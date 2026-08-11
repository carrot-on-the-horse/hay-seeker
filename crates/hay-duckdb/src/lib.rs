#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Durable embedded `DuckDB` storage with deterministic BM25 and exact vectors.
//!
//! `DuckDB` owns transactional persistence. Lexical statistics are stored in
//! regular tables so incremental correctness does not depend on `DuckDB`'s FTS
//! extension, whose index is not automatically refreshed. Dense retrieval is
//! an exact cosine scan until the ANN acceptance gate justifies an index.
//!
//! ```
//! use hay_duckdb::DuckDbIndex;
//! use hay_search::IndexManifest;
//!
//! let index = DuckDbIndex::open_in_memory(IndexManifest::lexical_v1(), None)?;
//! assert_eq!(index.document_count()?, 0);
//! # Ok::<(), hay_search::SearchError>(())
//! ```

mod vector;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use cast_index::{Embedder, EmbeddingInput};
use duckdb::{Config, Connection, Transaction, params, params_from_iter};
use futures::lock::Mutex as AsyncMutex;
use hay_search::{
    Candidate, Capabilities, DocId, IndexManifest, Quantization, Query, Retriever, SearchDocument,
    SearchError, SearchOpts, analyze_code_terms, fuse_ranked_results,
};
use ring::digest::{Context as DigestContext, SHA256};

use vector::{
    cosine_f32, cosine_int8, decode_f32, decode_int8, encode_f32, encode_int8, prepare_query,
};

const SCHEMA: &str = r"
CREATE TABLE IF NOT EXISTS hay_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);
CREATE TABLE IF NOT EXISTS hay_documents (
    document_id VARCHAR PRIMARY KEY,
    path VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    content VARCHAR NOT NULL,
    token_count BIGINT NOT NULL,
    embedding BLOB,
    embedding_dimensions BIGINT
);
CREATE TABLE IF NOT EXISTS hay_document_terms (
    document_id VARCHAR NOT NULL,
    term VARCHAR NOT NULL,
    term_frequency BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS hay_term_stats (
    term VARCHAR PRIMARY KEY,
    document_frequency BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS hay_embedding_cache (
    cache_key VARCHAR PRIMARY KEY,
    embedding BLOB NOT NULL,
    embedding_dimensions BIGINT NOT NULL
);
";
const EMBEDDING_BATCH_SIZE: usize = 128;
const DEFAULT_MEMORY_LIMIT: &str = "512MB";
const DEFAULT_WRITE_BUFFER_LIMIT: &str = "32MB";
const DEFAULT_THREADS: i64 = 2;
const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
const RESET_STAGING: &str = r"
DROP TABLE IF EXISTS hay_stage_document_terms;
DROP TABLE IF EXISTS hay_stage_documents;
CREATE TEMP TABLE hay_stage_documents AS
    SELECT * FROM hay_documents WHERE FALSE;
CREATE TEMP TABLE hay_stage_document_terms AS
    SELECT * FROM hay_document_terms WHERE FALSE;
";
const DROP_STAGING: &str = r"
DROP TABLE IF EXISTS hay_stage_document_terms;
DROP TABLE IF EXISTS hay_stage_documents;
";

/// Durable DuckDB-backed hybrid-search index.
pub struct DuckDbIndex {
    connection: Mutex<Connection>,
    rebuild: AsyncMutex<()>,
    runtime_manifest: IndexManifest,
    embedder: Option<Arc<dyn Embedder>>,
}

impl DuckDbIndex {
    /// Opens or creates a file-backed index and validates its stored manifest.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::ReindexRequired`] for an incompatible existing
    /// index and [`SearchError::Retriever`] for storage failures.
    pub fn open(
        path: impl AsRef<Path>,
        runtime_manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        let connection =
            Connection::open_with_flags(path, embedded_config()?).map_err(storage_error)?;
        Self::from_connection(connection, runtime_manifest, embedder)
    }

    /// Creates a non-persistent index for tests and ephemeral workflows.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for invalid configuration or initialization.
    pub fn open_in_memory(
        runtime_manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        let connection =
            Connection::open_in_memory_with_flags(embedded_config()?).map_err(storage_error)?;
        Self::from_connection(connection, runtime_manifest, embedder)
    }

    fn from_connection(
        connection: Connection,
        runtime_manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        runtime_manifest.validate()?;
        validate_duckdb_manifest(&runtime_manifest)?;
        validate_embedder(&runtime_manifest, embedder.as_deref())?;
        connection.execute_batch(SCHEMA).map_err(storage_error)?;
        let stored = read_manifest(&connection)?;
        if let Some(stored) = stored {
            stored.validate_runtime(&runtime_manifest)?;
        } else {
            let encoded = serde_json::to_string(&runtime_manifest).map_err(storage_error)?;
            connection
                .execute(
                    "INSERT INTO hay_metadata (key, value) VALUES ('manifest', ?)",
                    params![encoded],
                )
                .map_err(storage_error)?;
        }
        Ok(Self {
            connection: Mutex::new(connection),
            rebuild: AsyncMutex::new(()),
            runtime_manifest,
            embedder,
        })
    }

    /// Atomically replaces the complete document set and lexical statistics.
    ///
    /// Missing document embeddings are generated when an embedder is attached.
    /// This full-rebuild operation is the first product path; file-scoped
    /// incremental writes will reuse the same normalized tables.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for embedding, validation, or transaction errors.
    pub async fn replace_all(&self, documents: &[SearchDocument]) -> Result<(), SearchError> {
        self.replace_stream(documents.iter().cloned().map(Ok))
            .await
            .map(|_| ())
    }

    /// Atomically replaces the complete index from a bounded document stream.
    ///
    /// The iterator is consumed in small embedding/storage batches, so callers
    /// can scan and chunk repositories without retaining every source body in
    /// memory. Any iterator, embedding, validation, or storage error discards
    /// staging and preserves the previous searchable index.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for source, embedding, validation, manifest, or
    /// transaction errors.
    pub async fn replace_stream<I>(&self, documents: I) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
    {
        let _rebuild = self.rebuild.lock().await;
        let result = self.replace_stream_inner(documents).await;
        if result.is_err() {
            let _ = self.drop_staging();
        }
        result
    }

    async fn replace_stream_inner<I>(&self, documents: I) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
    {
        {
            let connection = self.lock_connection()?;
            self.verify_stored_manifest(&connection)?;
            connection
                .execute_batch(RESET_STAGING)
                .map_err(storage_error)?;
        }
        let mut documents = documents.into_iter();
        let mut indexed = 0_usize;
        loop {
            let batch = documents
                .by_ref()
                .take(EMBEDDING_BATCH_SIZE)
                .collect::<Result<Vec<_>, _>>()?;
            if batch.is_empty() {
                break;
            }
            let prepared = self.prepare_documents(&batch).await?;
            let mut connection = self.lock_connection()?;
            let transaction = connection.transaction().map_err(storage_error)?;
            append_documents_to(
                &transaction,
                "hay_stage_documents",
                "hay_stage_document_terms",
                &prepared,
                &self.runtime_manifest.quantization,
            )?;
            transaction.commit().map_err(storage_error)?;
            indexed = indexed.saturating_add(prepared.len());
        }
        let mut connection = self.lock_connection()?;
        let transaction = connection.transaction().map_err(storage_error)?;
        transaction
            .execute_batch(
                "DELETE FROM hay_document_terms;
                 DELETE FROM hay_term_stats;
                 DELETE FROM hay_documents;
                 INSERT INTO hay_documents SELECT * FROM hay_stage_documents;
                 INSERT INTO hay_document_terms SELECT * FROM hay_stage_document_terms;",
            )
            .map_err(storage_error)?;
        refresh_term_stats(&transaction)?;
        transaction
            .execute_batch(DROP_STAGING)
            .map_err(storage_error)?;
        transaction.commit().map_err(storage_error)?;
        Ok(indexed)
    }

    /// Atomically applies a bounded stream of changed documents and stale IDs.
    ///
    /// Changed documents are embedded into private staging tables first. The
    /// deletion list is resolved only after the source stream is exhausted,
    /// then deletions and inserts are committed in one transaction. A source,
    /// embedding, or storage failure preserves the previous searchable state.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for source, embedding, manifest, or transaction
    /// failures.
    pub async fn update_stream<I, F>(
        &self,
        documents: I,
        deletions: F,
    ) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
        F: FnOnce() -> Result<Vec<DocId>, SearchError>,
    {
        let _rebuild = self.rebuild.lock().await;
        let result = self.update_stream_inner(documents, deletions).await;
        if result.is_err() {
            let _ = self.drop_staging();
        }
        result
    }

    async fn update_stream_inner<I, F>(
        &self,
        documents: I,
        deletions: F,
    ) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
        F: FnOnce() -> Result<Vec<DocId>, SearchError>,
    {
        {
            let connection = self.lock_connection()?;
            self.verify_stored_manifest(&connection)?;
            connection
                .execute_batch(RESET_STAGING)
                .map_err(storage_error)?;
        }
        let mut documents = documents.into_iter();
        let mut indexed = 0_usize;
        loop {
            let batch = documents
                .by_ref()
                .take(EMBEDDING_BATCH_SIZE)
                .collect::<Result<Vec<_>, _>>()?;
            if batch.is_empty() {
                break;
            }
            let prepared = self.prepare_documents(&batch).await?;
            let mut connection = self.lock_connection()?;
            let transaction = connection.transaction().map_err(storage_error)?;
            append_documents_to(
                &transaction,
                "hay_stage_documents",
                "hay_stage_document_terms",
                &prepared,
                &self.runtime_manifest.quantization,
            )?;
            transaction.commit().map_err(storage_error)?;
            indexed = indexed.saturating_add(prepared.len());
        }
        let deletions = deletions()?;
        let mut connection = self.lock_connection()?;
        let transaction = connection.transaction().map_err(storage_error)?;
        for id in &deletions {
            transaction
                .execute(
                    "DELETE FROM hay_document_terms WHERE document_id = ?",
                    params![id.as_str()],
                )
                .map_err(storage_error)?;
            transaction
                .execute(
                    "DELETE FROM hay_documents WHERE document_id = ?",
                    params![id.as_str()],
                )
                .map_err(storage_error)?;
        }
        transaction
            .execute_batch(
                "INSERT INTO hay_documents SELECT * FROM hay_stage_documents;
                 INSERT INTO hay_document_terms SELECT * FROM hay_stage_document_terms;",
            )
            .map_err(storage_error)?;
        refresh_term_stats(&transaction)?;
        transaction
            .execute_batch(DROP_STAGING)
            .map_err(storage_error)?;
        transaction.commit().map_err(storage_error)?;
        Ok(indexed)
    }

    fn drop_staging(&self) -> Result<(), SearchError> {
        self.lock_connection()?
            .execute_batch(DROP_STAGING)
            .map_err(storage_error)
    }

    /// Atomically inserts or replaces the supplied documents.
    ///
    /// Existing term rows and embeddings are removed before the replacement is
    /// inserted, preventing stale lexical matches after a file changes.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for embedding, validation, or transaction errors.
    pub async fn upsert_documents(&self, documents: &[SearchDocument]) -> Result<(), SearchError> {
        let documents = self.prepare_documents(documents).await?;
        let mut connection = self.lock_connection()?;
        self.verify_stored_manifest(&connection)?;
        let transaction = connection.transaction().map_err(storage_error)?;
        for document in &documents {
            transaction
                .execute(
                    "DELETE FROM hay_document_terms WHERE document_id = ?",
                    params![document.doc_id.as_str()],
                )
                .map_err(storage_error)?;
            transaction
                .execute(
                    "DELETE FROM hay_documents WHERE document_id = ?",
                    params![document.doc_id.as_str()],
                )
                .map_err(storage_error)?;
            insert_document(&transaction, document, &self.runtime_manifest.quantization)?;
        }
        refresh_term_stats(&transaction)?;
        transaction.commit().map_err(storage_error)
    }

    /// Atomically deletes documents and refreshes lexical corpus statistics.
    ///
    /// IDs that are not present are ignored, making retries idempotent.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for manifest or transaction errors.
    pub fn delete_documents(&self, ids: &[DocId]) -> Result<(), SearchError> {
        let mut connection = self.lock_connection()?;
        self.verify_stored_manifest(&connection)?;
        let transaction = connection.transaction().map_err(storage_error)?;
        for id in ids {
            transaction
                .execute(
                    "DELETE FROM hay_document_terms WHERE document_id = ?",
                    params![id.as_str()],
                )
                .map_err(storage_error)?;
            transaction
                .execute(
                    "DELETE FROM hay_documents WHERE document_id = ?",
                    params![id.as_str()],
                )
                .map_err(storage_error)?;
        }
        refresh_term_stats(&transaction)?;
        transaction.commit().map_err(storage_error)
    }

    /// Returns the number of searchable documents currently stored.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Retriever`] when `DuckDB` cannot answer the query.
    pub fn document_count(&self) -> Result<usize, SearchError> {
        let connection = self.lock_connection()?;
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM hay_documents", [], |row| row.get(0))
            .map_err(storage_error)?;
        usize::try_from(count).map_err(|_| storage_error("negative document count"))
    }

    /// Loads documents by ID in the caller's requested order.
    ///
    /// Missing IDs are omitted.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Retriever`] for storage or decoding failures.
    pub fn documents(&self, ids: &[DocId]) -> Result<Vec<SearchDocument>, SearchError> {
        let connection = self.lock_connection()?;
        let mut statement = connection
            .prepare(
                "SELECT path, language, content, embedding, embedding_dimensions
                 FROM hay_documents WHERE document_id = ?",
            )
            .map_err(storage_error)?;
        let mut documents = Vec::with_capacity(ids.len());
        for id in ids {
            let mut rows = statement
                .query(params![id.as_str()])
                .map_err(storage_error)?;
            if let Some(row) = rows.next().map_err(storage_error)? {
                let blob: Option<Vec<u8>> = row.get(3).map_err(storage_error)?;
                let dimensions: Option<i64> = row.get(4).map_err(storage_error)?;
                documents.push(SearchDocument {
                    doc_id: id.clone(),
                    path: cast_index::NormalizedPath::new(
                        row.get::<_, String>(0).map_err(storage_error)?,
                    )
                    .map_err(storage_error)?,
                    language: cast_core::LanguageId::new(
                        row.get::<_, String>(1).map_err(storage_error)?,
                    ),
                    text: row.get(2).map_err(storage_error)?,
                    embedding: decode_optional_embedding(
                        blob,
                        dimensions,
                        &self.runtime_manifest.quantization,
                    )?,
                });
            }
        }
        Ok(documents)
    }

    /// Returns the exact runtime manifest enforced by every query.
    #[must_use]
    pub const fn runtime_manifest(&self) -> &IndexManifest {
        &self.runtime_manifest
    }

    async fn prepare_documents(
        &self,
        documents: &[SearchDocument],
    ) -> Result<Vec<SearchDocument>, SearchError> {
        let mut prepared = documents.to_vec();
        let originally_missing = prepared
            .iter()
            .enumerate()
            .filter_map(|(index, document)| document.embedding.is_none().then_some(index))
            .collect::<Vec<_>>();
        let missing = if self.embedder.is_some() {
            self.load_cached_embeddings(&mut prepared, &originally_missing)?
        } else {
            originally_missing.clone()
        };
        if let (Some(embedder), false) = (&self.embedder, missing.is_empty()) {
            let inputs = missing
                .iter()
                .map(|index| EmbeddingInput {
                    document_id: &prepared[*index].doc_id,
                    text: &prepared[*index].text,
                })
                .collect::<Vec<_>>();
            let vectors = embedder
                .embed_batch(&inputs)
                .await
                .map_err(|error| SearchError::Retriever(error.to_string()))?;
            if vectors.len() != missing.len() {
                return Err(SearchError::Retriever(format!(
                    "embedder returned {} vectors for {} documents",
                    vectors.len(),
                    missing.len()
                )));
            }
            for (index, vector) in missing.into_iter().zip(vectors) {
                prepared[index].embedding = Some(project_vector(
                    vector.values,
                    self.runtime_manifest.embed_dim,
                    self.runtime_manifest.mrl_dim,
                )?);
            }
        }
        for document in &prepared {
            document.validate(&self.runtime_manifest)?;
        }
        if self.embedder.is_some() && !originally_missing.is_empty() {
            self.persist_cached_embeddings(&prepared, &originally_missing)?;
        }
        Ok(prepared)
    }

    fn load_cached_embeddings(
        &self,
        documents: &mut [SearchDocument],
        missing: &[usize],
    ) -> Result<Vec<usize>, SearchError> {
        let connection = self.lock_connection()?;
        self.verify_stored_manifest(&connection)?;
        let mut statement = connection
            .prepare(
                "SELECT embedding, embedding_dimensions
                 FROM hay_embedding_cache WHERE cache_key = ?",
            )
            .map_err(storage_error)?;
        let mut unresolved = Vec::new();
        for index in missing {
            let key = embedding_cache_key(&documents[*index])?;
            let mut rows = statement.query(params![key]).map_err(storage_error)?;
            let Some(row) = rows.next().map_err(storage_error)? else {
                unresolved.push(*index);
                continue;
            };
            let blob: Vec<u8> = row.get(0).map_err(storage_error)?;
            let dimensions: i64 = row.get(1).map_err(storage_error)?;
            let actual_dimensions = usize::try_from(dimensions).map_err(storage_error)?;
            if actual_dimensions != self.runtime_manifest.mrl_dim {
                return Err(storage_error(format!(
                    "embedding cache row has {actual_dimensions} dimensions; expected {}",
                    self.runtime_manifest.mrl_dim
                )));
            }
            documents[*index].embedding = Some(decode_embedding(
                &blob,
                dimensions,
                &self.runtime_manifest.quantization,
            )?);
        }
        Ok(unresolved)
    }

    fn persist_cached_embeddings(
        &self,
        documents: &[SearchDocument],
        indices: &[usize],
    ) -> Result<(), SearchError> {
        let mut connection = self.lock_connection()?;
        self.verify_stored_manifest(&connection)?;
        let transaction = connection.transaction().map_err(storage_error)?;
        for index in indices {
            let document = &documents[*index];
            let vector = document.embedding.as_ref().ok_or_else(|| {
                storage_error(format!(
                    "document {} is missing an embedding after preparation",
                    document.doc_id
                ))
            })?;
            let key = embedding_cache_key(document)?;
            let encoded = encode_embedding(vector, &self.runtime_manifest.quantization)?;
            let dimensions = i64::try_from(vector.len()).map_err(storage_error)?;
            transaction
                .execute(
                    "INSERT INTO hay_embedding_cache
                     (cache_key, embedding, embedding_dimensions) VALUES (?, ?, ?)
                     ON CONFLICT (cache_key) DO NOTHING",
                    params![&key, &encoded, dimensions],
                )
                .map_err(storage_error)?;
            let (stored, stored_dimensions) = transaction
                .query_row(
                    "SELECT embedding, embedding_dimensions
                     FROM hay_embedding_cache WHERE cache_key = ?",
                    params![&key],
                    |row| Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?)),
                )
                .map_err(storage_error)?;
            if stored_dimensions != dimensions || stored != encoded {
                return Err(storage_error(format!(
                    "embedding cache conflict for document {}; the pinned embedder was not deterministic",
                    document.doc_id
                )));
            }
        }
        transaction.commit().map_err(storage_error)
    }

    fn lock_connection(&self) -> Result<MutexGuard<'_, Connection>, SearchError> {
        self.connection
            .lock()
            .map_err(|_| storage_error("DuckDB connection lock was poisoned"))
    }

    fn verify_stored_manifest(&self, connection: &Connection) -> Result<(), SearchError> {
        let stored = read_manifest(connection)?
            .ok_or_else(|| storage_error("DuckDB index manifest is missing"))?;
        stored.validate_runtime(&self.runtime_manifest)
    }
}

fn embedded_config() -> Result<Config, SearchError> {
    Config::default()
        .max_memory(DEFAULT_MEMORY_LIMIT)
        .and_then(|config| config.threads(DEFAULT_THREADS))
        .and_then(|config| config.with("preserve_insertion_order", "false"))
        .and_then(|config| {
            config.with(
                "write_buffer_row_group_memory_limit",
                DEFAULT_WRITE_BUFFER_LIMIT,
            )
        })
        .map_err(storage_error)
}

fn embedding_cache_key(document: &SearchDocument) -> Result<String, SearchError> {
    let mut digest = DigestContext::new(&SHA256);
    digest.update(b"hay-seeker/embedding-cache/v1\0");
    update_digest_field(&mut digest, document.doc_id.as_str().as_bytes())?;
    update_digest_field(&mut digest, document.text.as_bytes())?;
    let digest = digest.finish();
    let mut encoded = String::with_capacity(digest.as_ref().len().saturating_mul(2));
    for byte in digest.as_ref() {
        encoded.push(char::from(LOWER_HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(LOWER_HEX[usize::from(byte & 0x0f)]));
    }
    Ok(encoded)
}

fn update_digest_field(digest: &mut DigestContext, value: &[u8]) -> Result<(), SearchError> {
    let length = u64::try_from(value.len()).map_err(storage_error)?;
    digest.update(&length.to_le_bytes());
    digest.update(value);
    Ok(())
}

#[async_trait]
impl Retriever for DuckDbIndex {
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError> {
        query.validate()?;
        options.validate()?;
        let dense = if let Some(embedder) = &self.embedder {
            let vector = embedder
                .embed_query(&query.text)
                .await
                .map_err(|error| SearchError::Retriever(error.to_string()))?;
            Some(project_vector(
                vector.values,
                self.runtime_manifest.embed_dim,
                self.runtime_manifest.mrl_dim,
            )?)
        } else {
            None
        };

        let connection = self.lock_connection()?;
        self.verify_stored_manifest(&connection)?;
        let lexical = lexical_search(&connection, &query.text, options.candidate_limit.get())?;
        let dense = dense.map_or_else(
            || Ok(Vec::new()),
            |vector| {
                dense_search(
                    &connection,
                    &vector,
                    options.candidate_limit.get(),
                    &self.runtime_manifest.quantization,
                )
            },
        )?;
        Ok(fuse_ranked_results(&lexical, &dense, options.top_k.get()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            lexical: true,
            dense: self.embedder.is_some(),
            quantized_rescore: self.embedder.is_some()
                && self.runtime_manifest.quantization == Quantization::Int8PerVectorScaleOffset,
            ..Capabilities::default()
        }
    }
}

fn insert_document(
    transaction: &Transaction<'_>,
    document: &SearchDocument,
    quantization: &Quantization,
) -> Result<(), SearchError> {
    let fields = storage_fields(document, quantization)?;
    transaction
        .execute(
            "INSERT INTO hay_documents
             (document_id, path, language, content, token_count, embedding, embedding_dimensions)
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            params![
                document.doc_id.as_str(),
                document.path.as_str(),
                &document.language.0,
                &document.text,
                fields.token_count,
                fields.embedding,
                fields.dimensions
            ],
        )
        .map_err(storage_error)?;
    for (term, frequency) in fields.terms {
        transaction
            .execute(
                "INSERT INTO hay_document_terms (document_id, term, term_frequency)
                 VALUES (?, ?, ?)",
                params![document.doc_id.as_str(), term, i64::from(frequency)],
            )
            .map_err(storage_error)?;
    }
    Ok(())
}

fn append_documents_to(
    transaction: &Transaction<'_>,
    document_table: &str,
    term_table: &str,
    documents: &[SearchDocument],
    quantization: &Quantization,
) -> Result<(), SearchError> {
    let mut document_appender = transaction
        .appender(document_table)
        .map_err(storage_error)?;
    let mut term_appender = transaction.appender(term_table).map_err(storage_error)?;
    for document in documents {
        let fields = storage_fields(document, quantization)?;
        document_appender
            .append_row(params![
                document.doc_id.as_str(),
                document.path.as_str(),
                &document.language.0,
                &document.text,
                fields.token_count,
                fields.embedding,
                fields.dimensions
            ])
            .map_err(storage_error)?;
        for (term, frequency) in fields.terms {
            term_appender
                .append_row(params![
                    document.doc_id.as_str(),
                    term,
                    i64::from(frequency)
                ])
                .map_err(storage_error)?;
        }
    }
    document_appender.flush().map_err(storage_error)?;
    term_appender.flush().map_err(storage_error)
}

struct StorageFields {
    terms: BTreeMap<String, u32>,
    token_count: i64,
    embedding: Option<Vec<u8>>,
    dimensions: Option<i64>,
}

fn storage_fields(
    document: &SearchDocument,
    quantization: &Quantization,
) -> Result<StorageFields, SearchError> {
    let mut terms = analyze_code_terms(&document.text);
    add_weighted_terms(&mut terms, document.path.as_str(), 3);
    add_weighted_terms(&mut terms, document.doc_id.as_str(), 2);
    let token_count = terms.values().map(|count| u64::from(*count)).sum::<u64>();
    let token_count = i64::try_from(token_count).map_err(storage_error)?;
    let (embedding, dimensions) = match &document.embedding {
        Some(vector) => (
            Some(encode_embedding(vector, quantization)?),
            Some(i64::try_from(vector.len()).map_err(storage_error)?),
        ),
        None => (None, None),
    };
    Ok(StorageFields {
        terms,
        token_count,
        embedding,
        dimensions,
    })
}

fn refresh_term_stats(transaction: &Transaction<'_>) -> Result<(), SearchError> {
    transaction
        .execute_batch(
            "DELETE FROM hay_term_stats;
             INSERT INTO hay_term_stats
             SELECT term, COUNT(*) FROM hay_document_terms GROUP BY term;",
        )
        .map_err(storage_error)
}

fn add_weighted_terms(terms: &mut BTreeMap<String, u32>, text: &str, weight: u32) {
    for (term, frequency) in analyze_code_terms(text) {
        let count = terms.entry(term).or_default();
        *count = count.saturating_add(frequency.saturating_mul(weight));
    }
}

fn lexical_search(
    connection: &Connection,
    query: &str,
    limit: usize,
) -> Result<Vec<(DocId, f32)>, SearchError> {
    let terms = analyze_code_terms(query).into_keys().collect::<Vec<_>>();
    if terms.is_empty() {
        return Ok(Vec::new());
    }
    let values = std::iter::repeat_n("(?)", terms.len())
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "WITH query_terms(term) AS (VALUES {values}),
         corpus_stats AS (
             SELECT COUNT(*)::DOUBLE AS document_count,
                    COALESCE(AVG(token_count), 1)::DOUBLE AS average_length
             FROM hay_documents
         )
         SELECT d.document_id,
                CAST(SUM(
                    LN(1 + ((s.document_count - ts.document_frequency + 0.5) /
                            (ts.document_frequency + 0.5))) *
                    ((dt.term_frequency * 2.2) /
                     (dt.term_frequency + 1.2 *
                      (0.25 + 0.75 * d.token_count / s.average_length)))
                ) AS FLOAT) AS bm25
         FROM query_terms q
         JOIN hay_document_terms dt ON dt.term = q.term
         JOIN hay_term_stats ts ON ts.term = dt.term
         JOIN hay_documents d ON d.document_id = dt.document_id
         CROSS JOIN corpus_stats s
         GROUP BY d.document_id
         ORDER BY bm25 DESC, d.document_id ASC
         LIMIT {limit}"
    );
    let mut statement = connection.prepare(&sql).map_err(storage_error)?;
    let rows = statement
        .query_map(params_from_iter(terms.iter()), |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f32>(1)?))
        })
        .map_err(storage_error)?;
    rows.map(|row| {
        let (id, score) = row.map_err(storage_error)?;
        Ok((DocId::new(id).map_err(storage_error)?, score))
    })
    .collect()
}

fn dense_search(
    connection: &Connection,
    query: &[f32],
    limit: usize,
    quantization: &Quantization,
) -> Result<Vec<(DocId, f32)>, SearchError> {
    let mut statement = connection
        .prepare(
            "SELECT document_id, embedding, embedding_dimensions
             FROM hay_documents WHERE embedding IS NOT NULL",
        )
        .map_err(storage_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(storage_error)?;
    let prepared_query = prepare_query(query).map_err(storage_error)?;
    let mut scored = Vec::new();
    for row in rows {
        let (id, blob, dimensions) = row.map_err(storage_error)?;
        let dimensions = usize::try_from(dimensions).map_err(storage_error)?;
        if dimensions != query.len() {
            return Err(storage_error(format!(
                "document {id} vector has {dimensions} dimensions; query has {}",
                query.len()
            )));
        }
        let score = match quantization {
            Quantization::None => {
                let vector = decode_f32(&blob, dimensions).map_err(storage_error)?;
                cosine_f32(&prepared_query, &vector).map_err(storage_error)?
            }
            Quantization::Int8PerVectorScaleOffset => {
                cosine_int8(&prepared_query, &blob).map_err(storage_error)?
            }
            Quantization::ElasticBbq => {
                return Err(storage_error(
                    "Elastic BBQ quantization cannot be searched by DuckDB",
                ));
            }
        };
        scored.push((DocId::new(id).map_err(storage_error)?, score));
    }
    scored.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(limit);
    Ok(scored)
}

fn read_manifest(connection: &Connection) -> Result<Option<IndexManifest>, SearchError> {
    let mut statement = connection
        .prepare("SELECT value FROM hay_metadata WHERE key = 'manifest'")
        .map_err(storage_error)?;
    let mut rows = statement.query([]).map_err(storage_error)?;
    let Some(row) = rows.next().map_err(storage_error)? else {
        return Ok(None);
    };
    let encoded: String = row.get(0).map_err(storage_error)?;
    serde_json::from_str(&encoded)
        .map(Some)
        .map_err(storage_error)
}

fn validate_embedder(
    manifest: &IndexManifest,
    embedder: Option<&dyn Embedder>,
) -> Result<(), SearchError> {
    let Some(embedder) = embedder else {
        return Ok(());
    };
    let identity = embedder.identity();
    if identity.model != manifest.model_id
        || identity.profile != manifest.embedding_profile
        || !matches!(identity.dimensions, dimensions if dimensions == manifest.embed_dim || dimensions == manifest.mrl_dim)
    {
        return Err(SearchError::InvalidConfig(format!(
            "embedder identity {identity:?} does not match index manifest"
        )));
    }
    Ok(())
}

fn validate_duckdb_manifest(manifest: &IndexManifest) -> Result<(), SearchError> {
    if manifest.quantization == Quantization::ElasticBbq {
        return Err(SearchError::InvalidConfig(
            "DuckDB does not support Elasticsearch BBQ vector storage".into(),
        ));
    }
    Ok(())
}

fn project_vector(
    mut vector: Vec<f32>,
    embed_dimensions: usize,
    stored_dimensions: usize,
) -> Result<Vec<f32>, SearchError> {
    if vector.len() != embed_dimensions && vector.len() != stored_dimensions {
        return Err(SearchError::Retriever(format!(
            "embedding contains {} dimensions; expected {embed_dimensions} or {stored_dimensions}",
            vector.len()
        )));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(SearchError::Retriever(
            "embedding contains a non-finite value".into(),
        ));
    }
    vector.truncate(stored_dimensions);
    Ok(vector)
}

fn encode_embedding(vector: &[f32], quantization: &Quantization) -> Result<Vec<u8>, SearchError> {
    match quantization {
        Quantization::None => encode_f32(vector).map_err(storage_error),
        Quantization::Int8PerVectorScaleOffset => encode_int8(vector).map_err(storage_error),
        Quantization::ElasticBbq => Err(storage_error(
            "Elastic BBQ quantization cannot be stored by DuckDB",
        )),
    }
}

fn decode_optional_embedding(
    blob: Option<Vec<u8>>,
    dimensions: Option<i64>,
    quantization: &Quantization,
) -> Result<Option<Vec<f32>>, SearchError> {
    match (blob, dimensions) {
        (None, None) => Ok(None),
        (Some(blob), Some(dimensions)) => {
            decode_embedding(&blob, dimensions, quantization).map(Some)
        }
        _ => Err(storage_error("incomplete stored embedding")),
    }
}

fn decode_embedding(
    blob: &[u8],
    dimensions: i64,
    quantization: &Quantization,
) -> Result<Vec<f32>, SearchError> {
    let dimensions = usize::try_from(dimensions).map_err(storage_error)?;
    match quantization {
        Quantization::None => decode_f32(blob, dimensions).map_err(storage_error),
        Quantization::Int8PerVectorScaleOffset => {
            decode_int8(blob, dimensions).map_err(storage_error)
        }
        Quantization::ElasticBbq => Err(storage_error(
            "Elastic BBQ quantization cannot be decoded by DuckDB",
        )),
    }
}

fn storage_error(error: impl std::fmt::Display) -> SearchError {
    SearchError::Retriever(format!("DuckDB: {error}"))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cast_index::{
        BoxFuture, ContentHash, EmbeddingIdentity, EmbeddingVector, HashAlgorithm, IndexError,
    };
    use hay_search::{FdeParams, Quantization};
    use tempfile::tempdir;

    use super::*;

    struct TestEmbedder {
        identity: EmbeddingIdentity,
        embedded_documents: AtomicUsize,
    }

    impl TestEmbedder {
        fn new() -> Self {
            Self {
                identity: EmbeddingIdentity {
                    provider: "test".into(),
                    model: "hash-embedder".into(),
                    dimensions: 8,
                    profile: "symmetric-v1".into(),
                },
                embedded_documents: AtomicUsize::new(0),
            }
        }

        fn embedded_documents(&self) -> usize {
            self.embedded_documents.load(Ordering::SeqCst)
        }

        fn vector(text: &str) -> EmbeddingVector {
            let mut values = vec![0.0; 8];
            for term in analyze_code_terms(text).keys() {
                let bucket = term.bytes().fold(0_usize, |hash, byte| {
                    hash.wrapping_mul(31).wrapping_add(usize::from(byte))
                }) % values.len();
                values[bucket] += 1.0;
            }
            EmbeddingVector {
                identity: Self::new().identity,
                values,
            }
        }
    }

    impl Embedder for TestEmbedder {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }

        fn embed_batch<'a>(
            &'a self,
            inputs: &'a [EmbeddingInput<'a>],
        ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
            Box::pin(async move {
                self.embedded_documents
                    .fetch_add(inputs.len(), Ordering::SeqCst);
                Ok(inputs
                    .iter()
                    .map(|input| Self::vector(input.text))
                    .collect())
            })
        }

        fn embed_query<'a>(
            &'a self,
            text: &'a str,
        ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
            Box::pin(async move { Ok(Self::vector(text)) })
        }
    }

    fn manifest() -> IndexManifest {
        IndexManifest {
            model_id: "hash-embedder".into(),
            model_revision: "1".into(),
            embedding_profile: "symmetric-v1".into(),
            embed_dim: 8,
            mrl_dim: 8,
            quantization: Quantization::Int8PerVectorScaleOffset,
            tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "a".repeat(64)).unwrap(),
            chunker_version: "cast-rust-0.1".into(),
            fde_params: FdeParams::Disabled,
            schema_version: 1,
        }
    }

    fn document(id: &str, path: &str, text: &str) -> SearchDocument {
        SearchDocument {
            doc_id: DocId::new(id).unwrap(),
            path: cast_index::NormalizedPath::new(path).unwrap(),
            language: cast_core::LanguageId::new("rust"),
            text: text.into(),
            embedding: None,
        }
    }

    #[test]
    fn embedded_profile_bounds_threads_and_avoids_the_large_term_art_index() {
        let index = DuckDbIndex::open_in_memory(manifest(), None).unwrap();
        let connection = index.lock_connection().unwrap();
        let threads = connection
            .query_row("SELECT current_setting('threads')", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        let term_table_primary_keys = connection
            .prepare("PRAGMA table_info('hay_document_terms')")
            .unwrap()
            .query_map([], |row| row.get::<_, bool>(5))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .filter(|is_primary_key| *is_primary_key)
            .count();

        assert_eq!(threads, DEFAULT_THREADS);
        assert_eq!(term_table_primary_keys, 0);
    }

    fn options() -> SearchOpts {
        SearchOpts {
            top_k: NonZeroUsize::new(3).unwrap(),
            candidate_limit: NonZeroUsize::new(10).unwrap(),
            enable_late_interaction: false,
        }
    }

    #[tokio::test]
    async fn full_cycle_persists_bm25_vectors_and_manifest() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("index.duckdb");
        let embedder: Arc<dyn Embedder> = Arc::new(TestEmbedder::new());
        let index = DuckDbIndex::open(&path, manifest(), Some(embedder.clone())).unwrap();
        index
            .replace_all(&[
                document(
                    "manifest",
                    "src/manifest.rs",
                    "validate index manifest compatibility",
                ),
                document("chunker", "src/chunker.rs", "split syntax tree chunks"),
                document("gateway", "src/gateway.rs", "cloud gateway request"),
            ])
            .await
            .unwrap();
        {
            let connection = index.lock_connection().unwrap();
            let blob: Vec<u8> = connection
                .query_row(
                    "SELECT embedding FROM hay_documents WHERE document_id = 'manifest'",
                    [],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(blob.len(), 16 + manifest().mrl_dim);
            assert_eq!(&blob[..8], b"HAYI8\x01\0\0");
        }
        let results = index
            .search(&Query::new("manifest validation").unwrap(), &options())
            .await
            .unwrap();

        assert_eq!(results[0].doc_id.as_str(), "manifest");
        assert!(results[0].signals.lexical.is_some());
        assert!(results[0].signals.dense.is_some());
        assert_eq!(index.document_count().unwrap(), 3);
        drop(index);

        let reopened = DuckDbIndex::open(path, manifest(), Some(embedder)).unwrap();
        assert_eq!(reopened.document_count().unwrap(), 3);
    }

    #[tokio::test]
    async fn persistent_cache_reuses_only_exact_document_content() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("index.duckdb");
        let embedder = Arc::new(TestEmbedder::new());
        let index = DuckDbIndex::open(&path, manifest(), Some(embedder.clone())).unwrap();

        index
            .upsert_documents(&[document("same-id", "src/cache.rs", "branch alpha")])
            .await
            .unwrap();
        assert_eq!(embedder.embedded_documents(), 1);

        index
            .upsert_documents(&[document("same-id", "src/cache.rs", "branch beta")])
            .await
            .unwrap();
        assert_eq!(embedder.embedded_documents(), 2);
        drop(index);

        let reopened = DuckDbIndex::open(&path, manifest(), Some(embedder.clone())).unwrap();
        reopened
            .upsert_documents(&[document("same-id", "src/cache.rs", "branch alpha")])
            .await
            .unwrap();

        assert_eq!(embedder.embedded_documents(), 2);
        let connection = reopened.lock_connection().unwrap();
        let cache_rows: i64 = connection
            .query_row("SELECT COUNT(*) FROM hay_embedding_cache", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(cache_rows, 2);
    }

    #[tokio::test]
    async fn corrupt_cache_row_fails_closed_without_reembedding() {
        let embedder = Arc::new(TestEmbedder::new());
        let index = DuckDbIndex::open_in_memory(manifest(), Some(embedder.clone())).unwrap();
        let cached = document("cached", "src/cache.rs", "cache integrity");
        index
            .upsert_documents(std::slice::from_ref(&cached))
            .await
            .unwrap();
        assert_eq!(embedder.embedded_documents(), 1);
        let key = embedding_cache_key(&cached).unwrap();
        index
            .lock_connection()
            .unwrap()
            .execute(
                "UPDATE hay_embedding_cache SET embedding_dimensions = 7 WHERE cache_key = ?",
                params![key],
            )
            .unwrap();

        let error = index
            .upsert_documents(std::slice::from_ref(&cached))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("cache row has 7 dimensions"));
        assert_eq!(embedder.embedded_documents(), 1);
        assert_eq!(index.document_count().unwrap(), 1);
    }

    #[tokio::test]
    async fn incremental_update_removes_stale_terms_and_delete_is_idempotent() {
        let index = DuckDbIndex::open_in_memory(manifest(), None).unwrap();
        index
            .replace_all(&[
                document("changed", "src/old.rs", "obsolete unicorn parser"),
                document("kept", "src/kept.rs", "stable manifest validation"),
            ])
            .await
            .unwrap();

        index
            .upsert_documents(&[document(
                "changed",
                "src/new.rs",
                "current incremental indexer",
            )])
            .await
            .unwrap();

        let stale = index
            .search(&Query::new("obsolete unicorn").unwrap(), &options())
            .await
            .unwrap();
        let current = index
            .search(&Query::new("incremental indexer").unwrap(), &options())
            .await
            .unwrap();
        assert!(stale.is_empty());
        assert_eq!(current[0].doc_id.as_str(), "changed");

        let kept = DocId::new("kept").unwrap();
        index.delete_documents(std::slice::from_ref(&kept)).unwrap();
        index.delete_documents(&[kept]).unwrap();
        assert_eq!(index.document_count().unwrap(), 1);
        assert!(
            index
                .search(&Query::new("manifest validation").unwrap(), &options())
                .await
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn streamed_update_commits_changes_and_deletes_together() {
        let index = DuckDbIndex::open_in_memory(manifest(), None).unwrap();
        index
            .replace_all(&[
                document("old", "src/old.rs", "obsolete parser"),
                document("kept", "src/kept.rs", "stable manifest"),
            ])
            .await
            .unwrap();

        index
            .update_stream(
                [Ok(document("new", "src/new.rs", "current parser"))],
                || Ok(vec![DocId::new("old").unwrap()]),
            )
            .await
            .unwrap();
        assert_eq!(index.document_count().unwrap(), 2);
        assert!(
            index
                .documents(&[DocId::new("old").unwrap()])
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            index
                .documents(&[DocId::new("kept").unwrap(), DocId::new("new").unwrap()])
                .unwrap()
                .len(),
            2
        );

        let error = index
            .update_stream(
                [Ok(document("never", "src/never.rs", "not committed"))],
                || Err(SearchError::Corpus("checkpoint failure".into())),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("checkpoint failure"));
        assert_eq!(index.document_count().unwrap(), 2);
        assert!(
            index
                .documents(&[DocId::new("never").unwrap()])
                .unwrap()
                .is_empty()
        );
    }

    #[tokio::test]
    async fn streaming_rebuild_rolls_back_after_a_late_source_error() {
        let index = DuckDbIndex::open_in_memory(IndexManifest::lexical_v1(), None).unwrap();
        index
            .replace_all(&[document(
                "previous",
                "src/previous.rs",
                "previous searchable generation",
            )])
            .await
            .unwrap();
        let documents = (0..EMBEDDING_BATCH_SIZE)
            .map(|ordinal| {
                Ok(document(
                    &format!("new-{ordinal}"),
                    &format!("src/new-{ordinal}.rs"),
                    "new generation",
                ))
            })
            .chain(std::iter::once(Err(SearchError::Corpus(
                "simulated late read failure".into(),
            ))));

        let error = index.replace_stream(documents).await.unwrap_err();

        assert!(error.to_string().contains("late read failure"));
        assert_eq!(index.document_count().unwrap(), 1);
        let results = index
            .search(&Query::new("previous generation").unwrap(), &options())
            .await
            .unwrap();
        assert_eq!(results[0].doc_id.as_str(), "previous");
    }

    #[test]
    fn incompatible_manifest_hard_fails_on_open() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("index.duckdb");
        DuckDbIndex::open(&path, manifest(), None).unwrap();
        let incompatible = IndexManifest {
            chunker_version: "cast-rust-0.2".into(),
            ..manifest()
        };

        let error = DuckDbIndex::open(path, incompatible, None).err().unwrap();
        assert!(matches!(error, SearchError::ReindexRequired { .. }));
    }
}
