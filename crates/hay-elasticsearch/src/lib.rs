#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Elasticsearch implementation of the common Hay Seeker retrieval contract.
//!
//! Rebuilds use a new physical index followed by one atomic alias update. The
//! old index remains searchable if mapping creation or bulk indexing fails.
//! Ambiguous publication errors are reconciled against the exact alias before
//! an unpublished generation can be deleted.
//!
//! ```
//! use hay_elasticsearch::{ElasticsearchConfig, ElasticsearchIndex};
//! use hay_search::IndexManifest;
//!
//! let config = ElasticsearchConfig::new("http://127.0.0.1:9200", "hay-code");
//! let index = ElasticsearchIndex::new(config, IndexManifest::lexical_v1(), None)?;
//! # let _ = index;
//! # Ok::<(), hay_search::SearchError>(())
//! ```

use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use cast_index::{Embedder, EmbeddingInput};
use hay_search::{
    Candidate, Capabilities, DocId, IndexManifest, Query, Retriever, SearchDocument, SearchError,
    SearchOpts, analyze_code_terms, fuse_ranked_results,
};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use reqwest::{Client, Method, StatusCode, Url};
use serde_json::{Value, json};

const MAX_RESPONSE_BYTES: u64 = 8 * 1024 * 1024;
const TARGET_BULK_BYTES: usize = 5 * 1024 * 1024;
const EMBEDDING_BATCH_SIZE: usize = 128;
const DEFAULT_GENERATION_RETENTION: usize = 2;
const MAX_GENERATION_RETENTION: usize = 32;

/// Connection and index configuration for Elasticsearch.
pub struct ElasticsearchConfig {
    endpoint: String,
    index_alias: String,
    authorization: Option<String>,
    generation_retention: usize,
}

impl fmt::Debug for ElasticsearchConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElasticsearchConfig")
            .field("endpoint", &self.endpoint)
            .field("index_alias", &self.index_alias)
            .field("generation_retention", &self.generation_retention)
            .field(
                "authorization",
                &self.authorization.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl ElasticsearchConfig {
    /// Creates unauthenticated configuration, suitable for a local test node.
    #[must_use]
    pub fn new(endpoint: impl Into<String>, index_alias: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            index_alias: index_alias.into(),
            authorization: None,
            generation_retention: DEFAULT_GENERATION_RETENTION,
        }
    }

    /// Adds an Elasticsearch API key without logging or serializing it.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.authorization = Some(format!("ApiKey {}", api_key.into()));
        self
    }

    /// Adds a bearer token without logging or serializing it.
    #[must_use]
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.authorization = Some(format!("Bearer {}", token.into()));
        self
    }

    /// Sets the maximum number of Hay-owned physical generations to retain.
    ///
    /// At least two generations are required so the active index is never
    /// deleted before a replacement has been built and atomically published.
    #[must_use]
    pub const fn with_generation_retention(mut self, generations: usize) -> Self {
        self.generation_retention = generations;
        self
    }
}

/// Elasticsearch-backed lexical or hybrid index.
pub struct ElasticsearchIndex {
    client: Client,
    endpoint: Url,
    index_alias: String,
    runtime_manifest: IndexManifest,
    embedder: Option<Arc<dyn Embedder>>,
    generation_retention: usize,
}

impl fmt::Debug for ElasticsearchIndex {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ElasticsearchIndex")
            .field("endpoint", &self.endpoint)
            .field("index_alias", &self.index_alias)
            .field("runtime_manifest", &self.runtime_manifest)
            .field("embedder", &self.embedder.as_ref().map(|_| "configured"))
            .field("generation_retention", &self.generation_retention)
            .finish_non_exhaustive()
    }
}

impl ElasticsearchIndex {
    /// Builds a reusable backend client and validates local configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for an insecure remote URL,
    /// invalid alias, authorization value, manifest, or embedder identity.
    pub fn new(
        config: ElasticsearchConfig,
        runtime_manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        runtime_manifest.validate()?;
        validate_embedder(&runtime_manifest, embedder.as_deref())?;
        validate_index_name(&config.index_alias)?;
        if !(2..=MAX_GENERATION_RETENTION).contains(&config.generation_retention) {
            return Err(SearchError::InvalidConfig(format!(
                "Elasticsearch generation retention must be between 2 and {MAX_GENERATION_RETENTION}"
            )));
        }
        let endpoint = Url::parse(&config.endpoint)
            .map_err(|_| SearchError::InvalidConfig("invalid Elasticsearch endpoint".into()))?;
        if endpoint.scheme() != "https" && !is_loopback(&endpoint) {
            return Err(SearchError::InvalidConfig(
                "remote Elasticsearch endpoints must use HTTPS".into(),
            ));
        }
        let mut headers = HeaderMap::new();
        if let Some(authorization) = config.authorization {
            let mut value = HeaderValue::from_str(&authorization).map_err(|_| {
                SearchError::InvalidConfig("invalid Elasticsearch authorization".into())
            })?;
            value.set_sensitive(true);
            headers.insert(AUTHORIZATION, value);
        }
        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(transport_error)?;
        Ok(Self {
            client,
            endpoint,
            index_alias: config.index_alias,
            runtime_manifest,
            embedder,
            generation_retention: config.generation_retention,
        })
    }

    /// Builds a new physical index, bulk-indexes every document, and atomically
    /// switches the stable alias only after the new index is searchable.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for embedding, mapping, bulk-item, refresh, or
    /// alias-swap failures. The previous alias target remains intact on any
    /// failure before the final swap.
    pub async fn replace_all(&self, documents: &[SearchDocument]) -> Result<(), SearchError> {
        self.replace_stream(documents.iter().cloned().map(Ok))
            .await
            .map(|_| ())
    }

    /// Atomically replaces the alias target from a bounded document stream.
    ///
    /// Documents are embedded in bounded batches and serialized into bounded
    /// bulk requests. A source or indexing failure deletes the incomplete
    /// physical index and leaves the previous alias target searchable.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for source, embedding, mapping, bulk-item,
    /// refresh, or alias-swap failures.
    pub async fn replace_stream<I>(&self, documents: I) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
    {
        let obsolete = self.orphaned_generations_to_remove().await?;
        let physical = self.physical_index_name();
        self.request_json(
            Method::PUT,
            &physical,
            Some(create_index_body(&self.runtime_manifest)),
        )
        .await?;

        let indexed = match self.populate_index(&physical, documents).await {
            Ok(indexed) => indexed,
            Err(error) => {
                return Err(self.cleanup_unpublished_generation(&physical, error).await);
            }
        };
        self.publish_generation(&physical, obsolete).await?;
        Ok(indexed)
    }

    /// Builds an atomic incremental generation from the current alias target.
    ///
    /// Elasticsearch copies unchanged documents (including stored vectors) to
    /// a fresh physical index, applies the changed stream and stale-ID deletes,
    /// then swaps the alias. Failures delete the incomplete generation and
    /// leave the previous alias target searchable.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for manifest, reindex, source, embedding, bulk,
    /// refresh, or alias-swap failures.
    pub async fn update_stream<I, F>(
        &self,
        documents: I,
        deletions: F,
    ) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
        F: FnOnce() -> Result<Vec<DocId>, SearchError>,
    {
        self.verify_manifest().await?;
        let obsolete = self.orphaned_generations_to_remove().await?;
        let physical = self.physical_index_name();
        self.request_json(
            Method::PUT,
            &physical,
            Some(create_index_body(&self.runtime_manifest)),
        )
        .await?;

        let indexed = match async {
            let copied = self
                .request_json(
                    Method::POST,
                    "_reindex?wait_for_completion=true&refresh=false",
                    Some(json!({
                        "source": { "index": self.index_alias },
                        "dest": { "index": physical }
                    })),
                )
                .await?;
            validate_reindex_response(&copied)?;
            let indexed = self.populate_index(&physical, documents).await?;
            self.delete_from_index(&physical, &deletions()?).await?;
            self.request_json(Method::POST, &format!("{physical}/_refresh"), None)
                .await?;
            Ok(indexed)
        }
        .await
        {
            Ok(indexed) => indexed,
            Err(error) => {
                return Err(self.cleanup_unpublished_generation(&physical, error).await);
            }
        };
        self.publish_generation(&physical, obsolete).await?;
        Ok(indexed)
    }

    async fn populate_index<I>(&self, physical: &str, documents: I) -> Result<usize, SearchError>
    where
        I: IntoIterator<Item = Result<SearchDocument, SearchError>>,
    {
        let mut bulk = String::with_capacity(TARGET_BULK_BYTES);
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
            for document in &prepared {
                let record = bulk_record(document)?;
                if !bulk.is_empty() && bulk.len().saturating_add(record.len()) > TARGET_BULK_BYTES {
                    self.request_ndjson(Method::POST, &format!("{physical}/_bulk"), bulk)
                        .await?;
                    bulk = String::with_capacity(TARGET_BULK_BYTES);
                }
                bulk.push_str(&record);
            }
            indexed = indexed.saturating_add(prepared.len());
        }
        if !bulk.is_empty() {
            self.request_ndjson(Method::POST, &format!("{physical}/_bulk"), bulk)
                .await?;
        }
        self.request_json(Method::POST, &format!("{physical}/_refresh"), None)
            .await?;
        Ok(indexed)
    }

    async fn delete_from_index(&self, physical: &str, ids: &[DocId]) -> Result<(), SearchError> {
        let mut bulk = String::with_capacity(TARGET_BULK_BYTES);
        for id in ids {
            let record = format!(
                "{}\n",
                serde_json::to_string(&json!({ "delete": { "_id": id.as_str() } }))
                    .map_err(response_error)?
            );
            if !bulk.is_empty() && bulk.len().saturating_add(record.len()) > TARGET_BULK_BYTES {
                self.request_ndjson(Method::POST, &format!("{physical}/_bulk"), bulk)
                    .await?;
                bulk = String::with_capacity(TARGET_BULK_BYTES);
            }
            bulk.push_str(&record);
        }
        if !bulk.is_empty() {
            self.request_ndjson(Method::POST, &format!("{physical}/_bulk"), bulk)
                .await?;
        }
        Ok(())
    }

    /// Returns documents by ID in the requested order, omitting missing IDs.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Retriever`] for transport or response errors.
    pub async fn documents(&self, ids: &[DocId]) -> Result<Vec<SearchDocument>, SearchError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        self.verify_manifest().await?;
        let response = self
            .request_json(
                Method::POST,
                &format!("{}/_mget", self.index_alias),
                Some(json!({
                    "ids": ids.iter().map(DocId::as_str).collect::<Vec<_>>()
                })),
            )
            .await?;
        parse_mget_response(&response)
    }

    /// Returns the current document count behind the stable alias.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Retriever`] for transport or decoding failures.
    pub async fn document_count(&self) -> Result<usize, SearchError> {
        self.verify_manifest().await?;
        let response = self
            .request_json(Method::GET, &format!("{}/_count", self.index_alias), None)
            .await?;
        response
            .get("count")
            .and_then(Value::as_u64)
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| response_error("count response is missing count"))
    }

    /// Returns the number of strictly named physical generations owned by the
    /// configured Hay alias.
    ///
    /// This includes the active target and retained rollback generations, and
    /// excludes similarly prefixed indices that do not match Hay's complete
    /// timestamped naming contract.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Retriever`] for transport or response errors.
    pub async fn physical_generation_count(&self) -> Result<usize, SearchError> {
        let pattern = format!("{}-build-*/_alias", self.index_alias);
        let Some(generations) = self.request_optional_json(Method::GET, &pattern).await? else {
            return Ok(0);
        };
        let indices = generations
            .as_object()
            .ok_or_else(|| response_error("generation listing is not an object"))?;
        Ok(indices
            .keys()
            .filter(|name| physical_generation(name, &self.index_alias).is_some())
            .count())
    }

    async fn prepare_documents(
        &self,
        documents: &[SearchDocument],
    ) -> Result<Vec<SearchDocument>, SearchError> {
        let mut prepared = documents.to_vec();
        let missing = prepared
            .iter()
            .enumerate()
            .filter_map(|(index, document)| document.embedding.is_none().then_some(index))
            .collect::<Vec<_>>();
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
                return Err(response_error("embedder returned the wrong vector count"));
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
        Ok(prepared)
    }

    async fn verify_manifest(&self) -> Result<(), SearchError> {
        let mapping = self
            .request_json(Method::GET, &format!("{}/_mapping", self.index_alias), None)
            .await?;
        let manifests = mapping
            .as_object()
            .into_iter()
            .flat_map(|indices| indices.values())
            .filter_map(|index| index.pointer("/mappings/_meta/hay_manifest"))
            .map(|value| serde_json::from_value::<IndexManifest>(value.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(response_error)?;
        if manifests.is_empty() {
            return Err(response_error("index mapping is missing hay_manifest"));
        }
        for stored in manifests {
            stored.validate_runtime(&self.runtime_manifest)?;
        }
        Ok(())
    }

    fn physical_index_name(&self) -> String {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        format!(
            "{}-build-{}-{}",
            self.index_alias,
            elapsed.as_secs(),
            elapsed.subsec_nanos()
        )
    }

    async fn orphaned_generations_to_remove(&self) -> Result<Vec<String>, SearchError> {
        let pattern = format!("{}-build-*/_alias", self.index_alias);
        let Some(generations) = self.request_optional_json(Method::GET, &pattern).await? else {
            return Ok(Vec::new());
        };
        generation_cleanup_candidates(&generations, &self.index_alias, self.generation_retention)
    }

    async fn publish_generation(
        &self,
        physical: &str,
        obsolete: Vec<String>,
    ) -> Result<(), SearchError> {
        let publication = self
            .request_json(
                Method::POST,
                "_aliases",
                Some(alias_swap_body(&self.index_alias, physical, &obsolete)),
            )
            .await;
        let Err(publication_error) = publication else {
            return Ok(());
        };
        match self.alias_points_to(physical).await {
            Ok(true) => Ok(()),
            Ok(false) => Err(response_error(format!(
                "{publication_error}; generation {physical} was retained because a negative alias check cannot prove non-publication across every cluster node"
            ))),
            Err(verification_error) => Err(response_error(format!(
                "{publication_error}; publication state is ambiguous and generation {physical} was retained because alias verification failed: {verification_error}"
            ))),
        }
    }

    async fn alias_points_to(&self, physical: &str) -> Result<bool, SearchError> {
        let path = format!("{physical}/_alias/{}", self.index_alias);
        let Some(response) = self.request_optional_json(Method::GET, &path).await? else {
            return Ok(false);
        };
        Ok(alias_response_points_to(
            &response,
            physical,
            &self.index_alias,
        ))
    }

    async fn cleanup_unpublished_generation(
        &self,
        physical: &str,
        cause: SearchError,
    ) -> SearchError {
        match self.request_json(Method::DELETE, physical, None).await {
            Ok(_) => cause,
            Err(cleanup_error) => response_error(format!(
                "{cause}; failed to remove unpublished generation {physical}: {cleanup_error}"
            )),
        }
    }

    async fn request_json(
        &self,
        method: Method,
        path: &str,
        body: Option<Value>,
    ) -> Result<Value, SearchError> {
        let url = self.url(path);
        let mut request = self.client.request(method, url);
        if let Some(body) = body {
            request = request.json(&body);
        }
        decode_response(request.send().await.map_err(transport_error)?).await
    }

    async fn request_optional_json(
        &self,
        method: Method,
        path: &str,
    ) -> Result<Option<Value>, SearchError> {
        let response = self
            .client
            .request(method, self.url(path))
            .send()
            .await
            .map_err(transport_error)?;
        if response.status() == StatusCode::NOT_FOUND {
            let bytes = response.bytes().await.map_err(transport_error)?;
            if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
                return Err(response_error(
                    "Elasticsearch response exceeds safety limit",
                ));
            }
            return Ok(None);
        }
        decode_response(response).await.map(Some)
    }

    async fn request_ndjson(
        &self,
        method: Method,
        path: &str,
        body: String,
    ) -> Result<Value, SearchError> {
        let response = self
            .client
            .request(method, self.url(path))
            .header(CONTENT_TYPE, "application/x-ndjson")
            .body(body)
            .send()
            .await
            .map_err(transport_error)?;
        let value = decode_response(response).await?;
        if value.get("errors").and_then(Value::as_bool) == Some(true) {
            return Err(response_error("Elasticsearch bulk request had item errors"));
        }
        Ok(value)
    }

    fn url(&self, path: &str) -> Url {
        let mut endpoint = self.endpoint.clone();
        let base = endpoint.path().trim_end_matches('/');
        let (route, query) = path
            .split_once('?')
            .map_or((path, None), |(route, query)| (route, Some(query)));
        endpoint.set_path(&format!("{base}/{}", route.trim_start_matches('/')));
        endpoint.set_query(query);
        endpoint
    }
}

#[async_trait]
impl Retriever for ElasticsearchIndex {
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError> {
        query.validate()?;
        options.validate()?;
        self.verify_manifest().await?;
        let query_terms = expanded_terms(&query.text);
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let query_vector = if let Some(embedder) = &self.embedder {
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
        let search_path = format!("{}/_search", self.index_alias);
        let lexical_request = self.request_json(
            Method::POST,
            &search_path,
            Some(lexical_search_body(
                &query_terms,
                options.candidate_limit.get(),
            )),
        );
        let Some(query_vector) = query_vector else {
            let lexical = parse_ranked_response(&lexical_request.await?)?;
            return Ok(fuse_ranked_results(&lexical, &[], options.top_k.get()));
        };
        let dense_request = self.request_json(
            Method::POST,
            &search_path,
            Some(dense_search_body(
                &query_vector,
                options.candidate_limit.get(),
            )),
        );
        let (lexical, dense) = futures::try_join!(lexical_request, dense_request)?;
        let lexical = parse_ranked_response(&lexical)?;
        let dense = parse_ranked_response(&dense)?;
        Ok(fuse_ranked_results(&lexical, &dense, options.top_k.get()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            lexical: true,
            dense: self.embedder.is_some(),
            quantized_rescore: self.embedder.is_some(),
            ..Capabilities::default()
        }
    }
}

fn alias_swap_body(alias: &str, physical: &str, obsolete: &[String]) -> Value {
    let mut actions = vec![
        json!({
            "remove": {
                "index": "*",
                "alias": alias,
                "must_exist": false
            }
        }),
        json!({
            "add": {
                "index": physical,
                "alias": alias,
                "is_write_index": true
            }
        }),
    ];
    actions.extend(
        obsolete
            .iter()
            .map(|index| json!({ "remove_index": { "index": index } })),
    );
    json!({ "actions": actions })
}

fn alias_response_points_to(response: &Value, physical: &str, alias: &str) -> bool {
    response
        .get(physical)
        .and_then(|metadata| metadata.get("aliases"))
        .and_then(Value::as_object)
        .is_some_and(|aliases| aliases.contains_key(alias))
}

fn generation_cleanup_candidates(
    response: &Value,
    alias: &str,
    retention: usize,
) -> Result<Vec<String>, SearchError> {
    let indices = response
        .as_object()
        .ok_or_else(|| response_error("generation listing is not an object"))?;
    let mut active = 0_usize;
    let mut orphaned = Vec::new();
    for (name, metadata) in indices {
        let Some(generation) = physical_generation(name, alias) else {
            continue;
        };
        let is_active = metadata
            .get("aliases")
            .and_then(Value::as_object)
            .is_some_and(|aliases| aliases.contains_key(alias));
        if is_active {
            active = active.saturating_add(1);
        } else {
            orphaned.push((generation, name.clone()));
        }
    }
    if active == 0 {
        return Ok(Vec::new());
    }
    orphaned.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let retained_orphans = retention.saturating_sub(active.saturating_add(1));
    Ok(orphaned
        .into_iter()
        .skip(retained_orphans)
        .map(|(_, name)| name)
        .collect())
}

fn physical_generation(name: &str, alias: &str) -> Option<(u64, u32)> {
    let suffix = name.strip_prefix(&format!("{alias}-build-"))?;
    let (seconds, nanos) = suffix.split_once('-')?;
    if nanos.contains('-') {
        return None;
    }
    let seconds = seconds.parse().ok()?;
    let nanos = nanos.parse().ok()?;
    (nanos < 1_000_000_000).then_some((seconds, nanos))
}

fn create_index_body(manifest: &IndexManifest) -> Value {
    let index_type = match manifest.quantization {
        hay_search::Quantization::ElasticBbq => "bbq_hnsw",
        hay_search::Quantization::None | hay_search::Quantization::Int8PerVectorScaleOffset => {
            "int8_hnsw"
        }
    };
    let vector = (manifest.model_id != "none").then(|| {
        json!({
            "type": "dense_vector",
            "dims": manifest.mrl_dim,
            "index": true,
            "similarity": "cosine",
            "index_options": { "type": index_type }
        })
    });
    let mut properties = serde_json::Map::from_iter([
        ("doc_id".into(), json!({ "type": "keyword" })),
        ("path".into(), json!({ "type": "keyword" })),
        ("language".into(), json!({ "type": "keyword" })),
        ("text".into(), json!({ "type": "text", "index": false })),
        (
            "terms".into(),
            json!({ "type": "text", "analyzer": "whitespace" }),
        ),
    ]);
    if let Some(vector) = vector {
        properties.insert("embedding".into(), vector);
    }
    json!({
        "settings": { "number_of_shards": 1 },
        "mappings": {
            "dynamic": "strict",
            "_meta": { "hay_manifest": manifest },
            "properties": properties
        }
    })
}

fn bulk_record(document: &SearchDocument) -> Result<String, SearchError> {
    let action = json!({ "index": { "_id": document.doc_id.as_str() } });
    let mut source = json!({
        "doc_id": document.doc_id.as_str(),
        "path": document.path.as_str(),
        "language": document.language.0.as_str(),
        "text": &document.text,
        "terms": expanded_document_terms(document)
    });
    if let Some(embedding) = &document.embedding {
        source["embedding"] = json!(embedding);
    }
    Ok(format!(
        "{}\n{}\n",
        serde_json::to_string(&action).map_err(response_error)?,
        serde_json::to_string(&source).map_err(response_error)?
    ))
}

fn expanded_document_terms(document: &SearchDocument) -> String {
    let mut terms = analyze_code_terms(&document.text);
    add_weighted_terms(&mut terms, document.path.as_str(), 3);
    add_weighted_terms(&mut terms, document.doc_id.as_str(), 2);
    terms
        .into_iter()
        .flat_map(|(term, frequency)| std::iter::repeat_n(term, frequency as usize))
        .collect::<Vec<_>>()
        .join(" ")
}

fn expanded_terms(text: &str) -> String {
    analyze_code_terms(text)
        .into_keys()
        .collect::<Vec<_>>()
        .join(" ")
}

fn add_weighted_terms(
    terms: &mut std::collections::BTreeMap<String, u32>,
    text: &str,
    weight: u32,
) {
    for (term, frequency) in analyze_code_terms(text) {
        let count = terms.entry(term).or_default();
        *count = count.saturating_add(frequency.saturating_mul(weight));
    }
}

fn lexical_search_body(query: &str, limit: usize) -> Value {
    json!({
        "size": limit,
        "query": { "match": { "terms": query } },
        "sort": [ { "_score": "desc" }, { "doc_id": "asc" } ]
    })
}

fn dense_search_body(vector: &[f32], limit: usize) -> Value {
    json!({
        "size": limit,
        "knn": {
            "field": "embedding",
            "query_vector": vector,
            "k": limit,
            "num_candidates": limit
        },
        "sort": [ { "_score": "desc" }, { "doc_id": "asc" } ]
    })
}

fn parse_ranked_response(value: &Value) -> Result<Vec<(DocId, f32)>, SearchError> {
    let hits = value
        .pointer("/hits/hits")
        .and_then(Value::as_array)
        .ok_or_else(|| response_error("search response is missing hits"))?;
    hits.iter()
        .map(|hit| {
            let id = hit
                .get("_id")
                .and_then(Value::as_str)
                .ok_or_else(|| response_error("search hit is missing _id"))?;
            let score = hit
                .get("_score")
                .and_then(Value::as_f64)
                .ok_or_else(|| response_error("search hit is missing _score"))?;
            let score = score.to_string().parse::<f32>().map_err(response_error)?;
            Ok((DocId::new(id).map_err(response_error)?, score))
        })
        .collect()
}

fn validate_reindex_response(value: &Value) -> Result<(), SearchError> {
    if value.get("timed_out").and_then(Value::as_bool) == Some(true) {
        return Err(response_error("Elasticsearch reindex timed out"));
    }
    let failures = value
        .get("failures")
        .and_then(Value::as_array)
        .ok_or_else(|| response_error("reindex response is missing failures"))?;
    if !failures.is_empty() {
        return Err(response_error(format!(
            "Elasticsearch reindex had {} failures",
            failures.len()
        )));
    }
    Ok(())
}

fn parse_mget_response(value: &Value) -> Result<Vec<SearchDocument>, SearchError> {
    let documents = value
        .get("docs")
        .and_then(Value::as_array)
        .ok_or_else(|| response_error("mget response is missing docs"))?;
    documents
        .iter()
        .filter(|document| document.get("found").and_then(Value::as_bool) != Some(false))
        .map(|document| {
            let source = document
                .get("_source")
                .ok_or_else(|| response_error("mget document is missing _source"))?;
            Ok(SearchDocument {
                doc_id: DocId::new(required_string(source, "doc_id")?).map_err(response_error)?,
                path: cast_index::NormalizedPath::new(required_string(source, "path")?)
                    .map_err(response_error)?,
                language: cast_core::LanguageId::new(required_string(source, "language")?),
                text: required_string(source, "text")?.into(),
                embedding: source
                    .get("embedding")
                    .filter(|value| !value.is_null())
                    .map(|value| serde_json::from_value(value.clone()))
                    .transpose()
                    .map_err(response_error)?,
            })
        })
        .collect()
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str, SearchError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| response_error(format!("document is missing {field}")))
}

async fn decode_response(response: reqwest::Response) -> Result<Value, SearchError> {
    let status = response.status();
    let bytes = response.bytes().await.map_err(transport_error)?;
    decode_response_bytes(status, &bytes)
}

fn decode_response_bytes(status: StatusCode, bytes: &[u8]) -> Result<Value, SearchError> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_RESPONSE_BYTES {
        return Err(response_error(
            "Elasticsearch response exceeds safety limit",
        ));
    }
    if !status.is_success() {
        let error = serde_json::from_slice::<Value>(bytes).ok();
        let kind = error
            .as_ref()
            .and_then(|value| value.pointer("/error/type"))
            .and_then(Value::as_str)
            .unwrap_or("non-json-or-empty-response");
        let reason = error
            .as_ref()
            .and_then(|value| value.pointer("/error/reason"))
            .and_then(Value::as_str)
            .map(|reason| reason.chars().take(240).collect::<String>());
        let details = reason.map_or_else(|| kind.to_owned(), |reason| format!("{kind}: {reason}"));
        return Err(response_error(format!(
            "Elasticsearch returned {status} ({details})"
        )));
    }
    if bytes.is_empty() {
        return Err(response_error(format!(
            "Elasticsearch returned {status} with an empty response"
        )));
    }
    serde_json::from_slice::<Value>(bytes).map_err(response_error)
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
        return Err(SearchError::InvalidConfig(
            "embedder identity does not match Elasticsearch manifest".into(),
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
        return Err(response_error("embedding dimensions do not match manifest"));
    }
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(response_error("embedding contains a non-finite value"));
    }
    vector.truncate(stored_dimensions);
    Ok(vector)
}

fn validate_index_name(name: &str) -> Result<(), SearchError> {
    if name.is_empty()
        || name.len() > 180
        || name.starts_with(['_', '-', '+'])
        || name.chars().any(|character| {
            !character.is_ascii_lowercase()
                && !character.is_ascii_digit()
                && character != '-'
                && character != '_'
        })
    {
        return Err(SearchError::InvalidConfig(
            "Elasticsearch alias must use lowercase letters, digits, '-' or '_'".into(),
        ));
    }
    Ok(())
}

fn is_loopback(url: &Url) -> bool {
    matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"))
}

fn transport_error(error: impl fmt::Display) -> SearchError {
    SearchError::Retriever(format!("Elasticsearch transport: {error}"))
}

fn response_error(error: impl fmt::Display) -> SearchError {
    SearchError::Retriever(format!("Elasticsearch response: {error}"))
}

#[cfg(test)]
mod tests {
    use cast_core::LanguageId;
    use cast_index::{ContentHash, HashAlgorithm, NormalizedPath};
    use hay_search::{FdeParams, Quantization};

    use super::*;

    fn manifest(vector: bool) -> IndexManifest {
        if !vector {
            return IndexManifest::lexical_v1();
        }
        IndexManifest {
            model_id: "encoder".into(),
            model_revision: "revision".into(),
            embedding_profile: "retrieval-v1".into(),
            embed_dim: 768,
            mrl_dim: 256,
            quantization: Quantization::ElasticBbq,
            tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "a".repeat(64)).unwrap(),
            chunker_version: "cast-rust-0.1".into(),
            fde_params: FdeParams::Disabled,
            schema_version: 1,
        }
    }

    fn document() -> SearchDocument {
        SearchDocument {
            doc_id: DocId::new("manifest").unwrap(),
            path: NormalizedPath::new("src/manifest.rs").unwrap(),
            language: LanguageId::new("rust"),
            text: "validate manifest compatibility".into(),
            embedding: None,
        }
    }

    #[test]
    fn mapping_persists_manifest_and_vector_contract() {
        let body = create_index_body(&manifest(true));

        assert_eq!(
            body.pointer("/mappings/properties/embedding/dims"),
            Some(&json!(256))
        );
        assert_eq!(
            body.pointer("/mappings/properties/embedding/index_options/type"),
            Some(&json!("bbq_hnsw"))
        );
        assert!(body.pointer("/mappings/_meta/hay_manifest").is_some());
    }

    #[test]
    fn bulk_is_ndjson_and_expands_code_terms() {
        let body = bulk_record(&document()).unwrap();
        let lines = body.lines().collect::<Vec<_>>();

        assert_eq!(lines.len(), 2);
        assert!(body.ends_with('\n'));
        assert!(lines[1].contains("manifest"));
        assert!(lines[1].contains("compatibility"));
        assert!(!lines[1].contains("embedding"));
    }

    #[test]
    fn non_json_error_reports_status_instead_of_json_eof() {
        let error = decode_response_bytes(StatusCode::PAYLOAD_TOO_LARGE, &[]).unwrap_err();

        assert!(error.to_string().contains("413 Payload Too Large"));
        assert!(!error.to_string().contains("EOF"));
    }

    #[test]
    fn hybrid_query_uses_license_independent_native_candidate_searches() {
        let lexical = lexical_search_body("manifest", 50);
        let dense = dense_search_body(&vec![0.0; 256], 50);

        assert_eq!(lexical.pointer("/size"), Some(&json!(50)));
        assert_eq!(dense.pointer("/knn/k"), Some(&json!(50)));
        assert!(lexical.pointer("/retriever").is_none());
        assert!(dense.pointer("/retriever").is_none());
    }

    #[test]
    fn search_response_is_stably_decoded() {
        let response = json!({
            "hits": { "hits": [
                { "_id": "a", "_score": 2.0 },
                { "_id": "b", "_score": 1.0 }
            ] }
        });
        let candidates = parse_ranked_response(&response).unwrap();

        assert_eq!(candidates[0].0.as_str(), "a");
        assert!((candidates[0].1 - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reindex_response_rejects_timeouts_and_item_failures() {
        validate_reindex_response(&json!({ "timed_out": false, "failures": [] })).unwrap();
        assert!(validate_reindex_response(&json!({ "timed_out": true, "failures": [] })).is_err());
        assert!(
            validate_reindex_response(&json!({
                "timed_out": false,
                "failures": [{ "cause": { "type": "mapper_parsing_exception" } }]
            }))
            .is_err()
        );
    }

    #[test]
    fn generation_retention_is_bounded_and_preserves_one_rollback_index() {
        let response = json!({
            "search-build-300-3": { "aliases": { "search": {} } },
            "search-build-200-2": { "aliases": {} },
            "search-build-100-1": { "aliases": {} },
            "search-build-not-a-generation": { "aliases": {} },
            "someone-else-build-50-1": { "aliases": {} }
        });

        assert_eq!(
            generation_cleanup_candidates(&response, "search", 3).unwrap(),
            vec!["search-build-100-1"]
        );
        assert_eq!(
            generation_cleanup_candidates(&response, "search", 2).unwrap(),
            vec!["search-build-200-2", "search-build-100-1"]
        );
        assert_eq!(
            physical_generation("search-build-300-3", "search"),
            Some((300, 3))
        );
        assert_eq!(
            physical_generation("search-build-300-1000000000", "search"),
            None
        );

        let swap = alias_swap_body(
            "search",
            "search-build-400-4",
            &["search-build-100-1".into()],
        );
        assert_eq!(
            swap.pointer("/actions/2/remove_index/index"),
            Some(&json!("search-build-100-1"))
        );
        let published = json!({
            "search-build-400-4": { "aliases": { "search": { "is_write_index": true } } }
        });
        assert!(alias_response_points_to(
            &published,
            "search-build-400-4",
            "search"
        ));
        assert!(!alias_response_points_to(
            &published,
            "search-build-300-3",
            "search"
        ));
    }

    #[test]
    fn generation_cleanup_refuses_to_delete_without_an_active_alias_target() {
        let orphan_only = json!({
            "search-build-200-2": { "aliases": {} },
            "search-build-100-1": { "aliases": {} }
        });

        assert!(
            generation_cleanup_candidates(&orphan_only, "search", 2)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn invalid_generation_retention_fails_before_network_access() {
        let config = ElasticsearchConfig::new("http://127.0.0.1:9200", "search")
            .with_generation_retention(1);
        let error = ElasticsearchIndex::new(config, IndexManifest::lexical_v1(), None)
            .err()
            .unwrap();

        assert!(error.to_string().contains("retention must be between 2"));
    }
}
