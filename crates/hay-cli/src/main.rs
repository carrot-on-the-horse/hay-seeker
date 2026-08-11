#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::future::Future;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use cast_core::LanguageId;
use cast_index::{DocumentId, Embedder, NormalizedPath};
use clap::{Parser, Subcommand, ValueEnum};
use hay_duckdb::DuckDbIndex;
use hay_elasticsearch::{ElasticsearchConfig, ElasticsearchIndex};
use hay_runtime::{EmbeddingProvider, SearchRuntime, StorageBackend};
use hay_search::{
    Candidate, IndexManifest, Query, Retriever, SearchDocument, SearchError, SearchOpts,
};
use ring::digest::{SHA256, digest};
use serde::{Deserialize, Serialize};

mod repository;

use repository::{
    RepositoryCheckpoint, RepositoryChunkStream, RepositoryProgress, RepositoryStats,
};

#[derive(Debug, Parser)]
#[command(
    name = "hay",
    about = "Index and search source corpora with Hay Seeker"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Atomically rebuild an index from JSONL or a CAST-chunked repository.
    Index {
        /// Storage backend.
        #[arg(long, value_enum, default_value = "duckdb")]
        backend: Backend,
        /// Optional dense embedding provider.
        #[arg(long, value_enum, default_value = "none")]
        embeddings: Embeddings,
        /// Destination `DuckDB` file.
        #[arg(long, default_value = ".hay-seeker/index.duckdb")]
        database: PathBuf,
        /// Source JSONL containing `doc_id`, path, language, and text.
        #[arg(
            long,
            conflicts_with = "repository",
            required_unless_present = "repository"
        )]
        corpus: Option<PathBuf>,
        /// Git repository (or non-Git source directory) to scan and CAST-chunk.
        #[arg(long, conflicts_with = "corpus", required_unless_present = "corpus")]
        repository: Option<PathBuf>,
        /// Incremental repository checkpoint (derived automatically when omitted).
        #[arg(long, requires = "repository")]
        checkpoint: Option<PathBuf>,
        /// Abort after this many seconds without a completed or skipped file.
        #[arg(long, default_value_t = 600)]
        stall_timeout_seconds: u64,
        /// Emit repository progress JSON to stderr at this interval.
        #[arg(long, default_value_t = 5)]
        progress_interval_seconds: u64,
        /// Elasticsearch base URL.
        #[arg(long, default_value = "http://127.0.0.1:9200")]
        endpoint: String,
        /// Stable Elasticsearch index alias.
        #[arg(long, default_value = "hay-seeker")]
        index: String,
    },
    /// Search an existing local or remote index.
    Search {
        /// Storage backend.
        #[arg(long, value_enum, default_value = "duckdb")]
        backend: Backend,
        /// Must match the provider used when the index was built.
        #[arg(long, value_enum, default_value = "none")]
        embeddings: Embeddings,
        /// `DuckDB` index file.
        #[arg(long, default_value = ".hay-seeker/index.duckdb")]
        database: PathBuf,
        /// Elasticsearch base URL.
        #[arg(long, default_value = "http://127.0.0.1:9200")]
        endpoint: String,
        /// Stable Elasticsearch index alias.
        #[arg(long, default_value = "hay-seeker")]
        index: String,
        /// Number of final results.
        #[arg(long, default_value_t = 10)]
        top_k: usize,
        /// BM25/vector candidates retained before fusion.
        #[arg(long, default_value_t = 50)]
        candidate_limit: usize,
        /// Natural-language or code query.
        query: String,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Duckdb,
    Elasticsearch,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Embeddings {
    None,
    LocalOnnx,
    LocalStatic,
    Gemini,
    OpenAi,
    Voyage,
    CloudflareWorkersAi,
}

#[derive(Debug, Deserialize)]
struct RawDocument {
    doc_id: String,
    path: String,
    language: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct IndexResult {
    backend: &'static str,
    target: String,
    documents: usize,
    mode: &'static str,
    total_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    repository: Option<RepositoryStats>,
    manifest: IndexManifest,
}

#[derive(Debug, Serialize)]
struct SearchResult {
    backend: &'static str,
    query: String,
    results: Vec<SearchHit>,
}

#[derive(Debug, Serialize)]
struct SearchHit {
    candidate: Candidate,
    path: String,
    language: String,
    text: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    match Arguments::parse().command {
        Command::Index {
            backend,
            embeddings,
            database,
            corpus,
            repository,
            checkpoint,
            stall_timeout_seconds,
            progress_interval_seconds,
            endpoint,
            index,
        } => {
            index_source(
                backend,
                embeddings,
                &database,
                corpus.as_deref(),
                repository.as_deref(),
                checkpoint.as_deref(),
                stall_timeout_seconds,
                progress_interval_seconds,
                &endpoint,
                &index,
            )
            .await
        }
        Command::Search {
            backend,
            embeddings,
            database,
            endpoint,
            index,
            top_k,
            candidate_limit,
            query,
        } => {
            search(
                backend,
                embeddings,
                &database,
                &endpoint,
                &index,
                query,
                top_k,
                candidate_limit,
            )
            .await
        }
    }
}

enum IndexDocuments {
    Corpus(Vec<SearchDocument>),
    Repository {
        stream: Box<RepositoryChunkStream>,
        progress: RepositoryProgress,
        checkpoint: PathBuf,
        incremental: bool,
    },
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn index_source(
    backend: Backend,
    embeddings: Embeddings,
    database: &Path,
    corpus: Option<&Path>,
    repository: Option<&Path>,
    checkpoint: Option<&Path>,
    stall_timeout_seconds: u64,
    progress_interval_seconds: u64,
    endpoint: &str,
    index_alias: &str,
) -> Result<()> {
    let started = Instant::now();
    let stall_timeout = positive_duration(stall_timeout_seconds, "stall-timeout-seconds")?;
    let progress_interval =
        positive_duration(progress_interval_seconds, "progress-interval-seconds")?;
    let SearchRuntime { manifest, embedder } = search_runtime(backend, embeddings)?;
    let documents = match (corpus, repository) {
        (Some(corpus), None) => IndexDocuments::Corpus(load_corpus(corpus)?),
        (None, Some(repository)) => {
            let checkpoint = checkpoint.map_or_else(
                || default_checkpoint_path(backend, database, repository, index_alias),
                Path::to_owned,
            );
            let previous = load_checkpoint(&checkpoint)?;
            let compatible = previous
                .as_ref()
                .map_or(Ok(false), |state| state.matches(repository, &manifest))?;
            let incremental = compatible
                && match backend {
                    Backend::Duckdb => database.is_file(),
                    Backend::Elasticsearch => true,
                };
            let previous = incremental.then_some(previous).flatten();
            let (stream, progress) =
                RepositoryChunkStream::open_incremental(repository, &manifest, previous)?;
            progress.configure_progress_reporting(progress_interval)?;
            invalidate_checkpoint(&checkpoint)?;
            IndexDocuments::Repository {
                stream: Box::new(stream),
                progress,
                checkpoint,
                incremental,
            }
        }
        _ => bail!("exactly one of --corpus or --repository is required"),
    };
    let mode = match &documents {
        IndexDocuments::Corpus(_) => "corpus",
        IndexDocuments::Repository {
            incremental: true, ..
        } => "incremental",
        IndexDocuments::Repository { .. } => "full",
    };
    let (backend_name, target, document_count, repository) = match backend {
        Backend::Duckdb => {
            let (document_count, repository) = match documents {
                IndexDocuments::Repository {
                    stream,
                    progress,
                    checkpoint,
                    incremental: true,
                } => {
                    let index = DuckDbIndex::open(database, manifest.clone(), embedder)?;
                    let deletion_progress = progress.clone();
                    monitor_repository_run(
                        index.update_stream(*stream, move || deletion_progress.deletions()),
                        &progress,
                        stall_timeout,
                        progress_interval,
                    )
                    .await?;
                    let count = index.document_count()?;
                    persist_checkpoint(&checkpoint, &progress.checkpoint()?)?;
                    (count, Some(progress.snapshot()))
                }
                documents => {
                    let (count, repository) = rebuild_duckdb(
                        database,
                        manifest.clone(),
                        embedder,
                        documents,
                        stall_timeout,
                        progress_interval,
                    )
                    .await?;
                    (count, repository)
                }
            };
            (
                "duckdb",
                database.display().to_string(),
                document_count,
                repository,
            )
        }
        Backend::Elasticsearch => {
            let index = elasticsearch(endpoint, index_alias, manifest.clone(), embedder)?;
            let (document_count, repository) = match documents {
                IndexDocuments::Corpus(documents) => {
                    index.replace_all(&documents).await?;
                    (index.document_count().await?, None)
                }
                IndexDocuments::Repository {
                    stream,
                    progress,
                    checkpoint,
                    incremental,
                } => {
                    let indexed = if incremental {
                        let deletion_progress = progress.clone();
                        monitor_repository_run(
                            index.update_stream(*stream, move || deletion_progress.deletions()),
                            &progress,
                            stall_timeout,
                            progress_interval,
                        )
                        .await?
                    } else {
                        monitor_repository_run(
                            index.replace_stream(*stream),
                            &progress,
                            stall_timeout,
                            progress_interval,
                        )
                        .await?
                    };
                    persist_checkpoint(&checkpoint, &progress.checkpoint()?)?;
                    let total = index.document_count().await?;
                    debug_assert!(total >= indexed || !incremental);
                    (total, Some(progress.snapshot()))
                }
            };
            (
                "elasticsearch",
                format!("{endpoint}/{index_alias}"),
                document_count,
                repository,
            )
        }
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&IndexResult {
            backend: backend_name,
            target,
            documents: document_count,
            mode,
            total_ms: duration_millis(started.elapsed()),
            repository,
            manifest,
        })?
    );
    Ok(())
}

async fn rebuild_duckdb(
    database: &Path,
    manifest: IndexManifest,
    embedder: Option<Arc<dyn Embedder>>,
    documents: IndexDocuments,
    stall_timeout: Duration,
    progress_interval: Duration,
) -> Result<(usize, Option<RepositoryStats>)> {
    let parent = database
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create index directory {}", parent.display()))?;
    let temporary_directory = tempfile::Builder::new()
        .prefix(".hay-rebuild-")
        .tempdir_in(parent)
        .with_context(|| format!("create temporary index directory in {}", parent.display()))?;
    let temporary = temporary_directory.path().join("index.duckdb");
    let index = DuckDbIndex::open(&temporary, manifest, embedder)?;
    let result = match documents {
        IndexDocuments::Corpus(documents) => {
            index.replace_all(&documents).await?;
            (index.document_count()?, None)
        }
        IndexDocuments::Repository {
            stream,
            progress,
            checkpoint,
            ..
        } => {
            let indexed = monitor_repository_run(
                index.replace_stream(*stream),
                &progress,
                stall_timeout,
                progress_interval,
            )
            .await?;
            (
                indexed,
                Some((progress.snapshot(), checkpoint, progress.checkpoint()?)),
            )
        }
    };
    drop(index);
    std::fs::rename(&temporary, database)
        .with_context(|| format!("atomically replace DuckDB index {}", database.display()))?;
    match result {
        (count, Some((statistics, checkpoint, state))) => {
            persist_checkpoint(&checkpoint, &state)?;
            Ok((count, Some(statistics)))
        }
        (count, None) => Ok((count, None)),
    }
}

fn default_checkpoint_path(
    backend: Backend,
    database: &Path,
    repository: &Path,
    index_alias: &str,
) -> PathBuf {
    match backend {
        Backend::Duckdb => {
            let mut value = database.as_os_str().to_owned();
            value.push(".checkpoint.json");
            PathBuf::from(value)
        }
        Backend::Elasticsearch => repository.join(".hay-seeker").join(format!(
            "elasticsearch-{}.checkpoint.json",
            &sha256_hex(index_alias.as_bytes())[..16]
        )),
    }
}

fn load_checkpoint(path: &Path) -> Result<Option<RepositoryCheckpoint>> {
    match File::open(path) {
        Ok(file) => RepositoryCheckpoint::from_reader(BufReader::new(file)).map(Some),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error).with_context(|| format!("open checkpoint {}", path.display())),
    }
}

fn invalidate_checkpoint(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => {
            Err(error).with_context(|| format!("invalidate checkpoint {}", path.display()))
        }
    }
}

fn persist_checkpoint(path: &Path, checkpoint: &RepositoryCheckpoint) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("create checkpoint directory {}", parent.display()))?;
    let temporary = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create temporary checkpoint in {}", parent.display()))?;
    let mut writer = BufWriter::new(temporary);
    serde_json::to_writer(&mut writer, checkpoint).context("serialize repository checkpoint")?;
    writer
        .write_all(b"\n")
        .context("write repository checkpoint")?;
    writer.flush().context("flush repository checkpoint")?;
    let temporary = writer
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)
        .context("finish repository checkpoint")?;
    temporary
        .as_file()
        .sync_all()
        .context("sync repository checkpoint")?;
    temporary
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish checkpoint {}", path.display()))?;
    Ok(())
}

async fn monitor_repository_run<F, T>(
    future: F,
    progress: &RepositoryProgress,
    stall_timeout: Duration,
    progress_interval: Duration,
) -> Result<T>
where
    F: Future<Output = Result<T, SearchError>>,
{
    tokio::pin!(future);
    let cadence = progress_interval.min(stall_timeout);
    loop {
        tokio::select! {
            result = &mut future => return result.map_err(anyhow::Error::new),
            () = tokio::time::sleep(cadence) => {
                let inactive = progress.inactive_for()?;
                if inactive >= stall_timeout {
                    bail!(
                        "repository indexing stalled for {} seconds without a completed or skipped file",
                        inactive.as_secs()
                    );
                }
                progress.emit_progress_if_due();
            }
        }
    }
}

fn positive_duration(seconds: u64, name: &str) -> Result<Duration> {
    if seconds == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(Duration::from_secs(seconds))
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

#[allow(clippy::too_many_arguments)]
async fn search(
    backend: Backend,
    embeddings: Embeddings,
    database: &Path,
    endpoint: &str,
    index_alias: &str,
    query: String,
    top_k: usize,
    candidate_limit: usize,
) -> Result<()> {
    let top_k = NonZeroUsize::new(top_k).context("top-k must be greater than zero")?;
    let candidate_limit =
        NonZeroUsize::new(candidate_limit).context("candidate-limit must be greater than zero")?;
    let options = SearchOpts {
        top_k,
        candidate_limit,
        enable_late_interaction: false,
    };
    options.validate()?;
    let query = Query::new(query)?;
    let SearchRuntime { manifest, embedder } = search_runtime(backend, embeddings)?;
    let (backend_name, candidates, indexed_documents) = match backend {
        Backend::Duckdb => {
            let index = DuckDbIndex::open(database, manifest, embedder)?;
            let candidates = index.search(&query, &options).await?;
            let ids = candidate_ids(&candidates);
            ("duckdb", candidates, index.documents(&ids)?)
        }
        Backend::Elasticsearch => {
            let index = elasticsearch(endpoint, index_alias, manifest, embedder)?;
            let candidates = index.search(&query, &options).await?;
            let ids = candidate_ids(&candidates);
            ("elasticsearch", candidates, index.documents(&ids).await?)
        }
    };
    let documents = indexed_documents
        .into_iter()
        .map(|document| (document.doc_id.clone(), document))
        .collect::<BTreeMap<_, _>>();
    let results = candidates
        .into_iter()
        .map(|candidate| {
            let document = documents
                .get(&candidate.doc_id)
                .with_context(|| format!("missing result document {}", candidate.doc_id))?;
            Ok(SearchHit {
                candidate,
                path: document.path.as_str().to_owned(),
                language: document.language.0.clone(),
                text: document.text.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&SearchResult {
            backend: backend_name,
            query: query.text,
            results,
        })?
    );
    Ok(())
}

fn candidate_ids(candidates: &[Candidate]) -> Vec<DocumentId> {
    candidates
        .iter()
        .map(|candidate| candidate.doc_id.clone())
        .collect()
}

fn elasticsearch(
    endpoint: &str,
    index_alias: &str,
    manifest: IndexManifest,
    embedder: Option<Arc<dyn Embedder>>,
) -> Result<ElasticsearchIndex> {
    let mut config = ElasticsearchConfig::new(endpoint, index_alias);
    if let Ok(api_key) = std::env::var("ELASTICSEARCH_API_KEY") {
        config = config.with_api_key(api_key);
    } else if let Ok(token) = std::env::var("ELASTICSEARCH_BEARER_TOKEN") {
        config = config.with_bearer_token(token);
    }
    Ok(ElasticsearchIndex::new(config, manifest, embedder)?)
}

fn search_runtime(backend: Backend, embeddings: Embeddings) -> Result<SearchRuntime> {
    SearchRuntime::from_env(
        match backend {
            Backend::Duckdb => StorageBackend::DuckDb,
            Backend::Elasticsearch => StorageBackend::Elasticsearch,
        },
        match embeddings {
            Embeddings::None => EmbeddingProvider::None,
            Embeddings::LocalOnnx => EmbeddingProvider::LocalOnnx,
            Embeddings::LocalStatic => EmbeddingProvider::LocalStatic,
            Embeddings::Gemini => EmbeddingProvider::Gemini,
            Embeddings::OpenAi => EmbeddingProvider::OpenAi,
            Embeddings::Voyage => EmbeddingProvider::Voyage,
            Embeddings::CloudflareWorkersAi => EmbeddingProvider::CloudflareWorkersAi,
        },
    )
}

fn sha256_hex(value: &[u8]) -> String {
    const LOWER_HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = digest(&SHA256, value);
    let mut encoded = String::with_capacity(digest.as_ref().len() * 2);
    for byte in digest.as_ref() {
        encoded.push(char::from(LOWER_HEX[usize::from(*byte >> 4)]));
        encoded.push(char::from(LOWER_HEX[usize::from(*byte & 0x0f)]));
    }
    encoded
}

fn load_corpus(path: &Path) -> Result<Vec<SearchDocument>> {
    let file = File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    load_corpus_reader(BufReader::new(file), path)
}

fn load_corpus_reader(reader: impl BufRead, path: &Path) -> Result<Vec<SearchDocument>> {
    let mut ids = BTreeSet::new();
    let documents = reader
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            result => Some((index, result)),
        })
        .map(|(index, line)| {
            let line =
                line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            let raw: RawDocument = serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))?;
            let doc_id = DocumentId::new(raw.doc_id)
                .with_context(|| format!("validate document ID on line {}", index + 1))?;
            if !ids.insert(doc_id.clone()) {
                bail!("duplicate document ID {} on line {}", doc_id, index + 1);
            }
            Ok(SearchDocument {
                doc_id,
                path: NormalizedPath::new(raw.path)
                    .with_context(|| format!("validate path on line {}", index + 1))?,
                language: LanguageId::new(raw.language),
                text: raw.text,
                embedding: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if documents.is_empty() {
        bail!("corpus {} is empty", path.display());
    }
    Ok(documents)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn corpus_loader_rejects_duplicate_ids() {
        let input = concat!(
            r#"{"doc_id":"same","path":"one.rs","language":"rust","text":"one"}"#,
            "\n",
            r#"{"doc_id":"same","path":"two.rs","language":"rust","text":"two"}"#,
            "\n"
        );
        let error = load_corpus_reader(Cursor::new(input), Path::new("test.jsonl")).unwrap_err();
        assert!(error.to_string().contains("duplicate document ID"));
    }

    #[test]
    fn elasticsearch_checkpoint_name_does_not_trust_alias_as_a_path() {
        let path = default_checkpoint_path(
            Backend::Elasticsearch,
            Path::new("unused.duckdb"),
            Path::new("/repository"),
            "../../outside",
        );
        assert_eq!(path.parent(), Some(Path::new("/repository/.hay-seeker")));
        assert!(
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("elasticsearch-"))
        );
        assert_eq!(
            path.extension().and_then(|value| value.to_str()),
            Some("json")
        );
    }

    #[test]
    fn operational_durations_reject_zero_seconds() {
        assert!(positive_duration(0, "stall-timeout-seconds").is_err());
        assert_eq!(
            positive_duration(5, "stall-timeout-seconds").unwrap(),
            Duration::from_secs(5)
        );
    }

    #[tokio::test]
    async fn repository_monitor_aborts_a_stalled_run() {
        let directory = tempfile::tempdir().unwrap();
        let (stream, progress) = RepositoryChunkStream::open_incremental(
            directory.path(),
            &IndexManifest::lexical_v1(),
            None,
        )
        .unwrap();
        drop(stream);
        let error = monitor_repository_run(
            std::future::pending::<Result<(), SearchError>>(),
            &progress,
            Duration::from_millis(10),
            Duration::from_millis(2),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("indexing stalled"));
    }

    #[tokio::test]
    async fn failed_cli_rebuild_preserves_previous_duckdb_file() {
        let directory = tempfile::tempdir().unwrap();
        let database = directory.path().join("index.duckdb");
        let manifest = IndexManifest::lexical_v1();
        let previous = SearchDocument {
            doc_id: DocumentId::new("previous").unwrap(),
            path: NormalizedPath::new("src/previous.rs").unwrap(),
            language: LanguageId::new("rust"),
            text: "previous searchable generation".into(),
            embedding: None,
        };
        let index = DuckDbIndex::open(&database, manifest.clone(), None).unwrap();
        index.replace_all(&[previous]).await.unwrap();
        drop(index);
        let invalid = SearchDocument {
            doc_id: DocumentId::new("invalid").unwrap(),
            path: NormalizedPath::new("src/invalid.rs").unwrap(),
            language: LanguageId::new("rust"),
            text: "   ".into(),
            embedding: None,
        };

        rebuild_duckdb(
            &database,
            manifest.clone(),
            None,
            IndexDocuments::Corpus(vec![invalid]),
            Duration::from_secs(1),
            Duration::from_millis(10),
        )
        .await
        .unwrap_err();

        let preserved = DuckDbIndex::open(database, manifest, None).unwrap();
        assert_eq!(preserved.document_count().unwrap(), 1);
        assert_eq!(
            preserved
                .documents(&[DocumentId::new("previous").unwrap()])
                .unwrap()
                .len(),
            1
        );
    }
}
