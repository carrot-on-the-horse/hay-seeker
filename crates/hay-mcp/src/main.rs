#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use cast_core::LanguageId;
use cast_index::{DocumentId, Embedder, NormalizedPath};
use clap::parser::ValueSource;
use clap::{ArgAction, CommandFactory as _, FromArgMatches as _, Parser, ValueEnum};
use hay_duckdb::DuckDbIndex;
use hay_elasticsearch::{ElasticsearchConfig, ElasticsearchIndex};
use hay_runtime::{
    EmbeddingProvider, SearchRuntime, StorageBackend, Workspace, ensure_models, load_dotenv,
    report_to_stderr,
};
use hay_search::{
    Candidate, Capabilities, DeterministicPhase0Retriever, IndexManifest, Query, Retriever,
    SearchDocument, SearchError, SearchOpts,
};
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{
        router::tool::ToolRouter,
        wrapper::{Json, Parameters},
    },
    model::{Implementation, ServerCapabilities, ServerInfo},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::{Deserialize, Serialize};

const DEFAULT_TOP_K: usize = 10;
const DEFAULT_CANDIDATE_LIMIT: usize = 50;
const MAX_TOP_K: usize = 50;
const MAX_CANDIDATE_LIMIT: usize = 1_000;

#[derive(Debug, Parser)]
#[command(
    name = "hay-mcp",
    about = "Expose Hay Seeker's backend-neutral search contract over MCP stdio"
)]
struct Arguments {
    /// Search backend exposed through MCP.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_BACKEND",
        hide_env_values = true,
        value_enum,
        default_value = "duckdb"
    )]
    backend: Backend,
    /// Dense embedding provider. Must match the indexed manifest.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_EMBEDDINGS",
        hide_env_values = true,
        value_enum,
        default_value = "local-static"
    )]
    embeddings: Embeddings,
    /// Provision a missing local model bundle automatically.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_DOWNLOAD_MODELS",
        hide_env_values = true,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        action = ArgAction::Set
    )]
    download_models: bool,
    /// JSONL corpus used by the current Phase 0 search backend.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_CORPUS",
        hide_env_values = true,
        default_value = "evals/corpus.jsonl"
    )]
    corpus: PathBuf,
    /// Existing `DuckDB` index created by `hay index`.
    ///
    /// Defaults to the index of the Git repository containing the current
    /// directory, which is the one `hay index` writes.
    #[arg(long, env = "COTH_HAY_SEEKER_DATABASE", hide_env_values = true)]
    database: Option<PathBuf>,
    /// Elasticsearch base URL.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_ELASTICSEARCH_ENDPOINT",
        hide_env_values = true,
        default_value = "http://127.0.0.1:9200"
    )]
    endpoint: String,
    /// Stable Elasticsearch index alias.
    #[arg(
        long,
        env = "COTH_HAY_SEEKER_ELASTICSEARCH_INDEX",
        hide_env_values = true,
        default_value = "hay-seeker"
    )]
    index: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    Phase0,
    Duckdb,
    Elasticsearch,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
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

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchRequest {
    /// Natural-language or code-search query.
    query: String,
    /// Final number of results, from 1 through 50. Defaults to 10.
    top_k: Option<usize>,
    /// Cascade candidate ceiling, from `top_k` through 1000. Defaults to 50.
    candidate_limit: Option<usize>,
    /// Request late interaction if the active backend advertises support.
    enable_late_interaction: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
struct SearchResponse {
    backend: String,
    query: String,
    result_count: usize,
    capabilities: CapabilityResponse,
    results: Vec<SearchHit>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[allow(clippy::struct_excessive_bools)]
struct CapabilityResponse {
    lexical: bool,
    dense: bool,
    quantized_rescore: bool,
    late_interaction: bool,
    learned_sparse: bool,
    fde: bool,
}

impl From<Capabilities> for CapabilityResponse {
    fn from(value: Capabilities) -> Self {
        Self {
            lexical: value.lexical,
            dense: value.dense,
            quantized_rescore: value.quantized_rescore,
            late_interaction: value.late_interaction,
            learned_sparse: value.learned_sparse,
            fde: value.fde,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, JsonSchema)]
struct SearchHit {
    doc_id: String,
    score: f32,
    signals: SignalResponse,
    path: String,
    language: String,
    text: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, JsonSchema)]
struct SignalResponse {
    lexical: Option<f32>,
    dense: Option<f32>,
    late: Option<f32>,
}

#[derive(Clone)]
struct SearchServer {
    backend: Arc<str>,
    search_backend: Arc<dyn ToolBackend>,
    tool_router: ToolRouter<Self>,
}

#[async_trait]
trait ToolBackend: Send + Sync {
    async fn search_with_documents(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<(Vec<Candidate>, Vec<SearchDocument>), SearchError>;

    async fn document_count(&self) -> Result<usize, SearchError>;

    fn capabilities(&self) -> Capabilities;
}

#[async_trait]
trait DocumentCatalog: Send + Sync {
    async fn document_count(&self) -> Result<usize, SearchError>;
    async fn documents(&self, ids: &[DocumentId]) -> Result<Vec<SearchDocument>, SearchError>;
}

struct MemoryCatalog {
    documents: BTreeMap<String, SearchDocument>,
}

struct FixedBackend {
    retriever: Arc<dyn Retriever>,
    catalog: Arc<dyn DocumentCatalog>,
}

#[async_trait]
impl ToolBackend for FixedBackend {
    async fn search_with_documents(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<(Vec<Candidate>, Vec<SearchDocument>), SearchError> {
        let candidates = self.retriever.search(query, options).await?;
        let ids = candidates
            .iter()
            .map(|candidate| candidate.doc_id.clone())
            .collect::<Vec<_>>();
        let documents = self.catalog.documents(&ids).await?;
        Ok((candidates, documents))
    }

    async fn document_count(&self) -> Result<usize, SearchError> {
        self.catalog.document_count().await
    }

    fn capabilities(&self) -> Capabilities {
        self.retriever.capabilities()
    }
}

struct DuckDbToolBackend {
    database: PathBuf,
    manifest: IndexManifest,
    embedder: Option<Arc<dyn Embedder>>,
    capabilities: Capabilities,
}

impl DuckDbToolBackend {
    fn new(
        database: PathBuf,
        manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        let index = DuckDbIndex::open(&database, manifest.clone(), embedder.clone())?;
        let capabilities = index.capabilities();
        drop(index);
        Ok(Self {
            database,
            manifest,
            embedder,
            capabilities,
        })
    }

    fn open(&self) -> Result<DuckDbIndex, SearchError> {
        DuckDbIndex::open(&self.database, self.manifest.clone(), self.embedder.clone())
    }
}

#[async_trait]
impl ToolBackend for DuckDbToolBackend {
    async fn search_with_documents(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<(Vec<Candidate>, Vec<SearchDocument>), SearchError> {
        let index = self.open()?;
        let candidates = index.search(query, options).await?;
        let ids = candidates
            .iter()
            .map(|candidate| candidate.doc_id.clone())
            .collect::<Vec<_>>();
        let documents = index.documents(&ids)?;
        Ok((candidates, documents))
    }

    async fn document_count(&self) -> Result<usize, SearchError> {
        self.open()?.document_count()
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }
}

#[async_trait]
impl DocumentCatalog for MemoryCatalog {
    async fn document_count(&self) -> Result<usize, SearchError> {
        Ok(self.documents.len())
    }

    async fn documents(&self, ids: &[DocumentId]) -> Result<Vec<SearchDocument>, SearchError> {
        Ok(ids
            .iter()
            .filter_map(|id| self.documents.get(id.as_str()).cloned())
            .collect())
    }
}

#[async_trait]
impl DocumentCatalog for DuckDbIndex {
    async fn document_count(&self) -> Result<usize, SearchError> {
        DuckDbIndex::document_count(self)
    }

    async fn documents(&self, ids: &[DocumentId]) -> Result<Vec<SearchDocument>, SearchError> {
        DuckDbIndex::documents(self, ids)
    }
}

#[async_trait]
impl DocumentCatalog for ElasticsearchIndex {
    async fn document_count(&self) -> Result<usize, SearchError> {
        ElasticsearchIndex::document_count(self).await
    }

    async fn documents(&self, ids: &[DocumentId]) -> Result<Vec<SearchDocument>, SearchError> {
        ElasticsearchIndex::documents(self, ids).await
    }
}

impl SearchServer {
    fn phase0(documents: Vec<SearchDocument>) -> Result<Self> {
        let mut by_id = BTreeMap::new();
        for document in documents {
            let id = document.doc_id.as_str().to_owned();
            if by_id.insert(id.clone(), document).is_some() {
                bail!("duplicate corpus document ID {id}");
            }
        }
        if by_id.is_empty() {
            bail!("corpus must contain at least one document");
        }
        let document_ids = by_id
            .values()
            .map(|document| document.doc_id.clone())
            .collect();
        let catalog = MemoryCatalog { documents: by_id };
        Ok(Self::new(
            "phase0-deterministic-random",
            Arc::new(FixedBackend {
                retriever: Arc::new(DeterministicPhase0Retriever::new(document_ids)),
                catalog: Arc::new(catalog),
            }),
        ))
    }

    fn new(backend: impl Into<Arc<str>>, search_backend: Arc<dyn ToolBackend>) -> Self {
        Self {
            backend: backend.into(),
            search_backend,
            tool_router: Self::tool_router(),
        }
    }

    fn duckdb(
        database: PathBuf,
        manifest: IndexManifest,
        embedder: Option<Arc<dyn Embedder>>,
    ) -> Result<Self, SearchError> {
        let backend = Arc::new(DuckDbToolBackend::new(database, manifest, embedder)?);
        let name = if backend.capabilities().dense {
            "duckdb-hybrid"
        } else {
            "duckdb-bm25"
        };
        Ok(Self::new(name, backend))
    }

    fn elasticsearch(index: Arc<ElasticsearchIndex>) -> Self {
        let name = if index.capabilities().dense {
            "elasticsearch-hybrid"
        } else {
            "elasticsearch-bm25"
        };
        let retriever: Arc<dyn Retriever> = index.clone();
        let catalog: Arc<dyn DocumentCatalog> = index;
        Self::new(name, Arc::new(FixedBackend { retriever, catalog }))
    }

    async fn search_inner(&self, request: SearchRequest) -> Result<SearchResponse, SearchError> {
        let query = Query::new(request.query)?;
        let top_k = bounded_nonzero("top_k", request.top_k.unwrap_or(DEFAULT_TOP_K), MAX_TOP_K)?;
        let candidate_limit = bounded_nonzero(
            "candidate_limit",
            request
                .candidate_limit
                .unwrap_or(DEFAULT_CANDIDATE_LIMIT.max(top_k.get())),
            MAX_CANDIDATE_LIMIT,
        )?;
        let options = SearchOpts {
            top_k,
            candidate_limit,
            enable_late_interaction: request.enable_late_interaction.unwrap_or(false),
        };
        options.validate()?;

        let (candidates, documents) = self
            .search_backend
            .search_with_documents(&query, &options)
            .await?;
        let documents = documents
            .into_iter()
            .map(|document| (document.doc_id.clone(), document))
            .collect::<BTreeMap<_, _>>();
        let mut results = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let document = documents.get(&candidate.doc_id).ok_or_else(|| {
                SearchError::Retriever(format!(
                    "backend returned unknown document ID {}",
                    candidate.doc_id
                ))
            })?;
            results.push(SearchHit {
                doc_id: candidate.doc_id.as_str().to_owned(),
                score: candidate.score,
                signals: SignalResponse {
                    lexical: candidate.signals.lexical,
                    dense: candidate.signals.dense,
                    late: candidate.signals.late,
                },
                path: document.path.as_str().to_owned(),
                language: document.language.0.clone(),
                text: document.text.clone(),
            });
        }

        Ok(SearchResponse {
            backend: self.backend.to_string(),
            query: query.text,
            result_count: results.len(),
            capabilities: self.search_backend.capabilities().into(),
            results,
        })
    }
}

#[tool_router(router = tool_router)]
impl SearchServer {
    #[tool(
        description = "Search the configured Hay Seeker index through the common Retriever contract. Results include structured path, language, text, scores, and per-stage signals."
    )]
    async fn search(
        &self,
        Parameters(request): Parameters<SearchRequest>,
    ) -> Result<Json<SearchResponse>, String> {
        self.search_inner(request)
            .await
            .map(Json)
            .map_err(|error| error.to_string())
    }

    #[tool(
        description = "Report the active search backend and the retrieval stages it implements."
    )]
    async fn capabilities(&self) -> Result<Json<CapabilityToolResponse>, String> {
        self.search_backend
            .document_count()
            .await
            .map(|document_count| {
                Json(CapabilityToolResponse {
                    backend: self.backend.to_string(),
                    document_count,
                    capabilities: self.search_backend.capabilities().into(),
                })
            })
            .map_err(|error| error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
struct CapabilityToolResponse {
    backend: String,
    document_count: usize,
    capabilities: CapabilityResponse,
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for SearchServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "hay-search".into(),
                title: Some("Hay Seeker Search".into()),
                version: env!("CARGO_PKG_VERSION").into(),
                description: Some(
                    "MCP stdio adapter for Hay Seeker's backend-neutral Retriever contract".into(),
                ),
                ..Implementation::default()
            },
            instructions: Some(
                "Use search to query the configured index and capabilities to inspect active retrieval stages. DuckDB and Elasticsearch provide durable BM25 retrieval; phase0 remains an explicit deterministic integration stub."
                    .into(),
            ),
            ..ServerInfo::default()
        }
    }
}

fn bounded_nonzero(name: &str, value: usize, maximum: usize) -> Result<NonZeroUsize, SearchError> {
    if value == 0 || value > maximum {
        return Err(SearchError::InvalidConfig(format!(
            "{name} must be between 1 and {maximum}"
        )));
    }
    NonZeroUsize::new(value)
        .ok_or_else(|| SearchError::InvalidConfig(format!("{name} must not be zero")))
}

fn load_corpus(path: &Path) -> Result<Vec<SearchDocument>> {
    let file = File::open(path).with_context(|| format!("open corpus {}", path.display()))?;
    load_corpus_reader(BufReader::new(file), path)
}

fn load_corpus_reader(reader: impl BufRead, path: &Path) -> Result<Vec<SearchDocument>> {
    reader
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            result => Some((index, result)),
        })
        .map(|(index, line)| {
            let line =
                line.with_context(|| format!("read corpus {} line {}", path.display(), index + 1))?;
            let raw: RawDocument = serde_json::from_str(&line)
                .with_context(|| format!("parse corpus {} line {}", path.display(), index + 1))?;
            Ok(SearchDocument {
                doc_id: DocumentId::new(raw.doc_id)
                    .with_context(|| format!("validate document ID on line {}", index + 1))?,
                path: NormalizedPath::new(raw.path)
                    .with_context(|| format!("validate path on line {}", index + 1))?,
                language: LanguageId::new(raw.language),
                text: raw.text,
                embedding: None,
            })
        })
        .collect()
}

async fn search_runtime(
    backend: Backend,
    embeddings: Embeddings,
    download_models: bool,
) -> Result<SearchRuntime> {
    let backend = match backend {
        Backend::Duckdb => StorageBackend::DuckDb,
        Backend::Elasticsearch => StorageBackend::Elasticsearch,
        Backend::Phase0 => bail!("--embeddings is not supported with --backend phase0"),
    };
    let provider = match embeddings {
        Embeddings::None => EmbeddingProvider::None,
        Embeddings::LocalOnnx => EmbeddingProvider::LocalOnnx,
        Embeddings::LocalStatic => EmbeddingProvider::LocalStatic,
        Embeddings::Gemini => EmbeddingProvider::Gemini,
        Embeddings::OpenAi => EmbeddingProvider::OpenAi,
        Embeddings::Voyage => EmbeddingProvider::Voyage,
        Embeddings::CloudflareWorkersAi => EmbeddingProvider::CloudflareWorkersAi,
    };
    let models = ensure_models(provider, download_models, Some(&report_to_stderr)).await?;
    SearchRuntime::from_env_with_models(backend, provider, &models)
}

/// Parses arguments, reporting whether the caller chose the provider.
///
/// The Phase 0 corpus backend cannot embed. Defaulting the provider must not
/// break `--backend phase0`, so an unset provider is silently lexical there
/// while an explicitly requested one is still rejected.
/// Resolves the index to serve and refuses to serve one that does not exist.
///
/// A server has no operator to ask and cannot write a question to standard
/// output, which carries the MCP protocol. Starting anyway would open an empty
/// `DuckDB` file and answer every tool call with zero results, so a missing
/// index fails at startup where the client will show the reason.
fn resolve_database(configured: Option<PathBuf>) -> Result<PathBuf> {
    let database = match configured {
        Some(database) => database,
        None => Workspace::from_current_dir()?.default_database(),
    };
    if !database.is_file() {
        bail!(
            "no index at {}; build one with `hay index` before starting hay-mcp",
            database.display()
        );
    }
    Ok(database)
}

fn parse_arguments() -> Result<(Arguments, bool)> {
    let matches = Arguments::command().get_matches();
    let chosen = matches!(
        matches.value_source("embeddings"),
        Some(ValueSource::CommandLine | ValueSource::EnvVariable)
    );
    let arguments = Arguments::from_arg_matches(&matches)?;
    Ok((arguments, chosen))
}

#[tokio::main]
async fn main() -> Result<()> {
    load_dotenv()?;
    let (arguments, chose_embeddings) = parse_arguments()?;
    let server = match arguments.backend {
        Backend::Phase0 => {
            if chose_embeddings && arguments.embeddings != Embeddings::None {
                bail!("--embeddings is not supported with --backend phase0");
            }
            SearchServer::phase0(load_corpus(&arguments.corpus)?)?
        }
        Backend::Duckdb => {
            let database = resolve_database(arguments.database)?;
            let SearchRuntime { manifest, embedder } = search_runtime(
                arguments.backend,
                arguments.embeddings,
                arguments.download_models,
            )
            .await?;
            SearchServer::duckdb(database, manifest, embedder)?
        }
        Backend::Elasticsearch => {
            let SearchRuntime { manifest, embedder } = search_runtime(
                arguments.backend,
                arguments.embeddings,
                arguments.download_models,
            )
            .await?;
            let mut config = ElasticsearchConfig::new(arguments.endpoint, arguments.index);
            if let Ok(api_key) = std::env::var("ELASTICSEARCH_API_KEY") {
                config = config.with_api_key(api_key);
            } else if let Ok(token) = std::env::var("ELASTICSEARCH_BEARER_TOKEN") {
                config = config.with_bearer_token(token);
            }
            SearchServer::elasticsearch(Arc::new(ElasticsearchIndex::new(
                config, manifest, embedder,
            )?))
        }
    };
    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;
    use tempfile::tempdir;

    fn server() -> SearchServer {
        let input = concat!(
            r#"{"doc_id":"a","path":"docs/a.md","language":"markdown","text":"manifest validation"}"#,
            "\n",
            r#"{"doc_id":"b","path":"src/b.rs","language":"rust","text":"retriever trait"}"#,
            "\n"
        );
        let documents = load_corpus_reader(Cursor::new(input), Path::new("test.jsonl")).unwrap();
        SearchServer::phase0(documents).unwrap()
    }

    #[tokio::test]
    async fn search_is_deterministic_and_enriches_results() {
        let server = server();
        let request = || SearchRequest {
            query: "manifest".into(),
            top_k: Some(2),
            candidate_limit: Some(2),
            enable_late_interaction: None,
        };
        let first = server.search_inner(request()).await.unwrap();
        let second = server.search_inner(request()).await.unwrap();

        assert_eq!(first, second);
        assert_eq!(first.result_count, 2);
        assert!(first.results.iter().all(|hit| !hit.text.is_empty()));
    }

    #[tokio::test]
    async fn duckdb_server_searches_the_persisted_retriever_contract() {
        let directory = tempdir().unwrap();
        let database = directory.path().join("search.duckdb");
        let manifest = IndexManifest::lexical_v1();
        let index = DuckDbIndex::open(&database, manifest.clone(), None).unwrap();
        index
            .replace_all(&[SearchDocument {
                doc_id: DocumentId::new("manifest").unwrap(),
                path: NormalizedPath::new("src/manifest.rs").unwrap(),
                language: LanguageId::new("rust"),
                text: "validate stored index manifest compatibility".into(),
                embedding: None,
            }])
            .await
            .unwrap();
        drop(index);
        let response = SearchServer::duckdb(database, manifest, None)
            .unwrap()
            .search_inner(SearchRequest {
                query: "manifest compatibility".into(),
                top_k: Some(1),
                candidate_limit: Some(5),
                enable_late_interaction: None,
            })
            .await
            .unwrap();

        assert_eq!(response.backend, "duckdb-bm25");
        assert_eq!(response.results[0].doc_id, "manifest");
        assert!(response.capabilities.lexical);
        assert!(!response.capabilities.dense);
    }

    #[tokio::test]
    async fn search_rejects_unbounded_output() {
        let error = server()
            .search_inner(SearchRequest {
                query: "manifest".into(),
                top_k: Some(MAX_TOP_K + 1),
                candidate_limit: None,
                enable_late_interaction: None,
            })
            .await
            .unwrap_err();

        assert!(error.to_string().contains("top_k must be between"));
    }

    #[test]
    fn duplicate_document_ids_are_rejected() {
        let input = concat!(
            r#"{"doc_id":"same","path":"a.md","language":"markdown","text":"one"}"#,
            "\n",
            r#"{"doc_id":"same","path":"b.md","language":"markdown","text":"two"}"#,
            "\n"
        );
        let documents = load_corpus_reader(Cursor::new(input), Path::new("test.jsonl")).unwrap();
        let error = SearchServer::phase0(documents).err().unwrap();
        assert!(error.to_string().contains("duplicate corpus document ID"));
    }

    #[test]
    fn mcp_tools_publish_input_and_output_schemas() {
        let server = server();
        let tools = server.tool_router.list_all();

        assert_eq!(tools.len(), 2);
        assert!(tools.iter().any(|tool| tool.name == "capabilities"));
        let search = tools.iter().find(|tool| tool.name == "search").unwrap();
        assert!(search.input_schema.contains_key("properties"));
        assert!(search.output_schema.is_some());
    }
}
