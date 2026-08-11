#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use cast_core::LanguageId;
use cast_index::{ContentHash, DocumentId, HashAlgorithm, NormalizedPath};
use clap::{Parser, ValueEnum};
use hay_duckdb::DuckDbIndex;
use hay_elasticsearch::{ElasticsearchConfig, ElasticsearchIndex};
use hay_runtime::{
    BackendParityRuntime, EmbeddingProvider, SearchRuntime, StorageBackend, validate_backend_parity,
};
use hay_search::{
    Candidate, Chunker, ChunkerV1, CorpusDocument, DeterministicPhase0Retriever, FdeParams,
    IndexManifest, ManifestCheckedRetriever, Quantization, Query, Retriever, SearchDocument,
    SearchError, SearchOpts,
};
use serde::Deserialize;

const BINARY_PROBE_BYTES: usize = 8 * 1024;
const MAX_CONFIG_BYTES: u64 = 50 * 1024;
const MAX_SOURCE_BYTES: u64 = 5 * 1024 * 1024;
const MAX_BACKEND_NDCG_DELTA: f64 = 0.02;

#[derive(Debug, Parser)]
#[command(
    name = "eval",
    about = "Deterministic hybrid-search evaluation harness"
)]
struct Arguments {
    #[arg(long, value_enum)]
    backend: Backend,

    #[arg(long, value_enum, default_value = "none")]
    embeddings: Embeddings,

    #[arg(long, value_enum, default_value = "seed")]
    suite: Suite,

    #[arg(long)]
    eval_dir: Option<PathBuf>,

    #[arg(long, default_value = "evals/corpus.jsonl")]
    corpus: PathBuf,

    #[arg(long, default_value = ".bench-repos")]
    repo_root: PathBuf,

    #[arg(long, default_value = "benchmarks/repos.json")]
    repo_manifest: PathBuf,

    #[arg(long, default_value = ".hay-seeker/eval.duckdb")]
    database: PathBuf,

    /// Query an already-built `DuckDB` index instead of replacing its contents.
    #[arg(long, default_value_t = false)]
    reuse_duckdb_index: bool,

    #[arg(long, default_value = "http://127.0.0.1:9200")]
    endpoint: String,

    #[arg(long, default_value = "hay-seeker-eval")]
    index: String,

    /// Maximum documents retained by the shared parity embedding pass.
    #[arg(long, default_value = "8")]
    embedding_batch_size: NonZeroUsize,

    /// Exact `ChunkHound` executable used by the competitor adapter.
    #[arg(long)]
    chunkhound_bin: Option<PathBuf>,

    /// Explicit isolated `ChunkHound` configuration file.
    #[arg(long)]
    chunkhound_config: Option<PathBuf>,

    /// Directory containing one `<repository>.db` `ChunkHound` index per corpus.
    #[arg(long)]
    chunkhound_db_root: Option<PathBuf>,

    /// Optional frozen repository ID for a partial competitor checkpoint.
    #[arg(long)]
    chunkhound_repository: Option<String>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    /// Historical deterministic local label.
    Local,
    /// Historical deterministic remote label.
    Elastic,
    /// Production embedded BM25 backend.
    Duckdb,
    /// Production remote BM25 backend.
    Elasticsearch,
    /// Run both production backends and enforce the frozen parity gate.
    Parity,
    /// Evaluate `ChunkHound` single-hop semantic search over prebuilt indexes.
    Chunkhound,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Suite {
    Seed,
    Repos,
}

impl Backend {
    const fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Elastic => "elastic",
            Self::Duckdb => "duckdb",
            Self::Elasticsearch => "elasticsearch",
            Self::Parity => "parity",
            Self::Chunkhound => "chunkhound",
        }
    }

    const fn mrl_dim(self) -> usize {
        match self {
            Self::Local => 256,
            Self::Elastic => 768,
            Self::Duckdb | Self::Elasticsearch | Self::Parity | Self::Chunkhound => 1,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawCorpusDocument {
    doc_id: String,
    path: String,
    language: String,
    text: String,
}

#[derive(Clone, Debug, Deserialize)]
struct EvalCase {
    query: String,
    graded_doc_ids: BTreeMap<String, u8>,
}

#[derive(Debug, Deserialize)]
struct RepositoryManifest {
    schema_version: u32,
    repositories: Vec<RepositorySpec>,
}

#[derive(Debug, Deserialize)]
struct RepositorySpec {
    id: String,
    #[serde(rename = "url")]
    _url: String,
    revision: String,
    directory: String,
}

#[derive(Clone, Copy, Debug, Default)]
struct Metrics {
    ndcg_at_10: f64,
    recall_at_50: f64,
    mrr: f64,
    queries: usize,
}

#[derive(Debug, Default)]
struct LoadedDocuments {
    documents: Vec<SearchDocument>,
    candidate_to_judgment: BTreeMap<DocumentId, DocumentId>,
    source_documents: usize,
    skips: BTreeMap<&'static str, usize>,
}

impl LoadedDocuments {
    fn skip(&mut self, reason: &'static str) {
        *self.skips.entry(reason).or_default() += 1;
    }
}

#[tokio::main]
#[allow(clippy::too_many_lines)]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let arguments = Arguments::parse();
    if arguments.reuse_duckdb_index && !matches!(arguments.backend, Backend::Duckdb) {
        bail!("--reuse-duckdb-index requires --backend duckdb");
    }
    if matches!(arguments.backend, Backend::Chunkhound) {
        return run_chunkhound_evaluation(&arguments);
    }
    let (mut loaded, cases) = match arguments.suite {
        Suite::Seed => {
            let documents = load_corpus(&arguments.corpus)?;
            let eval_dir = arguments.eval_dir.as_deref().unwrap_or(Path::new("evals"));
            let cases = load_eval_cases(eval_dir, Some(&arguments.corpus))?;
            (
                LoadedDocuments {
                    candidate_to_judgment: documents
                        .iter()
                        .map(|document| (document.doc_id.clone(), document.doc_id.clone()))
                        .collect(),
                    source_documents: documents.len(),
                    documents,
                    skips: BTreeMap::new(),
                },
                cases,
            )
        }
        Suite::Repos => {
            let eval_dir = arguments
                .eval_dir
                .as_deref()
                .unwrap_or(Path::new("evals/repos"));
            let cases = load_eval_cases(eval_dir, None)?;
            let documents =
                load_repository_documents(&arguments.repo_root, &arguments.repo_manifest)?;
            (documents, cases)
        }
    };
    if cases.len() < 30 {
        bail!(
            "Phase 0 requires at least 30 evaluated queries; found {}",
            cases.len()
        );
    }

    let judgment_ids = loaded
        .candidate_to_judgment
        .values()
        .cloned()
        .collect::<Vec<_>>();
    validate_eval_set(&judgment_ids, &cases)?;

    let (retriever_label, metrics, repository_metrics, latencies) = match arguments.backend {
        Backend::Local | Backend::Elastic => {
            if arguments.embeddings != Embeddings::None {
                bail!("--embeddings is supported only by production backends");
            }
            let index_manifest = phase0_manifest(arguments.backend)?;
            let runtime_manifest = phase0_manifest(arguments.backend)?;
            let backend = DeterministicPhase0Retriever::new(
                loaded
                    .documents
                    .iter()
                    .map(|document| document.doc_id.clone())
                    .collect(),
            );
            let retriever =
                ManifestCheckedRetriever::new(backend, index_manifest, runtime_manifest);
            let (metrics, repository_metrics, latencies) = evaluate_suite(
                &retriever,
                &cases,
                arguments.suite,
                &loaded.candidate_to_judgment,
            )
            .await?;
            (
                "deterministic-random-stub",
                metrics,
                repository_metrics,
                latencies,
            )
        }
        Backend::Duckdb => {
            if arguments.reuse_duckdb_index && !arguments.database.is_file() {
                bail!(
                    "reused DuckDB index does not exist: {}",
                    arguments.database.display()
                );
            }
            if let Some(parent) = arguments.database.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create index directory {}", parent.display()))?;
            }
            let SearchRuntime { manifest, embedder } =
                search_runtime(StorageBackend::DuckDb, arguments.embeddings)?;
            let backend = DuckDbIndex::open(&arguments.database, manifest, embedder)?;
            if !arguments.reuse_duckdb_index {
                backend.replace_all(&loaded.documents).await?;
            }
            let (metrics, repository_metrics, latencies) = evaluate_suite(
                &backend,
                &cases,
                arguments.suite,
                &loaded.candidate_to_judgment,
            )
            .await?;
            (
                if arguments.embeddings == Embeddings::None {
                    "duckdb-bm25"
                } else {
                    "duckdb-hybrid"
                },
                metrics,
                repository_metrics,
                latencies,
            )
        }
        Backend::Elasticsearch => {
            let SearchRuntime { manifest, embedder } =
                search_runtime(StorageBackend::Elasticsearch, arguments.embeddings)?;
            let backend =
                ElasticsearchIndex::new(elasticsearch_config(&arguments), manifest, embedder)?;
            backend.replace_all(&loaded.documents).await?;
            let (metrics, repository_metrics, latencies) = evaluate_suite(
                &backend,
                &cases,
                arguments.suite,
                &loaded.candidate_to_judgment,
            )
            .await?;
            (
                if arguments.embeddings == Embeddings::None {
                    "elasticsearch-bm25"
                } else {
                    "elasticsearch-hybrid"
                },
                metrics,
                repository_metrics,
                latencies,
            )
        }
        Backend::Parity => {
            return run_backend_parity(&arguments, &mut loaded, &cases).await;
        }
        Backend::Chunkhound => bail!("ChunkHound evaluation dispatch invariant failed"),
    };

    println!("backend: {}", arguments.backend.label());
    println!("retriever: {retriever_label}");
    println!("documents: {}", loaded.documents.len());
    println!("source_documents: {}", loaded.source_documents);
    for (reason, count) in &loaded.skips {
        println!("skipped[{reason}]: {count}");
    }
    println!("queries: {}", metrics.queries);
    println!("nDCG@10: {:.6}", metrics.ndcg_at_10);
    println!("recall@50: {:.6}", metrics.recall_at_50);
    println!("MRR: {:.6}", metrics.mrr);
    print_latency_block("warm_in_process", &latencies)?;
    for (repository, metrics) in repository_metrics {
        println!(
            "repository[{repository}]: queries={} nDCG@10={:.6} recall@50={:.6} MRR={:.6}",
            metrics.queries, metrics.ndcg_at_10, metrics.recall_at_50, metrics.mrr
        );
    }
    Ok(())
}

fn run_chunkhound_evaluation(arguments: &Arguments) -> Result<()> {
    if arguments.suite != Suite::Repos {
        bail!("ChunkHound comparison requires --suite repos");
    }
    if arguments.embeddings != Embeddings::None {
        bail!("ChunkHound owns its embedding configuration; use --embeddings none");
    }
    let binary = arguments
        .chunkhound_bin
        .as_deref()
        .context("--chunkhound-bin is required for --backend chunkhound")?;
    let config = arguments
        .chunkhound_config
        .as_deref()
        .context("--chunkhound-config is required for --backend chunkhound")?;
    let database_root = arguments
        .chunkhound_db_root
        .as_deref()
        .context("--chunkhound-db-root is required for --backend chunkhound")?;
    let version = chunkhound_version(binary)?;
    if version != "chunkhound 5.2.1" {
        bail!("ChunkHound comparison requires version 5.2.1; found {version}");
    }
    let eval_dir = arguments
        .eval_dir
        .as_deref()
        .unwrap_or(Path::new("evals/repos"));
    let mut cases = load_eval_cases(eval_dir, None)?;
    if let Some(repository) = arguments.chunkhound_repository.as_deref() {
        cases.retain(|case| repository_for_case(case).is_ok_and(|value| value == repository));
        if cases.is_empty() {
            bail!("no frozen queries found for ChunkHound repository {repository}");
        }
    } else if cases.len() < 30 {
        bail!(
            "Phase 0 requires at least 30 evaluated queries; found {}",
            cases.len()
        );
    }
    let repositories = load_repository_specs(&arguments.repo_manifest)?;
    validate_chunkhound_eval_set(&arguments.repo_root, &repositories, &cases)?;

    let mut metrics = Metrics::default();
    let mut repository_metrics = BTreeMap::<String, Metrics>::new();
    let mut latencies = Vec::with_capacity(cases.len());
    for (ordinal, case) in cases.iter().enumerate() {
        let repository = repository_for_case(case)?;
        let spec = repositories.get(repository).ok_or_else(|| {
            anyhow::anyhow!("query references repository absent from manifest: {repository}")
        })?;
        let checkout = arguments.repo_root.join(&spec.directory);
        let database = database_root.join(format!("{repository}.db"));
        let started = Instant::now();
        let candidates = chunkhound_candidates(
            binary,
            config,
            &database,
            &checkout,
            repository,
            &case.query,
            50,
        )?;
        let elapsed = started.elapsed();
        latencies.push(elapsed);
        record_case_metrics(&mut metrics, &candidates, &case.graded_doc_ids);
        record_case_metrics(
            repository_metrics.entry(repository.into()).or_default(),
            &candidates,
            &case.graded_doc_ids,
        );
        eprintln!(
            "{{\"event\":\"chunkhound_query\",\"completed\":{},\"total\":{},\"repository\":\"{}\",\"elapsed_ms\":{:.3},\"unique_files\":{}}}",
            ordinal + 1,
            cases.len(),
            repository,
            elapsed.as_secs_f64() * 1_000.0,
            candidates.len()
        );
    }
    finalize_metrics(&mut metrics);
    for repository in repository_metrics.values_mut() {
        finalize_metrics(repository);
    }
    latencies.sort_unstable();
    print_chunkhound_report(metrics, &repository_metrics, &latencies, &version)
}

fn print_chunkhound_report(
    metrics: Metrics,
    repository_metrics: &BTreeMap<String, Metrics>,
    latencies: &[Duration],
    version: &str,
) -> Result<()> {
    println!("backend: chunkhound");
    println!("retriever: chunkhound-single-hop-semantic");
    println!("version: {version}");
    println!("queries: {}", metrics.queries);
    println!("nDCG@10: {:.6}", metrics.ndcg_at_10);
    println!("recall@50: {:.6}", metrics.recall_at_50);
    println!("MRR: {:.6}", metrics.mrr);
    println!(
        "cold_process_latency_p50_ms: {:.3}",
        percentile_duration(latencies, 50)?.as_secs_f64() * 1_000.0
    );
    println!(
        "cold_process_latency_p95_ms: {:.3}",
        percentile_duration(latencies, 95)?.as_secs_f64() * 1_000.0
    );
    for (repository, repository_metrics) in repository_metrics {
        println!(
            "repository[{repository}]: queries={} nDCG@10={:.6} recall@50={:.6} MRR={:.6}",
            repository_metrics.queries,
            repository_metrics.ndcg_at_10,
            repository_metrics.recall_at_50,
            repository_metrics.mrr
        );
    }
    Ok(())
}

fn chunkhound_version(binary: &Path) -> Result<String> {
    let output = Command::new(binary)
        .arg("--version")
        .output()
        .context("inspect ChunkHound version")?;
    if !output.status.success() {
        bail!(
            "ChunkHound version inspection failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8(output.stdout)
        .context("ChunkHound version returned non-UTF-8 stdout")?
        .trim()
        .into())
}

fn chunkhound_candidates(
    binary: &Path,
    config: &Path,
    database: &Path,
    checkout: &Path,
    repository: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<Candidate>> {
    const PAGE_SIZE: usize = 200;
    const MAX_PAGES: usize = 5;

    let mut seen = BTreeSet::new();
    let mut candidates = Vec::with_capacity(limit);
    for page in 0..MAX_PAGES {
        let offset = page * PAGE_SIZE;
        let output = Command::new(binary)
            .current_dir(checkout)
            .arg("search")
            .arg(query)
            .arg(".")
            .args(["--single-hop", "--page-size"])
            .arg(PAGE_SIZE.to_string())
            .arg("--offset")
            .arg(offset.to_string())
            .arg("--config")
            .arg(config)
            .arg("--db")
            .arg(database)
            .output()
            .with_context(|| format!("run ChunkHound query for {repository}"))?;
        if !output.status.success() {
            bail!(
                "ChunkHound query failed for {repository}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        let stdout = String::from_utf8(output.stdout)
            .context("ChunkHound search returned non-UTF-8 stdout")?;
        let mut page_results = 0_usize;
        for line in stdout.lines() {
            let Some((rank, path)) = parse_chunkhound_result_header(line) else {
                continue;
            };
            page_results = page_results.saturating_add(1);
            let normalized = NormalizedPath::new(path.replace('\\', "/"))
                .with_context(|| format!("validate ChunkHound result path {path}"))?;
            let parent = DocumentId::new(format!("{repository}:{normalized}"))?;
            if seen.insert(parent.clone()) {
                let bounded_rank = u16::try_from(rank).context("ChunkHound rank exceeds u16")?;
                let score = 1.0 / f32::from(bounded_rank);
                candidates.push(Candidate {
                    doc_id: parent,
                    score,
                    signals: hay_search::Signals {
                        lexical: None,
                        dense: Some(score),
                        late: None,
                    },
                });
                if candidates.len() == limit {
                    return Ok(candidates);
                }
            }
        }
        if page_results < PAGE_SIZE {
            break;
        }
    }
    Ok(candidates)
}

fn parse_chunkhound_result_header(line: &str) -> Option<(usize, &str)> {
    let rest = line.strip_prefix('[')?;
    let (rank, path) = rest.split_once("] ")?;
    let rank = rank.parse().ok()?;
    (rank > 0).then(|| (rank, path.trim()))
}

fn record_case_metrics(
    metrics: &mut Metrics,
    candidates: &[Candidate],
    judgments: &BTreeMap<String, u8>,
) {
    metrics.ndcg_at_10 += ndcg_at(candidates, judgments, 10);
    metrics.recall_at_50 += recall_at(candidates, judgments, 50);
    metrics.mrr += reciprocal_rank(candidates, judgments);
    metrics.queries = metrics.queries.saturating_add(1);
}

fn finalize_metrics(metrics: &mut Metrics) {
    if metrics.queries == 0 {
        return;
    }
    let denominator = metric_count(metrics.queries);
    metrics.ndcg_at_10 /= denominator;
    metrics.recall_at_50 /= denominator;
    metrics.mrr /= denominator;
}

fn percentile_duration(samples: &[Duration], percentile: usize) -> Result<Duration> {
    if samples.is_empty() || !(1..=100).contains(&percentile) {
        bail!("latency percentile requires samples and a percentile from 1 through 100");
    }
    let scaled = samples
        .len()
        .checked_mul(percentile)
        .context("latency percentile index overflow")?;
    let index = scaled.div_ceil(100).saturating_sub(1);
    Ok(samples[index])
}

fn repository_for_case(case: &EvalCase) -> Result<&str> {
    let repositories = case
        .graded_doc_ids
        .keys()
        .filter_map(|doc_id| doc_id.split_once(':').map(|(repository, _)| repository))
        .collect::<BTreeSet<_>>();
    if repositories.len() != 1 {
        bail!(
            "repository query must grade documents from exactly one repository: {}",
            case.query
        );
    }
    repositories
        .first()
        .copied()
        .context("repository query has no repository-prefixed judgment")
}

fn load_repository_specs(path: &Path) -> Result<BTreeMap<String, RepositorySpec>> {
    let bytes =
        fs::read(path).with_context(|| format!("read repository manifest {}", path.display()))?;
    let manifest: RepositoryManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse repository manifest {}", path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported repository manifest schema {}; expected 1",
            manifest.schema_version
        );
    }
    let repositories = manifest
        .repositories
        .into_iter()
        .map(|spec| (spec.id.clone(), spec))
        .collect::<BTreeMap<_, _>>();
    if repositories.len() != 4 {
        bail!("ChunkHound comparison requires the four frozen repositories");
    }
    Ok(repositories)
}

fn validate_chunkhound_eval_set(
    root: &Path,
    repositories: &BTreeMap<String, RepositorySpec>,
    cases: &[EvalCase],
) -> Result<()> {
    let mut queries = BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        Query::new(&case.query).with_context(|| format!("validate query {}", index + 1))?;
        if !queries.insert(case.query.as_str()) {
            bail!("duplicate evaluation query: {}", case.query);
        }
        let repository = repository_for_case(case)?;
        let spec = repositories.get(repository).ok_or_else(|| {
            anyhow::anyhow!("query references repository absent from manifest: {repository}")
        })?;
        let checkout = root.join(&spec.directory);
        verify_revision(&checkout, &spec.revision)?;
        for (document_id, grade) in &case.graded_doc_ids {
            if *grade == 0 {
                continue;
            }
            let (_, path) = document_id
                .split_once(':')
                .context("repository judgment must contain a colon")?;
            let normalized = NormalizedPath::new(path)?;
            if !checkout.join(normalized.as_str()).is_file() {
                bail!("query {} grades missing file {document_id}", index + 1);
            }
        }
    }
    Ok(())
}

async fn run_backend_parity(
    arguments: &Arguments,
    loaded: &mut LoadedDocuments,
    cases: &[EvalCase],
) -> Result<()> {
    if let Some(parent) = arguments.database.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create index directory {}", parent.display()))?;
    }
    let parity_runtime = BackendParityRuntime::from_env(embedding_provider(arguments.embeddings))?;
    validate_backend_parity(
        &parity_runtime.duckdb.manifest,
        &parity_runtime.elasticsearch.manifest,
    )?;
    let mut shared_embeddings = if arguments.embeddings == Embeddings::None {
        None
    } else {
        Some(
            precompute_parity_embeddings(
                &parity_runtime,
                &mut loaded.documents,
                arguments.embedding_batch_size,
            )
            .await?,
        )
    };
    let BackendParityRuntime {
        duckdb:
            SearchRuntime {
                manifest: duckdb_manifest,
                embedder: duckdb_embedder,
            },
        elasticsearch:
            SearchRuntime {
                manifest: elasticsearch_manifest,
                embedder: elasticsearch_embedder,
            },
    } = parity_runtime;
    let duckdb = DuckDbIndex::open(
        &arguments.database,
        duckdb_manifest.clone(),
        duckdb_embedder,
    )?;
    duckdb.replace_all(&loaded.documents).await?;
    let (duckdb_metrics, duckdb_repositories, duckdb_latencies) = evaluate_suite(
        &duckdb,
        cases,
        arguments.suite,
        &loaded.candidate_to_judgment,
    )
    .await?;
    drop(duckdb);

    let embedding_elapsed = shared_embeddings.as_ref().map(|shared| shared.elapsed);
    if let Some(shared_embeddings) = shared_embeddings.take() {
        install_elasticsearch_embeddings(&mut loaded.documents, shared_embeddings.elasticsearch)?;
    }
    let elasticsearch = ElasticsearchIndex::new(
        elasticsearch_config(arguments),
        elasticsearch_manifest,
        elasticsearch_embedder,
    )?;
    elasticsearch.replace_all(&loaded.documents).await?;
    let (elasticsearch_metrics, elasticsearch_repositories, elasticsearch_latencies) =
        evaluate_suite(
            &elasticsearch,
            cases,
            arguments.suite,
            &loaded.candidate_to_judgment,
        )
        .await?;

    print_parity_report(&ParityReport {
        document_count: loaded.documents.len(),
        source_document_count: loaded.source_documents,
        skips: &loaded.skips,
        embedding_elapsed,
        duckdb_metrics,
        duckdb_repositories: &duckdb_repositories,
        duckdb_latencies: &duckdb_latencies,
        elasticsearch_metrics,
        elasticsearch_repositories: &elasticsearch_repositories,
        elasticsearch_latencies: &elasticsearch_latencies,
    })
}

struct ParityReport<'a> {
    document_count: usize,
    source_document_count: usize,
    skips: &'a BTreeMap<&'static str, usize>,
    embedding_elapsed: Option<Duration>,
    duckdb_metrics: Metrics,
    duckdb_repositories: &'a BTreeMap<String, Metrics>,
    duckdb_latencies: &'a [Duration],
    elasticsearch_metrics: Metrics,
    elasticsearch_repositories: &'a BTreeMap<String, Metrics>,
    elasticsearch_latencies: &'a [Duration],
}

fn print_parity_report(report: &ParityReport<'_>) -> Result<()> {
    println!("backend: parity");
    println!("documents: {}", report.document_count);
    println!("source_documents: {}", report.source_document_count);
    if let Some(embedding_elapsed) = report.embedding_elapsed {
        let document_count = u32::try_from(report.document_count)
            .context("document count exceeds parity telemetry range")?;
        println!("parity.document_embedding_passes: 1");
        println!(
            "parity.document_embedding_seconds: {:.3}",
            embedding_elapsed.as_secs_f64()
        );
        println!(
            "parity.document_embeddings_per_second: {:.3}",
            f64::from(document_count) / embedding_elapsed.as_secs_f64()
        );
    }
    for (reason, count) in report.skips {
        println!("skipped[{reason}]: {count}");
    }
    print_metric_block("duckdb", report.duckdb_metrics, report.duckdb_repositories);
    print_latency_block("duckdb.warm_in_process", report.duckdb_latencies)?;
    print_metric_block(
        "elasticsearch",
        report.elasticsearch_metrics,
        report.elasticsearch_repositories,
    );
    print_latency_block(
        "elasticsearch.warm_in_process",
        report.elasticsearch_latencies,
    )?;
    let ndcg_delta = metric_delta(
        report.duckdb_metrics.ndcg_at_10,
        report.elasticsearch_metrics.ndcg_at_10,
    );
    println!("parity.nDCG@10_delta: {ndcg_delta:.6}");
    println!(
        "parity.recall@50_delta: {:.6}",
        metric_delta(
            report.duckdb_metrics.recall_at_50,
            report.elasticsearch_metrics.recall_at_50
        )
    );
    println!(
        "parity.MRR_delta: {:.6}",
        metric_delta(report.duckdb_metrics.mrr, report.elasticsearch_metrics.mrr)
    );
    for (repository, duckdb_metrics) in report.duckdb_repositories {
        let elasticsearch_metrics = report
            .elasticsearch_repositories
            .get(repository)
            .ok_or_else(|| {
                anyhow::anyhow!("Elasticsearch result is missing repository group {repository}")
            })?;
        println!(
            "parity.repository[{repository}].nDCG@10_delta: {:.6}",
            metric_delta(duckdb_metrics.ndcg_at_10, elasticsearch_metrics.ndcg_at_10)
        );
    }
    enforce_ndcg_parity(report.duckdb_metrics, report.elasticsearch_metrics)?;
    println!("parity.gate: pass");
    Ok(())
}

struct SharedParityEmbeddings {
    elasticsearch: Vec<Vec<f32>>,
    elapsed: Duration,
}

async fn precompute_parity_embeddings(
    runtime: &BackendParityRuntime,
    documents: &mut [SearchDocument],
    batch_size: NonZeroUsize,
) -> Result<SharedParityEmbeddings> {
    if documents
        .iter()
        .any(|document| document.embedding.is_some())
    {
        bail!("parity document precomputation requires source documents without vectors");
    }
    let started = Instant::now();
    let mut last_progress = started;
    let mut elasticsearch = Vec::with_capacity(documents.len());
    for start in (0..documents.len()).step_by(batch_size.get()) {
        let end = documents.len().min(start + batch_size.get());
        let inputs = documents[start..end]
            .iter()
            .map(|document| cast_index::EmbeddingInput {
                document_id: &document.doc_id,
                text: &document.text,
            })
            .collect::<Vec<_>>();
        let batch = runtime.embed_batch(&inputs).await?;
        drop(inputs);
        if batch.duckdb.len() != end - start || batch.elasticsearch.len() != end - start {
            bail!("shared parity embedder returned the wrong vector count");
        }
        for ((document, duckdb), elasticsearch_vector) in documents[start..end]
            .iter_mut()
            .zip(batch.duckdb)
            .zip(batch.elasticsearch)
        {
            document.embedding = Some(duckdb.values);
            document.validate(&runtime.duckdb.manifest)?;
            elasticsearch.push(elasticsearch_vector.values);
        }

        let now = Instant::now();
        if now.duration_since(last_progress) >= Duration::from_secs(5) || end == documents.len() {
            eprintln!(
                "{{\"event\":\"parity_embedding_progress\",\"completed\":{end},\"total\":{},\"elapsed_seconds\":{:.3}}}",
                documents.len(),
                now.duration_since(started).as_secs_f64()
            );
            last_progress = now;
        }
    }
    Ok(SharedParityEmbeddings {
        elasticsearch,
        elapsed: started.elapsed(),
    })
}

fn install_elasticsearch_embeddings(
    documents: &mut [SearchDocument],
    embeddings: Vec<Vec<f32>>,
) -> Result<()> {
    if documents.len() != embeddings.len() {
        bail!(
            "shared parity embedding count {} does not match document count {}",
            embeddings.len(),
            documents.len()
        );
    }
    for (document, embedding) in documents.iter_mut().zip(embeddings) {
        document.embedding = Some(embedding);
    }
    Ok(())
}

fn elasticsearch_config(arguments: &Arguments) -> ElasticsearchConfig {
    let mut config = ElasticsearchConfig::new(&arguments.endpoint, &arguments.index);
    if let Ok(api_key) = std::env::var("ELASTICSEARCH_API_KEY") {
        config = config.with_api_key(api_key);
    } else if let Ok(token) = std::env::var("ELASTICSEARCH_BEARER_TOKEN") {
        config = config.with_bearer_token(token);
    }
    config
}

fn print_metric_block(
    backend: &str,
    metrics: Metrics,
    repository_metrics: &BTreeMap<String, Metrics>,
) {
    println!("{backend}.queries: {}", metrics.queries);
    println!("{backend}.nDCG@10: {:.6}", metrics.ndcg_at_10);
    println!("{backend}.recall@50: {:.6}", metrics.recall_at_50);
    println!("{backend}.MRR: {:.6}", metrics.mrr);
    for (repository, metrics) in repository_metrics {
        println!(
            "{backend}.repository[{repository}]: queries={} nDCG@10={:.6} recall@50={:.6} MRR={:.6}",
            metrics.queries, metrics.ndcg_at_10, metrics.recall_at_50, metrics.mrr
        );
    }
}

fn print_latency_block(label: &str, sorted_latencies: &[Duration]) -> Result<()> {
    println!(
        "{label}_latency_p50_ms: {:.3}",
        percentile_duration(sorted_latencies, 50)?.as_secs_f64() * 1_000.0
    );
    println!(
        "{label}_latency_p95_ms: {:.3}",
        percentile_duration(sorted_latencies, 95)?.as_secs_f64() * 1_000.0
    );
    Ok(())
}

fn metric_delta(left: f64, right: f64) -> f64 {
    (left - right).abs()
}

fn enforce_ndcg_parity(duckdb: Metrics, elasticsearch: Metrics) -> Result<()> {
    let delta = metric_delta(duckdb.ndcg_at_10, elasticsearch.ndcg_at_10);
    if !delta.is_finite() {
        bail!("backend nDCG@10 parity delta is not finite");
    }
    if delta > MAX_BACKEND_NDCG_DELTA + f64::EPSILON {
        bail!("backend nDCG@10 delta {delta:.6} exceeds parity gate {MAX_BACKEND_NDCG_DELTA:.6}");
    }
    Ok(())
}

fn search_runtime(backend: StorageBackend, embeddings: Embeddings) -> Result<SearchRuntime> {
    SearchRuntime::from_env(backend, embedding_provider(embeddings))
}

const fn embedding_provider(embeddings: Embeddings) -> EmbeddingProvider {
    match embeddings {
        Embeddings::None => EmbeddingProvider::None,
        Embeddings::LocalOnnx => EmbeddingProvider::LocalOnnx,
        Embeddings::LocalStatic => EmbeddingProvider::LocalStatic,
        Embeddings::Gemini => EmbeddingProvider::Gemini,
        Embeddings::OpenAi => EmbeddingProvider::OpenAi,
        Embeddings::Voyage => EmbeddingProvider::Voyage,
        Embeddings::CloudflareWorkersAi => EmbeddingProvider::CloudflareWorkersAi,
    }
}

async fn evaluate_suite(
    retriever: &dyn Retriever,
    cases: &[EvalCase],
    suite: Suite,
    candidate_to_judgment: &BTreeMap<DocumentId, DocumentId>,
) -> Result<(Metrics, BTreeMap<String, Metrics>, Vec<Duration>)> {
    let options = SearchOpts {
        top_k: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
        candidate_limit: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
        enable_late_interaction: false,
    };
    let mut metrics = Metrics::default();
    let mut repository_metrics = BTreeMap::<String, Metrics>::new();
    let mut latencies = Vec::with_capacity(cases.len());

    for case in cases {
        let query = Query::new(&case.query)?;
        let started = Instant::now();
        let candidates = retriever.search(&query, &options).await?;
        latencies.push(started.elapsed());
        let candidates = collapse_candidates(candidates, candidate_to_judgment)?;
        record_case_metrics(&mut metrics, &candidates, &case.graded_doc_ids);
        if suite == Suite::Repos {
            let repository = repository_for_case(case)?;
            record_case_metrics(
                repository_metrics.entry(repository.into()).or_default(),
                &candidates,
                &case.graded_doc_ids,
            );
        }
    }

    finalize_metrics(&mut metrics);
    for repository in repository_metrics.values_mut() {
        finalize_metrics(repository);
    }
    latencies.sort_unstable();
    Ok((metrics, repository_metrics, latencies))
}

fn collapse_candidates(
    candidates: Vec<Candidate>,
    candidate_to_judgment: &BTreeMap<DocumentId, DocumentId>,
) -> Result<Vec<Candidate>, SearchError> {
    let mut seen = BTreeSet::new();
    let mut collapsed = Vec::with_capacity(candidates.len());
    for mut candidate in candidates {
        let judgment_id = candidate_to_judgment
            .get(&candidate.doc_id)
            .ok_or_else(|| {
                SearchError::Retriever(format!(
                    "retriever returned unknown candidate {}",
                    candidate.doc_id
                ))
            })?;
        if seen.insert(judgment_id.clone()) {
            candidate.doc_id = judgment_id.clone();
            collapsed.push(candidate);
        }
    }
    Ok(collapsed)
}

fn ndcg_at(candidates: &[Candidate], judgments: &BTreeMap<String, u8>, cutoff: usize) -> f64 {
    let dcg = candidates
        .iter()
        .take(cutoff)
        .enumerate()
        .map(|(rank, candidate)| {
            discounted_gain(
                judgments
                    .get(candidate.doc_id.as_str())
                    .copied()
                    .unwrap_or_default(),
                rank,
            )
        })
        .sum::<f64>();
    let mut ideal = judgments.values().copied().collect::<Vec<_>>();
    ideal.sort_unstable_by(|left, right| right.cmp(left));
    let idcg = ideal
        .into_iter()
        .take(cutoff)
        .enumerate()
        .map(|(rank, relevance)| discounted_gain(relevance, rank))
        .sum::<f64>();
    if idcg == 0.0 { 0.0 } else { dcg / idcg }
}

fn discounted_gain(relevance: u8, zero_based_rank: usize) -> f64 {
    (2_f64.powi(i32::from(relevance)) - 1.0) / (metric_count(zero_based_rank) + 2.0).log2()
}

fn recall_at(candidates: &[Candidate], judgments: &BTreeMap<String, u8>, cutoff: usize) -> f64 {
    let relevant = judgments
        .iter()
        .filter_map(|(doc_id, grade)| (*grade > 0).then_some(doc_id.as_str()))
        .collect::<BTreeSet<_>>();
    if relevant.is_empty() {
        return 0.0;
    }
    let retrieved = candidates
        .iter()
        .take(cutoff)
        .filter(|candidate| relevant.contains(candidate.doc_id.as_str()))
        .count();
    metric_count(retrieved) / metric_count(relevant.len())
}

fn reciprocal_rank(candidates: &[Candidate], judgments: &BTreeMap<String, u8>) -> f64 {
    candidates
        .iter()
        .position(|candidate| {
            judgments
                .get(candidate.doc_id.as_str())
                .is_some_and(|grade| *grade > 0)
        })
        .map_or(0.0, |rank| 1.0 / (metric_count(rank) + 1.0))
}

fn metric_count(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

fn load_corpus(path: &Path) -> Result<Vec<SearchDocument>> {
    let raw = read_jsonl::<RawCorpusDocument>(path)?;
    let documents = raw
        .into_iter()
        .map(|document| {
            Ok(SearchDocument {
                doc_id: DocumentId::new(document.doc_id)?,
                path: NormalizedPath::new(document.path)?,
                language: LanguageId::new(document.language),
                text: document.text,
                embedding: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(documents)
}

fn load_eval_cases(eval_dir: &Path, corpus_path: Option<&Path>) -> Result<Vec<EvalCase>> {
    let corpus_path = corpus_path
        .map(|path| {
            fs::canonicalize(path)
                .with_context(|| format!("resolve corpus path {}", path.display()))
        })
        .transpose()?;
    let mut files = fs::read_dir(eval_dir)
        .with_context(|| format!("read eval directory {}", eval_dir.display()))?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect::<Vec<_>>();
    files.sort();

    let mut cases = Vec::new();
    for file in files {
        if fs::canonicalize(&file).ok().as_ref() == corpus_path.as_ref() {
            continue;
        }
        cases.extend(read_jsonl::<EvalCase>(&file)?);
    }
    Ok(cases)
}

fn validate_eval_set(document_ids: &[DocumentId], cases: &[EvalCase]) -> Result<()> {
    let document_ids = document_ids
        .iter()
        .map(DocumentId::as_str)
        .collect::<BTreeSet<_>>();
    let mut queries = BTreeSet::new();

    for (index, case) in cases.iter().enumerate() {
        Query::new(&case.query).with_context(|| format!("validate query {}", index + 1))?;
        if !queries.insert(case.query.as_str()) {
            bail!(
                "duplicate evaluation query at case {}: {}",
                index + 1,
                case.query
            );
        }
        if case.graded_doc_ids.is_empty() || case.graded_doc_ids.values().all(|grade| *grade == 0) {
            bail!("query {} has no positively graded documents", index + 1);
        }
        for doc_id in case.graded_doc_ids.keys() {
            if !document_ids.contains(doc_id.as_str()) {
                bail!("query {} grades unknown document ID {doc_id}", index + 1);
            }
        }
    }
    Ok(())
}

fn load_repository_documents(root: &Path, manifest_path: &Path) -> Result<LoadedDocuments> {
    let manifest = fs::read_to_string(manifest_path)
        .with_context(|| format!("read repository manifest {}", manifest_path.display()))?;
    let manifest: RepositoryManifest = serde_json::from_str(&manifest)
        .with_context(|| format!("parse repository manifest {}", manifest_path.display()))?;
    if manifest.schema_version != 1 {
        bail!(
            "unsupported repository manifest schema {}; expected 1",
            manifest.schema_version
        );
    }

    let mut loaded = LoadedDocuments::default();
    let mut chunker = ChunkerV1::default();
    let runtime_chunker = IndexManifest::lexical_v1().chunker_version;
    if chunker.profile_id() != runtime_chunker {
        bail!("repository evaluator chunker does not match the product manifest");
    }
    for repository in manifest.repositories {
        let directory = NormalizedPath::new(&repository.directory)
            .with_context(|| format!("validate repository directory {}", repository.directory))?;
        let checkout = root.join(directory.as_str());
        verify_revision(&checkout, &repository.revision)?;
        collect_repository_documents(
            &repository.id,
            &checkout,
            &checkout,
            &mut chunker,
            &mut loaded,
        )?;
    }
    loaded
        .documents
        .sort_by(|left, right| left.doc_id.cmp(&right.doc_id));
    if loaded
        .documents
        .windows(2)
        .any(|pair| pair[0].doc_id == pair[1].doc_id)
    {
        bail!("repository chunker produced a duplicate document ID");
    }
    if loaded.documents.len() != loaded.candidate_to_judgment.len() {
        bail!("repository candidate mapping is incomplete");
    }
    Ok(loaded)
}

fn verify_revision(checkout: &Path, expected: &str) -> Result<()> {
    let output = Command::new("git")
        .arg("-C")
        .arg(checkout)
        .args(["rev-parse", "HEAD"])
        .output()
        .with_context(|| format!("inspect benchmark checkout {}", checkout.display()))?;
    if !output.status.success() {
        bail!(
            "cannot inspect benchmark checkout {}; run scripts/fetch-bench-repos.sh",
            checkout.display()
        );
    }
    let actual =
        String::from_utf8(output.stdout).context("git returned a non-UTF-8 benchmark revision")?;
    if actual.trim() != expected {
        bail!(
            "benchmark checkout {} is at {}, expected {}; rerun scripts/fetch-bench-repos.sh",
            checkout.display(),
            actual.trim(),
            expected
        );
    }
    Ok(())
}

fn collect_repository_documents(
    repository_id: &str,
    checkout: &Path,
    directory: &Path,
    chunker: &mut ChunkerV1,
    loaded: &mut LoadedDocuments,
) -> Result<()> {
    let mut entries = fs::read_dir(directory)
        .with_context(|| format!("read benchmark directory {}", directory.display()))?
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);

    for entry in entries {
        let file_type = entry
            .file_type()
            .with_context(|| format!("inspect benchmark path {}", entry.path().display()))?;
        if file_type.is_symlink() {
            continue;
        }
        let path = entry.path();
        if file_type.is_dir() {
            if !is_ignored_directory(&entry.file_name()) {
                collect_repository_documents(repository_id, checkout, &path, chunker, loaded)?;
            }
            continue;
        }
        if !file_type.is_file() || !is_supported_source(&path) {
            continue;
        }
        let metadata = entry
            .metadata()
            .with_context(|| format!("read benchmark metadata {}", path.display()))?;
        if metadata.len() > MAX_SOURCE_BYTES {
            loaded.skip("oversized_source");
            continue;
        }
        if is_config_or_data(&path) && metadata.len() > MAX_CONFIG_BYTES {
            loaded.skip("oversized_config");
            continue;
        }
        let bytes =
            fs::read(&path).with_context(|| format!("read benchmark source {}", path.display()))?;
        if bytes.iter().take(BINARY_PROBE_BYTES).any(|byte| *byte == 0) {
            loaded.skip("binary");
            continue;
        }
        let Ok(text) = String::from_utf8(bytes) else {
            loaded.skip("invalid_utf8");
            continue;
        };
        if text.trim().is_empty() {
            loaded.skip("empty");
            continue;
        }
        let relative = path
            .strip_prefix(checkout)
            .with_context(|| format!("make {} repository-relative", path.display()))?;
        let normalized = NormalizedPath::new(relative.to_string_lossy().replace('\\', "/"))?;
        let parent_id = DocumentId::new(format!("{repository_id}:{normalized}"))?;
        let corpus_document = CorpusDocument {
            doc_id: parent_id.clone(),
            path: NormalizedPath::new(format!("{repository_id}/{normalized}"))?,
            language: LanguageId::new(language_for_path(&path)),
            text,
        };
        let chunks = chunker.chunk(&corpus_document)?;
        let mut emitted = 0_usize;
        for chunk in chunks {
            if chunk.text.trim().is_empty() {
                loaded.skip("blank_chunk");
                continue;
            }
            if loaded
                .candidate_to_judgment
                .insert(chunk.chunk_id.clone(), parent_id.clone())
                .is_some()
            {
                bail!(
                    "repository chunker produced duplicate ID {}",
                    chunk.chunk_id
                );
            }
            loaded.documents.push(SearchDocument {
                doc_id: chunk.chunk_id,
                path: corpus_document.path.clone(),
                language: chunk.language,
                text: chunk.text,
                embedding: None,
            });
            emitted = emitted.saturating_add(1);
        }
        if emitted > 0 {
            loaded.source_documents = loaded.source_documents.saturating_add(1);
        }
    }
    Ok(())
}

fn is_config_or_data(path: &Path) -> bool {
    matches!(
        path.extension().and_then(std::ffi::OsStr::to_str),
        Some("json" | "toml" | "yaml" | "yml")
    )
}

fn language_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(std::ffi::OsStr::to_str) {
        Some("c" | "h") => "c",
        Some("cc" | "cpp") => "cpp",
        Some("go") => "go",
        Some("java") => "java",
        Some("js" | "jsx") => "javascript",
        Some("php") => "php",
        Some("py") => "python",
        Some("rb") => "ruby",
        Some("rs") => "rust",
        Some("sh") => "bash",
        Some("ts" | "tsx") => "typescript",
        Some("css") => "css",
        Some("html") => "html",
        Some("json") => "json",
        Some("md") => "markdown",
        Some("toml") => "toml",
        Some("yaml" | "yml") => "yaml",
        _ => "text",
    }
}

fn is_ignored_directory(name: &std::ffi::OsStr) -> bool {
    matches!(
        name.to_str(),
        Some(".git" | "node_modules" | "vendor" | ".venv" | "_output" | "target")
    )
}

fn is_supported_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(std::ffi::OsStr::to_str),
        Some(
            "c" | "cc"
                | "cpp"
                | "css"
                | "go"
                | "h"
                | "html"
                | "java"
                | "js"
                | "json"
                | "jsx"
                | "md"
                | "php"
                | "py"
                | "rb"
                | "rs"
                | "sh"
                | "toml"
                | "ts"
                | "tsx"
                | "yaml"
                | "yml"
        )
    )
}

fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    BufReader::new(file)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            result => Some((index, result)),
        })
        .map(|(index, line)| {
            let line =
                line.with_context(|| format!("read {} line {}", path.display(), index + 1))?;
            serde_json::from_str(&line)
                .with_context(|| format!("parse {} line {}", path.display(), index + 1))
        })
        .collect()
}

fn phase0_manifest(backend: Backend) -> Result<IndexManifest> {
    Ok(IndexManifest {
        model_id: "phase0-deterministic-random-stub".into(),
        model_revision: "1".into(),
        embedding_profile: "phase0-none".into(),
        embed_dim: 768,
        mrl_dim: backend.mrl_dim(),
        quantization: Quantization::None,
        tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "0".repeat(64))?,
        chunker_version: "cast-v1".into(),
        fde_params: FdeParams::Disabled,
        schema_version: 1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hay_search::Signals;

    fn candidate(doc_id: &str) -> Candidate {
        Candidate {
            doc_id: DocumentId::new(doc_id).unwrap(),
            score: 1.0,
            signals: Signals::default(),
        }
    }

    #[test]
    fn perfect_ranking_has_perfect_metrics() {
        let candidates = vec![candidate("best"), candidate("okay")];
        let judgments = BTreeMap::from([("best".into(), 3), ("okay".into(), 1)]);
        assert!((ndcg_at(&candidates, &judgments, 10) - 1.0).abs() < f64::EPSILON);
        assert!((recall_at(&candidates, &judgments, 50) - 1.0).abs() < f64::EPSILON);
        assert!((reciprocal_rank(&candidates, &judgments) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn chunk_candidates_collapse_to_their_first_ranked_parent_file() {
        let first_chunk = DocumentId::new("file-a:chunk:0").unwrap();
        let second_chunk = DocumentId::new("file-a:chunk:1").unwrap();
        let other_chunk = DocumentId::new("file-b:chunk:0").unwrap();
        let file_a = DocumentId::new("file-a").unwrap();
        let file_b = DocumentId::new("file-b").unwrap();
        let mapping = BTreeMap::from([
            (first_chunk.clone(), file_a.clone()),
            (second_chunk.clone(), file_a.clone()),
            (other_chunk.clone(), file_b.clone()),
        ]);

        let collapsed = collapse_candidates(
            vec![
                candidate(first_chunk.as_str()),
                candidate(second_chunk.as_str()),
                candidate(other_chunk.as_str()),
            ],
            &mapping,
        )
        .unwrap();

        assert_eq!(collapsed.len(), 2);
        assert_eq!(collapsed[0].doc_id, file_a);
        assert_eq!(collapsed[1].doc_id, file_b);
    }

    #[test]
    fn candidate_collapse_rejects_results_outside_the_loaded_corpus() {
        let error = collapse_candidates(vec![candidate("unknown")], &BTreeMap::new()).unwrap_err();
        assert!(error.to_string().contains("unknown candidate"));
    }

    #[test]
    fn chunkhound_result_headers_preserve_paths_with_spaces() {
        assert_eq!(
            parse_chunkhound_result_header("[17] src/a directory/file.rs"),
            Some((17, "src/a directory/file.rs"))
        );
        assert_eq!(parse_chunkhound_result_header("not a result"), None);
        assert_eq!(parse_chunkhound_result_header("[rank] file.rs"), None);
        assert_eq!(parse_chunkhound_result_header("[0] file.rs"), None);
    }

    #[test]
    fn latency_percentiles_use_nearest_rank() {
        let samples = [
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
        ];
        assert_eq!(
            percentile_duration(&samples, 50).unwrap(),
            Duration::from_millis(20)
        );
        assert_eq!(
            percentile_duration(&samples, 95).unwrap(),
            Duration::from_millis(40)
        );
        assert!(percentile_duration(&[], 50).is_err());
        assert!(percentile_duration(&samples, 0).is_err());
    }

    #[test]
    fn backend_parity_gate_is_inclusive_and_rejects_larger_drift() {
        let duckdb = Metrics {
            ndcg_at_10: 0.50,
            ..Metrics::default()
        };
        let at_limit = Metrics {
            ndcg_at_10: 0.48,
            ..Metrics::default()
        };
        let outside = Metrics {
            ndcg_at_10: 0.479,
            ..Metrics::default()
        };

        enforce_ndcg_parity(duckdb, at_limit).unwrap();
        assert!(enforce_ndcg_parity(duckdb, outside).is_err());
        assert!(
            enforce_ndcg_parity(
                duckdb,
                Metrics {
                    ndcg_at_10: f64::NAN,
                    ..Metrics::default()
                }
            )
            .is_err()
        );
    }
}
