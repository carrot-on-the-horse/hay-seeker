#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Shared provider wiring and exact index-manifest construction.
//!
//! All executable surfaces use this crate so an index built by `hay` is opened
//! with the identical provider identity by direct search, MCP, and evaluation.
//!
//! ```
//! use hay_runtime::{EmbeddingProvider, SearchRuntime, StorageBackend};
//!
//! let runtime = SearchRuntime::from_env(StorageBackend::DuckDb, EmbeddingProvider::None)?;
//! assert!(runtime.embedder.is_none());
//! assert_eq!(runtime.manifest.model_id, "none");
//! # Ok::<(), anyhow::Error>(())
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use cast_embeddings::{
    CloudflareVertexGemini2, CloudflareVertexGemini2Config, CloudflareWorkersAiEmbeddings,
    CloudflareWorkersAiEmbeddingsConfig, LocalOnnxConfig, LocalOnnxEmbedder, LocalStaticConfig,
    LocalStaticEmbedder, OpenAiEmbeddings, OpenAiEmbeddingsConfig, POTION_CODE_16M_V2_PROFILE,
    RetryPolicy, RetryingEmbedder, STATIC_RETRIEVAL_MRL_EN_V1_PROFILE, VOYAGE_DEFAULT_MODEL,
    VoyageEmbeddings, VoyageEmbeddingsConfig,
};
use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
    IndexErrorKind,
};
use hay_search::{IndexManifest, Quantization};
use ring::digest::{SHA256, digest};

mod models;
mod workspace;

pub use cast_embeddings::{ModelFetchEvent, ModelFetchReporter};
pub use models::{
    DOWNLOAD_MODELS_ENV, LOCAL_STATIC_DIR_ENV, MODEL_BASE_URL_ENV, MODEL_CACHE_DIR_ENV,
    ResolvedModels, ensure_models, model_cache_dir, report_to_stderr,
};
pub use workspace::{INDEX_DIRECTORY, INDEX_FILE, Workspace, git_root, prepare_index_directory};

/// Load the nearest `.env` file without overriding variables already exported
/// by the parent process.
///
/// A missing file is a valid configuration. Parse and I/O errors are returned
/// so executables never continue with an unexpectedly partial configuration.
///
/// # Errors
///
/// Returns an error when the discovered file cannot be read or parsed.
pub fn load_dotenv() -> Result<Option<PathBuf>> {
    match dotenvy::dotenv() {
        Ok(path) => Ok(Some(path)),
        Err(error) if error.not_found() => Ok(None),
        Err(error) => Err(error).context("load .env configuration"),
    }
}

/// Storage representation whose manifest is being constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageBackend {
    /// Embedded `DuckDB` with exact floating-point vectors.
    DuckDb,
    /// Elasticsearch with BBQ-quantized dense vectors.
    Elasticsearch,
}

/// Embedding provider selected for an index or query process.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EmbeddingProvider {
    /// Lexical-only retrieval.
    None,
    /// Checksum-pinned, fully offline ONNX checkpoint.
    LocalOnnx,
    /// Checksum-pinned, local-only static code checkpoint.
    LocalStatic,
    /// Gemini Embedding 2 through Cloudflare AI Gateway's Vertex route.
    Gemini,
    /// `OpenAI` text embeddings.
    OpenAi,
    /// `Voyage` code-specialized embeddings.
    Voyage,
    /// `Cloudflare Workers AI` with Qwen3 embeddings.
    CloudflareWorkersAi,
}

/// Exact manifest and optional query/document embedder for one process.
pub struct SearchRuntime {
    /// Manifest that must exactly match the persisted index.
    pub manifest: IndexManifest,
    /// Provider used for document and query vectors when dense search is enabled.
    pub embedder: Option<Arc<dyn Embedder>>,
}

/// Two production runtimes that share one provider session for parity runs.
///
/// The Elasticsearch runtime retains the provider's full embedding width. The
/// `DuckDB` runtime either shares that same embedder or wraps it with the approved
/// normalized 256-dimensional MRL projection for the offline ONNX profile.
pub struct BackendParityRuntime {
    /// `DuckDB` manifest and query embedder.
    pub duckdb: SearchRuntime,
    /// Elasticsearch manifest and full-width query/document embedder.
    pub elasticsearch: SearchRuntime,
}

/// One provider response projected into the two approved storage contracts.
pub struct BackendParityEmbeddingBatch {
    /// Vectors matching the `DuckDB` manifest's stored dimensions.
    pub duckdb: Vec<EmbeddingVector>,
    /// Vectors matching the Elasticsearch manifest's stored dimensions.
    pub elasticsearch: Vec<EmbeddingVector>,
}

impl SearchRuntime {
    /// Builds a runtime from the process environment.
    ///
    /// Each hosted provider reads its documented API key, model revision, and
    /// optional model/dimension/retry settings. See `.env.example`.
    ///
    /// # Errors
    ///
    /// Returns an error when required values are absent or any provider,
    /// retry, URL, dimension, or concurrency setting is invalid.
    pub fn from_env(backend: StorageBackend, provider: EmbeddingProvider) -> Result<Self> {
        Self::from_env_with_models(backend, provider, &ResolvedModels::default())
    }

    /// Builds a runtime from the environment using already-resolved bundles.
    ///
    /// Loading a local provider stays offline and checksum-verified. Callers
    /// that want automatic provisioning run [`ensure_models`] first and pass its
    /// result here; an explicit bundle directory in the environment still wins.
    ///
    /// # Errors
    ///
    /// Returns an error when required values are absent or any provider,
    /// retry, URL, dimension, or concurrency setting is invalid.
    pub fn from_env_with_models(
        backend: StorageBackend,
        provider: EmbeddingProvider,
        models: &ResolvedModels,
    ) -> Result<Self> {
        match provider {
            EmbeddingProvider::None => Ok(Self {
                manifest: IndexManifest::lexical_v1(),
                embedder: None,
            }),
            EmbeddingProvider::LocalOnnx => Self::local_onnx(backend),
            EmbeddingProvider::LocalStatic => Self::local_static(backend, models),
            EmbeddingProvider::Gemini => Self::gemini(backend),
            EmbeddingProvider::OpenAi => Self::openai(backend),
            EmbeddingProvider::Voyage => Self::voyage(backend),
            EmbeddingProvider::CloudflareWorkersAi => Self::cloudflare_workers_ai(backend),
        }
    }

    fn local_onnx(backend: StorageBackend) -> Result<Self> {
        let bundle_dir = std::env::var("HAY_LOCAL_MODEL_DIR")
            .context("HAY_LOCAL_MODEL_DIR is required for local ONNX embeddings")?;
        let stored_dimensions = match backend {
            StorageBackend::DuckDb => 256,
            StorageBackend::Elasticsearch => {
                environment_usize("HAY_LOCAL_RESEARCH_ELASTICSEARCH_DIMENSIONS", 768)?
            }
        };
        let provider = LocalOnnxEmbedder::new(
            LocalOnnxConfig::new(bundle_dir).with_stored_dimensions(stored_dimensions),
        )?;
        let base_dimensions = provider.base_dimensions();
        let revision = provider.model_revision().to_owned();
        let embedder = Arc::new(provider) as Arc<dyn Embedder>;
        Ok(local_runtime_from_embedder(
            backend,
            embedder,
            base_dimensions,
            &revision,
        ))
    }

    fn local_static(backend: StorageBackend, models: &ResolvedModels) -> Result<Self> {
        let bundle_dir = models::static_bundle_dir(models)?;
        let provider =
            LocalStaticEmbedder::new(LocalStaticConfig::new(&bundle_dir)).with_context(|| {
                format!(
                    "open the static embedding bundle at {}; if it is damaged, delete that \
                     directory and re-run to provision it again",
                    bundle_dir.display()
                )
            })?;
        let base_dimensions = provider.identity().dimensions;
        let revision = provider.model_revision().to_owned();
        let embedder = Arc::new(provider) as Arc<dyn Embedder>;
        Ok(local_runtime_from_embedder(
            backend,
            embedder,
            base_dimensions,
            &revision,
        ))
    }

    fn gemini(backend: StorageBackend) -> Result<Self> {
        let token = std::env::var("COTH_HAY_SEEKER_CF_AIG_TOKEN").context(
            "COTH_HAY_SEEKER_CF_AIG_TOKEN is required for Gemini (a Cloudflare AI Gateway Run token)",
        )?;
        let revision = required_revision("GEMINI_MODEL_REVISION", "Gemini")?;
        let endpoint = std::env::var("GEMINI_GATEWAY_URL")
            .context("GEMINI_GATEWAY_URL is required for Gemini")?;
        let dimensions = environment_usize("GEMINI_EMBEDDING_DIMENSIONS", 768)?;
        let concurrency = environment_usize("GEMINI_EMBEDDING_CONCURRENCY", 8)?;
        let max_attempts = environment_usize("GEMINI_EMBEDDING_MAX_ATTEMPTS", 4)?;
        let config = CloudflareVertexGemini2Config::new(endpoint.clone(), token)
            .with_dimensions(dimensions)
            .with_max_concurrency(concurrency);
        let provider = CloudflareVertexGemini2::new(config)?;
        let embedder = retrying(provider, max_attempts)?;
        let route_hash = sha256_hex(endpoint.as_bytes());
        Ok(runtime(
            backend,
            embedder,
            &format!("{revision};route-sha256:{route_hash}"),
        ))
    }

    fn openai(backend: StorageBackend) -> Result<Self> {
        let revision = required_revision("OPENAI_MODEL_REVISION", "OpenAI")?;
        let model = std::env::var("OPENAI_EMBEDDING_MODEL")
            .unwrap_or_else(|_| cast_embeddings::OPENAI_DEFAULT_MODEL.into());
        let dimensions = environment_usize("OPENAI_EMBEDDING_DIMENSIONS", 768)?;
        let max_attempts = environment_usize("OPENAI_EMBEDDING_MAX_ATTEMPTS", 4)?;
        let api_key = std::env::var("COTH_HAY_SEEKER_OPENAI_API_KEY")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let gateway_endpoint = std::env::var("OPENAI_GATEWAY_URL")
            .ok()
            .filter(|value| !value.trim().is_empty());
        let (mut config, endpoint) = if let Some(endpoint) = gateway_endpoint {
            let gateway_token = std::env::var("COTH_HAY_SEEKER_CF_AIG_TOKEN").context(
                "COTH_HAY_SEEKER_CF_AIG_TOKEN is required when OPENAI_GATEWAY_URL is set",
            )?;
            (
                OpenAiEmbeddingsConfig::through_cloudflare(endpoint.clone(), gateway_token),
                endpoint,
            )
        } else {
            let api_key = api_key
                .as_deref()
                .context("COTH_HAY_SEEKER_OPENAI_API_KEY is required for direct OpenAI")?;
            (
                OpenAiEmbeddingsConfig::direct(api_key),
                cast_embeddings::OPENAI_EMBEDDINGS_ENDPOINT.into(),
            )
        };
        if let Some(api_key) = api_key {
            config = config.with_api_key(api_key);
        }
        config = config.with_model(model).with_dimensions(dimensions);
        let provider = OpenAiEmbeddings::new(config)?;
        Ok(runtime(
            backend,
            retrying(provider, max_attempts)?,
            &format!(
                "{revision};route-sha256:{}",
                sha256_hex(endpoint.as_bytes())
            ),
        ))
    }

    fn voyage(backend: StorageBackend) -> Result<Self> {
        let api_key = std::env::var("VOYAGE_API_KEY").context("VOYAGE_API_KEY is required")?;
        let revision = required_revision("VOYAGE_MODEL_REVISION", "Voyage")?;
        let model =
            std::env::var("VOYAGE_EMBEDDING_MODEL").unwrap_or_else(|_| VOYAGE_DEFAULT_MODEL.into());
        let dimensions = environment_usize("VOYAGE_EMBEDDING_DIMENSIONS", 1_024)?;
        let max_attempts = environment_usize("VOYAGE_EMBEDDING_MAX_ATTEMPTS", 4)?;
        let provider = VoyageEmbeddings::new(
            VoyageEmbeddingsConfig::new(api_key)
                .with_model(model)
                .with_dimensions(dimensions),
        )?;
        Ok(runtime(
            backend,
            retrying(provider, max_attempts)?,
            &revision,
        ))
    }

    fn cloudflare_workers_ai(backend: StorageBackend) -> Result<Self> {
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .context("CLOUDFLARE_ACCOUNT_ID is required for Workers AI")?;
        let token = std::env::var("CLOUDFLARE_AI_TOKEN")
            .context("CLOUDFLARE_AI_TOKEN is required for Workers AI")?;
        let revision = required_revision(
            "CLOUDFLARE_WORKERS_AI_MODEL_REVISION",
            "Cloudflare Workers AI",
        )?;
        let max_attempts = environment_usize("CLOUDFLARE_AI_MAX_ATTEMPTS", 4)?;
        let provider = CloudflareWorkersAiEmbeddings::new(
            CloudflareWorkersAiEmbeddingsConfig::new(account_id, token),
        )?;
        Ok(runtime(
            backend,
            retrying(provider, max_attempts)?,
            &revision,
        ))
    }
}

impl BackendParityRuntime {
    /// Builds both backend runtimes around one provider session.
    ///
    /// This is the preferred evaluator/runtime path when the same corpus is
    /// written to both targets. Hosted providers use the identical embedder for
    /// both sides. The local ONNX provider is opened once at its full trained
    /// width and `DuckDB` receives a projecting wrapper over that same session.
    /// The static provider shares its native 256-dimensional output unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when provider configuration is invalid or the derived
    /// manifests are not an approved backend-parity pair.
    pub fn from_env(provider: EmbeddingProvider) -> Result<Self> {
        let elasticsearch = SearchRuntime::from_env(StorageBackend::Elasticsearch, provider)?;
        parity_runtime_from_elasticsearch(provider, elasticsearch)
    }

    /// Embeds a document batch once and returns vectors for both backends.
    ///
    /// # Errors
    ///
    /// Returns [`IndexError`] when no dense provider is configured, provider
    /// output violates its declared identity, or MRL projection fails.
    #[must_use]
    pub fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<BackendParityEmbeddingBatch, IndexError>> {
        Box::pin(async move {
            let embedder = self.elasticsearch.embedder.as_deref().ok_or_else(|| {
                runtime_embedding_error(
                    "parity_embeddings_disabled",
                    "cannot precompute dense parity vectors without an embedding provider",
                )
            })?;
            let vectors = embedder.embed_batch(inputs).await?;
            if vectors.len() != inputs.len() {
                return Err(runtime_embedding_error(
                    "parity_embedding_count",
                    format!(
                        "provider returned {} vectors for {} documents",
                        vectors.len(),
                        inputs.len()
                    ),
                ));
            }

            let mut duckdb = Vec::with_capacity(vectors.len());
            let mut elasticsearch = Vec::with_capacity(vectors.len());
            for vector in vectors {
                validate_provider_vector(&vector, embedder.identity())?;
                let normalize_duckdb =
                    self.duckdb.manifest.mrl_dim < self.elasticsearch.manifest.mrl_dim;
                duckdb.push(project_embedding(
                    vector.clone(),
                    self.duckdb.manifest.mrl_dim,
                    normalize_duckdb,
                )?);
                elasticsearch.push(project_embedding(
                    vector,
                    self.elasticsearch.manifest.mrl_dim,
                    false,
                )?);
            }
            Ok(BackendParityEmbeddingBatch {
                duckdb,
                elasticsearch,
            })
        })
    }
}

fn parity_runtime_from_elasticsearch(
    provider: EmbeddingProvider,
    elasticsearch: SearchRuntime,
) -> Result<BackendParityRuntime> {
    let mut duckdb_manifest = elasticsearch.manifest.clone();
    duckdb_manifest.quantization = match provider {
        EmbeddingProvider::None
        | EmbeddingProvider::Gemini
        | EmbeddingProvider::OpenAi
        | EmbeddingProvider::Voyage
        | EmbeddingProvider::CloudflareWorkersAi => Quantization::None,
        EmbeddingProvider::LocalOnnx | EmbeddingProvider::LocalStatic => {
            Quantization::Int8PerVectorScaleOffset
        }
    };
    if provider == EmbeddingProvider::LocalOnnx {
        duckdb_manifest.mrl_dim = 256;
    }

    let duckdb_embedder =
        match (&elasticsearch.embedder, provider) {
            (Some(embedder), EmbeddingProvider::LocalOnnx) => Some(Arc::new(
                MrlProjectingEmbedder::new(Arc::clone(embedder), duckdb_manifest.mrl_dim)?,
            )
                as Arc<dyn Embedder>),
            (embedder, _) => embedder.clone(),
        };
    let duckdb = SearchRuntime {
        manifest: duckdb_manifest,
        embedder: duckdb_embedder,
    };
    validate_backend_parity(&duckdb.manifest, &elasticsearch.manifest)?;
    Ok(BackendParityRuntime {
        duckdb,
        elasticsearch,
    })
}

struct MrlProjectingEmbedder {
    inner: Arc<dyn Embedder>,
    identity: EmbeddingIdentity,
}

impl MrlProjectingEmbedder {
    fn new(inner: Arc<dyn Embedder>, dimensions: usize) -> Result<Self> {
        if dimensions == 0 || dimensions > inner.identity().dimensions {
            bail!(
                "MRL projection dimensions {dimensions} must be within provider width {}",
                inner.identity().dimensions
            );
        }
        let mut identity = inner.identity().clone();
        identity.dimensions = dimensions;
        Ok(Self { inner, identity })
    }
}

impl Embedder for MrlProjectingEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        Box::pin(async move {
            let vectors = self.inner.embed_batch(inputs).await?;
            vectors
                .into_iter()
                .map(|vector| {
                    validate_provider_vector(&vector, self.inner.identity())?;
                    project_embedding(vector, self.identity.dimensions, true)
                })
                .collect()
        })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move {
            let vector = self.inner.embed_query(text).await?;
            validate_provider_vector(&vector, self.inner.identity())?;
            project_embedding(vector, self.identity.dimensions, true)
        })
    }
}

fn validate_provider_vector(
    vector: &EmbeddingVector,
    expected: &EmbeddingIdentity,
) -> Result<(), IndexError> {
    vector
        .validate()
        .map_err(|error| runtime_embedding_error("parity_embedding_invalid", error.to_string()))?;
    if &vector.identity != expected {
        return Err(runtime_embedding_error(
            "parity_embedding_identity",
            "provider output identity does not match the configured embedder",
        ));
    }
    Ok(())
}

fn project_embedding(
    mut vector: EmbeddingVector,
    dimensions: usize,
    normalize: bool,
) -> Result<EmbeddingVector, IndexError> {
    if dimensions == 0 || dimensions > vector.values.len() {
        return Err(runtime_embedding_error(
            "parity_embedding_dimensions",
            format!(
                "cannot project {} dimensions to {dimensions}",
                vector.values.len()
            ),
        ));
    }
    vector.values.truncate(dimensions);
    vector.identity.dimensions = dimensions;
    if normalize {
        let squared_norm = vector.values.iter().map(|value| value * value).sum::<f32>();
        if !squared_norm.is_finite() || squared_norm <= f32::EPSILON {
            return Err(runtime_embedding_error(
                "parity_embedding_norm",
                "projected embedding has a zero or non-finite norm",
            ));
        }
        let norm = squared_norm.sqrt();
        for value in &mut vector.values {
            *value /= norm;
        }
    }
    vector
        .validate()
        .map_err(|error| runtime_embedding_error("parity_embedding_invalid", error.to_string()))?;
    Ok(vector)
}

fn runtime_embedding_error(code: &str, message: impl Into<String>) -> IndexError {
    IndexError::new(IndexErrorKind::Embedding, code, message)
}

/// Validates that `DuckDB` and Elasticsearch manifests describe one relevance
/// contract with only the permitted backend representation differences.
///
/// Lexical profiles must be identical. Hosted dense profiles may differ only
/// by Elasticsearch BBQ storage. The offline ONNX profile additionally uses
/// the approved 256-dimensional `DuckDB` MRL projection versus the same
/// checkpoint's full-width Elasticsearch representation. The pinned static
/// code profile uses the same trained 256-dimensional width on both backends.
///
/// # Errors
///
/// Returns an error naming the first relevance-contract drift or an
/// unsupported dimension/quantization pair.
pub fn validate_backend_parity(
    duckdb: &IndexManifest,
    elasticsearch: &IndexManifest,
) -> Result<()> {
    duckdb.validate().map_err(anyhow::Error::new)?;
    elasticsearch.validate().map_err(anyhow::Error::new)?;
    parity_field("model_id", &duckdb.model_id, &elasticsearch.model_id)?;
    parity_field(
        "model_revision",
        &duckdb.model_revision,
        &elasticsearch.model_revision,
    )?;
    parity_field(
        "embedding_profile",
        &duckdb.embedding_profile,
        &elasticsearch.embedding_profile,
    )?;
    parity_field("embed_dim", &duckdb.embed_dim, &elasticsearch.embed_dim)?;
    parity_field(
        "tokenizer_hash",
        &duckdb.tokenizer_hash,
        &elasticsearch.tokenizer_hash,
    )?;
    parity_field(
        "chunker_version",
        &duckdb.chunker_version,
        &elasticsearch.chunker_version,
    )?;
    parity_field("fde_params", &duckdb.fde_params, &elasticsearch.fde_params)?;
    parity_field(
        "schema_version",
        &duckdb.schema_version,
        &elasticsearch.schema_version,
    )?;

    if duckdb.model_id == "none" {
        parity_field("mrl_dim", &duckdb.mrl_dim, &elasticsearch.mrl_dim)?;
        if duckdb.quantization != Quantization::None
            || elasticsearch.quantization != Quantization::None
        {
            bail!("lexical backend parity requires quantization=none on both manifests");
        }
        return Ok(());
    }

    match (&duckdb.quantization, &elasticsearch.quantization) {
        (Quantization::None, Quantization::ElasticBbq) => {
            parity_field("mrl_dim", &duckdb.mrl_dim, &elasticsearch.mrl_dim)
        }
        (Quantization::Int8PerVectorScaleOffset, Quantization::ElasticBbq)
            if duckdb.embed_dim == 768 && duckdb.mrl_dim == 256 && elasticsearch.mrl_dim == 768 =>
        {
            Ok(())
        }
        (Quantization::Int8PerVectorScaleOffset, Quantization::ElasticBbq)
            if duckdb.embedding_profile == STATIC_RETRIEVAL_MRL_EN_V1_PROFILE
                && duckdb.embed_dim == 1_024
                && duckdb.mrl_dim == 256
                && elasticsearch.mrl_dim == 1_024 =>
        {
            Ok(())
        }
        (Quantization::Int8PerVectorScaleOffset, Quantization::ElasticBbq)
            if duckdb.embedding_profile == POTION_CODE_16M_V2_PROFILE
                && duckdb.embed_dim == 256
                && duckdb.mrl_dim == 256
                && elasticsearch.mrl_dim == 256 =>
        {
            Ok(())
        }
        _ => bail!(
            "unsupported backend representation pair: DuckDB {:?}/{}d, Elasticsearch {:?}/{}d",
            duckdb.quantization,
            duckdb.mrl_dim,
            elasticsearch.quantization,
            elasticsearch.mrl_dim
        ),
    }
}

fn parity_field<T: std::fmt::Debug + PartialEq>(
    field: &str,
    duckdb: &T,
    elasticsearch: &T,
) -> Result<()> {
    if duckdb != elasticsearch {
        bail!(
            "backend parity drift in {field}: DuckDB {duckdb:?}, Elasticsearch {elasticsearch:?}"
        );
    }
    Ok(())
}

fn local_runtime_from_embedder(
    backend: StorageBackend,
    embedder: Arc<dyn Embedder>,
    base_dimensions: usize,
    revision: &str,
) -> SearchRuntime {
    let identity = embedder.identity();
    let manifest = IndexManifest {
        model_id: identity.model.clone(),
        model_revision: format!("{revision};hay-code-analyzer-v2-path3-id2"),
        embedding_profile: identity.profile.clone(),
        embed_dim: base_dimensions,
        mrl_dim: identity.dimensions,
        quantization: match backend {
            StorageBackend::DuckDb => Quantization::Int8PerVectorScaleOffset,
            StorageBackend::Elasticsearch => Quantization::ElasticBbq,
        },
        ..IndexManifest::lexical_v1()
    };
    SearchRuntime {
        manifest,
        embedder: Some(embedder),
    }
}

fn retrying<E: Embedder + 'static>(provider: E, max_attempts: usize) -> Result<Arc<dyn Embedder>> {
    let retry_policy = RetryPolicy::with_max_attempts(max_attempts)?;
    Ok(Arc::new(RetryingEmbedder::new(provider, retry_policy)))
}

fn runtime(
    backend: StorageBackend,
    embedder: Arc<dyn Embedder>,
    provider_revision: &str,
) -> SearchRuntime {
    let identity = embedder.identity();
    let manifest = IndexManifest {
        model_id: identity.model.clone(),
        model_revision: format!(
            "{};hay-code-analyzer-v2-path3-id2",
            provider_revision.trim()
        ),
        embedding_profile: identity.profile.clone(),
        embed_dim: identity.dimensions,
        mrl_dim: identity.dimensions,
        quantization: match backend {
            StorageBackend::DuckDb => Quantization::None,
            StorageBackend::Elasticsearch => Quantization::ElasticBbq,
        },
        ..IndexManifest::lexical_v1()
    };
    SearchRuntime {
        manifest,
        embedder: Some(embedder),
    }
}

fn required_revision(name: &'static str, provider: &str) -> Result<String> {
    let revision = std::env::var(name).with_context(|| {
        format!("{name} is required to make {provider} index invalidation explicit")
    })?;
    if revision.trim().is_empty() {
        bail!("{name} must not be blank");
    }
    Ok(revision.trim().to_owned())
}

fn environment_usize(name: &str, default: usize) -> Result<usize> {
    std::env::var(name).map_or(Ok(default), |value| {
        value
            .parse::<usize>()
            .with_context(|| format!("{name} must be a positive integer"))
    })
}

fn sha256_hex(value: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = digest(&SHA256, value);
    let mut encoded = String::with_capacity(digest.as_ref().len() * 2);
    for byte in digest.as_ref() {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use cast_index::{DocumentId, EmbeddingInput};

    struct TestEmbedder {
        identity: EmbeddingIdentity,
    }

    impl Embedder for TestEmbedder {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }

        fn embed_batch<'a>(
            &'a self,
            _inputs: &'a [EmbeddingInput<'a>],
        ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
            Box::pin(async { unreachable!("manifest test does not call the provider") })
        }

        fn embed_query<'a>(
            &'a self,
            _text: &'a str,
        ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
            Box::pin(async { unreachable!("manifest test does not call the provider") })
        }
    }

    struct VectorEmbedder {
        identity: EmbeddingIdentity,
        calls: Arc<AtomicUsize>,
    }

    impl Embedder for VectorEmbedder {
        fn identity(&self) -> &EmbeddingIdentity {
            &self.identity
        }

        fn embed_batch<'a>(
            &'a self,
            inputs: &'a [EmbeddingInput<'a>],
        ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(inputs
                    .iter()
                    .map(|_| EmbeddingVector {
                        identity: self.identity.clone(),
                        values: vec![1.0; self.identity.dimensions],
                    })
                    .collect())
            })
        }

        fn embed_query<'a>(
            &'a self,
            _text: &'a str,
        ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move {
                Ok(EmbeddingVector {
                    identity: self.identity.clone(),
                    values: vec![1.0; self.identity.dimensions],
                })
            })
        }
    }

    #[test]
    fn lexical_runtime_is_provider_free_and_backend_portable() {
        for backend in [StorageBackend::DuckDb, StorageBackend::Elasticsearch] {
            let runtime = SearchRuntime::from_env(backend, EmbeddingProvider::None).unwrap();
            assert_eq!(runtime.manifest, IndexManifest::lexical_v1());
            assert!(runtime.embedder.is_none());
        }
    }

    #[test]
    fn endpoint_fingerprint_is_stable() {
        assert_eq!(
            sha256_hex(b"https://gateway.example/v1/model"),
            "404b665a905e0645840b6d4258a9b348b8eb78e238f1c01d042637a0ad44815d"
        );
    }

    #[test]
    fn provider_manifest_is_shared_except_for_backend_quantization() {
        let make_embedder = || {
            Arc::new(TestEmbedder {
                identity: EmbeddingIdentity {
                    provider: "test".into(),
                    model: "test-model".into(),
                    dimensions: 768,
                    profile: "query-document-v1".into(),
                },
            }) as Arc<dyn Embedder>
        };
        let local = runtime(StorageBackend::DuckDb, make_embedder(), "revision-7");
        let remote = runtime(StorageBackend::Elasticsearch, make_embedder(), "revision-7");

        assert_eq!(local.manifest.model_id, remote.manifest.model_id);
        assert_eq!(
            local.manifest.model_revision,
            remote.manifest.model_revision
        );
        assert_eq!(local.manifest.embedding_profile, "query-document-v1");
        assert_eq!(local.manifest.quantization, Quantization::None);
        assert_eq!(remote.manifest.quantization, Quantization::ElasticBbq);
        validate_backend_parity(&local.manifest, &remote.manifest).unwrap();
    }

    #[test]
    fn local_checkpoint_uses_only_the_permitted_backend_representation_differences() {
        let make_embedder = |dimensions| {
            Arc::new(TestEmbedder {
                identity: EmbeddingIdentity {
                    provider: "local-onnx".into(),
                    model: "google/embeddinggemma-300m".into(),
                    dimensions,
                    profile: "embeddinggemma-code-retrieval-final-pooled-mrl-v1".into(),
                },
            }) as Arc<dyn Embedder>
        };
        let local = local_runtime_from_embedder(
            StorageBackend::DuckDb,
            make_embedder(256),
            768,
            "checkpoint-commit;bundle-sha256:abc",
        );
        let remote = local_runtime_from_embedder(
            StorageBackend::Elasticsearch,
            make_embedder(768),
            768,
            "checkpoint-commit;bundle-sha256:abc",
        );

        assert_eq!(local.manifest.model_id, remote.manifest.model_id);
        assert_eq!(
            local.manifest.model_revision,
            remote.manifest.model_revision
        );
        assert_eq!(
            local.manifest.embedding_profile,
            remote.manifest.embedding_profile
        );
        assert_eq!(local.manifest.embed_dim, remote.manifest.embed_dim);
        assert_eq!(local.manifest.mrl_dim, 256);
        assert_eq!(remote.manifest.mrl_dim, 768);
        assert_eq!(
            local.manifest.quantization,
            Quantization::Int8PerVectorScaleOffset
        );
        assert_eq!(remote.manifest.quantization, Quantization::ElasticBbq);
        validate_backend_parity(&local.manifest, &remote.manifest).unwrap();
    }

    #[test]
    fn static_research_checkpoint_requires_its_trained_1024_dimension_pair() {
        let make_embedder = |dimensions| {
            Arc::new(TestEmbedder {
                identity: EmbeddingIdentity {
                    provider: "local-onnx".into(),
                    model: "sentence-transformers/static-retrieval-mrl-en-v1".into(),
                    dimensions,
                    profile: STATIC_RETRIEVAL_MRL_EN_V1_PROFILE.into(),
                },
            }) as Arc<dyn Embedder>
        };
        let local = local_runtime_from_embedder(
            StorageBackend::DuckDb,
            make_embedder(256),
            1_024,
            "checkpoint;bundle-sha256:abc",
        );
        let remote = local_runtime_from_embedder(
            StorageBackend::Elasticsearch,
            make_embedder(1_024),
            1_024,
            "checkpoint;bundle-sha256:abc",
        );

        validate_backend_parity(&local.manifest, &remote.manifest).unwrap();

        let mut untrained_width = remote.manifest.clone();
        untrained_width.mrl_dim = 768;
        assert!(
            validate_backend_parity(&local.manifest, &untrained_width)
                .unwrap_err()
                .to_string()
                .contains("unsupported backend representation pair")
        );
    }

    #[test]
    fn static_code_checkpoint_requires_its_native_256_dimension_pair() {
        let make_embedder = || {
            Arc::new(TestEmbedder {
                identity: EmbeddingIdentity {
                    provider: "local-static".into(),
                    model: "minishlab/potion-code-16M-v2".into(),
                    dimensions: 256,
                    profile: POTION_CODE_16M_V2_PROFILE.into(),
                },
            }) as Arc<dyn Embedder>
        };
        let local = local_runtime_from_embedder(
            StorageBackend::DuckDb,
            make_embedder(),
            256,
            "checkpoint;bundle-sha256:abc",
        );
        let remote = local_runtime_from_embedder(
            StorageBackend::Elasticsearch,
            make_embedder(),
            256,
            "checkpoint;bundle-sha256:abc",
        );

        validate_backend_parity(&local.manifest, &remote.manifest).unwrap();

        let mut wrong_width = remote.manifest.clone();
        wrong_width.mrl_dim = 768;
        assert!(validate_backend_parity(&local.manifest, &wrong_width).is_err());
    }

    #[tokio::test]
    async fn parity_runtime_embeds_once_and_normalizes_the_duckdb_projection() {
        let calls = Arc::new(AtomicUsize::new(0));
        let embedder = Arc::new(VectorEmbedder {
            identity: EmbeddingIdentity {
                provider: "local-onnx".into(),
                model: "snowflake-arctic-embed-m-v2.0".into(),
                dimensions: 768,
                profile: "windowed-mrl-v1".into(),
            },
            calls: Arc::clone(&calls),
        }) as Arc<dyn Embedder>;
        let elasticsearch = local_runtime_from_embedder(
            StorageBackend::Elasticsearch,
            embedder,
            768,
            "checkpoint;bundle-sha256:abc",
        );
        let parity =
            parity_runtime_from_elasticsearch(EmbeddingProvider::LocalOnnx, elasticsearch).unwrap();
        let document_id = DocumentId::new("document").unwrap();
        let inputs = [EmbeddingInput {
            document_id: &document_id,
            text: "document text",
        }];

        let batch = parity.embed_batch(&inputs).await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(batch.duckdb[0].values.len(), 256);
        assert_eq!(batch.elasticsearch[0].values.len(), 768);
        let norm = batch.duckdb[0]
            .values
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1.0e-5);
        assert_eq!(
            parity
                .duckdb
                .embedder
                .as_ref()
                .unwrap()
                .identity()
                .dimensions,
            256
        );
        assert_eq!(
            parity
                .elasticsearch
                .embedder
                .as_ref()
                .unwrap()
                .identity()
                .dimensions,
            768
        );
    }

    #[test]
    fn backend_parity_rejects_relevance_drift_and_unapproved_dimensions() {
        let local = IndexManifest::lexical_v1();
        let mut drifted = local.clone();
        drifted.chunker_version = "different-chunker".into();
        assert!(
            validate_backend_parity(&local, &drifted)
                .unwrap_err()
                .to_string()
                .contains("chunker_version")
        );

        let make_embedder = |dimensions| {
            Arc::new(TestEmbedder {
                identity: EmbeddingIdentity {
                    provider: "local-onnx".into(),
                    model: "google/embeddinggemma-300m".into(),
                    dimensions,
                    profile: "embeddinggemma-code-retrieval-final-pooled-mrl-v1".into(),
                },
            }) as Arc<dyn Embedder>
        };
        let local = local_runtime_from_embedder(
            StorageBackend::DuckDb,
            make_embedder(256),
            768,
            "checkpoint",
        );
        let mut remote = local_runtime_from_embedder(
            StorageBackend::Elasticsearch,
            make_embedder(768),
            768,
            "checkpoint",
        );
        remote.manifest.mrl_dim = 512;

        assert!(
            validate_backend_parity(&local.manifest, &remote.manifest)
                .unwrap_err()
                .to_string()
                .contains("unsupported backend representation pair")
        );
    }
}
