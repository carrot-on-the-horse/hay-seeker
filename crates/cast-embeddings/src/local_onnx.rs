use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
    IndexErrorKind,
};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::{Encoding, PaddingParams, Tokenizer, TruncationParams};

#[cfg(target_os = "macos")]
use dispatch2::{DispatchQoS, DispatchQueue, GlobalQueueIdentifier};

const BUNDLE_MANIFEST: &str = "bundle.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const BUNDLE_SCHEMA_VERSION: u32 = 1;
const EMBEDDINGGEMMA_PROFILE: &str = "embeddinggemma-code-retrieval-final-pooled-mrl-v1";
const NOMIC_V1_5_PROFILE: &str = "nomic-embed-text-v1.5-search-mean-layernorm-mrl-v1";
const ARCTIC_M_V2_PROFILE: &str = "snowflake-arctic-embed-m-v2.0-query-cls-mrl-v1";
const ARCTIC_M_V2_WINDOWED_PROFILE: &str =
    "snowflake-arctic-embed-m-v2.0-query-cls-max256-doc-window256-stride224-token-mean-mrl-v1";
/// Exact inference profile for the official static retrieval research model.
pub const STATIC_RETRIEVAL_MRL_EN_V1_PROFILE: &str =
    "sentence-transformers-static-retrieval-mrl-en-v1-nospecial-mean-mrl-v1";

/// Hardware selected for local ONNX inference.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LocalExecutionProvider {
    /// Apple's Core ML execution provider.
    CoreMl,
    /// ONNX Runtime's portable CPU execution provider.
    Cpu,
}

/// Configuration for an offline ONNX embedding bundle.
#[derive(Clone, Debug)]
pub struct LocalOnnxConfig {
    bundle_dir: PathBuf,
    stored_dimensions: usize,
    prefer_core_ml: bool,
    max_batch_size: usize,
}

impl LocalOnnxConfig {
    /// Creates a configuration that stores 256 MRL dimensions and prefers
    /// Core ML on macOS.
    #[must_use]
    pub fn new(bundle_dir: impl Into<PathBuf>) -> Self {
        Self {
            bundle_dir: bundle_dir.into(),
            stored_dimensions: 256,
            prefer_core_ml: cfg!(target_os = "macos"),
            max_batch_size: 8,
        }
    }

    /// Selects the MRL width stored in the local index.
    #[must_use]
    pub const fn with_stored_dimensions(mut self, dimensions: usize) -> Self {
        self.stored_dimensions = dimensions;
        self
    }

    /// Enables or disables the Core ML preference on macOS.
    #[must_use]
    pub const fn with_core_ml(mut self, enabled: bool) -> Self {
        self.prefer_core_ml = enabled;
        self
    }

    /// Caps one ONNX invocation independently of storage-backend batch sizes.
    #[must_use]
    pub const fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }
}

/// Failure while validating or opening a local embedding bundle.
#[derive(Debug, Error)]
pub enum LocalOnnxError {
    /// The bundle manifest or one of its artifacts could not be read.
    #[error("failed to read local embedding bundle: {0}")]
    Read(String),
    /// The bundle manifest is malformed or violates the local model contract.
    #[error("invalid local embedding bundle: {0}")]
    Invalid(String),
    /// An artifact digest does not match its pinned value.
    #[error(
        "local embedding artifact checksum mismatch for {path}: expected {expected}, got {actual}"
    )]
    Checksum {
        /// Relative artifact path.
        path: String,
        /// SHA-256 declared by the bundle.
        expected: String,
        /// SHA-256 computed from the local file.
        actual: String,
    },
    /// The tokenizer could not be initialized.
    #[error("failed to initialize local tokenizer: {0}")]
    Tokenizer(String),
    /// ONNX Runtime could not initialize the model.
    #[error("failed to initialize local ONNX model: {0}")]
    Runtime(String),
}

/// Offline, checksum-pinned ONNX text embedder.
///
/// Each supported profile pins whether the graph returns token states or a
/// final sentence embedding, plus its output width, pooling, prompt,
/// document-window, MRL, and normalization rules. The adapter never downloads
/// model artifacts.
pub struct LocalOnnxEmbedder {
    identity: EmbeddingIdentity,
    base_dimensions: usize,
    model_revision: String,
    execution_provider: LocalExecutionProvider,
    fallback_reason: Option<String>,
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    manifest: BundleManifest,
    max_batch_size: usize,
}

impl LocalOnnxEmbedder {
    /// Verifies every bundle artifact before creating an ONNX session.
    ///
    /// # Errors
    ///
    /// Returns [`LocalOnnxError`] for malformed metadata, unsafe artifact
    /// paths, checksum differences, tokenizer errors, or ONNX initialization
    /// failures.
    pub fn new(config: LocalOnnxConfig) -> Result<Self, LocalOnnxError> {
        let LocalOnnxConfig {
            bundle_dir,
            stored_dimensions,
            prefer_core_ml,
            max_batch_size,
        } = config;
        let bundle_dir = bundle_dir
            .canonicalize()
            .map_err(|error| LocalOnnxError::Read(error.to_string()))?;
        let manifest_path = bundle_dir.join(BUNDLE_MANIFEST);
        let manifest_bytes = read_bounded(&manifest_path, MAX_MANIFEST_BYTES)?;
        let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| LocalOnnxError::Invalid(error.to_string()))?;
        manifest.validate(stored_dimensions)?;
        if !(1..=64).contains(&max_batch_size) {
            return Err(LocalOnnxError::Invalid(
                "local ONNX max batch size must be between 1 and 64".into(),
            ));
        }
        verify_artifacts(&bundle_dir, &manifest)?;

        let tokenizer_path = safe_artifact_path(&bundle_dir, &manifest.tokenizer_file)?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| LocalOnnxError::Tokenizer(error.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: manifest.max_length,
                stride: manifest.document_window_overlap_tokens,
                ..TruncationParams::default()
            }))
            .map_err(|error| LocalOnnxError::Tokenizer(error.to_string()))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let model_path = safe_artifact_path(&bundle_dir, &manifest.model_file)?;
        let core_ml_cache = bundle_dir.join(".coreml-cache");
        let core_ml_enabled = prefer_core_ml && manifest.core_ml_compatible;
        let (session, execution_provider, mut fallback_reason) =
            open_session(&model_path, &core_ml_cache, core_ml_enabled)?;
        if prefer_core_ml && !manifest.core_ml_compatible {
            fallback_reason = Some(
                "bundle declares Core ML incompatible for this graph; using the faster CPU path"
                    .into(),
            );
        }
        validate_graph_contract(&session, &manifest)?;

        let manifest_hash = hex_digest(&manifest_bytes);
        let model_revision = format!("{};bundle-sha256:{manifest_hash}", manifest.model_revision);
        let identity = EmbeddingIdentity {
            provider: "local-onnx".into(),
            model: manifest.model_id.clone(),
            dimensions: stored_dimensions,
            profile: manifest.embedding_profile.clone(),
        };

        Ok(Self {
            identity,
            base_dimensions: manifest.base_dimensions,
            model_revision,
            execution_provider,
            fallback_reason,
            tokenizer,
            session: Mutex::new(session),
            manifest,
            max_batch_size,
        })
    }

    /// Full output width produced by the portable checkpoint before MRL.
    #[must_use]
    pub const fn base_dimensions(&self) -> usize {
        self.base_dimensions
    }

    /// Immutable model revision plus the bundle-manifest checksum.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    /// Execution provider selected while opening the bundle.
    #[must_use]
    pub const fn execution_provider(&self) -> LocalExecutionProvider {
        self.execution_provider
    }

    /// Reason the configured Core ML preference was not used.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    fn embed_document_texts(
        &self,
        texts: &[String],
        prefix: &str,
    ) -> Result<Vec<EmbeddingVector>, IndexError> {
        run_at_indexing_qos(|| {
            self.embed_texts_inner(texts, prefix, self.manifest.document_aggregation)
        })
    }

    fn embed_texts_inner(
        &self,
        texts: &[String],
        prefix: &str,
        aggregation: DocumentAggregation,
    ) -> Result<Vec<EmbeddingVector>, IndexError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut ordered = texts.iter().enumerate().collect::<Vec<_>>();
        ordered.sort_by_key(|(index, text)| (text.len(), *index));
        let mut embeddings = std::iter::repeat_with(|| None)
            .take(texts.len())
            .collect::<Vec<_>>();
        for batch in ordered.chunks(self.max_batch_size) {
            let batch_texts = batch
                .iter()
                .map(|(_, text)| (*text).clone())
                .collect::<Vec<_>>();
            let batch_embeddings = self.embed_micro_batch(&batch_texts, prefix, aggregation)?;
            if batch_embeddings.len() != batch.len() {
                return Err(embedding_error(
                    "local_output_count",
                    "model output count does not match input count",
                ));
            }
            for ((index, _), embedding) in batch.iter().zip(batch_embeddings) {
                embeddings[*index] = Some(embedding);
            }
        }
        embeddings
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                embedding_error(
                    "local_output_count",
                    "model omitted one or more input embeddings",
                )
            })
    }

    fn embed_micro_batch(
        &self,
        texts: &[String],
        prefix: &str,
        aggregation: DocumentAggregation,
    ) -> Result<Vec<EmbeddingVector>, IndexError> {
        let prompted = texts
            .iter()
            .map(|text| format!("{prefix}{text}"))
            .collect::<Vec<_>>();
        let encodings = self
            .tokenizer
            .encode_batch(prompted, self.manifest.add_special_tokens)
            .map_err(|error| embedding_error("local_tokenize", error.to_string()))?;
        let (windows, owners) = document_windows(&encodings, aggregation);
        let mut window_embeddings = Vec::with_capacity(windows.len());
        for batch in windows.chunks(self.max_batch_size) {
            window_embeddings.extend(self.infer_encodings(batch)?);
        }
        let base_embeddings = match aggregation {
            DocumentAggregation::FirstWindow => window_embeddings,
            DocumentAggregation::TokenWeightedMeanWindows => aggregate_document_windows(
                &windows,
                &owners,
                window_embeddings,
                texts.len(),
                self.base_dimensions,
            )?,
        };
        finalize_embeddings(base_embeddings, self.identity.dimensions, &self.identity)
    }

    fn infer_encodings(&self, encodings: &[&Encoding]) -> Result<Vec<Vec<f32>>, IndexError> {
        let sequence_length = encodings
            .first()
            .map(|encoding| encoding.get_ids().len())
            .ok_or_else(|| embedding_error("local_tokenize", "tokenizer returned no encodings"))?;
        if sequence_length == 0
            || encodings
                .iter()
                .any(|encoding| encoding.get_ids().len() != sequence_length)
        {
            return Err(embedding_error(
                "local_tokenize",
                "tokenizer returned empty or ragged padded encodings",
            ));
        }

        let batch_size = encodings.len();
        let input_ids = encodings
            .iter()
            .flat_map(|encoding| encoding.get_ids().iter().map(|&value| i64::from(value)))
            .collect::<Vec<_>>();
        let attention_mask = encodings
            .iter()
            .flat_map(|encoding| {
                encoding
                    .get_attention_mask()
                    .iter()
                    .map(|&value| i64::from(value))
            })
            .collect::<Vec<_>>();
        let token_type_ids = self.manifest.token_type_ids_name.as_ref().map(|_| {
            encodings
                .iter()
                .flat_map(|encoding| {
                    encoding
                        .get_type_ids()
                        .iter()
                        .map(|&value| i64::from(value))
                })
                .collect::<Vec<_>>()
        });
        let input_tensor = Tensor::from_array(([batch_size, sequence_length], input_ids))
            .map_err(|error| embedding_error("local_tensor", error.to_string()))?;
        let mask_tensor =
            Tensor::from_array(([batch_size, sequence_length], attention_mask.clone()))
                .map_err(|error| embedding_error("local_tensor", error.to_string()))?;

        let mut session = self.session.lock().map_err(|_| {
            embedding_error("local_session_poisoned", "ONNX session lock was poisoned")
        })?;
        let outputs = if let (Some(name), Some(type_ids)) =
            (self.manifest.token_type_ids_name.as_deref(), token_type_ids)
        {
            let type_ids_tensor = Tensor::from_array(([batch_size, sequence_length], type_ids))
                .map_err(|error| embedding_error("local_tensor", error.to_string()))?;
            session.run(ort::inputs! {
                self.manifest.input_ids_name.as_str() => input_tensor,
                self.manifest.attention_mask_name.as_str() => mask_tensor,
                name => type_ids_tensor
            })
        } else {
            session.run(ort::inputs! {
                self.manifest.input_ids_name.as_str() => input_tensor,
                self.manifest.attention_mask_name.as_str() => mask_tensor
            })
        }
        .map_err(|error| embedding_error("local_inference", error.to_string()))?;
        let output = outputs
            .get(&self.manifest.output_name)
            .ok_or_else(|| embedding_error("local_output", "configured output is missing"))?;
        let (shape, values) = output
            .try_extract_tensor::<f32>()
            .map_err(|error| embedding_error("local_output", error.to_string()))?;
        let embeddings = postprocess_embeddings(
            self.manifest.output_transform,
            shape.as_ref(),
            values,
            &attention_mask,
            batch_size,
            sequence_length,
            self.base_dimensions,
        )?;
        Ok(embeddings)
    }
}

fn document_windows(
    encodings: &[Encoding],
    aggregation: DocumentAggregation,
) -> (Vec<&Encoding>, Vec<usize>) {
    let mut windows = Vec::new();
    let mut owners = Vec::new();
    for (owner, encoding) in encodings.iter().enumerate() {
        windows.push(encoding);
        owners.push(owner);
        if aggregation == DocumentAggregation::TokenWeightedMeanWindows {
            for overflow in encoding.get_overflowing() {
                windows.push(overflow);
                owners.push(owner);
            }
        }
    }
    (windows, owners)
}

fn aggregate_document_windows(
    windows: &[&Encoding],
    owners: &[usize],
    mut embeddings: Vec<Vec<f32>>,
    document_count: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>, IndexError> {
    if windows.len() != owners.len() || windows.len() != embeddings.len() {
        return Err(embedding_error(
            "local_window_count",
            "token windows, owners, and model outputs have different counts",
        ));
    }
    let mut sums = vec![vec![0.0_f32; dimensions]; document_count];
    let mut weights = vec![0.0_f32; document_count];
    for (position, ((window, owner), embedding)) in
        windows.iter().zip(owners).zip(&mut embeddings).enumerate()
    {
        if *owner >= document_count || embedding.len() != dimensions {
            return Err(embedding_error(
                "local_window_shape",
                format!("window {position} does not match its document or embedding width"),
            ));
        }
        normalize(embedding)?;
        let token_count = window
            .get_attention_mask()
            .iter()
            .map(|value| usize::try_from(*value).unwrap_or(usize::MAX))
            .sum::<usize>();
        let weight = f32::from(u16::try_from(token_count).map_err(|error| {
            embedding_error(
                "local_window_tokens",
                format!("window token count is too large: {error}"),
            )
        })?);
        if weight <= 0.0 {
            return Err(embedding_error(
                "local_window_tokens",
                "document window contains no unpadded tokens",
            ));
        }
        for (sum, value) in sums[*owner].iter_mut().zip(embedding) {
            *sum += *value * weight;
        }
        weights[*owner] += weight;
    }
    for (owner, (sum, weight)) in sums.iter_mut().zip(weights).enumerate() {
        if weight <= 0.0 {
            return Err(embedding_error(
                "local_window_count",
                format!("document {owner} produced no embedding windows"),
            ));
        }
        for value in sum {
            *value /= weight;
        }
    }
    Ok(sums)
}

fn finalize_embeddings(
    embeddings: Vec<Vec<f32>>,
    stored_dimensions: usize,
    identity: &EmbeddingIdentity,
) -> Result<Vec<EmbeddingVector>, IndexError> {
    embeddings
        .into_iter()
        .map(|embedding| {
            let values = embedding.get(..stored_dimensions).ok_or_else(|| {
                embedding_error(
                    "local_output_shape",
                    "model output is narrower than the stored MRL dimension",
                )
            })?;
            let mut values = values.to_vec();
            normalize(&mut values)?;
            Ok(EmbeddingVector {
                identity: identity.clone(),
                values,
            })
        })
        .collect()
}

#[cfg(target_os = "macos")]
fn run_at_indexing_qos<T: Send>(
    work: impl FnOnce() -> Result<T, IndexError> + Send,
) -> Result<T, IndexError> {
    let mut result = None;
    let queue = DispatchQueue::global_queue(GlobalQueueIdentifier::QualityOfService(
        DispatchQoS::Utility,
    ));
    queue.exec_sync(|| {
        result = Some(work());
    });
    result
        .ok_or_else(|| embedding_error("local_qos", "indexing QoS work did not return a result"))?
}

#[cfg(not(target_os = "macos"))]
fn run_at_indexing_qos<T>(work: impl FnOnce() -> Result<T, IndexError>) -> Result<T, IndexError> {
    work()
}

impl Embedder for LocalOnnxEmbedder {
    fn identity(&self) -> &EmbeddingIdentity {
        &self.identity
    }

    fn embed_batch<'a>(
        &'a self,
        inputs: &'a [EmbeddingInput<'a>],
    ) -> BoxFuture<'a, Result<Vec<EmbeddingVector>, IndexError>> {
        let texts = inputs
            .iter()
            .map(|input| input.text.to_owned())
            .collect::<Vec<_>>();
        Box::pin(async move { self.embed_document_texts(&texts, &self.manifest.document_prefix) })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move {
            let mut embeddings = self.embed_texts_inner(
                &[text.to_owned()],
                &self.manifest.query_prefix,
                DocumentAggregation::FirstWindow,
            )?;
            embeddings
                .pop()
                .ok_or_else(|| embedding_error("local_output", "model returned no query embedding"))
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BundleManifest {
    schema_version: u32,
    model_id: String,
    model_revision: String,
    model_file: String,
    tokenizer_file: String,
    artifacts: Vec<ArtifactDigest>,
    input_ids_name: String,
    attention_mask_name: String,
    #[serde(default)]
    token_type_ids_name: Option<String>,
    output_name: String,
    #[serde(default)]
    output_transform: OutputTransform,
    #[serde(default = "default_add_special_tokens")]
    add_special_tokens: bool,
    #[serde(default = "default_core_ml_compatible")]
    core_ml_compatible: bool,
    max_length: usize,
    #[serde(default)]
    document_window_overlap_tokens: usize,
    #[serde(default)]
    document_aggregation: DocumentAggregation,
    base_dimensions: usize,
    #[serde(default = "embeddinggemma_profile")]
    embedding_profile: String,
    document_prefix: String,
    query_prefix: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OutputTransform {
    #[default]
    FinalPooled,
    MeanPoolLayerNorm,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DocumentAggregation {
    #[default]
    FirstWindow,
    TokenWeightedMeanWindows,
}

fn embeddinggemma_profile() -> String {
    EMBEDDINGGEMMA_PROFILE.into()
}

const fn default_core_ml_compatible() -> bool {
    true
}

const fn default_add_special_tokens() -> bool {
    true
}

impl BundleManifest {
    fn validate(&self, stored_dimensions: usize) -> Result<(), LocalOnnxError> {
        if self.schema_version != BUNDLE_SCHEMA_VERSION {
            return Err(LocalOnnxError::Invalid(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        if self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.embedding_profile.trim().is_empty()
            || self.input_ids_name.trim().is_empty()
            || self.attention_mask_name.trim().is_empty()
            || self.output_name.trim().is_empty()
        {
            return Err(LocalOnnxError::Invalid(
                "model identity and graph tensor names must not be blank".into(),
            ));
        }
        if self.max_length == 0 || self.max_length > 2_048 {
            return Err(LocalOnnxError::Invalid(
                "max_length must be between 1 and 2048".into(),
            ));
        }
        if self.document_window_overlap_tokens >= self.max_length {
            return Err(LocalOnnxError::Invalid(
                "document window overlap must be smaller than max_length".into(),
            ));
        }
        let trained_dimensions = self.trained_dimensions()?;
        if !trained_dimensions.contains(&stored_dimensions)
            || stored_dimensions > self.base_dimensions
        {
            return Err(LocalOnnxError::Invalid(format!(
                "stored dimensions {stored_dimensions} are not a trained MRL width for profile {}",
                self.embedding_profile
            )));
        }
        if self.artifacts.is_empty() {
            return Err(LocalOnnxError::Invalid(
                "artifact checksum list must not be empty".into(),
            ));
        }
        Ok(())
    }

    fn trained_dimensions(&self) -> Result<&'static [usize], LocalOnnxError> {
        match self.embedding_profile.as_str() {
            EMBEDDINGGEMMA_PROFILE
                if self.output_transform == OutputTransform::FinalPooled
                    && self.add_special_tokens
                    && self.token_type_ids_name.is_none()
                    && self.base_dimensions == 768
                    && self.max_length == 2_048
                    && self.document_window_overlap_tokens == 0
                    && self.document_aggregation == DocumentAggregation::FirstWindow
                    && self.document_prefix == "title: none | text: "
                    && self.query_prefix == "task: code retrieval | query: " =>
            {
                Ok(&[128, 256, 512, 768])
            }
            NOMIC_V1_5_PROFILE
                if self.output_transform == OutputTransform::MeanPoolLayerNorm
                    && self.add_special_tokens
                    && self.token_type_ids_name.as_deref() == Some("token_type_ids")
                    && self.base_dimensions == 768
                    && self.max_length == 2_048
                    && self.document_window_overlap_tokens == 0
                    && self.document_aggregation == DocumentAggregation::FirstWindow
                    && self.document_prefix == "search_document: "
                    && self.query_prefix == "search_query: " =>
            {
                Ok(&[128, 256, 512, 768])
            }
            ARCTIC_M_V2_PROFILE
                if self.output_transform == OutputTransform::FinalPooled
                    && self.add_special_tokens
                    && self.token_type_ids_name.is_none()
                    && self.base_dimensions == 768
                    && self.max_length == 2_048
                    && self.document_window_overlap_tokens == 0
                    && self.document_aggregation == DocumentAggregation::FirstWindow
                    && self.document_prefix.is_empty()
                    && self.query_prefix == "query: " =>
            {
                Ok(&[128, 256, 512, 768])
            }
            ARCTIC_M_V2_WINDOWED_PROFILE
                if self.output_transform == OutputTransform::FinalPooled
                    && self.add_special_tokens
                    && self.token_type_ids_name.is_none()
                    && self.base_dimensions == 768
                    && self.max_length == 256
                    && self.document_window_overlap_tokens == 32
                    && self.document_aggregation
                        == DocumentAggregation::TokenWeightedMeanWindows
                    && self.document_prefix.is_empty()
                    && self.query_prefix == "query: " =>
            {
                Ok(&[128, 256, 512, 768])
            }
            STATIC_RETRIEVAL_MRL_EN_V1_PROFILE
                if self.output_transform == OutputTransform::FinalPooled
                    && !self.add_special_tokens
                    && self.token_type_ids_name.is_none()
                    && self.base_dimensions == 1_024
                    && self.max_length == 2_048
                    && self.document_window_overlap_tokens == 0
                    && self.document_aggregation
                        == DocumentAggregation::TokenWeightedMeanWindows
                    && self.document_prefix.is_empty()
                    && self.query_prefix.is_empty() =>
            {
                Ok(&[32, 64, 128, 256, 512, 1_024])
            }
            _ => {
                Err(LocalOnnxError::Invalid(
                    "bundle profile, output transform, inputs, and retrieval prompts do not match a supported pinned model contract".into(),
                ))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactDigest {
    path: String,
    sha256: String,
}

fn verify_artifacts(bundle_dir: &Path, manifest: &BundleManifest) -> Result<(), LocalOnnxError> {
    let mut paths = BTreeSet::new();
    for artifact in &manifest.artifacts {
        if !paths.insert(artifact.path.as_str()) {
            return Err(LocalOnnxError::Invalid(format!(
                "duplicate artifact path {}",
                artifact.path
            )));
        }
        if artifact.sha256.len() != 64
            || !artifact
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(LocalOnnxError::Invalid(format!(
                "artifact {} must have a lowercase SHA-256 digest",
                artifact.path
            )));
        }
        let path = safe_artifact_path(bundle_dir, &artifact.path)?;
        let actual = digest_file(&path)?;
        if actual != artifact.sha256 {
            return Err(LocalOnnxError::Checksum {
                path: artifact.path.clone(),
                expected: artifact.sha256.clone(),
                actual,
            });
        }
    }
    for required in [&manifest.model_file, &manifest.tokenizer_file] {
        if !paths.contains(required.as_str()) {
            return Err(LocalOnnxError::Invalid(format!(
                "required artifact {required} has no checksum"
            )));
        }
    }
    Ok(())
}

fn safe_artifact_path(bundle_dir: &Path, relative: &str) -> Result<PathBuf, LocalOnnxError> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(LocalOnnxError::Invalid(format!(
            "artifact path must be a simple relative path: {relative}"
        )));
    }
    let path = bundle_dir.join(relative_path);
    let canonical = path
        .canonicalize()
        .map_err(|error| LocalOnnxError::Read(error.to_string()))?;
    if !canonical.starts_with(bundle_dir) {
        return Err(LocalOnnxError::Invalid(format!(
            "artifact escapes bundle directory: {relative}"
        )));
    }
    Ok(canonical)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, LocalOnnxError> {
    let metadata = fs::metadata(path).map_err(|error| LocalOnnxError::Read(error.to_string()))?;
    if metadata.len() > limit {
        return Err(LocalOnnxError::Invalid(format!(
            "{} exceeds {limit} bytes",
            path.display()
        )));
    }
    fs::read(path).map_err(|error| LocalOnnxError::Read(error.to_string()))
}

fn digest_file(path: &Path) -> Result<String, LocalOnnxError> {
    let file = File::open(path).map_err(|error| LocalOnnxError::Read(error.to_string()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| LocalOnnxError::Read(error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hex_digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(target_os = "macos")]
fn open_session(
    model_path: &Path,
    core_ml_cache: &Path,
    prefer_core_ml: bool,
) -> Result<(Session, LocalExecutionProvider, Option<String>), LocalOnnxError> {
    if prefer_core_ml {
        match build_core_ml_session(model_path, core_ml_cache) {
            Ok(session) => {
                return Ok((session, LocalExecutionProvider::CoreMl, None));
            }
            Err(error) => {
                let reason = error.to_string();
                return match build_cpu_session(model_path) {
                    Ok(session) => Ok((session, LocalExecutionProvider::Cpu, Some(reason))),
                    Err(cpu_error) => Err(LocalOnnxError::Runtime(format!(
                        "Core ML failed ({reason}); CPU fallback failed ({cpu_error})"
                    ))),
                };
            }
        }
    }
    build_cpu_session(model_path).map(|session| (session, LocalExecutionProvider::Cpu, None))
}

#[cfg(not(target_os = "macos"))]
fn open_session(
    model_path: &Path,
    _core_ml_cache: &Path,
    _prefer_core_ml: bool,
) -> Result<(Session, LocalExecutionProvider, Option<String>), LocalOnnxError> {
    build_cpu_session(model_path).map(|session| (session, LocalExecutionProvider::Cpu, None))
}

#[cfg(target_os = "macos")]
fn build_core_ml_session(
    model_path: &Path,
    core_ml_cache: &Path,
) -> Result<Session, LocalOnnxError> {
    use ort::ep::{
        CoreML,
        coreml::{ModelFormat, SpecializationStrategy},
    };

    fs::create_dir_all(core_ml_cache).map_err(|error| {
        LocalOnnxError::Runtime(format!(
            "failed to create Core ML model cache {}: {error}",
            core_ml_cache.display()
        ))
    })?;
    let builder = Session::builder().map_err(|error| runtime_error(&error))?;
    let builder = builder
        .with_execution_providers([CoreML::default()
            .with_model_format(ModelFormat::NeuralNetwork)
            .with_specialization_strategy(SpecializationStrategy::FastPrediction)
            .with_model_cache_dir(core_ml_cache.to_string_lossy())
            .with_subgraphs(true)
            .build()])
        .map_err(|error| runtime_error(&error))?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| runtime_error(&error))?;
    builder
        .commit_from_file(model_path)
        .map_err(|error| runtime_error(&error))
}

fn build_cpu_session(model_path: &Path) -> Result<Session, LocalOnnxError> {
    let builder = Session::builder().map_err(|error| runtime_error(&error))?;
    let mut builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(|error| runtime_error(&error))?;
    builder
        .commit_from_file(model_path)
        .map_err(|error| runtime_error(&error))
}

fn runtime_error<T>(error: &ort::Error<T>) -> LocalOnnxError {
    LocalOnnxError::Runtime(error.to_string())
}

fn validate_graph_contract(
    session: &Session,
    manifest: &BundleManifest,
) -> Result<(), LocalOnnxError> {
    for required in [&manifest.input_ids_name, &manifest.attention_mask_name]
        .into_iter()
        .chain(manifest.token_type_ids_name.iter())
    {
        if !session
            .inputs()
            .iter()
            .any(|input| input.name() == required)
        {
            return Err(LocalOnnxError::Invalid(format!(
                "ONNX graph is missing input {required}"
            )));
        }
    }
    if !session
        .outputs()
        .iter()
        .any(|output| output.name() == manifest.output_name)
    {
        return Err(LocalOnnxError::Invalid(format!(
            "ONNX graph is missing output {}",
            manifest.output_name
        )));
    }
    Ok(())
}

fn dimensions2(first: usize, second: usize) -> Result<[i64; 2], IndexError> {
    Ok([
        i64::try_from(first)
            .map_err(|error| embedding_error("local_output_shape", error.to_string()))?,
        i64::try_from(second)
            .map_err(|error| embedding_error("local_output_shape", error.to_string()))?,
    ])
}

fn dimensions3(first: usize, second: usize, third: usize) -> Result<[i64; 3], IndexError> {
    Ok([
        i64::try_from(first)
            .map_err(|error| embedding_error("local_output_shape", error.to_string()))?,
        i64::try_from(second)
            .map_err(|error| embedding_error("local_output_shape", error.to_string()))?,
        i64::try_from(third)
            .map_err(|error| embedding_error("local_output_shape", error.to_string()))?,
    ])
}

fn postprocess_embeddings(
    transform: OutputTransform,
    shape: &[i64],
    values: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    sequence_length: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>, IndexError> {
    match transform {
        OutputTransform::FinalPooled => {
            let expected_shape = dimensions2(batch_size, dimensions)?;
            if shape != expected_shape {
                return Err(embedding_error(
                    "local_output_shape",
                    format!("final pooled output has shape {shape:?}; expected {expected_shape:?}"),
                ));
            }
            Ok(values
                .chunks_exact(dimensions)
                .map(<[f32]>::to_vec)
                .collect())
        }
        OutputTransform::MeanPoolLayerNorm => {
            let expected_shape = dimensions3(batch_size, sequence_length, dimensions)?;
            if shape != expected_shape {
                return Err(embedding_error(
                    "local_output_shape",
                    format!("token output has shape {shape:?}; expected {expected_shape:?}"),
                ));
            }
            mean_pool_layer_norm(
                values,
                attention_mask,
                batch_size,
                sequence_length,
                dimensions,
            )
        }
    }
}

fn mean_pool_layer_norm(
    token_embeddings: &[f32],
    attention_mask: &[i64],
    batch_size: usize,
    sequence_length: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>, IndexError> {
    let mut output = Vec::with_capacity(batch_size);
    for batch in 0..batch_size {
        let mut pooled = vec![0.0_f32; dimensions];
        let mut included = 0_usize;
        for token in 0..sequence_length {
            let mask_index = batch * sequence_length + token;
            if attention_mask.get(mask_index).copied() != Some(1) {
                continue;
            }
            let start = mask_index * dimensions;
            let values = token_embeddings
                .get(start..start + dimensions)
                .ok_or_else(|| {
                    embedding_error(
                        "local_output_shape",
                        "token output is shorter than its shape",
                    )
                })?;
            for (target, value) in pooled.iter_mut().zip(values) {
                *target += *value;
            }
            included += 1;
        }
        if included == 0 {
            return Err(embedding_error(
                "local_pooling",
                "attention mask contains no source tokens",
            ));
        }
        let divisor = f32::from(u16::try_from(included).map_err(|error| {
            embedding_error(
                "local_pooling",
                format!("token count is too large: {error}"),
            )
        })?);
        for value in &mut pooled {
            *value /= divisor;
        }
        layer_normalize(&mut pooled)?;
        output.push(pooled);
    }
    Ok(output)
}

fn layer_normalize(values: &mut [f32]) -> Result<(), IndexError> {
    let count = f32::from(u16::try_from(values.len()).map_err(|error| {
        embedding_error(
            "local_layer_normalize",
            format!("embedding width is too large: {error}"),
        )
    })?);
    if !count.is_finite() || count <= 0.0 {
        return Err(embedding_error(
            "local_layer_normalize",
            "cannot layer-normalize an empty embedding",
        ));
    }
    let mean = values.iter().sum::<f32>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f32>()
        / count;
    let denominator = (variance + 1e-5).sqrt();
    if !denominator.is_finite() || denominator <= f32::EPSILON {
        return Err(embedding_error(
            "local_layer_normalize",
            "model returned a non-finite embedding",
        ));
    }
    for value in values {
        *value = (*value - mean) / denominator;
    }
    Ok(())
}

fn normalize(values: &mut [f32]) -> Result<(), IndexError> {
    let magnitude = values
        .iter()
        .map(|value| *value * *value)
        .sum::<f32>()
        .sqrt();
    if !magnitude.is_finite() || magnitude <= f32::EPSILON {
        return Err(embedding_error(
            "local_normalize",
            "model returned a zero or non-finite embedding",
        ));
    }
    for value in values {
        *value /= magnitude;
    }
    Ok(())
}

fn embedding_error(code: &str, message: impl Into<String>) -> IndexError {
    IndexError::new(IndexErrorKind::Embedding, code, message)
}

#[cfg(test)]
mod tests {
    use tokenizers::Token;

    use super::*;

    fn manifest() -> BundleManifest {
        BundleManifest {
            schema_version: 1,
            model_id: "google/embeddinggemma-300m".into(),
            model_revision: "upstream-revision".into(),
            model_file: "model.onnx".into(),
            tokenizer_file: "tokenizer.json".into(),
            artifacts: vec![ArtifactDigest {
                path: "model.onnx".into(),
                sha256: "0".repeat(64),
            }],
            input_ids_name: "input_ids".into(),
            attention_mask_name: "attention_mask".into(),
            token_type_ids_name: None,
            output_name: "sentence_embedding".into(),
            output_transform: OutputTransform::FinalPooled,
            add_special_tokens: true,
            core_ml_compatible: true,
            max_length: 2_048,
            document_window_overlap_tokens: 0,
            document_aggregation: DocumentAggregation::FirstWindow,
            base_dimensions: 768,
            embedding_profile: EMBEDDINGGEMMA_PROFILE.into(),
            document_prefix: "title: none | text: ".into(),
            query_prefix: "task: code retrieval | query: ".into(),
        }
    }

    #[test]
    fn accepts_only_supported_mrl_dimensions_and_pinned_profiles() {
        let mut value = manifest();
        assert!(value.validate(256).is_ok());
        assert!(value.validate(300).is_err());
        value.query_prefix = "query: ".into();
        assert!(value.validate(256).is_err());

        value.embedding_profile = NOMIC_V1_5_PROFILE.into();
        value.output_transform = OutputTransform::MeanPoolLayerNorm;
        value.token_type_ids_name = Some("token_type_ids".into());
        value.document_prefix = "search_document: ".into();
        value.query_prefix = "search_query: ".into();
        assert!(value.validate(256).is_ok());

        value.embedding_profile = ARCTIC_M_V2_PROFILE.into();
        value.output_transform = OutputTransform::FinalPooled;
        value.token_type_ids_name = None;
        value.document_prefix.clear();
        value.query_prefix = "query: ".into();
        assert!(value.validate(256).is_ok());

        value.embedding_profile = ARCTIC_M_V2_WINDOWED_PROFILE.into();
        value.max_length = 256;
        value.document_window_overlap_tokens = 32;
        value.document_aggregation = DocumentAggregation::TokenWeightedMeanWindows;
        assert!(value.validate(256).is_ok());
        value.document_window_overlap_tokens = 256;
        assert!(value.validate(256).is_err());

        value.embedding_profile = STATIC_RETRIEVAL_MRL_EN_V1_PROFILE.into();
        value.add_special_tokens = false;
        value.base_dimensions = 1_024;
        value.max_length = 2_048;
        value.document_window_overlap_tokens = 0;
        value.document_aggregation = DocumentAggregation::TokenWeightedMeanWindows;
        value.query_prefix.clear();
        assert!(value.validate(256).is_ok());
        assert!(value.validate(1_024).is_ok());
        assert!(value.validate(768).is_err());
        value.add_special_tokens = true;
        assert!(value.validate(256).is_err());
    }

    #[test]
    fn mrl_truncation_is_renormalized() {
        let mut values = vec![3.0, 4.0];
        normalize(&mut values).unwrap();
        assert!((values[0] - 0.6).abs() < f32::EPSILON);
        assert!((values[1] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn mean_pooling_excludes_padding_and_applies_layer_normalization() {
        let tokens = [
            1.0, 2.0, 3.0, // first source token
            3.0, 4.0, 5.0, // second source token
            100.0, 100.0, 100.0, // padding must be ignored
        ];
        let pooled = mean_pool_layer_norm(&tokens, &[1, 1, 0], 1, 3, 3).unwrap();
        assert_eq!(pooled.len(), 1);
        assert!(pooled[0][0] < 0.0);
        assert!(pooled[0][1].abs() < f32::EPSILON);
        assert!(pooled[0][2] > 0.0);
        assert!(pooled[0].iter().sum::<f32>().abs() < 1e-5);
    }

    #[test]
    fn document_window_aggregation_preserves_owners_and_token_weights() {
        let tokens = |count: usize| {
            (0..count)
                .map(|ordinal| Token::new(u32::try_from(ordinal).unwrap(), "x".into(), (0, 1)))
                .collect()
        };
        let overflow = Encoding::from_tokens(tokens(1), 0);
        let mut root = Encoding::from_tokens(tokens(2), 0);
        root.set_overflowing(vec![overflow]);
        let encodings = [root];

        let (first, first_owners) = document_windows(&encodings, DocumentAggregation::FirstWindow);
        assert_eq!(first.len(), 1);
        assert_eq!(first_owners, [0]);

        let (windows, owners) =
            document_windows(&encodings, DocumentAggregation::TokenWeightedMeanWindows);
        assert_eq!(windows.len(), 2);
        assert_eq!(owners, [0, 0]);
        let aggregated = aggregate_document_windows(
            &windows,
            &owners,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            1,
            2,
        )
        .unwrap();
        assert!((aggregated[0][0] - (2.0 / 3.0)).abs() < 1e-6);
        assert!((aggregated[0][1] - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn indexing_qos_boundary_returns_values_and_errors() {
        assert_eq!(run_at_indexing_qos(|| Ok(42)).unwrap(), 42);
        let error =
            run_at_indexing_qos::<()>(|| Err(embedding_error("qos_test", "sentinel failure")))
                .unwrap_err();
        assert!(error.to_string().contains("sentinel failure"));
    }

    #[test]
    fn rejects_parent_and_absolute_artifact_paths_before_io() {
        let root = Path::new("/tmp/bundle");
        assert!(safe_artifact_path(root, "../model.onnx").is_err());
        assert!(safe_artifact_path(root, "/tmp/model.onnx").is_err());
    }
}
