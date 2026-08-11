use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use cast_index::{
    BoxFuture, Embedder, EmbeddingIdentity, EmbeddingInput, EmbeddingVector, IndexError,
    IndexErrorKind,
};
use half::f16;
use safetensors::{SafeTensors, tensor::Dtype};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokenizers::Tokenizer;

const BUNDLE_MANIFEST: &str = "static-bundle.json";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_MODEL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TOKENIZER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const BUNDLE_SCHEMA_VERSION: u32 = 1;
const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";
const CONFIG_FILE: &str = "config.json";
const POTION_CODE_16M_V2_REVISION: &str = "e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b";

/// Exact inference profile for the code-trained Potion static model.
pub const POTION_CODE_16M_V2_PROFILE: &str =
    "potion-code-16m-v2-nospecial-drop-unk-max16384-mean-l2-v1";

/// Configuration for a checksum-pinned, local-only static embedding bundle.
#[derive(Clone, Debug)]
pub struct LocalStaticConfig {
    bundle_dir: PathBuf,
    max_batch_size: usize,
}

impl LocalStaticConfig {
    /// Creates a static embedding configuration with bounded batches.
    #[must_use]
    pub fn new(bundle_dir: impl Into<PathBuf>) -> Self {
        Self {
            bundle_dir: bundle_dir.into(),
            max_batch_size: 1_024,
        }
    }

    /// Caps one tokenizer and pooling batch.
    #[must_use]
    pub const fn with_max_batch_size(mut self, max_batch_size: usize) -> Self {
        self.max_batch_size = max_batch_size;
        self
    }
}

/// Failure while validating or opening a local static embedding bundle.
#[derive(Debug, Error)]
pub enum LocalStaticError {
    /// The bundle manifest or one of its artifacts could not be read.
    #[error("failed to read local static embedding bundle: {0}")]
    Read(String),
    /// The manifest is malformed or violates the pinned model contract.
    #[error("invalid local static embedding bundle: {0}")]
    Invalid(String),
    /// An artifact digest does not match its pinned value.
    #[error(
        "local static embedding artifact checksum mismatch for {path}: expected {expected}, got {actual}"
    )]
    Checksum {
        /// Relative artifact path.
        path: String,
        /// SHA-256 declared by the bundle.
        expected: String,
        /// SHA-256 computed from the local file.
        actual: String,
    },
    /// The tokenizer or static tensor could not initialize.
    #[error("failed to initialize local static embedding model: {0}")]
    Runtime(String),
}

/// Local-only static embedder for a checksum-pinned code retrieval model.
///
/// This adapter has no model download or network path. It implements the
/// model's published no-special-token, unknown-token removal, mean-pooling,
/// and L2-normalization inference contract directly over the pinned table.
pub struct LocalStaticEmbedder {
    identity: EmbeddingIdentity,
    model_revision: String,
    tokenizer: Tokenizer,
    embeddings: Vec<f32>,
    vocabulary_size: usize,
    unknown_token_id: u32,
    max_tokens: usize,
    max_batch_size: usize,
}

impl LocalStaticEmbedder {
    /// Verifies every artifact before loading the static embedding table.
    ///
    /// # Errors
    ///
    /// Returns [`LocalStaticError`] for malformed metadata, unsafe artifact
    /// paths, checksum differences, invalid batch sizes, or model load errors.
    pub fn new(config: LocalStaticConfig) -> Result<Self, LocalStaticError> {
        let LocalStaticConfig {
            bundle_dir,
            max_batch_size,
        } = config;
        if !(1..=4_096).contains(&max_batch_size) {
            return Err(LocalStaticError::Invalid(
                "local static max batch size must be between 1 and 4096".into(),
            ));
        }
        let bundle_dir = bundle_dir
            .canonicalize()
            .map_err(|error| LocalStaticError::Read(error.to_string()))?;
        let manifest_path = bundle_dir.join(BUNDLE_MANIFEST);
        let manifest_bytes = read_bounded(&manifest_path, MAX_MANIFEST_BYTES)?;
        let manifest: StaticBundleManifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| LocalStaticError::Invalid(error.to_string()))?;
        manifest.validate()?;
        let artifacts = verify_and_read_artifacts(&bundle_dir, &manifest)?;

        let model_config: ModelConfig = serde_json::from_slice(&artifacts.config)
            .map_err(|error| LocalStaticError::Invalid(error.to_string()))?;
        if !model_config.normalize || model_config.embedding_dtype != "float16" {
            return Err(LocalStaticError::Invalid(format!(
                "unsupported static model configuration: normalize={}, embedding_dtype={}",
                model_config.normalize, model_config.embedding_dtype
            )));
        }
        let tokenizer = Tokenizer::from_bytes(&artifacts.tokenizer)
            .map_err(|error| LocalStaticError::Runtime(error.to_string()))?;
        if tokenizer.token_to_id("[UNK]") != Some(manifest.unknown_token_id) {
            return Err(LocalStaticError::Invalid(
                "tokenizer [UNK] id does not match the pinned bundle contract".into(),
            ));
        }
        let (embeddings, vocabulary_size) =
            load_embedding_table(&artifacts.model, manifest.dimensions)?;

        let manifest_hash = format!("{:x}", Sha256::digest(&manifest_bytes));
        Ok(Self {
            identity: EmbeddingIdentity {
                provider: "local-static".into(),
                model: manifest.model_id,
                dimensions: manifest.dimensions,
                profile: manifest.embedding_profile,
            },
            model_revision: format!("{};bundle-sha256:{manifest_hash}", manifest.model_revision),
            tokenizer,
            embeddings,
            vocabulary_size,
            unknown_token_id: manifest.unknown_token_id,
            max_tokens: manifest.max_tokens,
            max_batch_size,
        })
    }

    /// Immutable model revision plus the bundle-manifest checksum.
    #[must_use]
    pub fn model_revision(&self) -> &str {
        &self.model_revision
    }

    fn encode(&self, texts: &[String]) -> Result<Vec<EmbeddingVector>, IndexError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let dimensions = self.identity.dimensions;
        let mut output = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.max_batch_size) {
            let encodings = self
                .tokenizer
                .encode_batch(batch.to_vec(), false)
                .map_err(|error| embedding_error("local_static_tokenize", error.to_string()))?;
            for encoding in encodings {
                let mut values = vec![0.0_f32; dimensions];
                let mut token_count = 0_usize;
                for &token_id in encoding
                    .get_ids()
                    .iter()
                    .filter(|&&token_id| token_id != self.unknown_token_id)
                    .take(self.max_tokens)
                {
                    let token_index = usize::try_from(token_id).map_err(|_| {
                        embedding_error("local_static_token", "token id does not fit usize")
                    })?;
                    if token_index >= self.vocabulary_size {
                        return Err(embedding_error(
                            "local_static_token",
                            format!("token id {token_id} exceeds the embedding vocabulary"),
                        ));
                    }
                    let start = token_index * dimensions;
                    for (value, component) in values
                        .iter_mut()
                        .zip(&self.embeddings[start..start + dimensions])
                    {
                        *value += component;
                    }
                    token_count += 1;
                }
                if token_count == 0 {
                    return Err(embedding_error(
                        "local_static_empty_tokens",
                        "text contains no known tokens after tokenization",
                    ));
                }
                let bounded_token_count = u16::try_from(token_count).map_err(|_| {
                    embedding_error(
                        "local_static_token_count",
                        "known-token count exceeds the pinned 16384-token ceiling",
                    )
                })?;
                let denominator = f32::from(bounded_token_count);
                for value in &mut values {
                    *value /= denominator;
                }
                normalize_l2(&mut values)?;
                let vector = EmbeddingVector {
                    identity: self.identity.clone(),
                    values,
                };
                vector
                    .validate()
                    .map_err(|error| embedding_error("local_static_output", error.to_string()))?;
                output.push(vector);
            }
        }
        if output.len() != texts.len() {
            return Err(embedding_error(
                "local_static_output_count",
                "static model output count does not match input count",
            ));
        }
        Ok(output)
    }
}

impl Embedder for LocalStaticEmbedder {
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
        Box::pin(async move { self.encode(&texts) })
    }

    fn embed_query<'a>(
        &'a self,
        text: &'a str,
    ) -> BoxFuture<'a, Result<EmbeddingVector, IndexError>> {
        Box::pin(async move {
            let mut embeddings = self.encode(&[text.to_owned()])?;
            embeddings.pop().ok_or_else(|| {
                embedding_error(
                    "local_static_output",
                    "static model returned no query embedding",
                )
            })
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct StaticBundleManifest {
    schema_version: u32,
    model_id: String,
    model_revision: String,
    dimensions: usize,
    embedding_profile: String,
    unknown_token_id: u32,
    max_tokens: usize,
    artifacts: Vec<ArtifactDigest>,
}

impl StaticBundleManifest {
    fn validate(&self) -> Result<(), LocalStaticError> {
        if self.schema_version != BUNDLE_SCHEMA_VERSION {
            return Err(LocalStaticError::Invalid(format!(
                "unsupported schema version {}",
                self.schema_version
            )));
        }
        if self.model_id != "minishlab/potion-code-16M-v2"
            || self.model_revision != POTION_CODE_16M_V2_REVISION
            || self.dimensions != 256
            || self.embedding_profile != POTION_CODE_16M_V2_PROFILE
            || self.unknown_token_id != 1
            || self.max_tokens != 16_384
        {
            return Err(LocalStaticError::Invalid(
                "bundle does not match the pinned Potion code retrieval contract".into(),
            ));
        }
        if self.artifacts.len() != 3 {
            return Err(LocalStaticError::Invalid(
                "static bundle must pin exactly model.safetensors, tokenizer.json, and config.json"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactDigest {
    path: String,
    sha256: String,
}

struct StaticArtifacts {
    model: Vec<u8>,
    tokenizer: Vec<u8>,
    config: Vec<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelConfig {
    normalize: bool,
    embedding_dtype: String,
}

fn verify_and_read_artifacts(
    bundle_dir: &Path,
    manifest: &StaticBundleManifest,
) -> Result<StaticArtifacts, LocalStaticError> {
    let mut paths = BTreeSet::new();
    let mut files = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if !paths.insert(artifact.path.as_str()) {
            return Err(LocalStaticError::Invalid(format!(
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
            return Err(LocalStaticError::Invalid(format!(
                "artifact {} must have a lowercase SHA-256 digest",
                artifact.path
            )));
        }
        let path = safe_artifact_path(bundle_dir, &artifact.path)?;
        let limit = match artifact.path.as_str() {
            MODEL_FILE => MAX_MODEL_BYTES,
            TOKENIZER_FILE => MAX_TOKENIZER_BYTES,
            CONFIG_FILE => MAX_CONFIG_BYTES,
            _ => {
                return Err(LocalStaticError::Invalid(format!(
                    "unexpected static artifact path {}",
                    artifact.path
                )));
            }
        };
        let bytes = read_bounded(&path, limit)?;
        let actual = format!("{:x}", Sha256::digest(&bytes));
        if actual != artifact.sha256 {
            return Err(LocalStaticError::Checksum {
                path: artifact.path.clone(),
                expected: artifact.sha256.clone(),
                actual,
            });
        }
        files.insert(artifact.path.as_str(), bytes);
    }
    let required = [MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE]
        .into_iter()
        .collect::<BTreeSet<_>>();
    if paths != required {
        return Err(LocalStaticError::Invalid(
            "static bundle artifact paths do not match the required local layout".into(),
        ));
    }
    let model = files.remove(MODEL_FILE).ok_or_else(|| {
        LocalStaticError::Invalid("static bundle is missing model.safetensors".into())
    })?;
    let tokenizer = files.remove(TOKENIZER_FILE).ok_or_else(|| {
        LocalStaticError::Invalid("static bundle is missing tokenizer.json".into())
    })?;
    let config = files
        .remove(CONFIG_FILE)
        .ok_or_else(|| LocalStaticError::Invalid("static bundle is missing config.json".into()))?;
    Ok(StaticArtifacts {
        model,
        tokenizer,
        config,
    })
}

fn safe_artifact_path(bundle_dir: &Path, relative: &str) -> Result<PathBuf, LocalStaticError> {
    let relative_path = Path::new(relative);
    if relative.is_empty()
        || relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(LocalStaticError::Invalid(format!(
            "artifact path must be a simple relative path: {relative}"
        )));
    }
    let canonical = bundle_dir
        .join(relative_path)
        .canonicalize()
        .map_err(|error| LocalStaticError::Read(error.to_string()))?;
    if !canonical.starts_with(bundle_dir) {
        return Err(LocalStaticError::Invalid(format!(
            "artifact escapes bundle directory: {relative}"
        )));
    }
    Ok(canonical)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>, LocalStaticError> {
    let metadata = fs::metadata(path).map_err(|error| LocalStaticError::Read(error.to_string()))?;
    if metadata.len() > limit {
        return Err(LocalStaticError::Invalid(format!(
            "{} exceeds {limit} bytes",
            path.display()
        )));
    }
    fs::read(path).map_err(|error| LocalStaticError::Read(error.to_string()))
}

fn load_embedding_table(
    bytes: &[u8],
    dimensions: usize,
) -> Result<(Vec<f32>, usize), LocalStaticError> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|error| LocalStaticError::Runtime(error.to_string()))?;
    let tensor = tensors
        .tensor("embeddings")
        .map_err(|error| LocalStaticError::Runtime(error.to_string()))?;
    let shape = tensor.shape();
    if shape.len() != 2 || shape[1] != dimensions || shape[0] == 0 {
        return Err(LocalStaticError::Invalid(format!(
            "embeddings tensor shape must be [vocabulary, {dimensions}], got {shape:?}"
        )));
    }
    if tensor.dtype() != Dtype::F16 {
        return Err(LocalStaticError::Invalid(format!(
            "embeddings tensor must use F16, got {:?}",
            tensor.dtype()
        )));
    }
    let data = tensor.data();
    if data.len() % 2 != 0 {
        return Err(LocalStaticError::Invalid(
            "F16 embeddings tensor has an odd byte length".into(),
        ));
    }
    let values = data
        .chunks_exact(2)
        .map(|pair| f16::from_le_bytes([pair[0], pair[1]]).to_f32())
        .collect::<Vec<_>>();
    let expected = shape[0].checked_mul(dimensions).ok_or_else(|| {
        LocalStaticError::Invalid("embeddings tensor shape overflows usize".into())
    })?;
    if values.len() != expected {
        return Err(LocalStaticError::Invalid(format!(
            "embeddings tensor contains {} values; expected {expected}",
            values.len()
        )));
    }
    Ok((values, shape[0]))
}

fn normalize_l2(values: &mut [f32]) -> Result<(), IndexError> {
    let squared_norm = values.iter().map(|value| value * value).sum::<f32>();
    if !squared_norm.is_finite() || squared_norm <= f32::EPSILON {
        return Err(embedding_error(
            "local_static_output",
            "static model returned a zero or non-finite embedding",
        ));
    }
    let norm = squared_norm.sqrt();
    for value in values {
        *value /= norm;
    }
    Ok(())
}

fn embedding_error(code: &str, message: impl Into<String>) -> IndexError {
    IndexError::new(IndexErrorKind::Embedding, code, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> StaticBundleManifest {
        StaticBundleManifest {
            schema_version: 1,
            model_id: "minishlab/potion-code-16M-v2".into(),
            model_revision: POTION_CODE_16M_V2_REVISION.into(),
            dimensions: 256,
            embedding_profile: POTION_CODE_16M_V2_PROFILE.into(),
            unknown_token_id: 1,
            max_tokens: 16_384,
            artifacts: [MODEL_FILE, TOKENIZER_FILE, CONFIG_FILE]
                .into_iter()
                .map(|path| ArtifactDigest {
                    path: path.into(),
                    sha256: "0".repeat(64),
                })
                .collect(),
        }
    }

    #[test]
    fn accepts_only_the_pinned_code_model_contract() {
        let mut value = manifest();
        assert!(value.validate().is_ok());
        value.model_revision = "moving-main".into();
        assert!(value.validate().is_err());
        value.model_revision = POTION_CODE_16M_V2_REVISION.into();
        value.dimensions = 768;
        assert!(value.validate().is_err());
        value.dimensions = 256;
        value.embedding_profile = "generic-static".into();
        assert!(value.validate().is_err());
    }

    #[test]
    fn rejects_unsafe_and_incomplete_artifact_sets() {
        let root = Path::new("/tmp/static-bundle");
        assert!(safe_artifact_path(root, "../model.safetensors").is_err());
        assert!(safe_artifact_path(root, "/tmp/model.safetensors").is_err());

        let mut value = manifest();
        value.artifacts.pop();
        assert!(value.validate().is_err());
    }
}
