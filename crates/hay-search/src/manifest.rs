use cast_index::{ContentHash, HashAlgorithm};
use cast_tokenizers::OPENAI_O200K_ARTIFACT_SHA256;
use serde::{Deserialize, Serialize};

use crate::{ChunkerV1, ManifestMismatch, SearchError};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Vector storage representation used by an index.
pub enum Quantization {
    /// Full-precision or phase-zero placeholder values.
    None,
    /// Signed int8 values with per-vector scale and offset.
    Int8PerVectorScaleOffset,
    /// Elasticsearch Better Binary Quantization.
    ElasticBbq,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
/// Fixed-dimensional encoding configuration.
pub enum FdeParams {
    /// FDE is not present in this index.
    Disabled,
    /// MUVERA-compatible encoding parameters.
    Muvera {
        /// Encoding algorithm version.
        version: String,
        /// Number of independent projection repetitions.
        repetitions: usize,
        /// `SimHash` bits used by each projection.
        simhash_bits: usize,
        /// Explicit deterministic seed.
        seed: u64,
    },
}

/// Exact compatibility contract carried by every index.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndexManifest {
    /// Stable encoder family identifier.
    pub model_id: String,
    /// Exact immutable model revision or artifact digest.
    pub model_revision: String,
    /// Versioned input formatting, normalization, and pooling contract.
    pub embedding_profile: String,
    /// Native encoder output dimension.
    pub embed_dim: usize,
    /// Matryoshka dimension stored or queried.
    pub mrl_dim: usize,
    /// Vector quantization representation.
    pub quantization: Quantization,
    /// Cryptographic digest of tokenizer artifacts and configuration.
    pub tokenizer_hash: ContentHash,
    /// Chunker contract version.
    pub chunker_version: String,
    /// Exact fixed-dimensional encoding parameters.
    pub fde_params: FdeParams,
    /// Persisted storage schema version.
    pub schema_version: u32,
}

impl IndexManifest {
    /// Returns the frozen lexical-only manifest for the first `DuckDB` product
    /// path.
    ///
    /// The tokenizer hash pins the `o200k_base` chunk-sizing artifact. The BM25
    /// analyzer contract is pinned independently in `model_revision`; changing
    /// either identity requires a full rebuild.
    #[must_use]
    pub fn lexical_v1() -> Self {
        Self {
            model_id: "none".into(),
            model_revision: "lexical-bm25-v2-path3-id2".into(),
            embedding_profile: "none".into(),
            embed_dim: 1,
            mrl_dim: 1,
            quantization: Quantization::None,
            tokenizer_hash: ContentHash {
                algorithm: HashAlgorithm::Sha256,
                hex_digest: OPENAI_O200K_ARTIFACT_SHA256.into(),
            },
            chunker_version: ChunkerV1::default().profile_id(),
            fde_params: FdeParams::Disabled,
            schema_version: 1,
        }
    }

    /// Validates internal manifest invariants before an index is created.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when identifiers are blank,
    /// dimensions are zero or inconsistent, or the schema version is zero.
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.model_id.trim().is_empty()
            || self.model_revision.trim().is_empty()
            || self.embedding_profile.trim().is_empty()
            || self.chunker_version.trim().is_empty()
            || self.schema_version == 0
        {
            return Err(SearchError::InvalidConfig(
                "manifest requires model, revision, profile, chunker, and schema identities".into(),
            ));
        }
        if self.embed_dim == 0 || self.mrl_dim == 0 || self.mrl_dim > self.embed_dim {
            return Err(SearchError::InvalidConfig(
                "manifest requires 0 < mrl_dim <= embed_dim".into(),
            ));
        }
        Ok(())
    }

    /// Validates index metadata against the exact runtime configuration.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::ReindexRequired`] with every mismatched field.
    /// No field is silently ignored.
    pub fn validate_runtime(&self, runtime: &Self) -> Result<(), SearchError> {
        self.validate()?;
        runtime.validate()?;
        let mut mismatches = Vec::new();
        compare(
            &mut mismatches,
            "model_id",
            &self.model_id,
            &runtime.model_id,
        );
        compare(
            &mut mismatches,
            "model_revision",
            &self.model_revision,
            &runtime.model_revision,
        );
        compare(
            &mut mismatches,
            "embedding_profile",
            &self.embedding_profile,
            &runtime.embedding_profile,
        );
        compare(
            &mut mismatches,
            "embed_dim",
            &self.embed_dim,
            &runtime.embed_dim,
        );
        compare(&mut mismatches, "mrl_dim", &self.mrl_dim, &runtime.mrl_dim);
        compare(
            &mut mismatches,
            "quantization",
            &self.quantization,
            &runtime.quantization,
        );
        compare(
            &mut mismatches,
            "tokenizer_hash",
            &self.tokenizer_hash,
            &runtime.tokenizer_hash,
        );
        compare(
            &mut mismatches,
            "chunker_version",
            &self.chunker_version,
            &runtime.chunker_version,
        );
        compare(
            &mut mismatches,
            "fde_params",
            &self.fde_params,
            &runtime.fde_params,
        );
        compare(
            &mut mismatches,
            "schema_version",
            &self.schema_version,
            &runtime.schema_version,
        );

        if mismatches.is_empty() {
            Ok(())
        } else {
            Err(SearchError::ReindexRequired { mismatches })
        }
    }
}

fn compare<T: std::fmt::Debug + PartialEq>(
    mismatches: &mut Vec<ManifestMismatch>,
    field: &'static str,
    index: &T,
    runtime: &T,
) {
    if index != runtime {
        mismatches.push(ManifestMismatch {
            field,
            index_value: format!("{index:?}"),
            runtime_value: format!("{runtime:?}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use cast_index::{ContentHash, HashAlgorithm};

    use super::*;

    fn manifest() -> IndexManifest {
        IndexManifest {
            model_id: "encoder".into(),
            model_revision: "revision".into(),
            embedding_profile: "retrieval-v1".into(),
            embed_dim: 768,
            mrl_dim: 256,
            quantization: Quantization::Int8PerVectorScaleOffset,
            tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "a".repeat(64)).unwrap(),
            chunker_version: "cast-v1".into(),
            fde_params: FdeParams::Disabled,
            schema_version: 1,
        }
    }

    #[test]
    fn exact_manifest_passes() {
        let index = manifest();
        index.validate_runtime(&index).unwrap();
    }

    #[test]
    fn lexical_manifest_pins_the_executable_product_chunker() {
        assert_eq!(
            IndexManifest::lexical_v1().chunker_version,
            ChunkerV1::default().profile_id()
        );
    }

    #[test]
    fn every_manifest_field_hard_fails_with_reindex_required() {
        let index = manifest();
        let variations = vec![
            (
                "model_id",
                IndexManifest {
                    model_id: "other".into(),
                    ..manifest()
                },
            ),
            (
                "model_revision",
                IndexManifest {
                    model_revision: "other".into(),
                    ..manifest()
                },
            ),
            (
                "embedding_profile",
                IndexManifest {
                    embedding_profile: "other".into(),
                    ..manifest()
                },
            ),
            (
                "embed_dim",
                IndexManifest {
                    embed_dim: 384,
                    ..manifest()
                },
            ),
            (
                "mrl_dim",
                IndexManifest {
                    mrl_dim: 128,
                    ..manifest()
                },
            ),
            (
                "quantization",
                IndexManifest {
                    quantization: Quantization::None,
                    ..manifest()
                },
            ),
            (
                "tokenizer_hash",
                IndexManifest {
                    tokenizer_hash: ContentHash::new(HashAlgorithm::Sha256, "b".repeat(64))
                        .unwrap(),
                    ..manifest()
                },
            ),
            (
                "chunker_version",
                IndexManifest {
                    chunker_version: "cast-v2".into(),
                    ..manifest()
                },
            ),
            (
                "fde_params",
                IndexManifest {
                    fde_params: FdeParams::Muvera {
                        version: "1".into(),
                        repetitions: 4,
                        simhash_bits: 8,
                        seed: 42,
                    },
                    ..manifest()
                },
            ),
            (
                "schema_version",
                IndexManifest {
                    schema_version: 2,
                    ..manifest()
                },
            ),
        ];

        for (field, runtime) in variations {
            let error = index.validate_runtime(&runtime).unwrap_err();
            let SearchError::ReindexRequired { mismatches } = error else {
                panic!("expected reindex-required error for {field}");
            };
            assert_eq!(mismatches.len(), 1, "unexpected mismatches for {field}");
            assert_eq!(mismatches[0].field, field);
        }
    }
}
