use std::collections::BTreeSet;

use cast_core::{ChunkQuality, LanguageId, SourceRange};
use serde::{Deserialize, Serialize};

use crate::{
    BranchName, ContentHash, ContractError, DocumentId, NormalizedPath, RepositoryId, RevisionId,
    UnixMillis,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Stable identity of the unit used to size chunks.
pub struct SizerIdentity {
    /// Serialized sizer name.
    pub name: String,
    /// Optional implementation or vocabulary revision.
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Provider, model, and input-profile identity of an embedding vector.
pub struct EmbeddingIdentity {
    /// Stable embedding provider name.
    pub provider: String,
    /// Provider-specific model identifier.
    pub model: String,
    /// Number of scalar values in every produced vector.
    pub dimensions: usize,
    /// Versioned input formatting, normalization, and pooling contract.
    pub profile: String,
}

impl EmbeddingIdentity {
    /// Validates provider, model, profile, and dimensionality.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when an identity field is empty or dimensions
    /// are zero.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.provider.is_empty()
            || self.model.is_empty()
            || self.dimensions == 0
            || self.profile.is_empty()
        {
            return Err(ContractError::DocumentInvariant(
                "embedding identity requires provider, model, profile, and non-zero dimensions"
                    .into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Dense vector paired with the exact model contract that produced it.
pub struct EmbeddingVector {
    /// Embedding model and input-profile identity.
    pub identity: EmbeddingIdentity,
    /// Finite vector components in provider-defined order.
    pub values: Vec<f32>,
}

impl EmbeddingVector {
    /// Verifies declared dimensions and finite vector values.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when dimensions differ or a value is not
    /// finite.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.identity.validate()?;
        if self.identity.dimensions != self.values.len() {
            return Err(ContractError::DocumentInvariant(format!(
                "embedding declares {} dimensions but contains {} values",
                self.identity.dimensions,
                self.values.len()
            )));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(ContractError::DocumentInvariant(
                "embedding contains a non-finite value".into(),
            ));
        }
        Ok(())
    }
}

/// Inputs that determine whether stored chunks and vectors can be reused.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct IndexFingerprint {
    /// Schema version of stored index documents.
    pub document_schema_version: u32,
    /// Version of the chunk boundary algorithm.
    pub chunk_algorithm_version: String,
    /// Version of the complete parser grammar set.
    pub grammar_set_version: String,
    /// Sizer identity used to enforce chunk limits.
    pub sizer: SizerIdentity,
    /// Embedding contract, or `None` for lexical-only documents.
    pub embedding: Option<EmbeddingIdentity>,
}

impl IndexFingerprint {
    /// Validates all invalidation inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when a required version or sizer identity is
    /// empty, or the embedding identity is invalid.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.document_schema_version == 0
            || self.chunk_algorithm_version.is_empty()
            || self.grammar_set_version.is_empty()
            || self.sizer.name.is_empty()
        {
            return Err(ContractError::DocumentInvariant(
                "fingerprint requires schema, algorithm, grammar, and sizer identities".into(),
            ));
        }
        if let Some(embedding) = &self.embedding {
            embedding.validate()?;
        }
        Ok(())
    }
}

/// Backend-neutral, independently searchable chunk document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexDocument {
    /// Serialized document schema version.
    pub schema_version: u32,
    /// Deterministic identity of this chunk document.
    pub document_id: DocumentId,
    /// Repository that owns the source file.
    pub repository: RepositoryId,
    /// Normalized repository-relative source path.
    pub path: NormalizedPath,
    /// Language assigned by the chunk engine.
    pub language: LanguageId,
    /// Repository branches on which this exact content appears.
    pub branches: BTreeSet<BranchName>,
    /// Source revision read by the indexing run.
    pub revision: RevisionId,
    /// Hash of the complete source-file content.
    pub content_hash: ContentHash,
    /// Zero-based position among chunks from the source file.
    pub chunk_ordinal: usize,
    /// Total chunks emitted for the source file.
    pub total_chunks: usize,
    /// Non-overlapping source range owned by this chunk.
    pub core_range: SourceRange,
    /// Source range stored in `content`, including overlap.
    pub context_range: SourceRange,
    /// Sorted, unique syntax-node kinds represented by the chunk.
    pub node_kinds: Vec<String>,
    /// Searchable source content for the context range.
    pub content: String,
    /// Parser recovery and degraded-split indicators.
    pub quality: ChunkQuality,
    /// Complete invalidation identity for stored chunk and vector reuse.
    pub fingerprint: IndexFingerprint,
    /// Wall-clock time when the document was indexed.
    pub indexed_at: UnixMillis,
    /// Optional dense embedding for hybrid search.
    pub embedding: Option<EmbeddingVector>,
}

/// Canonical inputs to a deterministic document-id implementation.
#[derive(Clone, Copy, Debug)]
pub struct DocumentIdentityInput<'a> {
    /// Repository owning the source file.
    pub repository: &'a RepositoryId,
    /// Normalized repository-relative path.
    pub path: &'a NormalizedPath,
    /// Hash of the complete source file.
    pub content_hash: &'a ContentHash,
    /// Invalidation identity used to produce the chunk.
    pub fingerprint: &'a IndexFingerprint,
    /// Zero-based chunk position within the source file.
    pub chunk_ordinal: usize,
    /// Non-overlapping source range owned by the chunk.
    pub core_range: &'a SourceRange,
}

impl IndexDocument {
    /// Validates backend-independent document invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ContractError`] when ranges, ordinals, content, versions, node
    /// kinds, or embedding dimensions are inconsistent.
    pub fn validate(&self) -> Result<(), ContractError> {
        self.fingerprint.validate()?;
        if self.schema_version != crate::INDEX_DOCUMENT_SCHEMA_VERSION {
            return Err(ContractError::DocumentInvariant(format!(
                "unsupported document schema {}",
                self.schema_version
            )));
        }
        if self.total_chunks == 0 || self.chunk_ordinal >= self.total_chunks {
            return Err(ContractError::DocumentInvariant(
                "chunk ordinal must be smaller than a non-zero total".into(),
            ));
        }
        if self.core_range.start_byte > self.core_range.end_byte
            || self.context_range.start_byte > self.context_range.end_byte
        {
            return Err(ContractError::DocumentInvariant(
                "source ranges must be ordered".into(),
            ));
        }
        if self.core_range.start_byte < self.context_range.start_byte
            || self.core_range.end_byte > self.context_range.end_byte
        {
            return Err(ContractError::DocumentInvariant(
                "context range must contain the core range".into(),
            ));
        }
        if self.context_range.end_byte - self.context_range.start_byte != self.content.len() {
            return Err(ContractError::DocumentInvariant(
                "content byte length must equal context range length".into(),
            ));
        }
        if self.content.trim().is_empty() {
            return Err(ContractError::DocumentInvariant(
                "index document content must not be blank".into(),
            ));
        }
        if self.fingerprint.document_schema_version != self.schema_version {
            return Err(ContractError::DocumentInvariant(
                "fingerprint document schema does not match document".into(),
            ));
        }
        if self.node_kinds.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(ContractError::DocumentInvariant(
                "node kinds must be sorted and unique".into(),
            ));
        }
        if let Some(embedding) = &self.embedding {
            embedding.validate()?;
            if self.fingerprint.embedding.as_ref() != Some(&embedding.identity) {
                return Err(ContractError::DocumentInvariant(
                    "embedding identity does not match fingerprint".into(),
                ));
            }
        } else if self.fingerprint.embedding.is_some() {
            return Err(ContractError::DocumentInvariant(
                "fingerprint requires an embedding but document has none".into(),
            ));
        }
        Ok(())
    }
}
