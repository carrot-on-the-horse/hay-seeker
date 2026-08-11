use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cast_core::LanguageId;
use cast_index::NormalizedPath;

use crate::{DocId, IndexManifest, SearchError};

const RRF_K: f32 = 60.0;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Backend-neutral searchable chunk stored by local and remote indexes.
pub struct SearchDocument {
    /// Stable chunk identity.
    pub doc_id: DocId,
    /// Repository-relative source path.
    pub path: NormalizedPath,
    /// Resolved source language.
    pub language: LanguageId,
    /// Exact chunk text used for lexical retrieval and embedding.
    pub text: String,
    /// Optional dense vector matching the active index manifest dimensions.
    pub embedding: Option<Vec<f32>>,
}

impl SearchDocument {
    /// Validates content and an optional vector against an index manifest.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for blank content, a dimension
    /// mismatch, or a non-finite vector component.
    pub fn validate(&self, manifest: &IndexManifest) -> Result<(), SearchError> {
        if self.text.trim().is_empty() {
            return Err(SearchError::InvalidConfig(format!(
                "document {} has blank text",
                self.doc_id
            )));
        }
        if let Some(embedding) = &self.embedding {
            if embedding.len() != manifest.mrl_dim {
                return Err(SearchError::InvalidConfig(format!(
                    "document {} has {} embedding dimensions; expected {}",
                    self.doc_id,
                    embedding.len(),
                    manifest.mrl_dim
                )));
            }
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(SearchError::InvalidConfig(format!(
                    "document {} has a non-finite embedding",
                    self.doc_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Validated user query.
pub struct Query {
    /// Original nonblank query text.
    pub text: String,
}

impl Query {
    /// Creates a nonblank query.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for a blank query.
    pub fn new(text: impl Into<String>) -> Result<Self, SearchError> {
        let text = text.into();
        let query = Self { text };
        query.validate()?;
        Ok(query)
    }

    /// Revalidates a query received through deserialization or a struct literal.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] for blank text.
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.text.trim().is_empty() {
            return Err(SearchError::InvalidConfig("query must not be blank".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Backend-neutral limits and optional cascade stages.
pub struct SearchOpts {
    /// Number of final candidates requested by the caller.
    pub top_k: NonZeroUsize,
    /// Maximum candidates retained between cascade stages.
    pub candidate_limit: NonZeroUsize,
    /// Whether to run late interaction when the backend supports it.
    pub enable_late_interaction: bool,
}

impl Default for SearchOpts {
    fn default() -> Self {
        Self {
            top_k: NonZeroUsize::new(10).unwrap_or(NonZeroUsize::MIN),
            candidate_limit: NonZeroUsize::new(50).unwrap_or(NonZeroUsize::MIN),
            enable_late_interaction: false,
        }
    }
}

impl SearchOpts {
    /// Validates limits shared by all retrieval backends.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when the final result count is
    /// greater than the cascade candidate limit.
    pub fn validate(&self) -> Result<(), SearchError> {
        if self.top_k > self.candidate_limit {
            return Err(SearchError::InvalidConfig(
                "top_k must not exceed candidate_limit".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[allow(clippy::struct_excessive_bools)]
/// Retrieval stages implemented by one backend.
pub struct Capabilities {
    /// Lexical candidate generation is available.
    pub lexical: bool,
    /// Dense candidate generation is available.
    pub dense: bool,
    /// Exact or quantized dense rescoring is available.
    pub quantized_rescore: bool,
    /// Token-level late interaction is available.
    pub late_interaction: bool,
    /// Learned sparse retrieval is available.
    pub learned_sparse: bool,
    /// Fixed-dimensional encoding is available.
    pub fde: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
/// Optional per-stage evidence retained for debugging and evaluation.
pub struct Signals {
    /// Lexical stage score, when executed.
    pub lexical: Option<f32>,
    /// Dense stage score, when executed.
    pub dense: Option<f32>,
    /// Late-interaction stage score, when executed.
    pub late: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// One ranked search result.
pub struct Candidate {
    /// Stable document identity.
    pub doc_id: DocId,
    /// Final score used for ordering.
    pub score: f32,
    /// Available stage-level scores.
    pub signals: Signals,
}

/// The only query interface exposed by interchangeable retrieval backends.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Runs the backend's declared cascade and returns deterministically ordered
    /// candidates.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError`] for manifest, query, backend, or invariant
    /// failures.
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError>;

    /// Declares which stages this backend can execute.
    fn capabilities(&self) -> Capabilities;
}

/// Fuses lexical and dense ranked lists with reciprocal-rank fusion (`k=60`).
///
/// When only one stage produces results, its native score is preserved. This
/// keeps lexical-only and dense-only backends useful while giving hybrid
/// backends identical deterministic fusion and document-ID tie breaking.
#[must_use]
pub fn fuse_ranked_results(
    lexical: &[(DocId, f32)],
    dense: &[(DocId, f32)],
    limit: usize,
) -> Vec<Candidate> {
    if lexical.is_empty() {
        return single_stage(dense, limit, false);
    }
    if dense.is_empty() {
        return single_stage(lexical, limit, true);
    }

    let mut fused = BTreeMap::<DocId, Candidate>::new();
    for (rank, (doc_id, score)) in lexical.iter().enumerate() {
        let candidate = fused.entry(doc_id.clone()).or_insert_with(|| Candidate {
            doc_id: doc_id.clone(),
            score: 0.0,
            signals: Signals::default(),
        });
        candidate.score += 1.0 / (RRF_K + rank_as_f32(rank) + 1.0);
        candidate.signals.lexical = Some(*score);
    }
    for (rank, (doc_id, score)) in dense.iter().enumerate() {
        let candidate = fused.entry(doc_id.clone()).or_insert_with(|| Candidate {
            doc_id: doc_id.clone(),
            score: 0.0,
            signals: Signals::default(),
        });
        candidate.score += 1.0 / (RRF_K + rank_as_f32(rank) + 1.0);
        candidate.signals.dense = Some(*score);
    }
    let mut candidates = fused.into_values().collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    candidates.truncate(limit);
    candidates
}

fn single_stage(ranked: &[(DocId, f32)], limit: usize, lexical: bool) -> Vec<Candidate> {
    ranked
        .iter()
        .take(limit)
        .map(|(doc_id, score)| Candidate {
            doc_id: doc_id.clone(),
            score: *score,
            signals: if lexical {
                Signals {
                    lexical: Some(*score),
                    ..Signals::default()
                }
            } else {
                Signals {
                    dense: Some(*score),
                    ..Signals::default()
                }
            },
        })
        .collect()
}

fn rank_as_f32(rank: usize) -> f32 {
    u16::try_from(rank).map_or(f32::from(u16::MAX), f32::from)
}

/// Retriever decorator that enforces exact manifest compatibility per query.
///
/// Construct this at the backend boundary after loading the index manifest.
/// Keeping the check inside [`Retriever::search`] prevents a caller from
/// accidentally bypassing startup-only validation.
pub struct ManifestCheckedRetriever<R> {
    inner: R,
    index_manifest: IndexManifest,
    runtime_manifest: IndexManifest,
}

impl<R> ManifestCheckedRetriever<R> {
    /// Wraps a backend with its persisted and active manifests.
    #[must_use]
    pub const fn new(
        inner: R,
        index_manifest: IndexManifest,
        runtime_manifest: IndexManifest,
    ) -> Self {
        Self {
            inner,
            index_manifest,
            runtime_manifest,
        }
    }

    /// Consumes the decorator and returns the backend.
    #[must_use]
    pub fn into_inner(self) -> R {
        self.inner
    }
}

#[async_trait]
impl<R: Retriever> Retriever for ManifestCheckedRetriever<R> {
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError> {
        query.validate()?;
        options.validate()?;
        self.index_manifest
            .validate_runtime(&self.runtime_manifest)?;
        self.inner.search(query, options).await
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_query_is_rejected() {
        assert!(Query::new("  \n").is_err());
    }

    #[test]
    fn final_limit_cannot_exceed_candidate_limit() {
        let options = SearchOpts {
            top_k: NonZeroUsize::new(51).unwrap(),
            candidate_limit: NonZeroUsize::new(50).unwrap(),
            enable_late_interaction: false,
        };
        assert!(options.validate().is_err());
    }

    #[test]
    fn rrf_is_deterministic_and_preserves_stage_signals() {
        let lexical = vec![
            (DocId::new("lexical").unwrap(), 4.0),
            (DocId::new("both").unwrap(), 3.0),
        ];
        let dense = vec![
            (DocId::new("dense").unwrap(), 0.9),
            (DocId::new("both").unwrap(), 0.8),
        ];

        let results = fuse_ranked_results(&lexical, &dense, 3);
        assert_eq!(results[0].doc_id.as_str(), "both");
        assert_eq!(results[0].signals.lexical, Some(3.0));
        assert_eq!(results[0].signals.dense, Some(0.8));
    }
}
