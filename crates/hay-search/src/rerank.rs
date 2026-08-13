//! Final cascade stage: rescoring retrieved candidates against the query.
//!
//! A retriever ranks a passage without ever comparing it to the query directly —
//! BM25 counts term overlap, dense retrieval compares two independently produced
//! vectors. A cross-encoder reads the query and the passage together, which is
//! more accurate and far too slow to run over a corpus. It therefore only ever
//! sees candidates a retriever already selected, which is what makes the
//! candidate limit the thing that bounds reranking cost.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;

use cast_index::{RerankRequest, Reranker};

use crate::error::SearchError;
use crate::retriever::Candidate;
use crate::{DocId, Query};

/// Rescores `candidates` against `query` and returns the best `top_k`.
///
/// `passages` supplies the text for every candidate; callers already load it to
/// render results, so reranking adds no extra storage read. A candidate with no
/// passage is an error rather than a silent drop: it means the retriever and the
/// document store disagree about what exists, and quietly discarding it would
/// turn that inconsistency into a permanently missing search result.
///
/// Ordering is deterministic. Scores are provider-scaled and comparable only
/// within one response, so ties break on document identity rather than on the
/// order the retriever happened to produce.
///
/// # Errors
///
/// Returns [`SearchError`] when a candidate has no passage, when the reranker
/// returns a score count that does not match the passages it was given, or when
/// the reranker itself fails.
pub async fn rerank_candidates(
    reranker: &dyn Reranker,
    query: &Query,
    candidates: Vec<Candidate>,
    passages: &BTreeMap<DocId, String>,
    top_k: NonZeroUsize,
) -> Result<Vec<Candidate>, SearchError> {
    if candidates.is_empty() {
        return Ok(candidates);
    }
    let texts = candidates
        .iter()
        .map(|candidate| {
            passages
                .get(&candidate.doc_id)
                .map(String::as_str)
                .ok_or_else(|| {
                    SearchError::Retriever(format!(
                        "no passage text for reranking candidate {}",
                        candidate.doc_id
                    ))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;

    let scored = reranker
        .rerank(RerankRequest {
            query: query.text.as_str(),
            passages: &texts,
        })
        .await
        .map_err(|error| SearchError::Retriever(error.to_string()))?;

    if scored.scores.len() != candidates.len() {
        return Err(SearchError::Retriever(format!(
            "reranker returned {} scores for {} candidates",
            scored.scores.len(),
            candidates.len()
        )));
    }
    if let Some(position) = scored.scores.iter().position(|score| !score.is_finite()) {
        return Err(SearchError::Retriever(format!(
            "reranker returned a non-finite score at position {position}"
        )));
    }

    let mut ordered = candidates
        .into_iter()
        .zip(scored.scores)
        .map(|(mut candidate, score)| {
            candidate.score = score;
            candidate.signals.late = Some(score);
            candidate
        })
        .collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    ordered.truncate(top_k.get());
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use cast_index::{BoxFuture, IndexError, IndexErrorKind, RerankIdentity, RerankScores};

    use super::*;
    use crate::retriever::Signals;

    struct FixedScores {
        identity: RerankIdentity,
        scores: Vec<f32>,
    }

    impl FixedScores {
        fn new(scores: Vec<f32>) -> Self {
            Self {
                identity: RerankIdentity {
                    provider: "test".into(),
                    model: "fixed".into(),
                    revision: "v1".into(),
                },
                scores,
            }
        }
    }

    impl Reranker for FixedScores {
        fn identity(&self) -> &RerankIdentity {
            &self.identity
        }

        fn rerank<'a>(
            &'a self,
            _request: RerankRequest<'a>,
        ) -> BoxFuture<'a, Result<RerankScores, IndexError>> {
            Box::pin(async move {
                Ok(RerankScores {
                    identity: self.identity.clone(),
                    scores: self.scores.clone(),
                })
            })
        }
    }

    struct Failing;

    impl Reranker for Failing {
        fn identity(&self) -> &RerankIdentity {
            unreachable!("identity is not read on the failure path")
        }

        fn rerank<'a>(
            &'a self,
            _request: RerankRequest<'a>,
        ) -> BoxFuture<'a, Result<RerankScores, IndexError>> {
            Box::pin(async move {
                Err(IndexError::new(
                    IndexErrorKind::Embedding,
                    "test_failure",
                    "provider refused",
                ))
            })
        }
    }

    fn candidate(id: &str, score: f32) -> Candidate {
        Candidate {
            doc_id: DocId::new(id).unwrap(),
            score,
            signals: Signals {
                lexical: Some(score),
                dense: None,
                late: None,
            },
        }
    }

    fn passages(ids: &[&str]) -> BTreeMap<DocId, String> {
        ids.iter()
            .map(|id| (DocId::new(*id).unwrap(), format!("body of {id}")))
            .collect()
    }

    fn top(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap()
    }

    /// The whole point of the stage: an order the retriever got wrong is fixed.
    #[tokio::test]
    async fn rescoring_reorders_and_records_the_stage_signal() {
        let candidates = vec![
            candidate("a", 0.9),
            candidate("b", 0.5),
            candidate("c", 0.1),
        ];
        let reranker = FixedScores::new(vec![-2.0, 5.0, 1.0]);

        let ranked = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a", "b", "c"]),
            top(3),
        )
        .await
        .unwrap();

        assert_eq!(
            ranked.iter().map(|c| c.doc_id.as_str()).collect::<Vec<_>>(),
            ["b", "c", "a"]
        );
        assert!(
            (ranked[0].score - 5.0).abs() < f32::EPSILON,
            "the stage score replaces the retriever's: {}",
            ranked[0].score
        );
        assert!(
            ranked[0]
                .signals
                .late
                .is_some_and(|late| (late - 5.0).abs() < f32::EPSILON)
        );
        assert!(
            ranked[0]
                .signals
                .lexical
                .is_some_and(|lexical| (lexical - 0.5).abs() < f32::EPSILON),
            "the retriever's own signal is preserved"
        );
    }

    #[tokio::test]
    async fn only_the_requested_number_survives() {
        let candidates = vec![
            candidate("a", 0.9),
            candidate("b", 0.5),
            candidate("c", 0.1),
        ];
        let reranker = FixedScores::new(vec![1.0, 3.0, 2.0]);

        let ranked = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a", "b", "c"]),
            top(2),
        )
        .await
        .unwrap();

        assert_eq!(
            ranked.iter().map(|c| c.doc_id.as_str()).collect::<Vec<_>>(),
            ["b", "c"]
        );
    }

    /// Equal scores must not leave ordering to the retriever's accident.
    #[tokio::test]
    async fn equal_scores_break_ties_on_document_identity() {
        let candidates = vec![
            candidate("z", 0.9),
            candidate("m", 0.5),
            candidate("a", 0.1),
        ];
        let reranker = FixedScores::new(vec![1.0, 1.0, 1.0]);

        let ranked = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["z", "m", "a"]),
            top(3),
        )
        .await
        .unwrap();

        assert_eq!(
            ranked.iter().map(|c| c.doc_id.as_str()).collect::<Vec<_>>(),
            ["a", "m", "z"]
        );
    }

    /// A retriever and store that disagree is a bug to surface, not to hide by
    /// dropping a result that would then never be findable.
    #[tokio::test]
    async fn a_candidate_without_passage_text_fails_closed() {
        let candidates = vec![candidate("a", 0.9), candidate("missing", 0.5)];
        let reranker = FixedScores::new(vec![1.0, 2.0]);

        let error = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a"]),
            top(2),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("no passage text"), "{error}");
    }

    #[tokio::test]
    async fn a_score_count_mismatch_is_refused() {
        let candidates = vec![candidate("a", 0.9), candidate("b", 0.5)];
        let reranker = FixedScores::new(vec![1.0]);

        let error = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a", "b"]),
            top(2),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("1 scores for 2"), "{error}");
    }

    #[tokio::test]
    async fn a_non_finite_score_cannot_order_results() {
        let candidates = vec![candidate("a", 0.9), candidate("b", 0.5)];
        let reranker = FixedScores::new(vec![1.0, f32::NAN]);

        let error = rerank_candidates(
            &reranker,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a", "b"]),
            top(2),
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("non-finite"), "{error}");
    }

    #[tokio::test]
    async fn no_candidates_needs_no_provider_call() {
        let ranked = rerank_candidates(
            &Failing,
            &Query::new("q").unwrap(),
            Vec::new(),
            &BTreeMap::new(),
            top(5),
        )
        .await
        .unwrap();
        assert!(ranked.is_empty());
    }

    #[tokio::test]
    async fn a_provider_failure_is_reported_rather_than_ignored() {
        let candidates = vec![candidate("a", 0.9)];
        let error = rerank_candidates(
            &Failing,
            &Query::new("q").unwrap(),
            candidates,
            &passages(&["a"]),
            top(1),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("provider refused"), "{error}");
    }
}
