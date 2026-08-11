use async_trait::async_trait;

use crate::{Candidate, Capabilities, DocId, Query, Retriever, SearchError, SearchOpts, Signals};

/// Deterministic random-scoring retriever used only as the Phase 0 harness floor.
///
/// This backend proves integration and evaluation wiring before production
/// lexical or dense retrieval exists. It is deliberately not relevance-aware.
pub struct DeterministicPhase0Retriever {
    document_ids: Vec<DocId>,
}

impl DeterministicPhase0Retriever {
    /// Creates the Phase 0 retriever from a stable document universe.
    #[must_use]
    pub fn new(mut document_ids: Vec<DocId>) -> Self {
        document_ids.sort();
        document_ids.dedup();
        Self { document_ids }
    }
}

#[async_trait]
impl Retriever for DeterministicPhase0Retriever {
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError> {
        query.validate()?;
        options.validate()?;
        let mut candidates = self
            .document_ids
            .iter()
            .map(|doc_id| Candidate {
                score: stable_score(&query.text, doc_id.as_str()),
                doc_id: doc_id.clone(),
                signals: Signals::default(),
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.doc_id.cmp(&right.doc_id))
        });
        candidates.truncate(options.top_k.get().min(options.candidate_limit.get()));
        Ok(candidates)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }
}

fn stable_score(query: &str, doc_id: &str) -> f32 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in query
        .bytes()
        .chain(std::iter::once(0xff))
        .chain(doc_id.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    let fraction = u32::try_from(hash >> 41).unwrap_or_default();
    f32::from_bits(0x3f80_0000 | fraction) - 1.0
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use futures::executor::block_on;

    use super::*;

    #[test]
    fn results_are_stable_and_deduplicated() {
        let retriever = DeterministicPhase0Retriever::new(vec![
            DocId::new("second").unwrap(),
            DocId::new("first").unwrap(),
            DocId::new("first").unwrap(),
        ]);
        let query = Query::new("manifest").unwrap();
        let options = SearchOpts {
            top_k: NonZeroUsize::new(2).unwrap(),
            candidate_limit: NonZeroUsize::new(2).unwrap(),
            enable_late_interaction: false,
        };

        let first = block_on(retriever.search(&query, &options)).unwrap();
        let second = block_on(retriever.search(&query, &options)).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 2);
    }
}
