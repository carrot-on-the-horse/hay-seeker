#![forbid(unsafe_code)]
#![deny(missing_docs)]
//! Backend-neutral hybrid-search contracts.
//!
//! Phase 0 contains contracts, manifest validation, corpus/chunker abstractions,
//! and no production retrieval implementation.
//!
//! # Example
//!
//! ```
//! use hay_search::{Query, SearchOpts};
//!
//! let query = Query::new("where is the manifest validated")?;
//! let options = SearchOpts::default();
//! assert_eq!(query.text, "where is the manifest validated");
//! assert_eq!(options.top_k.get(), 10);
//! # Ok::<(), hay_search::SearchError>(())
//! ```

mod analysis;
mod chunker;
mod corpus;
mod error;
mod manifest;
#[cfg(feature = "phase0")]
mod phase0;
mod rerank;
mod retriever;

pub use analysis::analyze_code_terms;
pub use chunker::{Chunker, ChunkerV1, CorpusChunk, FixedWindowConfig};
pub use corpus::{Corpus, CorpusDocument};
pub use error::{ManifestMismatch, SearchError};
pub use manifest::{FdeParams, IndexManifest, Quantization};
#[cfg(feature = "phase0")]
pub use phase0::DeterministicPhase0Retriever;
pub use rerank::rerank_candidates;
pub use retriever::{
    Candidate, Capabilities, ManifestCheckedRetriever, Query, Retriever, SearchDocument,
    SearchOpts, Signals, fuse_ranked_results,
};

/// Stable identifier returned by retrieval backends.
pub use cast_index::DocumentId as DocId;
