use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
/// One field that differs between an index and the active runtime.
pub struct ManifestMismatch {
    /// Stable manifest field name.
    pub field: &'static str,
    /// Debug representation stored by the index.
    pub index_value: String,
    /// Debug representation requested by the runtime.
    pub runtime_value: String,
}

#[derive(Debug, Error)]
/// Failure returned by backend-neutral search contracts.
pub enum SearchError {
    /// The persisted index is incompatible with the active runtime.
    #[error("reindex required; manifest mismatch: {mismatches:?}")]
    ReindexRequired {
        /// Every incompatible field, in manifest declaration order.
        mismatches: Vec<ManifestMismatch>,
    },

    /// A caller supplied invalid options or input.
    #[error("invalid search configuration: {0}")]
    InvalidConfig(String),

    /// Reading the document corpus failed.
    #[error("corpus error: {0}")]
    Corpus(String),

    /// Splitting a document failed.
    #[error("chunker error: {0}")]
    Chunker(String),

    /// A retrieval backend failed.
    #[error("retriever error: {0}")]
    Retriever(String),
}
