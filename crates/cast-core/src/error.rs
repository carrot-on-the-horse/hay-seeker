use thiserror::Error;

#[derive(Debug, Error)]
/// Failure reported by a pluggable source sizer.
pub enum SizeError {
    /// Sizer-specific failure message.
    #[error("sizer failed: {0}")]
    Message(String),
}

#[derive(Debug, Error)]
/// Failure while resolving, parsing, sizing, or validating source chunks.
pub enum ChunkError {
    /// Chunk configuration is internally inconsistent.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Input exceeds the configured source byte limit.
    #[error("input is {actual} bytes; configured maximum is {maximum} bytes")]
    InputTooLarge {
        /// Observed input size in bytes.
        actual: usize,
        /// Configured maximum input size in bytes.
        maximum: usize,
    },

    /// No parser or fallback is available for the requested language.
    #[error("unsupported language: {0}")]
    UnsupportedLanguage(String),

    /// Available signals identify more than one possible language.
    #[error("language is ambiguous: {0}")]
    AmbiguousLanguage(String),

    /// The parser returned no syntax tree.
    #[error("parser could not produce a syntax tree")]
    ParseFailed,

    /// The parser exceeded its configured deadline.
    #[error("parsing exceeded the configured {milliseconds} ms deadline")]
    ParseTimeout {
        /// Configured deadline in milliseconds.
        milliseconds: u64,
    },

    /// Strict parsing rejected recovery or missing nodes.
    #[error("syntax tree contains {errors} error nodes and {missing} missing nodes")]
    ParseHasErrors {
        /// Number of error nodes in the syntax tree.
        errors: usize,
        /// Number of missing nodes in the syntax tree.
        missing: usize,
    },

    /// The strict limit cannot fit even one UTF-8 scalar value.
    #[error("strict size {maximum} cannot contain one complete UTF-8 character")]
    UnsplittableUnit {
        /// Configured maximum size.
        maximum: usize,
    },

    /// Requested behavior is intentionally outside the implemented contract.
    #[error("unsupported in the first draft: {0}")]
    UnsupportedFeature(&'static str),

    /// Produced chunks violate a core range or ordering invariant.
    #[error("chunk invariant failed: {0}")]
    Invariant(String),

    /// The configured sizer failed.
    #[error(transparent)]
    Size(#[from] SizeError),
}
