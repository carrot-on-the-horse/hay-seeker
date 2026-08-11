use std::num::NonZeroU64;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Debug, Eq, Error, PartialEq)]
/// Violation of a backend-neutral indexing data contract.
pub enum ContractError {
    /// A required string field is empty.
    #[error("{field} must not be empty")]
    Empty {
        /// Name of the invalid field.
        field: &'static str,
    },

    /// A field contains forbidden syntax or values.
    #[error("{field} contains an invalid value: {value}")]
    Invalid {
        /// Name of the invalid field.
        field: &'static str,
        /// Rejected value.
        value: String,
    },

    /// A hexadecimal digest has the wrong serialized length.
    #[error("{field} must contain {expected} hexadecimal characters")]
    InvalidHexLength {
        /// Name of the invalid digest field.
        field: &'static str,
        /// Required number of hexadecimal characters.
        expected: usize,
    },

    /// An index document or write batch violates its invariant.
    #[error("document invariant failed: {0}")]
    DocumentInvariant(String),

    /// A source descriptor and its loaded content disagree.
    #[error("source-file invariant failed: {0}")]
    SourceInvariant(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Stable category for failures crossing adapter boundaries.
pub enum IndexErrorKind {
    /// Invalid or unsupported configuration.
    Configuration,
    /// Repository discovery or file-read failure.
    Source,
    /// Language resolution, parsing, or chunking failure.
    Chunking,
    /// Embedding provider or model failure.
    Embedding,
    /// Index backend read or write failure.
    Storage,
    /// A bounded operation exceeded its deadline.
    Timeout,
    /// Cooperative cancellation was requested.
    Cancelled,
    /// An internal or cross-adapter invariant failed.
    Invariant,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Backend-neutral guidance for retrying an [`IndexError`].
pub enum RetryAdvice {
    /// Retrying the same operation is not expected to succeed.
    #[default]
    Never,
    /// The operation may be retried immediately.
    Immediate,
    /// The operation may be retried after the given delay.
    AfterMillis(NonZeroU64),
}

/// Error exchanged across index adapters without exposing provider-specific
/// error types.
#[derive(Clone, Debug, Eq, Error, PartialEq, Serialize, Deserialize)]
#[error("{kind:?} [{code}]: {message}")]
pub struct IndexError {
    /// Stable failure category.
    pub kind: IndexErrorKind,
    /// Stable machine-readable error code.
    pub code: String,
    /// Human-readable failure context safe to expose to callers.
    pub message: String,
    /// Recommended retry behavior.
    pub retry: RetryAdvice,
}

impl IndexError {
    /// Creates a non-retryable index error.
    #[must_use]
    pub fn new(kind: IndexErrorKind, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            kind,
            code: code.into(),
            message: message.into(),
            retry: RetryAdvice::Never,
        }
    }

    /// Replaces the retry advice attached to this error.
    #[must_use]
    pub const fn with_retry(mut self, retry: RetryAdvice) -> Self {
        self.retry = retry;
        self
    }
}
