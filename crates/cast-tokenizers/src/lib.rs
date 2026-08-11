//! Exact, locally bundled tokenizer implementations for CAST sizing.
//!
//! ```
//! use cast_core::Sizer;
//! use cast_tokenizers::{exact_or_openai_fallback, OPENAI_O200K_SIZER_ID};
//!
//! let sizer = exact_or_openai_fallback(None);
//! assert_eq!(sizer.name(), OPENAI_O200K_SIZER_ID);
//! assert_eq!(sizer.measure("hello world")?, 2);
//! # Ok::<(), cast_core::SizeError>(())
//! ```

#![deny(missing_docs)]

use std::sync::Arc;

use cast_core::{SizeError, Sizer};

/// Friendly name of the `OpenAI` fallback encoding.
pub const OPENAI_FALLBACK_ENCODING: &str = "o200k_base";

/// SHA-256 of the merge-rank artifact embedded by `tiktoken-rs` 0.12.0.
pub const OPENAI_O200K_ARTIFACT_SHA256: &str =
    "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";

/// Complete identity persisted in CAST output and index fingerprints.
pub const OPENAI_O200K_SIZER_ID: &str = concat!(
    "openai:o200k_base:tiktoken-rs@0.12.0:sha256:",
    "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
);

/// `OpenAI` `o200k_base` BPE sizing used when the selected model's exact
/// tokenizer is unavailable.
///
/// Merge ranks are embedded in the binary, so construction and measurement
/// require no network access.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenAiBpeSizer;

impl Sizer for OpenAiBpeSizer {
    fn name(&self) -> &'static str {
        OPENAI_O200K_SIZER_ID
    }

    fn measure(&self, text: &str) -> Result<usize, SizeError> {
        Ok(tiktoken_rs::o200k_base_singleton().count_ordinary(text))
    }
}

/// Uses a model's exact tokenizer when supplied, otherwise the pinned `OpenAI`
/// fallback.
///
/// Providers should pass an exact tokenizer only when its vocabulary,
/// pre-tokenizer, special-token policy, and artifact revision match the model.
#[must_use]
pub fn exact_or_openai_fallback(exact: Option<Arc<dyn Sizer>>) -> Arc<dyn Sizer> {
    exact.unwrap_or_else(|| Arc::new(OpenAiBpeSizer))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_examples_have_stable_counts() {
        let sizer = OpenAiBpeSizer;
        assert_eq!(sizer.measure("").unwrap(), 0);
        assert_eq!(sizer.measure("hello world").unwrap(), 2);
        assert_eq!(sizer.measure("2 + 2 = 4").unwrap(), 7);
        assert_eq!(sizer.measure("お誕生日おめでとう").unwrap(), 8);
    }

    #[test]
    fn identity_pins_implementation_and_artifact() {
        assert!(OPENAI_O200K_SIZER_ID.contains("tiktoken-rs@0.12.0"));
        assert!(OPENAI_O200K_SIZER_ID.ends_with(OPENAI_O200K_ARTIFACT_SHA256));
    }

    #[test]
    fn missing_exact_tokenizer_uses_openai_fallback() {
        assert_eq!(exact_or_openai_fallback(None).name(), OPENAI_O200K_SIZER_ID);

        let exact: Arc<dyn Sizer> = Arc::new(cast_core::UnicodeWordSizer);
        assert_eq!(
            exact_or_openai_fallback(Some(Arc::clone(&exact))).name(),
            exact.name()
        );
    }
}
