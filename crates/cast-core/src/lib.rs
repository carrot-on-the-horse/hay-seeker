//! Core contracts shared by CAST parser adapters and consumers.
//!
//! The crate keeps chunk configuration and output independent of a particular
//! parser, tokenizer, or index backend.
//!
//! ```
//! use cast_core::{ByteSizer, ChunkConfig, Sizer};
//!
//! let config = ChunkConfig::default();
//! config.validate(b"fn main() {}".len())?;
//! assert_eq!(ByteSizer.measure("fn main() {}")?, 12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

mod config;
mod error;
mod model;
mod sizer;

pub use config::{ChunkConfig, LimitPolicy, NodeKindMode, Overlap, ParsePolicy};
pub use error::{ChunkError, SizeError};
pub use model::{
    Chunk, ChunkOutput, ChunkQuality, ChunkStrategy, Diagnostic, DiagnosticSeverity, LanguageId,
    LanguageResolution, ResolutionMethod, SourcePoint, SourceRange,
};
pub use sizer::{ByteSizer, LineSizer, Sizer, UnicodeWordSizer};

/// Version of the serialized output contract.
pub const OUTPUT_SCHEMA_VERSION: u32 = 1;
/// Version of the chunk-boundary algorithm used by this draft.
pub const ALGORITHM_VERSION: &str = "cast-rust-0.1";
