use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
/// Stable language identifier such as `rust` or `python`.
pub struct LanguageId(pub String);

impl LanguageId {
    /// Creates a language identifier from its canonical name.
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }
}

impl From<&str> for LanguageId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Signal used to resolve a source language.
pub enum ResolutionMethod {
    /// The caller supplied an explicit language.
    Explicit,
    /// The source path extension selected the language.
    Extension,
    /// No grammar matched and generic chunking was selected.
    GenericFallback,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Resolved language and the signal used to select it.
pub struct LanguageResolution {
    /// Canonical selected language.
    pub language: LanguageId,
    /// Resolution signal that selected `language`.
    pub method: ResolutionMethod,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Human-readable position in UTF-8 source text.
pub struct SourcePoint {
    /// One-based line number.
    pub line: usize,
    /// Zero-based UTF-8 byte column.
    pub byte_column: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Half-open byte range with corresponding line and column positions.
pub struct SourceRange {
    /// Inclusive UTF-8 byte offset.
    pub start_byte: usize,
    /// Exclusive byte offset.
    pub end_byte: usize,
    /// Inclusive start position.
    pub start: SourcePoint,
    /// Exclusive end position.
    pub end: SourcePoint,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Boundary strategy used to produce an output.
pub enum ChunkStrategy {
    /// Every boundary was derived from the syntax tree.
    Ast,
    /// Every boundary was produced by syntax-agnostic splitting.
    Generic,
    /// AST boundaries and degraded splits were both used.
    Mixed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
/// Per-chunk indicators of parser recovery or degraded splitting.
pub struct ChunkQuality {
    /// The parser used recovery nodes while producing the chunk.
    pub recovered_parse: bool,
    /// At least one syntax boundary was split to satisfy a hard limit.
    pub degraded_split: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// One independently indexable source fragment.
pub struct Chunk {
    /// Zero-based position in the complete chunk output.
    pub ordinal: usize,
    /// Context text stored and searched for this chunk.
    pub text: String,
    /// Non-overlapping source range owned by this chunk.
    pub core_range: SourceRange,
    /// Source range represented by `text`, including overlap.
    pub context_range: SourceRange,
    /// Size of `text` reported by the configured sizer.
    pub measured_size: usize,
    /// Language used to parse or classify the source.
    pub language: LanguageId,
    /// Sorted syntax-node kinds represented by the chunk.
    pub node_kinds: Vec<String>,
    /// Degradation signals for downstream ranking or diagnostics.
    pub quality: ChunkQuality,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Importance assigned to a chunking diagnostic.
pub enum DiagnosticSeverity {
    /// Informational diagnostic that does not imply degradation.
    Info,
    /// Recoverable condition that may reduce chunk quality.
    Warning,
    /// Operation-level failure diagnostic.
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Machine-readable observation emitted during chunking.
pub struct Diagnostic {
    /// Stable diagnostic code.
    pub code: String,
    /// Diagnostic importance.
    pub severity: DiagnosticSeverity,
    /// Human-readable description.
    pub message: String,
    /// Optional source range associated with the diagnostic.
    pub range: Option<SourceRange>,
}

impl Diagnostic {
    /// Creates a warning diagnostic without an associated source range.
    #[must_use]
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: DiagnosticSeverity::Warning,
            message: message.into(),
            range: None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
/// Complete versioned result of one chunking operation.
pub struct ChunkOutput {
    /// Serialized output schema version.
    pub schema_version: u32,
    /// Boundary algorithm version used to create the chunks.
    pub algorithm_version: String,
    /// Ordered, independently indexable chunks.
    pub chunks: Vec<Chunk>,
    /// Resolved language and resolution method.
    pub language: LanguageResolution,
    /// Boundary strategy used across the output.
    pub strategy: ChunkStrategy,
    /// Stable name of the sizer used for `measured_size`.
    pub sizer: String,
    /// Non-fatal diagnostics collected during chunking.
    pub diagnostics: Vec<Diagnostic>,
}
