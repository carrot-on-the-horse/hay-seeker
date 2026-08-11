use std::num::NonZeroUsize;
use std::sync::Arc;

use cast_core::{
    ChunkConfig, ChunkStrategy, LanguageId, LimitPolicy, NodeKindMode, Overlap, ParsePolicy, Sizer,
    SourcePoint, SourceRange,
};
use cast_index::DocumentId;
use cast_tokenizers::exact_or_openai_fallback;
use cast_tree_sitter::{LanguageRegistry, TreeSitterChunker};

use crate::{CorpusDocument, SearchError};

/// Deterministic source-to-chunk transformation.
///
/// # Example
///
/// ```
/// use hay_search::{Chunker, ChunkerV1};
///
/// let chunker = ChunkerV1::default();
/// assert_eq!(chunker.version(), "cast-v1");
/// ```
pub trait Chunker: Send {
    /// Stable version persisted in [`crate::IndexManifest`].
    fn version(&self) -> &'static str;

    /// Complete tokenizer or sizing identity persisted in index fingerprints.
    fn sizer_name(&self) -> &'static str;

    /// Splits one corpus document deterministically.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::Chunker`] when parsing, sizing, or fallback
    /// invariants fail.
    fn chunk(&mut self, document: &CorpusDocument) -> Result<Vec<CorpusChunk>, SearchError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
/// A source-backed unit passed to indexing.
pub struct CorpusChunk {
    /// Stable child identity derived from parent and ordinal.
    pub chunk_id: DocumentId,
    /// Identity of the unsplit source document.
    pub parent_doc_id: DocumentId,
    /// Zero-based order within the source document.
    pub ordinal: usize,
    /// Context text indexed for this chunk.
    pub text: String,
    /// Non-overlapping range used to prove exact source coverage.
    pub core_range: SourceRange,
    /// Potentially overlapping range represented by `text`.
    pub context_range: SourceRange,
    /// Language used to select the chunking strategy.
    pub language: LanguageId,
    /// AST, generic, or mixed strategy actually used.
    pub strategy: ChunkStrategy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
/// Byte-window settings for non-code documents.
pub struct FixedWindowConfig {
    /// Maximum context window size before UTF-8 boundary adjustment.
    pub window_bytes: NonZeroUsize,
    /// Requested context overlap between adjacent windows.
    pub overlap_bytes: usize,
}

impl Default for FixedWindowConfig {
    fn default() -> Self {
        Self {
            window_bytes: NonZeroUsize::new(6_000).unwrap_or(NonZeroUsize::MIN),
            overlap_bytes: 600,
        }
    }
}

/// CAST chunker: Tree-sitter for compiled languages and fixed windows otherwise.
pub struct ChunkerV1 {
    ast: TreeSitterChunker,
    ast_config: ChunkConfig,
    fixed: FixedWindowConfig,
}

impl ChunkerV1 {
    /// Stable algorithm-family version, excluding concrete configuration.
    pub const ALGORITHM_VERSION: &str = "cast-v1";

    /// Builds CAST v1 with AST and fixed-window fallback settings.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when overlap is not smaller than
    /// the fixed window.
    pub fn new(ast_max_size: NonZeroUsize, fixed: FixedWindowConfig) -> Result<Self, SearchError> {
        Self::with_sizer(ast_max_size, fixed, exact_or_openai_fallback(None))
    }

    /// Builds CAST with an explicit sizing implementation.
    ///
    /// # Errors
    ///
    /// Returns [`SearchError::InvalidConfig`] when overlap is not smaller than
    /// the fixed window.
    pub fn with_sizer(
        ast_max_size: NonZeroUsize,
        fixed: FixedWindowConfig,
        sizer: Arc<dyn Sizer>,
    ) -> Result<Self, SearchError> {
        if fixed.overlap_bytes >= fixed.window_bytes.get() {
            return Err(SearchError::InvalidConfig(
                "fixed-window overlap must be smaller than the window".into(),
            ));
        }
        let ast_config = ChunkConfig {
            max_size: ast_max_size,
            ..ChunkConfig::default()
        };
        Ok(Self {
            ast: TreeSitterChunker::new(sizer),
            ast_config,
            fixed,
        })
    }

    /// Returns the complete relevance identity of this chunker instance.
    ///
    /// Unlike [`Chunker::version`], this includes every AST and fixed-window
    /// setting, the tokenizer implementation and artifact, and every compiled
    /// Tree-sitter grammar. Persist this value in [`crate::IndexManifest`].
    #[must_use]
    pub fn profile_id(&self) -> String {
        format!(
            concat!(
                "{};sizer={};grammars={};",
                "ast_max={};ast_overlap={};ast_limit={};ast_parse={};",
                "ast_node_kinds={};ast_max_input_bytes={};ast_max_chunk_bytes={};",
                "ast_parse_timeout_ms={};fixed_window_bytes={};fixed_overlap_bytes={}"
            ),
            Self::ALGORITHM_VERSION,
            self.ast.sizer_name(),
            LanguageRegistry.grammar_set_id(),
            self.ast_config.max_size,
            overlap_id(self.ast_config.overlap),
            limit_policy_id(self.ast_config.limit_policy),
            parse_policy_id(self.ast_config.parse_policy),
            node_kind_mode_id(self.ast_config.include_node_kinds),
            optional_usize_id(self.ast_config.max_input_bytes),
            optional_usize_id(self.ast_config.max_chunk_bytes),
            self.ast_config
                .parse_timeout_ms
                .map_or_else(|| "none".into(), |value| value.to_string()),
            self.fixed.window_bytes,
            self.fixed.overlap_bytes,
        )
    }
}

impl Default for ChunkerV1 {
    fn default() -> Self {
        Self {
            ast: TreeSitterChunker::new(exact_or_openai_fallback(None)),
            ast_config: ChunkConfig {
                max_size: NonZeroUsize::new(1_500).unwrap_or(NonZeroUsize::MIN),
                ..ChunkConfig::default()
            },
            fixed: FixedWindowConfig::default(),
        }
    }
}

impl Chunker for ChunkerV1 {
    fn version(&self) -> &'static str {
        Self::ALGORITHM_VERSION
    }

    fn sizer_name(&self) -> &'static str {
        self.ast.sizer_name()
    }

    fn chunk(&mut self, document: &CorpusDocument) -> Result<Vec<CorpusChunk>, SearchError> {
        if self.ast.supports(&document.language.0) {
            let output = self
                .ast
                .chunk(&document.text, &document.language.0, &self.ast_config)
                .map_err(|error| SearchError::Chunker(error.to_string()))?;
            return output
                .chunks
                .into_iter()
                .map(|chunk| {
                    Ok(CorpusChunk {
                        chunk_id: chunk_id(&document.doc_id, chunk.ordinal)?,
                        parent_doc_id: document.doc_id.clone(),
                        ordinal: chunk.ordinal,
                        text: chunk.text,
                        core_range: chunk.core_range,
                        context_range: chunk.context_range,
                        language: chunk.language,
                        strategy: output.strategy,
                    })
                })
                .collect();
        }

        fixed_window(document, self.fixed)
    }
}

fn overlap_id(value: Overlap) -> String {
    match value {
        Overlap::None => "none".into(),
        Overlap::Units(units) => format!("units:{units}"),
        Overlap::Percent(percent) => format!("percent:{percent}"),
    }
}

const fn limit_policy_id(value: LimitPolicy) -> &'static str {
    match value {
        LimitPolicy::Strict => "strict",
        LimitPolicy::PreserveAtomicNodes => "preserve_atomic_nodes",
    }
}

const fn parse_policy_id(value: ParsePolicy) -> &'static str {
    match value {
        ParsePolicy::RequireAst => "require_ast",
        ParsePolicy::Recover => "recover",
        ParsePolicy::GenericFallback => "generic_fallback",
    }
}

const fn node_kind_mode_id(value: NodeKindMode) -> &'static str {
    match value {
        NodeKindMode::None => "none",
        NodeKindMode::TopLevel => "top_level",
        NodeKindMode::AllNamed => "all_named",
    }
}

fn optional_usize_id(value: Option<NonZeroUsize>) -> String {
    value.map_or_else(|| "none".into(), |value| value.to_string())
}

fn fixed_window(
    document: &CorpusDocument,
    config: FixedWindowConfig,
) -> Result<Vec<CorpusChunk>, SearchError> {
    if document.text.is_empty() {
        return Ok(Vec::new());
    }
    let window = config.window_bytes.get();
    let step = window - config.overlap_bytes;
    let mut starts = vec![0];
    loop {
        let current = starts.last().copied().unwrap_or_default();
        if current + window >= document.text.len() {
            break;
        }
        let next = floor_char_boundary(&document.text, (current + step).min(document.text.len()));
        if next <= current {
            return Err(SearchError::Chunker(
                "fixed-window chunker could not make UTF-8 progress".into(),
            ));
        }
        starts.push(next);
    }

    let lines = LineIndex::new(&document.text);
    starts
        .iter()
        .enumerate()
        .map(|(ordinal, start)| {
            let core_end = starts
                .get(ordinal + 1)
                .copied()
                .unwrap_or(document.text.len());
            let context_end = floor_char_boundary(
                &document.text,
                start.saturating_add(window).min(document.text.len()),
            );
            Ok(CorpusChunk {
                chunk_id: chunk_id(&document.doc_id, ordinal)?,
                parent_doc_id: document.doc_id.clone(),
                ordinal,
                text: document.text[*start..context_end].to_owned(),
                core_range: lines.range(*start, core_end),
                context_range: lines.range(*start, context_end),
                language: document.language.clone(),
                strategy: ChunkStrategy::Generic,
            })
        })
        .collect()
}

fn chunk_id(parent: &DocumentId, ordinal: usize) -> Result<DocumentId, SearchError> {
    DocumentId::new(format!("{parent}#{ordinal}"))
        .map_err(|error| SearchError::Chunker(error.to_string()))
}

fn floor_char_boundary(text: &str, mut byte: usize) -> usize {
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    byte
}

struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(source: &str) -> Self {
        let mut starts = vec![0];
        starts.extend(
            source
                .bytes()
                .enumerate()
                .filter_map(|(index, byte)| (byte == b'\n').then_some(index + 1)),
        );
        Self { starts }
    }

    fn point(&self, byte: usize) -> SourcePoint {
        let line_index = self.starts.partition_point(|start| *start <= byte) - 1;
        SourcePoint {
            line: line_index + 1,
            byte_column: byte - self.starts[line_index],
        }
    }

    fn range(&self, start: usize, end: usize) -> SourceRange {
        SourceRange {
            start_byte: start,
            end_byte: end,
            start: self.point(start),
            end: self.point(end),
        }
    }
}

#[cfg(test)]
mod tests {
    use cast_core::UnicodeWordSizer;
    use cast_index::NormalizedPath;
    use cast_tokenizers::OPENAI_O200K_SIZER_ID;

    use super::*;

    fn document(language: &str, text: String) -> CorpusDocument {
        CorpusDocument {
            doc_id: DocumentId::new("document").unwrap(),
            path: NormalizedPath::new("source.txt").unwrap(),
            language: LanguageId::from(language),
            text,
        }
    }

    #[test]
    fn compiled_languages_use_ast_chunking() {
        let mut chunker = ChunkerV1::default();
        let cases = [
            ("rust", "fn main() {}\n"),
            ("php", "<?php function main() {}\n"),
            ("python", "def main():\n    pass\n"),
            ("go", "package main\nfunc main() {}\n"),
        ];
        for (language, source) in cases {
            let chunks = chunker
                .chunk(&document(language, source.into()))
                .unwrap_or_else(|error| panic!("{language} failed: {error}"));
            assert!(!chunks.is_empty());
            assert!(
                chunks
                    .iter()
                    .all(|chunk| chunk.strategy == ChunkStrategy::Ast)
            );
        }
    }

    #[test]
    fn constructors_use_openai_fallback_unless_exact_sizer_is_supplied() {
        let fixed = FixedWindowConfig::default();
        let chunker = ChunkerV1::new(NonZeroUsize::new(100).unwrap(), fixed).unwrap();
        assert_eq!(chunker.sizer_name(), OPENAI_O200K_SIZER_ID);

        let exact = ChunkerV1::with_sizer(
            NonZeroUsize::new(100).unwrap(),
            fixed,
            Arc::new(UnicodeWordSizer),
        )
        .unwrap();
        assert_eq!(exact.sizer_name(), "unicode_words");
    }

    #[test]
    fn product_profile_pins_every_chunking_dependency() {
        let chunker = ChunkerV1::default();
        let profile = chunker.profile_id();

        assert!(profile.starts_with("cast-v1;sizer=openai:o200k_base:"));
        assert!(profile.contains("grammars=tree-sitter@0.26.11;"));
        assert!(profile.contains("ast_max=1500"));
        assert!(profile.contains("ast_overlap=none"));
        assert!(profile.contains("ast_limit=strict"));
        assert!(profile.contains("ast_parse=generic_fallback"));
        assert!(profile.contains("ast_node_kinds=all_named"));
        assert!(profile.contains("ast_max_input_bytes=5242880"));
        assert!(profile.contains("ast_max_chunk_bytes=25000"));
        assert!(profile.contains("ast_parse_timeout_ms=60000"));
        assert!(profile.contains("fixed_window_bytes=6000"));
        assert!(profile.ends_with("fixed_overlap_bytes=600"));
    }

    #[test]
    fn fixed_window_overlap_is_source_backed_and_utf8_safe() {
        let fixed = FixedWindowConfig {
            window_bytes: NonZeroUsize::new(12).unwrap(),
            overlap_bytes: 4,
        };
        let mut chunker = ChunkerV1::new(NonZeroUsize::new(100).unwrap(), fixed).unwrap();
        let source = "🙂abcdef🙂ghijkl🙂".to_owned();
        let chunks = chunker.chunk(&document("text", source.clone())).unwrap();

        assert!(chunks.len() > 1);
        assert_eq!(
            chunks
                .iter()
                .map(|chunk| &source[chunk.core_range.start_byte..chunk.core_range.end_byte])
                .collect::<String>(),
            source
        );
        for pair in chunks.windows(2) {
            assert!(pair[0].context_range.end_byte > pair[1].context_range.start_byte);
        }
    }
}
