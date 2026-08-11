//! Tree-sitter parsing and syntax-aware CAST chunking.
//!
//! ```
//! use std::sync::Arc;
//! use cast_core::{ByteSizer, ChunkConfig, ChunkStrategy};
//! use cast_tree_sitter::TreeSitterChunker;
//!
//! let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
//! let output = chunker.chunk("fn answer() -> u8 { 42 }", "rust", &ChunkConfig::default())?;
//! assert_eq!(output.strategy, ChunkStrategy::Ast);
//! # Ok::<(), cast_core::ChunkError>(())
//! ```

#![deny(missing_docs)]

mod algorithm;
mod language;

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cast_core::{ChunkConfig, ChunkError, ChunkOutput, Diagnostic, ParsePolicy, Sizer};
use tree_sitter::{ParseOptions, Parser};

pub use language::{GrammarVersion, LanguageRegistry, ResolvedLanguage};

/// A mutable parser/chunker intended to be owned and reused by one worker.
pub struct TreeSitterChunker {
    parser: Parser,
    registry: LanguageRegistry,
    sizer: Arc<dyn Sizer>,
}

impl TreeSitterChunker {
    /// Creates a worker-local chunker using the provided sizing strategy.
    #[must_use]
    pub fn new(sizer: Arc<dyn Sizer>) -> Self {
        Self {
            parser: Parser::new(),
            registry: LanguageRegistry,
            sizer,
        }
    }

    /// Returns whether this build contains a grammar for `language`.
    #[must_use]
    pub fn supports(&self, language: &str) -> bool {
        self.registry.supports(language)
    }

    /// Returns the complete sizing identity written into chunk and index
    /// metadata.
    #[must_use]
    pub fn sizer_name(&self) -> &'static str {
        self.sizer.name()
    }

    /// Chunks source using an explicit language id.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] for invalid configuration, unsupported language,
    /// parse-policy failures, sizing failures, or violated output invariants.
    pub fn chunk(
        &mut self,
        source: &str,
        language: &str,
        config: &ChunkConfig,
    ) -> Result<ChunkOutput, ChunkError> {
        let resolved = self
            .registry
            .resolve_explicit(language, config.parse_policy)?;
        self.chunk_resolved(source, resolved, config)
    }

    /// Detects a language from a path and chunks its UTF-8 source.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError`] for invalid configuration, unknown extensions,
    /// parse-policy failures, sizing failures, or violated output invariants.
    pub fn chunk_path(
        &mut self,
        source: &str,
        path: &Path,
        config: &ChunkConfig,
    ) -> Result<ChunkOutput, ChunkError> {
        let resolved = self.registry.resolve_path(path, config.parse_policy)?;
        self.chunk_resolved(source, resolved, config)
    }

    fn chunk_resolved(
        &mut self,
        source: &str,
        resolved: ResolvedLanguage,
        config: &ChunkConfig,
    ) -> Result<ChunkOutput, ChunkError> {
        config.validate(source.len())?;

        if !matches!(config.overlap, cast_core::Overlap::None) {
            return Err(ChunkError::UnsupportedFeature(
                "source-backed overlap is planned for milestone 3",
            ));
        }

        if resolved.id.0 == "generic" {
            let mut output =
                algorithm::chunk_generic(source, resolved.resolution, config, self.sizer.as_ref())?;
            output.diagnostics.push(Diagnostic::warning(
                "generic_fallback",
                "no compiled grammar matched; generic UTF-8 chunking was used",
            ));
            return Ok(output);
        }

        let grammar = self
            .registry
            .grammar(&resolved.id)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(resolved.id.0.clone()))?;
        self.parser
            .set_language(&grammar)
            .map_err(|_| ChunkError::UnsupportedLanguage(resolved.id.0.clone()))?;

        let parse_started = Instant::now();
        let timeout_ms = config.parse_timeout_ms.map(std::num::NonZeroU64::get);
        let timeout = timeout_ms.map(Duration::from_millis);
        let mut timed_out = false;
        let bytes = source.as_bytes();
        let tree = {
            let mut reader = |offset: usize, _| bytes.get(offset..).unwrap_or_default();
            let mut progress = |_: &tree_sitter::ParseState| {
                if timeout.is_some_and(|deadline| parse_started.elapsed() >= deadline) {
                    timed_out = true;
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            };
            let options = ParseOptions::new().progress_callback(&mut progress);
            self.parser
                .parse_with_options(&mut reader, None, Some(options))
        };
        let tree = match tree {
            Some(tree) => tree,
            None if timed_out => {
                return Err(timeout_ms.map_or(ChunkError::ParseFailed, |milliseconds| {
                    ChunkError::ParseTimeout { milliseconds }
                }));
            }
            None => return Err(ChunkError::ParseFailed),
        };
        let root = tree.root_node();
        let (error_nodes, missing_nodes) = algorithm::count_parse_issues(root);
        let recovered = error_nodes > 0 || missing_nodes > 0;

        if recovered && matches!(config.parse_policy, ParsePolicy::RequireAst) {
            return Err(ChunkError::ParseHasErrors {
                errors: error_nodes,
                missing: missing_nodes,
            });
        }

        let mut diagnostics = Vec::new();
        if recovered {
            diagnostics.push(Diagnostic::warning(
                "recovered_parse",
                format!(
                    "Tree-sitter recovered with {error_nodes} error nodes and {missing_nodes} missing nodes"
                ),
            ));
        }

        let mut output = algorithm::chunk_tree(
            source,
            root,
            resolved.resolution,
            config,
            self.sizer.as_ref(),
            recovered,
        )?;
        output.diagnostics.splice(0..0, diagnostics);
        Ok(output)
    }
}

#[cfg(all(test, feature = "popular-languages"))]
mod grammar_tests {
    use std::num::NonZeroUsize;

    use cast_core::{ByteSizer, ChunkStrategy, ParsePolicy};

    use super::*;

    #[test]
    fn every_default_grammar_parses_representative_source_as_ast() {
        let samples = [
            ("bash", "#!/usr/bin/env bash\nprintf '%s\\n' \"hello\"\n"),
            ("c", "int add(int a, int b) { return a + b; }\n"),
            (
                "cpp",
                "class Counter { public: int value() const { return 1; } };\n",
            ),
            ("csharp", "class Counter { int Value() { return 1; } }\n"),
            ("go", "package main\nfunc main() { println(\"hello\") }\n"),
            (
                "java",
                "class Main { public static void main(String[] args) {} }\n",
            ),
            (
                "javascript",
                "export function add(a, b) { return a + b; }\n",
            ),
            ("php", "<?php\nfunction add($a, $b) { return $a + $b; }\n"),
            ("python", "def add(a, b):\n    return a + b\n"),
            ("ruby", "def add(a, b)\n  a + b\nend\n"),
            ("rust", "fn add(a: i32, b: i32) -> i32 { a + b }\n"),
            (
                "typescript",
                "export function add(a: number, b: number): number { return a + b; }\n",
            ),
            ("tsx", "export const App = () => <main>Hello</main>;\n"),
        ];
        let config = ChunkConfig {
            max_size: NonZeroUsize::new(10_000).unwrap_or(NonZeroUsize::MIN),
            parse_policy: ParsePolicy::RequireAst,
            ..ChunkConfig::default()
        };
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));

        for (language, source) in samples {
            let output = chunker
                .chunk(source, language, &config)
                .unwrap_or_else(|error| panic!("{language} sample failed: {error}"));
            assert_eq!(output.strategy, ChunkStrategy::Ast, "language: {language}");
            assert_eq!(output.language.language.0, language);
            assert!(!output.chunks.is_empty(), "language: {language}");
        }
    }
}
