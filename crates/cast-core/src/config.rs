use std::num::{NonZeroU64, NonZeroUsize};

use serde::{Deserialize, Serialize};

use crate::ChunkError;

/// Configuration for one chunking operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ChunkConfig {
    /// Maximum chunk size measured by the selected [`crate::Sizer`].
    pub max_size: NonZeroUsize,
    /// Context copied from adjacent chunks.
    pub overlap: Overlap,
    /// Policy used when an indivisible syntax node exceeds `max_size`.
    pub limit_policy: LimitPolicy,
    /// Policy used when parsing fails or produces recovery nodes.
    pub parse_policy: ParsePolicy,
    /// Syntax-node metadata included in each chunk.
    pub include_node_kinds: NodeKindMode,
    /// Maximum accepted source-file size in bytes, or no limit.
    pub max_input_bytes: Option<NonZeroUsize>,
    /// Hard per-chunk byte ceiling, independent of the selected sizing unit.
    pub max_chunk_bytes: Option<NonZeroUsize>,
    /// Tree-sitter parse deadline in milliseconds.
    pub parse_timeout_ms: Option<NonZeroU64>,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            max_size: NonZeroUsize::new(1_500).unwrap_or(NonZeroUsize::MIN),
            overlap: Overlap::None,
            limit_policy: LimitPolicy::Strict,
            parse_policy: ParsePolicy::GenericFallback,
            include_node_kinds: NodeKindMode::AllNamed,
            max_input_bytes: NonZeroUsize::new(5 * 1024 * 1024),
            max_chunk_bytes: NonZeroUsize::new(25_000),
            parse_timeout_ms: NonZeroU64::new(60_000),
        }
    }
}

impl ChunkConfig {
    /// Validates values that depend on both configuration and input size.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkError::InvalidConfig`] for an invalid overlap or
    /// [`ChunkError::InputTooLarge`] when the input exceeds its configured cap.
    pub fn validate(&self, input_bytes: usize) -> Result<(), ChunkError> {
        match self.overlap {
            Overlap::Units(units) if units >= self.max_size.get() => {
                return Err(ChunkError::InvalidConfig(
                    "overlap units must be smaller than max_size".into(),
                ));
            }
            Overlap::Percent(percent) if percent > 50 => {
                return Err(ChunkError::InvalidConfig(
                    "overlap percent must be between 0 and 50".into(),
                ));
            }
            Overlap::None | Overlap::Units(_) | Overlap::Percent(_) => {}
        }

        if let Some(maximum) = self.max_input_bytes {
            if input_bytes > maximum.get() {
                return Err(ChunkError::InputTooLarge {
                    actual: input_bytes,
                    maximum: maximum.get(),
                });
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Amount of neighboring content copied into a chunk's context range.
pub enum Overlap {
    /// Do not include content from neighboring chunks.
    #[default]
    None,
    /// Include up to this many sizing units of context.
    Units(usize),
    /// Include this percentage of `max_size`, limited to 50 percent.
    Percent(u8),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Behavior when a syntax-preserving boundary exceeds the configured limit.
pub enum LimitPolicy {
    /// Enforce the configured size ceiling by splitting the source safely.
    #[default]
    Strict,
    /// Keep atomic syntax nodes intact even when they exceed the size target.
    PreserveAtomicNodes,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Behavior when the parser cannot produce a clean syntax tree.
pub enum ParsePolicy {
    /// Fail unless the parser returns a tree without recovery nodes.
    RequireAst,
    /// Use the recovered syntax tree and report degraded quality.
    #[default]
    Recover,
    /// Fall back to syntax-agnostic chunking when AST parsing is unusable.
    GenericFallback,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Controls which syntax-node kinds are recorded on chunks.
pub enum NodeKindMode {
    /// Do not attach syntax-node kinds.
    None,
    /// Attach kinds for top-level nodes represented by the chunk.
    #[default]
    TopLevel,
    /// Attach every named node kind represented by the chunk.
    AllNamed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_overlap() {
        let config = ChunkConfig {
            overlap: Overlap::Percent(51),
            ..ChunkConfig::default()
        };

        assert!(matches!(
            config.validate(0),
            Err(ChunkError::InvalidConfig(_))
        ));
    }

    #[test]
    fn rejects_oversized_input() {
        let config = ChunkConfig {
            max_input_bytes: NonZeroUsize::new(3),
            ..ChunkConfig::default()
        };

        assert!(matches!(
            config.validate(4),
            Err(ChunkError::InputTooLarge {
                actual: 4,
                maximum: 3
            })
        ));
    }

    #[test]
    fn defaults_preserve_proven_go_pipeline_limits() {
        let config = ChunkConfig::default();

        assert_eq!(config.max_size.get(), 1_500);
        assert_eq!(config.max_input_bytes.unwrap().get(), 5 * 1024 * 1024);
        assert_eq!(config.max_chunk_bytes.unwrap().get(), 25_000);
        assert_eq!(config.parse_timeout_ms.unwrap().get(), 60_000);
        assert_eq!(config.parse_policy, ParsePolicy::GenericFallback);
        assert_eq!(config.include_node_kinds, NodeKindMode::AllNamed);
    }
}
