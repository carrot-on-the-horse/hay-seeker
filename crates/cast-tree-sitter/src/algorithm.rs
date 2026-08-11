use std::collections::{BTreeSet, HashMap};
use std::ops::Range;

use cast_core::{
    Chunk, ChunkConfig, ChunkError, ChunkOutput, ChunkQuality, ChunkStrategy, Diagnostic,
    LanguageResolution, LimitPolicy, NodeKindMode, Sizer, SourcePoint, SourceRange,
};
use tree_sitter::Node;

const MAX_RECURSION_DEPTH: usize = 512;

#[derive(Debug)]
struct Candidate {
    range: Range<usize>,
    node_kinds: Vec<String>,
    degraded: bool,
}

#[derive(Clone, Copy)]
struct Segment<'tree> {
    node: Node<'tree>,
    start: usize,
    end: usize,
}

pub(crate) fn chunk_tree(
    source: &str,
    root: Node<'_>,
    language: LanguageResolution,
    config: &ChunkConfig,
    sizer: &dyn Sizer,
    recovered: bool,
) -> Result<ChunkOutput, ChunkError> {
    if source.is_empty() {
        return Ok(empty_output(language, ChunkStrategy::Ast, sizer));
    }

    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();
    let mut measurements = HashMap::new();
    split_node(
        source,
        root,
        0..source.len(),
        config,
        sizer,
        &mut measurements,
        0,
        &mut candidates,
        &mut diagnostics,
    )?;

    finalize(
        source,
        candidates,
        language,
        ChunkStrategy::Ast,
        sizer,
        recovered,
        diagnostics,
        config,
    )
}

pub(crate) fn chunk_generic(
    source: &str,
    language: LanguageResolution,
    config: &ChunkConfig,
    sizer: &dyn Sizer,
) -> Result<ChunkOutput, ChunkError> {
    if source.is_empty() {
        return Ok(empty_output(language, ChunkStrategy::Generic, sizer));
    }

    let generic_kinds = vec!["generic".into()];
    let mut measurements = HashMap::new();
    let candidates = lexical_split(
        source,
        0..source.len(),
        &generic_kinds,
        config.max_size.get(),
        config.max_chunk_bytes,
        sizer,
        &mut measurements,
        false,
    )?;

    finalize(
        source,
        candidates,
        language,
        ChunkStrategy::Generic,
        sizer,
        false,
        Vec::new(),
        config,
    )
}

#[allow(clippy::too_many_arguments)]
fn split_node(
    source: &str,
    node: Node<'_>,
    assigned: Range<usize>,
    config: &ChunkConfig,
    sizer: &dyn Sizer,
    measurements: &mut HashMap<(usize, usize), usize>,
    depth: usize,
    output: &mut Vec<Candidate>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(), ChunkError> {
    let measured = measure_range(source, &assigned, sizer, measurements)?;
    if measured <= config.max_size.get() && within_byte_cap(&assigned, config) {
        output.push(Candidate {
            range: assigned,
            node_kinds: kinds_for_node(node, config.include_node_kinds),
            degraded: false,
        });
        return Ok(());
    }

    if depth >= MAX_RECURSION_DEPTH {
        diagnostics.push(Diagnostic::warning(
            "maximum_tree_depth",
            "maximum CAST recursion depth reached; using a lexical split",
        ));
        output.extend(lexical_split(
            source,
            assigned,
            &kinds_for_node(node, config.include_node_kinds),
            config.max_size.get(),
            config.max_chunk_bytes,
            sizer,
            measurements,
            true,
        )?);
        return Ok(());
    }

    let children = named_children(node);
    let makes_progress = children
        .iter()
        .any(|child| child.start_byte() > assigned.start || child.end_byte() < assigned.end);

    if children.is_empty() || !makes_progress {
        return handle_atomic(
            source,
            node,
            assigned,
            config,
            sizer,
            measurements,
            measured,
            output,
            diagnostics,
        );
    }

    let segments = partition_children(&children, &assigned);
    let mut group: Option<Candidate> = None;

    for segment in segments {
        let segment_range = segment.start..segment.end;
        let segment_size = measure_range(source, &segment_range, sizer, measurements)?;

        if segment_size > config.max_size.get() || !within_byte_cap(&segment_range, config) {
            flush_group(&mut group, output);
            split_node(
                source,
                segment.node,
                segment_range,
                config,
                sizer,
                measurements,
                depth + 1,
                output,
                diagnostics,
            )?;
            continue;
        }

        let segment_kinds = kinds_for_node(segment.node, config.include_node_kinds);
        if let Some(current) = group.as_mut() {
            let proposed = current.range.start..segment.end;
            if measure_range(source, &proposed, sizer, measurements)? <= config.max_size.get()
                && within_byte_cap(&proposed, config)
            {
                current.range.end = segment.end;
                current.node_kinds.extend(segment_kinds);
                continue;
            }
            flush_group(&mut group, output);
        }

        group = Some(Candidate {
            range: segment_range,
            node_kinds: segment_kinds,
            degraded: false,
        });
    }

    flush_group(&mut group, output);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_atomic(
    source: &str,
    node: Node<'_>,
    assigned: Range<usize>,
    config: &ChunkConfig,
    sizer: &dyn Sizer,
    measurements: &mut HashMap<(usize, usize), usize>,
    measured: usize,
    output: &mut Vec<Candidate>,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(), ChunkError> {
    let kinds = kinds_for_node(node, config.include_node_kinds);
    if !within_byte_cap(&assigned, config) {
        diagnostics.push(Diagnostic::warning(
            "hard_chunk_byte_cap",
            format!(
                "atomic node {:?} exceeded the hard {} byte chunk ceiling",
                node.kind(),
                config
                    .max_chunk_bytes
                    .map_or(assigned.len(), std::num::NonZeroUsize::get)
            ),
        ));
        output.extend(lexical_split(
            source,
            assigned,
            &kinds,
            config.max_size.get(),
            config.max_chunk_bytes,
            sizer,
            measurements,
            true,
        )?);
        return Ok(());
    }

    match config.limit_policy {
        LimitPolicy::Strict => {
            diagnostics.push(Diagnostic::warning(
                "degraded_split",
                format!(
                    "oversized atomic node {:?} required a lexical split",
                    node.kind()
                ),
            ));
            output.extend(lexical_split(
                source,
                assigned,
                &kinds,
                config.max_size.get(),
                config.max_chunk_bytes,
                sizer,
                measurements,
                true,
            )?);
        }
        LimitPolicy::PreserveAtomicNodes => {
            diagnostics.push(Diagnostic::warning(
                "oversized_atomic_node",
                format!(
                    "preserved atomic node {:?} measured {measured}, above maximum {}",
                    node.kind(),
                    config.max_size
                ),
            ));
            output.push(Candidate {
                range: assigned,
                node_kinds: kinds,
                degraded: false,
            });
        }
    }
    Ok(())
}

fn named_children(node: Node<'_>) -> Vec<Node<'_>> {
    let mut cursor = node.walk();
    node.named_children(&mut cursor)
        .filter(|child| child.end_byte() > child.start_byte())
        .collect()
}

fn partition_children<'tree>(
    children: &[Node<'tree>],
    assigned: &Range<usize>,
) -> Vec<Segment<'tree>> {
    children
        .iter()
        .enumerate()
        .map(|(index, child)| Segment {
            node: *child,
            start: if index == 0 {
                assigned.start
            } else {
                child.start_byte()
            },
            end: children
                .get(index + 1)
                .map_or(assigned.end, Node::start_byte),
        })
        .filter(|segment| segment.end > segment.start)
        .collect()
}

fn flush_group(group: &mut Option<Candidate>, output: &mut Vec<Candidate>) {
    if let Some(candidate) = group.take() {
        output.push(candidate);
    }
}

#[allow(clippy::too_many_arguments)]
fn lexical_split(
    source: &str,
    range: Range<usize>,
    node_kinds: &[String],
    maximum: usize,
    maximum_bytes: Option<std::num::NonZeroUsize>,
    sizer: &dyn Sizer,
    measurements: &mut HashMap<(usize, usize), usize>,
    degraded: bool,
) -> Result<Vec<Candidate>, ChunkError> {
    let mut chunks = Vec::new();
    let mut start = range.start;

    while start < range.end {
        let maximum_end = largest_fitting_end(
            source,
            start,
            range.end,
            maximum,
            maximum_bytes,
            sizer,
            measurements,
        )?;
        let preferred_end = preferred_boundary(source, start, maximum_end);
        let end = if preferred_end < maximum_end
            && measure_range(source, &(start..preferred_end), sizer, measurements)? > maximum
        {
            maximum_end
        } else {
            preferred_end
        };
        if end <= start {
            return Err(ChunkError::UnsplittableUnit { maximum });
        }

        chunks.push(Candidate {
            range: start..end,
            node_kinds: node_kinds.to_owned(),
            degraded,
        });
        start = end;
    }

    Ok(chunks)
}

fn largest_fitting_end(
    source: &str,
    start: usize,
    end: usize,
    maximum: usize,
    maximum_bytes: Option<std::num::NonZeroUsize>,
    sizer: &dyn Sizer,
    measurements: &mut HashMap<(usize, usize), usize>,
) -> Result<usize, ChunkError> {
    let mut hard_end = maximum_bytes.map_or(end, |cap| end.min(start.saturating_add(cap.get())));
    while hard_end > start && !source.is_char_boundary(hard_end) {
        hard_end -= 1;
    }
    if hard_end == start {
        return Err(ChunkError::UnsplittableUnit { maximum });
    }

    let mut boundaries: Vec<usize> = source[start..hard_end]
        .char_indices()
        .skip(1)
        .map(|(offset, _)| start + offset)
        .collect();
    boundaries.push(hard_end);

    let mut low = 0;
    let mut high = boundaries.len();
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate_end = boundaries[middle];
        if measure_range(source, &(start..candidate_end), sizer, measurements)? <= maximum {
            low = middle + 1;
        } else {
            high = middle;
        }
    }

    if low == 0 {
        return Err(ChunkError::UnsplittableUnit { maximum });
    }

    Ok(boundaries[low - 1])
}

fn preferred_boundary(source: &str, start: usize, maximum_end: usize) -> usize {
    let text = &source[start..maximum_end];
    let mut newline = None;
    let mut whitespace = None;
    let mut punctuation = None;

    for (offset, character) in text.char_indices() {
        let after = start + offset + character.len_utf8();
        if character == '\n' {
            newline = Some(after);
        } else if character.is_whitespace() {
            whitespace = Some(after);
        } else if character.is_ascii_punctuation() {
            punctuation = Some(after);
        }
    }

    newline
        .or(whitespace)
        .or(punctuation)
        .unwrap_or(maximum_end)
}

fn kinds_for_node(node: Node<'_>, mode: NodeKindMode) -> Vec<String> {
    match mode {
        NodeKindMode::None => Vec::new(),
        NodeKindMode::TopLevel => vec![node.kind().to_owned()],
        NodeKindMode::AllNamed => {
            let mut kinds = BTreeSet::new();
            let mut pending = vec![node];
            while let Some(current) = pending.pop() {
                if current.is_named() {
                    kinds.insert(current.kind().to_owned());
                }
                let mut cursor = current.walk();
                pending.extend(current.children(&mut cursor));
            }
            kinds.into_iter().collect()
        }
    }
}

fn measure_range(
    source: &str,
    range: &Range<usize>,
    sizer: &dyn Sizer,
    measurements: &mut HashMap<(usize, usize), usize>,
) -> Result<usize, ChunkError> {
    let key = (range.start, range.end);
    if let Some(measured) = measurements.get(&key) {
        return Ok(*measured);
    }
    let measured = sizer.measure(&source[range.clone()])?;
    measurements.insert(key, measured);
    Ok(measured)
}

fn within_byte_cap(range: &Range<usize>, config: &ChunkConfig) -> bool {
    config
        .max_chunk_bytes
        .is_none_or(|maximum| range.len() <= maximum.get())
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    source: &str,
    mut candidates: Vec<Candidate>,
    language: LanguageResolution,
    strategy: ChunkStrategy,
    sizer: &dyn Sizer,
    recovered: bool,
    diagnostics: Vec<Diagnostic>,
    config: &ChunkConfig,
) -> Result<ChunkOutput, ChunkError> {
    candidates.sort_by_key(|candidate| candidate.range.start);
    let lines = LineIndex::new(source);
    let mut chunks = Vec::with_capacity(candidates.len());

    for (ordinal, mut candidate) in candidates.into_iter().enumerate() {
        candidate.node_kinds.sort_unstable();
        candidate.node_kinds.dedup();
        let range = lines.source_range(candidate.range.start, candidate.range.end);
        let text = source[candidate.range.clone()].to_owned();
        let measured_size = sizer.measure(&text)?;

        if matches!(config.limit_policy, LimitPolicy::Strict)
            && measured_size > config.max_size.get()
        {
            return Err(ChunkError::Invariant(format!(
                "chunk {ordinal} measures {measured_size}, above maximum {}",
                config.max_size
            )));
        }
        if !within_byte_cap(&candidate.range, config) {
            return Err(ChunkError::Invariant(format!(
                "chunk {ordinal} is {} bytes, above hard ceiling {}",
                candidate.range.len(),
                config
                    .max_chunk_bytes
                    .map_or(candidate.range.len(), std::num::NonZeroUsize::get)
            )));
        }

        chunks.push(Chunk {
            ordinal,
            text,
            core_range: range,
            context_range: range,
            measured_size,
            language: language.language.clone(),
            node_kinds: candidate.node_kinds,
            quality: ChunkQuality {
                recovered_parse: recovered,
                degraded_split: candidate.degraded,
            },
        });
    }

    validate_coverage(source, &chunks)?;
    Ok(ChunkOutput {
        schema_version: cast_core::OUTPUT_SCHEMA_VERSION,
        algorithm_version: cast_core::ALGORITHM_VERSION.to_owned(),
        chunks,
        language,
        strategy,
        sizer: sizer.name().to_owned(),
        diagnostics,
    })
}

fn empty_output(
    language: LanguageResolution,
    strategy: ChunkStrategy,
    sizer: &dyn Sizer,
) -> ChunkOutput {
    ChunkOutput {
        schema_version: cast_core::OUTPUT_SCHEMA_VERSION,
        algorithm_version: cast_core::ALGORITHM_VERSION.to_owned(),
        chunks: Vec::new(),
        language,
        strategy,
        sizer: sizer.name().to_owned(),
        diagnostics: Vec::new(),
    }
}

fn validate_coverage(source: &str, chunks: &[Chunk]) -> Result<(), ChunkError> {
    let mut cursor = 0;
    for chunk in chunks {
        if chunk.core_range.start_byte != cursor {
            return Err(ChunkError::Invariant(format!(
                "expected chunk {} to start at byte {cursor}, got {}",
                chunk.ordinal, chunk.core_range.start_byte
            )));
        }
        if chunk.core_range.end_byte > source.len() {
            return Err(ChunkError::Invariant(format!(
                "chunk {} ends outside the source",
                chunk.ordinal
            )));
        }
        if chunk.text != source[chunk.context_range.start_byte..chunk.context_range.end_byte] {
            return Err(ChunkError::Invariant(format!(
                "chunk {} text does not match its context range",
                chunk.ordinal
            )));
        }
        cursor = chunk.core_range.end_byte;
    }

    if cursor != source.len() {
        return Err(ChunkError::Invariant(format!(
            "core ranges cover {cursor} of {} source bytes",
            source.len()
        )));
    }
    Ok(())
}

pub(crate) fn count_parse_issues(root: Node<'_>) -> (usize, usize) {
    let mut errors = 0;
    let mut missing = 0;
    let mut pending = vec![root];

    while let Some(node) = pending.pop() {
        errors += usize::from(node.is_error());
        missing += usize::from(node.is_missing());
        let mut cursor = node.walk();
        pending.extend(node.children(&mut cursor));
    }

    (errors, missing)
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

    fn source_range(&self, start: usize, end: usize) -> SourceRange {
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
    use std::fmt::Write as _;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use cast_core::{ByteSizer, ParsePolicy, UnicodeWordSizer};

    use crate::TreeSitterChunker;

    use super::*;

    fn config(maximum: usize) -> ChunkConfig {
        ChunkConfig {
            max_size: NonZeroUsize::new(maximum).unwrap(),
            max_input_bytes: None,
            ..ChunkConfig::default()
        }
    }

    #[test]
    fn small_rust_source_is_one_exact_chunk() {
        let source = "fn answer() -> usize { 42 }\n";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk(source, "rust", &config(100)).unwrap();

        assert_eq!(output.chunks.len(), 1);
        assert_eq!(output.chunks[0].text, source);
        assert_eq!(output.chunks[0].core_range.start_byte, 0);
        assert_eq!(output.chunks[0].core_range.end_byte, source.len());
    }

    #[test]
    fn split_chunks_reconstruct_source_and_respect_strict_limit() {
        let source =
            "// lead\nfn one() { println!(\"one\"); }\n\nfn two() { println!(\"two\"); }\n";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk(source, "rust", &config(42)).unwrap();

        assert!(output.chunks.len() > 1);
        assert!(output.chunks.iter().all(|chunk| chunk.measured_size <= 42));
        assert_eq!(
            output
                .chunks
                .iter()
                .map(
                    |chunk| source[chunk.core_range.start_byte..chunk.core_range.end_byte]
                        .to_owned()
                )
                .collect::<String>(),
            source
        );
    }

    #[test]
    fn oversized_unicode_atomic_node_splits_on_utf8_boundaries() {
        let source = "const VALUE: &str = \"🙂🙂🙂🙂🙂🙂🙂🙂\";";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk(source, "rust", &config(12)).unwrap();

        assert!(
            output
                .chunks
                .iter()
                .any(|chunk| chunk.quality.degraded_split)
        );
        assert!(output.chunks.iter().all(|chunk| chunk.measured_size <= 12));
        assert_eq!(
            output
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            source
        );
    }

    #[test]
    fn recovered_parse_is_visible() {
        let source = "fn broken( { let value = ;";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk(source, "rust", &config(100)).unwrap();

        assert!(
            output
                .chunks
                .iter()
                .all(|chunk| chunk.quality.recovered_parse)
        );
        assert!(
            output
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "recovered_parse")
        );
    }

    #[test]
    fn strict_parse_policy_rejects_recovered_tree() {
        let source = "fn broken( {";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let strict = ChunkConfig {
            parse_policy: ParsePolicy::RequireAst,
            ..config(100)
        };

        assert!(matches!(
            chunker.chunk(source, "rust", &strict),
            Err(ChunkError::ParseHasErrors { .. })
        ));
    }

    #[test]
    fn generic_fallback_is_explicit() {
        let source = "some unknown but valid UTF-8 🙂 content";
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let fallback = ChunkConfig {
            parse_policy: ParsePolicy::GenericFallback,
            ..config(12)
        };
        let output = chunker.chunk(source, "unknown", &fallback).unwrap();

        assert_eq!(output.strategy, ChunkStrategy::Generic);
        assert!(
            output
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "generic_fallback")
        );
        assert_eq!(
            output
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            source
        );
    }

    #[test]
    fn empty_source_produces_no_chunks() {
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk("", "rust", &config(10)).unwrap();
        assert!(output.chunks.is_empty());
    }

    #[test]
    fn hard_byte_cap_contains_minified_single_token_input() {
        let source = "a".repeat(60_000);
        let mut chunker = TreeSitterChunker::new(Arc::new(UnicodeWordSizer));
        let fallback = ChunkConfig {
            max_input_bytes: None,
            ..ChunkConfig::default()
        };
        let output = chunker.chunk(&source, "generic", &fallback).unwrap();

        assert!(output.chunks.len() >= 3);
        assert!(output.chunks.iter().all(|chunk| chunk.text.len() <= 25_000));
        assert_eq!(
            output
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            source
        );
    }

    #[test]
    fn lexical_split_rechecks_preferred_boundary_for_non_monotonic_sizers() {
        struct RetokenizingSizer;

        impl Sizer for RetokenizingSizer {
            fn name(&self) -> &'static str {
                "retokenizing_test"
            }

            fn measure(&self, text: &str) -> Result<usize, cast_core::SizeError> {
                Ok(usize::from(text.len() == 4) + 1)
            }
        }

        let source = "abc\nxxxxx";
        let chunks = lexical_split(
            source,
            0..source.len(),
            &[],
            1,
            None,
            &RetokenizingSizer,
            &mut HashMap::new(),
            true,
        )
        .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].range, 0..source.len());
    }

    #[test]
    fn output_carries_schema_and_algorithm_versions() {
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk("fn main() {}", "rust", &config(100)).unwrap();

        assert_eq!(output.schema_version, cast_core::OUTPUT_SCHEMA_VERSION);
        assert_eq!(output.algorithm_version, cast_core::ALGORITHM_VERSION);
    }

    #[test]
    fn large_rust_file_scale_smoke_preserves_all_bytes() {
        let mut source = String::new();
        for index in 0..2_000 {
            writeln!(source, "fn generated_{index}() -> usize {{ {index} }}")
                .expect("writing to a String cannot fail");
        }
        let mut chunker = TreeSitterChunker::new(Arc::new(ByteSizer));
        let output = chunker.chunk(&source, "rust", &config(1_500)).unwrap();

        assert!(output.chunks.len() > 20);
        assert!(
            output
                .chunks
                .iter()
                .all(|chunk| chunk.measured_size <= 1_500)
        );
        assert_eq!(
            output
                .chunks
                .iter()
                .map(|chunk| chunk.text.as_str())
                .collect::<String>(),
            source
        );
    }
}
