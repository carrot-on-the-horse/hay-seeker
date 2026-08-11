# CAST Parser and Semantic Chunker — Rust Design Specification

Status: Draft; Milestone 1 vertical slice started
Primary implementation language: Rust
Reference implementation: private legacy `chunkenator` repository (Go)
Repository: `hay-seeker`

## 1. Summary

This project will provide a Rust library and a small reference CLI for splitting source code into semantically coherent chunks using CAST (Chunking via Abstract Syntax Trees).

Tree-sitter parses source text into a concrete syntax tree. CAST walks that tree, recursively splits oversized syntax nodes, groups adjacent nodes that fit the configured size budget, and emits source-backed chunks suitable for search, retrieval, and embedding.

The first deliverable is the parsing and chunking engine. Repository indexing, embeddings, vector databases, and retrieval are deliberately outside the core scope.

## 2. Terminology

- **CAST** means Chunking via Abstract Syntax Trees. It is the chunking algorithm, not a parser implementation.
- **Parser** means the Tree-sitter parser that produces a syntax tree.
- **Chunker** means the CAST layer that turns the syntax tree into source ranges.
- **Core range** is the unique, non-overlapping range owned by a chunk.
- **Context range** is the source range returned as chunk content after optional overlap is added.
- **Sizer** is the configured unit counter, such as BPE tokens, bytes, lines, or words.
- **Degraded split** is a size-enforcement split that crosses a syntax boundary because no smaller useful AST node exists.

The working assumption is that “cAST parser” refers to the CAST algorithm used by the Go project. If the intended scope is specifically a parser for the C language, the architecture still supports it, but the language rollout in section 12 should be narrowed.

## 3. Goals

1. Implement the production code in Rust.
2. Preserve meaningful syntax boundaries whenever the configured size budget permits.
3. Produce deterministic output for the same source, language, configuration, grammar versions, and tokenizer version.
4. Preserve exact source bytes within every emitted range.
5. Cover the complete input with ordered, non-overlapping core ranges.
6. Enforce a strict final size limit when strict mode is selected.
7. Support multiple Tree-sitter grammars without coupling the core algorithm to any one language.
8. Report parse recovery, generic fallback, and degraded splitting instead of hiding them.
9. Be safe to use in parallel repository-processing pipelines.
10. Retain the useful behavior and fixture corpus of the Go prototype without requiring byte-for-byte output compatibility.

## 4. Non-goals

The first release will not:

- generate embeddings;
- index Git repositories;
- integrate with Elasticsearch or another vector store;
- implement semantic search or ranking;
- download grammars at runtime;
- expose an editor/LSP incremental-update protocol;
- guarantee compatibility with the Go package API or its JSON shape;
- promise that every chunk is independently compilable;
- invent a new programming-language grammar.

Incremental Tree-sitter parsing may be added later, but the initial API processes complete source files.

## 5. Lessons retained from the Go prototype

The Go project establishes the useful baseline:

- recursively split an oversized syntax node;
- greedily group adjacent child nodes under a size budget;
- retain source content, line numbers, byte offsets, node types, and language;
- allow pluggable token counters;
- fall back to generic text chunking;
- test against real source files in several languages;
- create one parser per indexing worker because parser instances are mutable.

The Rust implementation must not carry forward these prototype weaknesses:

- An oversized leaf can currently exceed the configured maximum.
- Group sizes are calculated from node texts while the emitted span also contains inter-node whitespace, so an emitted chunk can exceed the measured budget.
- Percentage overlap mutates offsets after inserting synthetic newlines, which makes content and coordinates disagree.
- Generic multiline chunks do not always have real byte offsets.
- Character splitting can cut a multibyte UTF-8 code point.
- Single-line generic content receives a fixed 10% overlap even when overlap is disabled or configured differently.
- Parser failure silently becomes generic chunking, and Tree-sitter recovery errors are not surfaced.
- `.h` can resolve nondeterministically because both C and C++ claim it.
- Invalid values such as a zero maximum size are not rejected early.
- Collecting every descendant node type can add substantial traversal and output cost.

## 6. Proposed workspace

```text
hay-seeker/
├── Cargo.toml
├── crates/
│   ├── cast-core/          # Parser-independent CAST algorithm and public types
│   ├── cast-tree-sitter/   # Tree-sitter adapter, registry, and grammar features
│   ├── cast-index/         # Runtime-neutral indexing domain and adapter contracts
│   └── cast-cli/           # Thin reference CLI and JSONL output
├── fixtures/
│   ├── sources/            # Ported and expanded real-source corpus
│   └── expected/           # Stable snapshot outputs
├── benches/                # Criterion benchmarks and benchmark corpus
└── DESIGN.md
```

`cast-core` must not depend on CLI, indexing, networking, or database crates. Grammar dependencies belong in `cast-tree-sitter` and should be gated by Cargo features to control compile time and binary size.

## 7. High-level architecture

```text
UTF-8 source + explicit/auto language + ChunkConfig
                         |
                         v
               deterministic detection
                         |
                         v
                 Tree-sitter parse
                         |
             tree + parse diagnostics
                         |
                         v
           AST candidate range extraction
                         |
                         v
       recursive split + adjacent range grouping
                         |
                         v
         exact coverage and budget validation
                         |
                         v
             optional source-backed overlap
                         |
                         v
             ChunkOutput + diagnostics
```

The implementation must keep parsing and segmentation separate. This allows the core chunking invariants to be tested using synthetic trees without loading every grammar.

## 8. Public data model

The exact Rust names may change during implementation, but the semantic contract should remain stable.

```rust
pub struct ChunkConfig {
    pub max_size: NonZeroUsize,
    pub overlap: Overlap,
    pub limit_policy: LimitPolicy,
    pub parse_policy: ParsePolicy,
    pub include_node_kinds: NodeKindMode,
    pub max_input_bytes: Option<NonZeroUsize>,
    pub max_chunk_bytes: Option<NonZeroUsize>,
    pub parse_timeout_ms: Option<NonZeroU64>,
}

pub enum Overlap {
    None,
    Units(usize),
    Percent(u8), // validated as 0..=50
}

pub enum LimitPolicy {
    Strict,
    PreserveAtomicNodes,
}

pub enum ParsePolicy {
    RequireAst,
    Recover,
    GenericFallback,
}

pub struct SourceRange {
    pub start_byte: usize,
    pub end_byte: usize, // exclusive
    pub start: SourcePoint,
    pub end: SourcePoint,
}

pub struct SourcePoint {
    pub line: usize,        // 1-based
    pub byte_column: usize, // 0-based
}

pub struct Chunk {
    pub ordinal: usize,
    pub text: String,
    pub core_range: SourceRange,
    pub context_range: SourceRange,
    pub measured_size: usize,
    pub language: LanguageId,
    pub node_kinds: Vec<String>,
    pub quality: ChunkQuality,
}

pub enum ChunkStrategy {
    Ast,
    Generic,
    Mixed,
}

pub struct ChunkQuality {
    pub recovered_parse: bool,
    pub degraded_split: bool,
}

pub struct ChunkOutput {
    pub chunks: Vec<Chunk>,
    pub language: LanguageResolution,
    pub strategy: ChunkStrategy,
    pub diagnostics: Vec<Diagnostic>,
}
```

Recovery and degradation are independent flags: a chunk may come from a recovered tree and still require a degraded lexical split. `measured_size` describes the final context-backed `text`, not only the non-overlapping core.

### 8.1 Range contract

- Byte ranges are half-open: `[start_byte, end_byte)`.
- Ranges always index the original UTF-8 input.
- `text` must equal `source[context_range.start_byte..context_range.end_byte]` exactly.
- No synthetic newline or other separator may be inserted into `text`.
- Core ranges must be ordered, non-empty for non-empty input, non-overlapping, and collectively reconstruct the original input.
- Context ranges may overlap but must contain their core range.
- Empty input returns zero chunks and an otherwise successful output.
- CRLF, trailing newlines, leading whitespace, comments, and a UTF-8 BOM are preserved as source bytes.

Keeping core and context ranges separate prevents overlap from corrupting provenance.

### 8.2 Node kinds

`NodeKindMode` should support:

- `None` for minimum overhead;
- `TopLevel` for kinds directly represented by the chunk;
- `AllNamed` for the sorted unique named descendant kinds.

The compatibility default is `AllNamed`, matching the metadata proven in the Go pipeline. `TopLevel` remains available for lower overhead. Anonymous punctuation tokens are excluded. Ordering must be deterministic.

## 9. Core API

```rust
pub trait Sizer: Send + Sync {
    fn name(&self) -> &'static str;
    fn measure(&self, text: &str) -> Result<usize, SizeError>;
}

pub struct Chunker {
    // Owns one mutable Tree-sitter parser and reusable traversal buffers.
}

impl Chunker {
    pub fn new(registry: Arc<LanguageRegistry>, sizer: Arc<dyn Sizer>) -> Self;

    pub fn chunk(
        &mut self,
        source: &str,
        language: LanguageRequest,
        config: &ChunkConfig,
    ) -> Result<ChunkOutput, ChunkError>;
}
```

`Chunker` is mutable and is not shared concurrently. A parallel caller creates one `Chunker` per worker and reuses it. Shared immutable language definitions and sizers may use `Arc`.

The initial library accepts UTF-8 `&str`. A file API must reject invalid UTF-8 with a typed diagnostic rather than using lossy conversion. A later range-only byte API can support non-UTF-8 source without weakening the first API's guarantees.

## 10. Sizing

The Go name `TokenCounter` is too narrow because it also supports byte and line counts. Rust uses `Sizer` and states the unit in configuration and output.

Required implementations:

- `ByteSizer` for exact byte budgets and low-overhead testing;
- `UnicodeWordSizer` for a lightweight deterministic approximation;
- `LineSizer` for debugging and compatibility tests;
- `BpeSizer` for a named, pinned tokenizer encoding.

The CLI and search chunker use the exact model tokenizer when it is available.
Otherwise they default to a pinned OpenAI® `o200k_base` encoding. The artifact
digest and tokenizer implementation version are part of the sizer identity.
`UnicodeWordSizer` remains available for compatibility with the Go
`SimpleTokenCounter`. The selected sizer or encoding must be printed in
metadata; changing it invalidates existing chunks. The high-level `Chunker`
exposes the resolved identity through `sizer_name()` so index writers do not
have to reconstruct or guess it.

GigaToken is benchmarked against the stable implementation in an isolated,
pinned nightly crate. It must not become a production dependency until it can
compile on the stable workspace toolchain without weakening workspace checks.

An independent `max_chunk_bytes` ceiling prevents one-token minified input or giant identifiers from producing multi-megabyte chunks. The compatibility default is 25,000 bytes, replacing the Go indexer's UTF-8-unsafe truncation with source-preserving lexical splits.

Sizing must always measure the actual contiguous source range that would be emitted, including whitespace and comments between syntax nodes.

## 10.1 Rust implementation baseline

- Use the Rust 2024 edition.
- Set and test an explicit minimum supported Rust version when the workspace is created.
- Use the official `tree-sitter` Rust binding and individual grammar crates.
- Keep application, CAST, registry, CLI, and test code in Rust.
- Accept that the standard Tree-sitter runtime and many generated grammar crates compile bundled C/C11 sources behind their Rust APIs; this does not introduce a Go runtime or Go production code.
- Use `thiserror` for typed library errors, `serde` for versioned output types, and `clap` for the reference CLI.
- Keep `tracing`, Rayon, tokenizer implementations, and individual language grammars behind the crate boundary or features so core users do not inherit unnecessary dependencies.
- Use a committed `Cargo.lock` for the CLI/workspace and pin compatible grammar/runtime versions during dependency upgrades.

## 11. CAST algorithm

### 11.1 Parse

1. Validate configuration and input size before parsing.
2. Resolve the language deterministically.
3. Assign the corresponding Tree-sitter language to the worker-local parser.
4. Parse with a cancellation/deadline mechanism where supported.
5. Inspect the resulting root for error and missing nodes.
6. Apply `ParsePolicy`:
   - `RequireAst`: any parse recovery node is an error;
   - `Recover`: use the recovered tree and add diagnostics;
   - `GenericFallback`: use the recovered tree when available; use generic chunking only when a grammar is unavailable or parsing cannot produce a tree.

Syntax errors must not silently erase AST behavior. Tree-sitter is designed to return useful recovered trees.

### 11.2 Split and group

The algorithm operates on contiguous source ranges, not concatenated node strings.

```text
split(node, inherited_range):
    candidate = exact source range assigned to node/group

    if measure(candidate) <= core_budget:
        emit candidate
        return

    children = useful named children in source order

    if children provide no strictly smaller progress:
        emit lexical_split(candidate)
        mark degraded when a syntax node is crossed
        return

    partition inherited_range around child boundaries
    greedily group adjacent partitions by measuring each full proposed span

    for each oversized child/group:
        recurse
```

Important rules:

- Every recursion must reduce byte length or terminate in lexical splitting.
- Group measurement uses the complete proposed source slice.
- Partition boundaries assign all separators, whitespace, and comments to exactly one core range.
- Prefer named nodes over anonymous punctuation nodes for semantic boundaries.
- Preserve declaration-leading documentation comments with the declaration where the grammar exposes a reliable relationship or a language rule defines it.
- When no safe language-specific comment rule exists, exact coverage and deterministic output take priority over heuristic attachment.
- Do not emit whitespace-only chunks unless the complete input is whitespace. Attach separator-only ranges to an adjacent semantic chunk.

### 11.3 Strict lexical fallback

In `LimitPolicy::Strict`, a chunk must not exceed `max_size`, even when one leaf or atomic syntax node is too large.

The lexical fallback finds the largest UTF-8-safe prefix whose measured size is within budget. It prefers boundaries in this order:

1. line boundary;
2. whitespace boundary;
3. punctuation boundary;
4. UTF-8 character boundary.

If measuring arbitrary prefixes is expensive, a monotonic binary search over valid character boundaries may be used. The algorithm must guarantee forward progress. Chunks produced by this path receive `ChunkQuality::Degraded` and a diagnostic identifying the oversized node kind.

In `LimitPolicy::PreserveAtomicNodes`, an atomic syntax node may exceed the limit. The output must report the exception and its measured size.

### 11.4 Overlap

Overlap is derived only after non-overlapping core chunks pass coverage validation.

- It expands a chunk's context range into adjacent original source.
- It never inserts text.
- It prefers complete neighboring semantic ranges when they fit.
- It falls back to UTF-8-safe lexical boundaries when only part of a neighbor fits.
- Under `Strict`, the core budget is reduced to reserve context space so the final measured chunk remains at or below `max_size`.
- The first and last chunk naturally have one-sided overlap.
- `Percent` is calculated from `max_size`, not from raw byte length.

The implementation may return less overlap than requested when a minimum semantic core would otherwise be lost.

### 11.5 Validation pass

Before returning, debug builds and tests must validate:

1. all ranges are within input bounds and on UTF-8 boundaries;
2. every `text` exactly matches its context range;
3. core ranges are ordered and do not overlap;
4. concatenating core slices reproduces the source exactly;
5. every context range contains its core range;
6. strict chunks are within the configured budget;
7. ordinals are contiguous from zero.

Production builds should retain low-cost bounds and strict-size checks. Full reconstruction validation may be feature-gated if profiling shows material overhead.

## 12. Language registry and rollout

```rust
pub struct LanguageDefinition {
    pub id: LanguageId,
    pub extensions: &'static [&'static str],
    pub exact_filenames: &'static [&'static str],
    pub grammar: tree_sitter::Language,
    pub comment_policy: CommentPolicy,
}
```

Resolution precedence:

1. caller-provided language;
2. exact filename;
3. unambiguous extension;
4. configured project preference;
5. an explicit ambiguous/unknown result.

The registry must not depend on hash-map iteration order. `.h` is ambiguous between C and C++; automatic mode returns an ambiguity unless configuration selects a preference. TypeScript and TSX must select their correct grammar variants.

### 12.1 Initial grammar set

Tier 1 is chosen to cover the old real-source fixtures and common repository code:

- Rust
- Go
- Python
- JavaScript
- TypeScript and TSX
- Java
- C
- C++
- PHP
- Bash
- C#
- Ruby
- Generic UTF-8 text fallback

Tier 2 may add CSS, Dockerfile, HTML, JSON, Kotlin, Lua, Markdown, SQL, Swift, TOML, YAML, and the remaining languages from the Go registry. Each compiled grammar is a Cargo feature. `all-languages` is an opt-in convenience feature, not the minimal library default.

Grammar and Tree-sitter runtime versions must be pinned together and upgraded through dependency-update pull requests with the full snapshot suite.

## 13. Diagnostics and error handling

Use typed errors for failures that prevent output and structured diagnostics for recoverable conditions.

Fatal error categories:

- invalid configuration;
- input too large;
- invalid UTF-8 in the file API;
- unsupported or ambiguous language when policy requires AST;
- incompatible grammar ABI;
- parse cancellation or timeout;
- sizer failure;
- invariant violation.

Recoverable diagnostics include:

- language inferred from extension;
- ambiguous extension resolved by project preference;
- syntax tree contains error or missing nodes;
- generic fallback used;
- strict lexical split crossed a syntax node;
- preserved atomic node exceeded the configured size;
- requested overlap could not be fully satisfied.

Diagnostics should include a stable code, severity, message, and optional source range. Do not make callers parse human-readable error strings.

## 14. CLI

The CLI exists to exercise the library and provide a useful local tool; it is not the architecture boundary.

Proposed commands:

```text
cast chunk <path|-> [--language <id>|auto]
                     [--max-size <n>]
                     [--sizer <bytes|words|lines|bpe:encoding>]
                     [--overlap <n>|<n>%]
                     [--strict]
                     [--format jsonl|json|text]

cast languages
cast inspect <path|-> [--language <id>|auto]
```

`chunk` writes chunks and diagnostics. `inspect` prints the Tree-sitter S-expression and parse diagnostics for debugging. Machine-readable output goes to stdout; logs and human diagnostics go to stderr. JSON output includes a schema version.

The parser/chunker CLI requires no network access. Optional provider adapters,
such as `cast-embeddings`, are separate crates and perform network I/O only when
explicitly composed into an indexing or evaluation application.

## 15. Configuration

The library uses typed Rust configuration only. The CLI may later read `cast.toml`, but command-line flags are sufficient for the first milestone.

Recommended initial defaults for the CLI:

- `max_size`: 1500 units;
- `overlap`: none;
- `limit_policy`: strict;
- `parse_policy`: generic fallback with explicit diagnostics;
- `node_kinds`: all named descendants;
- `max_input_bytes`: 5 MiB;
- `max_chunk_bytes`: 25,000 bytes;
- `parse_timeout_ms`: 60,000;
- `language`: auto.

The selected sizer or BPE encoding must be explicit in the CLI help and serialized metadata. Environment-variable configuration is postponed until there is a concrete deployment need.

## 16. Concurrency and performance

- A Tree-sitter `Parser` is mutable. Each worker owns and reuses one `Chunker`.
- The registry and immutable grammar handles are shared.
- Parsing and chunking are synchronous CPU work. The core library does not introduce an async runtime.
- Repository-level callers may use Rayon or their existing worker pool.
- Traversal should be iterative where practical to avoid stack growth on pathological trees.
- Reuse node/range buffers between calls.
- Cache range measurements within one chunk operation by byte-range key.
- Materialize owned chunk strings only after ranges are finalized.
- Do not optimize node-kind collection until the core range algorithm is correct.

Performance is subordinate to correctness in milestone 1. Benchmarks prevent major regressions; they are not acceptance criteria until a representative corpus and target hardware are agreed.

Track at least:

- parse time;
- CAST traversal/grouping time;
- tokenizer time;
- total time;
- bytes per second;
- allocations and peak resident memory;
- chunk count and degraded-split count.

## 17. Security and resource limits

- Treat source code as untrusted input.
- Compile known grammar crates into the binary; do not load arbitrary native grammar libraries.
- Enforce a configurable input-size limit in the CLI.
- Support parse cancellation/deadlines to contain pathological input.
- Validate every offset before slicing.
- Avoid recursion that can overflow the Rust stack on adversarial trees.
- Do not execute repository code, build scripts from target repositories, or network calls.
- Run dependency auditing in CI.
- Consider WASM-sandboxed runtime grammars only as a separate future design.

## 18. Testing strategy

### 18.1 Unit tests

- configuration validation, including zero sizes and invalid percentages;
- language detection and every ambiguous extension;
- exact range and line/column conversion;
- each sizer;
- grouping at exactly below, equal to, and above the limit;
- empty, whitespace-only, and comment-only source;
- oversized atomic nodes;
- parse-recovery policies;
- overlap at file boundaries;
- BOM, CRLF, tabs, emoji, combining characters, and non-Latin identifiers;
- deterministic node-kind ordering.

### 18.2 Property tests

For generated valid UTF-8 source and synthetic trees:

- the algorithm terminates;
- ranges never leave input bounds;
- core ranges reconstruct the input;
- no strict output exceeds its budget;
- overlap never changes source bytes;
- repeated runs are identical.

Use `proptest` for this layer.

### 18.3 Snapshot and compatibility tests

The first Rust approval suite records the complete serialized `ChunkOutput` for
three compact synthetic cases: clean Go AST chunking, recovered Python syntax,
and generic fallback for minified unknown input. This locks text, ranges,
measured sizes, language resolution, node kinds, quality flags, diagnostics,
algorithm version, and tokenizer identity in one reviewed contract.

The snapshots use the Go suite's `.approved.json` naming convention and an
explicit `UPDATE_CAST_GOLDENS=1` regeneration switch. Regeneration is refused
in CI, and every changed snapshot must be inspected as a behavior diff. Keep
approval fixtures small enough to review. Real WordPress, Django, Kubernetes,
and Ollama sources belong in the pinned benchmark/evaluation corpus, not in
large copied snapshots.

Expand the compact corpus across Rust, JavaScript, TypeScript, Java, C, C++,
PHP, and other supported modes as behavior-specific regressions are found.

The Go output is reference evidence, not the golden contract. Snapshot changes are expected where the Rust design fixes coverage, range, overlap, fallback, or size-limit behavior. Every snapshot update must be reviewed as a behavior change.

### 18.4 Fuzzing

Fuzz at minimum:

- generic splitting;
- source-range partitioning;
- language detection;
- overlap expansion;
- malformed-but-valid-UTF-8 source through each Tier 1 grammar.

Primary invariants are no panic, no invalid slice, termination, and exact core coverage.

### 18.5 Benchmarks

Use Criterion with:

- small and large source files;
- minified single-line input;
- files with one oversized string/comment/token;
- recovered syntax errors;
- every sizer;
- overlap disabled and enabled;
- Tier 1 languages;
- sequential reuse and one-chunker-per-worker parallel operation.

## 19. Observability

The library should expose data rather than emit logs. The CLI can map it to tracing events.

Per-file summary fields:

- language and how it was resolved;
- grammar version/identity where available;
- source bytes;
- parse duration;
- chunk duration;
- number of chunks;
- strategy used;
- number of parse errors/missing nodes;
- number of degraded splits;
- maximum and average measured chunk size.

This makes fallback rates and grammar regressions visible in a future indexer.

## 20. Versioning and compatibility

- Start JSON output at `schema_version: 1`.
- Use semantic versioning for public crates.
- Treat range semantics, strategy meanings, diagnostic codes, and JSON fields as public contracts.
- Chunk boundaries may change after grammar or tokenizer upgrades; serialize grammar and sizer identity so indexes can be invalidated deliberately.
- A future indexer should include algorithm version, grammar set version, and tokenizer identity in its content fingerprint.

## 21. Milestones

### Milestone 0 — skeleton and contracts

- Create the Cargo workspace and three crates.
- Implement public types, configuration validation, `Sizer`, and diagnostics.
- Add CI for formatting, Clippy, tests, dependency audit, and minimum supported Rust version.

Exit criterion: core types compile, invalid configurations fail predictably, and CI is green.

### Milestone 1 — correct single-language vertical slice

- Integrate Tree-sitter Rust grammar.
- Implement exact range partitioning, recursive CAST grouping, strict lexical fallback, and validation.
- Add UTF-8/property tests and initial benchmarks.

Exit criterion: all core invariants hold for Rust fixtures, including oversized atomic input and recovered syntax errors.

### Milestone 2 — Tier 1 languages

- Add the Tier 1 grammar registry and deterministic detection.
- Port the Go fixture corpus.
- Implement comment policies only where tests justify them.

Exit criterion: snapshots and invariants pass across all Tier 1 languages; ambiguous `.h` behavior is explicit and tested.

### Milestone 3 — overlap and CLI

- Implement source-backed context overlap within strict budgets.
- Add JSONL schema, `chunk`, `languages`, and `inspect` commands.
- Add cancellation, input limits, and structured timing.

Exit criterion: the CLI chunks a mixed repository deterministically without network access, panics, range corruption, or silent fallback.

### Milestone 4 — hardening

- Fuzz the critical range paths.
- Profile representative repositories.
- Reduce avoidable allocation and tokenizer work.
- Document public APIs and release the initial crate versions.

Exit criterion: agreed performance baseline, no known invariant violations, and release documentation complete.

## 22. Acceptance criteria for version 1

Version 1 is ready when:

1. production implementation is Rust;
2. Tier 1 grammars are feature-gated and version-pinned;
3. the complete UTF-8 source is recoverable by concatenating core ranges;
4. chunk text always matches its original-source context range;
5. strict mode never exceeds the configured size;
6. malformed code produces useful recovered chunks or an explicit policy-driven error;
7. generic fallback and degraded splits are visible in diagnostics;
8. language detection is deterministic and ambiguity is explicit;
9. one chunker per worker is documented and tested under parallel load;
10. unit, snapshot, property, fuzz smoke, and benchmark suites exist;
11. the Go fixture corpus is represented;
12. the core crate contains no embedding, database, repository-walking, or network concerns.

## 23. Decisions to confirm before implementation

These choices do not block the architecture, but should be confirmed before Milestone 1 is finalized:

1. Is CAST the intended meaning, or is the product specifically a C-language AST parser?
2. Which BPE encoding/model will the first real consumer use?
3. Should version 1 expose only the library, or ship the reference CLI in the same release?
4. Is strict maximum size required for every downstream consumer, or should preserving atomic syntax nodes be the default?
5. Which Tier 1 languages are required on day one?

## 24. References

- Old Go prototype: private legacy `chunkenator` repository
- Tree-sitter introduction: <https://tree-sitter.github.io/tree-sitter/>
- Tree-sitter Rust binding: <https://docs.rs/tree-sitter/latest/tree_sitter/>
- Tree-sitter Rust grammar example: <https://docs.rs/tree-sitter-rust/latest/tree_sitter_rust/>
