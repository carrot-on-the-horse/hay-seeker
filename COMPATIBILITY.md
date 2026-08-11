# Go Production Compatibility Baseline

The Go `chunkenator` implementation successfully processed more than 1,000 repositories. Its operational behavior is therefore evidence, not throwaway prototype detail. The Rust rewrite may correct known correctness defects, but it must preserve or deliberately replace the scale protections listed here.

Reference: the private legacy Go `chunkenator` repository.

## Preserved in the current Rust draft

| Proven Go behavior | Rust status |
|---|---|
| Default chunk budget of 1,500 sizing units | Preserved |
| Lightweight whitespace-token sizing when BPE is not selected | Preserved as the CLI's `words` default |
| Full named descendant node kinds in metadata | Preserved as `NodeKindMode::AllNamed` default |
| Code files up to 5 MiB | Preserved as the default input ceiling |
| 25,000-byte protection before storage/embedding | Moved into the chunker as a UTF-8-safe hard chunk ceiling |
| One mutable Tree-sitter parser per worker | Preserved by `TreeSitterChunker`, which requires mutable worker-local use |
| Generic fallback keeps a run moving | Preserved, with an explicit diagnostic instead of silent fallback |
| Per-file parse protection | Preserved as a cancellable 60-second Tree-sitter parse deadline |
| Stable chunk ordering and source coordinates | Preserved and strengthened with exact range validation |
| Repeated size checks can be expensive with BPE | Range measurements are cached within each chunk operation |
| Empty chunks are not useful downstream | Empty input returns no chunks; whitespace retention remains exact in core output |

The hard byte ceiling is intentionally independent of the selected sizer. Without it, a minified file or giant identifier can count as one whitespace token and create a multi-megabyte chunk. The old indexer truncated such content to 25,000 bytes; Rust splits it safely at UTF-8 boundaries instead, preserving the complete source.

## Required in the future repository/indexer layer

The first product full-rebuild path now implements:

- streaming `git ls-files` enumeration for tracked and non-ignored untracked
  files, with a deterministic filesystem fallback outside Git;
- hidden, symlink, unsupported-language, size, binary, invalid-UTF-8, blank,
  and large data-like filtering with counters in CLI output;
- immediate per-file CAST chunking with only one source file and one chunk set
  resident before bounded 128-document backend batches;
- deterministic SHA-256 document IDs over normalized path, complete source
  content, backend-neutral relevance fingerprint, chunk ordinal, and core byte
  range;
- bounded DuckDB staging followed by one atomic promotion, and bounded
  Elasticsearch bulk requests followed by one atomic alias swap;
- rollback/preservation of the previous searchable generation after a late
  source error;
- versioned, backend-neutral repository checkpoints with content-hash reuse;
- explicit stale-chunk deletion for changed, deleted, and newly filtered files;
- bounded DuckDB delta staging with one insert/delete commit and Elasticsearch
  copy-on-write incremental generations that retain unchanged stored vectors;
- flush and searchable-generation publication before checkpoint publication,
  with pre-run invalidation so interrupted runs fall back to a full rebuild;
- periodic structured progress, a configurable 10-minute no-completion stall
  abort, and full/incremental, source-byte, reuse/delete/skip, discovery, read,
  chunk, and total-time metrics.

The remaining production-indexer work is:

- default worker count from available CPU parallelism;
- instantiate and reuse one chunker per worker;
- impose a 60-second whole-file job deadline in addition to the parser deadline;
- add cross-branch deduplication and branch-scoped checkpoint selection;
- generalize bounded batches and typed retry execution to future providers;
- expose worker utilization and separate embedding/storage timings.

The Rust indexer must use bounded queues. A 1,000-repository run must not load every file body or every chunk document into memory at once.

## Deliberate corrections, not compatibility regressions

The following Go behaviors must not be copied:

- nondeterministic `.h` detection caused by C and C++ sharing a map-driven extension registry;
- overlap that inserts synthetic newlines and then reports offsets as if they came from the source;
- generic chunks with zero or incorrect byte offsets;
- UTF-8-unsafe byte truncation;
- silently hiding fallback and recovered syntax errors;
- allowing invalid zero limits that can prevent forward progress;
- measuring grouped node bodies without the whitespace later included in output;
- abandoning a stuck parser without a cancellation mechanism;
- treating collected deleted paths as sufficient without verifying storage deletion.

## Migration contract

Rust chunk boundaries are intentionally not byte-for-byte compatible with Go because range coverage, limits, fallback diagnostics, and UTF-8 handling are corrected. Existing indexes require a full rebuild when switching implementations.

Every output includes:

- `schema_version` for serialization compatibility;
- `algorithm_version` for chunk-boundary invalidation;
- `sizer` identity;
- language-resolution method;
- explicit diagnostics.

The future index fingerprint must also include grammar versions and complete tokenizer identity. Do not reuse Go document IDs across the migration merely because file content hashes match.

## Parity gaps still open

- Thirteen popular language modes are compiled; the remaining languages from
  the roughly 30-language Go registry are still open parity work.
- Pinned OpenAI® `o200k_base` sizing is implemented; exact tokenizers for
  non-OpenAI embedding models remain open parity work.
- The Cloudflare/Vertex Gemini Embedding 2 adapter is implemented, including
  bounded concurrency and typed retries. Full repository rebuild, DuckDB and
  Elasticsearch hybrid retrieval, product CLI selection, content-hash
  incrementality, explicit deletion, and crash-safe checkpoint fallback are
  wired; branch-scoped reuse and the remaining provider adapters remain open.
- Whole-job timeouts and 10-minute stall detection belong to the future indexer.
- A compact Rust-native approval corpus now locks complete chunk output for
  clean AST, recovered-parse, and generic-fallback behavior. The wider Go
  real-source approval corpus remains open and should be ported selectively as
  behavior-specific regression fixtures, rather than copied wholesale.
- File-level cross-repository lexical benchmarks pass for DuckDB and
  Elasticsearch; chunk-level dense recall and peak-RSS gates remain open.

Until these gaps close, the current code is a correct parser/chunker vertical slice, not a replacement for the production Go indexing pipeline.
