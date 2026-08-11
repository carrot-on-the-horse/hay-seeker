# Common Index Contracts

Status: Initial contract draft
Implementation: `crates/cast-index`
Scope: Backend-neutral repository indexing contracts

## Boundary

`cast-index` defines the shared language between repository sources, file policy, worker-local chunkers, embedders, storage drivers, progress observers, and orchestration.

It does not implement Git traversal, provider cloning, filtering policy, a worker pool, embeddings, Elasticsearch, or another storage backend. Those components depend on these contracts rather than depending on each other.

The crate is async-runtime neutral. External adapters return object-safe boxed futures from the standard library, so an application can use Tokio, async-std, a custom executor, or synchronous test doubles.

## Pipeline contract

```text
RepositorySnapshot + prior SyncCheckpoint
                  |
                  v
       pull-based FileCursor
                  |
                  v
       eligibility and limits
                  |
             bounded queue
                  |
                  v
       read one UTF-8 SourceFile
                  |
        worker-local ChunkEngine
                  |
                  v
        versioned IndexDocuments
                  |
       lookup exact fingerprint
          /               \
 attach existing       embed missing
    branch                 |
          \               /
           file-scoped WriteBatch
                  |
             store flush
                  |
         save SyncCheckpoint
```

Required sequencing:

1. Open an inventory for an immutable `RepositorySnapshot`.
2. Pull descriptors only when bounded queue capacity is available.
3. Turn deleted descriptors directly into `DeleteFile`; never attempt to read them.
4. Apply eligibility policy before reading large file bodies.
5. Read and validate one UTF-8 `SourceFile` per job.
6. Use one `ChunkEngine` created by `ChunkEngineFactory` per worker.
7. Map nonblank chunks into `IndexDocument` values without trimming or changing source-backed text.
8. Calculate document IDs through `DocumentIdFactory` using the complete identity input.
9. Use `FileLookup` only when repository, normalized path, content hash, and full index fingerprint match.
10. If exact content already exists and the store supports it, emit `AttachBranch`; otherwise embed and upsert.
11. Group operations for only one repository file into each `WriteBatch` and call `validate` before storage.
12. Treat a file as processed only after its batch is accepted.
13. Flush all buffered writes.
14. Save `SyncCheckpoint` only after flush confirms durability and no required file operation failed.

Partial runs may report progress but must not advance the checkpoint. Retrying from the previous checkpoint must be safe.

## Identity

### Repository

`RepositoryId` is `provider + namespace + name`. Namespace supports nested groups such as `platform/search/team`. This avoids collisions across providers and organizations.

### Paths

`NormalizedPath` is repository-relative and always serialized with `/`. Absolute paths, Windows drive paths, NUL bytes, and `..` traversal are rejected. Machine-specific checkout roots never enter document IDs.

### Content

`ContentHash` names both its algorithm and canonical hexadecimal digest. SHA-256 is the initial contract. Hashes are computed from the complete original file bytes before chunking.

### Documents

`DocumentIdFactory` must include every `DocumentIdentityInput` field:

- repository identity;
- normalized relative path;
- file content hash;
- complete index fingerprint;
- chunk ordinal;
- core byte range.

The exact hash encoding belongs to the factory implementation, but it must be deterministic and versioned. It must not reuse the Go ID formula, which omitted the chunking/tokenizer fingerprint.

## Fingerprint and invalidation

`IndexFingerprint` contains:

- index-document schema version;
- CAST algorithm version;
- complete grammar-set version;
- sizer name and revision;
- embedding provider, model, and dimensions when embeddings are stored.

Any difference makes previous chunks ineligible for content reuse and forces reprocessing. A checkpoint with a different fingerprint cannot seed an incremental run.

Future tokenizer identities must include merge-table/encoding revision, not only a friendly model name. Grammar-set identity must change when any compiled grammar version changes.

The chunker exposes its complete resolved sizing identity through
`sizer_name()`. Index adapters must copy that value into `SizerIdentity` (or a
lossless structured equivalent) rather than deriving identity from provider or
model display names.

`EmbeddingIdentity.profile` versions all transformations around the remote
model: retrieval prefixes or task instructions, pooling, truncation, and
normalization. Changing any of these requires a different profile and therefore
invalidates stored vectors even when provider, model alias, and dimensions stay
the same.

## Index document

`IndexDocument` is the common searchable record. It includes repository, path, branch set, revision, content hash, chunk ordinal/count, core/context source ranges, language, node kinds, exact content, quality flags, fingerprint, timestamp, and optional embedding.

Validation requires:

- supported document schema;
- valid ordinal and total;
- ordered ranges with context containing core;
- content byte length equal to the context range length;
- nonblank content;
- sorted, unique node kinds;
- fingerprint and document schema agreement;
- finite embedding values with exact declared dimensions;
- embedding identity equal to the fingerprint identity.

Whitespace-only core chunks are skipped by the document mapper. Nonblank chunks retain exact `Chunk.text`; trimming would invalidate their source range.

## Source inventory

`FileCursor` is pull-based rather than returning every repository file in a `Vec`. This is required for bounded memory when processing large repositories or many repositories concurrently.

`FileDescriptor` carries status, normalized path, byte length, optional language hint, and optional precomputed content hash. Providers should populate a hash only when it is trustworthy for the immutable snapshot.

`Eligibility` and `SkipReason` make filtering observable. The first repository implementation must preserve the proven Go reasons: hidden, ignored, unsupported language, binary, invalid UTF-8, oversized config, oversized source, data-like file, and read/policy failures.

## Writes and deletion

`WriteOperation` has three explicit forms:

- `Upsert` an independently searchable document;
- `DeleteFile` for a repository/path/branch;
- `AttachBranch` to exact matching documents without re-embedding.

`WriteBatch` is file-scoped. Stores advertising `transactional_file_batches` must apply the batch atomically. Other stores must make operations idempotent and report a non-durable receipt until `flush` succeeds.

Deleted paths are not statistics alone: every deletion must result in a successful `DeleteFile` operation before checkpoint advancement.

## Errors and retries

Adapters return `IndexError` with a stable code, category, message, and `RetryAdvice`:

- `Never` for permanent input/configuration failures;
- `Immediate` for transient optimistic conflicts;
- `AfterMillis` for rate limits and provider backoff.

The orchestrator owns retry count, jitter, and total retry budget. Adapters describe retryability but must not sleep internally unless their provider contract requires it.

Timeout, cancellation, and stall are distinct outcomes. A 60-second per-file deadline includes reading, chunking, embedding, and storage; the parser also retains its own cancellable 60-second ceiling. Ten minutes without a completed or skipped file marks the run stalled.

## Configuration defaults

`IndexConfig` carries the proven operational limits:

- workers: available parallelism;
- bounded queue: two jobs per worker;
- write batch: 100 documents;
- embedding batch: 100 documents, capped further by provider limits;
- code file: 5 MiB;
- config/data file: 50 KiB;
- binary probe: first 8 KiB;
- per-file timeout: 60 seconds;
- stall timeout: 10 minutes.

All are application-configurable. Providers may impose smaller limits, never silently larger ones.

## Progress and completion

`IndexObserver` receives lightweight events and must not block worker threads. Expensive logging or telemetry should be queued by the observer implementation.

`IndexSummary` distinguishes completed, completed-with-skips, cancelled, stalled, and failed runs and records repository/file/chunk counts plus phase timings. Skips must remain visible by reason in events or higher-level metrics.

## Compatibility rule

These contracts preserve the Go pipeline's successful scale behavior while fixing ambiguous identity, unbounded inventory collection, silent fallback, deletion accounting, and checkpoint ordering. The Rust indexer is not production-parity until real source providers, the full grammar set, BPE sizing, a durable store adapter, incremental tests, and multi-repository benchmarks implement these contracts.
