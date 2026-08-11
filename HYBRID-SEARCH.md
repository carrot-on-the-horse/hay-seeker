# Embedded Hybrid Search Design

Status: DuckDB and Elasticsearch product slice implemented
Language: Rust 2024
Backends: local Apple Silicon and Elasticsearch

## Mission and boundary

`hay-search` is an embedded library with one backend-neutral query contract. A caller constructs a `Retriever`, submits the same `Query` and `SearchOpts`, and receives the same `Candidate` shape regardless of storage. Backend selection is composition-time wiring, never a branch in application query logic.

The project will not introduce a network server, daemon, REST API, web UI,
custom ANN implementation, hosted embedding dependency on the local path,
ELSER, or another proprietary Elastic model. The executable surfaces are thin
development, evaluation, and MCP stdio CLIs. The MCP process is launched and
owned by its client and delegates to the same embedded `Retriever`; it is not a
separate search service.

## Compatibility boundary

Every built index persists this complete manifest:

```text
model_id
model_revision
embedding_profile
embed_dim
mrl_dim
quantization
tokenizer_hash
chunker_version
fde_params
schema_version
```

The query path compares all fields to runtime configuration before retrieval. Any difference returns `SearchError::ReindexRequired` with each mismatched field. There is no warning-only path and no automatic fallback. Model ID and revision name an immutable portable checkpoint shared by both backends. The one allowed representation difference is `mrl_dim`: 256 locally and 768 remotely.

The manifest is about relevance compatibility, not only storage decoding.
Changing tokenization, chunking, model revision, embedding input profile,
dimensions, quantization, FDE projections, or schema invalidates the index even
if old bytes remain readable. The embedding profile includes task prefixes,
pooling, truncation, and normalization behavior.

The lexical manifest pins the OpenAI `o200k_base` merge-rank artifact used for
chunk sizing in `tokenizer_hash`; `model_revision` independently pins the Hay
BM25 analyzer contract. Gemini manifests additionally require an
operator-supplied model revision and bind the complete Gateway route hash.
The offline ONNX manifest binds the bundle-manifest checksum, every model and
tokenizer artifact digest, the final pooled-output tensor contract, the
official code-query/document prompts, and post-MRL normalization. It uses the
same 768-dimensional checkpoint for both targets, storing 256 dimensions in
DuckDB as per-vector scale/offset int8 values and 768 in Elasticsearch with
BBQ. DuckDB scans the quantized values exactly in deterministic Rust code; ANN
remains absent through the 250,000-chunk acceptance threshold.

## Common query contract

```rust
#[async_trait]
pub trait Retriever: Send + Sync {
    async fn search(
        &self,
        query: &Query,
        options: &SearchOpts,
    ) -> Result<Vec<Candidate>, SearchError>;

    fn capabilities(&self) -> Capabilities;
}
```

`Candidate` contains a stable document ID, final score, and optional lexical, dense, and late-interaction signals. Both backends use the same cascade:

```text
lexical + dense candidate generation
                |
                v
       quantized rescore
                |
                v
 optional late-interaction rerank
                |
                v
 stable score order, then document-ID tie break
```

A backend skips a stage only when its `Capabilities` value says the stage is unavailable. It must not replace that stage with a backend-specific approximation. Ranking inputs never include wall-clock values or nondeterministic map iteration.

## Corpus and chunking

`Corpus` is pull-based and yields stable `CorpusDocument` values. This retains the bounded-memory behavior that allowed the Go system to process more than 1,000 repositories. The indexing contracts in [INDEX-CONTRACTS.md](./INDEX-CONTRACTS.md) further require bounded queues, one mutable Tree-sitter parser per worker, content-hash incremental processing, durable file batches, and flush-before-checkpoint ordering.

The product CLI now consumes this boundary directly with `hay index
--repository`: Git enumeration and per-file CAST chunking feed bounded backend
batches without collecting the repository's source bodies or chunks in a
single Rust vector. DuckDB uses staging tables plus atomic promotion;
the CLI additionally replaces incompatible database files through a sibling
temporary build. Elasticsearch uses bounded bulk requests plus atomic alias
replacement. Versioned repository checkpoints now make later runs
content-hash incremental: unchanged chunks and vectors are reused, while stale
IDs from changed, deleted, or newly filtered files are removed. DuckDB stages
the complete delta before one transaction; Elasticsearch copies the prior
generation, applies the delta, and swaps the alias. Checkpoints are invalidated
before mutation and republished only after the searchable generation succeeds,
so interruption forces a safe full rebuild rather than stale-state reuse.
DuckDB also retains a manifest-scoped embedding cache across incremental runs.
Keys bind the stable document ID and exact text, which permits safe reuse when
switching back to old branch content without allowing changed content under a
reused caller-supplied ID to receive a stale vector. Cache rows are decoded
with the active quantization and dimension contract and fail closed on any
shape, codec, or deterministic-output conflict. Full database rebuilds are the
explicit cache-compaction boundary when they replace the database file.

`ChunkerV1` selects:

- Rust: the CAST Tree-sitter chunker with a 1,500 Unicode-word budget;
- non-code: 6,000-byte fixed windows with 600 bytes of source-backed overlap.

The persisted `chunker_version` is the complete executable profile rather than
an informal release label. It pins the algorithm family, AST size, overlap,
limit/parse/node-kind policies, input and chunk byte caps, parse timeout,
fixed-window parameters, sizing tokenizer identity, and every compiled
Tree-sitter grammar package/version. Repository indexing compares this value
to the constructed `ChunkerV1` and fails before reading source when they differ.

Core ranges cover the complete source exactly once. Context ranges may overlap. Both remain valid UTF-8 byte ranges into the original text; the implementation never manufactures separators or truncates a code point.

## Phase plan and gates

Each phase is an independent Cargo feature and one independently revertible PR. A failed acceptance gate is reported and stops later work; judgments are never edited to make a phase pass.

### Phase 0 — harness first (`phase0`)

Contracts, exact manifest validation, AST/fixed-window chunking, deterministic random stub, and checked-in JSONL evaluation. The harness reports nDCG@10, recall@50, and MRR for at least 30 graded queries.

Acceptance: both backend labels run the same stub and print a reproducible baseline. No production retrieval implementation is allowed in this phase.

### Phase 1 — local floor (implemented lexical baseline, optimization gates open)

- DuckDB transactional storage with persisted BM25 statistics and code
  identifier analysis retaining original, camelCase, and snake_case parts.
- Portable EmbeddingGemma-300M-class ONNX checkpoint through `ort` and CoreML.
- 256-dimension MRL truncation; per-vector scale/offset int8 storage in a memory-mapped slab.
- Exact cosine scan; ANN remains gated by measured corpus size and latency.
- RRF with `k = 60`.
- Durable/resumable segments, observer callbacks, and macOS low-QoS embedding work.

Implemented: full rebuild, content-hash incremental indexing, atomic staged
upsert/delete, crash-safe checkpoint fallback, restart persistence, exact
manifest validation, BM25, exact cosine, RRF, Gemini CLI, and MCP adapter.
Indexing, direct search, MCP, and evaluation share one `hay-runtime` provider
and manifest constructor, preventing surface-specific identity drift. OpenAI,
Voyage code embeddings, and Cloudflare Workers AI Qwen3 adapters implement the
same contract with exact response-order/shape validation and mandatory model
revisions.
The CLI also emits periodic structured progress, enforces a configurable
no-completion stall timeout, and reports discovery/read/chunk/total timings.
Remaining acceptance: CAST-chunked dense recall and latency/RSS gates. The
product-correct lexical suite contains 89,183 chunks from 29,772 files;
first-ranked chunk collapse produced DuckDB nDCG@10 0.434779 and recall@50
0.887500.

On macOS, each local ONNX document tokenizer/inference batch is synchronously
submitted to libdispatch's Utility QoS queue. This keeps the caller-facing
async contract and deterministic batch order while ensuring long-running
indexing work yields to interactive applications; no caller or Tokio worker
thread has its QoS mutated permanently. Query encoding remains at the caller's
priority so the cold-query latency gate is not weakened.

### Phase 2 — Elasticsearch parity (lexical gate passed)

Use the same chunker and pinned encoder. Store BM25 text and a BBQ
`dense_vector`, run native BM25 and kNN candidates, then apply the shared Hay
RRF with the same fusion parameters. This avoids making the product path depend
on Elastic's separately licensed native RRF retriever.

Implemented: physical-index rebuild, mapping manifest, bounded 5 MiB bulk
ingest, bounded 128-document embedding batches, final refresh, failed-build
cleanup, atomic alias swap, BM25, BBQ dense kNN, shared license-independent RRF,
copy-on-write incremental generations with explicit stale-ID deletion, bounded
physical-generation retention in the atomic publish request, post-error alias
reconciliation at the ambiguous commit boundary, Gemini CLI, MCP adapter, and
opt-in live lifecycle test. The official
Elasticsearch 9.3.2 live run indexed the same 89,183 CAST chunks and produced
nDCG@10 0.433320 versus DuckDB 0.434779, a difference of 0.001459; both
achieved recall@50 0.887500. The lexical parity gate passes. A Basic-license
live test now covers dense indexing and hybrid query signals.
`eval --backend parity` validates the two runtime manifests, runs the identical
loaded chunks and parent-file judgments through both production adapters,
prints absolute metric deltas, and hard-fails above `0.02` nDCG@10. Dense
CAST-chunked recall parity remains open before late interaction.

### Phase 3 — late interaction (`late`)

Use a pinned portable `answerai-colbert-small` ONNX checkpoint. Store local per-token int8 vectors in memory-mapped form. MaxSim reranks roughly the top 100 union candidates and fully replaces earlier scores. Elastic uses `rank_vectors` and the same MaxSim rule.

Acceptance: nDCG@10 improves by at least 0.03 or the feature remains off by default. Criterion covers MaxSim rescore.

### Phase 4 — learned sparse (`sparse`)

Use a portable SPLADE ONNX checkpoint. Local Tantivy reads stored term impacts through a custom scorer; Elastic uses `sparse_vector`.

Acceptance: measurable gain on the frozen paraphrase-query subset.

### Phase 5 — MUVERA FDE (`fde`)

Attempt only if Phase 3 wins quality but loses latency. Implement the algorithm from arXiv:2405.19504: SimHash partitions, summed query token embeddings, and averaged document token embeddings. FDE becomes the dense retrieval key while MaxSim remains the rescore.

## Phase 0 evaluation format

Corpus records are JSONL objects with `doc_id`, `path`, `language`, and `text`. Evaluation files other than `corpus.jsonl` contain:

```json
{"query":"How are equal scores ordered?","graded_doc_ids":{"determinism":3}}
```

Run either backend label:

```bash
cargo run --bin eval -- --backend local
cargo run --bin eval -- --backend elastic
```

The harness sorts input file names, rejects blank or duplicate queries, rejects unknown graded document IDs, requires a positive judgment per query, and refuses fewer than 30 cases.

The extended repository suite inventories pinned WordPress, Django, Kubernetes, and Ollama checkouts and evaluates 40 graded file-retrieval queries. See [BENCHMARKS.md](./BENCHMARKS.md). Its random Phase 0 result is a harness floor; production recall comparisons begin with Phase 1.

## Open model decision

“EmbeddingGemma-300M class” and “answerai-colbert-small” are model families, not immutable artifacts. Before Phase 1 code begins, pin the exact model IDs, revisions or artifact digests, ONNX graph variants, tokenizer artifacts, pooling/normalization rules, and license provenance. Implementing the encoder before that choice would violate the manifest contract.

The embedded backend investigation and its acceptance gates are recorded in
[EMBEDDED-SEARCH-RESEARCH.md](./EMBEDDED-SEARCH-RESEARCH.md). Qdrant Edge is the
next gated experiment; DuckDB remains the supported correctness floor.
