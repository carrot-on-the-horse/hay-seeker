# Embedded and On-Device Search Backends

Status: decision record and benchmark plan
Date: 2026-08-10

## Decision

Keep the implemented DuckDB backend as the supported embedded product floor.
Add **Qdrant Edge** as the next time-boxed backend experiment. Do not replace
DuckDB or add another permanent backend until it passes the same frozen recall,
restart, incremental-update, and resource gates.

Qdrant Edge is the closest match to Hay Seeker's common contracts: it runs in
the Rust process, persists a shard with a WAL, stores named dense and sparse
vectors, provides an offline BM25 document/query encoder, and exposes the query
building blocks needed for dense+sparse fusion. Its current Rust crate is
`0.7.2`, so API and migration stability are the primary risks.

The first experiment should not change `Retriever`, `SearchDocument`, or
`IndexManifest`. It is an adapter behind those contracts.

For the existing DuckDB and Elasticsearch full cycle, retain
**Snowflake Arctic Embed m v2** as the original 256d/768d relevance reference,
not the promotable product encoder. Its
official checkpoint is Apache-2.0, ungated, produces 768 dimensions, and was
trained for 256-dimensional Matryoshka truncation. The frozen seed run passed
Hay's required 256d local / 768d remote parity gate at a 0.009678 nDCG delta.
The pinned product input profile preserves 1,500-token CAST chunks but encodes
complete documents in 256-token overlapping model windows before shared 768d
aggregation; this measured 29.7% faster than the prior 2,048-token
first-window path on the Hay repository and removes tail truncation.
Nomic Embed Text v1.5 remains a useful permissive comparison, but its required
256d/768d pair missed the gate at 0.027223 even though a diagnostic 768d/768d
control passed. EmbeddingGemma remains supported for caller-provisioned licensed
bundles rather than blocking the product acceptance path. Snowflake now fails
both cold startup and the 1 GiB indexing gate, so a replacement checkpoint must
pass a cheap model-only screen before another repository-scale dense run.

The first promotable low-footprint alternative is
**minishlab/potion-code-16M-v2**. It is a code-trained static 256d token table,
not a transformer and not an untrained truncation. Hay's direct safe-Rust
runtime starts in 87.59 ms and uses 136.4 MB model-only RSS. It passed the full
89,183-chunk backend-parity suite and the 79,949-chunk Kubernetes product build
under 1 GiB. Adopting it is an explicit 256d/256d representation-contract
change; its distinct profile prevents silent compatibility with 256d/768d
indexes. Its repository-scale relevance gain is marginal and mixed, so it is
an efficient hybrid option rather than proof that dense always improves
lexical retrieval.

## Cold-start encoder finding

Snowflake Arctic m v2 meets relevance and licensing requirements but does not
meet the Phase 1 cold-query gate. Full product initialization on Apple Silicon
measured 867.4 ms from the original int8 ONNX bundle. An ONNX Runtime
fully-optimized ONNX graph measured 826.9 ms, and an official fixed CPU `.ort`
conversion measured 851.8 ms. All three produced equivalent roughly 5-6 ms
queries and 22 ms eight-document batches. Serialization is therefore not the
missing order-of-magnitude improvement; safely checksum-verifying and opening
the roughly 311 MB checkpoint dominates startup.

Do not add an optimization-format setting or daemon. The mission forbids a
daemon, and neither serialized format approaches 100 ms. A replacement model
must still produce 768 dimensions, be trained for 256-dimensional MRL, pass the
same CAST-chunked 40-query parity suite, and preserve offline portability.

The memory result is equally conclusive. Snowflake used 1.314 GB peak RSS in a
model-only smoke process and 1.640 GB after only 289 streamed Kubernetes chunks.
Disabling ONNX Runtime's CPU arena, memory pattern, and prepacking made the
model-only result worse at 1.367 GB. Nomic used 1.060 GB model-only, leaving no
space for the product and still failing the required 256d/768d parity gate.

The reviewed smaller Snowflake Arctic models have 22M/33M parameters but only
384-dimensional outputs, so they violate the fixed 768d remote contract.
EmbeddingGemma has the correct 768d/256d MRL contract but is a 308M-parameter
model, not an evidence-backed cold-start fix. Newer Granite 311M and mmBERT
307M options are also larger rather than startup candidates. A 149M USER2-base
model advertises 768d MRL, but it is Russian-focused and needs code/English
recall plus license/artifact review before any implementation work.

The first generic low-footprint architecture screened was Sentence Transformers'
Apache-2.0 `static-retrieval-mrl-en-v1`: a static token-embedding model with a
roughly 31.3 MB quantized ONNX artifact and 1024-dimensional retrieval output.
The published checkpoint's actual Matryoshka widths are 1024, 512, 256, 128,
64, and 32—not 768. The generic training recipe demonstrates that 768 can be
included, but that does not make an untrained 768-element slice of the released
checkpoint contract-safe. It is therefore an architecture candidate with two
honest paths: train a code-aware `[768, 256]` checkpoint, or explicitly change
the remote-width contract and benchmark the published model at a trained
width. Silent 1024-to-768 truncation is rejected. At its trained DuckDB 256d /
Elasticsearch 1024d widths, the frozen seed run missed parity by `0.027488`
nDCG and DuckDB hybrid relevance remained below lexical, so the published
general checkpoint was rejected before repository scale.

Potion Code 16M v2 resolves the footprint and domain mismatch at a native 256d
width. It was distilled for code retrieval from CodeRankEmbed using code query
and document pairs. Its released tensor is about 32.5 MB, and direct inference
avoids ONNX Runtime and a duplicate tokenizer dependency. The full repository
results were DuckDB nDCG `0.438550` / recall `0.875000` and Elasticsearch nDCG
`0.433076` / recall `0.887500`; aggregate parity passed at `0.005473` nDCG
drift. The Kubernetes full build used 988,889,088 bytes RSS, while unchanged
incremental reuse finished in 8.91 seconds.

## Evidence matrix

| Candidate | Lexical / substring | Vector | Rust / deployment | Main product risk | Verdict |
| --- | --- | --- | --- | --- | --- |
| Current DuckDB adapter | Hay's persisted BM25 term tables update transactionally. DuckDB's own FTS extension is not used because its index does not auto-update. | Exact cosine today. DuckDB VSS HNSW persistence is experimental and not recommended for production. | Primary Rust client; one local file; single read-write process. | Exact vector scan eventually stops scaling; multi-process writes are not the embedded concurrency model. | Supported floor and correctness oracle. |
| Qdrant Edge | Built-in offline BM25 sparse encoder, stemming/stopwords, word/prefix/whitespace/multilingual tokenizers, separate document/query paths. | Named dense and sparse vectors, quantization, local shard, universal query types. | Official in-process Rust crate; local directory; WAL; no service or network. | Pre-1.0 crate and newer operational surface; code-identifier recall must beat Hay's analyzer. | **Build a gated adapter spike next.** |
| LanceDB | BM25 FTS plus an FM index for arbitrary byte substrings. | Exact and IVF/HNSW variants with flat, scalar, product, and RabitQ quantization; RRF support. | Official embedded Rust SDK over Lance/Arrow. | Larger dependency/build footprint; Rust hybrid and incremental index-freshness behavior needs a lifecycle spike. | Strong runner-up if Qdrant Edge fails stability or recall gates. |
| SQLite FTS5 + sqlite-vec | Mature built-in BM25, weighted columns, prefixes, custom tokenizers, and trigram substring search. | `sqlite-vec` supports float/int8/binary vectors and Rust static linking, but is pre-v1 and currently brute-force only. | Small, single-file, ubiquitous. The current Rust registration example crosses an unsafe FFI boundary that would need an isolated audited crate. | No ANN today; FTS external-content tables require application-maintained consistency. | Best conservative small-device fallback, not the first large-corpus target. |
| Tantivy + vector sidecar | Native Rust inverted index, BM25 query support, n-gram and custom analyzer support. | Requires a second engine such as USearch or an exact vector slab. | Pure-Rust lexical core and excellent code-analysis control. | Two persistence domains make atomic update, recovery, compaction, and manifest coordination our responsibility. | Use if lexical recall/control dominates unified-store simplicity. |
| SurrealDB embedded | Integrated analyzers and BM25 full-text index. | HNSW and newer disk-backed DiskANN. | Rust embedded mode with memory or file-backed engines. | Much broader database and query-language surface than this library needs; version-3 index semantics are moving. | Watch, do not add now. |
| Qdrant server | BM25 sparse, text payload indexes, dense+sparse RRF/DBSF. | Mature vector server. | Rust client, but requires a separate process/container. | Violates the no-service embedded target. | Remote alternative only, not on-device. |

## Primary-source notes

- Snowflake's official model card documents the Apache-2.0 license, 768d
  output, query prefix, and 256d Matryoshka training:
  <https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0>
- Nomic's official model card documents Apache-2.0 licensing, retrieval
  prefixes, mean pooling, layer normalization, and supported Matryoshka widths:
  <https://huggingface.co/nomic-ai/nomic-embed-text-v1.5>
- ONNX Runtime documents Core ML model formats, model caching, and the limits
  of dynamic input shapes:
  <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>
- ONNX Runtime documents fixed/runtime `.ort` conversion and notes that full
  runtimes load both ONNX and ORT formats. Both paths were measured above:
  <https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html>
  and
  <https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-model-runtime-optimization.html>
- Snowflake's model family card records 384d outputs for its 22M and 33M
  variants and 768d for its 110M medium model:
  <https://huggingface.co/Snowflake/snowflake-arctic-embed-m>
- Google's EmbeddingGemma release documents 308M parameters, 768d output, and
  trained 512/256/128-dimensional MRL truncation:
  <https://huggingface.co/blog/embeddinggemma>
- USER2's model card documents its Russian focus and the 149M base model's
  768d/256d MRL option:
  <https://huggingface.co/onnx-community/USER2-small-ONNX>
- Sentence Transformers' static retrieval model card records its Apache-2.0
  license, 1024-dimensional output, and trained Matryoshka widths; the official
  static-embeddings article documents the architecture and training workflow:
  <https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1>
  and
  <https://huggingface.co/blog/static-embeddings>
- Potion Code 16M v2's model card records its MIT license, code-retrieval
  training sources, 256-dimensional static architecture, and benchmark scope:
  <https://huggingface.co/minishlab/potion-code-16M-v2>
- MinishLab's Rust implementation documents the no-special-token tokenizer,
  unknown-token removal, mean pooling, and optional L2 normalization semantics
  reproduced by Hay's smaller direct loader:
  <https://github.com/MinishLab/model2vec-rs>

- SQLite FTS5 documents built-in BM25, per-column weights, prefix indexes,
  custom tokenizers, and a trigram tokenizer for general substring matching:
  <https://www.sqlite.org/fts5.html>
- `sqlite-vec` documents Rust static linking and states that its current vector
  path is brute-force; its repository also warns that it is pre-v1:
  <https://alexgarcia.xyz/sqlite-vec/rust.html>,
  <https://alexgarcia.xyz/sqlite-vec/guides/binary-quant.html>, and
  <https://github.com/asg017/sqlite-vec>
- Qdrant Edge's official quickstart describes a local persisted shard, WAL,
  dense and sparse vectors, and Rust usage. Its BM25 guide documents the
  asymmetric document/query encoder and offline operation:
  <https://qdrant.tech/documentation/edge/edge-quickstart/> and
  <https://qdrant.tech/documentation/edge/edge-bm25/>
- The current Qdrant Edge Rust API is published as crate `0.7.2`:
  <https://docs.rs/qdrant-edge/latest/qdrant_edge/>
- LanceDB's Rust API exposes BM25 FTS, FM substring search, multiple vector
  indexes, and RRF building blocks:
  <https://docs.rs/lancedb/latest/lancedb/index/enum.Index.html> and
  <https://docs.lancedb.com/search/hybrid-search>
- Tantivy's Rust API documents committed searchable segments, custom
  tokenizers, stemming, and `NgramTokenizer`:
  <https://docs.rs/tantivy/latest/tantivy/> and
  <https://docs.rs/tantivy/latest/tantivy/tokenizer/>
- DuckDB officially warns that FTS does not auto-update and persistent VSS HNSW
  is experimental, must fit in RAM, and accumulates stale deleted entries:
  <https://duckdb.org/docs/lts/core_extensions/full_text_search> and
  <https://duckdb.org/docs/lts/core_extensions/vss>
- SurrealDB documents embedded Rust engines and integrated BM25, HNSW, and
  DiskANN indexes:
  <https://surrealdb.com/docs/reference/rust/embedding> and
  <https://surrealdb.com/docs/reference/query-language/statements/define/indexes>

## Qdrant Edge experiment

Build `hay-qdrant-edge` with lexical-only and hybrid configurations. Preserve
Hay's analyzer as one benchmark arm and Qdrant's built-in BM25 analyzer as the
other; do not silently change the frozen analyzer manifest.

The adapter must demonstrate:

1. create, bulk ingest, query, close, reopen, query;
2. idempotent upsert and delete with no stale lexical or dense match;
3. exact manifest rejection before every query;
4. deterministic document-ID tie breaking at Hay's boundary;
5. dense + BM25 candidate fusion through the shared contract;
6. bounded batches and measured peak RSS on the 89,183-CAST-chunk repository suite;
7. no network access after dependencies and models are present;
8. recovery after interruption between WAL update and optimization;
9. migration behavior between the exact pinned crate versions used in the
   experiment.

## Acceptance gates

Run the frozen WordPress, Django, Kubernetes, and Ollama judgments for DuckDB,
Elasticsearch, and the candidate adapter. Record aggregate and per-repository
nDCG@10, recall@50, MRR, cold/warm p50/p95 latency, index time, index bytes,
peak RSS, and update/delete latency.

A candidate becomes supported only when:

- recall@50 is not below DuckDB lexical by more than `0.01` absolute;
- nDCG@10 is not below DuckDB lexical by more than `0.02` absolute;
- every restart, stale-term, delete, and manifest test passes;
- peak RSS remains bounded under the configured indexing batch;
- no ignored external-service test is required for its embedded mode;
- dependency license and binary-size review is accepted;
- the adapter removes at least as much product complexity as it adds.

## N-gram policy

Do not make trigrams the only lexical index. They are useful for partial
identifiers, punctuation-heavy symbols, and substring lookup, but they expand
the term space and can dilute natural-language BM25. Benchmark them as a
separate candidate leg and fuse ranks with word/code BM25. The initial queries
should include camelCase, snake_case, short symbols, exact error text, file
paths, and natural-language paraphrases.
