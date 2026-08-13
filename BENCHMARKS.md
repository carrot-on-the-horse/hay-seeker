# Repository Benchmarks

Status: CAST-chunked lexical and code-static hybrid product cycles established
Snapshot date: 2026-08-10

## Frozen repositories

The suite uses immutable commits rather than moving branches:

| Corpus | Revision | Primary language |
| --- | --- | --- |
| WordPress `wordpress-develop` | `7b887ba4820e0ee87bbf3f14a0e8385b33f1a6fd` | PHP |
| Django | `dfc52e53f1d19a2730854d68b602fb4dba8bf0c5` | Python |
| Kubernetes | `4f5591ab57b75c0b8cabbff3031c9b956075c1ed` | Go |
| Ollama | `144893850fa778c8c81ff931f26614d62e6689c1` | Go |

The machine-readable source of truth is [benchmarks/repos.json](./benchmarks/repos.json). Checkouts live in ignored `.bench-repos/` storage and are not vendored into the crate.

```bash
./scripts/fetch-bench-repos.sh
```

The fetcher refuses to modify a dirty checkout and verifies every exact revision before evaluation.

## Recall suite

[evals/repos/queries.jsonl](./evals/repos/queries.jsonl) contains 40 natural
developer queries: ten per repository, with graded stable file IDs. The
production evaluator applies the legacy eligibility policy and exact product
`ChunkerV1` profile, producing 89,183 searchable CAST chunks from 29,772
eligible files. Results collapse deterministically by first-ranked chunk to the
parent file ID before scoring. This keeps judgments stable while measuring the
same retrieval unit shipped by the indexer.

```bash
cargo run --bin eval -- --backend local --suite repos
cargo run --bin eval -- --backend elastic --suite repos
cargo run --bin eval -- --backend duckdb --suite repos
cargo run --bin eval -- --backend elasticsearch --suite repos \
  --endpoint https://search.example.com --index hay-seeker-eval
```

Add `--embeddings gemini` or `--embeddings open-ai` to either production
backend to measure the shared hybrid cascade with the same provider and exact
manifest used by `hay` and `hay-mcp`. Both are intentionally opt-in because
they transmit the complete evaluation corpus and queries externally. OpenAI
can call the official API directly or use the provider-native
`/v1/{account}/{gateway}/openai/embeddings` Cloudflare route. Use a disposable
DuckDB path or Elasticsearch alias because the harness rebuilds it.

The harness validates checkout revisions, rejects missing graded paths and
unknown returned chunks, and reports aggregate and per-repository nDCG@10,
recall@50, and MRR.

Historical whole-file deterministic random-stub baseline for both backend labels:

```text
documents: 30587
queries: 40
nDCG@10: 0.000000
recall@50: 0.000000
MRR: 0.000000
```

Zero was expected for the historical whole-file random floor. The current
chunked stub has a different candidate cardinality and collapses repeated file
hits; it remains only a harness check, not relevance evidence.

### CAST-chunked lexical parity

The 2026-08-07 product-corrected run indexed 89,183 chunks from 29,772 files
through both production adapters. Fifty-five blank chunks were discarded. File
eligibility skipped 625 empty files, one invalid-UTF-8 fixture, 188 oversized
config/data files, and one oversized source file.

| Backend / corpus | Queries | nDCG@10 | Recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| DuckDB / all | 40 | 0.434779 | 0.887500 | 0.364071 |
| Elasticsearch / all | 40 | 0.433320 | 0.887500 | 0.362592 |
| DuckDB / Django | 10 | 0.664871 | 1.000000 | 0.590238 |
| DuckDB / Kubernetes | 10 | 0.340489 | 0.800000 | 0.281513 |
| DuckDB / Ollama | 10 | 0.396358 | 0.850000 | 0.315996 |
| DuckDB / WordPress | 10 | 0.337399 | 0.900000 | 0.268535 |

Aggregate nDCG differs by 0.001459, recall is identical, and the hard 0.02
parity gate passes. The release evaluator completed both targets in 129.35
seconds against Elasticsearch 9.3.2. It peaked at 1,095,155,712 bytes RSS while
holding the complete four-repository corpus in memory; this is harness
telemetry, not evidence for the separately streamed Kubernetes product-index
memory gate.

A prior dense attempt over 29,772 whole-file vectors was stopped after 20.7
minutes at only 2,888 files and 2.69 GB maximum RSS. It is rejected evidence:
the path bypassed CAST, averaged files as large as 5 MiB into one vector, and
did not represent shipped query semantics.

### Historical whole-file DuckDB BM25 diagnostic

The 2026-08-06 production-adapter run applied the legacy eligibility policy
before indexing: blank files, invalid UTF-8, configs/data above 50 KiB, and
sources above 5 MiB were skipped with explicit counters. It indexed 29,772
files and reported:

| Corpus | Queries | nDCG@10 | Recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| All | 40 | 0.443676 | 0.887500 | 0.364366 |
| Django | 10 | 0.734125 | 1.000000 | 0.645000 |
| Kubernetes | 10 | 0.312533 | 0.900000 | 0.280274 |
| Ollama | 10 | 0.374457 | 0.850000 | 0.283382 |
| WordPress | 10 | 0.353588 | 0.800000 | 0.248810 |

Skip counts were 625 empty files, one invalid-UTF-8 fixture, 188 oversized
config/data files, and one oversized source file. The full rebuild uses
DuckDB's bulk appender and persisted Hay BM25 term tables. These are file-level
documents from the superseded evaluator and remain historical diagnostics, not
the current product retrieval baseline.

### Historical whole-file Elasticsearch BM25 diagnostic

The same 29,772 eligible files and 40 frozen judgments were run against an
official Elasticsearch 9.4.2 node using the production adapter. The adapter
used bounded 5 MiB bulk requests, a final explicit refresh, and an atomic alias
swap:

| Corpus | Queries | nDCG@10 | Recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| All | 40 | 0.446111 | 0.887500 | 0.367461 |
| Django | 10 | 0.721032 | 1.000000 | 0.628333 |
| Kubernetes | 10 | 0.312533 | 0.900000 | 0.280131 |
| Ollama | 10 | 0.384200 | 0.850000 | 0.295356 |
| WordPress | 10 | 0.366681 | 0.800000 | 0.266026 |

Aggregate nDCG@10 differs from DuckDB by 0.002435, comfortably inside the
0.02 parity gate, and recall@50 is identical. The 60-document seed suite also
passed with DuckDB nDCG@10 0.808138 versus Elasticsearch 0.796605, a difference
of 0.011533; both returned recall@50 0.937500. These results establish lexical
backend parity, not dense or chunk-level parity.

Generation retention was verified separately against the same cached 9.4.2
Basic-license image on 2026-08-06. Two consecutive executions of the live
create/update/failure/dense cycle left exactly two lexical physical indices
and two dense physical indices per alias. The active alias pointed at the
newest lexical generation; failed builds did not publish or remove the prior
target. This exercises the third-generation pruning path, not only its JSON
request builder.

The one-command production parity path re-ran the 60-document seed lexical
suite against a disposable Elasticsearch 9.4.2 node and DuckDB index. It
reported DuckDB nDCG@10 0.808138, recall@50 0.937500, and MRR 0.803993 versus
Elasticsearch 0.796605, 0.937500, and 0.788368. The hard nDCG delta gate passed
at 0.011533; recall delta was zero. A real-model hosted run remains an explicit
data-egress gate; a rejected or unapproved external call is not reported as
recall evidence.

### DuckDB repository-build memory gate

The release indexer must build the largest pinned repository with peak RSS
below 1 GiB. On macOS, the reproducible command is:

```bash
cargo build --release -p hay-cli
/usr/bin/time -l target/release/hay index \
  --backend duckdb --embeddings none \
  --repository .bench-repos/kubernetes \
  --database /private/tmp/hay-kubernetes.duckdb \
  --checkpoint /private/tmp/hay-kubernetes.checkpoint.json
```

At Kubernetes revision `4f5591ab57b75c0b8cabbff3031c9b956075c1ed`,
the 2026-08-06 clean build produced 79,949 chunks from 26,366 files:

| Storage profile | Peak RSS | Total time | Database size |
| --- | ---: | ---: | ---: |
| DuckDB defaults plus compound document/term primary key | 3.56 GB | 163.4 s | not retained |
| Hay embedded profile, no redundant compound ART index | 787 MB | 160.0 s | 384 MiB |

The accepted profile limits DuckDB's buffer manager to 512 MB, uses two
threads, disables insertion-order preservation, and caps buffered row groups at
32 MB. The term table deliberately has no `(document_id, term)` primary key:
terms are already unique per document when emitted from the analyzer, while
the redundant compound ART index was not useful to the term-first BM25 join
and was not governed by DuckDB's buffer-manager limit. Blank CAST chunks are
discarded and reported; this corpus contained 22.

Indexes created before this schema change remain readable, but retain their old
compound index until the next full rebuild.

## DuckDB int8 dense scan

Dense DuckDB indexes with the offline encoder store 256-dimensional vectors as
a versioned int8 blob with one float32 scale and offset per vector. The exact
Rust scan prepares the query norm once, dequantizes each component while
accumulating cosine dot/norm values, and uses document ID as the deterministic
score tie-break. No ANN index is present.

```bash
cargo bench -p hay-duckdb --bench dense_scan -- --noplot
```

The 2026-08-06 Apple Silicon run measured:

| Path | Vectors | Dimensions | Median time | Throughput |
| --- | ---: | ---: | ---: | ---: |
| In-memory kernel | 30,000 | 256 | 3.322 ms | 2.31B elements/s |
| In-memory kernel | 250,000 | 256 | 27.761 ms | 2.31B elements/s |
| Full DuckDB query | 30,000 | 256 | 9.671 ms | 794M elements/s |

The full-query arm reads every persisted blob through DuckDB, scores it, sorts
with the production tie-break, and returns the top 10 from 50 candidates. It
excludes model encoding time, so it proves the vector-storage/scan portion is
comfortably below the 100 ms query budget. The measured local-model encoding
and process-startup costs are recorded below.

The model-side acceptance tools use generated strings and activate when
`COTH_HAY_SEEKER_LOCAL_MODEL_DIR` points to a reviewed bundle:

```bash
cargo run -p cast-embeddings --example local_onnx_smoke
cargo bench -p cast-embeddings --bench local_encode -- --noplot
```

The smoke command reports cold session startup, first/warm query latency,
Core ML policy diagnostics, output dimensions/norm, and one eight-document
batch. The Criterion suite measures warm document batches of one and eight plus
query encoding. Missing model artifacts are reported as a skipped benchmark,
not replaced by a hosted provider.

### Offline model comparison

Both measured profiles are Apache-2.0 and use official checksum-pinned ONNX
artifacts. The runs used ONNX Runtime CPU because the dynamic quantized graphs
are not suitable Core ML production candidates. Times are Apple Silicon release
builds; document batches contain short generated inputs.

| Model | Session startup | First query | Warm query | 8 documents | Seed 256d/768d parity |
| --- | ---: | ---: | ---: | ---: | --- |
| Nomic Embed Text v1.5 | 396 ms | 5.98 ms | 4.54 ms | 26.4 ms | Failed: 0.027223 nDCG delta |
| Snowflake Arctic Embed m v2 | 900 ms | 6.19 ms | 6.15 ms | 22.15 ms | Passed: 0.009678 nDCG delta |

The Nomic failure was isolated to MRL truncation: a diagnostic 768d/768d run
passed at a 0.000731 nDCG delta, while the required DuckDB 256d / Elasticsearch
768d pair exceeded the 0.02 gate. It is retained as a supported research
profile, not the current acceptance model.

Snowflake cold loading was also tested with two offline serializations using
the same graph, tokenizer, prompts, pooling, and checksum enforcement:

| Snowflake artifact | Product startup | First query | 8 documents | Verdict |
| --- | ---: | ---: | ---: | --- |
| Original dynamic-int8 ONNX | 867.4 ms | 6.16 ms | 22.59 ms | Baseline; fails cold gate |
| ONNX Runtime fully optimized ONNX | 826.9 ms | 5.64 ms | 22.08 ms | Reject; only 4.7% faster startup |
| ONNX Runtime fixed CPU `.ort` | 851.8 ms | 6.11 ms | 22.72 ms | Reject; only 1.8% faster startup |

The official ORT conversion and optimized ONNX outputs were disposable
experiments and were deleted after measurement. Neither changes inference
throughput materially or approaches the required cold query under 100 ms, so
no artifact-format knob was added to the product.

Snowflake Arctic produced these frozen seed metrics:

| Backend | Width / storage | nDCG@10 | Recall@50 | MRR |
| --- | --- | ---: | ---: | ---: |
| DuckDB | 256d per-vector int8 | 0.880102 | 1.000000 | 0.913690 |
| Elasticsearch | 768d BBQ | 0.870425 | 1.000000 | 0.900670 |

Compared with the lexical seed baselines, hybrid nDCG gained 0.071964 on
DuckDB and 0.073820 on Elasticsearch. The full 40-query repository dense run
remains open; seed parity is not substituted for broader acceptance evidence.

### Code-trained static hybrid profile

The low-footprint product candidate is the MIT-licensed
`minishlab/potion-code-16M-v2` static table at immutable revision
`e9d2a44ca6a05ac6685f3b23709ea57eb7352d5b`. Hay verifies all artifacts and
implements the published no-special-token, unknown-token removal, mean-pooling,
and normalization contract directly in safe Rust. The model's only trained
width is 256 dimensions, so both backends use 256d; this is an explicit profile
change from the original 256d/768d ONNX contract.

Release smoke on Apple Silicon:

| Measurement | Result |
| --- | ---: |
| Verified cold initialization | 87.59 ms |
| First query | 0.269 ms |
| Warm query | 0.019 ms |
| Eight short documents | 0.607 ms |
| Generated 5.2 KiB document | 0.868 ms |
| Model smoke peak RSS | 136,396,800 bytes |

A separate cold CLI process over the completed 79,949-chunk Kubernetes DuckDB
index took 190 ms and peaked at 302,612,480 bytes RSS. The static encoder clears
the model-startup screen, but the actual end-to-end process still misses the
strict 100 ms cold-query gate. Warm in-process use is comfortably below it;
startup and index-open costs remain an explicit optimization target.

The frozen 60-document seed suite passed with identical results on DuckDB int8
and Elasticsearch BBQ: nDCG@10 `0.862000`, recall@50 `1.000000`, and MRR
`0.873214`. That is a `0.053862` DuckDB gain over the lexical seed baseline.

The product-correct 89,183-CAST-chunk repository suite embedded the complete
corpus once in 7.171 seconds and passed the hard backend-parity gate:

| Backend / corpus | Queries | nDCG@10 | Recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| DuckDB / all | 40 | 0.438550 | 0.875000 | 0.364914 |
| Elasticsearch / all | 40 | 0.433076 | 0.887500 | 0.367426 |
| DuckDB / Django | 10 | 0.580775 | 0.900000 | 0.519167 |
| DuckDB / Kubernetes | 10 | 0.336783 | 0.750000 | 0.305765 |
| DuckDB / Ollama | 10 | 0.376581 | 0.850000 | 0.301902 |
| DuckDB / WordPress | 10 | 0.460061 | 1.000000 | 0.332821 |

Aggregate nDCG drift was `0.005473`, recall drift `0.012500`, and MRR drift
`0.002512`. The largest per-repository nDCG drift was Kubernetes at `0.028906`;
the frozen hard gate is aggregate, so the run passes while retaining that local
warning. Against lexical, DuckDB gained `0.003771` nDCG but lost `0.0125`
recall; Elasticsearch was effectively flat (`-0.000244` nDCG, equal recall).
The result supports a fast hybrid option, not a claim that dense retrieval wins
uniformly on this small 40-query repository suite. The evaluator peaked at
1,456,816,128 bytes RSS because it holds the whole corpus and both result sets;
that is harness telemetry, not the streamed product memory gate.

### Static versus OpenAI comparison gate

The evaluator can run the same frozen documents, queries, chunker, DuckDB
backend, and metrics with Potion or OpenAI. Use separate databases because the
embedding profile and route are manifest-bound:

```bash
COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR=/path/to/reviewed/potion-bundle \
  cargo run --release -p hay-eval -- \
  --backend duckdb --embeddings local-static --suite repos \
  --database /path/to/disposable-static.duckdb

COTH_HAY_SEEKER_CF_AIG_TOKEN=... \
COTH_HAY_SEEKER_OPENAI_API_KEY=... \
COTH_HAY_SEEKER_OPENAI_GATEWAY_URL=https://gateway.ai.cloudflare.com/v1/account/gateway/openai/embeddings \
COTH_HAY_SEEKER_OPENAI_MODEL_REVISION=approved-revision \
  cargo run --release -p hay-eval -- \
  --backend duckdb --embeddings open-ai --suite repos \
  --database /path/to/disposable-openai.duckdb
```

The example uses Cloudflare AI Gateway. Remove `COTH_HAY_SEEKER_OPENAI_GATEWAY_URL` and the
Gateway token to benchmark the direct API with only
`COTH_HAY_SEEKER_OPENAI_API_KEY`. For gateway BYOK or Unified Billing, omit the
OpenAI key instead. Record nDCG@10, recall@50, MRR, total indexing and query
time, provider or Gateway request/token counts and cost, and peak RSS. No live
OpenAI result is committed yet, so the Potion measurements above remain
standalone evidence rather than a claimed head-to-head win.

### Controlled ChunkHound 5.2.1 comparison

The competitor run pins [ChunkHound 5.2.1](https://pypi.org/project/chunkhound/)
and uses its stable DuckDB backend. Both systems receive vectors from Hay's
exact checksum-pinned Potion table. A benchmark-only loopback OpenAI adapter
exposes `LocalStaticEmbedder`; it performs no external network access and accepts the
`text-embedding-3-small` compatibility name so ChunkHound applies its supported
OpenAI sizing fallback. The returned vectors are still the exact 256-dimensional
Potion vectors. This follows the project rule to use an OpenAI tokenizer when a
model-specific tokenizer integration is unavailable.

This controls the model, source revisions, file-size policy, Git ignore policy,
queries, and file-level judgments. It does **not** pretend the retrieval
cascades are identical: Hay runs BM25 plus dense RRF, while ChunkHound runs its
available no-reranker `--single-hop` semantic mode. ChunkHound results are
collapsed from ranked chunks to the first occurrence of each file before the
same nDCG@10, recall@50, and MRR functions are applied.

Reproducible setup:

```bash
uv venv /private/tmp/hay-chunkhound-5.2.1 --python 3.12
UV_CACHE_DIR=/private/tmp/hay-chunkhound-uv-cache \
  uv pip install --python /private/tmp/hay-chunkhound-5.2.1/bin/python \
  'chunkhound==5.2.1'

COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR=/path/to/reviewed/potion-bundle \
  cargo run --release -p cast-embeddings --example openai_static_server

/private/tmp/hay-chunkhound-5.2.1/bin/chunkhound index \
  .bench-repos/ollama \
  --config benchmarks/chunkhound-5.2.1.json \
  --db /private/tmp/hay-chunkhound-bench-5.2.1/ollama.db \
  --discovery-backend git_only

cargo run --release --bin eval -- \
  --backend chunkhound --suite repos --chunkhound-repository ollama \
  --chunkhound-bin /private/tmp/hay-chunkhound-5.2.1/bin/chunkhound \
  --chunkhound-config benchmarks/chunkhound-5.2.1.json \
  --chunkhound-db-root /private/tmp/hay-chunkhound-bench-5.2.1
```

The evaluator refuses other ChunkHound versions and validates every frozen Git
revision and positively judged path. The Rust loopback adapter and config are
benchmark tooling, not a supported hosted-provider shim.

Three corpora completed and produced relevance scores. Kubernetes could not be
scored because the competitor index hit the explicit disk safety guard:

| Corpus / retriever | Queries | nDCG@10 | Recall@50 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Django / Hay hybrid | 10 | 0.580775 | 0.900000 | 0.519167 |
| Django / ChunkHound semantic | 10 | 0.293196 | 0.800000 | 0.277434 |
| Ollama / Hay hybrid | 10 | 0.376581 | 0.850000 | 0.301902 |
| Ollama / ChunkHound semantic | 10 | 0.038685 | 0.550000 | 0.043583 |
| WordPress / Hay hybrid | 10 | 0.460061 | 1.000000 | 0.332821 |
| WordPress / ChunkHound semantic | 10 | 0.411006 | 1.000000 | 0.303021 |

Across the 30 completed queries, Hay's macro-average was nDCG@10 `0.472472`,
recall@50 `0.916667`, and MRR `0.384630`. ChunkHound's was nDCG@10 `0.247629`,
recall@50 `0.783333`, and MRR `0.208013`. Hay therefore led by `0.224843`
nDCG, `0.133334` recall, and `0.176617` MRR on the safely completed subset.
This is a hybrid-versus-semantic product comparison under a controlled model,
not an attribution of the entire difference to chunking or vector storage.

Single clean-index process measurements on the same Apple Silicon host:

| Corpus | Processed files | Chunks | Index time | Peak RSS | Final DB disk use |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ollama | 1,006 | 54,779 | 390.17 s | 3,077,226,496 | 328 MiB |
| Django | 4,296 | 133,858 | 839.58 s | 3,800,154,112 | 499 MiB |
| WordPress | 4,833 | 124,002 | 915.47 s | 3,794,026,496 | 515 MiB |
| Kubernetes | Not completed | Not reported | >5,161.02 s | 9,829,793,792 | No usable index |

The final compacted WordPress file is 515 MiB. DuckDB temporarily reached 4.6
GiB during ChunkHound's end-of-run compaction, so transient disk consumption is
material even though the final file is much smaller. These are one-shot system
measurements rather than confidence intervals.

Kubernetes was stopped after 86.0 minutes when the temporary DuckDB file
crossed the predeclared 120 GB guard, reaching 155 GB while the process was
active and 164 GB by cleanup. Peak RSS was 9,829,793,792 bytes with no swaps.
The exact aborted temporary database and 101-byte root sidecar were then
deleted, restoring 164 GB of workstation disk. ChunkHound repeatedly retained
or held open `compact_backup`/replacement files while rewriting the database;
the visible path oscillated from low single-digit GB to tens of GB. This is a
measured resource failure, so no Kubernetes recall is invented from a partial
index. By comparison, Hay completed the same frozen Kubernetes corpus in
198.76 seconds at 988,889,088 bytes RSS and produced 79,949 chunks.

Query latency separates a fresh CLI process from a persistent service. The
steady-state numbers exclude the first query after service construction:

| Corpus | Cold process p50 / p95 | Persistent first | Steady p50 / p95 |
| --- | ---: | ---: | ---: |
| Ollama | 2,642 / 2,750 ms | 272 ms | 79 / 82 ms |
| Django | 3,321 / 5,758 ms | 339 ms | 153 / 160 ms |
| WordPress | 2,985 / 6,686 ms | 353 ms | 149 / 156 ms |

The cold harness launches the official CLI once per query. The persistent
measurement constructs ChunkHound's service layer once and requests 200 chunks
per query, matching the first page used to collect 50 unique judged files.

Hay's read-only reuse run over the existing 89,183-chunk, four-repository index
reported warm hybrid p50 `89.725` ms and p95 `103.073` ms across all 40
queries. It reproduced nDCG@10 `0.438550`, recall@50 `0.875000`, and MRR
`0.364914` without replacing the index. The combined 27 steady-state
ChunkHound samples from the three completed corpora were p50 `147.060` ms and
p95 `158.089` ms. ChunkHound was faster on Ollama alone, but Hay was faster on
Django and WordPress and across the controlled completed subset. Hay's separate
actual cold CLI sample was 190 ms; it is not promoted to a p50/p95 distribution.

```bash
COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR=/path/to/reviewed/potion-bundle \
  cargo run --release --bin eval -- \
  --backend duckdb --embeddings local-static --suite repos \
  --database /path/to/completed/repos.duckdb --reuse-duckdb-index
```

`--reuse-duckdb-index` is deliberately restricted to the DuckDB evaluator and
fails when the file is missing. The normal evaluator remains rebuild-by-default.

ChunkHound's unchanged Ollama cycle discovered 1,009 files, processed zero,
and finished in 2.68 seconds (3.54 seconds wall) at 1,588,215,808 bytes peak
RSS. A controlled one-line edit caused exactly one file to be processed and 61
chunks to be embedded in 6.43 seconds (7.56 seconds wall) at 1,980,039,168
bytes RSS. The edit was reversed and the database re-indexed back to the exact
frozen commit. This verifies functional incrementality while exposing a much
larger incremental memory floor than Hay's 432,832,512-byte unchanged
Kubernetes cycle.

### Dense Kubernetes resource gate

The product CLI was run against the frozen Kubernetes checkout with the
Snowflake profile, DuckDB persistence, resumable checkpoints, progress
reporting, and `/usr/bin/time -l`. The run was stopped after 289 chunks across
six files because the hard failure was already measurable:

| Measurement | Peak RSS | Verdict |
| --- | ---: | --- |
| Snowflake model-only smoke | 1,313,685,504 bytes | Fails the complete product's 1 GiB budget before DuckDB opens |
| Snowflake streamed Kubernetes build at 289 chunks | 1,640,480,768 bytes | Fails; stopped instead of spending hours on a result that cannot pass |
| Snowflake model-only with CPU arena, memory pattern, and prepacking disabled | 1,366,638,592 bytes | Reject; memory increased and throughput fell |
| Nomic model-only comparison | 1,059,651,584 bytes | Barely below 1 GiB but leaves no product headroom and already fails required 256d/768d parity |
| Potion static model-only smoke | 136,396,800 bytes | Passes cold and model-memory screens |
| Potion streamed Kubernetes full build | 988,889,088 bytes | Passes at 79,949 chunks, with limited headroom |
| Potion unchanged Kubernetes incremental build | 432,832,512 bytes | Passes; reused all 79,949 chunks in 8.91 s |

No partial database or checkpoint was retained. The model-only decomposition
shows that DuckDB buffer limits, streaming batch size, and index transaction
strategy cannot make the current Snowflake checkpoint pass. A smaller encoder
must first pass model-only cold-start and RSS screening. Potion is the first
screened candidate to pass: the clean Kubernetes hybrid build completed in
198.76 seconds from 26,366 files and 215,856,679 source bytes. Peak RSS was only
84,852,736 bytes below the 1 GiB ceiling, so future grammar/model/index growth
must keep this gate running in CI or release qualification.

The product profile does not silently truncate the reviewed 1,500-token CAST
chunks. It encodes documents in 256-token model windows with 32-token overlap,
normalizes each window, takes a non-padding-token-weighted mean at 768d, then
applies the backend's MRL projection. On the current 194-chunk/96-file Hay
repository, a clean DuckDB dense build took 33.2 seconds versus 47.2 seconds
for the legacy 2,048-token first-window profile; an intermediate 512-token
window took 42.0 seconds. This is a 29.7% improvement while covering the whole
chunk. Increasing the ONNX microbatch from 8 to 16 improved only another 1.1%
and was rejected because it weakens the still-open Kubernetes RSS gate.

Length-sorted microbatches reduced Nomic's equivalent earlier run from 69.1 to
49.4 seconds. Even the preferred Arctic profile remains too slow for an
economical full 79,949-chunk Kubernetes run, and its measured memory is already
a hard failure. Repository-scale relevance remains open rather than inferred
from the seed.

Criterion for the final 256/32 profile measured 3.456-3.505 ms for one short
document, 20.952-21.049 ms for eight (380-382 documents/s), 4.644-4.675 ms for
a query, and 274.2-297.8 ms for the generated 5.2 KiB windowed document.

## Chunking throughput

Criterion measures one representative production file from each repository:

```bash
cargo bench -p hay-search --bench repository_chunking
```

Set `COTH_HAY_SEEKER_BENCH_REPOS` when the checkouts are elsewhere. A ten-sample development
run of the earlier generic fixed-window path produced:

| Corpus file | Median time | Approximate throughput |
| --- | ---: | ---: |
| WordPress `class-wp-query.php` | 77.3 us | 1.98 GiB/s |
| Django `query.py` | 53.2 us | 2.08 GiB/s |
| Kubernetes `kubelet.go` | 67.6 us | 2.15 GiB/s |
| Ollama `routes.go` | 43.6 us | 1.92 GiB/s |

The multi-language CAST implementation now produces:

| Corpus file | Median time | Approximate throughput |
| --- | ---: | ---: |
| WordPress `class-wp-query.php` | 16.5 ms | 9.48 MiB/s |
| Django `query.py` | 10.1 ms | 11.2 MiB/s |
| Kubernetes `kubelet.go` | 15.2 ms | 9.79 MiB/s |
| Ollama `routes.go` | 11.8 ms | 7.26 MiB/s |

The benchmark asserts that every representative file uses its compiled PHP,
Python, or Go grammar. All four files parsed without recovery diagnostics or
degraded splits. The generic numbers are retained only as historical context;
they are not directly comparable because they did not parse or traverse an AST.

## Tokenizer throughput

The stable suite compares the old Go-compatible Unicode word approximation to
the production fallback, OpenAI `o200k_base` through `tiktoken-rs` 0.12.0:

```bash
./scripts/bench-tokenizers.sh
```

Ten-sample development run:

| Corpus file | Unicode words | OpenAI `o200k_base` |
| --- | ---: | ---: |
| WordPress `class-wp-query.php` | 214.77 us / 728.88 MiB/s | 6.6995 ms / 23.366 MiB/s |
| Django `query.py` | 102.92 us / 1.0774 GiB/s | 3.3637 ms / 33.758 MiB/s |
| Kubernetes `kubelet.go` | 141.36 us / 1.0269 GiB/s | 4.4016 ms / 33.770 MiB/s |
| Ollama `routes.go` | 69.033 us / 1.2128 GiB/s | 2.9031 ms / 29.531 MiB/s |

The word counter is only an approximation and cannot enforce a model context
window. The BPE result is the default when an exact model tokenizer is
unavailable.

The isolated nightly suite compares identical `o200k_base` tokenization in the
stable implementation and GigaToken. It first asserts full token-ID equality
for every source file. `gigatoken_warm` reuses the worker-local cache;
`gigatoken_cold_cache` starts each measured encode with GigaToken's freshly
seeded cache while keeping the model tables loaded. Fork construction is
Criterion setup and is excluded from the timed encode.

| Corpus file | `tiktoken-rs` | GigaToken warm | GigaToken cold cache |
| --- | ---: | ---: | ---: |
| WordPress | 5.5463 ms | 194.06 us (28.6x) | 331.14 us (16.8x) |
| Django | 3.2062 ms | 112.41 us (28.5x) | 231.82 us (13.8x) |
| Kubernetes | 4.2966 ms | 164.06 us (26.2x) | 390.66 us (11.0x) |
| Ollama | 2.9736 ms | 99.964 us (29.7x) | 217.48 us (13.7x) |

These results pin GigaToken revision
`fac0114b37120ec8a76362e9ee8e1c742aaafaef` and the OpenAI rank artifact SHA-256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.
GigaToken remains benchmark-only because it requires nightly Rust; it is not a
production dependency of the stable workspace.

## Cloudflare/Gemini repository smoke test

The generated Go fixture exercises the complete CAST chunking, bounded Gemini
batch, query embedding, and cosine-ranking path without transmitting local or
third-party repository source:

```bash
COTH_HAY_SEEKER_GEMINI_SMOKE_CHUNK_TOKENS=80 cargo run -p cast-embeddings \
  --example repo_smoke -- \
  crates/cast-embeddings/tests/fixtures/synthetic_routes.go \
  "where are API routes registered?"
```

The 2026-08-06 live run produced 9 chunks and 9 document vectors plus one query
vector, all at 768 dimensions, in 1.62 seconds. Route-registration code ranked
in the top two results. This validates request authentication, response shape,
bounded concurrency, output ordering, and vector scoring; it is not a recall or
relevance benchmark.

The same date's synthetic-only Cloudflare Workers AI smoke exercised Qwen3's
native document and code-query modes through the shared adapter. Both vectors
had 1,024 dimensions and cosine similarity 0.644688. The fixture contains only
generated route-registration text; no repository contents were transmitted.

```bash
cargo run -p cast-embeddings --example workers_ai_smoke
```
