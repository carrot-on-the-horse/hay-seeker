# Product acceptance ledger

Status date: 2026-08-10

This ledger maps the original mission to current executable evidence. A row is
complete only when the stated acceptance scope is directly measured; passing a
smaller unit test does not close a broader product gate.

| Requirement | Status | Current evidence / missing proof |
| --- | --- | --- |
| Rust embedded library, DuckDB and Elasticsearch targets | Complete | `hay-search::Retriever` is implemented by `DuckDbIndex` and `ElasticsearchIndex`; the CLI and MCP process compose either without application-side query branching. |
| Fully on-device local path; no hosted dependency | Complete | The code-trained Potion static bundle is checksum-pinned and opened directly by safe Rust with no HTTP or download path. It completed the full frozen Kubernetes build and served hybrid queries locally. The ONNX research path remains available separately. |
| Same pinned checkpoint; only 256 local / 768 remote MRL divergence | Complete for seed, repository proof pending | The Snowflake Arctic seed parity run used one pinned graph and tokenizer: DuckDB stored 256d int8 and Elasticsearch stored 768d BBQ. DuckDB nDCG@10 was 0.880102 versus Elasticsearch 0.870425, a 0.009678 delta. Both returned recall@50 1.0. The executable manifest-pair validator rejects every other drift. |
| Exact manifest and hard `reindex required` failure | Complete | Every manifest field is compared, mismatch tests cover every field, and both production adapters validate before query/update. The local bundle revision includes its manifest SHA-256. The chunker identity now pins every AST/fixed-window setting, tokenizer implementation/artifact, and compiled grammar version; the repository stream rejects an identity that differs from its executable chunker. |
| Common query/candidate/signals/capabilities contract | Complete | The object-safe `Retriever` contract and deterministic RRF boundary are shared by both adapters and MCP. |
| Deterministic cascade and tie-break | Complete for lexical+dense | Both targets generate BM25+dense candidates and fuse with RRF `k=60`; final ordering uses score then document ID. DuckDB reports its int8 stage, Elasticsearch reports BBQ. Late/sparse/FDE correctly remain unavailable. |
| Phase 0 harness and at least 30 queries | Complete | Seed suite has 32 queries; frozen repository suite has 40 queries across WordPress, Django, Kubernetes, and Ollama and reports nDCG@10, recall@50, and MRR. Repository sources now pass through the exact product `ChunkerV1`; chunk results collapse deterministically to frozen parent-file judgments. |
| Code-aware lexical retrieval | Complete with approved DuckDB substitution | User selected DuckDB as the first embedded target. Hay persists code-aware BM25 terms, retaining originals and camel/snake components; incremental stale-term tests pass. |
| Local 256d int8 exact scan; no ANN through 250k | Complete for storage/scan | Versioned per-vector scale+offset codec, malformed-blob tests, transactional persistence, and exact deterministic scan are implemented. Criterion: 3.322 ms at 30k, 27.761 ms at 250k; full 30k DuckDB query 9.671 ms. |
| Clear Phase 1 relevance gain | Complete for seed; mixed at repository scale | Potion hybrid seed nDCG@10 was 0.862000 versus DuckDB lexical 0.808138, a 0.053862 gain. On the 40-query repository suite DuckDB gained 0.003771 nDCG but lost 0.0125 recall; Elasticsearch was effectively flat. The evidence supports an optional hybrid profile, not uniform dense superiority. |
| Cold end-to-end query under 100 ms on Apple Silicon | Failed end-to-end; encoder now passes | Potion verified model initialization was 87.59 ms and warm query encoding 0.019 ms, but a cold CLI search over the complete Kubernetes DuckDB index took 190 ms and 302,612,480 bytes RSS. Snowflake remains rejected at 867.4 ms model startup. The remaining startup/open/query overhead must be profiled without introducing a daemon. |
| Elasticsearch parity within 0.02 nDCG@10 | Complete for lexical and Potion repository hybrid | The product-correct Potion repository run reported DuckDB nDCG 0.438550 and Elasticsearch 0.433076, an aggregate delta of 0.005473. Recall delta was 0.0125 and MRR delta 0.002512. The seed run was identical across backends. Kubernetes local nDCG drift was 0.028906 and remains a warning beyond the aggregate hard gate. |
| Incremental, resumable, crash-safe, progress reporting | Complete | Content-hash checkpoints, unchanged reuse, changed/deleted handling, atomic DuckDB staging, Elasticsearch copy-on-write alias swap, bounded physical-generation retention, and ambiguous-publication reconciliation are present alongside progress and stall aborts. DuckDB's manifest-scoped vector cache survives process/branch changes, keys exact ID plus text, and fails closed on corrupt rows. Local ONNX document tokenizer/inference batches run on libdispatch's Utility QoS queue on macOS without permanently demoting the caller thread; query encoding retains caller priority. |
| Indexing below 1 GiB on largest frozen repository | Complete for Potion, failed for Snowflake | Potion indexed all 79,949 Kubernetes chunks in 198.76 seconds at 988,889,088 bytes peak RSS. An unchanged incremental run reused every chunk in 8.91 seconds at 432,832,512 bytes. This passes with only 84,852,736 bytes headroom. Snowflake remains rejected at 1.640 GB after 289 chunks. |
| Criterion encode, brute-force, MaxSim benches | Partial | Snowflake Arctic Criterion measured 4.64-4.67 ms queries, 20.95-21.05 ms for eight short documents, and 274-298 ms for a generated 5.2 KiB document through the complete 256-token-window profile. Nomic Criterion measured 4.57-4.65 ms queries and 25.38-27.02 ms eight-document batches. Brute-force scan is measured. MaxSim is Phase 3 and remains gated by repository-scale dense parity. |
| Phase 3 late interaction | Correctly gated | Not started. Potion passes parity/resources but its 40-query hybrid gain is marginal and mixed, so adding a costlier cascade stage is not yet justified. |
| Phase 4 learned sparse | Correctly gated | Not started. |
| Phase 5 MUVERA FDE | Correctly gated | Not started. |
| Full code-standard audit | Partial | Rust 2024, forbidden unsafe, strict Clippy, and library `thiserror`/contract errors pass current checks. A repository-wide AST regression test rejects `unwrap`/`expect` outside `#[cfg(test)]`, `#[test]`, and the actual `main` function. Every library crate denies missing public documentation and has a compiling top-level usage example; all 10 doctests pass. Per-item compiling examples beyond the crate entry points remain open. |
| Controlled ChunkHound comparison | Complete with one competitor resource failure | ChunkHound 5.2.1 used the exact Potion vectors through a loopback-only compatibility adapter and the same frozen file judgments. Across the 30 safely completed Django/Ollama/WordPress queries, Hay led nDCG@10 0.472472 to 0.247629, recall@50 0.916667 to 0.783333, and MRR 0.384630 to 0.208013. ChunkHound's Kubernetes build was stopped after 5,161 seconds at 9.83 GB RSS when its temporary database crossed the 120 GB disk guard; Hay completed that corpus in 198.76 seconds at 988,889,088 bytes RSS. |

## Competitive result and next acceptance run

The controlled comparison is now executable and recorded in
`BENCHMARKS.md`. It establishes a clear current product advantage without
claiming identical cascades: Hay uses BM25+dense RRF, while ChunkHound's
controlled OpenAI-compatible provider exposes no reranker and therefore runs
single-hop semantic search.

The next evidence gate is to broaden human judgments beyond ten queries per
repository and add a safe changed-file incremental check on a corpus larger
than Ollama. Kubernetes should not be retried with ChunkHound 5.2.1 unless a
hard external disk quota or upstream compaction fix prevents its observed
backup/rewrite amplification. Preserve the original Snowflake 256d/768d
evidence as the relevance reference, but do not promote that resource-failing
checkpoint.

No Phase 3 work begins unless competitor evidence or a broader judged suite
shows a material gap that late interaction is likely to close.
