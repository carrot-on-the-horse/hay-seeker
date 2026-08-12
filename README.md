# Hay Seeker

An embedded Rust hybrid-search library, starting from production-proven AST-aware semantic source chunking. Tree-sitter produces the syntax tree; CAST (Chunking via Abstract Syntax Trees) turns it into exact, source-backed chunks.

This repository currently contains the first executable vertical slice from [DESIGN.md](./DESIGN.md):

- Rust 2024 workspace pinned to Rust 1.97.1;
- parser-independent public contracts and byte/word/line sizers;
- Tree-sitter grammars for Rust, Bash, C, C++, C#, Go, Java, JavaScript,
  PHP, Python, Ruby, TypeScript, and TSX;
- recursive AST splitting and adjacent-node grouping;
- strict UTF-8-safe fallback for oversized atomic nodes;
- exact, non-overlapping core ranges that reconstruct the input;
- explicit parse-recovery and generic-fallback diagnostics;
- compatibility defaults and safety limits carried over from the proven Go pipeline;
- a typed Gemini Embedding 2 adapter through Cloudflare AI Gateway;
- a small JSON CLI.

See [COMPATIBILITY.md](./COMPATIBILITY.md) for the audited 1,000-repository behaviors, deliberate corrections, and remaining parity gaps.

Common repository-indexing domain types and adapter interfaces are defined in [`cast-index`](./crates/cast-index) and documented in [INDEX-CONTRACTS.md](./INDEX-CONTRACTS.md).

The backend-neutral search contract, exact index manifest, phased local/Elasticsearch architecture, and acceptance gates are documented in [HYBRID-SEARCH.md](./HYBRID-SEARCH.md). Phase 0 adds `hay-search`, the `eval` harness, and a frozen seed set of 32 realistic queries.

The current embedded/on-device backend comparison and Qdrant Edge experiment
gates are in [EMBEDDED-SEARCH-RESEARCH.md](./EMBEDDED-SEARCH-RESEARCH.md).

`hay`, `hay-mcp`, DuckDB, and Elasticsearch all use that same `Retriever`
contract. DuckDB is the zero-service default; Elasticsearch is the first
remote target. `hay-mcp` exposes either index to MCP clients over local stdio.

Real-repository recall and chunking benchmarks use pinned WordPress, Django,
Kubernetes, and Ollama snapshots. The controlled ChunkHound 5.2.1 comparison
uses the same Potion vectors and file-level judgments; Hay leads all safely
completed corpora, while ChunkHound's Kubernetes build trips the disk safety
guard. Setup, exact commands, resource telemetry, baseline numbers, and the
retrieval-mode caveat are in [BENCHMARKS.md](./BENCHMARKS.md).

Hay Seeker is available under the [MIT License](./LICENSE). Contributions are
welcome; see [CONTRIBUTING.md](./CONTRIBUTING.md), and report security issues
privately according to [SECURITY.md](./SECURITY.md). Coding agents and anyone
looking for the exact commands, pull-request workflow, or
[release and publish steps](./AGENTS.md#release) should read
[AGENTS.md](./AGENTS.md).

## Try it

```bash
cargo run -p cast-cli -- path/to/source.rs --max-size 1500 --sizer bytes --pretty
```

Read Rust from stdin:

```bash
printf 'fn main() {}\n' | cargo run -p cast-cli -- - --language rust --pretty
```

CAST-chunk a Git repository into a durable local index, then query it:

```bash
cargo run -p hay-cli -- index --backend duckdb \
  --database .hay-seeker/index.duckdb --repository /path/to/repository
cargo run -p hay-cli -- search --backend duckdb \
  --database .hay-seeker/index.duckdb --top-k 5 \
  "where is manifest compatibility validated?"
```

Those commands need no credentials and no manual model staging. `hay` defaults
to `--embeddings local-static` and provisions the pinned
[Potion Code 16M v2](./models/potion-code-16m-v2/README.md) bundle (MIT, 31 MiB)
into a per-user cache on first use, then runs entirely locally. See
[Automatic model provisioning](#automatic-model-provisioning) to point it at a
mirror, pre-stage the bundle, or turn downloading off.

### Configure with `.env`

`hay`, `hay-mcp`, and `eval` find the nearest `.env` file in the current
directory or its parents. Copy the template before editing it; `.env` and
`.env.*` are ignored by Git:

```bash
cp .env.example .env
```

Hay Seeker's own settings use the `COTH_HAY_SEEKER_*` namespace. Precedence is
an explicit command-line value, an already-exported process variable, `.env`,
then the built-in default. A missing `.env` is allowed; a malformed or
unreadable file stops startup instead of silently using partial configuration.
For example:

```dotenv
COTH_HAY_SEEKER_BACKEND=duckdb
COTH_HAY_SEEKER_EMBEDDINGS=none
COTH_HAY_SEEKER_DATABASE=.hay-seeker/index.duckdb
COTH_HAY_SEEKER_REPOSITORY=/path/to/repository
COTH_HAY_SEEKER_TOP_K=5
```

With that file, indexing and searching need only the subcommand and query:

```bash
cargo run -p hay-cli -- index
cargo run -p hay-cli -- search "where is manifest compatibility validated?"
```

The query itself can optionally come from `COTH_HAY_SEEKER_QUERY`. Other
supported settings are `COTH_HAY_SEEKER_CORPUS`,
`COTH_HAY_SEEKER_CHECKPOINT`, `COTH_HAY_SEEKER_STALL_TIMEOUT_SECONDS`,
`COTH_HAY_SEEKER_PROGRESS_INTERVAL_SECONDS`,
`COTH_HAY_SEEKER_ELASTICSEARCH_ENDPOINT`,
`COTH_HAY_SEEKER_ELASTICSEARCH_INDEX`, and
`COTH_HAY_SEEKER_CANDIDATE_LIMIT`. Set only one of
`COTH_HAY_SEEKER_REPOSITORY` and `COTH_HAY_SEEKER_CORPUS`. The complete
template is in [`.env.example`](./.env.example), and each command's `--help`
output shows the variable accepted by every option without displaying its
current value.

Hay-specific Cloudflare and OpenAI credentials are namespaced as
`COTH_HAY_SEEKER_CF_AIG_TOKEN` and `COTH_HAY_SEEKER_OPENAI_API_KEY`. Other
provider credentials and model locations keep their provider-native names,
such as `ELASTICSEARCH_API_KEY` and `HAY_LOCAL_MODEL_DIR`.

#### Embedding model examples

Select one embedding profile in `.env`, and use that same file for indexing,
searching, and MCP. Lexical-only search needs no model or credentials:

```dotenv
COTH_HAY_SEEKER_EMBEDDINGS=none
```

The ONNX and code-specific static profiles run from checksum-pinned local
bundles. The ONNX profile is always caller-provisioned. The static profile is
provisioned automatically unless you stage it yourself or disable downloads:

```dotenv
# Snowflake Arctic Embed m v2, Nomic v1.5, or EmbeddingGemma bundle
COTH_HAY_SEEKER_EMBEDDINGS=local-onnx
HAY_LOCAL_MODEL_DIR=/absolute/path/to/snowflake-arctic-bundle
```

```dotenv
# Potion Code 16M v2 bundle
COTH_HAY_SEEKER_EMBEDDINGS=local-static
HAY_LOCAL_STATIC_MODEL_DIR=/absolute/path/to/potion-code-16m-v2
```

Gemini Embedding 2 runs through the complete Cloudflare AI Gateway Vertex
route. The token must have AI Gateway Run permission:

```dotenv
COTH_HAY_SEEKER_EMBEDDINGS=gemini
COTH_HAY_SEEKER_CF_AIG_TOKEN=replace-with-gateway-run-token
GEMINI_MODEL_REVISION=approved-2026-08-11
GEMINI_GATEWAY_URL=https://gateway.ai.cloudflare.com/v1/account/gateway/google-vertex-ai/v1/projects/project/locations/location/publishers/google/models/gemini-embedding-2:embedContent
```

The other hosted providers use their native credentials and require an
explicit approved model revision:

Direct OpenAI API:

```dotenv
COTH_HAY_SEEKER_EMBEDDINGS=open-ai
COTH_HAY_SEEKER_OPENAI_API_KEY=replace-with-api-key
OPENAI_MODEL_REVISION=approved-2026-08-11
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

OpenAI through Cloudflare AI Gateway:

```dotenv
COTH_HAY_SEEKER_EMBEDDINGS=open-ai
COTH_HAY_SEEKER_CF_AIG_TOKEN=replace-with-gateway-run-token
OPENAI_GATEWAY_URL=https://gateway.ai.cloudflare.com/v1/account/gateway/openai/embeddings
OPENAI_MODEL_REVISION=approved-2026-08-11
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
# Optional with Cloudflare BYOK or Unified Billing
COTH_HAY_SEEKER_OPENAI_API_KEY=replace-with-api-key
```

```dotenv
# Voyage
COTH_HAY_SEEKER_EMBEDDINGS=voyage
VOYAGE_API_KEY=replace-with-api-key
VOYAGE_MODEL_REVISION=approved-2026-08-11
VOYAGE_EMBEDDING_MODEL=voyage-code-3
```

```dotenv
# Cloudflare Workers AI
COTH_HAY_SEEKER_EMBEDDINGS=cloudflare-workers-ai
CLOUDFLARE_ACCOUNT_ID=replace-with-account-id
CLOUDFLARE_AI_TOKEN=replace-with-api-token
CLOUDFLARE_WORKERS_AI_MODEL_REVISION=approved-2026-08-11
```

The optional dimension, concurrency, and retry controls for each provider are
listed in [`.env.example`](./.env.example). Never commit real tokens.

The repository path prefers streaming `git ls-files` enumeration (tracked plus
non-ignored untracked files) and falls back to a deterministic filesystem walk
for non-Git directories. It preserves the production Go filters for hidden and
ignored paths, 50 KiB config/data files, 5 MiB source files, the 8 KiB binary
probe, invalid UTF-8, and large data-like files. JSONL remains available with
`--corpus evals/corpus.jsonl` for controlled imports and evaluation fixtures.
Chunk IDs are SHA-256 identities over normalized path, complete source content,
the backend-neutral relevance fingerprint, ordinal, and core byte range;
content or relevance-contract changes therefore cannot silently reuse an old
chunk identity, while storage-only quantization differences preserve IDs across
backends. The manifest's chunker identity includes the CAST algorithm, every
AST limit and policy, fixed-window size and overlap, exact sizing tokenizer,
and the complete compiled Tree-sitter grammar set. Repository indexing refuses
to run if that identity does not match the executable chunker.

Repository indexing writes a versioned content-hash checkpoint after the new
searchable generation is published. A second run reuses unchanged chunks and
stored vectors, embeds only changed chunks, and removes chunks for changed,
deleted, or newly filtered files. DuckDB stages the delta and commits inserts
and deletes together. Elasticsearch copies unchanged documents into a fresh
physical index, applies the delta, then atomically swaps the alias. The default
checkpoint is `<database>.checkpoint.json` for DuckDB and
`<repository>/.hay-seeker/elasticsearch-<alias-hash>.checkpoint.json` for
Elasticsearch (the hash prevents aliases from becoming path input); override it
with `--checkpoint`. The prior checkpoint is
invalidated before mutation, so an interrupted run safely falls back to a full
rebuild instead of trusting stale sync state.

For dense DuckDB profiles, a manifest-scoped embedding cache is retained in
the same database across incremental runs. Its SHA-256 key binds both the
document ID and exact text, so switching back to previously indexed content
can reuse its vector while changed text under a reused ID is always a miss.
Cache blobs use the index's exact vector codec and dimensions; corruption or a
conflicting result from the pinned embedder fails closed. Historical entries
are intentionally retained for branch switching. A full rebuild into a new
database starts with an empty cache and is the current compaction boundary.

Long repository runs emit compact `index_progress` JSON records to stderr every
five seconds and abort after ten minutes without a completed or skipped file.
Use `--progress-interval-seconds` and `--stall-timeout-seconds` to tune those
bounds. The final stdout result remains one JSON object and now includes
`mode`, `total_ms`, source bytes, and cumulative discovery/read/chunk timings.

Run the frozen evaluation baseline:

```bash
cargo run --bin eval -- --backend local
cargo run --bin eval -- --backend elastic
cargo run -p hay-eval -- --backend duckdb --embeddings none --suite seed
```

Run both production backends over the exact same loaded documents and
judgments, validate their manifest pair, and enforce the `0.02` nDCG@10 gate
in one process with:

```bash
cargo run -p hay-eval -- --backend parity --embeddings none --suite seed \
  --database /path/to/disposable-parity.duckdb \
  --endpoint http://127.0.0.1:9200 --index hay-parity-seed
```

The parity backend prints both complete metric sets plus absolute nDCG@10,
recall@50, and MRR deltas, and exits nonzero when nDCG drift exceeds `0.02`.
Use a disposable DuckDB path and Elasticsearch alias because both indexes are
rebuilt. `--embeddings local-onnx` applies the same command to the pinned ONNX
checkpoint and its approved MRL representation pair. `--embeddings
local-static` uses the native 256-dimensional Potion code model on both
backends. Dense parity opens one provider session and embeds every document
once in bounded batches; ONNX full-width results are retained for
Elasticsearch while DuckDB gets the normalized 256-dimensional MRL projection.
The static profile shares its already normalized native output. Adjust the evaluator's memory/request boundary with
`--embedding-batch-size` (default `8`); telemetry must still report
`parity.document_embedding_passes: 1`.

The production evaluation path accepts `--embeddings gemini` for DuckDB and
Elasticsearch. This sends every evaluated document and query to the configured
Cloudflare AI Gateway, so run it only for corpora approved for that external
provider. `hay`, `hay-mcp`, and `hay-eval` all construct provider identities and
exact manifests through the shared `hay-runtime` crate.

## Fully offline embeddings

`--embeddings local-onnx` is the private, airplane-mode product path. Set
`HAY_LOCAL_MODEL_DIR` to a checksum-pinned bundle following one of the reviewed
contracts. [Snowflake Arctic Embed m v2](./models/snowflake-arctic-embed-m-v2.0/README.md)
is the current acceptance candidate: it is Apache-2.0, ungated, and passed the
seed 256d DuckDB / 768d Elasticsearch parity gate. The
[Nomic v1.5](./models/nomic-embed-text-v1.5/README.md) and
[EmbeddingGemma](./models/embeddinggemma/README.md) profiles remain supported
for comparison and caller-provisioned deployments.

Hay verifies every artifact before loading it, and never downloads an ONNX
bundle: `HAY_LOCAL_MODEL_DIR` must already exist. A bundle pins its exact graph
inputs, output transform, retrieval prompts, and embedding profile as part of
the index identity. Core ML is tried
only when that exact graph declares compatibility; otherwise ONNX Runtime uses
CPU explicitly. DuckDB stores 256 re-normalized MRL dimensions as versioned
per-vector int8 values with scale and offset, while Elasticsearch uses the same
checkpoint at 768 dimensions with BBQ.

The preferred Arctic profile keeps the reviewed 1,500-token CAST chunks, then
encodes complete documents as 256-token model windows with 32-token overlap and
a token-weighted mean at 768 dimensions. MRL projection happens after that
shared aggregation, so DuckDB and Elasticsearch differ only in the approved
stored width/quantization rather than in which part of a chunk they see.

For the promotable low-footprint path, `--embeddings local-static` uses the
code-trained [Potion Code 16M v2 bundle](./models/potion-code-16m-v2/README.md).
The Rust adapter directly opens its checksum-pinned F16 embedding table and
tokenizer; it never invokes an external runtime, and it reads only from disk.
Provisioning that bundle is a separate step that runs before loading, described
in [Automatic model provisioning](#automatic-model-provisioning). Both
backends use the model's native trained 256 dimensions. This deliberately
changes the original 256d-local/768d-remote width contract, and the distinct
embedding-profile identity makes old indexes fail closed instead of silently
mixing representations.

```bash
export HAY_LOCAL_MODEL_DIR=/absolute/path/to/snowflake-arctic-bundle

cargo run -p hay-cli -- index --backend duckdb --embeddings local-onnx \
  --database .hay-seeker/offline.duckdb --repository /path/to/repository

cargo run -p hay-cli -- search --backend duckdb --embeddings local-onnx \
  --database .hay-seeker/offline.duckdb \
  "where is repository checkpoint compatibility validated?"
```

The matching MCP command is fully local as well:

```bash
cargo build --release -p hay-mcp --locked
HAY_LOCAL_MODEL_DIR=/absolute/path/to/snowflake-arctic-bundle \
  target/release/hay-mcp --backend duckdb --embeddings local-onnx \
  --database /absolute/path/to/offline.duckdb
```

The equivalent static code-model cycle, with the bundle staged by hand rather
than provisioned, is:

```bash
export HAY_LOCAL_STATIC_MODEL_DIR=/absolute/path/to/potion-code-16m-v2
cargo run -p hay-cli -- index --backend duckdb --embeddings local-static \
  --database .hay-seeker/code.duckdb --repository /path/to/repository
cargo run -p hay-cli -- search --backend duckdb --embeddings local-static \
  --database .hay-seeker/code.duckdb "where is request validation handled?"
```

## Automatic model provisioning

`--embeddings local-static` is the default, so `hay` and `hay-mcp` need a
Potion Code 16M v2 bundle before they can open an index. They resolve one in a
fixed order:

1. `HAY_LOCAL_STATIC_MODEL_DIR`, when set. A staged bundle always wins and is
   never compared against the network.
2. The per-user cache, when it already holds the pinned artifacts.
3. A download, when `COTH_HAY_SEEKER_DOWNLOAD_MODELS` is true (the default).

Provisioning is not a trust decision. The catalog in `cast-embeddings` pins the
upstream revision, each artifact's exact byte length, and each artifact's
SHA-256, and a transfer that misses any of them is deleted rather than used.
The manifest written into the cache is byte-identical to
[`static-bundle.example.json`](./models/potion-code-16m-v2/static-bundle.example.json),
so a provisioned bundle and a hand-staged one produce the same index identity
and the same `bundle-sha256` in the manifest. That is why a mirror over plain
HTTP is safe: it can serve the bytes faster, but it cannot serve different ones.

| Setting | Default | Purpose |
| --- | --- | --- |
| `COTH_HAY_SEEKER_DOWNLOAD_MODELS` | `true` | Allow provisioning. `false` fails with instructions instead of downloading. |
| `COTH_HAY_SEEKER_MODEL_CACHE_DIR` | platform cache | Where bundles are stored. |
| `COTH_HAY_SEEKER_MODEL_BASE_URL` | `https://huggingface.co` | Mirror or proxy to fetch from. |
| `HAY_LOCAL_STATIC_MODEL_DIR` | unset | Use this staged bundle and skip provisioning entirely. |

The default cache is `$XDG_CACHE_HOME/hay-seeker/models` (or `~/.cache`),
`~/Library/Caches/hay-seeker/models` on macOS, and `%LOCALAPPDATA%\hay-seeker\models`
on Windows. Bundles are keyed by model and revision, so a future catalog
revision provisions beside the current one instead of replacing it.

Air-gapped and reproducible builds keep their previous behavior by pinning both
settings:

```bash
export COTH_HAY_SEEKER_DOWNLOAD_MODELS=false
export HAY_LOCAL_STATIC_MODEL_DIR=/absolute/path/to/potion-code-16m-v2
```

A damaged cache repairs itself: an artifact whose length no longer matches the
catalog is re-fetched, and the loader still verifies every digest before use, so
a corrupted file fails closed rather than producing wrong vectors.

### Upgrading an index built before this default

The default provider changed from `none` to `local-static`, so an index built
lexically is now opened with a different manifest and fails closed with
`reindex required`. Keep the old index by naming its provider explicitly:

```bash
cargo run -p hay-cli -- search --backend duckdb --embeddings none \
  --database .hay-seeker/index.duckdb "where is request validation handled?"
```

Provisioning runs before the index manifest is compared, so a mismatched query
still pays for the download once. Passing `--embeddings none` avoids it.

## MCP search CLI

Build the MCP stdio binary:

```bash
cargo build --release -p hay-mcp --locked
```

Configure an MCP client to launch it with absolute paths:

```bash
codex mcp add hay-search -- \
  /absolute/path/to/hay-seeker/target/release/hay-mcp \
  --backend duckdb \
  --database /absolute/path/to/hay-seeker/.hay-seeker/index.duckdb
```

Or use the equivalent generic client configuration:

```json
{
  "mcpServers": {
    "hay-search": {
      "command": "/absolute/path/to/hay-seeker/target/release/hay-mcp",
      "args": [
        "--backend",
        "duckdb",
        "--database",
        "/absolute/path/to/hay-seeker/.hay-seeker/index.duckdb"
      ]
    }
  }
}
```

The server publishes two typed tools:

- `search`: accepts `query`, optional `top_k`, `candidate_limit`, and
  `enable_late_interaction`; returns structured candidates with path, language,
  source text, final score, and stage signals.
- `capabilities`: reports the active backend, document count, and supported
  cascade stages.

The default backend is `duckdb`; `--backend elasticsearch --endpoint ...
--index ...` exposes the same tools over Elasticsearch. Authentication is read
from `ELASTICSEARCH_API_KEY` or `ELASTICSEARCH_BEARER_TOKEN` in the environment
or `.env`. `--backend phase0 --corpus ...` remains only as a deterministic
integration stub. Only JSON-RPC protocol messages are written to stdout;
startup errors go to stderr.

The DuckDB MCP adapter validates the index at startup but does not keep the
database file open while the server is idle. Each tool call opens a short-lived
snapshot, and `search` performs retrieval plus result enrichment through that
same connection. A long-running MCP client therefore does not block `hay index`
from publishing an incremental update between requests.

For an index created with dense embeddings, pass the same provider to MCP. For
example, add `--embeddings gemini` for a Gemini-built index. The server loads
`COTH_HAY_SEEKER_CF_AIG_TOKEN`, `GEMINI_MODEL_REVISION`, and the optional
Gemini tuning variables from `.env`, validates the exact stored manifest
before serving, and reports a hybrid backend with dense capability enabled.

## Elasticsearch lifecycle

The Elasticsearch path uses a stable alias. Every rebuild creates a new
physical index, stores the exact `IndexManifest` in mapping metadata, bulk
indexes in bounded 5 MiB requests, performs one final refresh, and swaps the
alias only after success. The same atomic alias request removes excess
Hay-owned orphan generations; by default the active index and one rollback
generation are retained. Names must match the strict
`<alias>-build-<seconds>-<nanoseconds>` format and an active alias target must
exist before any cleanup is planned. Library users can configure retention
from 2 through 32 generations with
`ElasticsearchConfig::with_generation_retention`.

Publication is treated as an ambiguous commit boundary. If the alias request
returns an error, Hay reads the exact new physical index's alias state before
cleanup. An already-active generation is accepted. A negative read alone is
not treated as proof of non-publication across every cluster node, so the
generation is retained with an explicit error. The next successful build can
reclaim it through strict generation retention once a stable active target is
known. Hay never deletes a possibly live target merely because the response
was lost.

```bash
ELASTICSEARCH_API_KEY=... cargo run -p hay-cli -- index \
  --backend elasticsearch --endpoint https://search.example.com \
  --index hay-seeker --repository /path/to/repository

ELASTICSEARCH_API_KEY=... cargo run -p hay-cli -- search \
  --backend elasticsearch --endpoint https://search.example.com \
  --index hay-seeker "where is manifest compatibility validated?"
```

Remote endpoints must use HTTPS. A local loopback Elasticsearch node may use
HTTP. Both backends fail closed when the stored analyzer, chunker, schema, or
embedding manifest differs from the runtime contract.

Run the opt-in live lifecycle test only against a disposable alias:

```bash
ELASTICSEARCH_TEST_URL=http://127.0.0.1:9200 \
ELASTICSEARCH_TEST_INDEX=hay-seeker-live-test \
cargo test -p hay-elasticsearch --test live_cycle -- --ignored
```

Use a disposable alias. Repeated runs may remove older Hay-owned physical
generations under that exact alias according to the configured retention, and
the retained generations are intentionally left for inspection.

Run the pinned repository suite:

```bash
./scripts/fetch-bench-repos.sh
cargo run --bin eval -- --backend local --suite repos
cargo run --bin eval -- --backend duckdb --suite repos
cargo bench -p hay-search --bench repository_chunking
```

Repository evaluation runs every eligible source through the same executable
`ChunkerV1` profile pinned by product manifests. Retrieval occurs over chunks,
then candidates collapse by first-ranked occurrence to the parent file IDs used
by the frozen judgments. This keeps recall labels reviewable without replacing
the product's AST-aware retrieval unit with whole-file vectors.

The sizing options are `open-ai`, `bytes`, `words`, and `lines`. The default is
the pinned OpenAI® `o200k_base` encoding when a model-specific tokenizer is not
available. `words` retains compatibility with the Go `SimpleTokenCounter`.
Regardless of sizing unit, chunks have a UTF-8-safe 25,000-byte hard ceiling.

Run the stable and GigaToken comparison suite with:

```bash
./scripts/bench-tokenizers.sh
```

GigaToken currently requires nightly Rust, so its pinned benchmark crate is
isolated from the stable production workspace. The benchmark checks token-ID
parity before reporting both warm-cache and cold-cache encode throughput. Use
`./scripts/bench-tokenizers.sh --giga-only` to skip the stable sizing run.

## Gemini through Cloudflare AI Gateway

The `cast-embeddings` crate implements the common `Embedder` contract against
the pinned provider-native Vertex endpoint. Copy `.env.example` to `.env` and
set a Cloudflare token with AI Gateway Run permission:

```bash
cargo run -p cast-embeddings --example gemini_gateway -- \
  query "where is the chunk size configured?"
```

Run the complete CAST-to-ranking smoke path on the generated Go fixture:

```bash
GEMINI_SMOKE_CHUNK_TOKENS=80 cargo run -p cast-embeddings \
  --example repo_smoke -- \
  crates/cast-embeddings/tests/fixtures/synthetic_routes.go \
  "where are API routes registered?"
```

The token is sent only as `cf-aig-authorization: Bearer ...`. Do not put the
old `CH_GEMINI_API_KEY` in `COTH_HAY_SEEKER_CF_AIG_TOKEN`: that value is a
Google API key, while the selected endpoint expects Cloudflare Gateway
authentication and Vertex BYOK credentials configured on the gateway.

Gemini Embedding 2 uses versioned retrieval prefixes instead of `taskType`.
Documents use `title: none | text: ...`; queries use
`task: search result | query: ...`. The default is 768 dimensions. Because the
Vertex `embedContent` route accepts one content at a time, batch calls use eight
bounded concurrent requests and restore exact input order. Retryable transport,
timeout, HTTP 429, and upstream 5xx failures use at most four total attempts;
`Retry-After` is respected with a 30-second per-delay cap. Authentication and
other permanent request failures are never retried. A provider-neutral
`RetryingEmbedder` orchestration wrapper owns attempts, jitter, and the total
budget; the Gemini adapter only returns typed `RetryAdvice`. Set
`GEMINI_EMBEDDING_MAX_ATTEMPTS` to a value from 1 through 10 to tune or disable
retries.

The product index and query cycle enables that adapter with the same flag on
both commands. `GEMINI_MODEL_REVISION` is deliberately required: use the
deployment/model revision your team has approved, and change it whenever the
provider-managed model changes.

```bash
GEMINI_MODEL_REVISION=approved-2026-08-06 \
cargo run -p hay-cli -- index --backend duckdb --embeddings gemini \
  --database .hay-seeker/gemini.duckdb --repository /path/to/repository

GEMINI_MODEL_REVISION=approved-2026-08-06 \
cargo run -p hay-cli -- search --backend duckdb --embeddings gemini \
  --database .hay-seeker/gemini.duckdb \
  "where are API routes registered?"
```

The manifest also binds the complete Gateway route hash, output dimensions,
retrieval-prefix profile, OpenAI fallback tokenizer artifact, and BM25 analyzer
revision. Elasticsearch uses native BM25 and BBQ kNN candidate searches, then
Hay applies its shared deterministic RRF in Rust. This works on the Basic
license; Elastic's native RRF retriever is not required.

## Other hosted embedding providers

The same `Embedder` and manifest contracts support three additional adapters:

| CLI value | Default model | Retrieval contract | Required environment |
| --- | --- | --- | --- |
| `open-ai` | `text-embedding-3-small`, 768 dimensions | Symmetric float embeddings, direct or through Cloudflare AI Gateway | `OPENAI_MODEL_REVISION`; direct requires `COTH_HAY_SEEKER_OPENAI_API_KEY`; gateway requires `COTH_HAY_SEEKER_CF_AIG_TOKEN` and `OPENAI_GATEWAY_URL` |
| `voyage` | `voyage-code-3`, 1,024 dimensions | Provider-native `document` and `query` input types; truncation disabled | `VOYAGE_API_KEY`, `VOYAGE_MODEL_REVISION` |
| `cloudflare-workers-ai` | `@cf/qwen/qwen3-embedding-0.6b`, 1,024 dimensions | Native document mode and code-search query instruction | `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_AI_TOKEN`, `CLOUDFLARE_WORKERS_AI_MODEL_REVISION` |

Each adapter batches without reordering output, rejects incomplete response
indices or shapes, caps response bytes, redacts credentials from diagnostics,
and returns the same retry classification consumed by `RetryingEmbedder`.
Model revision is mandatory because hosted aliases can change without changing
their names. Without `OPENAI_GATEWAY_URL`, OpenAI calls the official API
directly using `COTH_HAY_SEEKER_OPENAI_API_KEY`. When the gateway URL is set,
`COTH_HAY_SEEKER_CF_AIG_TOKEN` is sent in `cf-aig-authorization`; the optional
namespaced OpenAI key is sent separately in `Authorization`, or omitted for
Cloudflare BYOK or Unified Billing. The exact route hash is part of the index
identity. Optional model, dimension, and retry variables are listed in
[.env.example](./.env.example).

Verify Workers AI's native contract using only generated, non-repository text:

```bash
cargo run -p cast-embeddings --example workers_ai_smoke
```

Use the same provider value for indexing, direct search, MCP, and evaluation:

```bash
cargo run -p hay-cli -- index --backend duckdb --embeddings voyage \
  --database .hay-seeker/voyage.duckdb --repository /path/to/repository
cargo run -p hay-cli -- search --backend duckdb --embeddings voyage \
  --database .hay-seeker/voyage.duckdb "where is retry policy applied?"
```

OpenAI follows its official [embeddings API](https://developers.openai.com/api/reference/resources/embeddings/methods/create)
through Cloudflare's [provider-native OpenAI route](https://developers.cloudflare.com/ai-gateway/usage/providers/openai/),
Voyage follows its [text embeddings API](https://docs.voyageai.com/reference/embeddings-api-1),
and Workers AI uses Cloudflare's current
[Qwen3 embedding model](https://developers.cloudflare.com/workers-ai/models/qwen3-embedding-0.6b/).

## Verify

Run the complete offline suite from the repository root:

```bash
./scripts/verify.sh
```

This checks formatting, strict Clippy across every target and feature, all unit,
integration, golden, and compiling documentation tests, plus warning-free
Rustdoc generation. Ignored live-provider and live-Elasticsearch tests remain
explicit because they require credentials or external services.

The Tree-sitter integration has reviewed, full-output approval tests for clean
AST chunking, recovered syntax, and generic fallback. Run them normally with:

```bash
cargo test -p cast-tree-sitter --test golden
```

When an intentional chunking-contract change occurs, regenerate the snapshots
locally and review the resulting diff before committing it:

```bash
UPDATE_CAST_GOLDENS=1 cargo test -p cast-tree-sitter --test golden
git diff -- crates/cast-tree-sitter/tests/goldens
```

Snapshot regeneration is refused in CI. Fixtures are compact, synthetic, and
checked in beside the integration test; production repository sources remain
in the benchmark corpus instead of becoming multi-hundred-kilobyte snapshots.

## Current boundaries

- Popular grammars are enabled by the `cast-tree-sitter/popular-languages`
  default feature; each non-Rust grammar also has an individual `lang-*` feature.
- Grammar crates and the Tree-sitter runtime are exact-version pinned, and the
  registry exposes a deterministic grammar-set ID for index invalidation.
- CAST code overlap is intentionally deferred; Phase 0 non-code fixed windows have exact source-backed overlap.
- Exact Gemini tokenizer support, the remaining legacy grammar set, the wider
  legacy approval corpus, and fuzzing are not implemented yet.
- The CLI rejects non-UTF-8 files instead of converting them lossily.
- DuckDB production retrieval currently uses persisted BM25 statistics and an
  exact cosine scan. Full rebuild, atomic upsert, deletion, restart persistence,
  and manifest invalidation are implemented; ANN remains acceptance-gated. Its
  embedded profile bounds DuckDB to a 512 MB buffer pool and two threads; the
  pinned Kubernetes lexical full-build gate peaks at 787 MB RSS for 79,949
  chunks. The Potion hybrid profile completes the same build at 988.9 MB RSS
  and reuses all chunks on an unchanged second run.
- DuckDB and Elasticsearch accept the same bounded repository-to-CAST-chunk
  stream. Initial DuckDB builds use a sibling temporary database; incremental
  runs stage changed chunks and commit them with stale-ID deletes in one
  transaction. Elasticsearch initial builds and incremental copy-on-write runs
  publish through an atomic alias swap. Unchanged stored vectors are retained.
  Source, chunking, embedding, storage, or checkpoint-resolution failure
  preserves the prior searchable generation.
- Repository runs expose full/incremental mode, source bytes, skip/delete/reuse
  counters, discovery/read/chunk/total timings, periodic stderr progress, and a
  configurable no-progress stall abort (ten minutes by default).
- Elasticsearch implements atomic rebuilds, BM25, dense-vector mappings, kNN,
  and shared license-independent RRF. Its lexical live parity gate passes;
  each deployment environment should still run the opt-in network lifecycle
  test against a disposable alias.
- Gemini, OpenAI, Voyage, and Cloudflare Workers AI are wired through the
  product CLI, MCP, and evaluation runtime for DuckDB exact-vector and
  Elasticsearch BBQ-kNN hybrid retrieval. Their request/response contracts are
  tested locally; provider-specific live recall gates remain opt-in. Late
  interaction, learned sparse retrieval, and FDE remain gated.

Current product evidence and remaining acceptance gates are tracked in
[PRODUCT-ACCEPTANCE.md](./PRODUCT-ACCEPTANCE.md).
