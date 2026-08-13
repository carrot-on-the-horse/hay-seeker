# AGENTS.md

Operating guide for coding agents working in this repository. Human
contributors should read [CONTRIBUTING.md](./CONTRIBUTING.md) first; this file
adds the concrete commands, workflow, and release steps an agent needs.

Instructions in this file apply to the whole repository. A nested `AGENTS.md`,
if one is ever added, wins for files under its directory. Direct instructions
from the user always win over both.

## Repository map

| Path | Purpose |
| --- | --- |
| `crates/cast-core` | Data contracts and sizing primitives for CAST chunking |
| `crates/cast-tokenizers` | Pinned tokenizer-backed sizers |
| `crates/cast-tree-sitter` | Tree-sitter grammar registry and AST adapter |
| `crates/cast-index` | Runtime-agnostic repository-indexing contracts |
| `crates/cast-embeddings` | Embedding provider adapters (local and hosted) |
| `crates/cast-cli` | `cast` binary: chunking reference CLI |
| `crates/hay-search` | Backend-neutral search contract and Phase 0 harness |
| `crates/hay-duckdb` | Embedded DuckDB backend (zero-service default) |
| `crates/hay-elasticsearch` | Elasticsearch backend |
| `crates/hay-runtime` | Shared embedding and manifest runtime |
| `crates/hay-cli` | `hay` binary: index and search |
| `crates/hay-mcp` | `hay-mcp` binary: MCP stdio adapter |
| `crates/hay-eval` | `eval` binary: retrieval evaluation harness |
| `benchmarks/`, `evals/`, `scripts/` | Benchmark corpora, eval sets, tooling |

Design and contract documents are the source of truth for behavior:
[DESIGN.md](./DESIGN.md), [INDEX-CONTRACTS.md](./INDEX-CONTRACTS.md),
[HYBRID-SEARCH.md](./HYBRID-SEARCH.md), [COMPATIBILITY.md](./COMPATIBILITY.md),
[PRODUCT-ACCEPTANCE.md](./PRODUCT-ACCEPTANCE.md), and
[BENCHMARKS.md](./BENCHMARKS.md). Update the relevant document in the same
change that alters the behavior it describes.

## Environment

The toolchain is pinned in `rust-toolchain.toml` (Rust 1.97.1 with `clippy`,
`rustfmt`, and `rust-src`); do not add `+nightly` invocations or edit the pin to
work around a compile error. DuckDB is vendored through the `bundled` feature,
so a cold `cargo build` compiles C++ and takes several minutes — expect it and
do not treat a long first build as a hang.

The default DuckDB and static-embedding path needs no credentials and no
service. It does provision the pinned Potion Code 16M v2 bundle over the
network on first use; every artifact is checked against a pinned length and
SHA-256 before it is used, and `COTH_HAY_SEEKER_DOWNLOAD_MODELS=false` plus a
staged `COTH_HAY_SEEKER_LOCAL_STATIC_MODEL_DIR` restores a fully air-gapped run. Once a
bundle is present, indexing and search are offline. Hosted providers and
Elasticsearch are opt-in and read credentials from an ignored local `.env`; see
[.env.example](./.env.example). Tests that need a hosted provider or a
disposable Elasticsearch node must stay `#[ignore]`d and explicitly opt-in, and
no test may download a model.

## Verify before you finish

Run the complete verifier from the repository root and leave it passing:

```bash
./scripts/verify.sh
```

It enforces `cargo fmt --check`, `cargo clippy --workspace --all-targets
--all-features -- -D warnings`, `cargo test --workspace --all-features`, and
warning-free `cargo doc`. While iterating, narrow the loop first:

```bash
cargo test -p hay-duckdb
cargo clippy -p hay-duckdb --all-targets --all-features -- -D warnings
```

Tree-sitter chunking has full-output approval tests. Regenerate snapshots only
for an intentional contract change, and review the diff before committing:

```bash
UPDATE_CAST_GOLDENS=1 cargo test -p cast-tree-sitter --test golden
git diff -- crates/cast-tree-sitter/tests/goldens
```

Snapshot regeneration is refused in CI.

## Conventions

- `unsafe_code` is forbidden workspace-wide; Clippy runs with `all` and
  `pedantic` at warn and is promoted to `-D warnings` in the verifier.
- Dependencies, grammars, and the Tree-sitter runtime are exact-version pinned
  on purpose. Do not relax a pin, and do not add a dependency without a reason
  stated in the pull request.
- Shared dependencies and metadata live in `[workspace.dependencies]` and
  `[workspace.package]`; member crates inherit with `<dep>.workspace = true`.
- Every environment variable this code reads carries the `COTH_HAY_SEEKER_`
  prefix, with no exceptions and no fallback to the bare name — not for provider
  credentials, endpoints, model revisions, or bundle directories, even where the
  provider documents a bare name. An unprefixed variable can be supplied by an
  unrelated project's `.env` or the developer's shell, and the run would then use
  someone else's credential or endpoint while appearing to succeed. New settings
  also go in `.env.example` and in `ISOLATED_ENV_VARS`
  (`crates/hay-cli/tests/env_config.rs`) so tests cannot inherit a local value.
- Nothing prompts an automated caller. A question requires both standard input
  and standard error to be terminals, and any CI variable withdraws it
  (`crates/hay-cli/src/interaction.rs`). Standard output carries result JSON and
  the MCP protocol, so questions and progress go to standard error only. Any new
  implicit action fails closed and names the command that would do it.
- Changes to chunk identities, tokenizer or grammar versions, file eligibility,
  incremental checkpoints, index manifests, or backend scoring are
  compatibility changes: make migrations fail closed with a clear reindex
  requirement and record the evidence in the design and benchmark documents.
- Never commit `.env`, credentials, gateway URLs carrying private
  infrastructure identifiers, repository source, generated indexes
  (`.hay-seeker/`, `*.duckdb`), model weights, or benchmark checkouts. Use
  generated or publicly licensed fixtures.
- Match the surrounding prose style in Markdown: wrapped lines, no marketing
  language, and claims backed by a command or a document reference.

## Recommended workflow: pull requests

Land every change as a pull request against `main`. Do not commit or push to
`main` directly, and do not rewrite published history.

1. **Start from an up-to-date `main` on a topic branch.** One branch per
   logical change, named `<type>/<slug>` — for example `feat/duckdb-ann-gate`,
   `fix/checkpoint-stale-ids`, or `docs/release-steps`.

   ```bash
   git switch main && git pull --ff-only
   git switch -c feat/duckdb-ann-gate
   ```

2. **Confirm the task and its blast radius before editing.** Read the design
   document that owns the behavior, and say in the pull request when a change
   crosses a compatibility boundary listed above.

3. **Keep the change focused.** Unrelated cleanups belong in their own pull
   request. Commit messages use Conventional Commits (`feat:`, `fix:`,
   `docs:`, `refactor:`, `test:`, `chore:`) with an imperative subject and a
   body explaining why, not what.

4. **Verify locally, then scan for secrets.** Both must pass before you open
   the pull request:

   ```bash
   ./scripts/verify.sh
   gitleaks git . --redact
   ```

5. **Review your own diff before pushing.** `git diff main...HEAD` and
   `git status --short` — confirm no generated index, model artifact, `.env`,
   or stray scratch file is staged.

6. **Push the branch and open the pull request.**

   ```bash
   git push -u origin feat/duckdb-ann-gate
   gh pr create --base main --fill
   ```

   The description states the problem, the approach, the verification actually
   run (paste the commands and their outcome), the compatibility impact, and
   anything deliberately left out. Never claim a check passed that you did not
   run. Link the issue it closes.

7. **Keep the branch mergeable and let review finish.** Rebase on `main`
   (`git pull --rebase origin main`) rather than merging `main` into the
   branch, re-run the verifier after any rebase or review fix, and reply to
   each review comment. Merging is the maintainer's call; agents do not
   self-merge.

8. **Squash-merge, then delete the branch.** The squashed subject keeps the
   Conventional Commits form so release notes can be assembled from history.

For security-relevant findings, follow [SECURITY.md](./SECURITY.md) — private
reporting, not a public pull request or issue.

## Release

Releases publish the workspace crates to crates.io and attach built binaries to
a GitHub release. Every step below runs from a clean checkout of `main` after
the release pull request has merged.

### 1. Pre-flight

```bash
git switch main && git pull --ff-only
git status --short          # must be empty
./scripts/verify.sh
gitleaks git . --redact
```

### 2. Publishable manifest invariants

The manifests are already publish-ready; keep them that way. Every internal
dependency in `[workspace.dependencies]` carries a version next to its path,
because `cargo package` rejects a bare path entry with
`all dependencies must have a version requirement specified when packaging`:

```toml
cast-core = { path = "crates/cast-core", version = "0.1.0" }
```

Cargo uses the path in-workspace and rewrites it to the version requirement in
the published package. Every member crate sets `description` and inherits
`license`, `repository`, and `readme` from `[workspace.package]`; the inherited
`readme = "README.md"` resolves against the workspace root, so the top-level
README is packaged into each crate and rendered on crates.io. A new crate added
to the workspace needs the same four inherited fields plus its own
`description`, and a versioned entry in `[workspace.dependencies]` if other
members depend on it.

### 3. Bump the version

Versions are unified through `[workspace.package]`. Edit the single `version`
field in the root `Cargo.toml`, update the internal dependency requirements in
`[workspace.dependencies]` to the same version, refresh the lockfile, and
re-verify:

```bash
cargo update --workspace
./scripts/verify.sh
```

Open the bump as its own pull request following the workflow above.

### 4. Build release artifacts

```bash
cargo build --workspace --release --locked
```

This produces the shipped binaries in `target/release/`: `cast` (cast-cli),
`hay` (hay-cli), `hay-mcp`, and `eval` (hay-eval). Smoke-test them from the
release directory before publishing anything:

```bash
printf 'fn main() {}\n' | ./target/release/cast - --language rust --pretty
./target/release/hay --help
./target/release/hay-mcp --help
```

For a distributable per-platform archive, build on each target host (or with a
cross toolchain) and package the four binaries plus `README.md` and `LICENSE`:

```bash
cargo build --workspace --release --locked --target aarch64-apple-darwin
tar -czf hay-seeker-v0.1.0-aarch64-apple-darwin.tar.gz \
  -C target/aarch64-apple-darwin/release cast hay hay-mcp eval
```

DuckDB is compiled from vendored sources, so each target needs a working C++
toolchain and the build is slow; do not add `--no-verify` shortcuts to hide a
target that does not compile.

### 5. Dry-run the publish

Cargo resolves the publish order from the dependency graph:

```bash
cargo publish --workspace --dry-run --locked
```

Fix every warning that would become a permanent crates.io artifact (missing
metadata, unintended packaged files) before continuing. To rehearse a single
crate:

```bash
cargo publish -p cast-core --dry-run --locked
```

Before the first release, confirm the names are still unclaimed — a 404 from the
sparse index means available:

```bash
curl -sI https://index.crates.io/ca/st/cast-core | head -1
```

Note the first-publish asymmetry: rehearsing a dependent crate on its own
(`cargo publish -p hay-search --dry-run`) fails with
`no matching package named cast-index found` until its dependencies exist on
crates.io. `cargo publish --workspace` handles that itself by ordering the
uploads, so use the workspace form for the initial release.

### 6. Publish

```bash
cargo publish --workspace --locked
```

If a crate must be published individually — or `--workspace` fails partway and
you resume — use this dependency order, waiting for each crate to become
resolvable on the index before the next tier:

1. `cast-core`
2. `cast-tokenizers`, `cast-tree-sitter`, `cast-index`
3. `cast-embeddings`
4. `hay-search`
5. `hay-runtime`, `hay-duckdb`, `hay-elasticsearch`
6. `cast-cli`, `hay-cli`, `hay-mcp`, `hay-eval`

crates.io releases are permanent: a version can be yanked but never replaced.
Publishing is a maintainer action with a registry token, so an agent stops at
the dry run unless the user explicitly asks for the real publish.

### 7. Tag and publish the GitHub release

```bash
git tag -a v0.1.0 -m "Hay Seeker v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "Hay Seeker v0.1.0" --generate-notes \
  hay-seeker-v0.1.0-aarch64-apple-darwin.tar.gz
```

Then confirm the release notes state the compatibility impact — in particular
whether existing indexes must be rebuilt — and that no credential, private
gateway URL, or generated index was attached.
