# Contributing

Thanks for helping improve Hay Seeker.

## Development

Hay Seeker uses the Rust toolchain pinned in `rust-toolchain.toml`. Clone the
repository, make focused changes, and run the complete verifier before opening
a pull request:

```bash
./scripts/verify.sh
```

The verifier enforces formatting, strict Clippy checks, all workspace tests,
doctests, and warning-free documentation. Tests that need a hosted provider or
disposable Elasticsearch node must remain explicitly opt-in.

## Pull requests

Land every change as a pull request against `main` from a topic branch; do not
push to `main` directly. Keep the change focused, use Conventional Commits, run
`./scripts/verify.sh` and `gitleaks git . --redact` before opening the pull
request, and describe the verification you actually ran plus the compatibility
impact. The step-by-step workflow, including rebase and merge expectations, is
in [AGENTS.md](./AGENTS.md#recommended-workflow-pull-requests); it applies to
human contributors and coding agents alike.

## Releasing

Maintainers publish the workspace crates and the release binaries following the
[release steps in AGENTS.md](./AGENTS.md#release): pre-flight verification,
publishable manifest metadata, the workspace version bump,
`cargo build --workspace --release --locked`, `cargo publish --workspace
--dry-run --locked`, the real publish, and the tagged GitHub release.

## Compatibility expectations

Changes to chunk identities, tokenizer or grammar versions, file eligibility,
incremental checkpoints, index manifests, or backend scoring are compatibility
changes. Update the relevant design and benchmark evidence and make migrations
fail closed with a clear reindex requirement.

## Security and privacy

- Never commit `.env`, model credentials, gateway URLs containing private
  infrastructure identifiers, repository source, or generated indexes.
- Use generated or publicly licensed fixtures.
- Run `gitleaks git . --redact` before publishing changes that touch provider
  configuration, examples, scripts, or documentation.
- Report vulnerabilities according to `SECURITY.md`, not in a public issue.

By contributing, you agree that your contribution is licensed under the MIT
License.
