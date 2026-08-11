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
