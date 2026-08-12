# CLAUDE.md

Read [AGENTS.md](./AGENTS.md) before making changes. It is the single guide for
coding agents in this repository and covers the crate map, the pinned
toolchain, the verifier, the repository conventions, the recommended
pull-request workflow, and the release and publish steps.

Nothing in this file overrides AGENTS.md; it exists so Claude Code loads that
guide automatically. Quick reminders:

- Verify with `./scripts/verify.sh` and scan with `gitleaks git . --redact`
  before opening a pull request.
- Work on a topic branch and land changes through a pull request against
  `main`; never push to `main` and never self-merge.
- Report only checks you actually ran.
