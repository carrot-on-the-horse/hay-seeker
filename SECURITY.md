# Security policy

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue.

Use GitHub's private vulnerability reporting for this repository. If private
reporting is unavailable, contact the maintainers through a private channel of
the `carrot-on-the-horse` GitHub organization.

Include the affected revision, impact, reproduction steps, and any suggested
mitigation. Please avoid including real credentials or private source code in
the report.

## Secrets and external services

Hay Seeker does not require hosted services for its local DuckDB and static
embedding path. Hosted embedding and Elasticsearch integrations are opt-in.
Keep credentials in an ignored local `.env` or your secret manager; never add
them to fixtures, examples, logs, issues, or benchmark output.
