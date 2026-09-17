# Security policy

## Reporting a vulnerability

Open a [GitHub issue](https://github.com/Maxim-Mazurok/llm-context-benchmark/issues)
with a concise reproduction, the affected version or commit, and the potential
impact. Security reports are handled through the same public issue tracker as
other bugs.

Do not include credentials, access tokens, private model data, personal
information, or other secrets in an issue. Revoke any credential immediately
if it has already been exposed.

## Viewer network model

The results viewer is a trusted-network administration tool, not a hardened
multi-user service. By default it listens on all network interfaces and does
not authenticate requests. Anyone who can reach it can inspect local model and
benchmark metadata and can start, stop, or resume resource-intensive runs.

Use it only on a trusted network. To restrict access to the local machine:

```bash
HOST=127.0.0.1 npm run dev
```

Do not expose the viewer directly to the public internet.

The separate single-run dashboard deletion server binds only to `127.0.0.1`,
uses a random per-session token, and moves deleted runs to Trash.

## Sensitive benchmark output

Generated run bundles may contain absolute model/output paths, platform
details, model names, task and attempt identifiers, and error messages. Review
and scrub run artifacts before publishing them. The repository ignores
`runs/` by default.
