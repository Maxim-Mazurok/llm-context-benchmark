# Contributing

Thanks for helping improve LLM Context Benchmark. Bug reports, reproducible
benchmark findings, documentation fixes, runtime adapters, and focused pull
requests are welcome.

## Development setup

The core project requires Python 3.11 or newer. Install the test environment:

```bash
uv sync --extra test
uv run pytest -q
```

Running real Apple Silicon inference also requires the MLX dependencies:

```bash
uv sync --extra test --extra mlx
```

The results viewer requires a current Node.js release:

```bash
npm ci
npm run test:viewer
npm run build
```

Use the mock adapter for fast end-to-end development without a model:

```bash
uv run llm-context-bench --adapter mock \
  --max-context 3000 --chunk-tokens 1000 \
  --short-decode-tokens 16 --long-decode-tokens 32 \
  --long-decode-interval 2000
```

## Repository layout

- `src/llm_context_benchmark/`: benchmark runner, adapters, monitoring, recovery,
  reporting, and CLI.
- `viewer/`: React/Chart.js results browser and its Node.js backend.
- `tests/`: Python behavior and end-to-end tests.
- `viewer/server/*.test.mjs`: viewer data and queue tests.
- `bin/llm-context-bench`: launcher that can use OMLX's bundled runtime.

Generated `runs/`, virtual environments, dependencies, and viewer builds are
intentionally ignored and must not be committed.

## Runtime adapter contract

An adapter implements `load`, `make_input_tokens`, `restore_context`,
`append_and_decode`, `mlx_metrics`, `reset_peak_memory`, and `metadata` from
`adapters/base.py`.

The important behavioral contract is that `append_and_decode` extends one
persistent cache and calls `on_prefill_complete` exactly when decode begins.
Resume support must reconstruct the exact completed-cycle token checkpoint.

## Making changes

- Keep benchmark semantics explicit and reproducible.
- Do not silently merge results produced with incompatible models, decode
  modes, cache settings, input profiles, or measurement schemas.
- Add regression tests for fixes and tests for new scheduling or aggregation
  behavior.
- Run both test suites and the production viewer build before opening a pull
  request.
- Keep the README focused on installation, operation, interpretation, and
  safety for benchmark users. Put implementation guidance here.
- Avoid committing local model paths, machine identifiers, raw runs, prompts,
  telemetry, or other personal data.

## Pull requests

Describe the user-visible behavior, measurement impact, and verification
performed. If a change modifies recorded fields or benchmark semantics, note
whether old and new runs remain comparable and update the schema version when
needed.

Small, reviewable pull requests are preferred. Open an issue first for major
changes to benchmark methodology, storage formats, or runtime support.

## AI-assisted development

AI-assisted contributions are welcome. Contributors remain responsible for
reviewing, testing, licensing, and accurately describing submitted work. Do
not include generated material copied from incompatible sources or private
context that should not be published.
