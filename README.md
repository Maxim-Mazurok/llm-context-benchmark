# LLM Context Benchmark

Find the largest *usable* context for a local LLM on Apple Silicon, not merely the largest prompt that allocates successfully.

The benchmark keeps one model and one KV cache alive while it repeatedly:

1. appends about 5,000 input tokens;
2. starts with a sustained 1,000-token decode to establish an accurate baseline;
3. generates 256-token decode samples between sustained 1,000-token probes around each 25,000-token milestone;
4. ends a planned model/context-limit run with another sustained decode at the absolute maximum;
5. samples process, MLX, compression, swap, and system memory every 100 ms.

Generated tokens remain in the context. Every result and chart therefore uses the **actual total context**, not just cumulative prompt tokens.

The final sustained probe is guaranteed when the endpoint is known in advance, such as a declared model limit or `--max-context`. A sudden allocation failure, OS termination, or manual interruption cannot safely schedule an after-the-fact final probe.

## Quick start with installed OMLX

The included launcher uses the Python, MLX, `mlx-lm`, Transformers, and psutil versions packaged inside the globally installed `/Applications/oMLX.app`. It does not require an OMLX source checkout or a separate project environment:

```bash
cd ~/llm-context-benchmark
./bin/llm-context-bench \
  --model ~/.omlx/models/mlx-community/Llama-3.2-1B-Instruct-4bit
```

For a shorter validation run:

```bash
./bin/llm-context-bench \
  --model ~/.omlx/models/mlx-community/Llama-3.2-1B-Instruct-4bit \
  --chunk-tokens 1000 --short-decode-tokens 32 \
  --long-decode-interval 3000 --long-decode-tokens 64 \
  --max-context 5000 --sample-ms 50
```

### Benchmark every installed OMLX model

Preview which locally installed models are eligible:

```bash
./bin/llm-context-bench --list-omlx-models
```

Run all eligible models sequentially with the same benchmark settings:

```bash
./bin/llm-context-bench --all-omlx-models
```

The model listing also reports speculative capabilities and the selected local drafter, when available. Raw autoregressive decoding remains the default.

### Speculative decoding

Let the benchmark choose the best supported local path:

```bash
./bin/llm-context-bench \
  --model ~/.omlx/models/Qwen/Qwen3.5-0.8B \
  --speculative
```

`auto` prefers a valid OMLX Lightning MTP head: embedded weights, a prefixed local `mtp.safetensors` sidecar, or a compatible MTP-only model elsewhere in the OMLX library. MTP helpers are paired by their inner text architecture and dense/MoE dimensions, then ranked by model release and tokenizer affinity. This handles helpers whose outer config is labeled `qwen3_5_mtp` while the target is labeled `qwen3_5_moe`. One helper may serve multiple compatible targets, such as the original and uncensored variants of the same Qwen base model. Otherwise the benchmark selects the smallest installed generative model with an identical `tokenizer.json` and a rollback-capable long-context cache, then uses MLX-LM target/draft speculation. Override either choice explicitly:

```bash
# Force an ordinary target + draft model pairing
./bin/llm-context-bench \
  --model /path/to/full-attention-target \
  --speculative --speculative-backend mlx-draft \
  --draft-model /path/to/tokenizer-compatible-smaller-draft

# Require an embedded OMLX Lightning MTP head
./bin/llm-context-bench \
  --model ~/.omlx/models/Qwen/Qwen3.5-0.8B \
  --speculative --speculative-backend omlx-mtp \
  --num-draft-tokens 2
```

A speculative batch sweep runs only targets with a detected compatible path and reports skipped models:

```bash
./bin/llm-context-bench --all-omlx-models --speculative
```

Run raw and speculative variants as separate, directly comparable results:

```bash
./bin/llm-context-bench --all-omlx-models --also-speculative
```

Models without a compatible speculative path still run once in raw mode. The
maximum MTP draft depth comes from that model's OMLX setting when configured;
otherwise it defaults to 3. `--num-draft-tokens N` overrides it.

Each run records `decode_mode`, `speculative_backend`, `draft_model`, `mtp_sidecar`, `mtp_helper`, and `num_draft_tokens` in `run-metadata.json`; the results viewer adds the backend/drafter to the model label. OMLX model settings are consulted for enabled native MTP, and the checkpoint is independently checked for both declared MTP heads and MTP tensors. Sidecars and external MTP helpers are loaded through a temporary overlay; the original OMLX model directories are never modified.

Ordinary MLX target/draft speculation requires cache rollback. Models with recurrent/linear-attention caches, or sliding caches that cease to be trimmable at long context, are not advertised for that backend. This prevents a pairing that works only for a short prompt from failing partway through a context benchmark.

Specialized DFlash and VLM assistant checkpoints are deliberately not passed to MLX-LM's ordinary causal-draft API. They use different OMLX engines and cache contracts; the benchmark reports them as auxiliary rather than producing a misleading “speculative” result.

All normal controls apply, so a bounded comparison sweep can use, for example:

```bash
./bin/llm-context-bench --all-omlx-models \
  --max-context 65536 --swap-stop-gib 4
```

Discovery defaults to `~/.omlx/models`; override it with `--omlx-models-dir`. Provider directories and individual models may be symlinks, including links into `~/.lmstudio/models`; discovery, helper matching, direct `--model` runs, and metadata retain the OMLX library identity while loading the resolved files. Batch mode launches each model in a fresh subprocess so model allocations and caches are fully released between runs. A supplied `--output` is treated as the parent directory for the per-model run folders.

Directories identified as MTP, assistant, draft/speculator, embedding, reranker, reward, or other non-generative model configurations are reported and skipped as standalone benchmark targets. Compatible MTP-only directories still participate as helper models. A main model is not excluded merely because its directory also contains an `mtp.safetensors` sidecar.

Or create a standalone environment using upstream `mlx-lm`:

```bash
cd ~/llm-context-benchmark
uv sync --extra mlx
uv run llm-context-bench --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

## Useful controls

```text
--chunk-tokens 5000
--short-decode-tokens 256
--long-decode-tokens 1000
--long-decode-interval 25000
--sample-ms 100                 # 50 is also reasonable
--max-context N                 # optional safety/test cap
--no-max-context                # clear an old cap when resuming
--swap-stop-gib 4               # primary default safety stop
--severe-decode-ratio 0.5       # optional; disabled unless specified
--stop-on-critical-pressure     # optional; disabled by default
--practical-decode-ratio 0.8
--practical-swap-growth-gib 1
--kv-bits {4,8}                 # optional quantized KV cache
```

## Import useful-task observations

Eval Workbench can opt into correlated request telemetry when it runs real,
independent benchmark tasks through oMLX. Import one completed Workbench run:

```bash
./bin/llm-context-bench \
  --import-workbench-run ~/llm-eval-workbench/benchmark-runs/<run-directory>
```

The importer joins each Workbench result to its oMLX request by exact attempt
ID. It preserves task score/status, exact prompt and completion token counts,
oMLX timing/rate fields, and peak host/MLX values from the request's 100 ms
sample window. These records remain independent useful-task observations; they
are never represented as phases of one persistent cache.

The output is a standalone bundle under `runs/`:

- `useful-tasks.json` and `useful-tasks.csv`: normalized observations;
- `useful-task-summary.json`: counts, score, maximum prompt, models, and sources;
- `report.md`: concise task table and provenance boundary;
- `useful-task-dashboard.html`: local responsive observation ledger.

Use `--output <directory>` to choose the bundle directory.

## Run the separate capacity tail

Synthetic high-context capacity testing remains a direct `mlx-lm` persistent
cache run. Start it normally with a staged `--max-context`, then resume the
checkpoint explicitly as a capacity tail:

```bash
./bin/llm-context-bench --capacity-tail runs/<run-directory> \
  --max-context 131072
```

Repeat with larger caps, or finish at the model/runtime limit:

```bash
./bin/llm-context-bench --capacity-tail runs/<run-directory> --no-max-context
```

`--capacity-tail` is the semantic resume entry point for this synthetic tail;
it uses the existing exact token checkpoint, cache reconstruction, sampling,
and interruption rollback. It does not append useful-task requests into the
cache or observation bundle.

By default, throughput degradation, compression, and macOS memory-pressure status are recorded but do **not** stop the run. The benchmark stops only after peak swap growth reaches 4 GiB, at a known model/runtime context limit, on allocation/runtime failure, or on manual interruption. Peak growth is retained even if macOS later reclaims swap before the cycle ends. Speed-based and critical-pressure stopping remain available as explicit opt-ins. `Ctrl-C` rolls back an in-flight partial cycle, writes the report from completed cycles, and preserves a resumable token checkpoint.

Resume a gracefully stopped or staged run from its last completed cycle:

```bash
./bin/llm-context-bench --resume runs/<run-directory> --max-context 131072
```

Omit the target on a new run, or clear a saved target when resuming, to continue
until the model/runtime limit:

```bash
./bin/llm-context-bench --resume runs/<run-directory> --no-max-context
```

The model is reloaded and its live cache is reconstructed from the saved token sequence. Raw and OMLX Lightning MTP caches are supported. Runs created before checkpoint support cannot be resumed exactly, but their existing raw samples can be recovered for viewing:

```bash
./bin/llm-context-bench --recover-run runs/<older-interrupted-run>
```

## Outputs

Each run gets its own timestamped directory under `runs/` unless `--output` is supplied:

- `samples.csv` and `samples.jsonl`: raw 100 ms time series, flushed continuously;
- `phases.csv` and `phases.json`: prefill/decode duration, throughput, RSS mean/median/p95/max, MLX peaks, compression and swap deltas;
- `tokens.csv` and `tokens.json`: per-token decode intervals and instantaneous throughput;
- `summary.json`: max tested, known hard limit, no-swap and <1 GiB swap limits, degradation crossings, and practical context;
- `checkpoint.json` and `run-state.json`: last completed token sequence and live/interrupted/completed status for safe resume and UI updates;
- `memory-vs-context.svg` and `throughput-vs-context.svg`: dependency-free charts;
- `dashboard.html`: interactive prefill, decode, and memory charts with exact point values;
- `prefill-speed-vs-context.svg`, `decode-speed-vs-context.svg`, and `memory-indicators-vs-context.svg`: separate static charts;
- `report.md`: concise human-readable result.

Raw records are retained so practical-context criteria can be recalculated without repeating a long model run.

The live progress display emphasizes unified-memory signals rather than process RSS: MLX active memory, system-available memory, compressed memory, current total swap, peak swap growth since model load, swap-out traffic, and macOS pressure status. Phase records also retain minimum available memory and the fraction of samples spent at warning or critical pressure. RSS remains in the raw files for completeness, but Metal-backed pageable allocations make it misleading on Apple Silicon.

### Open a dashboard with run deletion enabled

A browser cannot delete local files from a directly opened `file://` dashboard. Start the localhost-only viewer to enable the dashboard's **Delete run** button:

```bash
./bin/llm-context-bench --serve-run runs/20260917-002440 --open-browser
```

Deletion requires a confirmation dialog and moves the entire run directory to macOS Trash, where it remains recoverable until Trash is emptied. The viewer binds only to `127.0.0.1` and uses a random per-session token for the deletion request.

## Results viewer

Start the local results browser with one command:

```bash
npm install
npm run dev
```

It prefers port `5173`, then tries `5174`, `5175`, and so on until one is available. It listens on all network interfaces and prints both the localhost and LAN URLs. Browser requests use same-origin `/api` paths, so the printed network URL reaches the backend without localhost leakage or separate proxy configuration. The viewer discovers live, interrupted, and completed folders under `runs/`; no import step is needed.

For a lower-memory server without Vite, hot reload, or source transforms, build
once and run the production viewer:

```bash
npm run build
npm start
```

The **Run benchmarks** panel can launch one or many installed OMLX models, choose raw, speculative, or both variants, override the MTP draft depth, stop the active model gracefully, and resume checkpointed runs. **Round-robin stages** runs every selected model to 8K, then every model to 16K, then 32K, and so on. With no final target it adds one last uncapped continuation to the model/runtime limit. **Finish each model** keeps the traditional one-model-at-a-time schedule. Intermediate phase results are written after every cycle and refresh in the browser while inference continues.

The opening ledger identifies every benchmark by model and the run's local date/time, with its maximum context, final decode rate, peak MLX allocation, swap growth, and stop reason. Open a run for detailed charts. Every chart uses the same comparison controls, allowing you to:

- overlay a metric from several models or runs;
- overlay different metrics from one run, with separate unit axes where needed;
- remap the horizontal axis to another metric, such as swap growth versus prefill speed;
- add or remove series without leaving the run detail page.

Short and sustained decode probes share one decode line. Sustained probes use larger dots. Prefill points are the rate of each individual append phase—not a cumulative average from the beginning of the run—so local slowdowns or recoveries remain visible.

## What “continuous” means

The direct `mlx-lm` adapter creates `make_prompt_cache(model)` once and updates it in place for the entire benchmark. `generate_step` leaves its final returned token one token ahead of the cache; the adapter deliberately feeds that token back at the start of the next append. Thus no token is dropped or duplicated between measurement phases.

This is why the default adapter runs in the model process rather than through OMLX's OpenAI HTTP endpoint: ordinary independent HTTP requests do not expose an unambiguous, client-owned live KV cache. The launcher uses the exact MLX/`mlx-lm` layer shipped by the installed OMLX app.

If OMLX is installed somewhere other than `/Applications/oMLX.app`, set `OMLX_APP_PATH` to its `.app` path. The normal `omlx` shell command is a native CLI launcher and does not directly provide a general-purpose Python subcommand, which is why this project launcher selects the packaged runtime explicitly.

## Runtime adapter interface

A new adapter implements `load`, `make_input_tokens`, `append_and_decode`, `mlx_metrics`, `reset_peak_memory`, and `metadata` from `adapters/base.py`. The important contract is that `append_and_decode` extends one persistent cache and calls `on_prefill_complete` exactly when decode begins. The included `mock` adapter supports fast end-to-end testing:

```bash
./bin/llm-context-bench --adapter mock \
  --max-context 3000 --chunk-tokens 1000 \
  --short-decode-tokens 16 --long-decode-tokens 32 --long-decode-interval 2000
```

## Measurement notes

- Sampling happens in one lightweight thread using in-process macOS/psutil APIs and MLX counters; it never repeatedly launches `ps`, `vm_stat`, or `sysctl`.
- A tiny four-token throwaway generation warms the model/JIT before the persistent benchmark cache is created; warmup tokens never enter the measured context.
- Prefill throughput includes the one-token cache catch-up after the first cycle and time-to-first-token work. The appended input count and catch-up count are stored separately.
- Decode throughput excludes time-to-first-token and is `(N - 1) / (last_token_time - first_token_time)`.
- `memory_pressure_level` uses the kernel pressure flag when exposed and an available-memory fallback otherwise.
- The no-swap default tolerates 64 MiB of background noise. All swap criteria are growth relative to the post-load baseline, not total machine swap already in use.
- Decode measurements are raw target-model autoregressive generation unless `--speculative` is supplied. Speculative metadata is always written so raw, external-draft, and native-MTP results remain distinguishable.

## Tests

```bash
uv sync --extra test
uv run pytest -q
```
