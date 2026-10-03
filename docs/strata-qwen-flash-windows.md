# Fast Qwen3.8 Flash Next with Strata on Windows

This runbook records the September 2026 investigation of long-context Strata
performance on an 8 GB NVIDIA laptop GPU. It covers the supported upstream
configuration, safe detached startup, and a repeatable speed probe. Raw model paths,
prompts, telemetry, and run directories remain local and must not be committed.

## Reference system

- NVIDIA RTX PRO 2000 Blackwell Laptop GPU, 8,151 MiB VRAM
- Intel Core Ultra 9 285H, AVX2 expert kernels
- 64 GiB system RAM
- Qwen3.8 Flash Next IQ3_XXS native expert pack
- Strata v0.1.38 server and engine, with a side-by-side v0.1.14 rollback engine
- 262,144-token context with Q4_0 KV and 32,768 resident cells per QSA layer
- MTP speculation: four drafts, minimum probability 0.5

These settings are a measured reference, not universal defaults. Larger GPUs may lend
more expert-cache slots to prefill and select a larger automatic chunk.

## Findings

KV capacity and prefill workspace are separate VRAM consumers. Increasing the prompt
chunk improves GPU utilization and amortizes expert streaming, but older Strata builds
retained chunk-sized prompt buffers during decode. On an already-full 8 GB GPU, Windows
could page those allocations instead of reporting an OOM, reducing decode from about
20 tokens/s to about 2 tokens/s.

Strata v0.1.13 introduced shared prompt scratch and `--prefill auto`. The current
server and engine are v0.1.38. Its default 700 MiB reserve was slower for this
prompt-heavy workload on the reference 8 GB GPU because it crossed an automatic
workspace threshold. A measured 590 MiB reserve restores the larger prompt workspace
without paging or allocation failures. Do not restore the old experimental
`Prefill::release()` patch: upstream's allocator owns this lifetime now.

There is no standalone upstream changelog. Relevant commits between v0.1.14 and
v0.1.38 add grouped prompt expert launches, Q4_0 tensor-core prompt attention,
unbuffered Windows model loading, split prompt-buffer improvements, long-context RAM
accounting fixes, request timing and monitoring, cancellation and API fixes,
conversation caching, low-RAM modes, and server restart/lifecycle improvements.

Measured results:

| Engine and prompt path | Prefill | Decode | Interpretation |
|---|---:|---:|---|
| Older engine, chunk 128, 4,096-token probe | 49.40 tok/s | 14.76 tok/s single-token sample | Original production setting |
| Older engine, chunk 1,024, 4,096-token probe | 188.71 tok/s | 12.89 tok/s single-token sample | Fast safe fixed chunk |
| Older engine, chunk 1,920, 4,096-token probe | 217.86 tok/s | 2.26 tok/s | VRAM paging; reject |
| Older engine, chunk 1,024, 11,687-token coding prompt | 162.72 tok/s | 19.84 tok/s | Sustained decode control |
| Experimental buffer release, chunk 1,920, same coding prompt | 233.39 tok/s | 19.14 tok/s | Root-cause proof only |
| v0.1.14, auto, 6,953-token API probe | 290.20 tok/s | 20.30 tok/s | Previous supported configuration |

The current API prompt is not token-identical to the historical native probes, so the
table is an operational comparison rather than a controlled engine benchmark. It does
show the intended outcome: much faster cold prefill without sacrificing sustained
decode. Normal observed decode across longer agent sessions was about 18-24 tokens/s.
Prefix-reused requests can report low rates for a tiny newly read suffix because fixed
request overhead dominates; keep reused and read token counts when interpreting logs.

The model's full context capacity is 262,144 tokens. Upstream identifies 259,943 as
the full usable prompt shape; the remaining capacity accommodates generation and
server overhead. The final controlled A/B used isolated local builds, the same
v0.1.38 server wrapper, identical engine arguments, cold starts, the exact same prompt
construction, and 256 generated tokens. The full 32K -> 131K -> 259,943 sequence was
repeated for each engine:

| Release | Prompt tokens | Reused | Fresh prefill | Decode | Total time |
|---|---:|---:|---:|---:|---:|
| **v0.1.14** | **32,768** | **0** | **355.5 tok/s** | **23.0 tok/s** | **103.3 s** |
| **v0.1.14** | **131,072** | **16,384** | **335.4 tok/s** | **25.4 tok/s** | **352.1 s** |
| **v0.1.14** | **259,943** | **114,688** | **282.9 tok/s** | **24.3 tok/s** | **523.9 s** |
| v0.1.38, cache 3 | 32,768 | 0 | 313.8 tok/s | 24.7 tok/s | 114.8 s |
| v0.1.38, cache 3 | 131,072 | 16,384 | 300.1 tok/s | 25.9 tok/s | 392.1 s |
| v0.1.38, cache 3 | 259,943 | 114,688 | 267.7 tok/s | 24.6 tok/s | 553.1 s |

With the default reserve, v0.1.38 took 11.1% longer at 32K, 11.4% longer at 131K, and
5.6% longer at the maximum prompt than v0.1.14. Its decode was 1-7% faster, but prefill
was 5-12% slower and dominates agent request time. There was no decode collapse in
either release.

The mechanism is visible at startup: v0.1.14 keeps 512 expert slots and selects a
512-token prompt chunk, while v0.1.38 keeps 448 slots and selects 256 tokens. Asking
v0.1.38 for a fixed 512-token chunk is safely clamped back to 256 because its buffers
do not fit the cache. The controlling condition is the mandatory 128-slot resident
floor, not the 85-90% lending cap.

A bounded v0.1.38 test reduced the reserve from 700 to 590 MiB. This produced 513
expert slots, restored the 512-token automatic chunk, borrowed 336 slots for a 0.55
GiB workspace, and left 287 MiB free after loading MTP and verifier buffers. The same
32K -> 131K -> 259,943 sequence then produced:

| Release | Prompt tokens | Reused | Fresh prefill | Decode | Total time |
|---|---:|---:|---:|---:|---:|
| **v0.1.38, 590 MiB** | **32,768** | **0** | **471.1 tok/s** | **24.9 tok/s** | **79.8 s** |
| **v0.1.38, 590 MiB** | **131,072** | **16,384** | **451.7 tok/s** | **24.4 tok/s** | **264.4 s** |
| **v0.1.38, 590 MiB** | **259,943** | **114,688** | **396.0 tok/s** | **25.9 tok/s** | **376.8 s** |

This is 23-28% faster end to end than the matched v0.1.14 runs. All three requests
completed with zero RAM/file expert fallback, stable system memory, and no watchdog,
allocation, or decode failure. The 287 MiB internal headroom is intentionally narrow;
590 MiB is hardware-specific and must not be generalized to other cards or workloads.

The initial untuned v0.1.38 process with six checkpoints reached 95% memory load after
the 131K probe and the production watchdog stopped it. Three checkpoints retain the
synthetic test's useful 114,688-token prefix while staying safe in the full-context
test. Keep the watchdog enabled at 262K. The side-by-side v0.1.14 engine remains the
rollback option if a driver, display workload, or future engine changes the margin.

### Future headless Linux option

Native headless Linux is primarily a headroom and stability option, not an expected
repeat of the Windows 256-to-512-token chunk improvement. Tuned Windows already holds
98-99% GPU utilization without paging. Estimate 2-8% higher prefill, 0-5% higher
decode, and 2-7% lower end-to-end latency when the automatic chunk remains 512 tokens.
A realistic upper bound is about 10-15% end to end only if reclaimed VRAM crosses
another useful automatic-workspace threshold.

Reclaiming at least 110 MiB would already allow the safer 700 MiB reserve while
retaining the current 513-slot, 512-token layout. Prefer spending recovered VRAM on
margin before enlarging the cache. Validate a Linux configuration from a cold start at
32K, 131K, and 259,943 tokens, recording the selected chunk, cache slots, final free
VRAM, expert fallback, and decode rate. WSL is not equivalent because CUDA still uses
the Windows WDDM driver. Driving every display from the integrated GPU may recover
some dGPU memory on Windows, but is not as deterministic as native headless Linux.

### Local patches and reproducibility

The v0.1.38 performance recovery itself is configuration-only: a 590 MiB reserve
crosses the 513-slot threshold, and three prompt checkpoints preserve useful prefix
reuse. No prefill or cache-sizing source code was changed for the speed result.

Production also uses local source changes that must be preserved across upgrades:

- `src/core/pinned.cu` adds `STRATA_ARENA_REGISTER=0`. The launcher sets it to avoid
  registering the approximately 40 GiB arena with CUDA on this Windows system.
- `serve/chat_template.jinja`, `serve/frontend.py`, and `serve/server.py` restore
  `continue_final_message` behavior required by the API client.
- `serve/server.py` also adds opt-in sanitized JSONL request metrics through
  `STRATA_METRICS_LOG`, including fresh-prefill throughput.
- `serve/test_server.py` verifies continuation rendering and validation.

The deployed engine's `BUILD.json.src` is a content fingerprint, not a Git commit.
For the tested build it is `474634f09371eae1`, exactly matching the current patched
engine sources at v0.1.38. Commit the five tracked source files to the Strata fork
before the next update so the engine and server can be rebuilt reproducibly. Do not
commit local model configuration, binaries, model paths, prompts, or run telemetry.
Keep the arena opt-out, continuation support, and metrics as separate commits when
practical so each can be reviewed or dropped if upstream adds an equivalent fix.

## Install or update

Inspect local changes before updating. Preserve them with a commit or stash, fetch
tags, then fast-forward to the desired release:

```powershell
$StrataRoot='C:\path\to\Strata'
git -C $StrataRoot status --short
git -C $StrataRoot fetch origin --tags
git -C $StrataRoot pull --ff-only
git -C $StrataRoot describe --tags --exact-match HEAD
```

For an installation whose `BUILD.json` records `source: local`, rebuild the deployed
engine from that checkout:

```powershell
Push-Location $StrataRoot
try {
    & .\.venv\Scripts\python.exe -c "import setup; setup.update_installed_engine(setup.PREBUILT_URL)"
} finally {
    Pop-Location
}
```

Verify both the checkout and configured engine report the expected release before a
long run. The selected layout keeps production v0.1.38 in `engine/` and the rollback
v0.1.14 binary in `engine-v0.1.14/`; the model config's `exe` selects the former. This
prevents a server update from destroying either tested binary.

## Performance configuration

Let Strata setup generate the model-specific JSON, then verify its `args` contain this
core configuration:

```json
[
  "--expert-cache", "auto",
  "--vram-reserve-mib", "590",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--max-context", "262144",
  "--kv", "q4_0",
  "--kv-resident", "32768",
  "--prompt-cache", "3"
]
```

Set the config's top-level `"draft_vocab": "en"` and refresh the MTP vocabulary by
starting once through Strata setup with `--draft-vocab en`. This supported English/code
subset reduces the draft head from 180 MiB to 68 MiB on this model and is 1-2% faster
for English according to upstream. Use the larger default subset when CJK generation
speed matters.

The 590 MiB reserve is the end-to-end tested value for this exact 8 GB system. It is
the smallest relaxation that crosses the 512-token workspace threshold while retaining
287 MiB of engine-reported headroom. Leave `--prefill auto` in control. A different
GPU, display workload, model, MTP vocabulary, or engine build requires a fresh staged
startup, 32K, 131K, and maximum-context validation under the watchdog.

The 262K configuration stores the complete Q4 K/V cache in about 1.69 GiB of pinned
host RAM while limiting each QSA layer to 32,768 resident GPU cells. Tuned v0.1.38
starts with 513 expert slots, selects a 512-token chunk, and borrows 336 slots for a
0.55 GiB workspace. Three prompt checkpoints preserve useful growing-prefix reuse
without returning to the unsafe six-checkpoint RAM footprint. Keep 131K when other
memory-heavy applications must run beside Strata.

Set `STRATA_ARENA_REGISTER=0`. Registering the approximately 40 GiB host expert arena
with CUDA consumed VRAM needed by MTP on the reference machine. The launcher below
sets it only for the child process. It also sets `STRATA_METRICS_LOG` inside the run
directory and attaches the repository's stop-first memory watchdog to `strata.exe`.

## Start detached

From this repository on Windows PowerShell:

```powershell
.\scripts\start-strata-fast.ps1 `
  -StrataRoot 'C:\path\to\Strata' `
  -ConfigPath 'C:\path\to\Strata\strata-iq3_xxs.json' `
  -MaximumMemoryLoadPercent 95 `
  -MinimumAvailableRamGiB 3
```

The launcher refuses configs without `--prefill auto`, `--expert-cache auto`, or an
explicit VRAM reserve. It creates an untracked `runs/strata-TIMESTAMP` directory,
waits for the actual engine child, starts the watchdog, and returns only after
`/health` succeeds. It does not replace an existing listener. When the port is already
healthy, it returns success only after verifying the server and config paths, engine
executable and version, model, context limit, and watchdog. A stale or differently
configured service must be stopped explicitly before launching the requested setup.

To launch without automatic RAM-pressure termination, add the explicit switch:

```powershell
.\scripts\start-strata-fast.ps1 `
  -StrataRoot 'C:\path\to\Strata' `
  -ConfigPath 'C:\path\to\Strata\strata-iq3_xxs.json' `
  -DisableMemoryWatchdog
```

Telemetry and `STRATA_ARENA_REGISTER=0` remain enabled. The run's `launch.json`
records `memory_watchdog_enabled: false`, a null watchdog PID, and null inactive
thresholds. Without the watchdog, Windows may become unresponsive or page heavily
under host-memory pressure; monitor the system and stop Strata manually if needed.

Check the endpoint directly:

```powershell
Invoke-RestMethod http://127.0.0.1:8081/health | ConvertTo-Json
```

Stop only the verified Strata process tree when the GPU is needed elsewhere:

```powershell
.\scripts\stop-strata.ps1 -StrataRoot 'C:\path\to\Strata'
```

The stop script checks that the port owner is the `serve/server.py` under the supplied
Strata checkout. It refuses to terminate an unrelated service on the same port.

Do not run another GPU inference engine concurrently. The watchdog intentionally
terminates only the recorded Strata engine after two consecutive unsafe RAM samples;
it is not a restart supervisor.

## Measure the active server

Run a unique cold prompt and parse Strata's native timing line:

```powershell
.\scripts\benchmark-strata.ps1 `
  -Endpoint http://127.0.0.1:8081 `
  -EngineLogPath 'C:\path\to\Strata\strata-iq3_xxs.log'
```

The probe defaults to roughly 7,000 prompt tokens and 128 generated tokens. It writes
`summary.json` and the API response under an untracked `runs/strata-probe-TIMESTAMP`
directory. Compare `read_tokens`, not total prompt tokens, when prefix reuse is nonzero.
Use at least 128 generated tokens for a meaningful decode rate; one-token decode
figures mostly measure fixed overhead.

To view the durable metrics in the repository dashboard, point the viewer at the
launcher run directory before starting it:

```powershell
$env:STRATA_METRICS_PATH='C:\path\to\llm-context-benchmark\runs\strata-TIMESTAMP\inference-metrics.jsonl'
npm run dev
```

Without that environment variable, the dashboard opens a compact sanitized example
containing measurements only. The repository does not include prompts, completions,
machine paths, local addresses, or the full private run history.

For controlled comparisons, keep model files, context limit, KV type, MTP settings,
prompt token IDs, generated-token count, expert-cache state, and background GPU load
fixed. Report cold and warm prompts separately.

## Troubleshooting

- **Too few expert slots:** increase available VRAM or use only a reserve validated on
  that exact system. On the reference 8 GB GPU, 700 MiB produced 448 variable-sized
  slots and a 256-token chunk; 590 MiB produced 513 slots and a 512-token chunk.
- **Prompt path allocates its own buffers:** this is a valid automatic fallback when
  too few cache slots are lendable. Keep `--prefill auto` rather than forcing a chunk.
- **Decode falls near 2 tok/s:** check dedicated/shared GPU memory and confirm the
  deployed engine is the expected release with `--prefill auto`; this resembles WDDM paging from
  retained or oversized workspaces.
- **MTP allocation fails:** confirm `STRATA_ARENA_REGISTER=0` reached the engine and no
  other inference process owns VRAM.
- **Startup takes several minutes:** loading and locking the approximately 40 GiB
  expert arena is expected. Inspect both the run's server stderr and the engine log
  named by the Strata config before changing settings.