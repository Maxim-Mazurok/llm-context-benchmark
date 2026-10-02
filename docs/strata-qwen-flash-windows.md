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
- Strata v0.1.14
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

Strata v0.1.13 introduced shared prompt scratch and `--prefill auto`; v0.1.14 is the
tested release. The supported path borrows expert-cache storage when enough slots are
available. Otherwise server mode chooses a 1,024-token own-buffer fallback. Do not add
the experimental local `Prefill::release()` patch: upstream's allocator owns this
lifetime now.

Measured results:

| Engine and prompt path | Prefill | Decode | Interpretation |
|---|---:|---:|---|
| Older engine, chunk 128, 4,096-token probe | 49.40 tok/s | 14.76 tok/s single-token sample | Original production setting |
| Older engine, chunk 1,024, 4,096-token probe | 188.71 tok/s | 12.89 tok/s single-token sample | Fast safe fixed chunk |
| Older engine, chunk 1,920, 4,096-token probe | 217.86 tok/s | 2.26 tok/s | VRAM paging; reject |
| Older engine, chunk 1,024, 11,687-token coding prompt | 162.72 tok/s | 19.84 tok/s | Sustained decode control |
| Experimental buffer release, chunk 1,920, same coding prompt | 233.39 tok/s | 19.14 tok/s | Root-cause proof only |
| **v0.1.14, auto, 6,953-token API probe** | **290.20 tok/s** | **20.30 tok/s** | Current supported configuration |

The current API prompt is not token-identical to the historical native probes, so the
table is an operational comparison rather than a controlled engine benchmark. It does
show the intended outcome: much faster cold prefill without sacrificing sustained
decode. Normal observed decode across longer agent sessions was about 18-24 tokens/s.
Prefix-reused requests can report low rates for a tiny newly read suffix because fixed
request overhead dominates; keep reused and read token counts when interpreting logs.

The model's full context capacity is 262,144 tokens. Upstream identifies 259,943 as
the full usable prompt shape; the remaining capacity accommodates generation and
server overhead. A controlled prefix-growing test reached that prompt length on the
reference laptop:

| Prompt tokens | Reused | Fresh prefill | Decode | Total time |
|---:|---:|---:|---:|---:|
| 32,768 | 0 | 333.8 tok/s | 24.1 tok/s | 108.8 s |
| 131,072 | 16,384 | 315.9 tok/s | 21.4 tok/s | 375.1 s |
| **259,943** | **114,688** | **264.0 tok/s** | **20.9 tok/s** | **562.5 s** |

At the maximum prompt, fresh-prefill speed was 21% below the 32K point and decode was
13% lower. There was no decode collapse. Q4 KV streaming kept 96.23% of block reads
in VRAM and read 890.2 MiB from RAM. After completion, the engine had a 41.59 GiB
working set, the system had 8.12 GiB physical RAM free at 87.2% memory load, and the
GPU had 1,216 MiB VRAM free. This fits, but the RAM margin is narrow enough that the
memory watchdog remains strongly recommended.

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

Verify both the checkout and installed engine report the expected release before a
long run. The measurements in this document use v0.1.14.

## Performance configuration

Let Strata setup generate the model-specific JSON, then verify its `args` contain this
core configuration:

```json
[
  "--expert-cache", "auto",
  "--vram-reserve-mib", "700",
  "--prefill", "auto",
  "--spec", "4",
  "--spec-min-p", "0.5",
  "--max-context", "262144",
  "--kv", "q4_0",
  "--kv-resident", "32768"
]
```

The 700 MiB reserve is the tested 8 GB value. A 1,280 MiB reserve left no expert slots
on this system and server mode refused to start. Do not reduce the reserve merely to
force a larger chunk: leave `--prefill auto` in control and retain headroom for MTP,
the verifier, display use, and transient allocations.

The 262K configuration stores the complete Q4 K/V cache in about 1.69 GiB of pinned
host RAM while limiting each QSA layer to 32,768 resident GPU cells. On this system,
automatic prefill selected a 512-token chunk, borrowed 336 expert-cache slots for its
0.55 GiB workspace, and started with 512 expert slots. Strata setup may conservatively
clamp IQ3_XXS to 131K on systems below 90 GiB RAM; that is a setup heuristic rather
than an engine limit. Keep 131K when other memory-heavy applications must run beside
Strata.

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
`/health` succeeds. It does not replace an existing listener.

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

- **Zero expert slots:** increase available VRAM or reduce the reserve only to the
  tested hardware-appropriate value. On the reference 8 GB GPU, 700 MiB produced 169
  slots; 1,280 MiB produced zero.
- **Prompt path allocates its own buffers:** this is a valid automatic fallback when
  too few cache slots are lendable. In v0.1.14 server mode it uses chunk 1,024.
- **Decode falls near 2 tok/s:** check dedicated/shared GPU memory and confirm the
  deployed engine is v0.1.14 with `--prefill auto`; this resembles WDDM paging from
  retained or oversized workspaces.
- **MTP allocation fails:** confirm `STRATA_ARENA_REGISTER=0` reached the engine and no
  other inference process owns VRAM.
- **Startup takes several minutes:** loading and locking the approximately 40 GiB
  expert arena is expected. Inspect both the run's server stderr and the engine log
  named by the Strata config before changing settings.