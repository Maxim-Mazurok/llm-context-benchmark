# Qwen3.8 Flash Next distributed split search

This note preserves the September 2026 Windows + Apple Silicon investigation in a
portable form. Raw run directories and logs remain intentionally untracked.

## Test system and workload

- Windows parent: 64 GiB RAM, one 8 GiB CUDA GPU
- Apple Silicon worker: M5-class system, 32 GiB unified memory
- Link: direct Thunderbolt IP network
- Model: `Qwen3.8-Flash-Next-UD-Q2_K_XL`, three GGUF shards
- Server context: 150,000 tokens
- KV cache: Q4_0 keys and values
- Concurrency: one slot and one inference stream

Passing the first GGUF shard to llama.cpp is correct. The GGUF split metadata makes
llama.cpp discover the remaining sibling shards; do not pass all shard paths as
separate models.

## Reliability findings

The original freezes were not normal steady-state CPU inference:

1. `--lazy-mode auto` performed repeated on-demand reads from the slow USB HDD.
   The visible pattern was sustained 100% disk activity followed by short CPU bursts.
   Use `--lazy-mode off` for this model.
2. CUDA pinned host buffers retained pages used to upload remote tensors. Setting
   `GGML_CUDA_NO_PINNED=1` makes host buffers pageable.
3. Pageable mmap still left uploaded pages in the Windows process working set. After
   `/health` reports ready, call `trim-process-working-set.py` once. This changed the
   measured post-load trajectory from the memory guard firing below 12 GiB available
   to tens of GiB remaining available.
4. Direct I/O attempted to allocate toward the full model size and was not safe on the
   64 GiB parent.
5. The RPC tensor hash cache needed to be disabled on both peers with
   `GGML_RPC_NO_HASH_CACHE=1` for this patched llama.cpp revision.

The guards are stop-first safety mechanisms, not supervisors. The Windows guard stops
only the recorded llama PID after two consecutive readings below 12 GiB available or
at/above 82% physical load. The Mac guard stops only the recorded RPC PID after two
consecutive readings below 10% free-pressure or above 6 GiB swap.

## Measured placements

All placements kept seven layers on the Windows CUDA device. `Mac / CUDA / CPU` below
describes transformer-layer placement; the remaining layers execute on the Windows
CPU.

| Placement | Result | Prefill | Controlled decode | Stream decode | Important observation |
|---|---|---:|---:|---:|---|
| 24 / 7 / 17 | rejected after sustained use | 64.03 tok/s | 7.88 tok/s | 6.61 tok/s | Passed the short benchmark, then Mac pressure reached 9% twice under agent traffic and the guard correctly stopped RPC. |
| 22 / 7 / 19 | sustained-safe candidate | 59.25 tok/s | 8.32 tok/s | 4.37 tok/s | Held 18–19% Mac free-pressure and left about 36 GiB Windows RAM free during the deterministic benchmark. |

The 22-layer run incurred a 779-second cold full-trace warm-up on the USB HDD. After
the pages became resident, its three prefill stages measured 61.50, 59.47, and 55.13
tok/s. Cold loading must therefore be reported separately from warm inference.

These two points do **not** prove that 22 Mac layers is performance-optimal. Moving two
layers to the Windows CPU reduced prefill by about 7.5%, slightly improved the forced
controlled-decode measurement, and reduced the output-dependent stream result. That
mixed result plus more than 30 GiB of observed parent headroom justifies a CPU-heavier
search on the faster replacement drive.

RPC does not resend remote weights for every token. Weights remain resident on the
remote backend; per-evaluation costs are activation copies, graph/RPC dispatch, and
synchronization at backend boundaries. Moving a layer to the parent removes that tax
but makes the Windows CPU stream and compute that layer's weights on every evaluation.
Free RAM establishes capacity, not which side is faster.

## Next split-search plan

Keep context, KV type, batching, CUDA placement, clocks, and workload fixed. Warm each
candidate fully before scoring it.

1. Use `22 / 7 / 19` as the safe reference.
2. Test `14 / 7 / 27` as the coarse midpoint.
3. If 14 Mac layers wins on end-to-end time, test `10 / 7 / 31`; otherwise test
   `18 / 7 / 23`.
4. Refine the winning interval in two-layer increments.
5. Stop exploring toward the CPU when Windows approaches 16 GiB available during a
   real agent trace. The emergency guard remains at 12 GiB.

Score both the deterministic trace and an agent-shaped replay. The latter should
include approximately 2K, 8K, and 32K prompts with at least 512 generated tokens. Use
median warm results from at least two repetitions; keep cold load time, prompt rate,
decode rate, total wall time, page reads, Windows available RAM, Mac pressure/swap, and
GPU memory as separate fields. Do not select a placement from a 4K synthetic trace
alone, and do not treat a short pass as sustained-safe.

## Portable launch outline

The launcher requires paths and endpoints explicitly so a new Windows installation
does not inherit machine-specific assumptions:

```powershell
$run = Join-Path $PWD 'runs\qwen-split-14-7'
$launch = .\scripts\start-split-experiment.ps1 `
  -ReplacePid 999999 -AllowStopped `
  -RunDirectory $run `
  -LlamaServerExecutable 'D:\llama.cpp\bin\llama-server.exe' `
  -ModelPath 'D:\models\Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf' `
  -RpcServers 'MAC_THUNDERBOLT_ADDRESS:50053' `
  -Devices 'RPC0,CUDA0' `
  -GpuLayers 21 -TensorSplit '14,7' `
  -ContextSize 150000 `
  -LoadMode mmap -HostMemoryMode pageable `
  -CacheTypeK q4_0 -CacheTypeV q4_0 `
  -Threads 16 -Batch 256 -UBatch 64 `
  -PythonExecutable 'python'

.\scripts\run-split-experiment.ps1 `
  -RunDirectory $run `
  -TelemetryScript sample-mac-split-experiment.ps1 `
  -PythonExecutable 'python'
```

Before Mac telemetry, set `LLAMA_MAC_SSH_KEY`, `LLAMA_MAC_KNOWN_HOSTS`,
`LLAMA_MAC_HOST`, and `LLAMA_MAC_USER` in the launching PowerShell session. Credentials
and private keys must never be written to run artifacts or committed.
