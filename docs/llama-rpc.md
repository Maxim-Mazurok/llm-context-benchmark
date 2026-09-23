# Distributed llama.cpp RPC

This setup runs `llama-server` on the 32 GB Apple Silicon Mac and exposes one
or more remote devices through llama.cpp RPC. The model file stays on the Mac;
the RPC client transfers assigned tensors to each worker. Worker `--cache`
stores transferred weight tensors on local disk so later starts can skip their
network transfer; it does not retain them in VRAM.

RPC is proof-of-concept software with no authentication or encryption. Bind it
only on a trusted private network, restrict the Windows firewall rule to the
Mac's address, and never forward port 50052 from the router.

## Recommended first model

Start with
`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q4_K_M` (22.3 GB). The 26.2 GB Q5_K_L
build leaves too little practical headroom on a 32 GB Mac plus an 8 GB Windows
GPU once macOS, Windows, Metal/CUDA buffers, and KV cache are included. Test a
larger quantization only when another worker is available:

```bash
export LLAMA_HUGGING_FACE_REPOSITORY='bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q6_K'
```

The downloaded OMLX and LM Studio Qwen3.6/Gemma checkpoints use MLX
`safetensors`; llama.cpp RPC requires GGUF. The one GGUF already installed is
Qwen3.8 27B Q4_K_M at:

```text
~/.lmstudio/models/OBLITERATUS/Qwen3.8-27B-OBLITERATED/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf
```

It is useful for a smoke test, but its 16.8 GB size does not stress the combined
memory pool. Qwen3.6 is the stronger first agentic-coding comparison. Gemma 4
31B dense remains the useful follow-up for measuring compute acceleration; use
a current GGUF only after confirming the installed llama.cpp revision supports
that architecture.

## 1. Build the Mac host

The setup scripts pin llama.cpp to release `b11094` so all machines speak the
same RPC protocol.

```bash
./scripts/llama-rpc/setup-llama-cpp-macos.sh
```

Override checkout location or revision with `LLAMA_CPP_DIRECTORY` and
`LLAMA_CPP_REVISION`.

## 2. Install the Windows CUDA worker

Install a current NVIDIA driver. In PowerShell from this cloned repository,
download and verify the official pinned CUDA 12.4 llama.cpp binaries:

```powershell
.\scripts\llama-rpc\setup-llama-cpp-windows.ps1
```

The script includes the matching CUDA runtime, so CUDA Toolkit, Git, CMake, and
Visual Studio Build Tools are not required on Windows. Override the installation
location with `-LlamaCppDirectory`; the release is intentionally pinned to keep
it compatible with the Mac host.

Start the worker from Administrator PowerShell. It creates a firewall rule that
allows any source address on networks classified as Private:

```powershell
.\scripts\llama-rpc\start-llama-rpc-worker-windows.ps1
```

To restrict access to the Mac later, pass
`-AllowedClientAddress 192.168.0.10`. Public-profile traffic remains blocked.

Keep this process running. Its startup output must list `CUDA0` and roughly
8 GB total memory. If it does not, fix CUDA detection before starting the Mac
host.

## 3. Start distributed inference on the Mac

Launch without configuring the Dell's address:

```bash
./scripts/llama-rpc/start-distributed-llama-server-macos.sh
```

When `LLAMA_RPC_SERVERS` is unset, the launcher scans port 50052 across the
Mac's current IPv4 `/24` subnet, lists reachable workers, and offers to use all
of them. If the subnet cannot be detected, it scans `192.168.0.1-254`. Override
the scan range or port when needed:

```bash
export LLAMA_RPC_SUBNET='192.168.1'
export LLAMA_RPC_PORT=50052
export LLAMA_RPC_SCAN_PARALLELISM=64
```

Each connection attempt has a one-second timeout. The scan detects any TCP
service listening on that port; the subsequent llama.cpp connection verifies
that it is a compatible RPC worker. If no worker is found, an interactive
launch asks for comma-separated addresses manually.
For noninteractive use, or to skip discovery, set `LLAMA_RPC_SERVERS` first:

```bash
export LLAMA_RPC_SERVERS='192.168.0.20:50052'
```

When neither model variable is set, an interactive launch scans
`~/.omlx/models` and `~/.lmstudio/models` recursively for downloaded GGUF
files. Select one from the prompt, or select the recommended Hugging Face model
to download it into llama.cpp's cache. If no local GGUF exists, the recommended
model is selected automatically. Noninteractive launches also use the
recommended model automatically.

The Mac sends assigned tensors over the LAN. Later starts can reuse both local
and RPC caches. Wait for `http://127.0.0.1:8080/health` to return
`{"status":"ok"}`.

Distributed mode passes `--load-mode none`. Do not remove this option. The
default mmap loader advises macOS that the entire GGUF will be needed, which can
make tensors assigned to RPC workers resident in Mac memory even though the
workers own their persistent execution copies. Non-mmap loading reads each
tensor from the Mac's model file and either loads its local allocation or sends
it to the assigned RPC worker without retaining a mapped copy of remote
tensors. The Mac must still read the complete GGUF once during startup, so
macOS may keep some data in reclaimable filesystem cache.

After loading, the Mac retains only its assigned model tensors, local-layer KV
and recurrent state, Metal compute buffers, and normal server overhead. Each
RPC worker retains its assigned tensors and cache state. `--load-mode none`
changes how weights enter memory; it does not change tensor placement or cause
weights to load on demand during inference. Mac-only mode intentionally keeps
llama.cpp's default mmap behavior because every model tensor is local there.

With exactly two workers, the launcher defaults to the RPC-first tensor split
`1,1,1.1`. With the recommended Q4_K_M model and two 8 GB NVIDIA cards, this
places about 7 GB on each worker while retaining CUDA headroom. llama.cpp's
automatic fit does not account for macOS unified host-memory pressure, so a
larger Metal fit target alone does not move these layers off the Mac.

Other worker counts use automatic layer fitting, 8,192 MiB of reserved Mac
memory, and 512 MiB reserved on each worker. All modes default to 32,768 context
tokens, 512-token batches, 128-token physical microbatches, Q8 K/V cache, one
server slot, and text-only loading. Fit targets follow llama.cpp's model-device
order: all RPC workers first, then the local Metal device. Additional workers
add another 512 MiB target before the final 8,192 MiB Mac target. Override the
defaults with environment variables:

```bash
export LLAMA_MODEL_PATH='/absolute/path/to/model.gguf'
export LLAMA_CONTEXT_SIZE=65536
export LLAMA_FIT_TARGET='512,8192'
export LLAMA_BATCH_SIZE=512
export LLAMA_MICROBATCH_SIZE=128
export LLAMA_SERVER_PORT=8081
```

llama.cpp can also cap reasoning before allowing the model to continue with its
final answer. Configure the server defaults through its supported environment
variables:

```bash
export LLAMA_ARG_REASONING=on
export LLAMA_ARG_THINK_BUDGET=4096
export LLAMA_ARG_THINK_BUDGET_MESSAGE='I have enough information. I will now provide the final answer.'
```

The reasoning budget accepts `-1` for unlimited reasoning, `0` to end reasoning
immediately, or a positive token count. Use both budget settings for user-facing
responses: the token count enforces the limit, while the short message helps the
model transition cleanly before llama.cpp forces the end-of-thinking token. The
message is optional and defaults to none; omitting it is preferable for controlled
benchmarks where injected guidance would affect comparability. The overall
`max_tokens` or `n_predict` limit still needs enough room for the final answer.
Qwen3.6 35B A3B supports reasoning on or off through `enable_thinking`; its chat
template does not expose graded reasoning-effort or model-native budget controls.
The llama.cpp budget is an external hard limit and remains usable with this model.

`--list-devices` prints Metal before RPC, but implicit model allocation places
RPC workers first to reduce network transfers. Fit targets and tensor-split
values follow that model order. With one worker, `1,3` means 25% Windows and 75%
Mac; `3,1` means 75% Windows and will exceed an 8 GB GPU for the recommended
model. Setting `LLAMA_TENSOR_SPLIT` replaces the two-worker default or disables
llama.cpp's automatic fit calculation for other topologies. Set
`LLAMA_TENSOR_SPLIT=auto` to force automatic fitting. If Windows Task Manager
reports shared GPU memory use, the CUDA allocation has exceeded dedicated VRAM;
stop the server and increase the final Mac share or use a smaller model.

## 4. Benchmark

Run a short validation first:

```bash
uv run llm-context-bench \
  --adapter llama-server \
  --server-url http://127.0.0.1:8080 \
  --max-context 5000 \
  --chunk-tokens 1000 \
  --short-decode-tokens 32 \
  --long-decode-interval 3000 \
  --long-decode-tokens 64
```

Then compare identical model, context, cache types, and benchmark arguments.
Start distributed mode for one run and Mac-only mode for the other:

```bash
./scripts/llama-rpc/start-distributed-llama-server-macos.sh --local
```

Add `--mtp` to enable MTP speculative decoding. It uses three draft blocks by
default; override that with `--mtp-blocks NUMBER`.

The adapter keeps one exact token sequence and asks llama-server to reuse its
prompt cache. Run metadata records adapter, server URL, model alias, and context
limit. Host memory samples cover the Mac only; collect `nvidia-smi` telemetry
separately when GPU memory/power measurements matter.

## Direct USB4 networking

<!-- cspell:words Gbps iperf Mbps -->

A direct USB4 cable can carry llama.cpp RPC between macOS and Windows through
the USB4 peer-to-peer network adapters. A connected 20 Gbps USB4 fabric does
not by itself make RPC use the cable: both peer adapters need IPv4 addresses,
the Windows network profile and firewall must admit RPC, and the Mac launcher
must use the Windows USB4 address.

The tested pair assigned link-local addresses automatically:

- Mac Thunderbolt Bridge: `169.254.117.5/16`
- Windows USB4 P2P Network Adapter: `169.254.185.10/16`

Find the current Windows values in Administrator PowerShell:

```powershell
Get-NetAdapter |
  Format-Table Name, InterfaceDescription, Status, LinkSpeed, MacAddress, `
    ifIndex -Auto
Get-NetIPAddress -AddressFamily IPv4 |
  Format-Table InterfaceAlias, IPAddress, PrefixLength, AddressState -Auto
Get-NetConnectionProfile |
  Format-Table InterfaceAlias, NetworkCategory, IPv4Connectivity -Auto
```

The tested Windows adapter appeared as `Ethernet 2`, with interface description
`USB4(TM) P2P Network Adapter`. Windows initially classified it as Public, so
the RPC firewall rule created by the worker launcher did not apply. Change only
the USB4 adapter to Private:

```powershell
Set-NetConnectionProfile -InterfaceAlias 'Ethernet 2' -NetworkCategory Private
```

Keep Wi-Fi's gateway and DNS configuration unchanged. The link-local USB4
adapter needs no default gateway. Confirm that the existing
`llama.cpp RPC on private networks` rule allows inbound TCP port 50052 on the
Private profile and that the worker listens on `0.0.0.0:50052`.

On the Mac, verify the direct endpoint and route:

```bash
nc -vz 169.254.185.10 50052
route -n get 169.254.185.10
```

The route must show `interface: bridge0`. Start the server with the direct
address to bypass Wi-Fi discovery:

```bash
LLAMA_RPC_SERVERS='169.254.185.10:50052' \
  ./scripts/llama-rpc/start-distributed-llama-server-macos.sh
```

### Measured USB4 versus Wi-Fi performance

Measurements from 2026-09-22 used `iperf3` 3.21 and the same Windows worker.
The Windows Wi-Fi adapter reported a 600 Mbps link rate; its actual TCP
throughput was substantially lower.

| Path | Direction | Streams | Receiver throughput |
| --- | --- | ---: | ---: |
| USB4 | Mac to Windows | 1 | 17.856 Gbps |
| USB4 | Windows to Mac | 1 | 11.192 Gbps |
| USB4 | Mac to Windows | 4 | 18.861 Gbps |
| USB4 | Windows to Mac | 4 | 9.907 Gbps |
| Wi-Fi | Mac to Windows | 1 | 121.765 Mbps |
| Wi-Fi | Windows to Mac | 1 | 114.086 Mbps |
| Wi-Fi | Mac to Windows | 4 | 127.650 Mbps |
| Wi-Fi | Windows to Mac | 4 | 88.628 Mbps |

One hundred TCP connections to RPC port 50052 produced these latency results:

| Path | Minimum | Median | Mean | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| USB4 | 0.463 ms | 0.561 ms | 0.722 ms | 1.054 ms | 9.856 ms |
| Wi-Fi | 6.157 ms | 19.587 ms | 52.744 ms | 131.835 ms | 150.894 ms |

For this pair, USB4 delivered about 87 to 165 times more TCP throughput and
about 35 times lower median TCP connection latency. ICMP behavior was
inconsistent across Windows firewall states, so TCP connection latency is the
relevant comparison for RPC.

The clearest inference benefit is uncached tensor transfer when loading a
model. At the measured one-stream rates, ideal transfer time for 7 GiB is about
3.4 seconds over USB4 versus 8.2 minutes over Wi-Fi. Protocol, storage, tensor
allocation, and initialization overhead increase real load time. Worker
`--cache` reduces this difference on later loads.

Steady-state token throughput also benefits from USB4's bandwidth and latency
when execution crosses the local/remote device boundary. The improvement is
model-, split-, prompt-, and GPU-dependent; it does not scale directly with the
87-to-165-times network throughput ratio because GPU compute and local memory
operations remain part of every token. Compare prompt and decode rates with
identical model, tensor split, context, cache, and benchmark arguments to
measure the end-to-end gain.

## Additional workers

For another Windows NVIDIA machine, run the same worker script on another IP
and list both endpoints:

```bash
export LLAMA_RPC_SERVERS='192.168.0.20:50052,192.168.0.21:50052'
```

For an older Intel Mac CPU worker, build with the macOS setup script, then:

```bash
export LLAMA_CPP_METAL=OFF
./scripts/llama-rpc/setup-llama-cpp-macos.sh
export LLAMA_RPC_DEVICE=CPU
./scripts/llama-rpc/start-llama-rpc-worker-macos.sh
```

A 1 Gbps link can make a slow CPU worker reduce total throughput. Add it only
after measuring Mac+RTX, and compare prompt and decode rates separately.

## Common questions

### Will `LLAMA_HUGGING_FACE_REPOSITORY` download the model?

Yes. An explicit `LLAMA_MODEL_PATH` takes precedence. Otherwise, an explicit
`LLAMA_HUGGING_FACE_REPOSITORY` is passed to `llama-server` as `--hf-repo`.
llama.cpp downloads missing GGUF files into its local cache and reuses them on
later starts. The value must identify a llama.cpp-compatible GGUF repository
and may include a quantization selector, for example:

```bash
export LLAMA_HUGGING_FACE_REPOSITORY='bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q6_K'
```

With neither variable set, interactive Mac launchers offer discovered OMLX and
LM Studio GGUF files before the recommended download. MLX `safetensors`
repositories are not interchangeable with GGUF repositories.

### Does each RPC worker download or store the model?

No full model download is required on a worker. The Mac owns the model file and
sends each worker its assigned tensors over RPC. With worker caching enabled,
weight tensors are cached on the worker's local disk to avoid later network
transfers, but the worker does not become the authoritative model store.

### Must Mac and Windows use the same llama.cpp revision?

Yes. Both setup scripts currently pin the same static llama.cpp commit. Run the
corresponding setup script on every host after changing that pin; mixing builds
can cause RPC protocol or tensor-transfer failures.

### Can the Windows worker accept any client on the home network?

Yes. The Windows worker script defaults `AllowedClientAddress` to `Any`, but its
firewall rule applies only to the Windows Private network profile. Keep the
network classified as Private and do not expose the RPC port through the
router. Pass the Mac's private IP with `-AllowedClientAddress` when tighter
access is preferred.

## Troubleshooting

Confirm basic reachability from the Mac:

```bash
nc -vz 192.168.0.20 50052
```

Enable RPC diagnostics on either process with `GGML_RPC_DEBUG=1`. Force TCP
with `GGML_RPC_NO_RDMA=1`. Keep identical pinned revisions across hosts; RPC
protocol mismatches often appear as connection or tensor-transfer failures.

An RPC worker started with `--cache` stores reusable weight tensors under its
local llama.cpp cache directory. GPU allocations are released when the Mac
server disconnects. Windows shared GPU memory use indicates CUDA fallback into
system RAM, not useful additional VRAM; stop that run rather than benchmarking
it.

Warnings about unused `nextn` tensors in Qwen3.6 describe optional speculative
decoding weights and do not cause the Metal failure. `Insufficient Memory`
during warm-up or a request means the Mac allocation is too aggressive. Keep
the default 32K context and reduced batch sizes for the first successful run;
then raise one setting at a time.

If distributed startup makes Mac wired memory grow by approximately the full
GGUF size, confirm the running command contains `--load-mode none`. Expected
growth is the local tensor share plus local cache and compute buffers. Total
macOS memory used can grow further from reclaimable filesystem cache and is not
equivalent to llama-server resident or wired memory.
