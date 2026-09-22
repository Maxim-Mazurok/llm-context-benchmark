# Distributed llama.cpp RPC

This setup runs `llama-server` on the 32 GB Apple Silicon Mac and exposes one
or more remote devices through llama.cpp RPC. The model file stays on the Mac;
the RPC client transfers assigned tensors to each worker. `--cache` lets a
worker retain those tensors for later starts.

RPC is proof-of-concept software with no authentication or encryption. Bind it
only on a trusted private network, restrict the Windows firewall rule to the
Mac's address, and never forward port 50052 from the router.

## Recommended first model

Start with
`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Q5_K_L` (26.23 GB). It is large enough
that a 65K Q8 KV cache makes the 8 GB RTX useful, but still leaves more combined
headroom than the 30.95 GB Q6_K build. Test Q6_K after the first setup works,
or when a second 8 GB worker is available:

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

The setup scripts pin llama.cpp to a verified revision so all machines speak
the same RPC protocol.

```bash
./scripts/llama-rpc/setup-llama-cpp-macos.sh
```

Override checkout location or revision with `LLAMA_CPP_DIRECTORY` and
`LLAMA_CPP_REVISION`.

## 2. Build the Windows CUDA worker

Install current NVIDIA drivers, CUDA Toolkit, Git, CMake, and Visual Studio 2022
with **Desktop development with C++**. In Developer PowerShell from this cloned
repository:

```powershell
.\scripts\llama-rpc\setup-llama-cpp-windows.ps1
```

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
```

The scan detects any TCP service listening on that port; the subsequent
llama.cpp connection verifies that it is a compatible RPC worker. If no worker
is found, an interactive launch asks for comma-separated addresses manually.
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

Defaults are layer splitting, automatic memory-proportional allocation, 65,536
context tokens, Q8 K/V cache, one server slot, and text-only loading. Override
them with environment variables:

```bash
export LLAMA_MODEL_PATH='/absolute/path/to/model.gguf'
export LLAMA_CONTEXT_SIZE=32768
export LLAMA_TENSOR_SPLIT='3,1'
export LLAMA_SERVER_PORT=8081
```

Do not set `LLAMA_TENSOR_SPLIT` initially. Device order and available memory are
printed during load; only tune the proportions after recording that order.

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
./scripts/llama-rpc/start-local-llama-server-macos.sh
```

The adapter keeps one exact token sequence and asks llama-server to reuse its
prompt cache. Run metadata records adapter, server URL, model alias, and context
limit. Host memory samples cover the Mac only; collect `nvidia-smi` telemetry
separately when GPU memory/power measurements matter.

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
after measuring Mac+RTX, and compare prompt and decode rates separately. A
direct Thunderbolt link may help only when both machines and operating systems
support the transport; ordinary USB networking still uses TCP.

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
those transferred tensors can be retained for later starts, but the worker does
not become the authoritative model store.

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