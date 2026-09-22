#!/usr/bin/env bash
set -euo pipefail

llama_cpp_directory="${LLAMA_CPP_DIRECTORY:-$HOME/llama.cpp}"
rpc_server_path="$llama_cpp_directory/build-rpc-metal/bin/ggml-rpc-server"
host_address="${LLAMA_RPC_HOST_ADDRESS:-0.0.0.0}"
port="${LLAMA_RPC_PORT:-50052}"
device="${LLAMA_RPC_DEVICE:-CPU}"

if [[ ! -x "$rpc_server_path" ]]; then
    printf 'RPC worker not found: %s\n' "$rpc_server_path" >&2
    printf 'Run setup-llama-cpp-macos.sh first.\n' >&2
    exit 1
fi

printf 'WARNING: RPC has no authentication or encryption. Use only on a trusted private network.\n' >&2
exec "$rpc_server_path" \
    --host "$host_address" \
    --port "$port" \
    --device "$device" \
    --cache