#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/select-llama-model-macos.sh"

llama_cpp_directory="${LLAMA_CPP_DIRECTORY:-$HOME/llama.cpp}"
llama_server_path="$llama_cpp_directory/build-rpc-metal/bin/llama-server"
rpc_servers="${LLAMA_RPC_SERVERS:-}"
model_path="${LLAMA_MODEL_PATH:-}"
hugging_face_repository="${LLAMA_HUGGING_FACE_REPOSITORY:-}"
context_size="${LLAMA_CONTEXT_SIZE:-65536}"
server_port="${LLAMA_SERVER_PORT:-8080}"
tensor_split="${LLAMA_TENSOR_SPLIT:-}"
rpc_port="${LLAMA_RPC_PORT:-50052}"
rpc_subnet="${LLAMA_RPC_SUBNET:-}"
rpc_scan_parallelism="${LLAMA_RPC_SCAN_PARALLELISM:-64}"

discover_rpc_servers() {
    local default_interface
    local discovered_address
    local local_address
    local use_discovered_servers
    local -a discovered_addresses=()

    if ! command -v nc >/dev/null 2>&1; then
        return
    fi

    if [[ -z "$rpc_subnet" ]]; then
        default_interface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"
        if [[ -n "$default_interface" ]]; then
            local_address="$(ipconfig getifaddr "$default_interface" 2>/dev/null || true)"
            if [[ "$local_address" =~ ^([0-9]+\.[0-9]+\.[0-9]+)\.[0-9]+$ ]]; then
                rpc_subnet="${BASH_REMATCH[1]}"
            fi
        fi
    fi
    rpc_subnet="${rpc_subnet:-192.168.0}"

    printf 'Scanning %s.1-254 on RPC port %s...\n' "$rpc_subnet" "$rpc_port" >&2
    while IFS= read -r discovered_address; do
        discovered_addresses+=("$discovered_address")
    done < <(
        seq 1 254 |
            xargs -P "$rpc_scan_parallelism" -I '{}' sh -c \
                'nc -z -G 1 -w 1 "$1.{}" "$2" >/dev/null 2>&1 && printf "%s.{}\n" "$1"' \
                sh "$rpc_subnet" "$rpc_port" |
            sort -t . -k 4,4n
    )

    if (( ${#discovered_addresses[@]} == 0 )); then
        printf 'No RPC workers found.\n' >&2
        return
    fi

    printf 'Discovered RPC workers:\n' >&2
    for discovered_address in "${discovered_addresses[@]}"; do
        printf '  %s:%s\n' "$discovered_address" "$rpc_port" >&2
    done

    if [[ -t 0 ]]; then
        read -r -p 'Use all discovered workers? [Y/n] ' use_discovered_servers
        if [[ "$use_discovered_servers" =~ ^[Nn]$ ]]; then
            return
        fi
    fi

    for discovered_address in "${discovered_addresses[@]}"; do
        rpc_servers+="${rpc_servers:+,}${discovered_address}:${rpc_port}"
    done
}

if [[ ! -x "$llama_server_path" ]]; then
    printf 'llama-server not found: %s\n' "$llama_server_path" >&2
    printf 'Run setup-llama-cpp-macos.sh first.\n' >&2
    exit 1
fi
if [[ -z "$rpc_servers" ]]; then
    discover_rpc_servers
fi
if [[ -z "$rpc_servers" ]]; then
    if [[ ! -t 0 ]]; then
        printf 'Set LLAMA_RPC_SERVERS or LLAMA_RPC_SUBNET to locate workers.\n' >&2
        exit 1
    fi

    while [[ -z "$rpc_servers" ]]; do
        read -r -p 'RPC worker addresses (comma-separated host:port): ' rpc_servers
    done
fi

resolve_llama_model_selection

arguments=(
    --rpc "$rpc_servers"
    --n-gpu-layers all
    --split-mode layer
    --ctx-size "$context_size"
    --cache-type-k q8_0
    --cache-type-v q8_0
    --parallel 1
    --host 127.0.0.1
    --port "$server_port"
    --metrics
    --jinja
    --no-mmproj
    --alias distributed-local
)

if [[ -n "$model_path" ]]; then
    arguments+=(--model "$model_path")
else
    arguments+=(--hf-repo "$hugging_face_repository")
fi
if [[ -n "$tensor_split" ]]; then
    arguments+=(--tensor-split "$tensor_split")
fi

exec "$llama_server_path" "${arguments[@]}"