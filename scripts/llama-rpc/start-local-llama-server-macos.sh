#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_directory/select-llama-model-macos.sh"

llama_cpp_directory="${LLAMA_CPP_DIRECTORY:-$HOME/llama.cpp}"
llama_server_path="$llama_cpp_directory/build-rpc-metal/bin/llama-server"
model_path="${LLAMA_MODEL_PATH:-}"
hugging_face_repository="${LLAMA_HUGGING_FACE_REPOSITORY:-}"
context_size="${LLAMA_CONTEXT_SIZE:-65536}"
server_port="${LLAMA_SERVER_PORT:-8080}"
mtp_enabled=false
mtp_blocks=3

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mtp)
            mtp_enabled=true
            shift
            ;;
        --mtp-blocks)
            if [[ $# -lt 2 ]]; then
                printf '%s requires a value.\n' "$1" >&2
                exit 1
            fi
            mtp_blocks="$2"
            shift 2
            ;;
        --help|-h)
            printf 'Usage: %s [--mtp] [--mtp-blocks NUMBER]\n' "${0##*/}"
            printf '  --mtp                Enable MTP speculative decoding.\n'
            printf '  --mtp-blocks NUMBER  Set MTP draft blocks (default: 3).\n'
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 1
            ;;
    esac
done

if [[ ! "$mtp_blocks" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MTP blocks must be a positive integer: %s\n' "$mtp_blocks" >&2
    exit 1
fi

if [[ ! -x "$llama_server_path" ]]; then
    printf 'llama-server not found: %s\n' "$llama_server_path" >&2
    printf 'Run setup-llama-cpp-macos.sh first.\n' >&2
    exit 1
fi

resolve_llama_model_selection

arguments=(
    --n-gpu-layers all
    --ctx-size "$context_size"
    --cache-type-k q8_0
    --cache-type-v q8_0
    --parallel 1
    --host 127.0.0.1
    --port "$server_port"
    --metrics
    --jinja
    --no-mmproj
    --alias mac-local
)

if [[ "$mtp_enabled" == true ]]; then
    arguments+=(--spec-type draft-mtp --spec-draft-num "$mtp_blocks")
fi

if [[ -n "$model_path" ]]; then
    arguments+=(--model "$model_path")
else
    arguments+=(--hf-repo "$hugging_face_repository")
fi

exec "$llama_server_path" "${arguments[@]}"