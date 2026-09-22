#!/usr/bin/env bash
set -euo pipefail

llama_cpp_directory="${LLAMA_CPP_DIRECTORY:-$HOME/llama.cpp}"
llama_cpp_revision="${LLAMA_CPP_REVISION:-58367713a6935c0810103378144008df32e3d5db}"
metal_backend="${LLAMA_CPP_METAL:-ON}"
build_directory="$llama_cpp_directory/build-rpc-metal"

for command_name in git cmake; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$command_name" >&2
        exit 1
    fi
done

if [[ ! -d "$llama_cpp_directory/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$llama_cpp_directory"
fi

git -C "$llama_cpp_directory" fetch origin "$llama_cpp_revision"
git -C "$llama_cpp_directory" checkout --detach "$llama_cpp_revision"
cmake -S "$llama_cpp_directory" -B "$build_directory" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_METAL="$metal_backend" \
    -DGGML_RPC=ON
cmake --build "$build_directory" --config Release --parallel

printf 'llama-server: %s\n' "$build_directory/bin/llama-server"
printf 'RPC worker:   %s\n' "$build_directory/bin/ggml-rpc-server"