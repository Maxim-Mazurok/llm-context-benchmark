#!/usr/bin/env bash
# Run on the Mac only. Builds a separate checkout; does not start inference.
set -euo pipefail
revision=7fe450e19305b828c199d602c23a8337aaa1f03b
checkout="${QWEN_MAC_CHECKOUT:-$HOME/llama.cpp-qwen-b11146}"
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires Apple Silicon macOS'; exit 1; }
for tool in git cmake xcrun; do
    command -v "$tool" >/dev/null || { echo "Missing $tool: install it before continuing"; exit 1; }
done
xcrun --find clang >/dev/null
sw_vers
sysctl hw.memsize hw.physicalcpu hw.logicalcpu
vm_stat
sysctl vm.swapusage
df -h "$HOME"
networksetup -listallhardwareports
if [[ -e "$checkout" ]]; then
    [[ -d "$checkout/.git" ]] || { echo 'Existing target is not a Git checkout'; exit 1; }
    [[ "$(git -C "$checkout" rev-parse HEAD)" == "$revision" ]] || { echo 'Existing checkout has a different revision; refusing to change it'; exit 1; }
    [[ -z "$(git -C "$checkout" status --porcelain)" ]] || { echo 'Existing checkout has changes; refusing to change it'; exit 1; }
else
    git clone --no-checkout https://github.com/ggml-org/llama.cpp.git "$checkout"
    git -C "$checkout" checkout --detach "$revision"
fi
git -C "$checkout" apply --check "$script_dir/qwen-rpc-no-hash-cache.patch"
git -C "$checkout" apply "$script_dir/qwen-rpc-no-hash-cache.patch"
cmake -S "$checkout" -B "$checkout/build-rpc-metal" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_RPC=ON -DGGML_CUDA=OFF
cmake --build "$checkout/build-rpc-metal" --config Release --parallel 4 \
    --target ggml-rpc-server llama-server
git -C "$checkout" rev-parse HEAD
"$checkout/build-rpc-metal/bin/llama-server" --version
"$checkout/build-rpc-metal/bin/llama-server" --list-devices
echo 'Build complete. No worker/server has been started and no model downloaded.'
echo 'The checkout now contains the intentional hash-cache switch patch; rerunning refuses a dirty checkout.'
