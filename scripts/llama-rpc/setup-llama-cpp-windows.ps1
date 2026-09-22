param(
    [string]$LlamaCppDirectory = "$HOME\llama.cpp",
    [string]$LlamaCppRevision = "58367713a6935c0810103378144008df32e3d5db"
)

$ErrorActionPreference = "Stop"
$buildDirectory = Join-Path $LlamaCppDirectory "build-rpc-cuda"

foreach ($commandName in @("git", "cmake", "nvcc")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $commandName"
    }
}

if (-not (Test-Path (Join-Path $LlamaCppDirectory ".git"))) {
    git clone https://github.com/ggml-org/llama.cpp.git $LlamaCppDirectory
}

git -C $LlamaCppDirectory fetch origin $LlamaCppRevision
git -C $LlamaCppDirectory checkout --detach $LlamaCppRevision
cmake -S $LlamaCppDirectory -B $buildDirectory `
    -DGGML_CUDA=ON `
    -DGGML_RPC=ON
cmake --build $buildDirectory --config Release --parallel

$rpcServerPath = Join-Path $buildDirectory "bin\Release\ggml-rpc-server.exe"
Write-Host "RPC worker: $rpcServerPath"