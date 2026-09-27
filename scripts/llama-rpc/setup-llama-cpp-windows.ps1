param(
    [string]$LlamaCppDirectory = "$HOME\llama.cpp",
    [string]$LlamaCppRelease = "b11094"
)

$ErrorActionPreference = "Stop"
$buildDirectory = Join-Path $LlamaCppDirectory "build-rpc-cuda"
$binaryDirectory = Join-Path $buildDirectory "bin\Release"
$releaseBaseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaCppRelease"
$releaseArchives = @{
    "b11094" = @(
        @{
            Name = "llama-b11094-bin-win-cuda-12.4-x64.zip"
            Sha256 = "0bc6f1ee5f781425f32ff4cf2981fd031f096a16de43e55a1c654534728ba39f"
        },
        @{
            Name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
            Sha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
        }
    )
    "b11146" = @(
        @{
            Name = "llama-b11146-bin-win-cuda-12.4-x64.zip"
            Sha256 = "3c806a6ceccc3dae1c743ceb1a1fb2cce5b76f40bfbd4c6b7b8afb6ef45a5807"
        },
        @{
            Name = "cudart-llama-bin-win-cuda-12.4-x64.zip"
            Sha256 = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
        }
    )
}
if (-not $releaseArchives.ContainsKey($LlamaCppRelease)) {
    throw "Unsupported llama.cpp release $LlamaCppRelease. Verified releases: $($releaseArchives.Keys -join ', ')"
}
$archives = $releaseArchives[$LlamaCppRelease]
$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "llama-cpp-$LlamaCppRelease-$([guid]::NewGuid())"
$stagedBuildDirectory = Join-Path $temporaryDirectory "build-rpc-cuda"
$stagedBinaryDirectory = Join-Path $stagedBuildDirectory "bin\Release"

New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null
try {
    New-Item -ItemType Directory -Path $stagedBinaryDirectory -Force | Out-Null

    foreach ($archive in $archives) {
        $archivePath = Join-Path $temporaryDirectory $archive.Name
        Invoke-WebRequest -Uri "$releaseBaseUrl/$($archive.Name)" -OutFile $archivePath

        $actualSha256 = (Get-FileHash -Path $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $archive.Sha256) {
            throw "SHA256 mismatch for $($archive.Name): expected $($archive.Sha256), got $actualSha256"
        }

        Expand-Archive -Path $archivePath -DestinationPath $stagedBinaryDirectory -Force
    }

    $stagedRpcServerPath = Join-Path $stagedBinaryDirectory "ggml-rpc-server.exe"
    if (-not (Test-Path $stagedRpcServerPath)) {
        throw "Official llama.cpp archive did not contain ggml-rpc-server.exe"
    }
    $stagedLlamaServerPath = Join-Path $stagedBinaryDirectory "llama-server.exe"
    if (-not (Test-Path $stagedLlamaServerPath)) {
        throw "Official llama.cpp archive did not contain llama-server.exe"
    }

    Set-Content -Path (Join-Path $stagedBuildDirectory "release.txt") -Value $LlamaCppRelease
    New-Item -ItemType Directory -Path $LlamaCppDirectory -Force | Out-Null
    Remove-Item $buildDirectory -Recurse -Force -ErrorAction SilentlyContinue
    Move-Item -Path $stagedBuildDirectory -Destination $buildDirectory
}
finally {
    Remove-Item $temporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
}

$rpcServerPath = Join-Path $binaryDirectory "ggml-rpc-server.exe"
$llamaServerPath = Join-Path $binaryDirectory "llama-server.exe"
Write-Host "Installed llama.cpp $LlamaCppRelease"
Write-Host "llama-server: $llamaServerPath"
Write-Host "RPC worker:   $rpcServerPath"