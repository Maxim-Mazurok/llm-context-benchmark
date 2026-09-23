<#
.SYNOPSIS
Start llama-server on Windows as the distributed llama.cpp RPC parent process.

.DESCRIPTION
Mirrors start-distributed-llama-server-macos.sh. The Windows host owns the GGUF
model file and distributes tensors to discovered RPC workers. Partial GPU
offloading is available through -GpuLayers, -CpuMoeLayers, and -CpuFfnLayers so
part of the model can stay in ordinary system RAM on the CPU device.
#>
[CmdletBinding()]
param(
    [switch]$Local,
    [switch]$Mtp,
    [ValidateRange(1, 1024)]
    [int]$MtpBlocks = 3,
    [ValidatePattern('^$|^all$|^auto$|^[0-9]+$')]
    [string]$GpuLayers = $env:LLAMA_GPU_LAYERS,
    [ValidatePattern('^$|^[0-9]+$')]
    [string]$CpuMoeLayers = $env:LLAMA_CPU_MOE_LAYERS,
    [ValidatePattern('^$|^[0-9]+$')]
    [string]$CpuFfnLayers = $env:LLAMA_CPU_FFN_LAYERS,
    [string]$LlamaCppDirectory = $env:LLAMA_CPP_DIRECTORY,
    [string]$RpcServers = $env:LLAMA_RPC_SERVERS,
    [string]$ModelPath = $env:LLAMA_MODEL_PATH,
    [string]$HuggingFaceRepository = $env:LLAMA_HUGGING_FACE_REPOSITORY,
    [string]$TensorSplit = $env:LLAMA_TENSOR_SPLIT,
    [string]$FitTarget = $env:LLAMA_FIT_TARGET,
    [string]$RpcSubnet = $env:LLAMA_RPC_SUBNET,
    [int]$ContextSize = 0,
    [int]$BatchSize = 0,
    [int]$MicrobatchSize = 0,
    [int]$PromptCacheMebibytes = -1,
    [int]$LocalFitTargetMebibytes = 0,
    [int]$WorkerFitTargetMebibytes = 0,
    [int]$ServerPort = 0,
    [int]$RpcPort = 0,
    [int]$RpcScanParallelism = 0
)

$ErrorActionPreference = 'Stop'

function Get-IntegerSetting {
    param(
        [int]$ProvidedValue,
        [string]$EnvironmentVariableName,
        [int]$DefaultValue,
        [int]$UnsetValue = 0
    )

    if ($ProvidedValue -ne $UnsetValue) {
        return $ProvidedValue
    }
    $environmentValue = [Environment]::GetEnvironmentVariable($EnvironmentVariableName)
    $parsedValue = 0
    if ($environmentValue -and [int]::TryParse($environmentValue, [ref]$parsedValue)) {
        return $parsedValue
    }
    return $DefaultValue
}

$contextSize = Get-IntegerSetting $ContextSize 'LLAMA_CONTEXT_SIZE' 32768
$batchSize = Get-IntegerSetting $BatchSize 'LLAMA_BATCH_SIZE' 512
$microbatchSize = Get-IntegerSetting $MicrobatchSize 'LLAMA_MICROBATCH_SIZE' 128
$promptCacheMebibytes = Get-IntegerSetting $PromptCacheMebibytes 'LLAMA_PROMPT_CACHE_MEBIBYTES' 0 -UnsetValue -1
$localFitTargetMebibytes = Get-IntegerSetting $LocalFitTargetMebibytes 'LLAMA_LOCAL_FIT_TARGET_MEBIBYTES' 2048
$workerFitTargetMebibytes = Get-IntegerSetting $WorkerFitTargetMebibytes 'LLAMA_WORKER_FIT_TARGET_MEBIBYTES' 512
$serverPort = Get-IntegerSetting $ServerPort 'LLAMA_SERVER_PORT' 8080
$rpcPort = Get-IntegerSetting $RpcPort 'LLAMA_RPC_PORT' 50052
$rpcScanParallelism = Get-IntegerSetting $RpcScanParallelism 'LLAMA_RPC_SCAN_PARALLELISM' 64

if (-not $LlamaCppDirectory) {
    $LlamaCppDirectory = Join-Path $HOME 'llama.cpp'
}

$scriptDirectory = Split-Path -Parent $PSCommandPath
. (Join-Path $scriptDirectory 'select-llama-model-windows.ps1')

$llamaServerPath = Join-Path $LlamaCppDirectory 'build-rpc-cuda\bin\Release\llama-server.exe'
if (-not (Test-Path -LiteralPath $llamaServerPath)) {
    throw "llama-server not found at $llamaServerPath. Run setup-llama-cpp-windows.ps1 first."
}

function Get-LocalSubnet {
    foreach ($networkInterface in [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()) {
        if ($networkInterface.OperationalStatus -ne 'Up') {
            continue
        }
        if ($networkInterface.NetworkInterfaceType -eq 'Loopback') {
            continue
        }
        $interfaceProperties = $networkInterface.GetIPProperties()
        $hasGateway = $interfaceProperties.GatewayAddresses | Where-Object {
            $_.Address.AddressFamily -eq 'InterNetwork' -and $_.Address.ToString() -ne '0.0.0.0'
        }
        if (-not $hasGateway) {
            continue
        }
        foreach ($unicastAddress in $interfaceProperties.UnicastAddresses) {
            if ($unicastAddress.Address.AddressFamily -ne 'InterNetwork') {
                continue
            }
            $addressText = $unicastAddress.Address.ToString()
            if ($addressText -match '^(\d+\.\d+\.\d+)\.\d+$') {
                return $Matches[1]
            }
        }
    }
    return ''
}

function Find-RpcWorker {
    param(
        [string]$Subnet,
        [int]$Port,
        [int]$Parallelism
    )

    $discoveredAddresses = [System.Collections.Generic.List[string]]::new()
    $hostNumbers = 1..254
    for ($batchStart = 0; $batchStart -lt $hostNumbers.Count; $batchStart += $Parallelism) {
        $batchEnd = [Math]::Min($batchStart + $Parallelism, $hostNumbers.Count) - 1
        $pendingConnections = @()
        foreach ($hostNumber in $hostNumbers[$batchStart..$batchEnd]) {
            $address = "$Subnet.$hostNumber"
            $tcpClient = [System.Net.Sockets.TcpClient]::new()
            $pendingConnections += [pscustomobject]@{
                Address     = $address
                TcpClient   = $tcpClient
                ConnectTask = $tcpClient.ConnectAsync($address, $Port)
            }
        }
        try {
            [System.Threading.Tasks.Task]::WaitAll(
                [System.Threading.Tasks.Task[]]@($pendingConnections.ConnectTask),
                1000
            ) | Out-Null
        }
        catch {
        }
        foreach ($pendingConnection in $pendingConnections) {
            if ($pendingConnection.TcpClient.Connected) {
                $discoveredAddresses.Add($pendingConnection.Address)
            }
            $pendingConnection.TcpClient.Dispose()
        }
    }

    return @($discoveredAddresses | Sort-Object { [int](($_ -split '\.')[3]) })
}

function Get-DiscoveredRpcServer {
    if (-not $RpcSubnet) {
        $RpcSubnet = Get-LocalSubnet
    }
    if (-not $RpcSubnet) {
        $RpcSubnet = '192.168.0'
    }

    Write-Host "Scanning $RpcSubnet.1-254 on RPC port $rpcPort..."
    $discoveredAddresses = Find-RpcWorker -Subnet $RpcSubnet -Port $rpcPort -Parallelism $rpcScanParallelism

    if ($discoveredAddresses.Count -eq 0) {
        Write-Host 'No RPC workers found.'
        return [pscustomobject]@{ RpcServers = ''; UseLocalServer = $false }
    }

    Write-Host 'Discovered RPC workers:'
    foreach ($discoveredAddress in $discoveredAddresses) {
        Write-Host "  ${discoveredAddress}:$rpcPort"
    }

    if (-not [Console]::IsInputRedirected) {
        $response = Read-Host 'Use all discovered workers? [Y/n/x local-only]'
        if ($response -match '^[Xx]$') {
            return [pscustomobject]@{ RpcServers = ''; UseLocalServer = $true }
        }
        if ($response -match '^[Nn]$') {
            return [pscustomobject]@{ RpcServers = ''; UseLocalServer = $false }
        }
    }

    $endpoints = $discoveredAddresses | ForEach-Object { "${_}:$rpcPort" }
    return [pscustomobject]@{ RpcServers = ($endpoints -join ','); UseLocalServer = $false }
}

$useLocalServer = [bool]$Local

if (-not $RpcServers -and -not $useLocalServer) {
    $discoveryResult = Get-DiscoveredRpcServer
    $RpcServers = $discoveryResult.RpcServers
    $useLocalServer = $discoveryResult.UseLocalServer
}

if (-not $RpcServers -and -not $useLocalServer) {
    if ([Console]::IsInputRedirected) {
        throw 'Set LLAMA_RPC_SERVERS or LLAMA_RPC_SUBNET to locate workers, or pass -Local.'
    }
    while (-not $RpcServers) {
        $RpcServers = Read-Host 'RPC worker addresses (comma-separated host:port)'
    }
}

if (-not $useLocalServer) {
    $rpcServerCount = ($RpcServers -split ',').Count
    if ($TensorSplit -eq 'auto') {
        $TensorSplit = ''
    }
    if (-not $FitTarget) {
        $fitTargetValues = @(1..$rpcServerCount | ForEach-Object { $workerFitTargetMebibytes })
        $fitTargetValues += $localFitTargetMebibytes
        $FitTarget = $fitTargetValues -join ','
    }
}

$modelSelection = Resolve-LlamaModelSelection `
    -ModelPath $ModelPath `
    -HuggingFaceRepository $HuggingFaceRepository `
    -LlamaCppDirectory $LlamaCppDirectory
$ModelPath = $modelSelection.ModelPath
$HuggingFaceRepository = $modelSelection.HuggingFaceRepository

$arguments = @(
    '--ctx-size', $contextSize
    '--batch-size', $batchSize
    '--ubatch-size', $microbatchSize
    '--cache-type-k', 'q8_0'
    '--cache-type-v', 'q8_0'
    '--cache-ram', $promptCacheMebibytes
    '--ctx-checkpoints', '0'
    '--parallel', '1'
    '--host', '0.0.0.0'
    '--port', $serverPort
    '--metrics'
    '--jinja'
    '--no-mmproj'
)

if ($Mtp) {
    $arguments += @('--spec-type', 'draft-mtp', '--spec-draft-n-max', $MtpBlocks)
}

if ($useLocalServer) {
    $localGpuLayers = if ($GpuLayers) { $GpuLayers } else { 'all' }
    $arguments += @('--n-gpu-layers', $localGpuLayers, '--alias', 'windows-local')
}
else {
    $arguments += @('--rpc', $RpcServers, '--load-mode', 'none', '--split-mode', 'layer')
    if ($TensorSplit) {
        Write-Host "Using RPC-first tensor split $TensorSplit; monitor dedicated and shared GPU memory."
        $arguments += @('--fit', 'off', '--tensor-split', $TensorSplit)
    }
    else {
        $arguments += @('--fit', 'on', '--fit-target', $FitTarget)
    }
    if ($GpuLayers) {
        $arguments += @('--n-gpu-layers', $GpuLayers)
    }
    $arguments += @('--alias', 'distributed-local')
}

if ($CpuMoeLayers) {
    Write-Host "Keeping Mixture of Experts weights of the first $CpuMoeLayers layers in system RAM."
    $arguments += @('--n-cpu-moe', $CpuMoeLayers)
}
if ($CpuFfnLayers) {
    Write-Host "Keeping dense feed-forward weights of the first $CpuFfnLayers layers in system RAM."
    $arguments += @('--n-cpu-ffn', $CpuFfnLayers)
}

if ($ModelPath) {
    $arguments += @('--model', $ModelPath)
}
else {
    $arguments += @('--hf-repo', $HuggingFaceRepository)
}

& $llamaServerPath @arguments
exit $LASTEXITCODE
