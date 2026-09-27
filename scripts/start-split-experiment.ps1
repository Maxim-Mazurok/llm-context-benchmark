param(
    [Parameter(Mandatory=$true)][int]$ReplacePid,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][int]$GpuLayers,
    [Parameter(Mandatory=$true)][string]$TensorSplit,
    [Parameter(Mandatory=$true)][string]$ModelPath,
    [Parameter(Mandatory=$true)][string]$LlamaServerExecutable,
    [int]$ContextSize=100000,
    [string]$Devices,
    [string]$Alias='distributed-local',
    [Parameter(Mandatory=$true)][string]$RpcServers,
    [ValidateSet('auto','none','mmap','mlock','mmap+mlock','dio')][string]$LoadMode='none',
    [ValidateSet('pinned','pageable')][string]$HostMemoryMode='pinned',
    [ValidateSet('f16','bf16','q8_0','q4_0')][string]$CacheTypeK='q8_0',
    [ValidateSet('f16','bf16','q8_0','q4_0')][string]$CacheTypeV='q8_0',
    [int]$Threads=16,
    [int]$Batch=512,
    [int]$UBatch=128,
    [string]$PythonExecutable='python',
    [switch]$AllowStopped
)
$ErrorActionPreference='Stop'
$exe=(Resolve-Path -LiteralPath $LlamaServerExecutable).Path
$old=Get-CimInstance Win32_Process -Filter "ProcessId=$ReplacePid"
if (-not $old -and $AllowStopped) {
    if (Get-NetTCPConnection -State Listen -LocalPort 8080 -ErrorAction SilentlyContinue) { throw 'Port 8080 is already in use' }
    if (Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'" | Where-Object CommandLine -like "*--alias $Alias*") {
        throw 'Another distributed-local server exists'
    }
} elseif (-not $old -or $old.ExecutablePath -ne $exe -or $old.CommandLine -notlike '*--alias distributed-local*') {
    throw 'Refusing to replace unexpected process'
}
if (Test-Path -LiteralPath $RunDirectory) { throw 'Run directory already exists; preserve previous artifacts' }
if ($old) {
    $slots=Invoke-RestMethod http://127.0.0.1:8080/slots -TimeoutSec 5
    if ($null -ne $slots.value) { $slots=$slots.value }
    foreach ($slot in $slots) { if ($slot.is_processing) { throw 'Server is busy' } }
}
New-Item -ItemType Directory -Path $RunDirectory | Out-Null
if ($old) {
    Stop-Process -Id $ReplacePid -Force
    Wait-Process -Id $ReplacePid -ErrorAction SilentlyContinue
}
foreach ($name in @('LLAMA_ARG_NO_HOST','GGML_CUDA_NO_PINNED','GGML_CUDA_REGISTER_HOST')) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
if ($HostMemoryMode -eq 'pageable') { $env:GGML_CUDA_NO_PINNED='1' }
$env:GGML_RPC_NO_HASH_CACHE='1'
$arguments=@('--ctx-size',"$ContextSize",'--batch-size',"$Batch",'--ubatch-size',"$UBatch",
    '--cache-type-k',$CacheTypeK,'--cache-type-v',$CacheTypeV,'--cache-ram','0','--ctx-checkpoints','0',
    '--parallel','1','--host','0.0.0.0','--port','8080','--metrics','--jinja','--no-mmproj',
    '--rpc',$RpcServers,'--load-mode',$LoadMode,'--lazy-mode','off','--split-mode','layer','--fit','off',
    '--tensor-split',$TensorSplit,'--n-gpu-layers',"$GpuLayers",'--threads',"$Threads",'--threads-batch',"$Threads")
if ($Devices) { $arguments += @('--device',$Devices) }
$arguments += @('--alias',$Alias,'--model',$ModelPath)
$logBase=Join-Path $RunDirectory 'server'
$process=Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory (Split-Path $exe) `
    -WindowStyle Hidden -RedirectStandardOutput "$logBase.stdout.log" -RedirectStandardError "$logBase.stderr.log" -PassThru
$record=[ordered]@{pid=$process.Id;started_utc=[DateTime]::UtcNow.ToString('o');arguments=$arguments;
    gpu_layers=$GpuLayers;tensor_split=$TensorSplit;model_path=$ModelPath;context_size=$ContextSize;devices=$Devices;alias=$Alias;rpc_servers=$RpcServers;
    threads=$Threads;batch=$Batch;ubatch=$UBatch;cache_type_k=$CacheTypeK;cache_type_v=$CacheTypeV;
    host_memory_mode=$HostMemoryMode;load_mode=$LoadMode;lazy_mode='off';GGML_RPC_NO_HASH_CACHE='1';
    stdout="$logBase.stdout.log";stderr="$logBase.stderr.log"}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RunDirectory 'launch.json')
@{status='loading';started_utc=$record.started_utc} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDirectory 'run-state.json')
$watchdog=Start-Process -FilePath $PythonExecutable -ArgumentList @(
    (Join-Path $PSScriptRoot 'windows-memory-watchdog.py'),'--pid',"$($process.Id)",
    '--run-directory',$RunDirectory,'--minimum-available-gib','12','--maximum-memory-load','82','--interval','0.5') `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunDirectory 'memory-watchdog.stdout.log') `
    -RedirectStandardError (Join-Path $RunDirectory 'memory-watchdog.stderr.log') -PassThru
$record['watchdog_pid']=$watchdog.Id
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RunDirectory 'launch.json')
$record | ConvertTo-Json -Depth 5
