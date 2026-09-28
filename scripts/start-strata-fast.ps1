param(
    [Parameter(Mandatory=$true)][string]$StrataRoot,
    [string]$ConfigPath,
    [int]$Port=8081,
    [string]$RunDirectory,
    [ValidateRange(1, 99)][int]$MaximumMemoryLoadPercent=95,
    [ValidateRange(0.5, 1024)][double]$MinimumAvailableRamGiB=3,
    [ValidateRange(30, 1800)][int]$LoadTimeoutSeconds=600,
    [string]$PythonExecutable,
    [switch]$DisableMemoryWatchdog
)

$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $StrataRoot).Path
if (-not $ConfigPath) { $ConfigPath=Join-Path $root 'strata-iq3_xxs.json' }
$config=(Resolve-Path -LiteralPath $ConfigPath).Path
if (-not $PythonExecutable) { $PythonExecutable=Join-Path $root '.venv\Scripts\python.exe' }
$python=(Resolve-Path -LiteralPath $PythonExecutable).Path
$server=(Resolve-Path -LiteralPath (Join-Path $root 'serve\server.py')).Path

$listener=Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($listener) {
    $health=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5
    Write-Output "Strata is already healthy on port $Port (PID $($listener.OwningProcess), model $($health.model))."
    exit 0
}

$settings=Get-Content -LiteralPath $config -Raw | ConvertFrom-Json
$engineArgs=@($settings.args)
function Get-EngineArgument([string]$Name) {
    $index=[Array]::IndexOf($engineArgs, $Name)
    if ($index -lt 0 -or $index + 1 -ge $engineArgs.Count) { return $null }
    return $engineArgs[$index + 1]
}

if ((Get-EngineArgument '--prefill') -ne 'auto') {
    throw 'The Strata config must use "--prefill", "auto" for the supported adaptive fast path.'
}
if ((Get-EngineArgument '--expert-cache') -ne 'auto') {
    throw 'The Strata config must use "--expert-cache", "auto".'
}
if (-not (Get-EngineArgument '--vram-reserve-mib')) {
    throw 'The Strata config must set --vram-reserve-mib explicitly.'
}

if (-not $RunDirectory) {
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $RunDirectory=Join-Path $PSScriptRoot "..\runs\strata-$stamp"
}
if (Test-Path -LiteralPath $RunDirectory) { throw 'Run directory already exists; preserve previous artifacts.' }
$run=New-Item -ItemType Directory -Path $RunDirectory

$stdout=Join-Path $run.FullName 'server.stdout.log'
$stderr=Join-Path $run.FullName 'server.stderr.log'
$arguments=@("`"$server`"",'--engine','strata','--config',"`"$config`"",'--host','0.0.0.0','--port',"$Port")
$previousArenaRegister=$env:STRATA_ARENA_REGISTER
$previousMetricsLog=$env:STRATA_METRICS_LOG
try {
    $env:STRATA_ARENA_REGISTER='0'
    $env:STRATA_METRICS_LOG=Join-Path $run.FullName 'inference-metrics.jsonl'
    $launcher=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
        -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
} finally {
    if ($null -eq $previousArenaRegister) { Remove-Item Env:STRATA_ARENA_REGISTER -ErrorAction SilentlyContinue }
    else { $env:STRATA_ARENA_REGISTER=$previousArenaRegister }
    if ($null -eq $previousMetricsLog) { Remove-Item Env:STRATA_METRICS_LOG -ErrorAction SilentlyContinue }
    else { $env:STRATA_METRICS_LOG=$previousMetricsLog }
}

$deadline=(Get-Date).AddSeconds($LoadTimeoutSeconds)
$engine=$null
while ((Get-Date) -lt $deadline -and -not $engine) {
    $processes=Get-CimInstance Win32_Process
    $descendants=@($launcher.Id)
    do {
        $before=$descendants.Count
        $children=$processes | Where-Object { $_.ParentProcessId -in $descendants } | Select-Object -ExpandProperty ProcessId
        $descendants=@($descendants + $children | Select-Object -Unique)
    } while ($descendants.Count -gt $before)
    $engine=$processes | Where-Object { $_.Name -eq 'strata.exe' -and $_.ProcessId -in $descendants } |
        Select-Object -First 1
    if (-not $engine) {
        if ($launcher.HasExited) { throw "Strata launcher exited before creating the engine; see $stderr" }
        Start-Sleep -Milliseconds 500
    }
}
if (-not $engine) {
    Stop-Process -Id $launcher.Id -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for strata.exe; see $stderr"
}

$watchdog=$null
if (-not $DisableMemoryWatchdog) {
    $watchdog=Start-Process -FilePath $python -ArgumentList @(
        "`"$(Join-Path $PSScriptRoot 'windows-memory-watchdog.py')`"",'--pid',"$($engine.ProcessId)",
        '--run-directory',"`"$($run.FullName)`"",'--minimum-available-gib',"$MinimumAvailableRamGiB",
        '--maximum-memory-load',"$MaximumMemoryLoadPercent",'--interval','0.5') `
        -WindowStyle Hidden -RedirectStandardOutput (Join-Path $run.FullName 'memory-watchdog.stdout.log') `
        -RedirectStandardError (Join-Path $run.FullName 'memory-watchdog.stderr.log') -PassThru
} else {
    Write-Warning 'RAM protection is disabled; Strata will not be stopped automatically under memory pressure.'
}

$record=[ordered]@{
    started_utc=[DateTime]::UtcNow.ToString('o')
    launcher_pid=$launcher.Id
    engine_pid=$engine.ProcessId
    memory_watchdog_enabled=(-not $DisableMemoryWatchdog)
    watchdog_pid=if ($watchdog) { $watchdog.Id } else { $null }
    config=$config
    port=$Port
    strata_arena_register='0'
    maximum_memory_load_percent=if ($DisableMemoryWatchdog) { $null } else { $MaximumMemoryLoadPercent }
    minimum_available_ram_gib=if ($DisableMemoryWatchdog) { $null } else { $MinimumAvailableRamGiB }
    stdout=$stdout
    stderr=$stderr
}
$record | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $run.FullName 'launch.json')

$health=$null
while ((Get-Date) -lt $deadline -and -not $health) {
    if (-not (Get-Process -Id $engine.ProcessId -ErrorAction SilentlyContinue)) {
        throw "Strata exited before it was ready; see $stderr and $($settings.log)"
    }
    try {
        $health=Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $health) {
    Stop-Process -Id $engine.ProcessId -Force -ErrorAction SilentlyContinue
    throw "Timed out waiting for Strata health on port $Port"
}

Write-Output "Strata is ready at http://127.0.0.1:$Port/ (engine PID $($engine.ProcessId))."
Write-Output "Model: $($health.model); context: $($health.max_context); run: $($run.FullName)"