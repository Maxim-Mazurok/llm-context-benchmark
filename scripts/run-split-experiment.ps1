param([Parameter(Mandatory=$true)][string]$RunDirectory,[string]$WaitForRun,
    [ValidateSet('split-search-bench.py','repeat-stream-probe.py')][string]$BenchmarkScript='split-search-bench.py',
    [ValidateSet('sample-split-experiment.ps1','sample-mac-split-experiment.ps1')][string]$TelemetryScript='sample-split-experiment.ps1',
    [int]$LoadTimeoutMinutes=45,
    [string]$PythonExecutable='python')
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot
if ($WaitForRun) {
    $waitDeadline=[DateTime]::UtcNow.AddMinutes(30)
    while ($true) {
        $prior=Get-Content (Join-Path $WaitForRun 'run-state.json') -Raw | ConvertFrom-Json
        if ($prior.status -eq 'completed') { break }
        if ($prior.status -notin @('running','loading')) { throw "Prior run stopped: $($prior.status)" }
        if ([DateTime]::UtcNow -gt $waitDeadline) { throw 'Prior run wait exceeded deadline' }
        Start-Sleep -Seconds 5
    }
}
$launch=Get-Content (Join-Path $RunDirectory 'launch.json') -Raw | ConvertFrom-Json
$telemetry=Start-Process powershell.exe -ArgumentList @('-NoProfile','-File',
    (Join-Path $PSScriptRoot $TelemetryScript),'-RunDirectory',$RunDirectory) `
    -WindowStyle Hidden -RedirectStandardOutput (Join-Path $RunDirectory 'telemetry.stdout.log') `
    -RedirectStandardError (Join-Path $RunDirectory 'telemetry.stderr.log') -PassThru
$telemetryHandle=$telemetry.Handle
Remove-Item Env:SPLIT_WORKER_PASSWORD -ErrorAction SilentlyContinue
$deadline=[DateTime]::UtcNow.AddMinutes($LoadTimeoutMinutes)
while ($true) {
    if (-not (Get-Process -Id $launch.pid -ErrorAction SilentlyContinue)) {
        @{status='failed_load';error='Server exited while loading';scored=$false} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $RunDirectory 'run-state.json')
        throw 'Server exited while loading'
    }
    try { $health=Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 3; if ($health.status -eq 'ok') { break } } catch { }
    if ([DateTime]::UtcNow -gt $deadline) {
        @{status='failed_load';error='Load deadline exceeded';scored=$false} | ConvertTo-Json |
            Set-Content -LiteralPath (Join-Path $RunDirectory 'run-state.json')
        throw 'Load deadline exceeded'
    }
    Start-Sleep -Seconds 10
}
& $PythonExecutable (Join-Path $PSScriptRoot 'trim-process-working-set.py') `
    --pid $launch.pid --output (Join-Path $RunDirectory 'working-set-trim.json')
if ($LASTEXITCODE -ne 0) { throw "Working-set trim failed: $LASTEXITCODE" }
Start-Sleep -Seconds 2
Write-Output "$(Get-Date -Format o) Ready; starting $BenchmarkScript"
& $PythonExecutable -u (Join-Path $PSScriptRoot $BenchmarkScript) $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed: $LASTEXITCODE" }
$state=Get-Content (Join-Path $RunDirectory 'run-state.json') -Raw | ConvertFrom-Json
if ($state.status -ne 'completed') { throw "Unexpected final state: $($state.status)" }
$telemetry.WaitForExit(30000) | Out-Null
Write-Output "$(Get-Date -Format o) Experiment completed; server remains running"
