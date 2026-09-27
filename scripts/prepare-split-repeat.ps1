param(
    [Parameter(Mandatory=$true)][string]$SourceRun,
    [Parameter(Mandatory=$true)][string]$RunDirectory,
    [Parameter(Mandatory=$true)][string]$LlamaServerExecutable
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $RunDirectory) { throw 'Run directory already exists; preserve artifacts' }
$launch=Get-Content (Join-Path $SourceRun 'launch.json') -Raw | ConvertFrom-Json
$process=Get-CimInstance Win32_Process -Filter "ProcessId=$($launch.pid)"
$expectedExecutable=(Resolve-Path -LiteralPath $LlamaServerExecutable).Path
if (-not $process -or $process.ExecutablePath -ne $expectedExecutable -or
    $process.CommandLine -notlike '*--alias distributed-local*') { throw 'Expected server is not running' }
$slots=Invoke-RestMethod http://127.0.0.1:8080/slots -TimeoutSec 5
if ($null -ne $slots.value) { $slots=$slots.value }
foreach ($slot in $slots) { if ($slot.is_processing) { throw 'Server is busy' } }
New-Item -ItemType Directory -Path $RunDirectory | Out-Null
$launch | Add-Member -NotePropertyName reused_from -NotePropertyValue $SourceRun -Force
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $RunDirectory 'launch.json')
@{status='loading';reused_server=$true;prepared_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunDirectory 'run-state.json')
Write-Output "Prepared repeat with existing server PID $($launch.pid). Run run-split-experiment.ps1 on the new directory."
