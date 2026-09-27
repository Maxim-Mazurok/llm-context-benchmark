param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$destination=Join-Path $RunDirectory 'windows-telemetry.jsonl'
while ($true) {
    $server=Get-Process -Name llama-server -ErrorAction SilentlyContinue
    if (@($server).Count -ne 1) { throw 'Expected exactly one llama-server' }
    $gpu=(Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage').CounterSamples |
        Where-Object { $_.InstanceName -like "pid_$($server.Id)_*" }
    $os=Get-CimInstance Win32_OperatingSystem
    $row=[ordered]@{
        timestamp=[DateTime]::UtcNow.ToString('o')
        pid=$server.Id
        dedicated_bytes=($gpu|Where-Object Path -like '*Dedicated Usage'|Measure-Object CookedValue -Sum).Sum
        shared_bytes=($gpu|Where-Object Path -like '*Shared Usage'|Measure-Object CookedValue -Sum).Sum
        working_set_bytes=$server.WorkingSet64
        private_bytes=$server.PrivateMemorySize64
        cpu_seconds=$server.CPU
        free_ram_bytes=([double]$os.FreePhysicalMemory*1024)
        gpu=( & nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits )
    }
    $row | ConvertTo-Json -Compress | Add-Content -LiteralPath $destination
    $state=Get-Content -LiteralPath (Join-Path $RunDirectory 'run-state.json') -Raw | ConvertFrom-Json
    if ($state.status -notin @('running','loading','restoring')) { break }
    Start-Sleep -Seconds 20
}
