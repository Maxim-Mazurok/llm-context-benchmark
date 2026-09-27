param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$launch=Get-Content (Join-Path $RunDirectory 'launch.json') -Raw | ConvertFrom-Json
$ssh=if ($env:LLAMA_MAC_SSH_EXE) { $env:LLAMA_MAC_SSH_EXE } else { (Get-Command ssh -ErrorAction Stop).Source }
$key=$env:LLAMA_MAC_SSH_KEY
$known=$env:LLAMA_MAC_KNOWN_HOSTS
$macHost=$env:LLAMA_MAC_HOST
$macUser=$env:LLAMA_MAC_USER
if (-not $key -or -not $known -or -not $macHost -or -not $macUser) {
    throw 'Set LLAMA_MAC_SSH_KEY, LLAMA_MAC_KNOWN_HOSTS, LLAMA_MAC_HOST, and LLAMA_MAC_USER'
}
foreach ($path in @($ssh,$key,$known)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
$iteration=0
while ($true) {
    $server=Get-Process -Id $launch.pid -ErrorAction SilentlyContinue
    if (-not $server) { break }
    $counters=(Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
        '\Memory\Page Reads/sec','\Memory\% Committed Bytes In Use',
        '\Processor Information(_Total)\% Processor Performance').CounterSamples
    $gpu=$counters | Where-Object InstanceName -like "pid_$($launch.pid)_*"
    $os=Get-CimInstance Win32_OperatingSystem
    $row=[ordered]@{timestamp=[DateTime]::UtcNow.ToString('o');pid=$launch.pid;
        dedicated_bytes=($gpu|Where-Object Path -like '*Dedicated Usage'|Measure-Object CookedValue -Sum).Sum;
        shared_bytes=($gpu|Where-Object Path -like '*Shared Usage'|Measure-Object CookedValue -Sum).Sum;
        working_set_bytes=$server.WorkingSet64;private_bytes=$server.PrivateMemorySize64;cpu_seconds=$server.CPU;
        free_ram_bytes=([double]$os.FreePhysicalMemory*1024);
        committed_percent=($counters|Where-Object Path -like '*\Memory\% Committed Bytes In Use').CookedValue;
        system_page_reads_per_second=($counters|Where-Object Path -like '*\Memory\Page Reads/sec').CookedValue;
        cpu_performance_percent=($counters|Where-Object Path -like '*\% Processor Performance').CookedValue;
        gpu=(& nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits)}
    $state=Get-Content (Join-Path $RunDirectory 'run-state.json') -Raw | ConvertFrom-Json
    $row['status']=$state.status
    if ($iteration%3 -eq 0) {
        $remote='memory_pressure -Q; sysctl vm.swapusage; ps -o pid=,rss=,vsz=,pcpu= -p $(cat "$HOME/qwen-mac-rpc-50053.pid")'
        $row['mac_raw']=(& $ssh -i $key -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes `
            -o "UserKnownHostsFile=$known" "${macUser}@${macHost}" $remote 2>&1) -join "`n"
    }
    $row | ConvertTo-Json -Depth 4 -Compress | Add-Content (Join-Path $RunDirectory 'telemetry.jsonl')
    if ($row.free_ram_bytes -lt 8GB -or $row.dedicated_bytes -gt 7.4GB) {
        @{reason='Windows memory safety limit';measurement=$row} | ConvertTo-Json -Depth 4 |
            Set-Content (Join-Path $RunDirectory 'abort.json')
    }
    if ($row.free_ram_bytes -lt 8GB -or $row.dedicated_bytes -gt 7.7GB) {
        @{reason='Emergency Windows memory pressure';measurement=$row} | ConvertTo-Json -Depth 4 |
            Set-Content (Join-Path $RunDirectory 'abort.json')
        Stop-Process -Id $launch.pid -Force
        break
    }
    if ($state.status -notin @('loading','running')) { break }
    $iteration++
    Start-Sleep -Seconds 5
}
