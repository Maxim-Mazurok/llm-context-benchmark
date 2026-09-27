param([Parameter(Mandatory=$true)][string]$RunDirectory)
$ErrorActionPreference='Stop'
$launch=Get-Content (Join-Path $RunDirectory 'launch.json') -Raw | ConvertFrom-Json
$minimumAvailableRamGiB=if ($null -ne $launch.minimum_available_ram_gib) { [double]$launch.minimum_available_ram_gib } else { 12 }
$maximumGpuMemoryGiB=if ($null -ne $launch.maximum_gpu_memory_gib) { [double]$launch.maximum_gpu_memory_gib } else { 7 }
$workerSession=$null
if ($env:SPLIT_WORKER_PASSWORD) {
    if (-not $env:SPLIT_WORKER_ADDRESS) { throw 'Set SPLIT_WORKER_ADDRESS when using remote telemetry' }
    $workerUser=if ($env:SPLIT_WORKER_USER) { $env:SPLIT_WORKER_USER } else { 'Admin' }
    $credential=[pscredential]::new($workerUser,(ConvertTo-SecureString $env:SPLIT_WORKER_PASSWORD -AsPlainText -Force))
    Remove-Item Env:SPLIT_WORKER_PASSWORD
    $workerSession=New-PSSession -ComputerName $env:SPLIT_WORKER_ADDRESS -Credential $credential
}
$iteration=0
try {
    while ($true) {
        $server=Get-Process -Id $launch.pid -ErrorAction SilentlyContinue
        if (-not $server) { break }
        $counters=(Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
            '\Memory\Page Reads/sec','\Processor Information(_Total)\% Processor Performance').CounterSamples
        $gpu=$counters | Where-Object InstanceName -like "pid_$($launch.pid)_*"
        $os=Get-CimInstance Win32_OperatingSystem
        $row=[ordered]@{timestamp=[DateTime]::UtcNow.ToString('o');pid=$launch.pid;
            dedicated_bytes=($gpu|Where-Object Path -like '*Dedicated Usage'|Measure-Object CookedValue -Sum).Sum;
            shared_bytes=($gpu|Where-Object Path -like '*Shared Usage'|Measure-Object CookedValue -Sum).Sum;
            working_set_bytes=$server.WorkingSet64;private_bytes=$server.PrivateMemorySize64;cpu_seconds=$server.CPU;
            free_ram_bytes=([double]$os.FreePhysicalMemory*1024);
            system_page_reads_per_second=($counters|Where-Object Path -like '*\Memory\Page Reads/sec').CookedValue;
            cpu_performance_percent=($counters|Where-Object Path -like '*\% Processor Performance').CookedValue;
            gpu=(& nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits)}
        $state=Get-Content (Join-Path $RunDirectory 'run-state.json') -Raw | ConvertFrom-Json
        $row['status']=$state.status
        if ($workerSession -and ($iteration%3 -eq 0)) {
            $row['worker']=Invoke-Command -Session $workerSession -ScriptBlock {
                $process=Get-Process -Name ggml-rpc-server
                $remoteCounters=(Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
                    '\Memory\Page Reads/sec','\Processor Information(_Total)\% Processor Performance').CounterSamples
                $gpu=$remoteCounters | Where-Object InstanceName -like "pid_$($process.Id)_*"
                [pscustomobject]@{pid=$process.Id;working_set_bytes=$process.WorkingSet64;cpu_seconds=$process.CPU;
                    free_ram_bytes=([double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024);
                    dedicated_bytes=($gpu|Where-Object Path -like '*Dedicated Usage'|Measure-Object CookedValue -Sum).Sum;
                    shared_bytes=($gpu|Where-Object Path -like '*Shared Usage'|Measure-Object CookedValue -Sum).Sum;
                    system_page_reads_per_second=($remoteCounters|Where-Object Path -like '*\Memory\Page Reads/sec').CookedValue;
                    cpu_performance_percent=($remoteCounters|Where-Object Path -like '*\% Processor Performance').CookedValue;
                    gpu=(& nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits)}
            } | Select-Object pid,working_set_bytes,cpu_seconds,free_ram_bytes,dedicated_bytes,shared_bytes,system_page_reads_per_second,cpu_performance_percent,gpu
        }
        $row | ConvertTo-Json -Depth 5 -Compress | Add-Content (Join-Path $RunDirectory 'telemetry.jsonl')
        $minimumAvailableRamBytes=$minimumAvailableRamGiB*1GB
        $maximumGpuMemoryBytes=$maximumGpuMemoryGiB*1GB
        if ($row.free_ram_bytes -lt $minimumAvailableRamBytes -or $row.dedicated_bytes -gt $maximumGpuMemoryBytes) {
            @{reason='Local memory safety limit';measurement=$row} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $RunDirectory 'abort.json')
            Stop-Process -Id $launch.pid -Force
            break
        }
        if ($workerSession -and ($row.worker.free_ram_bytes -lt $minimumAvailableRamBytes -or
            $row.worker.dedicated_bytes -gt $maximumGpuMemoryBytes)) {
            @{reason='RPC worker memory safety limit';measurement=$row} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $RunDirectory 'abort.json')
            $workerPid=[int]$row.worker.pid
            Invoke-Command -Session $workerSession -ArgumentList $workerPid -ScriptBlock {
                param($processId)
                Stop-Process -Id $processId -Force
            }
            break
        }
        if ($state.status -notin @('loading','running')) { break }
        $iteration++
        Start-Sleep -Seconds 20
    }
} finally {
    if ($workerSession) { Remove-PSSession $workerSession }
}
