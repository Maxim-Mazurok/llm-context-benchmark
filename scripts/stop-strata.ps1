param(
    [Parameter(Mandatory=$true)][string]$StrataRoot,
    [int]$Port=8081
)

$ErrorActionPreference='Stop'
$root=(Resolve-Path -LiteralPath $StrataRoot).Path
$serverPath=(Resolve-Path -LiteralPath (Join-Path $root 'serve\server.py')).Path
$listener=Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $listener) {
    Write-Output "Nothing is listening on port $Port."
    exit 0
}

$processes=@(Get-CimInstance Win32_Process)
$server=$processes | Where-Object ProcessId -eq $listener.OwningProcess
if ([string]::IndexOf($server.CommandLine,$serverPath,[StringComparison]::OrdinalIgnoreCase) -lt 0) {
    throw "Refusing to stop unexpected port $Port owner PID $($listener.OwningProcess): $($server.CommandLine)"
}

$processIds=@($server.ProcessId)
do {
    $before=$processIds.Count
    $children=$processes | Where-Object {
        $_.ParentProcessId -in $processIds -and
        ($_.Name -eq 'strata.exe' -or
            [string]::IndexOf($_.CommandLine,$root,[StringComparison]::OrdinalIgnoreCase) -ge 0)
    } | Select-Object -ExpandProperty ProcessId
    $processIds=@($processIds + $children | Select-Object -Unique)
} while ($processIds.Count -gt $before)

$parent=$processes | Where-Object ProcessId -eq $server.ParentProcessId
if ($parent -and [string]::IndexOf($parent.CommandLine,$root,[StringComparison]::OrdinalIgnoreCase) -ge 0) {
    $processIds=@($processIds + $parent.ProcessId | Select-Object -Unique)
}

$processIds | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
$processIds | ForEach-Object { Wait-Process -Id $_ -ErrorAction SilentlyContinue }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is still listening after stopping Strata."
}
Write-Output "Stopped Strata process IDs: $($processIds -join ', ')"