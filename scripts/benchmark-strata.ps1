param(
    [string]$Endpoint='http://127.0.0.1:8081',
    [Parameter(Mandatory=$true)][string]$EngineLogPath,
    [ValidateRange(1, 10000)][int]$PromptRecords=350,
    [ValidateRange(1, 4096)][int]$MaxTokens=128,
    [string]$OutputDirectory
)

$ErrorActionPreference='Stop'
$log=(Resolve-Path -LiteralPath $EngineLogPath).Path
if (-not $OutputDirectory) {
    $stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
    $OutputDirectory=Join-Path $PSScriptRoot "..\runs\strata-probe-$stamp"
}
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Output directory already exists; preserve previous results.' }
$output=New-Item -ItemType Directory -Path $OutputDirectory

$baseUri=$Endpoint.TrimEnd('/')
$health=Invoke-RestMethod -Uri "$baseUri/health" -TimeoutSec 10
$nonce=[Guid]::NewGuid().ToString('N')
$records=1..$PromptRecords | ForEach-Object {
    "Probe $nonce record $_ contains stable words for a cold prompt throughput measurement."
}
$prompt=($records -join "`n")+"`nSummarize the records in one sentence."
$body=@{
    model=$health.model
    messages=@(@{role='user';content=$prompt})
    max_tokens=$MaxTokens
    temperature=0
} | ConvertTo-Json -Depth 5

$logOffset=(Get-Item -LiteralPath $log).Length
$started=Get-Date
$response=Invoke-RestMethod -Method Post -Uri "$baseUri/v1/chat/completions" `
    -ContentType 'application/json' -Body $body -TimeoutSec 900
$finished=Get-Date

$stream=[System.IO.File]::Open($log,[System.IO.FileMode]::Open,[System.IO.FileAccess]::Read,[System.IO.FileShare]::ReadWrite)
$reader=$null
try {
    $null=$stream.Seek($logOffset,[System.IO.SeekOrigin]::Begin)
    $reader=[System.IO.StreamReader]::new($stream)
    $newLog=$reader.ReadToEnd()
} finally {
    if ($reader) { $reader.Dispose() } else { $stream.Dispose() }
}

$pattern='prompt (?<total>\d+) tokens = (?<reused>\d+) reused \+ (?<read>\d+) read in (?<prefill_ms>[\d.]+) ms \((?<prefill_tps>[\d.]+) tok/s\), (?<generated>\d+) generated in (?<decode_ms>[\d.]+) ms \((?<decode_tps>[\d.]+) tok/s\)'
$matches=[regex]::Matches($newLog,$pattern)
if ($matches.Count -eq 0) {
    $newLog | Set-Content -LiteralPath (Join-Path $output.FullName 'engine-log-fragment.txt')
    throw 'The request completed, but no Strata timing line appeared in the engine log.'
}
$timing=$matches[$matches.Count - 1]
$summary=[ordered]@{
    timestamp_utc=[DateTime]::UtcNow.ToString('o')
    endpoint=$Endpoint
    model=$health.model
    max_context=[int64]$health.max_context
    prompt_tokens=[int]$timing.Groups['total'].Value
    reused_tokens=[int]$timing.Groups['reused'].Value
    read_tokens=[int]$timing.Groups['read'].Value
    prefill_ms=[double]$timing.Groups['prefill_ms'].Value
    prefill_tokens_per_second=[double]$timing.Groups['prefill_tps'].Value
    generated_tokens=[int]$timing.Groups['generated'].Value
    decode_ms=[double]$timing.Groups['decode_ms'].Value
    decode_tokens_per_second=[double]$timing.Groups['decode_tps'].Value
    wall_seconds=($finished-$started).TotalSeconds
    finish_reason=$response.choices[0].finish_reason
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $output.FullName 'summary.json')
$response | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $output.FullName 'response.json')
$summary | ConvertTo-Json -Depth 4