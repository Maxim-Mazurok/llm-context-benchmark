param(
    [string]$LlamaCppDirectory = "$HOME\llama.cpp",
    [string]$HostAddress = "0.0.0.0",
    [ValidateRange(1, 65535)]
    [int]$Port = 50052,
    [string]$Device = "CUDA0",
    [int]$Threads = 0,
    [string]$AllowedClientAddress = "Any"
)

$ErrorActionPreference = "Stop"
$rpcServerPath = Join-Path $LlamaCppDirectory "build-rpc-cuda\bin\Release\ggml-rpc-server.exe"
if (-not (Test-Path $rpcServerPath)) {
    throw "RPC worker not found at $rpcServerPath. Run setup-llama-cpp-windows.ps1 first."
}

$isAdministrator = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdministrator) {
    throw "Run PowerShell as Administrator to configure the firewall rule."
}
$firewallRuleName = "llama.cpp RPC on private networks"
Remove-NetFirewallRule -DisplayName $firewallRuleName -ErrorAction SilentlyContinue
New-NetFirewallRule `
    -DisplayName $firewallRuleName `
    -Direction Inbound `
    -Action Allow `
    -Protocol TCP `
    -LocalPort $Port `
    -RemoteAddress $AllowedClientAddress `
    -Profile Private | Out-Null

Write-Warning "RPC has no authentication or encryption. Use only on a trusted private network."

$arguments = @('--host', $HostAddress, '--port', $Port, '--device', $Device, '--cache')
if ($Threads -gt 0) {
    $arguments += @('--threads', $Threads)
}

if ($Device -match ',') {
    Write-Host "Exposing multiple devices ($Device); the host tensor split decides the GPU/RAM ratio on this worker."
}

& $rpcServerPath @arguments
exit $LASTEXITCODE