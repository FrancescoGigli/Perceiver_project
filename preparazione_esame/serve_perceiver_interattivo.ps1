param(
  [int]$Port = 8770,
  [string]$BindAddress = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "logs"
$LogPath = Join-Path $LogDir "perceiver_interattivo_server.log"

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

function Write-ServerLog {
  param([string]$Message)
  $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  Add-Content -LiteralPath $LogPath -Value "[$stamp] $Message"
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) {
  throw "python.exe not found in PATH"
}

if ([string]::IsNullOrWhiteSpace($BindAddress)) {
  $Tailscale = (Get-Command tailscale -ErrorAction SilentlyContinue).Source
  if (-not $Tailscale) {
    throw "tailscale.exe not found in PATH"
  }

  Write-ServerLog "Waiting for Tailscale IPv4..."
  while (-not $BindAddress) {
    $BindAddress = (& $Tailscale ip -4 2>$null | Where-Object { $_ -match '^100\.' } | Select-Object -First 1)
    if (-not $BindAddress) {
      Start-Sleep -Seconds 5
    }
  }
}

$Existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
  Where-Object { $_.LocalAddress -eq $BindAddress -or $_.LocalAddress -eq "0.0.0.0" }

if ($Existing) {
  Write-ServerLog "Port $Port is already listening on $($Existing[0].LocalAddress). Nothing to start."
  exit 0
}

Set-Location -LiteralPath $Root
Write-ServerLog "Serving $Root on http://${BindAddress}:$Port/"

& $Python -m http.server $Port --bind $BindAddress 2>&1 |
  ForEach-Object { Write-ServerLog $_.ToString() }
