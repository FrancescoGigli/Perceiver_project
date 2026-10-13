# HTTPS temporaneo per il tutor locale. Nessun account, installazione o regola firewall.
# Docs: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/
# Il backend deve partire in modalita --remote e richiedere TUTOR_TOKEN.
# Il collegamento segreto per lo studente usa un frammento #token=... creato dal launcher;
# questo script non stampa, registra o passa il token sulla riga di comando.
[CmdletBinding()]
param(
    [ValidateSet('Prepare', 'Start', 'Status', 'Stop')]
    [string]$Action = 'Start',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8765,
    [ValidateRange(5, 120)]
    [int]$TimeoutSeconds = 45
)

$ErrorActionPreference = 'Stop'
$release = '2026.10.0'
$checksum = '86aee4017b26625cee8484c113558f48effa4cd47f7aa05fcf425604e5d2b23c'
$cacheDirectory = Join-Path $env:LOCALAPPDATA 'PerceiverTutor\phone'
$binaryPath = Join-Path $cacheDirectory "cloudflared-$release.exe"
$statePath = Join-Path $cacheDirectory 'active.json'

function Get-TunnelState {
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    $process = Get-Process -Id $state.ProcessId -ErrorAction SilentlyContinue
    # Un PID riutilizzato non deve permettere di fermare un'altra applicazione.
    if ($null -eq $process -or $process.Path -ne $binaryPath) { return $null }
    return $state
}

if ($Action -eq 'Status') {
    $state = Get-TunnelState
    if ($null -eq $state) {
        [pscustomobject]@{ Active = $false; Url = $null }
    } else {
        [pscustomobject]@{ Active = $true; Url = $state.Url; ProcessId = $state.ProcessId }
    }
    return
}

if ($Action -eq 'Stop') {
    $state = Get-TunnelState
    if ($null -ne $state) { Stop-Process -Id $state.ProcessId }
    if (Test-Path -LiteralPath $statePath) { Remove-Item -LiteralPath $statePath }
    [pscustomobject]@{ Active = $false; Stopped = ($null -ne $state) }
    return
}

# Valida il segreto prima di scaricare file o aprire un tunnel.
if ($Action -eq 'Start' -and ([string]::IsNullOrWhiteSpace($env:TUTOR_TOKEN) -or $env:TUTOR_TOKEN.Length -lt 24)) {
    throw 'Imposta TUTOR_TOKEN nel processo del launcher (almeno 24 caratteri). Il tunnel non e stato avviato.'
}

New-Item -ItemType Directory -Path $cacheDirectory -Force | Out-Null
if (-not (Test-Path -LiteralPath $binaryPath)) {
    $downloadUrl = "https://github.com/cloudflare/cloudflared/releases/download/$release/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $binaryPath -UseBasicParsing
}
$actualChecksum = (Get-FileHash -LiteralPath $binaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualChecksum -ne $checksum) {
    throw 'Checksum cloudflared non valido: il programma non verra eseguito. Elimina solo il file indicato nella cache e riprova.'
}
if ($Action -eq 'Prepare') {
    [pscustomobject]@{ Prepared = $true; Version = $release; ChecksumVerified = $true; BinaryPath = $binaryPath }
    return
}

$existing = Get-TunnelState
if ($null -ne $existing) {
    if ($existing.Port -ne $Port) { throw 'Un tunnel usa gia un altro backend. Esegui -Action Stop prima di cambiarlo.' }
    [pscustomobject]@{ Active = $true; Url = $existing.Url; ProcessId = $existing.ProcessId; Reused = $true }
    return
}

$origin = "http://127.0.0.1:$Port"
# La porta deve appartenere al backend autorizzato: prima verifica autenticazione,
# poi verifica che una chiamata senza token sia rifiutata. Nessun token nei log.
$authHeaders = @{ 'X-Tutor-Token' = $env:TUTOR_TOKEN }
try {
    $health = Invoke-RestMethod -Uri "$origin/api/health" -Headers $authHeaders -TimeoutSec 10
} catch {
    throw 'Il tutor non risponde sulla porta richiesta o il token non coincide. Avvia prima il backend --remote.'
}
try {
    Invoke-WebRequest -Uri "$origin/api/health" -UseBasicParsing -TimeoutSec 10 | Out-Null
    throw 'AUTHENTICATION_NOT_REQUIRED'
} catch {
    $statusCode = $null
    if ($null -ne $_.Exception.Response) { $statusCode = [int]$_.Exception.Response.StatusCode }
    if ($statusCode -ne 401 -and $statusCode -ne 403) {
        throw 'Il backend non rifiuta gli accessi senza token. Nessun tunnel e stato aperto: usa --remote.'
    }
}

# Configurazione isolata: non legge o altera eventuali tunnel dell'utente.
$configPath = Join-Path $cacheDirectory 'quick-tunnel.yml'
Set-Content -LiteralPath $configPath -Value '{}' -Encoding UTF8
$logId = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
$stdoutPath = Join-Path $cacheDirectory "$logId.out.log"
$stderrPath = Join-Path $cacheDirectory "$logId.err.log"
$arguments = @('tunnel', '--config', "`"$configPath`"", '--no-autoupdate', '--url', $origin, '--loglevel', 'info', '--metrics', '127.0.0.1:0')
$tunnelProcess = Start-Process -FilePath $binaryPath -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$url = $null
while ((Get-Date) -lt $deadline -and -not $tunnelProcess.HasExited) {
    $logText = ''
    foreach ($logPath in @($stdoutPath, $stderrPath)) {
        if (Test-Path -LiteralPath $logPath) { $logText += Get-Content -LiteralPath $logPath -Raw }
    }
    $urlMatch = [regex]::Match($logText, 'https://[a-z0-9-]+\.trycloudflare\.com')
    if ($urlMatch.Success) { $url = $urlMatch.Value; break }
    Start-Sleep -Milliseconds 250
    $tunnelProcess.Refresh()
}
if ($null -eq $url) {
    if (-not $tunnelProcess.HasExited) { Stop-Process -Id $tunnelProcess.Id }
    throw "Cloudflare non ha restituito un URL. I log senza token sono nella cache $cacheDirectory."
}

$state = [pscustomobject]@{
    Url = $url
    ProcessId = $tunnelProcess.Id
    Port = $Port
    StartedAt = (Get-Date).ToUniversalTime().ToString('o')
}
$state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
# URL pubblico del trasporto, senza il frammento segreto. Il backend resta protetto.
[pscustomobject]@{ Active = $true; Url = $url; ProcessId = $tunnelProcess.Id; Reused = $false }
