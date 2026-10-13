[CmdletBinding()]
param(
    [ValidateSet('Start','Status','Stop')][string]$Action = 'Start',
    [ValidateRange(1024,65535)][int]$Port = 8765,
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
$tutorCache = Join-Path $env:LOCALAPPDATA 'PerceiverTutor'
$sessionFile = Join-Path $tutorCache 'live-session.json'
$appPath = Join-Path $PSScriptRoot 'app.py'
$pythonPath = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
$state = $null
if (Test-Path -LiteralPath $sessionFile) { $state = Get-Content -LiteralPath $sessionFile -Raw | ConvertFrom-Json }
$backend = if ($state) { Get-CimInstance Win32_Process -Filter "ProcessId=$($state.ProcessId)" -ErrorAction SilentlyContinue } else { $null }
$ownedBackend = $backend -and $backend.CommandLine.Contains($appPath)
if ($Action -eq 'Status') {
    [pscustomobject]@{ Active=[bool]$ownedBackend; Url=if($ownedBackend){$state.Url}else{$null}; LinkFile=if($ownedBackend){Join-Path $tutorCache 'apri-tutor.html'}else{$null} }
    return
}
if ($Action -eq 'Stop') {
    & (Join-Path $PSScriptRoot 'phone_access.ps1') -Action Stop | Out-Null
    if ($ownedBackend) { Stop-Process -Id $backend.ProcessId }
    if (Test-Path -LiteralPath $sessionFile) { Remove-Item -LiteralPath $sessionFile }
    Write-Output 'Collegamento del tutor chiuso.'
    return
}
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Python 3.10 non trovato sul PC.' }
New-Item -ItemType Directory -Path $tutorCache -Force | Out-Null
if ($ownedBackend) {
    $Port = [int]$state.Port
    $tutorToken = $state.Token
} else {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw 'La porta e gia occupata. Scegli -Port con una porta libera.' }
    $bytes = New-Object byte[] 32
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    $random.GetBytes($bytes)
    $random.Dispose()
    $tutorToken = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
    $env:TUTOR_TOKEN = $tutorToken
    $env:TUTOR_QUIET_TOKEN = '1'
    $env:TUTOR_REMOTE = '1'
    $start = Start-Process -FilePath $pythonPath -ArgumentList @("`"$appPath`"",'--remote','--port',$Port) -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $tutorCache 'backend.out.log') -RedirectStandardError (Join-Path $tutorCache 'backend.err.log')
    $deadline = (Get-Date).AddSeconds(25)
    $ready = $false
    while ((Get-Date) -lt $deadline -and -not $start.HasExited) {
        try { Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -Headers @{'X-Tutor-Token'=$tutorToken} -TimeoutSec 2 | Out-Null; $ready=$true; break } catch { Start-Sleep -Milliseconds 300 }
        $start.Refresh()
    }
    if (-not $ready) { if(-not $start.HasExited){Stop-Process -Id $start.Id}; throw 'Il tutor non si e avviato. Controlla i log nella cache PerceiverTutor.' }
    $state = [pscustomobject]@{ProcessId=$start.Id;Port=$Port;Token=$tutorToken;Url=$null}
    $state | ConvertTo-Json | Set-Content -LiteralPath $sessionFile -Encoding UTF8
}
$env:TUTOR_TOKEN = $tutorToken
try {
    $tunnel = & (Join-Path $PSScriptRoot 'phone_access.ps1') -Action Start -Port $Port
    $state.Url = $tunnel.Url
    $state | ConvertTo-Json | Set-Content -LiteralPath $sessionFile -Encoding UTF8
    $link = $tunnel.Url + '/sito/studio/#tutor-token=' + $tutorToken
    $safeLink = [System.Net.WebUtility]::HtmlEncode($link)
    $sharePage = '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Apri il tuo tutor Perceiver</title><style>body{font:18px/1.6 Segoe UI,sans-serif;max-width:680px;margin:40px auto;padding:20px;color:#1a237e}a{overflow-wrap:anywhere}</style><h1>Il tuo tutor Perceiver</h1><p>Apri questo link sul cellulare. Il PC deve restare acceso.</p><p><a href="'+$safeLink+'">Apri il tutor</a></p><p>Il link contiene il tuo accesso: condividilo solo con chi vuoi far usare il tutor.</p></html>'
    [IO.File]::WriteAllText((Join-Path $tutorCache 'apri-tutor.html'), $sharePage, [Text.UTF8Encoding]::new($false))
    $playerLink = $tunnel.Url + '/sito/studio/player/#tutor-token=' + $tutorToken
    $safePlayerLink = [System.Net.WebUtility]::HtmlEncode($playerLink)
    $playerPage = '<!doctype html><html lang="it"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Il tuo player Perceiver</title><style>body{font:18px/1.6 Segoe UI,sans-serif;max-width:680px;margin:40px auto;padding:20px;color:#1a237e}a{overflow-wrap:anywhere}</style><h1>Il tuo player sul cellulare</h1><p>Apri questo link sul telefono per usare il player con i video generati sul PC. Il video parte solo quando lo chiedi.</p><p><a href="'+$safePlayerLink+'">Apri il player vocale</a></p><p>Su iPhone aprilo in Safari e usa Condividi, poi Aggiungi alla schermata Home. Per conservare l&#8217;accesso nell&#8217;app puoi selezionare Ricorda l&#8217;accesso su questo telefono nel player.</p><p>Per i comandi vocali lascia acceso il PC. Puoi scegliere un MP4 salvato sul telefono per guardarlo offline con i pulsanti.</p></html>'
    [IO.File]::WriteAllText((Join-Path $tutorCache 'apri-player.html'), $playerPage, [Text.UTF8Encoding]::new($false))
    if ($Quiet) { [pscustomobject]@{Active=$true;Url=$tunnel.Url;LinkFile=(Join-Path $tutorCache 'apri-tutor.html');PlayerLinkFile=(Join-Path $tutorCache 'apri-player.html')} }
    else { Write-Output "Apri il player dal cellulare: $playerLink"; Write-Output "Apri il tutor dal cellulare: $link"; Write-Output ('Pagina con il link: '+(Join-Path $tutorCache 'apri-player.html')) }
} finally {
    Remove-Item Env:TUTOR_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:TUTOR_QUIET_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:TUTOR_REMOTE -ErrorAction SilentlyContinue
}
