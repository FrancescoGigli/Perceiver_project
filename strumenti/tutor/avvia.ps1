param(
    [switch]$Lan,
    [switch]$Remote,
    [int]$Port = 8765
)
$tutorPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python310\python.exe'
if (-not (Test-Path -LiteralPath $tutorPython)) {
    throw 'Python 3.10 non trovato. Installa i requisiti e lancia python strumenti/tutor/app.py.'
}
$tutorVoiceConfigPath = Join-Path $env:LOCALAPPDATA 'PerceiverTutor\voice-runtime.json'
if (Test-Path -LiteralPath $tutorVoiceConfigPath) {
    $tutorVoiceConfig = Get-Content -LiteralPath $tutorVoiceConfigPath -Raw | ConvertFrom-Json
    if (-not $env:TUTOR_TTS_ROOT -and $tutorVoiceConfig.root) {
        $env:TUTOR_TTS_ROOT = [string]$tutorVoiceConfig.root
    }
    if (-not $env:TUTOR_TTS_PYTHON -and $tutorVoiceConfig.python) {
        $env:TUTOR_TTS_PYTHON = [string]$tutorVoiceConfig.python
    }
}
$tutorArguments = @((Join-Path $PSScriptRoot 'app.py'), '--port', $Port)
if ($Lan) { $tutorArguments += '--lan' }
if ($Remote) { $tutorArguments += '--remote' }
& $tutorPython @tutorArguments
