# Esporta alcune slide del deck in PNG, per controllare a occhio le modifiche.
#   powershell -File _diag/render.ps1 -Slides "14,15,16" -Out C:\percorso
# LibreOffice non c'e' su questa macchina: si passa da PowerPoint via COM.
param(
  [string]$Deck = "$PSScriptRoot\..\Perceiver & Perceiver IO.pptx",
  [string]$Slides = "",
  [string]$Out = "$env:TEMP\pptx_render"
)

$Deck = (Resolve-Path $Deck).Path
$lock = Join-Path (Split-Path $Deck) ("~$" + (Split-Path $Deck -Leaf))
if (Test-Path $lock) { Write-Error "Il deck e' aperto in PowerPoint: chiudilo."; exit 1 }
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($Deck, $true, $false, $false)   # ReadOnly
try {
  $indici = if ($Slides) { $Slides -split ',' | ForEach-Object { [int]$_.Trim() } }
            else { 1..$pres.Slides.Count }
  foreach ($i in $indici) {
    $png = Join-Path $Out ("slide{0:D2}.png" -f $i)
    $pres.Slides.Item($i).Export($png, "PNG", 1600, 900)
    Write-Output $png
  }
} finally {
  $pres.Close()
  $ppt.Quit()
}
