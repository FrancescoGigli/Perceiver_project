# Esporta il .pptx in PDF via PowerPoint (COM). Serve al copione per l'orale:
# includendo le pagine del PDF le slide restano vettoriali, invece di 41 PNG.
param(
  [string]$Deck = "$PSScriptRoot\..\Perceiver & Perceiver IO.pptx",
  [string]$Out  = "$PSScriptRoot\..\_copione\deck.pdf"
)

$Deck = (Resolve-Path $Deck).Path
$lock = Join-Path (Split-Path $Deck) ("~$" + (Split-Path $Deck -Leaf))
if (Test-Path $lock) { Write-Error "Il deck e' aperto in PowerPoint: chiudilo."; exit 1 }

$dir = Split-Path $Out
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
$Out = Join-Path (Resolve-Path $dir).Path (Split-Path $Out -Leaf)

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Open($Deck, $true, $false, $false)
try {
  $pres.SaveAs($Out, 32)          # 32 = ppSaveAsPDF
  Write-Output $Out
} finally {
  $pres.Close()
  $ppt.Quit()
}
