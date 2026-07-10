param(
  [switch]$CreateZip
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageName = "perceiver_condivisibile"
$Package = Join-Path $Root $PackageName
$ZipPath = Join-Path $Root "$PackageName.zip"

$RootResolved = (Resolve-Path -LiteralPath $Root).Path
if (Test-Path -LiteralPath $Package) {
  $PackageResolved = (Resolve-Path -LiteralPath $Package).Path
  if (-not $PackageResolved.StartsWith($RootResolved, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to update package outside workspace: $PackageResolved"
  }
  Remove-Item -LiteralPath $PackageResolved -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $Package | Out-Null

$RequiredFiles = @(
  "perceiver_interattivo\index.html",
  "perceiver_interattivo\css\style.css",
  "perceiver_interattivo\css\interactive-labs.css",
  "perceiver_interattivo\js\app.js",
  "perceiver_interattivo\js\interactive-labs.js",
  "appunti_ml_definitivo.pdf",
  "appunti_ml_definitivo.tex",
  "perceiver_assets\bottleneck.svg",
  "perceiver_assets\byte_array.svg",
  "perceiver_assets\latent_array.svg",
  "perceiver_assets\attention_pipeline.svg",
  "perceiver_assets\output_head.svg",
  "perceiver_assets\backward_flow.svg",
  "perceiver_assets\cnn_pipeline_hierarchy.svg",
  "perceiver_assets\cnn_pooling_pipeline.svg",
  "perceiver_assets\cnn_resnet_bridge.svg",
  "perceiver_assets\transformer_family_map.svg",
  "perceiver_assets\transformer_block_split.svg",
  "perceiver_assets\transformer_masked_attention.svg",
  "appunti_images\media\image89.jpeg",
  "appunti_images\media\image2.png",
  "appunti_images\media\image5.png",
  "appunti_images\media\image7.png",
  "appunti_images\media\image9.png",
  "appunti_images\media\image11.png",
  "appunti_images\media\image12.png",
  "appunti_images\media\image17.png",
  "appunti_images\media\image43.png",
  "appunti_images\media\image75.png",
  "appunti_images\media\image18.png",
  "appunti_images\media\image22.png",
  "appunti_images\media\image23.jpg",
  "appunti_images\media\image24.png",
  "appunti_images\media\image25.png",
  "appunti_images\media\image26.png",
  "appunti_images\media\image27.jpeg",
  "appunti_images\media\image29.png",
  "appunti_images\media\image30.png",
  "appunti_images\media\image31.png",
  "appunti_images\media\image33.png",
  "appunti_images\papers\fig_positional_encoding.png",
  "appunti_images\papers\perceiver_orig_fig1_full.png",
  "appunti_images\papers\fig_softmax.png",
  "appunti_images\papers\fig_cross_entropy.png",
  "appunti_images\papers\fig_normalization_methods.png",
  "appunti_images\papers\fig_relu_gelu.png",
  "appunti_images\papers\fig_residual_block.png",
  "appunti_images\papers\fig_gradient_descent.png",
  "appunti_images\papers\fig_optimizer_contour.png",
  "appunti_images\papers\fig_adam_convergence.png",
  "appunti_images\papers\fig_lr_cosine_warmup.png",
  "interactive_trainer\img\cnn_architecture_clean.jpg",
  "interactive_trainer\img\cnn_feature_hierarchy_clean.jpeg",
  "interactive_trainer\img\docx_resnet50_arch.png",
  "interactive_trainer\img\docx_transformer_full_zoom.jpg",
  "interactive_trainer\img\docx_vit_architecture.png",
  "interactive_trainer\img\fig1_architecture.png",
  "interactive_trainer\img\fig3_attention_maps.png",
  "interactive_trainer\img\perceiver_io_fig2_architecture.png",
  "interactive_trainer\img\perceiver_io_fig3_output_queries.png",
  "lezioni\paper_figures\table7_weight_sharing.png"
)

foreach ($RelativePath in $RequiredFiles) {
  $Source = Join-Path $Root $RelativePath
  if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
    throw "Missing required file: $RelativePath"
  }
  $Destination = Join-Path $Package $RelativePath
  $DestinationDir = Split-Path -Parent $Destination
  New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
  Copy-Item -LiteralPath $Source -Destination $Destination -Force
}

$Readme = @"
Perceiver - pacchetto condivisibile
===================================

Apri questo file nel browser:

  perceiver_interattivo/index.html

La cartella contiene solo i file necessari alla lezione interattiva:

- HTML principale
- immagini usate nella lezione
- appunti_ml_definitivo.pdf
- appunti_ml_definitivo.tex

Nota: le formule usano MathJax da CDN. Con internet attivo vengono renderizzate in modo pulito.
Senza internet resta comunque visibile il contenuto fallback delle formule, ma meno elegante.

I link nel pannello "Fonte negli appunti" aprono il PDF con #page=N. Se il browser non rispetta
il numero pagina, apri manualmente appunti_ml_definitivo.pdf e vai al range indicato.
"@

Set-Content -LiteralPath (Join-Path $Package "LEGGIMI.txt") -Value $Readme -Encoding UTF8

$CopiedIndex = Join-Path $Package "perceiver_interattivo\index.html"
$Html = Get-Content -LiteralPath $CopiedIndex -Raw
$Matches = [regex]::Matches($Html, '(?:src|href)="\.\./([^"#]+)(?:#[^"]*)?"')
$Missing = New-Object System.Collections.Generic.List[string]
foreach ($Match in $Matches) {
  $Relative = $Match.Groups[1].Value.Replace("/", [System.IO.Path]::DirectorySeparatorChar)
  $Resolved = Join-Path $Package $Relative
  if (-not (Test-Path -LiteralPath $Resolved -PathType Leaf)) {
    $Missing.Add($Match.Groups[1].Value)
  }
}
if ($Missing.Count -gt 0) {
  throw "Package validation failed. Missing referenced files: $($Missing -join ', ')"
}

if ($CreateZip) {
  if (Test-Path -LiteralPath $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
  }
  Compress-Archive -LiteralPath $Package -DestinationPath $ZipPath -Force
}

Write-Host "Package ready: $Package"
if ($CreateZip) {
  Write-Host "Zip ready: $ZipPath"
}
Write-Host "Referenced local files validated: $($Matches.Count)"
