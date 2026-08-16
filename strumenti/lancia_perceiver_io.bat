@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul

REM ============================================================================
REM  Lancia in sequenza le run Perceiver IO che mancano.
REM
REM  - Si puo' interrompere con Ctrl+C e rilanciare: riprende da dove era.
REM  - Salta da sola le run che hanno gia' un results.json.
REM  - Se una run fallisce non blocca le altre: passa alla successiva.
REM
REM  Ordine: dalla piu' corta alla piu' lunga, cosi' i primi risultati
REM  arrivano presto anche se poi interrompi.
REM ============================================================================

cd /d "%~dp0\..\progetto"
if errorlevel 1 (
  echo ERRORE: non trovo la cartella progetto\
  pause
  exit /b 1
)

REM Il fine-tuning GLUE parte dal checkpoint del pre-training MLM.
if not exist "logs\io_mlm\checkpoints\best_model.pt" (
  echo.
  echo ERRORE: manca logs\io_mlm\checkpoints\best_model.pt
  echo Le run io_glue_* partono da li'. Lancia prima:  python experiments.py --run io_mlm
  echo.
  pause
  exit /b 1
)

set LISTA=io_glue_rte io_glue_rte_scratch io_glue_mrpc io_glue_stsb io_glue_cola io_glue_sst2 io_glue_sst2_scratch io_glue_qnli io01_cifar io02_cifar_seed1 io_glue_qqp io_glue_mnli io_glue_multitask

set /a TOT=0
set /a FATTE=0
set /a SALTATE=0
set /a FALLITE=0
for %%R in (%LISTA%) do set /a TOT+=1

echo.
echo ================================================================
echo   PERCEIVER IO  -  %TOT% run in coda
echo   Ctrl+C per fermare. Rilanciando, riprende da dove era.
echo ================================================================
echo.

set /a I=0
for %%R in (%LISTA%) do (
  set /a I+=1

  if exist "logs\%%R\results.json" (
    echo [!I!/%TOT%] %%R  -  gia' fatta, salto
    set /a SALTATE+=1
  ) else (
    echo.
    echo ----------------------------------------------------------------
    echo [!I!/%TOT%] %%R  -  avvio alle !TIME:~0,5!
    echo ----------------------------------------------------------------

    python experiments.py --run %%R

    if exist "logs\%%R\results.json" (
      echo   ^> %%R completata
      set /a FATTE+=1
    ) else (
      echo   ^> %%R NON ha prodotto results.json - proseguo con la prossima
      set /a FALLITE+=1
    )
  )
)

echo.
echo ================================================================
echo   RIEPILOGO
echo     completate ora : !FATTE!
echo     gia' fatte     : !SALTATE!
echo     non riuscite   : !FALLITE!
echo ================================================================
echo.
echo Stato completo del registro:
python check.py
echo.
echo Analisi dei risultati:
echo   python analyze_v2.py
echo.
pause
