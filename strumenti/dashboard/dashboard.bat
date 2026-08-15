@echo off
REM Avvia il pannello esperimenti fuori dall'IDE (doppio clic su questo file).
REM La finestra console resta aperta: mostra eventuali errori.
title Perceiver dashboard
python "%~dp0app.py"
echo.
echo (dashboard chiusa) - premi un tasto per uscire
pause >nul
