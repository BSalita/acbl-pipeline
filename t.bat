@echo off
setlocal EnableExtensions
:: Resume after 5c stopped mid tournament Pct_NS (2026-09-15 08:31).
:: Already done this run:
::   1e-4, 5a, 5b
::   5c club DD / Contract / Pct_NS
::   5c tournament DD / Contract
:: Tournament Pct_NS: schema + 16 shards written 08:16-08:25; epochs 1-7
:: ran then the process exited. .pth is still 2026-08-26. Restart that
:: target only, reusing the leftover shards (epochs restart from 1).
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  echo Create it with: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set MPLBACKEND=Agg
set "STEP_OK=%TEMP%\acbl_all_step.ok"
set "PTH_CHECK=e:\bridge\data\acbl\SavedModels\acbl_tournament_predicted_pct_ns_torch_model.pth"
echo ======================================================================
echo  ACBL pipeline resume (5c tournament Pct_NS only)
echo  Skipped: 1a-4, 5a, 5b, club 5c, tournament DD/Contract
echo  Running: 5c  acbl_prediction_train.py --reuse-shards --tournament --target Pct_NS
echo           5d  acbl_prediction_charts.py  (interactive charts; close windows to finish)
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

echo [Stage 5] ML model pipeline...
echo   [5c] Training tournament Pct_NS (reuse leftover shards)...
call :pyrun 5c acbl_prediction_train.py --reuse-shards --tournament --target Pct_NS
if errorlevel 1 goto :error
call :freshpth "%PTH_CHECK%" %PIPE_T0%
if errorlevel 1 goto :error

:: 5d: same as acbl_all.bat. Shows every completed model's charts
:: (club + tournament, all targets that have artifacts), not just Pct_NS.
echo   [5d] Showing prediction charts (close the figure windows to finish)...
set "MPLBACKEND="
call :pyrun 5d acbl_prediction_charts.py
if errorlevel 1 goto :error

echo.
echo ======================================================================
echo  Resume complete: %date% %time%
call :now PIPE_T1
set /a PIPE_ELAPSED=PIPE_T1-PIPE_T0
set /a PIPE_H=PIPE_ELAPSED/3600
set /a PIPE_M=(PIPE_ELAPSED %% 3600)/60
set /a PIPE_S=PIPE_ELAPSED %% 60
echo  TIME[total]: %PIPE_ELAPSED%s (%PIPE_H%h %PIPE_M%m %PIPE_S%s)
echo  Tournament sessions are as of the last successful 1d (JWT still expired).
echo  Re-run acbl_all.bat from 1c after a new ACBL_API_KEY.
echo ======================================================================
del /q "%STEP_OK%" 2>nul
goto :eof

:pyrun
set "STEP_LABEL=%~1"
shift
del /q "%STEP_OK%" 2>nul
call :now STEP_T0
"%PY%" %1 %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
if not exist "%STEP_OK%" exit /b 1
call :toc %STEP_LABEL%
exit /b 0

:now
for /f %%t in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "%~1=%%t"
goto :eof

:freshpth
:: Fail if %1 does not exist or its mtime is older than unix seconds %2.
powershell -NoProfile -Command "$p='%~1'; $t0=[int64]'%~2'; if (-not (Test-Path -LiteralPath $p)) { Write-Host ('*** FAILED: missing model {0}' -f $p); exit 1 }; $u=[DateTimeOffset](Get-Item -LiteralPath $p).LastWriteTime.ToUniversalTime(); if ($u.ToUnixTimeSeconds() -lt $t0) { Write-Host ('*** FAILED: stale model {0}' -f $p); exit 1 }; Write-Host ('Verified fresh model: {0}' -f $p); exit 0"
if errorlevel 1 exit /b 1
goto :eof

:toc
call :now STEP_T1
set /a STEP_ELAPSED=STEP_T1-STEP_T0
set /a STEP_H=STEP_ELAPSED/3600
set /a STEP_M=(STEP_ELAPSED %% 3600)/60
set /a STEP_S=STEP_ELAPSED %% 60
echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
echo.
goto :eof

:error
echo.
echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
del /q "%STEP_OK%" 2>nul
exit /b 1
