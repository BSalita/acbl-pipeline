@echo off
setlocal EnableExtensions
:: Resume after hardware reboot during 5c (OOM).
:: Already done this run:
::   5a  model data (2026-09-12 22:35)
::   5b  prediction parquets (club 00:45, tournament 01:06 on 2026-09-13)
:: 5c club Declarer_Direction: schema + 62 shards (~1.4 TB) written this
:: morning; epochs 1-2 ran then the box died. No new .pth (still 2026-08-24).
:: Restart 5c from club Declarer_Direction using leftover shards, then
:: Contract, Pct_NS, and all three tournament targets.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  echo Create it with: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "STEP_OK=%TEMP%\acbl_all_step.ok"
echo ======================================================================
echo  ACBL pipeline resume (5c only, reuse leftover club DD shards)
echo  Skipped: 1a-4, 5a, 5b
echo  Running: 5c  acbl_prediction_train.py --reuse-shards
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

echo [Stage 5] ML model pipeline...
echo   [5c] Training prediction models (reuse leftover shards)...
call :pyrun 5c acbl_prediction_train.py --reuse-shards
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
