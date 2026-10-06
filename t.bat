@echo off
setlocal EnableExtensions
:: Rebuild model-data shards after the in-progress 5b finishes.
:: Do not start this while that 5b is still reading the shard files.
:: 1a-4 and the first 5a/5b are already done. 5a skipped months whose
:: source row count had not changed, so Par_Contract_NS, Par_Contract_EW,
:: and Is_Sacrifice_Opportunity are only on the newest shards.
:: Deleting the shard dirs forces a cold 5a. Last measured cold rebuild:
:: club ~10.5 h (2026-09-12), tournament 889s (2026-08-16).
:: 5b is then rerun from the new shards. 5c and 5d stay out.
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
echo ======================================================================
echo  ACBL pipeline resume: rebuild model shards, then 5b
echo  Skipped: 1a through 4, and the 5b that is already running
echo  Running: delete shards_club_model_data and shards_tournament_model_data
echo           5a acbl_model_data.py
echo           5b acbl_prediction_data.py
echo  Not in this bat: 5c train, 5d charts
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

echo.
echo [5a] Removing model-data shards so every month is rebuilt...
set "STEP_LABEL=5a-shards"
set "SHARD_ROOT=e:\bridge\data\acbl"
for %%D in (shards_club_model_data shards_tournament_model_data) do (
  if exist "%SHARD_ROOT%\%%D" (
    echo   rmdir %SHARD_ROOT%\%%D
    rmdir /s /q "%SHARD_ROOT%\%%D"
    if exist "%SHARD_ROOT%\%%D" goto :error
  ) else (
    echo   already absent: %SHARD_ROOT%\%%D
  )
)

echo.
echo [5a] Building model data...
call :pyrun 5a acbl_model_data.py
if errorlevel 1 goto :error

echo.
echo [5b] Preparing prediction data from the rebuilt shards...
call :pyrun 5b acbl_prediction_data.py
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
echo  Model shards now include the par-contract and sacrifice columns.
echo  Train and test files still omit them: 5b reads game states 0-4.
echo  5c train and 5d charts are not in this bat.
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
