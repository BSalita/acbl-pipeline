@echo off
setlocal EnableExtensions
:: Resume after the 2026-10-01 club download.
:: Already done and not repeated here:
::   1a  club JSON (1944 clubs, 7689 sessions, Failed: 0)
::   1b  acbl_club_results.sqlite (139.34 GB, err=0, finished 2026-10-02 06:40)
:: That console printed the 1c banner and then Stage 3 four seconds later.
:: 1c-2b never ran, so 3a-4 rewrote the previous cleaned parquets
:: (club Date max 2026-09-09) and 5a skipped every monthly shard.
:: 5b was reading shards whose Date max was 2026-08-12.
:: 1c-1e finished on 2026-10-04. 2a then stopped because Windows
:: blocked the ADBC SQLite driver; cleaning now reads through DuckDB.
:: This resume starts at 2a and rebuilds through prediction data.
:: 5c and 5d stay out until those prediction parquets exist.
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
echo  ACBL pipeline resume (2a through 5b)
echo  Skipped: 1a, 1b, 1c, 1d, 1e
echo  Running: 2a 2b  3a 3b 3c  4  5a 5b
echo  Not in this bat: 5c train, 5d charts
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

echo.
echo [Stage 2] Cleaning...
echo   [2a] Cleaning hand records...
call :pyrun 2a acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error

echo   [2b] Cleaning board results...
call :pyrun 2b acbl_sql_to_board_results_clean.py
if errorlevel 1 goto :error

echo.
echo [Stage 3] Augmentation...
echo   [3a] Augmenting hand records (DD + SD + Par)...
call :pyrun 3a acbl_hand_records_augment.py
if errorlevel 1 goto :error

echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

echo   [3c] Augmenting board results (step 2: join hand records + full augmentation)...
call :pyrun 3c acbl_board_results_augment_step2.py
if errorlevel 1 goto :error

echo.
echo [Stage 4] Elo ratings...
echo   [4] Computing Elo ratings (player + pair)...
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

echo.
echo [Stage 5] ML model pipeline...
echo   [5a] Building model data...
call :pyrun 5a acbl_model_data.py
if errorlevel 1 goto :error

echo   [5b] Preparing prediction data (train/test split)...
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
echo  Prediction parquets now include the post-1b clean. Run 5c after this.
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
