@echo off
setlocal EnableExtensions
:: Resume acbl_all.bat after Stage 1 club ingest and the blocked
:: tournament API downloads. Skips:
::   1a, 1b  club JSON + club SQLite (already updated)
::   1c, 1d  tournament events/sessions (need a fresh ACBL_API_KEY JWT)
:: Starts at 1e (existing session JSON -> SQLite; no API) and runs
:: through 5c so club + current tournament data finish the update.
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
echo  ACBL pipeline resume (skip 1a-1d)
echo  Skipped: club download/SQL (1a-1b), tournament API (1c-1d, JWT)
echo  Running: 1e, 2a, 2b, 3a, 3b, 3c, 4, 5a, 5b, 5c
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

:: ---- 1e ----
:: Existing tournaments/sessions/*.session.json -> .session.sql + sqlite.
:: Does not call the ACBL API.
echo   [1e] Loading tournament sessions into SQLite...
call :pyrun 1e acbl_tournament_sessions_json_to_sql.py
if errorlevel 1 goto :error

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

echo   [5c] Training prediction models...
call :pyrun 5c acbl_prediction_train.py
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
