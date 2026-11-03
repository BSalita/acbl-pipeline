@echo off
setlocal EnableExtensions
:: One-time ACBL upgrade: add the opening-lead and SuitC columns to the
:: existing ACBL data (no new downloads, no model retraining).
::   1. fill the shared SuitC cache (E:\bridge\data\suitc\combos)   minutes if
::      the FFBridge upgrade already filled it, else ~1-2 h
::   2. 3a hand records with SuitC (SC_* columns)                    ~2-3 h
::   3. 3c board results with the SD lead score (20 samples). Adds
::      DD_Lead_Cost, Lead_Is_DD_Best, SD_Lead_*, DD_Tricks_Diff_SC.
::      Club only for the lead columns: tournaments record no lead.
::      The first run solves ~5.7M lead positions (~33 h) on top of
::      the ~16.5 h rewrite                                          ~50 h
::   4. Elo ratings (keeps the lead and SuitC columns)                ~40 min
::   5. publish: acbl-pipeline\u.bat (board results to X1 acbl-stage),
::      elo\u.bat (Elo parquets), bridgestats-acbl\u.bat (narrow hand records)
::   6. redeploy Elo, postmortem and BridgeStats-ACBL on X1
:: Run it after ffbridge_upgrade_lead_suitc.bat has finished: both are CPU
:: bound, and elo\u.bat also publishes the FFBridge archive.
:: Each step stops the run on failure; rerun the script as is: step 1 solves
:: only uncached holdings, 3a skips a side that is already done, 3c resumes
:: club years already written (tournament is redone, ~3.3 h) and keeps
:: solved lead positions in acbl\acbl_club_sd_lead_cache\.
:: Stage 5 (model data, training) is not run: the models do not use these
:: columns.
set "PIPE_DIR=%~dp0"
set "SRC=%PIPE_DIR%.."
set "PY=%PIPE_DIR%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
:: Success marker: Ctrl+C then "Terminate batch job? N" clears ERRORLEVEL
:: to 0, so only a file created after a clean exit counts (as in acbl_all.bat).
set "STEP_OK=%TEMP%\acbl_upgrade_step.ok"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_STAMP=%%t"
set "LOG_DIR=%PIPE_DIR%logs\acbl_upgrade_%RUN_STAMP%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo [upgrade] start %DATE% %TIME%
echo [upgrade] logs %LOG_DIR%
pushd "%PIPE_DIR%"

echo [upgrade] 1. filling the SuitC cache... %TIME%
call :pyrun 1_suitc suitc_backfill.py --workers 64
if errorlevel 1 goto :error

echo [upgrade] 2. 3a hand records with SuitC... %TIME%
call :pyrun 3a acbl_hand_records_augment.py --suitc
if errorlevel 1 goto :error

echo [upgrade] 3. 3c board results with the SD lead score (20)... %TIME%
call :pyrun 3c acbl_board_results_augment_step2.py --sd-lead-samples 20
if errorlevel 1 goto :error

echo [upgrade] 4. Elo ratings... %TIME%
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

echo [upgrade] 5. publishing to src\data and X1... %TIME%
set "STEP_LABEL=5"
call u.bat
if errorlevel 1 goto :error
pushd "%SRC%\elo"
call u.bat
set "STEP_EXIT=%ERRORLEVEL%"
popd
if not "%STEP_EXIT%"=="0" goto :error
pushd "%SRC%\bridgestats-acbl"
call u.bat
set "STEP_EXIT=%ERRORLEVEL%"
popd
if not "%STEP_EXIT%"=="0" goto :error

echo [upgrade] 6. redeploying Elo, postmortem and BridgeStats-ACBL on X1... %TIME%
:: run_in_desktop runs each deploy in X1's desktop session; wslc build fails
:: from a plain SSH session. The containers pull code from GitHub master.
set "STEP_LABEL=6"
for %%S in (deploy_elo_ratings.ps1 deploy_postmortem.ps1 deploy_bridgestats_acbl.ps1) do (
    echo [upgrade] 6. %%S %TIME%
    ssh -o BatchMode=yes -o ConnectTimeout=10 X1-PRO-470-1TB "powershell -NoProfile -ExecutionPolicy Bypass -File C:\sw\bridge\ML-Contract-Bridge\src\ops\run_in_desktop.ps1 -Script %%S"
    if errorlevel 1 goto :error
)

popd
del /q "%STEP_OK%" 2>nul
echo [upgrade] done %DATE% %TIME%
echo Later runs of acbl_all.bat must set ACBL_SUITC=1 and ACBL_SD_LEAD_SAMPLES=20:
echo without them 3a and 3c recompute everything and drop the new columns.
exit /b 0

:: usage: call :pyrun LABEL script.py [args...]  (output also goes to LOG_DIR)
:pyrun
set "STEP_LABEL=%~1"
shift
del /q "%STEP_OK%" 2>nul
"%PY%" "%PIPE_DIR%acbl_step_log.py" "%LOG_DIR%\%STEP_LABEL%.log" %1 %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
if not exist "%STEP_OK%" exit /b 1
echo [upgrade] %STEP_LABEL% ended %DATE% %TIME%
exit /b 0

:error
popd
del /q "%STEP_OK%" 2>nul
echo *** [upgrade] failed at step %STEP_LABEL% %DATE% %TIME%; fix the cause and rerun (finished steps resume or skip).
echo  Step log: %LOG_DIR%\%STEP_LABEL%.log
exit /b 1
