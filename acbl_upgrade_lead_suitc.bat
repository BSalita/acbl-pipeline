@echo off
setlocal EnableExtensions
:: One-time ACBL upgrade: add the opening-lead and SuitC columns to the
:: existing ACBL data (no new downloads, no model retraining).
:: Times are from the run of 2026-11-15 17:02 to 2026-11-16 09:48 (16h 46m).
::   1. fill the shared SuitC cache (E:\bridge\data\suitc\combos)    12 s
::      (already filled: 783,075 holdings; from empty ~1-2 h, estimate)
::   2. 3a hand records with SuitC (SC_* columns)                    1h 47m
::   3. 3c board results with the SD lead score (20 samples). Adds
::      DD_Lead_Cost, Lead_Is_DD_Best, SD_Lead_*, DD_Tricks_Diff_SC.
::      Club only for the lead columns: tournaments record no lead.
::      Club 14h 2m, of it 12h 44m solving 5.51M lead positions at
::      120 positions/s; tournament 20m                              14h 22m
::   4. Elo ratings (keeps the lead and SuitC columns)                7 min
::   5. publish: acbl-pipeline\u.bat (board results to X1 acbl-stage),
::      elo\u.bat (Elo parquets), bridgestats-acbl\u.bat (narrow hand records)
::   6. redeploy Elo, postmortem and BridgeStats-ACBL on X1
::      Steps 5 and 6 took 29 min together with only the board results
::      (93 GB) published; the Elo and BridgeStats files, published by
::      hand afterwards, took 3 min, and the three redeploys 3 min.
:: Run it after ffbridge_upgrade_lead_suitc.bat has finished: both are CPU
:: bound, and elo\u.bat also publishes the FFBridge archive.
:: Each step stops the run on failure; rerun the script as is: step 1 solves
:: only uncached holdings, 3a skips a side that is already done, 3c
:: recomputes only the blocks not yet written and keeps solved lead
:: positions in acbl\acbl_club_sd_lead_cache\.
:: Stage 5 (model data, training) is not run: the models do not use these
:: columns.
:: NOTE 2026-11-15: step 4 now also applies the pair-rating fix
::      (compute_event_start_end_elo_columns; see step 4's NOTE in
::      acbl_all.bat), so steps 4-6 put corrected pair ratings on prod while
::      the models still carry the old ones. Set ACBL_UPGRADE_STOP_AFTER_3C=1
::      to stop after step 3 and decide later; rerunning without it resumes
::      (steps 1-3 skip or resume) and goes on with 4-6.
set "PIPE_DIR=%~dp0"
:: Not named SRC: acbl-pipeline\u.bat sets SRC to the data folder, and on
:: 2026-11-16 that sent the two pushd calls of step 5 to folders that do not
:: exist, so u.bat of this folder ran three times and the Elo and BridgeStats
:: files were not published. A pushd that fails now stops the run.
set "SRC_ROOT=%PIPE_DIR%.."
set "PY=%PIPE_DIR%.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
:: DD solver: the same library as acbl_all.bat (see the notes there).
if not defined DDSS_DLL_PATH for %%I in ("%~dp0..\bridge-solver-cpp\build\dds-single.dll") do set "DDSS_DLL_PATH=%%~fI"
if not exist "%DDSS_DLL_PATH%" (
  echo *** FAILED: DD solver library not found: %DDSS_DLL_PATH%
  echo Build it with build.sh in ..\bridge-solver-cpp, or set DDSS_DLL_PATH.
  exit /b 1
)
:: Success marker: Ctrl+C then "Terminate batch job? N" clears ERRORLEVEL
:: to 0, so only a file created after a clean exit counts (as in acbl_all.bat).
set "STEP_OK=%TEMP%\acbl_upgrade_step.ok"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_STAMP=%%t"
set "LOG_DIR=%PIPE_DIR%logs\acbl_upgrade_%RUN_STAMP%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo [upgrade] start %DATE% %TIME%
echo [upgrade] logs %LOG_DIR%
echo [upgrade] DD solver %DDSS_DLL_PATH%
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

if "%ACBL_UPGRADE_STOP_AFTER_3C%"=="1" goto :held

echo [upgrade] 4. Elo ratings... %TIME%
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

echo [upgrade] 5. publishing to src\data and X1... %TIME%
set "STEP_LABEL=5"
call u.bat
if errorlevel 1 goto :error
pushd "%SRC_ROOT%\elo" || goto :error
call u.bat
set "STEP_EXIT=%ERRORLEVEL%"
popd
if not "%STEP_EXIT%"=="0" goto :error
pushd "%SRC_ROOT%\bridgestats-acbl" || goto :error
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
echo acbl_all.bat keeps these columns: ACBL_SUITC=1 and ACBL_SD_LEAD_SAMPLES=20 are its
echo defaults. Setting either to 0 recomputes 3a and 3c and drops the new columns.
exit /b 0

:held
popd
del /q "%STEP_OK%" 2>nul
echo [upgrade] steps 1-3 done %DATE% %TIME%; stopped before step 4 (ACBL_UPGRADE_STOP_AFTER_3C=1).
echo Not done: 4 Elo ratings, 5 publish, 6 redeploy. Rerun without the variable to finish.
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
