@echo off
setlocal EnableExtensions
:: This script's folder. Inside a CALLed label %~dp0 is not the script's folder.
set "PIPE_DIR=%~dp0"
:: Use project venv (has requests, polars, torch, ...). Bare `python` on PATH
:: is often the system install and will fail with ModuleNotFoundError.
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo *** FAILED: project venv not found: %PY%
  echo Create it with: python -m venv .venv ^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
:: Success marker for :pyrun. Ctrl+C then "Terminate batch job? N" clears
:: ERRORLEVEL to 0, so `if errorlevel 1` alone lets the next step run.
:: Only a successful python exit creates this file (via &&); missing file = abort.
set "STEP_OK=%TEMP%\acbl_all_step.ok"
echo ======================================================================
echo  ACBL Full Pipeline
echo  Produces data files consumed by:
echo    ..\elo          (player/pair Elo parquets)
echo    ..\postmortem-acbl (SavedModels, Elo parquets)
echo.
echo  Approximate end-to-end wall time on the dev box
echo  (512 GB RAM, 64-core CPU, NVMe E:/F:, RTX 5080, 1.8 TB K: pagefile):
echo    Cold start (no caches, fresh DD/SD work): ~4.5-5 days
echo      (~38 h Stage 3a + ~18 h Stage 3c + ~45 h Stage 5c + other stages).
echo    Warm rewrite measured 2026-10-01 -^> 10-03 (log captured mid-5b):
echo      1a 10h01m + 1b 7h56m + 3a 2h10m (DD/SD cache hit) + 3b 2m
echo      + 3c 16h27m + 4 40m + 5a 4m (every shard skipped). 5b+ unfinished.
echo    Stage 5c current baseline: 45h 31m for all 6 models (2026-10-06 15:50 -^> 10-08 13:21).
echo      club 39h 19m, tournament 6h 12m. Clean run; no OOM resume.
echo    Stage 5d shows the prediction charts 5c no longer opens (close windows to finish).
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Latest measured stages: 1a-4 from 2026-10-01 -^> 10-03. Stage 5 from
echo    the 2026-10-06 -^> 10-08 rebuild (5a shard writes, 5b, 5c).
echo  Model training results history: RESULTS.md (append an entry after each 5c run).
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
:: Each step's output is also saved to logs\acbl_all_<start time>\<step>.log
:: (via acbl_step_log.py); summary.log collects the TIME[...] lines.
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "RUN_STAMP=%%t"
set "LOG_DIR=%PIPE_DIR%logs\acbl_all_%RUN_STAMP%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
echo Logs:  %LOG_DIR%
>"%LOG_DIR%\summary.log" echo Start: %date% %time%
echo.
call :now PIPE_T0

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   MEASURED 2026-10-02 08:52 -^> 10-03 01:19: 59213s (16h 27m).
::         Club 47221s, 70,827,715 rows, 82.4 GB. Tournament 11970s,
::         18,667,922 rows, 19.6 GB. Club Date max 2026-09-09.
::         Joins ~6k-col hand-record features into board results.
:: OPTION: set ACBL_SD_LEAD_SAMPLES=20 to add the single-dummy lead score
::         (SD_Lead_* columns, club only: tournaments record no lead). The first
::         club run solves ~5.7M lead positions, ~33 h at 20 (48 positions/s);
::         acbl/acbl_club_sd_lead_cache/ keeps them, so later runs solve only new
::         ones. Changing the value recomputes every year shard.
if not defined ACBL_SD_LEAD_SAMPLES set "ACBL_SD_LEAD_SAMPLES=0"
echo   [3c] Augmenting board results (step 2: join hand records + full augmentation; SD lead samples %ACBL_SD_LEAD_SAMPLES%)...
call :pyrun 3c acbl_board_results_augment_step2.py --sd-lead-samples %ACBL_SD_LEAD_SAMPLES%
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 4: ELO RATINGS
:: ====================================================================
echo.
echo [Stage 4] Elo ratings...

:: ---- 4 ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_elo_ratings.parquet
::         acbl/acbl_{club,tournament}_player_elo_ratings.parquet  -> elo, postmortem-acbl
::         acbl/acbl_{club,tournament}_pair_elo_ratings.parquet    -> elo, postmortem-acbl
:: TIME:   MEASURED 2026-10-03 01:19-01:59: 2418s (40m). Club 1937s
::         (70.83M rows), tournament 475s (18.67M rows). Walks games
::         chronologically; mostly single-threaded.
echo   [4] Computing Elo ratings (player + pair)...
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

goto :eof

:: --------------------------------------------------------------------
:: Run one python step. Returns exit /b 1 on failure OR Ctrl+C.
:: Callers MUST check:  if errorlevel 1 goto :error
::
:: Two Windows quirks this defends against:
::  1) Ctrl+C then "Terminate batch job? N" clears ERRORLEVEL to 0.
::     STEP_OK is only created via `&&` after a successful python exit,
::     so a missing marker still fails the step.
::  2) `goto :error` + `exit /b` from inside a CALLed subroutine only
::     returns to the caller -- the pipeline would continue. So :pyrun
::     returns /b 1 and the top-level caller jumps to :error.
::
:: usage: call :pyrun LABEL script.py [args...]
:: --------------------------------------------------------------------
:pyrun
set "STEP_LABEL=%~1"
shift
del /q "%STEP_OK%" 2>nul
call :now STEP_T0
"%PY%" "%PIPE_DIR%acbl_step_log.py" "%LOG_DIR%\%STEP_LABEL%.log" %1 %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
if not exist "%STEP_OK%" exit /b 1
call :toc %STEP_LABEL%
exit /b 0

:: --------------------------------------------------------------------
:: Timing helpers. Epoch seconds via PowerShell so steps longer than
:: 24 h (e.g. 5c) are measured correctly; %time% arithmetic wraps at
:: midnight and cannot represent multi-day steps.
:: --------------------------------------------------------------------

:: usage: call :now VARNAME  -- store current unix epoch seconds in VARNAME
:now
for /f %%t in ('powershell -NoProfile -Command "[DateTimeOffset]::Now.ToUnixTimeSeconds()"') do set "%~1=%%t"
goto :eof

:: usage: call :toc LABEL  -- print elapsed time since STEP_T0
:toc
call :now STEP_T1
set /a STEP_ELAPSED=STEP_T1-STEP_T0
set /a STEP_H=STEP_ELAPSED/3600
set /a STEP_M=(STEP_ELAPSED %% 3600)/60
set /a STEP_S=STEP_ELAPSED %% 60
echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
>>"%LOG_DIR%\%~1.log" echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
>>"%LOG_DIR%\summary.log" echo   TIME[%~1]: %STEP_ELAPSED%s (%STEP_H%h %STEP_M%m %STEP_S%s) ended %date% %time%
echo.
goto :eof

:error
echo.
echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
>>"%LOG_DIR%\summary.log" echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
echo  Step log: %LOG_DIR%\%STEP_LABEL%.log
del /q "%STEP_OK%" 2>nul
exit /b 1
