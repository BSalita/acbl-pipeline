@echo off
setlocal EnableExtensions
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
echo    Cold start (no caches, fresh DD/SD work): ~4 days
echo      (~38 h Stage 3a + ~45 h Stage 5c + other stages).
echo    Warm rerun (3a cache hits): ~55-57 h end-to-end
echo      (~10-12 h Stages 1-5b + ~45 h Stage 5c).
echo    Stage 5c current baseline: ~45 h for all 6 models (2026-08-23 -^> 08-26).
echo    Stage 5d shows the prediction charts 5c no longer opens (close windows to finish).
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Latest complete outputs: 2026-08-14 -^> 2026-08-26 (5c used resumed target runs).
echo  Model training results history: RESULTS.md (append an entry after each 5c run).
echo ======================================================================
echo.
echo Using: %PY%
echo Start: %date% %time%
echo.
call :now PIPE_T0

:: ====================================================================
:: STAGE 1: DATA INGESTION -- download raw data from ACBL
:: ====================================================================
echo [Stage 1] Data ingestion...

:: ---- 1a ----
:: READS:  (web scrape from my.acbl.org)
:: WRITES: acbl/club-results/{club_id}/details/{session_id}.data.json
:: TIME:   incremental; only fetches sessions not already on disk.
::         Cold start (full ACBL backlog): many hours, network-bound.
::         Daily incremental: ~minutes.
echo   [1a] Downloading club results to JSON...
call :pyrun 1a acbl_club_download_to_json.py --sleep 2
if errorlevel 1 goto :error

:: ---- 1b ----
:: READS:  acbl/club-results/*/details/*.data.json
:: WRITES: acbl/club_results_parquet/*.parquet
::         acbl/acbl_club_results.sqlite   (same schema as legacy)
:: TIME:   parallel JSON->Parquet->SQLite (option F). Cold rebuild target
::         ~0.5-1.5 h on 64-core/512GB vs ~4 h legacy .data.sql path.
::         Use: python acbl_club_json_to_sql.py --legacy-sql-scripts
echo   [1b] Loading club JSON into SQLite (via Parquet)...
call :pyrun 1b acbl_club_json_to_sql.py
if errorlevel 1 goto :error

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
echo   [1c] Downloading tournament sanctioned events...
call :pyrun 1c acbl_tournament_download_sanctioned_events.py
if errorlevel 1 goto :error

:: ---- 1d ----
:: READS:  acbl/tournaments/events/*.sanction.json
:: WRITES: acbl/tournaments/sessions/{session_id}.session.json
:: TIME:   incremental; ~minutes daily, several hours cold start (API-bound).
::         90s read timeout covers large NABC full_monty payloads.
::         HTTP 400/404 (unpublished / no boards) do not fail the step.
echo   [1d] Downloading tournament sessions...
call :pyrun 1d acbl_tournament_download_sessions_using_sanctioned_events.py --timeout 90
if errorlevel 1 goto :error

:: ---- 1e ----
:: READS:  acbl/tournaments/sessions/*.session.json
:: WRITES: acbl/tournaments/sessions/*.session.sql
::         acbl/acbl_tournament_results.sqlite
:: TIME:   ~minutes incremental; ~30-60 min on a full rebuild.
echo   [1e] Loading tournament sessions into SQLite...
call :pyrun 1e acbl_tournament_sessions_json_to_sql.py
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 2: CLEANING -- SQLite -> cleaned parquets
:: ====================================================================
echo.
echo [Stage 2] Cleaning...

:: ---- 2a ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: hand_records, sessions)
::         acbl/acbl_tournament_results.sqlite   (tables: handrecord, session)
:: WRITES: acbl/acbl_club_hand_records_cleaned.parquet
::         acbl/acbl_tournament_hand_records_cleaned.parquet
:: TIME:   ~5-10 min total (both club + tournament). SQLite read-bound.
echo   [2a] Cleaning hand records...
call :pyrun 2a acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error

:: ---- 2b ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: events, board_results, boards, ...)
::         acbl/acbl_tournament_results.sqlite
::         acbl/acbl_club_board_results_cleaned.parquet  (for tournament enrichment)
:: WRITES: acbl/acbl_club_board_results_cleaned.parquet
::         acbl/acbl_tournament_board_results_cleaned.parquet
:: TIME:   ~15-30 min total. Wider tables, more joins than 2a.
echo   [2b] Cleaning board results...
call :pyrun 2b acbl_sql_to_board_results_clean.py
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 3: AUGMENTATION -- DD/SD/Par analysis + board result enrichment
:: ====================================================================
echo.
echo [Stage 3] Augmentation...

:: ---- 3a ----
:: READS:  acbl/acbl_{club,tournament}_hand_records_cleaned.parquet
::         acbl/acbl_club_hand_records_cache_df.parquet  (DD+SD cache, shared by club+tournament)
:: WRITES: acbl/acbl_club_hand_records_cache_df.parquet  (updated incrementally every 50K PBNs)
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented_small.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented_narrow.parquet
:: TIME:   COLD START ~38 h for 747K novel PBNs (batched SD pipeline; CPU-bound).
::         WARM (cache hits): ~minutes incremental for daily new PBNs.
::         By far the longest single step in a cold pipeline rebuild.
echo   [3a] Augmenting hand records (DD + SD + Par)...
call :pyrun 3a acbl_hand_records_augment.py
if errorlevel 1 goto :error

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   ~5-15 min total (club is the bigger half).
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   ~30-60 min total. Joins ~6k-col hand-record features into board results.
echo   [3c] Augmenting board results (step 2: join hand records + full augmentation)...
call :pyrun 3c acbl_board_results_augment_step2.py
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
:: TIME:   ~30-60 min total (tournament ~10 min, club ~30-45 min).
::         Walks games chronologically; mostly single-threaded.
echo   [4] Computing Elo ratings (player + pair)...
call :pyrun 4 acbl_elo_ratings_create.py
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 5: ML MODEL DATA
:: ====================================================================
echo.
echo [Stage 5] ML model pipeline...

:: ---- 5a ----
:: READS:  acbl/acbl_{club,tournament}_hand_records_augmented.parquet
::         acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/shards_{club,tournament}_model_data/*.parquet + manifest.json
::         (default --no-merge-shards since 2026-08-16: the single-file merge
::         took ~12 h for club and made downstream reads SLOWER; consumers
::         now scan the shard glob with file-level Date pruning instead)
:: TIME:   tournament ~15 min (16.7M rows x 6786 cols -> 132 monthly shards,
::                   15.3 GB; measured 889 s on 2026-08-16).
::         club      ~60-75 min cold (69.4M rows x 6778 cols -> 96 monthly
::                   shards, 86 GB; ~60 min measured 2026-04 at 59.7M rows).
::         Resume (all shards valid): ~1 min per mode (club measured 54 s
::         on 2026-08-16). Logs: logs/05a_model_data_noshardmerge_*.log.
echo   [5a] Building model data...
call :pyrun 5a acbl_model_data.py
if errorlevel 1 goto :error

:: ---- 5b ----
:: READS:  acbl/acbl_{club,tournament}_model_data_d.pkl
::         acbl/shards_{club,tournament}_model_data/*.parquet (or legacy
::         acbl_{club,tournament}_model_data.parquet if no shard dir)
::         acbl/acbl_{club,tournament}_player_elo_ratings.parquet
::         acbl/acbl_{club,tournament}_pair_elo_ratings.parquet
:: WRITES: acbl/acbl_{club,tournament}_prediction_data_train.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test.parquet
:: TIME:   MEASURED 2026-04-20 -> 2026-04-21 (at 59.5M club rows, reading
::         the single merged model_data file):
::           tournament ~30-60 min (~16M rows, train+test ~42 GB).
::           club       ~7.3 h (55.2M train + 4.3M test rows; 166 GB train,
::                      12 GB test), of which 3 h was one anomalous year
::                      (2022) -- see script docstring "KNOWN ISSUE".
::           Non-anomalous sink rate: ~0.26 ms/row.
::         ESTIMATE at current volume (69.4M club rows, 2026-08-16): club
::         ~5 h (+2.5 h if the 2022 anomaly repeats), tournament ~0.5-1 h,
::         total ~6 h expected / ~8.5 h worst case. Source is now the
::         monthly shard dir (per-year scans prune to ~12 shard files vs
::         scanning the whole 86 GB merged file), which may shave another
::         10-30% off the scan side. Remeasure on next full run.
echo   [5b] Preparing prediction data (train/test split)...
call :pyrun 5b acbl_prediction_data.py
if errorlevel 1 goto :error

:: ---- 5c ----
:: READS:  acbl/acbl_{club,tournament}_prediction_data_train_v2.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test_v2.parquet
:: WRITES: acbl/SavedModels/{model_name}_schema.json                -> Chatbot
::         acbl/SavedModels/{model_name}.pth                        -> Chatbot
::         acbl/SavedModels/*model_shard_*.pt                       (transient)
::         acbl/SavedModels/{model_name}_importance.csv
::         acbl/debug_input_{club,tournament}_{target}.parquet
::         acbl/debug_predictions_{club,tournament}_{target}.parquet
::         Charts are not shown here (MPLBACKEND=Agg). Use 5d.
:: TIME:   CURRENT BASELINE ~45 h for all 6 models, 20 epochs each.
::         Reconstructed from successful target-specific runs on
::         2026-08-23 -> 2026-08-26; a clean uninterrupted run avoids duplicate
::         parquet loading and is expected to take ~44-46 h.
::         Input sizes:
::           club:      61.71M train + 7.74M test rows, ~218.5 + 21.9 GB
::           tournament:15.70M train + 1.03M test rows, ~44.5 + 2.9 GB
::         club (~36-37 h clean-run estimate):
::           Declarer_Direction: shards 4h28m + epochs 9h39m (~1737 s/epoch)
::           Contract:           shards 4h30m + epochs 10h10m (~1830 s/epoch)
::           Pct_NS (pruned):    shards   42m + epochs 1h26m (~259 s/epoch)
::           Measured resumed Contract+Pct_NS process: 22.03 h total.
::         tournament (~8-8.5 h clean-run estimate):
::           Declarer_Direction: shards 1h11m + epochs 2h31m (~454 s/epoch)
::           Contract:           shards   56m + epochs 2h38m (~473 s/epoch)
::           Pct_NS (pruned):    shards   10m + epochs   18m (~53 s/epoch)
::           Measured resumed Contract+Pct_NS process: 4.49 h total.
::         HOST MEMORY: this is the bottleneck, not VRAM. Full-width club data
::         drove Python to ~488 GB working set. The dev box uses 512 GB physical
::         RAM plus a fixed 1.8 TB K: pagefile (~2.27 TB total commit). Do not
::         disable/reduce that pagefile. mlBridgeAiLib.df_to_float32_matrix
::         builds inference/shard matrices in bounded column/row chunks; do not
::         replace it with select(...).to_numpy().astype(float32), which caused
::         45-373 GB transient allocations and commit-limit failures.
::         GPU MEMORY: observed peak was only ~0.38 GB Declarer_Direction,
::         ~0.58 GB Contract, and ~0.43 GB Pct_NS on the RTX 5080.
::         DISK: full-width club models create 62 transient shards (61 x
::         ~22.48 GiB + one ~15.86 GiB = ~1.36 TiB) under SavedModels.
::         E:\...\SavedModels is a junction to F:\...\SavedModels; ensure at
::         least 1.5 TiB free on F: before 5c. Shards are deleted before the
::         next target, so only one target's shard set should exist at a time.
::         Run this step alone (not via acbl_all.bat) when iterating on models.
echo   [5c] Training prediction models...
set MPLBACKEND=Agg
call :pyrun 5c acbl_prediction_train.py
if errorlevel 1 goto :error

:: ---- 5d ----
:: READS:  acbl/debug_predictions_{club,tournament}_{target}.parquet
::         (fallback: acbl/debug_predictions_{target}.parquet)
::         acbl/SavedModels/acbl_{club,tournament}_predicted_{target}_torch_model_importance.csv
:: WRITES: (none; interactive matplotlib windows)
:: TIME:   user-bound; close the figure windows to continue.
::         Charts were removed from 5c (Agg backend) after a GUI close
::         on a leftover figure killed tournament Pct_NS (2026-09-15).
echo   [5d] Showing prediction charts (close the figure windows to finish)...
set "MPLBACKEND="
call :pyrun 5d acbl_prediction_charts.py
if errorlevel 1 goto :error

:: ====================================================================
:: DONE
:: ====================================================================
echo.
echo ======================================================================
echo  Pipeline complete: %date% %time%
call :now PIPE_T1
set /a PIPE_ELAPSED=PIPE_T1-PIPE_T0
set /a PIPE_H=PIPE_ELAPSED/3600
set /a PIPE_M=(PIPE_ELAPSED %% 3600)/60
set /a PIPE_S=PIPE_ELAPSED %% 60
echo  TIME[total]: %PIPE_ELAPSED%s (%PIPE_H%h %PIPE_M%m %PIPE_S%s)
echo.
echo  Downstream consumers and their required files:
echo.
echo  ..\elo:
echo    acbl_{club,tournament}_elo_ratings.parquet         (full Elo history)
echo    acbl_{club,tournament}_player_elo_ratings.parquet  (player lookup)
echo    acbl_{club,tournament}_pair_elo_ratings.parquet    (pair lookup)
echo.
echo  ..\postmortem-acbl:
echo    acbl_{club,tournament}_player_elo_ratings.parquet  (player lookup)
echo    acbl_{club,tournament}_pair_elo_ratings.parquet    (pair lookup)
echo    SavedModels\*_schema.json                         (model schemas)
echo    SavedModels\*.pth                                 (trained weights)
echo ======================================================================
del /q "%STEP_OK%" 2>nul
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
"%PY%" %1 %2 %3 %4 %5 %6 %7 %8 %9 && echo.>"%STEP_OK%"
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
echo.
goto :eof

:error
echo.
echo *** FAILED/INTERRUPTED at step %STEP_LABEL% %date% %time% ***
del /q "%STEP_OK%" 2>nul
exit /b 1
