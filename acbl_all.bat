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
echo.
call :now PIPE_T0

:: ====================================================================
:: STAGE 1: DATA INGESTION -- download raw data from ACBL
:: ====================================================================
echo [Stage 1] Data ingestion...

:: ---- 1a ----
:: READS:  (web scrape from my.acbl.org)
:: WRITES: acbl/club-results/{club_id}/details/{session_id}.data.json
:: TIME:   MEASURED 2026-10-01: 36066s (10h 1m), ended 22:44:56.
::         1944 clubs, 7689 sessions extracted; last club 208 processed,
::         2 already on disk, Failed: 0. Incremental, network-bound.
::         A small daily catch-up is minutes; this pass was a full sweep.
echo   [1a] Downloading club results to JSON...
call :pyrun 1a acbl_club_download_to_json.py --sleep 2
if errorlevel 1 goto :error

:: ---- 1b ----
:: READS:  acbl/club-results/*/details/*.data.json
:: WRITES: acbl/club_results_parquet/*.parquet
::         acbl/acbl_club_results.sqlite   (same schema as legacy)
:: TIME:   MEASURED 2026-10-01 22:45 -^> 10-02 06:40: 28533s (7h 56m).
::         1,335,293 JSON files, err=0. Parquet merge 16m20s, sqlite
::         preflight 6m44s, table load ~7h20m. Wrote acbl_club_results.sqlite
::         139.34 GB. xmlFiles.parquet skipped (no schema table).
::         Use: python acbl_club_json_to_sql.py --legacy-sql-scripts
echo   [1b] Loading club JSON into SQLite (via Parquet)...
call :pyrun 1b acbl_club_json_to_sql.py
if errorlevel 1 goto :error

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
::         A rejected or expired ACBL_API_KEY falls back to live.acbl.org.
::         2026-09 cycle skipped (expired JWT). Newest sanction JSON 2026-08-14.
::         2026-10-02 console printed this banner, then Stage 3 started 4s
::         after 1b (06:40:29 -^> 06:40:33). 1c-2b were not remeasured;
::         cleaned parquets were not rebuilt from the new sqlite.
echo   [1c] Downloading tournament sanctioned events...
call :pyrun 1c acbl_tournament_download_sanctioned_events.py
if errorlevel 1 goto :error

:: ---- 1d ----
:: READS:  acbl/tournaments/events/*.sanction.json
:: WRITES: acbl/tournaments/sessions/{session_id}.session.json
:: TIME:   incremental; ~minutes daily, several hours cold start (API-bound).
::         90s read timeout covers large NABC full_monty payloads.
::         HTTP 400/404 (unpublished / no boards) do not fail the step.
::         A rejected or expired ACBL_API_KEY falls back to live.acbl.org
::         (same Chrome/Cloudflare download path as club results).
::         2026-09 cycle skipped (expired JWT). Newest session JSON 2026-09-02.
echo   [1d] Downloading tournament sessions...
call :pyrun 1d acbl_tournament_download_sessions_using_sanctioned_events.py --timeout 90
if errorlevel 1 goto :error

:: ---- 1e ----
:: READS:  acbl/tournaments/sessions/*.session.json
:: WRITES: acbl/tournaments/sessions/*.session.sql
::         acbl/acbl_tournament_results.sqlite
:: TIME:   ~minutes incremental; was ~30-60 min on a smaller full rebuild.
::         MEASURED 2026-09-10/11: finished 00:16, sqlite 15.0 GB. If this
::         started after 1b (18:00), the rebuild was ~6 h.
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
:: TIME:   MEASURED 2026-09-11 00:16-03:54 (~3.6 h). Club 0.89 GB @ 03:37,
::         tournament 0.08 GB @ 03:54. Was ~5-10 min at smaller volume.
echo   [2a] Cleaning hand records...
call :pyrun 2a acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error

:: ---- 2b ----
:: READS:  acbl/acbl_club_results.sqlite        (tables: events, board_results, boards, ...)
::         acbl/acbl_tournament_results.sqlite
::         acbl/acbl_club_board_results_cleaned.parquet  (for tournament enrichment)
:: WRITES: acbl/acbl_club_board_results_cleaned.parquet
::         acbl/acbl_tournament_board_results_cleaned.parquet
:: TIME:   MEASURED 2026-09-11 03:54-04:27 (~33 min). Club 11.6 GB,
::         tournament 0.83 GB.
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
::         WARM compute is cache hits, but this step still rewrites the full
::         augmented parquets. MEASURED 2026-10-02 06:40-08:50: 7777s (2h 10m).
::         Club 7214s (20.93M rows, 71.2 GB), tournament 556s (2.05M rows, 4.1 GB).
::         DD/Par/SD pbns_to_process=0. Previous warm rewrite 2026-09-11 was ~12 h.
::         3c is longer on a warm rewrite.
echo   [3a] Augmenting hand records (DD + SD + Par)...
call :pyrun 3a acbl_hand_records_augment.py
if errorlevel 1 goto :error

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   MEASURED 2026-10-02 08:50-08:52: 128s. Club 105s
::         (152.4M cleaned -^> 120.87M step1, 9.7 GB), tournament 17s
::         (31.77M -^> 27.77M, 0.54 GB).
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   MEASURED 2026-10-02 08:52 -^> 10-03 01:19: 59213s (16h 27m).
::         Club 47221s, 70,827,715 rows, 82.4 GB. Tournament 11970s,
::         18,667,922 rows, 19.6 GB. Club Date max 2026-09-09.
::         Joins ~6k-col hand-record features into board results.
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
:: TIME:   MEASURED 2026-10-03 01:19-01:59: 2418s (40m). Club 1937s
::         (70.83M rows), tournament 475s (18.67M rows). Walks games
::         chronologically; mostly single-threaded.
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
:: TIME:   COLD REBUILD MEASURED 2026-10-06 from shard-dir creation
::         (after the dirs were deleted) through manifest.json:
::           club       12:42:51-13:31:45 (49 min). 96 shards,
::                      71,773,521 rows, 88.43 GB.
::           tournament 13:31:49-13:48:46 (17 min). 132 shards,
::                      18,970,284 rows, 15.52 GB.
::         Skip-all pass remains 231s (2026-10-03 02:01-02:03). Skip
::         rebuilds a month when manifest source_rows changed, or, for
::         shards written before that field, when the window Date max moved.
::         Earlier cold club rebuild was ~10.5 h on 2026-09-12. Tournament
::         cold was 889 s / 15.3 GB on 2026-08-16.
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
:: TIME:   MEASURED 2026-10-06 13:49-15:50 (2h 2m) from the rebuilt
::         monthly shards:
::           club       13:49-15:29 (1h 40m). 61,705,397 train + 10,068,124
::                      test rows; 203.51 + 26.83 GB. 6,042 columns.
::           tournament 15:29-15:50 (22 min). 17,698,014 train + 1,272,270
::                      test rows; 47.01 + 3.32 GB. 6,033 columns.
::         Test cutoff 2026-01-01. Earlier 2026-09-12 run was ~2.5 h
::         (club 61.71M+7.74M, tournament 15.70M+1.03M). The 2026-04 run
::         against the single merged file was club 7.3 h.
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
:: TIME:   CURRENT BASELINE 45h 31m for all 6 models.
::         MEASURED 2026-10-06 15:50 -^> 10-08 13:21 from artifact mtimes
::         (club 39h 19m, tournament 6h 12m). Clean run.
::         schema mtime = load+prepare done; .pth = shards+epochs done;
::         importance.csv = eval done.
::         Input sizes (test cutoff 2026-01-01):
::           club:      61.71M train + 10.07M test rows, 203.51 + 26.83 GB
::           tournament:17.70M train + 1.27M test rows, 47.01 + 3.32 GB
::         club (39h 19m):
::           Declarer_Direction: load 3h 10m + shards 4h 18m + epochs 9h 35m
::                               + eval 56m. .pth 10-07 08:53, importance 09:50.
::           Contract:           load 3h 03m + shards+epochs 14h 22m + eval 59m.
::                               .pth 10-08 03:14, importance 04:13.
::           Pct_NS (pruned):    load 1h 25m + shards+epochs 1h 22m + eval 10m.
::                               .pth 10-08 07:00, importance 07:09.
::         tournament (6h 12m):
::           Declarer_Direction: load 16m + shards+epochs 2h 18m + eval 5m.
::                               .pth 10-08 09:43, importance 09:49.
::           Contract:           load 13m + shards+epochs 2h 30m + eval 6m.
::                               .pth 10-08 12:32, importance 12:38.
::           Pct_NS (pruned):    load 13m + shards+epochs 29m + eval 1m.
::                               .pth 10-08 13:20, importance 13:21.
::         ACCURACY on the held-out test parquets (debug_predictions_*):
::           club Declarer_Direction: accuracy 0.6644, macro F1 0.664,
::                               majority baseline 0.2623 (N). 60,268 test
::                               rows have a null direction and count as misses.
::           club Contract:      accuracy 0.3795, macro F1 0.182,
::                               majority baseline 0.0408 (3NN). 141 test classes.
::           club Pct_NS:        MAE 0.2595, variance ratio 0.167
::                               (pred_std/actual_std). y max 1.25.
::           tournament Declarer_Direction: accuracy 0.6522, macro F1 0.652,
::                               majority baseline 0.2623 (N).
::           tournament Contract: accuracy 0.3824, macro F1 0.073,
::                               majority baseline 0.0434 (3NN). 378 test classes.
::           tournament Pct_NS:  MAE 0.2514, variance ratio 0.164.
::                               y max 9.99 (sentinel; still in the test file).
::         Earlier 2026-09-13 -^> 09-15 run was ~45 h and resumed after an
::         OOM reboot (club DD) and an AppHang (tournament Pct_NS).
::         HOST MEMORY: this is the bottleneck, not VRAM. Loading train+test
::         together drove Python to ~488 GB and OOM-rebooted club DD
::         (2026-09-13). After loading them one at a time, resume RSS was
::         ~68 GB. The dev box uses 512 GB physical RAM plus a fixed 1.8 TB
::         K: pagefile (~2.27 TB total commit). Do not disable/reduce that
::         pagefile. mlBridgeAiLib.df_to_float32_matrix
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
