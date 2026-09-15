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
echo    Warm rerun (3a DD/SD cache hits; 3a/3c still rewrite full parquets): ~3.5 days
echo      (Stages 2a-4 ~36 h + 5b ~2.5 h + 5c ~45 h; measured 2026-09-11 -^> 09-15).
echo    Stage 5c current baseline: ~45 h for all 6 models (2026-09-13 -^> 09-15).
echo      club ~40 h, tournament ~5 h.
echo    Stage 5d shows the prediction charts 5c no longer opens (close windows to finish).
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Latest complete outputs: 2026-09-10 -^> 2026-09-15
echo    (1c/1d skipped, expired JWT; 5c resumed after OOM and AppHang).
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
::         Last run: parquet dir newest 2026-09-10 10:00, sqlite 18:00
::         (138.7 GB). Use: python acbl_club_json_to_sql.py --legacy-sql-scripts
echo   [1b] Loading club JSON into SQLite (via Parquet)...
call :pyrun 1b acbl_club_json_to_sql.py
if errorlevel 1 goto :error

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
::         2026-09 cycle skipped (expired JWT). Newest sanction JSON 2026-08-14.
echo   [1c] Downloading tournament sanctioned events...
call :pyrun 1c acbl_tournament_download_sanctioned_events.py
if errorlevel 1 goto :error

:: ---- 1d ----
:: READS:  acbl/tournaments/events/*.sanction.json
:: WRITES: acbl/tournaments/sessions/{session_id}.session.json
:: TIME:   incremental; ~minutes daily, several hours cold start (API-bound).
::         90s read timeout covers large NABC full_monty payloads.
::         HTTP 400/404 (unpublished / no boards) do not fail the step.
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
::         augmented parquets. MEASURED 2026-09-11 04:27-16:30 (~12 h):
::         club 71.2 GB @ 16:13, tournament 4.1 GB @ 16:30.
::         Daily incremental (few new PBNs, no full rewrite) is still ~minutes.
::         Longest single step in a cold rebuild; 3c is longer on a warm rewrite.
echo   [3a] Augmenting hand records (DD + SD + Par)...
call :pyrun 3a acbl_hand_records_augment.py
if errorlevel 1 goto :error

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   MEASURED 2026-09-11 16:30-16:33 (~3 min). Club 9.7 GB,
::         tournament 0.54 GB. Was ~5-15 min.
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   MEASURED 2026-09-11 16:33 -^> 09-12 10:59 (~18.4 h). Club 82.4 GB
::         @ 08:07, tournament 19.6 GB @ 10:59. Was ~30-60 min at smaller
::         volume. Joins ~6k-col hand-record features into board results.
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
:: TIME:   MEASURED 2026-09-12 10:59-12:01 (~62 min). Club ~50 min,
::         tournament ~12 min. Walks games chronologically; mostly
::         single-threaded.
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
:: TIME:   MEASURED 2026-09-12: successful resume (shards already valid)
::         ~1 min both modes (manifests 22:36). Club 96 monthly shards +
::         manifest, 85.9 GB. Tournament 132 shards + manifest, 14.3 GB.
::         A failed club pass the same day ran ~10.5 h (12:02-22:31) before
::         the unregistered-column assert; that is the current cold-club
::         cost. Tournament was 889 s / 15.3 GB on 2026-08-16.
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
:: TIME:   MEASURED 2026-09-12 22:36 -^> 09-13 01:06 (~2.5 h) reading
::         monthly shards (not the old merged file):
::           club       ~2.1 h (61.71M train + 7.74M test; 203.5 + 20.4 GB).
::           tournament ~21 min (15.70M train + 1.03M test; 41.6 + 2.7 GB).
::         Earlier 2026-04 run against the single merged file was club 7.3 h
::         / tournament ~30-60 min (see script docstring "KNOWN ISSUE").
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
::         MEASURED 2026-09-13 -^> 09-15 from artifact mtimes (club ~40 h,
::         tournament ~5 h). First club DD reused leftover shards after an
::         OOM reboot; tournament Pct_NS reused shards after AppHang.
::         A clean uninterrupted run should be similar (~44-46 h).
::         Input sizes:
::           club:      61.71M train + 7.74M test rows, 203.5 + 20.4 GB
::           tournament:15.70M train + 1.03M test rows, 41.6 + 2.7 GB
::         club (~40 h; schema mtime = load+prepare done, then shards+epochs
::         until .pth, then eval until importance.csv):
::           Declarer_Direction: load ~3h27m + shards ~4h26m + epochs 9h07m
::                               (~1637 s/epoch) + eval 50m; .pth 09-14 04:23
::           Contract:           load 3h06m + shards+epochs 15h18m + eval 51m
::                               .pth 09-14 23:37
::           Pct_NS (pruned):    load 1h13m + shards+epochs 1h45m + eval 7m
::                               .pth 09-15 03:26
::         tournament (~5 h):
::           Declarer_Direction: load 10m + shards+epochs 2h00m + eval 4m
::                               .pth 09-15 05:43
::           Contract:           load 10m + shards+epochs 2h07m + eval 4m
::                               .pth 09-15 08:04
::           Pct_NS (pruned):    load 8m + shards ~9m + epochs ~18m (~52 s/epoch)
::                               + eval 1m; .pth 09-15 09:22 (resumed).
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
