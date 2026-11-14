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
:: DD solver for every step: bridge-solver-cpp's DDS-API library (branch
:: faster-search) instead of ddss. mlBridge\dds_ddss.py loads the library
:: named by DDSS_DLL_PATH before its built-in ddss path; each step's log
:: shows the path it loaded. Same results as ddss, so no cache is rebuilt.
:: MEASURED 2026-11-14, revision b12bb3f, 2000 club deals at 16 threads:
::   DD tables 17.4s vs 46.5s, tables+par 8.8s vs 27.0s (1000 deals),
::   opening leads 4.2s vs 12.4s; every result identical to ddss.
:: To use another library (ddss:
::   C:\sw\bridge\ML-Contract-Bridge\src\ddss\build-cmake\Release\dds.dll),
:: set DDSS_DLL_PATH before running this file.
if not defined DDSS_DLL_PATH for %%I in ("%~dp0..\bridge-solver-cpp\build\dds-single.dll") do set "DDSS_DLL_PATH=%%~fI"
if not exist "%DDSS_DLL_PATH%" (
  echo *** FAILED: DD solver library not found: %DDSS_DLL_PATH%
  echo Build it with build.sh in ..\bridge-solver-cpp, or set DDSS_DLL_PATH.
  exit /b 1
)
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
echo      (~38 h Stage 3a + ~1.5 h Stage 3c + ~45 h Stage 5c + other stages).
echo    Latest run 2026-11-12 -^> 11-14 (six weeks of new data, 251K new deals):
echo      1a 8h18m + 1b 27m + 1c 20m + 1d 3h + 1e 7m + 2a 3m + 2b 16m
echo      + 3a 8h41m (6h43m of it new DD/SD) + 3b 2m + 3c 6h03m (est. 2h35m
echo      alone, both sides in blocks) + 4 5m + 5a 39m (cold rebuild)
echo      + 5b 56m + 5v 8s.
echo    Stage 5c current baseline: 45h 31m for all 6 models (2026-10-06 15:50 -^> 10-08 13:21).
echo      club 39h 19m, tournament 6h 12m. Clean run; no OOM resume.
echo    Stage 5d shows the prediction charts 5c no longer opens (close windows to finish).
echo    Empirical bottlenecks per stage are noted as "TIME:" tags below.
echo    Each step prints its own measured elapsed time as "TIME[step]: ..." lines.
echo  Latest measured stages: 1a-5v from 2026-11-12 -^> 11-14. 5c from
echo    the 2026-10-06 -^> 10-08 rebuild.
echo  Model training results history: RESULTS.md (append an entry after each 5c run).
echo ======================================================================
echo.
echo Using: %PY%
echo DD solver: %DDSS_DLL_PATH%
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

:: ====================================================================
:: STAGE 1: DATA INGESTION -- download raw data from ACBL
:: ====================================================================
echo [Stage 1] Data ingestion...

:: ---- 1a ----
:: READS:  (web scrape from my.acbl.org)
:: WRITES: acbl/club-results/{club_id}/details/{session_id}.data.json
:: TIME:   MEASURED 2026-11-12 06:14-14:32: 29870s (8h 18m) at the 3.3 s
::         default, no 403. 1948 clubs, 6856 sessions extracted, Failed: 0;
::         six weeks since the last sweep. Incremental, network-bound.
::         2026-10-01 at 5 s: 36066s (10h 1m), 1944 clubs, 7689 sessions.
::         Since 2026-10-24: session requests start at most once per --sleep
::         (no extra fixed sleep) and listing pages wait for their content
::         instead of a fixed 2 s. --concurrency N (default 1) is opt-in; it
::         only helps if page loads exceed --sleep.
::         Since 2026-11-12: --sleep applies to every page request (club list,
::         club listing pages, sessions). The wait comes right after each
::         page is fetched: the rest of --sleep since its request started.
::         Listing pages count toward ACBL's limit: the 06:05 run fetched 121
::         listing pages about 2 s apart, no sessions, and got the 403. The
::         05:37 run got it after 58 listing pages and 56 sessions.
::         A catch-up run now takes at least 1948 clubs x --sleep (1.8 h at
::         3.3 s) even when nothing is new.
::         A small daily catch-up is minutes; this pass was a full sweep.
:: 403:    MEASURED 2026-11-12 by test_acbl_download_delay.py: "403 Forbidden"
::         came at 2.7 s (request 113), 2.9 s (122) and 3.0 s (257); 3.1 s
::         and 3.5 s passed 400 requests; 3.3 s passed 1000 requests. Each
::         403 cleared in 1-2 min. If a 403 does come, the downloader waits
::         until it clears and resumes the same session.
::         --sleep is not passed: the downloader's DEFAULT_SLEEP_SECONDS
::         (3.3, was 5) applies. Rerun the test with --apply to remeasure
::         and update that default.
echo   [1a] Downloading club results to JSON...
call :pyrun 1a acbl_club_download_to_json.py
if errorlevel 1 goto :error

:: ---- 1b ----
:: READS:  acbl/club-results/*/details/*.data.json
:: WRITES: acbl/club_results_parquet/*.parquet
::         (acbl/acbl_club_results.sqlite only with --build-sqlite; see below)
:: TIME:   Since 2026-10-24 parsed JSON is cached per 200-file chunk in
::         acbl/club_results_parquet_cache/ (~20 GB); only new or changed files
::         are re-parsed (--rebuild-cache forces a full parse). Measured on the
::         full corpus: nothing new 45 s; cold (first run) ~38 min; a typical
::         run with a few thousand new files ~7-8 min (merge is the floor).
::         MEASURED 2026-11-12 16:19-16:45: 1599s (27m). 900,751 files parsed
::         in 5,876 chunks (2,740 chunks reused), err=0, 34 tables merged.
::         Before the cache: ~16 min without the SQLite build (2026-10-01).
::         MEASURED 2026-10-01 22:45 -^> 10-02 06:40 WITH the SQLite build:
::         28533s (7h 56m). 1,335,293 JSON files, err=0. Parquet merge
::         16m20s, sqlite preflight 6m44s, table load ~7h20m, 139.34 GB.
::         SQLite is skipped by default since 2026-10-24: 2a/2b read the
::         parquets directly. acbl_compare_sqlite_vs_parquet.py showed all 9
::         club queries give identical results from both. The 2026-10-02
::         sqlite is left as is (stale). To rebuild it, add --build-sqlite.
::         Use: python acbl_club_json_to_sql.py --legacy-sql-scripts
echo   [1b] Converting club JSON to Parquet...
call :pyrun 1b acbl_club_json_to_sql.py
if errorlevel 1 goto :error

:: ---- 1c ----
:: READS:  (ACBL API via ACBL_API_KEY)
:: WRITES: acbl/tournaments/events/{sanction_id}.sanction.json
:: TIME:   incremental; ~minutes daily, ~1-2 h cold start (API-rate-limited).
::         MEASURED 2026-11-12 16:45-17:05: 1204s (20m) on the live.acbl.org
::         path, 808 tournaments checked at 1.5 s each, 94 events written.
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
::         MEASURED 2026-11-12 on the live.acbl.org path, two passes:
::         17:05-19:02 6993s (stopped by hand) + 19:05-20:08 3811s = 3h.
::         172,215 sessions already saved, 6,574 asked for, 75 written; the
::         other 6,499 were team sessions with no pair scorecards, now
::         listed as unavailable, so the next run does not ask again.
::         90s read timeout covers large NABC full_monty payloads.
::         HTTP 400/404 (unpublished / no boards) do not fail the step.
::         A rejected or expired ACBL_API_KEY falls back to live.acbl.org
::         (same Chrome/Cloudflare download path as club results).
::         --sleep is not passed: DEFAULT_SLEEP_SECONDS (0 = no pause) applies.
::         On the live.acbl.org path it is a pause after every pair scorecard
::         page. test_acbl_download_delay.py --target tournament measures it.
::         The live.acbl.org path cannot save team sessions (no pair
::         scorecards). Once seen that way, 30+ days after the event, they
::         are listed in sessions\_web_no_pair_scorecards.json and skipped;
::         --recheck-web-unavailable asks for them again. The API skips none.
::         2026-09 cycle skipped (expired JWT). Newest session JSON 2026-09-02.
echo   [1d] Downloading tournament sessions...
call :pyrun 1d acbl_tournament_download_sessions_using_sanctioned_events.py --timeout 90
if errorlevel 1 goto :error

:: ---- 1e ----
:: READS:  acbl/tournaments/sessions/*.session.json
:: WRITES: acbl/tournament_results_parquet/*.parquet  (one file per table, Utf8)
::         acbl/tournaments/sessions/*.session.sql + acbl/acbl_tournament_results.sqlite
::           only with --build-sqlite (off by default since 2026-10-24, like 1b).
::           2a/2b read the parquets. Check with
::           acbl_compare_sqlite_vs_parquet.py --side tournament.
:: TIME:   parquet build: full rebuild of 172K session files, see TIME[1e].
::         MEASURED 2026-11-12 20:24-20:31: 411s (7m). 172,288 session files,
::         15 tables, no SQLite build.
::         MEASURED 2026-09-10/11 WITH the SQLite build: finished 00:16,
::         sqlite 15.0 GB. If this started after 1b (18:00), it was ~6 h.
echo   [1e] Converting tournament sessions to Parquet...
call :pyrun 1e acbl_tournament_sessions_json_to_sql.py
if errorlevel 1 goto :error

:: ====================================================================
:: STAGE 2: CLEANING -- raw parquet tables -> cleaned parquets
:: ====================================================================
echo.
echo [Stage 2] Cleaning...

:: ---- 2a ----
:: READS:  acbl/club_results_parquet/{hand_records,sessions}.parquet
::         acbl/tournament_results_parquet/{handrecord,session}.parquet
:: WRITES: acbl/acbl_club_hand_records_cleaned.parquet
::         acbl/acbl_tournament_hand_records_cleaned.parquet
:: TIME:   MEASURED 2026-11-12 20:31-20:33: 155s. Club 135s (21,651,288
::         rows), tournament 15s (2,081,812 rows). Club alone, now written in
::         session_id order for 3c: 102s on 2026-11-13; 103s on 2026-11-14,
::         the first run to drop identical repeated club rows (24,757 of
::         them; 21,626,531 rows left).
::         From the SQLite database, 2026-09-11: ~3.6 h.
echo   [2a] Cleaning hand records...
call :pyrun 2a acbl_sql_to_hand_records_clean.py
if errorlevel 1 goto :error

:: ---- 2b ----
:: READS:  acbl/club_results_parquet/{events,board_results,boards,...}.parquet
::         acbl/tournament_results_parquet/{event,board_results,sections}.parquet
::         acbl/acbl_club_board_results_cleaned.parquet  (for tournament enrichment)
:: WRITES: acbl/acbl_club_board_results_cleaned.parquet
::         acbl/acbl_tournament_board_results_cleaned.parquet
:: TIME:   MEASURED 2026-11-12 20:33-20:49: 953s (16m). Club 407s
::         (156.00M rows, 11.6 GB), tournament 546s (32.69M rows, 0.92 GB).
::         2026-09-11: ~33 min.
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
::         augmented parquets.
::         MEASURED 2026-11-12 20:50 -^> 11-13 05:30 with 251,482 new deals:
::         31246s (8h 41m). Club 30634s (21.65M rows): DD 23m, SD 6h 20m,
::         everything else 1h 48m. Tournament 595s (2.08M rows, 4.5 GB).
::         MEASURED 2026-11-13 20:41-22:32, warm, club only, rows now in
::         session_id order (47.1 GB, was 78.5 GB): 6677s (1h 51m), with a
::         DD verification using most cores from 20:57.
::         MEASURED 2026-11-14 09:26-10:53, warm, club only (21,626,531
::         rows, 47.1 GB), nothing else running: 5220s (1h 27m).
::         2026-10-02 warm rewrite: 7777s (2h 10m), club 7214s.
::         Since 2026-10-24 a side is skipped (~1 s check) when its cleaned
::         input, cache rows, code and versions match
::         acbl_{club,tournament}_hand_records_augmented_manifest.json;
::         --force recomputes. The first run after this change recomputes.
:: OPTION: set ACBL_SUITC=1 to add the SC_* SuitC single-suit columns (needs
::         ..\SuitC\build\Release\SuitC.exe). Holdings are cached in
::         E:\bridge\data\suitc\combos\ (shared with FFBridge); the first run
::         solves ~800k holdings (~1-2 h). Changing it recomputes 3a and 3c.
set "SUITC_ARGS="
if "%ACBL_SUITC%"=="1" set "SUITC_ARGS=--suitc"
echo   [3a] Augmenting hand records (DD + SD + Par; SuitC %ACBL_SUITC%)...
call :pyrun 3a acbl_hand_records_augment.py %SUITC_ARGS%
if errorlevel 1 goto :error

:: ---- 3b ----
:: READS:  acbl/acbl_{club,tournament}_board_results_cleaned.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
:: TIME:   MEASURED 2026-11-13 05:30-05:32: 132s. Club 106s
::         (156.00M cleaned -^> 123.97M step1, 9.8 GB), tournament 18s
::         (32.69M -^> 28.35M, 0.59 GB). Club alone 113s on 2026-11-14.
::         Since 2026-11-14 club Tricks is always total tricks: where ACBL's
::         tricks_taken is tricks over book, 6 is added (1,724 rows then).
echo   [3b] Augmenting board results (step 1: contracts + vulnerability)...
call :pyrun 3b acbl_board_results_augment_step1.py
if errorlevel 1 goto :error

:: ---- 3c ----
:: READS:  acbl/acbl_{club,tournament}_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_hand_records_augmented.parquet
:: WRITES: acbl/acbl_{club,tournament}_board_results_augmented.parquet
:: TIME:   Both sides run in blocks of sessions, kept in
::         acbl/acbl_{club,tournament}_board_results_augmented_blocks. A
::         rerun recomputes only blocks whose rows changed; editing this
::         script, mlBridgeAugmentLib.py or mlBridgeLib.py recomputes all.
::         All blocks with the machine otherwise idle: ~1h 35m (club 1h 13m
::         measured, tournament ~20m estimated).
::         Club, 62 blocks of 25,000 session ids. MEASURED 2026-11-14
::         10:53-12:06, all blocks, nothing else running: 4393s (1h 13m),
::         73,347,299 rows, 75.4 GB; about 70 s per block of 1.4M rows, then
::         24 s for the ratings and 250 s to assemble the file.
::         2026-11-13 23:15 -^> 11-14 02:51: 12958s (3h 36m), 73,404,432 rows
::         (57,133 of them repeats), with a DD verification using most cores
::         for the first 2h 40m.
::         With no block changed a side is only reassembled: tournament 222s
::         on 2026-11-14 (147 blocks checked in 10 s, ratings 13 s, file 151 s).
::         Tournament, 147 blocks (a month of sanctions or a year of NABCs).
::         MEASURED 2026-11-14 06:39-07:04: 1523s (25m), 18,980,984 rows,
::         15.4 GB, while 5a and 5b were running. All columns equal the
::         load-everything run except the three player ratings, which now
::         average over the rows in the file (1,744 rows are dropped after
::         the old code had already counted them).
::         Before blocks: tournament 8799s (2h 27m) on 2026-11-14, paging;
::         2026-10-02 both sides 59213s (16h 27m), club 47221s.
::         Joins ~6k-col hand-record features into board results.
:: OPTION: set ACBL_SD_LEAD_SAMPLES=20 to add the single-dummy lead score
::         (SD_Lead_* columns, club only: tournaments record no lead). The first
::         club run solves ~5.7M lead positions, ~33 h at 20 (48 positions/s);
::         acbl/acbl_club_sd_lead_cache/ keeps them, so later runs solve only new
::         ones. Changing the value recomputes every club block.
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
:: TIME:   MEASURED 2026-11-14 05:18-05:24: 321s (5m). Club 256s
::         (73.40M rows), tournament 58s (18.98M rows).
::         2026-10-03: 2418s (40m), club 1937s, tournament 475s.
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
:: TIME:   COLD CLUB REBUILD MEASURED 2026-11-14 06:57-07:38: 2464s (41m),
::         while a 3c tournament test and a file comparison were running.
::           club       2413s (40 min). 96 shards, 73,347,299 rows,
::                      88.95 GB. Estimate alone: ~35 min.
::           tournament 43s, all 132 shards skipped (18,980,984 rows,
::                      15.53 GB).
::         Cold rebuild of both, same day 05:32-06:10, nothing else running:
::         2316s (39m), club 2110s, tournament 199s.
::         2026-10-06 cold rebuild: club 49 min (71,773,521 rows),
::         tournament 17 min (18,970,284 rows).
::         The 05:32 run wrote 73,760,368 club rows, more than 3c had
::         (73,404,432): the club hand records held 24,757 identical extra
::         rows (7,906 boards in 376 sessions listed 2-10 times). 3c
::         repeated those boards' results
::         (57,133 extra rows) and the 5a join repeated them again (355,936
::         more). Since 2026-11-14 5a joins to one row per deal and keeps
::         one row per club board_result_id, and 2a drops the repeated hand
::         records, so from the next full run 3c has none either.
::         The column checks at the start fail in under 30 s when 3c adds
::         or drops a column; add it to the catalog in acbl_model_data.py.
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
:: WRITES: acbl/acbl_{club,tournament}_prediction_data_train_v2.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test_v2.parquet
:: TIME:   MEASURED 2026-11-14 07:38-08:34: 3347s (56m), nothing else
::         running:
::           club       2329s (39 min). 61,291,826 train + 12,055,473
::                      test rows; 57.90 + 9.72 GB. 6,042 columns.
::           tournament 1010s (17 min). 17,697,878 train + 1,283,106
::                      test rows; 9.90 + 0.79 GB. 6,036 columns.
::         The files are about a quarter of the 2026-10-06 sizes for the
::         same columns; null counts of 60 sampled columns equal the 5a
::         shards on both sides.
::         2026-10-06: 2h 2m, club 1h 40m (61,705,397 + 10,068,124 rows,
::         203.51 + 26.83 GB), tournament 22 min (47.01 + 3.32 GB).
::         Test cutoff 2026-01-01. Earlier 2026-09-12 run was ~2.5 h
::         (club 61.71M+7.74M, tournament 15.70M+1.03M). The 2026-04 run
::         against the single merged file was club 7.3 h.
echo   [5b] Preparing prediction data (train/test split)...
call :pyrun 5b acbl_prediction_data.py
if errorlevel 1 goto :error

:: ---- 5v ----
:: READS:  acbl/acbl_club_board_results_augmented_step1.parquet
::         acbl/acbl_{club,tournament}_board_results_augmented.parquet
::         acbl/acbl_{club,tournament}_{,player_,pair_}elo_ratings.parquet
::         acbl/acbl_club_prediction_data_test_v2.parquet
:: WRITES: (none; exits 1 on any FAIL)
:: TIME:   MEASURED 2026-11-14: 8s.
::         That first full run FAILED 2 of 33 checks, both on the club step1
::         file: 1,724 rows in 8 sessions of 2026-09 where ACBL's
::         tricks_taken holds tricks over book (Result is right), and
::         Result null in 1.63% of played contracts against the 1% limit
::         (ACBL gave none for 3.2% of 2019 and 11.9% of 2020). The 3c
::         output passed both: none of the 1,724 rows reach it and 0.25%
::         of its contracts have a null Result.
::         Fixed the same day: 3b converts those trick counts, and the
::         null-Result limit counts contracts from 2021 on (0.06% in step1).
::         33 of 33 passed at 09:31 after rerunning 3b.
::         Verifies the 2026-10 data fixes: doubled contracts kept, typed
::         Result/Tricks, player ID 0 nulled, unrated seats, ordered pair keys.
::         Runs before 5c so bad data does not cost ~45 h of training.
::         After the first passing run, merge acbl-pipeline branch
::         after-nov-2026-acbl-rebuild (drops the club API raw-result override).
echo   [5v] Verifying data fixes...
call :pyrun 5v acbl_verify_data_fixes.py
if errorlevel 1 goto :error

:: ---- 5c ----
:: READS:  acbl/acbl_{club,tournament}_prediction_data_train_v2.parquet
::         acbl/acbl_{club,tournament}_prediction_data_test_v2.parquet
:: WRITES: acbl/SavedModels/{model_name}_schema.json                -> Chatbot
::         acbl/SavedModels/{model_name}.pth                        -> Chatbot
::         acbl/SavedModels/*model_shard_*.pt                       (transient)
::         acbl/SavedModels/{model_name}_importance.csv
::         acbl/debug_predictions_{club,tournament}_{target}.parquet
::         acbl/debug_input_{club,tournament}_{target}.parquet  (only with
::           --save-debug-input since 2026-10-24; ~27 GB per club target)
::         acbl/acbl_{club,tournament}_prediction_rank_metrics.json  (after
::           Pct_NS: Spearman per section/direction vs a seat-Elo line,
::           via acbl_prediction_rank_eval.py; ~1 min.)
::         Ids and dates (event/session/section_id, Club, Date, game_date)
::           are not model inputs (mlBridgeAiLib schema blacklist) since
::           2026-10-19. Tournament ids stay in the 5b parquets as metadata.
::         Charts are not shown here (MPLBACKEND=Agg). Use 5d.
::         Joint contract/declarer pick (what postmortem-acbl serves):
::           acbl_prediction_rank_eval.py --side club|tournament (~3 min).
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
::           club rank (2026-10-18): Spearman 0.456 vs seat-Elo line 0.433
::                               (92,573 section/directions). Section
::                               centering keeps ranks, pair MAE 0.0556 ->
::                               0.0544; board centering adds nothing.
::           club joint pick (declarer weight 0.3, 305k boards): contract
::                               0.3787 -^> 0.3787, declarer 0.6630 -^> 0.6644,
::                               contract/declarer agree 85.4%% -^> 100%%.
::           tournament joint pick (318k boards): contract 0.3826 -^> 0.3830,
::                               declarer 0.6529 -^> 0.6598.
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
>>"%LOG_DIR%\summary.log" echo  TIME[total]: %PIPE_ELAPSED%s (%PIPE_H%h %PIPE_M%m %PIPE_S%s)
echo  Logs: %LOG_DIR%
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
