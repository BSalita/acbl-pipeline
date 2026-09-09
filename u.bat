rem Publish ACBL club/tournament parquets to X1 elo\data\_wslc_host\acbl-stage.
rem Production containers mount that tree (postmortem ACBL_CLUB_* and
rem BridgeStats extra-data). OneDrive does not sync this; copy over UNC
rem unless this script is already running on X1.
rem Do not copy into this repo's club_results_parquet or club-results-recent.
rem
rem Five small relationship tables support listings and lookups. The Stage 3c
rem augmented monolith supplies complete historical club postmortems. The
rem tournament monolith provides the same API/MCP path for tournaments.
rem Recent session-details JSON (last 30 days) is the archive tier so results
rem are served without live Cloudflare scrapes.

set "SRC=e:\bridge\data\acbl"
set "STAGE=\\X1-pro-470-1tb\c\sw\bridge\ML-Contract-Bridge\src\elo\data\_wslc_host\acbl-stage"
if /i "%COMPUTERNAME%"=="X1-PRO-470-1TB" (
    set "STAGE=%~dp0..\elo\data\_wslc_host\acbl-stage"
)
if not exist "%STAGE%\" (
    echo *** FAILED: _wslc_host acbl-stage not found: %STAGE%
    exit /b 1
)

if not exist "%STAGE%\club_results_parquet\" (
    mkdir "%STAGE%\club_results_parquet"
    if errorlevel 1 exit /b 1
)

xcopy "%SRC%\club_results_parquet\events.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\club_results_parquet\players.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\club_results_parquet\pair_summaries.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\club_results_parquet\sections.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\club_results_parquet\sessions.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\acbl_club_board_results_augmented.parquet" "%STAGE%\club_results_parquet\" /D /Y
xcopy "%SRC%\acbl_tournament_board_results_augmented.parquet" "%STAGE%\club_results_parquet\" /D /Y
if errorlevel 1 exit /b 1

rem Obsolete Stage 1b reconstruction files. Recent sessions missing from the
rem monolith use the bounded JSON archive/live fallback instead.
for %%F in (club.parquet boards.parquet board_results.parquet hand_records.parquet) do (
    del /q "%STAGE%\club_results_parquet\%%F" 2>nul
)

if not exist "%STAGE%\club-results-recent\" (
    mkdir "%STAGE%\club-results-recent"
    if errorlevel 1 exit /b 1
)
robocopy "%SRC%\club-results" "%STAGE%\club-results-recent" *.data.json /S /MAXAGE:30 /NDL /NFL /NJH /NJS /R:2 /W:2
if errorlevel 8 exit /b 1
forfiles /P "%STAGE%\club-results-recent" /S /M *.data.json /D -45 /C "cmd /c del @path" 2>nul

exit /b 0
