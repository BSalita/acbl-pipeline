@echo off
setlocal EnableExtensions
:: Bulk-update recent ACBL club games into e:\bridge\data\acbl\recent.
:: Schedule one of: hour, day, week, quarter.
::   hour    yesterday..today, 200 clubs, cursor continues next run
::   day     last 2 days, every club
::   week    last 8 days, every club
::   quarter last 95 days, every club
cd /d "%~dp0"
set "EVERY=%~1"
if "%EVERY%"=="" set "EVERY=day"
".venv\Scripts\python.exe" acbl_recent_update.py --every %EVERY%
exit /b %ERRORLEVEL%
