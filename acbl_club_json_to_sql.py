"""
Club JSON → Parquet (and optionally SQLite) builder.

Default path (option F): parallel JSON → Parquet using
acbl_club_json_to_parquet_sql.py. Since 2026-10-24 the SQLite build is off
by default (stages 2a/2b read the Parquet tables); pass --build-sqlite to
also write acbl_club_results.sqlite (same acbl_club_results_schema.sql).

Legacy path (--legacy-sql-scripts): per-session .data.sql text files then
executescript into an in-memory SQLite (slow; kept for comparison).

Takes ~4h historically via the legacy path for ~1.3M JSON → ~140GB SQLite.
"""

import argparse
import os
import pathlib
import sys
import time
import traceback  # kept for parity with the notebook

# ---------------------------------------------------------------------------
# Graceful Ctrl+C handling on Windows
# ---------------------------------------------------------------------------
_ctrl_c_pressed = False
_handler_refs = []


def _install_windows_ctrl_handler() -> None:
    if sys.platform != "win32":
        return

    import ctypes
    from ctypes import wintypes

    CTRL_C_EVENT = 0
    CTRL_BREAK_EVENT = 1
    HANDLER_ROUTINE = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
    kernel32 = ctypes.windll.kernel32

    def handler(ctrl_type: int) -> bool:
        global _ctrl_c_pressed
        if ctrl_type in (CTRL_C_EVENT, CTRL_BREAK_EVENT):
            if not _ctrl_c_pressed:
                _ctrl_c_pressed = True
                sys.stderr.write("\n\nCtrl+C detected. Exiting gracefully.\n")
                sys.stderr.flush()
            os._exit(130)
        return False

    _handler_func = HANDLER_ROUTINE(handler)
    kernel32.SetConsoleCtrlHandler(_handler_func, True)
    _handler_refs.append(_handler_func)


import mlBridge.mlBridgeLib as mlBridgeLib
import mlBridge.mlBridgeAcblLib as mlBridgeAcblLib

_install_windows_ctrl_handler()

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")


def _run_legacy(keep_bad_sql: bool) -> int:
    from mlBridge import print_ended, print_started

    program_start_time = print_started()
    mlBridgeLib.pd_options_display()

    json_unstemed = {
        str(file).removesuffix(".data.json")
        for file in acblPath.joinpath("club-results").rglob("*/details/*.json")
    }
    sql_unstemed = {
        str(file).removesuffix(".data.sql")
        for file in acblPath.joinpath("club-results").rglob("*/details/*.sql")
    }
    json_files_without_sql_files = [
        pathlib.Path(fn).with_suffix(".data.json") for fn in (json_unstemed - sql_unstemed)
    ]
    print(f"json_files_without_sql_files: {len(json_files_without_sql_files)}")

    mlBridgeAcblLib.club_results_json_to_sql(json_files_without_sql_files)

    create_tables_sql_file = "acbl_club_results_schema.sql"
    db_file = "acbl_club_results.sqlite"
    db_file_connection_string = "sqlite:///" + acblPath.joinpath(db_file).as_posix()
    db_file_path = acblPath.joinpath(db_file)

    mlBridgeAcblLib.club_results_create_sql_db(
        db_file_connection_string,
        create_tables_sql_file,
        db_file_path,
        acblPath,
        "club-results",
        delete_bad_sql_files=not keep_bad_sql,
    )
    print_ended(program_start_time)
    return 0


def main(argv: list[str] | None = None) -> int:
    _install_windows_ctrl_handler()

    parser = argparse.ArgumentParser(
        description=(
            "Build club_results_parquet/ from club-results JSON. "
            "Default: parallel JSON→Parquet; --build-sqlite also writes "
            "acbl_club_results.sqlite. "
            "Use --legacy-sql-scripts for the old .data.sql→SQLite path."
        )
    )
    parser.add_argument(
        "--legacy-sql-scripts",
        action="store_true",
        help="Use legacy per-session .data.sql + executescript path (slow).",
    )
    parser.add_argument(
        "--keep-bad-sql",
        action="store_true",
        help="(Legacy only) On SQL execution errors, keep the offending *.data.sql.",
    )
    # Forwarded to the Parquet path (ignored for --legacy-sql-scripts).
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-parquet", action="store_true")
    parser.add_argument(
        "--build-sqlite", action="store_true",
        help="Also build acbl_club_results.sqlite (off by default; ~7h20m).",
    )
    parser.add_argument(
        "--skip-sqlite", action="store_true",
        help="Accepted for old command lines; skipping SQLite is now the default.",
    )
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate Parquet PK/UNIQUE after coerce; do not build SQLite.",
    )
    args, forward = parser.parse_known_args(argv)

    if args.legacy_sql_scripts:
        return _run_legacy(keep_bad_sql=args.keep_bad_sql)

    # Default: option F
    from acbl_club_json_to_parquet_sql import main as parquet_sql_main

    forwarded: list[str] = list(forward)
    if args.workers is not None:
        forwarded.extend(["--workers", str(args.workers)])
    if args.flush_every is not None:
        forwarded.extend(["--flush-every", str(args.flush_every)])
    if args.limit:
        forwarded.extend(["--limit", str(args.limit)])
    if args.skip_parquet:
        forwarded.append("--skip-parquet")
    if args.build_sqlite:
        forwarded.append("--build-sqlite")
    if args.skip_sqlite:
        forwarded.append("--skip-sqlite")
    if args.keep_shards:
        forwarded.append("--keep-shards")
    if args.preflight_only:
        forwarded.append("--preflight-only")
    return parquet_sql_main(forwarded)


if __name__ == "__main__":
    sys.exit(main())
