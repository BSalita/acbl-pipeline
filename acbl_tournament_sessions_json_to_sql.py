"""
ACBL Tournament Sessions JSON -> SQL -> SQLite

Takes 23m to process 150,000 session files.

Converts `tournaments/sessions/*.session.json` files to per-session `.session.sql`
files, then loads those SQL scripts into a SQLite database.

This is a script version of `acbl_tournament_session_json_to_sql.ipynb`, and is
structured similarly to `acbl_club_json_to_sql.py`.

High-level pipeline:
  1) Read `*.session.json` files
  2) Convert JSON to SQL statements (via `mlBridge.mlBridgeAcblLib`)
  3) Write a `*.session.sql` file next to each JSON
  4) Execute `*.session.sql` scripts into SQLite in batched transactions

Notes:
  - The schema is defined by `acbl_tournament_sessions_schema.sql`.
  - Some rare session files may fail due to schema drift (e.g. bracketed events).
    This script matches the notebook behavior: it deletes the offending `.sql`
    and continues.
  - Load uses heap tables (no PK/FK), bulk-import PRAGMAs, and batched
    BEGIN/COMMIT. Unique indexes are created after all session scripts run.
    Bare executescript() plus in-memory PK upserts get slower as the DB grows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
import traceback
from collections import defaultdict
from typing import Iterable, Optional

import sqlalchemy
import sqlalchemy_utils

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")

# One transaction per this many session scripts (or this many bytes of SQL).
DEFAULT_LOAD_BATCH_FILES = 250
DEFAULT_LOAD_BATCH_BYTES = 32 * 1024 * 1024
_ON_CONFLICT_RE = re.compile(
    r"\s*ON CONFLICT\([^)]+\)\s+DO UPDATE SET.*?;",
    re.IGNORECASE | re.DOTALL,
)
_TRAILING_COMMA_BEFORE_CLOSE_RE = re.compile(
    r",(?P<tail>(?:\s*--[^\n]*)*\s*)\)",
)
# Negative cache_size is KiB. 4 GiB is cheap on a 512 GiB machine and keeps
# the growing B-trees in the pager cache during the bulk load.
LOAD_CACHE_SIZE_KIB = 4 * 1024 * 1024


def _iter_files_sorted(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    # Deterministic ordering for reproducible runs.
    return sorted(paths, key=lambda p: p.as_posix())


def _schema_sql_without_wal(schema_text: str) -> str:
    """Drop journal_mode=WAL from schema so bulk load can use journal_mode=OFF."""
    lines = []
    for line in schema_text.splitlines():
        if line.strip().upper().startswith("PRAGMA JOURNAL_MODE"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _schema_sql_for_bulk_load(schema_text: str) -> str:
    """Heap tables: no WAL, no PRIMARY KEY, no FOREIGN KEY.

    Random VARCHAR PK upserts are why the load rate falls as the DB grows.
    Indexes are created once after all session scripts have been inserted.
    """
    lines = []
    for line in _schema_sql_without_wal(schema_text).splitlines():
        if line.strip().upper().startswith("FOREIGN KEY"):
            continue
        line = re.sub(r"\s+PRIMARY KEY\b", "", line, flags=re.IGNORECASE)
        lines.append(line)
    return _TRAILING_COMMA_BEFORE_CLOSE_RE.sub(r"\g<tail>)", "\n".join(lines))


def _strip_upserts(sql_script: str) -> str:
    """Turn ON CONFLICT DO UPDATE into a plain INSERT (heap tables have no unique index)."""
    return _ON_CONFLICT_RE.sub(";", sql_script)


def _pragma(conn, sql: str) -> None:
    result = conn.execute(sql)
    fetchone = getattr(result, "fetchone", None)
    if fetchone is not None:
        fetchone()


def _set_page_size(conn) -> None:
    """page_size only sticks on an empty DB; VACUUM applies it after create_database()."""
    _pragma(conn, "PRAGMA page_size=32768;")
    conn.execute("VACUUM")
    commit = getattr(conn, "commit", None)
    if commit is not None:
        commit()


def _apply_bulk_load_pragmas(conn) -> None:
    """Match the club parquet→SQLite fast path. Must run before schema + inserts."""
    _pragma(conn, "PRAGMA foreign_keys=OFF;")
    # journal_mode returns a row; fetch it so SQLite actually switches modes.
    _pragma(conn, "PRAGMA journal_mode=OFF;")
    _pragma(conn, "PRAGMA synchronous=OFF;")
    _pragma(conn, "PRAGMA temp_store=MEMORY;")
    _pragma(conn, f"PRAGMA cache_size=-{LOAD_CACHE_SIZE_KIB};")
    _pragma(conn, "PRAGMA locking_mode=EXCLUSIVE;")
    _pragma(conn, "PRAGMA mmap_size=8589934592;")  # 8 GiB; no-op for :memory:


def _restore_runtime_pragmas(conn) -> None:
    _pragma(conn, "PRAGMA journal_mode=WAL;")
    _pragma(conn, "PRAGMA synchronous=NORMAL;")
    _pragma(conn, "PRAGMA foreign_keys=ON;")


def _transactional_script(sql_script: str) -> str:
    # executescript() COMMITs first, then sqlite3_exec auto-commits every
    # statement unless the script itself opens a transaction.
    return "BEGIN;\n" + sql_script + "\nCOMMIT;\n"


def _finalize_unique_indexes(conn) -> None:
    """Add UNIQUE(id) after the heap load so lookups still have a primary key."""
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    tables = [row[0] for row in result.fetchall()]
    t0 = time.time()
    for i, table in enumerate(tables, 1):
        idx = f"ux_{table}_id"
        print(f"Creating unique index on {table} ({i}/{len(tables)})...", flush=True)
        try:
            conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx}" ON "{table}"("id")')
        except Exception as exc:
            print(f"  duplicates in {table} ({type(exc).__name__}); keeping first id")
            conn.execute(
                f'DELETE FROM "{table}" WHERE rowid NOT IN '
                f'(SELECT MIN(rowid) FROM "{table}" GROUP BY "id")'
            )
            conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{idx}" ON "{table}"("id")')
        commit = getattr(conn, "commit", None)
        if commit is not None:
            commit()
    print(f"Unique indexes ready in {time.time() - t0:.1f}s", flush=True)


class ACBLTournamentSessionSqliteBuilder:
    def __init__(
        self,
        sessions_dir: pathlib.Path,
        schema_file: pathlib.Path,
        db_file: pathlib.Path,
        write_direct_to_disk: bool,
        create_engine_echo: bool,
    ) -> None:
        self.sessions_dir = sessions_dir
        self.schema_file = schema_file
        self.db_file = db_file
        self.write_direct_to_disk = write_direct_to_disk
        self.create_engine_echo = create_engine_echo

        self.db_memory_connection_string = "sqlite://"
        self.db_file_connection_string = "sqlite:///" + self.db_file.as_posix()

    def find_session_json_files(self) -> list[pathlib.Path]:
        return _iter_files_sorted(self.sessions_dir.rglob("*.session.json"))

    def find_session_sql_files(self) -> list[pathlib.Path]:
        return _iter_files_sorted(self.sessions_dir.rglob("*.session.sql"))

    def generate_sql_files(
        self,
        starting_nfile: int = 0,
        ending_nfile: int = 0,
        skip_existing_files: bool = True,
        initially_delete_all_output_files: bool = False,
        delete_bad_json: bool = True,
    ) -> tuple[int, int, int]:
        """
        Convert JSON session files to SQL files using mlBridge helpers.

        Returns:
            (total_urls, total_files_written, total_skipped)
        """
        try:
            import mlBridge.mlBridgeAcblLib as mlBridgeLib  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency: cannot import `mlBridge.mlBridgeAcblLib`.\n"
                "This script expects the same environment as the original notebook."
            ) from e

        urls = self.find_session_json_files()
        if ending_nfile == 0:
            ending_nfile = len(urls)
        filtered_urls = urls[starting_nfile:ending_nfile]

        if initially_delete_all_output_files:
            for url in filtered_urls:
                url.with_suffix(".sql").unlink(missing_ok=True)

        start_time = time.time()
        total_files_written = 0
        total_skipped = 0

        skip_count_streak = 0
        for n, url in enumerate(filtered_urls, start=1):
            json_file = url
            sql_file = url.with_suffix(".sql")

            if (
                skip_existing_files
                and sql_file.exists()
                and sql_file.stat().st_ctime > json_file.stat().st_ctime
            ):
                print(f"Skipping ({n}/{len(filtered_urls)}): file:{json_file.as_posix()}")
                total_skipped += 1
                skip_count_streak += 1
                continue

            print(
                f"Processing ({n}/{len(filtered_urls)}): file:{json_file.as_posix()} skipped:{skip_count_streak}"
            )
            skip_count_streak = 0

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data_json = json.load(f)
            except Exception as e:
                print(f"Error: {e}: file:{json_file.as_posix()}")
                if delete_bad_json:
                    json_file.unlink(missing_ok=True)
                continue

            tables = defaultdict(lambda: defaultdict(dict))

            # Notebook behavior:
            # - When 'id' exists at top-level, use it as the primary key.
            # - When 'session_id' exists, still uses primary_keys=['id'] but validates 'session'.
            if "id" in data_json:
                primary_keys = ["id"]
                mlBridgeLib.json_to_sql_walk(tables, "session", "", "", data_json, primary_keys)
            elif "session_id" in data_json:
                if isinstance(data_json.get("session"), dict) or len(data_json.get("session", [])) == 0:
                    print(f"Invalid file: session is dict or empty: {json_file.as_posix()}")
                    continue
                primary_keys = ["id"]
                mlBridgeLib.json_to_sql_walk(tables, "session", "", "", data_json, primary_keys)
            else:
                print(f"Invalid file (missing both id and session): {json_file.as_posix()}")
                continue

            with open(sql_file, "w", encoding="utf-8") as f:
                mlBridgeLib.CreateSqlFile(tables, f, primary_keys)

            total_files_written += 1
            print(f"Writing: session id:{data_json.get(primary_keys[0])} file:{sql_file.as_posix()}")

        print(
            "All files processed:"
            f"{len(filtered_urls)} files written:{total_files_written} skipped:{total_skipped} "
            f"total time:{round(time.time() - start_time, 2)}"
        )
        return (len(filtered_urls), total_files_written, total_skipped)

    def load_sql_files_into_db(
        self,
        starting_nfile: int = 0,
        ending_nfile: int = 0,
        recreate_db: bool = True,
        create_tables: bool = True,
        perform_integrity_checks: bool = False,
        delete_bad_sql: bool = True,
        load_batch_files: int = DEFAULT_LOAD_BATCH_FILES,
        load_batch_bytes: int = DEFAULT_LOAD_BATCH_BYTES,
    ) -> int:
        """
        Execute all `.session.sql` scripts into SQLite.

        Returns:
            total_scripts_executed
        """
        def _create_tables(raw_connection) -> None:
            print(f"Creating heap tables (indexes after load) from:{self.schema_file.as_posix()}")
            with open(self.schema_file, "r", encoding="utf-8") as f:
                create_sql = _schema_sql_for_bulk_load(f.read())
            raw_connection.executescript(create_sql)
            # Schema may have toggled journal_mode; keep the bulk-load setting.
            _apply_bulk_load_pragmas(raw_connection)

        def _handle_script_error(sql_file: pathlib.Path, exc: Exception) -> bool:
            """Retry once on missing session table. Returns True if retry worked."""
            msg = str(exc)
            if not (
                "no such table" in msg
                and ("session" in msg or "Session" in msg)
                and create_tables
            ):
                return False
            print("Detected missing table; recreating schema and retrying once.")
            _create_tables(raw_connection)
            return True

        def _report_bad_sql(sql_file: pathlib.Path, exc: Exception) -> None:
            print(f"Error: {type(exc).__name__} while processing file:{sql_file.as_posix()}")
            print(traceback.format_exc())
            if delete_bad_sql:
                print(f"Removing {sql_file.as_posix()}")
                sql_file.unlink(missing_ok=True)

        def _execute_one(sql_file: pathlib.Path, sql_script: str) -> bool:
            try:
                raw_connection.executescript(_transactional_script(sql_script))
                return True
            except Exception as e:
                if _handle_script_error(sql_file, e):
                    try:
                        raw_connection.executescript(_transactional_script(sql_script))
                        return True
                    except Exception as retry_exc:
                        _report_bad_sql(sql_file, retry_exc)
                        return False
                _report_bad_sql(sql_file, e)
                return False

        def _flush_batch(batch_files: list[pathlib.Path], batch_scripts: list[str]) -> int:
            if not batch_scripts:
                return 0
            if len(batch_scripts) == 1:
                return 1 if _execute_one(batch_files[0], batch_scripts[0]) else 0
            try:
                raw_connection.executescript(_transactional_script("\n".join(batch_scripts)))
                return len(batch_scripts)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(
                    f"Batch of {len(batch_scripts)} scripts failed ({type(e).__name__}); "
                    "retrying each file."
                )
                return sum(
                    1
                    for sql_file, sql_script in zip(batch_files, batch_scripts)
                    if _execute_one(sql_file, sql_script)
                )

        if recreate_db and sqlalchemy_utils.functions.database_exists(self.db_file_connection_string):
            print(f"Deleting db:{self.db_file_connection_string}")
            sqlalchemy_utils.functions.drop_database(self.db_file_connection_string)

        if self.write_direct_to_disk:
            db_connection_string = self.db_file_connection_string
        else:
            db_connection_string = self.db_memory_connection_string

        # For in-memory SQLite ("sqlite://"), `database_exists()` semantics are not useful.
        # Always ensure the DB exists, then (by default) always execute the schema script
        # to create required tables (the schema contains DROP TABLE IF EXISTS statements).
        if not sqlalchemy_utils.functions.database_exists(db_connection_string):
            print(f"Creating db:{db_connection_string}")
            sqlalchemy_utils.functions.create_database(db_connection_string)

        engine = sqlalchemy.create_engine(db_connection_string, echo=self.create_engine_echo)
        raw_connection = engine.raw_connection()

        try:
            _apply_bulk_load_pragmas(raw_connection)
            if create_tables:
                _set_page_size(raw_connection)
            print(
                f"Bulk-load PRAGMAs: journal_mode=OFF synchronous=OFF "
                f"page_size=32KiB cache_size={LOAD_CACHE_SIZE_KIB // 1024}MiB "
                f"heap_tables={create_tables} "
                f"batch_files={load_batch_files} batch_bytes={load_batch_bytes // (1024 * 1024)}MiB"
            )
            if create_tables:
                _create_tables(raw_connection)

            urls = self.find_session_sql_files()
            if ending_nfile == 0:
                ending_nfile = len(urls)
            filtered_urls = urls[starting_nfile:ending_nfile]
            file_sizes = [sql_file.stat().st_size for sql_file in filtered_urls]
            total_sql_bytes = sum(file_sizes)
            print(
                f"SQL payload: {len(filtered_urls):,} files, "
                f"{total_sql_bytes / 1e9:.2f} GB. Later years are much larger "
                f"per file; watch MB/s, not files/s."
            )

            total_scripts_executed = 0
            start_time = time.time()
            canceled = False
            batch_files: list[pathlib.Path] = []
            batch_scripts: list[str] = []
            batch_bytes = 0
            batch_limit = max(1, int(load_batch_files))
            bytes_limit = max(1, int(load_batch_bytes))
            done_bytes = 0
            last_mark_n = 0
            last_mark_t = start_time
            last_mark_bytes = 0

            for nfile, sql_file in enumerate(filtered_urls):
                if nfile % 1000 == 0:
                    now = time.time()
                    elapsed = max(now - start_time, 1e-6)
                    interval_t = max(now - last_mark_t, 1e-6)
                    interval_n = nfile - last_mark_n
                    interval_b = done_bytes - last_mark_bytes
                    interval_mbs = (interval_b / 1e6) / interval_t
                    cum_mbs = (done_bytes / 1e6) / elapsed
                    remain_b = max(total_sql_bytes - done_bytes, 0)
                    eta_s = remain_b / (interval_b / interval_t) if interval_b else 0
                    print(
                        f"Executing SQL script ({nfile}/{len(filtered_urls)}): "
                        f"total_time:{round(elapsed, 1)} "
                        f"last1000:{interval_n / interval_t:.1f}/s {interval_b / 1e6:.1f}MB {interval_mbs:.1f}MB/s "
                        f"cum:{nfile / elapsed:.1f}/s {done_bytes / 1e9:.2f}GB {cum_mbs:.1f}MB/s "
                        f"eta:{eta_s / 60:.0f}m "
                        f"file:{sql_file.as_posix()}",
                        flush=True,
                    )
                    last_mark_n = nfile
                    last_mark_t = now
                    last_mark_bytes = done_bytes

                try:
                    sql_script = sql_file.read_text(encoding="utf-8")
                    if create_tables:
                        sql_script = _strip_upserts(sql_script)
                except KeyboardInterrupt:
                    print(f"KeyboardInterrupt while processing file:{sql_file.as_posix()}")
                    canceled = True
                    break
                except Exception as e:
                    _report_bad_sql(sql_file, e)
                    continue

                batch_files.append(sql_file)
                batch_scripts.append(sql_script)
                batch_bytes += len(sql_script)
                done_bytes += file_sizes[nfile]
                if len(batch_scripts) >= batch_limit or batch_bytes >= bytes_limit:
                    try:
                        total_scripts_executed += _flush_batch(batch_files, batch_scripts)
                    except KeyboardInterrupt:
                        print(f"KeyboardInterrupt while processing file:{sql_file.as_posix()}")
                        canceled = True
                        batch_files = []
                        batch_scripts = []
                        break
                    batch_files = []
                    batch_scripts = []
                    batch_bytes = 0

            if not canceled and batch_scripts:
                try:
                    total_scripts_executed += _flush_batch(batch_files, batch_scripts)
                except KeyboardInterrupt:
                    print("KeyboardInterrupt while flushing final batch")
                    canceled = True

            elapsed = max(time.time() - start_time, 1e-6)
            print(
                f"SQL scripts executed ({total_scripts_executed}/{len(filtered_urls)}/{len(urls)}): "
                f"total changes:{raw_connection.total_changes} total time:{round(elapsed, 2)}: "
                f"avg script execution time:{round(elapsed / max(1, total_scripts_executed), 3)} "
                f"rate:{total_scripts_executed / elapsed:.1f}/s"
            )

            if not canceled and create_tables:
                print("Building unique indexes after heap load...")
                _finalize_unique_indexes(raw_connection)

            if not canceled and perform_integrity_checks:
                # Warning: these can take a long time on large DBs.
                print("Performing quick_check")
                raw_connection.execute("PRAGMA quick_check;")
                print("Performing foreign_key_check")
                raw_connection.execute("PRAGMA foreign_key_check;")
                print("Performing integrity_check")
                raw_connection.execute("PRAGMA integrity_check;")

            if not canceled and self.write_direct_to_disk:
                print("Restoring WAL / normal synchronous after bulk load")
                _restore_runtime_pragmas(raw_connection)

            if not canceled and not self.write_direct_to_disk:
                print(f"Writing memory db to file:{self.db_file_connection_string}")
                engine_file = sqlalchemy.create_engine(self.db_file_connection_string)
                raw_connection_file = engine_file.raw_connection()
                raw_connection.backup(raw_connection_file.driver_connection)
                _restore_runtime_pragmas(raw_connection_file)
                raw_connection_file.close()
                engine_file.dispose()

            return total_scripts_executed
        finally:
            raw_connection.close()
            engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert ACBL tournament session JSON to SQL and load into SQLite"
    )
    parser.add_argument(
        "--sessions-dir",
        default="tournaments/sessions",
        help=f"Input subdirectory relative to {acblPath} (default: tournaments/sessions)",
    )
    parser.add_argument(
        "--schema-file",
        type=str,
        default="acbl_tournament_sessions_schema.sql",
        help="Schema SQL filename (default: acbl_tournament_sessions_schema.sql in script directory)",
    )
    parser.add_argument(
        "--db-file",
        default="acbl_tournament_results.sqlite",
        help="SQLite database filename placed in acblPath (default: acbl_tournament_results.sqlite)",
    )
    parser.add_argument(
        "--write-direct-to-disk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write into the DB file during load (default: True). "
            "Use --no-write-direct-to-disk for an in-memory DB then backup "
            "(slows down as the DB grows past a few GB)."
        ),
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        default=False,
        help="SQLAlchemy echo (default: False)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting file index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=0,
        help="Ending file index, 0 means all (default: 0)",
    )
    parser.add_argument(
        "--no-generate-sql",
        action="store_true",
        help="Skip generating `.session.sql` files (default: generate)",
    )
    parser.add_argument(
        "--no-load-sql",
        action="store_true",
        help="Skip loading `.session.sql` into SQLite (default: load)",
    )
    parser.add_argument(
        "--skip-existing-sql",
        action="store_true",
        default=True,
        help="Skip generating SQL if output `.sql` exists and is newer than `.json` (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing-sql",
        action="store_false",
        dest="skip_existing_sql",
        help="Do not skip SQL generation when output `.sql` exists",
    )
    parser.add_argument(
        "--rebuild-sql",
        action="store_true",
        default=False,
        help="Delete all output `.sql` files before generation (default: False)",
    )
    parser.add_argument(
        "--delete-bad-json",
        action="store_true",
        default=True,
        help="Delete invalid/unreadable JSON files during generation (default: True)",
    )
    parser.add_argument(
        "--no-delete-bad-json",
        action="store_false",
        dest="delete_bad_json",
        help="Keep invalid/unreadable JSON files during generation",
    )
    parser.add_argument(
        "--recreate-db",
        action="store_true",
        default=True,
        help="Delete and recreate DB file if it exists (default: True)",
    )
    parser.add_argument(
        "--no-recreate-db",
        action="store_false",
        dest="recreate_db",
        help="Do not delete DB file if it exists",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        default=True,
        help="Execute schema SQL before loading scripts (default: True)",
    )
    parser.add_argument(
        "--no-create-tables",
        action="store_false",
        dest="create_tables",
        help="Do not execute schema SQL before loading scripts",
    )
    parser.add_argument(
        "--integrity-checks",
        action="store_true",
        default=False,
        help="Run PRAGMA integrity checks at end (slow; default: False)",
    )
    parser.add_argument(
        "--delete-bad-sql",
        action="store_true",
        default=True,
        help="Delete SQL scripts that fail to execute (default: True)",
    )
    parser.add_argument(
        "--no-delete-bad-sql",
        action="store_false",
        dest="delete_bad_sql",
        help="Keep SQL scripts that fail to execute",
    )
    parser.add_argument(
        "--load-batch-size",
        type=int,
        default=DEFAULT_LOAD_BATCH_FILES,
        help=(
            "Session SQL files per SQLite transaction "
            f"(default: {DEFAULT_LOAD_BATCH_FILES})"
        ),
    )

    args = parser.parse_args()

    sessions_dir = acblPath.joinpath(args.sessions_dir)
    schema_file = pathlib.Path(__file__).with_name(args.schema_file)
    db_file = acblPath.joinpath(args.db_file)

    print("=" * 70)
    print("ACBL Tournament Sessions JSON -> SQLite")
    print("=" * 70)
    print(f"Sessions dir: {sessions_dir}")
    print(f"Schema file:  {schema_file}")
    print(f"DB file:      {db_file}")
    print(f"Write direct: {args.write_direct_to_disk}")
    print()

    builder = ACBLTournamentSessionSqliteBuilder(
        sessions_dir=sessions_dir,
        schema_file=schema_file,
        db_file=db_file,
        write_direct_to_disk=args.write_direct_to_disk,
        create_engine_echo=args.echo,
    )

    if not args.no_generate_sql:
        builder.generate_sql_files(
            starting_nfile=args.start,
            ending_nfile=args.end,
            skip_existing_files=args.skip_existing_sql,
            initially_delete_all_output_files=args.rebuild_sql,
            delete_bad_json=args.delete_bad_json,
        )

    if not args.no_load_sql:
        builder.load_sql_files_into_db(
            starting_nfile=args.start,
            ending_nfile=args.end,
            recreate_db=args.recreate_db,
            create_tables=args.create_tables,
            perform_integrity_checks=args.integrity_checks,
            delete_bad_sql=args.delete_bad_sql,
            load_batch_files=args.load_batch_size,
        )

    print()
    print("Done!")
    return 0


if __name__ == "__main__":
    from mlBridge import print_started, print_ended
    program_start_time = print_started()
    try:
        raise SystemExit(main())
    finally:
        print_ended(program_start_time)

