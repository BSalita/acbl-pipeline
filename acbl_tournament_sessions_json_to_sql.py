"""
ACBL Tournament Sessions JSON -> Parquet (-> SQL -> SQLite on request)

Default (since 2026-10-24): `tournaments/sessions/*.session.json` -> one
Parquet file per table under `tournament_results_parquet/` (all Utf8, like
club_results_parquet/). Stages 2a/2b read those tables directly.

With --build-sqlite it also runs the original path below, which converts the
JSON files to per-session `.session.sql` files and loads them into
`acbl_tournament_results.sqlite`. That took 23m for 150,000 session files.

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

import polars as pl
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


# ---------------------------------------------------------------------------
# JSON -> Parquet (default since 2026-10-24; SQLite only with --build-sqlite)
# ---------------------------------------------------------------------------
DEFAULT_PARQUET_SUBDIR = "tournament_results_parquet"
DEFAULT_SHARD_SUBDIR = "tournament_results_parquet_shards"
DEFAULT_PARQUET_WORKERS = 32
DEFAULT_PARQUET_FLUSH_EVERY = 1000
_ORDER_COL = "__load_order"
_CREATE_TABLE_RE = re.compile(r'CREATE TABLE\s+"?([A-Za-z_][A-Za-z0-9_]*)"?\s*\(', re.IGNORECASE)
_COLUMN_RE = re.compile(r'^\s*"([^"]+)"\s+[A-Za-z]+')


def parse_schema_columns(schema_file: pathlib.Path) -> dict[str, tuple[frozenset, frozenset]]:
    """{table: (columns, not_null_columns)} from the tournament schema SQL."""
    out: dict[str, tuple[frozenset, frozenset]] = {}
    table = None
    cols: set[str] = set()
    not_null: set[str] = set()
    for line in schema_file.read_text(encoding="utf-8").splitlines():
        m = _CREATE_TABLE_RE.search(line)
        if m:
            table, cols, not_null = m.group(1), set(), set()
            continue
        if table is None:
            continue
        if line.strip().startswith(");"):
            out[table] = (frozenset(cols), frozenset(not_null))
            table = None
            continue
        c = _COLUMN_RE.match(line)
        if c:
            cols.add(c.group(1))
            if "NOT NULL" in line.upper():
                not_null.add(c.group(1))
    return out


def session_rows_from_json(json_file: pathlib.Path) -> tuple[Optional[dict], Optional[str]]:
    """Rows per table for one session file, using the same walk and the same
    accept/skip rules as generate_sql_files(). Returns (rows, None) or
    (None, reason)."""
    import mlBridge.mlBridgeAcblLib as mlBridgeLib  # type: ignore
    from mlBridge.mlBridgeLib import tables_to_db_rows  # type: ignore

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data_json = json.load(f)
    except Exception as e:
        return None, f"unreadable JSON ({type(e).__name__}: {e})"
    tables = defaultdict(lambda: defaultdict(dict))
    if "id" in data_json:
        mlBridgeLib.json_to_sql_walk(tables, "session", "", "", data_json, ["id"])
    elif "session_id" in data_json:
        if isinstance(data_json.get("session"), dict) or len(data_json.get("session", [])) == 0:
            return None, "session is dict or empty"
        mlBridgeLib.json_to_sql_walk(tables, "session", "", "", data_json, ["id"])
    else:
        return None, "missing both id and session"
    return tables_to_db_rows(tables), None


def schema_violation(rows: dict, schema: dict) -> Optional[str]:
    """Why SQLite would reject this session's script, else None.

    The SQLite load runs one script per session; any unknown table, unknown
    column, or NULL in a NOT NULL column fails the whole script and the
    session is dropped. The Parquet path drops the same sessions.
    """
    for table, table_rows in rows.items():
        if table not in schema:
            return f"no such table: {table}"
        cols, not_null = schema[table]
        for row in table_rows:
            extra = set(row) - cols
            if extra:
                return f"table {table} has no column(s) {sorted(extra)[:3]}"
            for col in not_null:
                if row.get(col) is None:
                    return f"NULL in NOT NULL column {table}.{col}"
    return None


def _parquet_worker(
    indexed_paths: list[tuple[int, str]],
    shard_dir: str,
    batch_id: int,
    schema_file: str,
) -> tuple[int, int, list[str]]:
    """Process one batch of session files into per-table Parquet shards.

    Each row carries __load_order = (file index, row index) so the merge can
    keep the first row per id, the same row the SQLite load keeps.
    """
    from acbl_club_json_to_parquet_sql import _rows_to_utf8_df

    schema = parse_schema_columns(pathlib.Path(schema_file))
    buffers: dict[str, list[dict]] = defaultdict(list)
    ok = 0
    skipped: list[str] = []
    for file_index, path_str in indexed_paths:
        path = pathlib.Path(path_str)
        rows, reason = session_rows_from_json(path)
        if rows is not None:
            reason = schema_violation(rows, schema)
        if reason:
            skipped.append(f"{path.as_posix()}: {reason}")
            continue
        ok += 1
        for table, table_rows in rows.items():
            for row_index, row in enumerate(table_rows):
                buffers[table].append({**row, _ORDER_COL: file_index * 1_000_000 + row_index})
    shard_path = pathlib.Path(shard_dir)
    shard_path.mkdir(parents=True, exist_ok=True)
    for table, table_rows in buffers.items():
        df = _rows_to_utf8_df(table_rows).with_columns(pl.col(_ORDER_COL).cast(pl.Int64))
        df.write_parquet(shard_path.joinpath(f"{table}__{batch_id:06d}.parquet"), compression="zstd")
    return ok, len(skipped), skipped


def build_parquet_from_session_json(
    json_files: list[pathlib.Path],
    parquet_dir: pathlib.Path,
    shard_dir: pathlib.Path,
    schema_file: pathlib.Path,
    workers: int = DEFAULT_PARQUET_WORKERS,
    flush_every: int = DEFAULT_PARQUET_FLUSH_EVERY,
    keep_shards: bool = False,
) -> list[str]:
    """Session JSON -> per-table Parquet (all Utf8, like club_results_parquet/).

    Duplicate ids keep the first row in sorted file order, matching the
    SQLite heap load + "keep MIN(rowid)" dedup in _finalize_unique_indexes.
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from acbl_club_json_to_parquet_sql import _batches, _max_process_workers, _progress_line

    t0 = time.time()
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    indexed = [(i, str(p)) for i, p in enumerate(json_files)]
    n_files = len(indexed)
    workers = max(1, min(workers, _max_process_workers(), n_files or 1))
    batches = _batches(indexed, max(1, flush_every))
    print(f"[parquet] session files={n_files:,} workers={workers} batches={len(batches):,}")

    total_ok = total_skipped = files_done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_parquet_worker, batch, str(shard_dir), i, str(schema_file)): len(batch)
            for i, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            ok, n_skipped, skipped = fut.result()
            total_ok += ok
            total_skipped += n_skipped
            files_done += futures[fut]
            for msg in skipped[:5]:
                print(f"  SKIP {msg}", flush=True)
            print(_progress_line("parquet", files_done, n_files, t0,
                                 extra=f"ok={total_ok:,} skipped={total_skipped:,}"), flush=True)

    tables = sorted({p.name.split("__", 1)[0] for p in shard_dir.glob("*.parquet")})
    for table in tables:
        t_table = time.time()
        parts = sorted(shard_dir.glob(f"{table}__*.parquet"))
        df = (
            pl.concat([pl.read_parquet(p) for p in parts], how="diagonal_relaxed")
            .sort(_ORDER_COL)
        )
        if "id" in df.columns:
            df = df.unique(subset=["id"], keep="first", maintain_order=True)
        df = df.drop(_ORDER_COL)
        out = parquet_dir.joinpath(f"{table}.parquet")
        df.write_parquet(out, compression="zstd")
        print(f"[parquet] {out.name}: rows={df.height:,} cols={df.width} "
              f"in {time.time() - t_table:.1f}s", flush=True)
        del df

    if not keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
    print(f"[parquet] done: ok={total_ok:,} skipped={total_skipped:,} tables={len(tables)} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return tables


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
        description=(
            "Convert ACBL tournament session JSON to Parquet tables "
            "(default) and optionally to SQL + SQLite (--build-sqlite)"
        )
    )
    parser.add_argument(
        "--parquet-dir",
        default=DEFAULT_PARQUET_SUBDIR,
        help=f"Parquet output subdirectory relative to {acblPath} (default: {DEFAULT_PARQUET_SUBDIR})",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Do not build the Parquet tables",
    )
    parser.add_argument(
        "--build-sqlite",
        action="store_true",
        help="Also generate .session.sql files and build the SQLite DB (off by default)",
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Accepted for old command lines; skipping SQLite is now the default.",
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_PARQUET_WORKERS,
                        help=f"Parquet worker processes (default: {DEFAULT_PARQUET_WORKERS})")
    parser.add_argument("--flush-every", type=int, default=DEFAULT_PARQUET_FLUSH_EVERY,
                        help=f"Session files per Parquet shard batch (default: {DEFAULT_PARQUET_FLUSH_EVERY})")
    parser.add_argument("--keep-shards", action="store_true",
                        help="Keep the intermediate Parquet shards")
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
    if args.build_sqlite and args.skip_sqlite:
        parser.error("--build-sqlite and --skip-sqlite contradict each other")

    sessions_dir = acblPath.joinpath(args.sessions_dir)
    parquet_dir = acblPath.joinpath(args.parquet_dir)
    schema_file = pathlib.Path(__file__).with_name(args.schema_file)
    db_file = acblPath.joinpath(args.db_file)

    print("=" * 70)
    print("ACBL Tournament Sessions JSON -> Parquet" + (" + SQLite" if args.build_sqlite else ""))
    print("=" * 70)
    print(f"Sessions dir: {sessions_dir}")
    print(f"Schema file:  {schema_file}")
    print(f"Parquet dir:  {parquet_dir}{' (skipped)' if args.skip_parquet else ''}")
    print(f"DB file:      {db_file}{'' if args.build_sqlite else ' (skipped; pass --build-sqlite)'}")
    print()

    builder = ACBLTournamentSessionSqliteBuilder(
        sessions_dir=sessions_dir,
        schema_file=schema_file,
        db_file=db_file,
        write_direct_to_disk=args.write_direct_to_disk,
        create_engine_echo=args.echo,
    )

    if not args.skip_parquet:
        json_files = builder.find_session_json_files()
        if args.end:
            json_files = json_files[args.start:args.end]
        elif args.start:
            json_files = json_files[args.start:]
        build_parquet_from_session_json(
            json_files,
            parquet_dir,
            acblPath.joinpath(DEFAULT_SHARD_SUBDIR),
            schema_file,
            workers=args.workers,
            flush_every=args.flush_every,
            keep_shards=args.keep_shards,
        )

    if not args.build_sqlite:
        print()
        print("[sqlite] skipped (default; pass --build-sqlite to build it)")
        print("Done!")
        return 0

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

