"""
Club results ingest option F: parallel JSON -> Parquet (-> SQLite on request).

Uses the same json_to_sql_walk extraction as the legacy .data.sql path, so
row shapes match CreateSqlFile / acbl_club_results_schema.sql.

Since 2026-10-24 the SQLite build is off by default: stages 2a/2b read the
Parquet tables directly (acbl_compare_sqlite_vs_parquet.py showed identical
query results), and the build took ~7h20m. Pass --build-sqlite to also
write acbl_club_results.sqlite.

Pipeline:
  1) ProcessPool over *.data.json in batches of --flush-every files
     (each batch writes Parquet shards then frees buffers)
  2) Stream-merge shards -> one Parquet file per table under club_results_parquet/
  3) Only with --build-sqlite: apply acbl_club_results_schema.sql and bulk
     INSERT from Parquet (batched)

Usage:
  python acbl_club_json_to_parquet_sql.py
  python acbl_club_json_to_parquet_sql.py --workers 32 --flush-every 1000
  python acbl_club_json_to_parquet_sql.py --build-sqlite                 # Parquet + SQLite
  python acbl_club_json_to_parquet_sql.py --skip-parquet --build-sqlite  # SQLite from existing Parquet

Resource notes (64-core / 128-thread, 512 GB RAM):
  - Windows ProcessPoolExecutor caps max_workers at 61 (~50% of logical CPUs).
  - Default workers=32 and flush-every=1000 bound peak RAM to roughly
    workers x flush-every files of Python row dicts (not the whole corpus).
  - Without flush-every, a prior run filled RAM/pagefile; do not set
    flush-every to 0 unless you have huge spare RAM.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sqlite3
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import polars as pl

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")

DEFAULT_SCHEMA = pathlib.Path(__file__).resolve().parent.joinpath("acbl_club_results_schema.sql")
DEFAULT_PARQUET_DIR = acblPath.joinpath("club_results_parquet")
DEFAULT_SHARD_DIR = acblPath.joinpath("club_results_parquet_shards")
DEFAULT_DB = acblPath.joinpath("acbl_club_results.sqlite")

# Conservative defaults after a pagefile thrash with 61 workers holding full chunks.
DEFAULT_WORKERS = 32
DEFAULT_FLUSH_EVERY = 1000


def _extract_tables_from_json(json_path: pathlib.Path) -> Dict[str, List[dict]]:
    from mlBridge.mlBridgeLib import json_to_sql_walk, tables_to_db_rows

    with open(json_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)
    tables = defaultdict(lambda: defaultdict(dict))
    json_to_sql_walk(tables, "events", "", "", data_json, ["id"])
    return tables_to_db_rows(tables)


def _rows_to_utf8_df(rows: List[dict]) -> pl.DataFrame:
    """
    Build a DataFrame with every non-null value as Utf8.

    Club JSON columns are heterogeneous across events (null/int early, then an
    ISO timestamp string, etc.). Polars schema inference then fails mid-batch.
    Utf8 Parquet + SQLite column affinity / explicit coerce on load matches the
    legacy SQL-text path, which also stringifies freely.
    """
    if not rows:
        return pl.DataFrame()
    normalized = [
        {k: (None if v is None else str(v)) for k, v in row.items()} for row in rows
    ]
    return pl.DataFrame(normalized, infer_schema_length=len(normalized))


def _write_table_shards(
    buffers: Dict[str, List[dict]],
    shard_path: pathlib.Path,
    batch_id: int,
) -> int:
    """Flush in-memory row buffers to Parquet shards. Returns tables written."""
    n_tables = 0
    for table, rows in buffers.items():
        if not rows:
            continue
        df = _rows_to_utf8_df(rows)
        out = shard_path.joinpath(f"{table}__{batch_id:06d}.parquet")
        df.write_parquet(out, compression="zstd")
        n_tables += 1
    buffers.clear()
    return n_tables


def _coerce_sqlite_value(
    declared_type: str,
    value: Any,
    *,
    not_null: bool = False,
) -> Any:
    """Coerce a Utf8 Parquet cell back to a Python value for SQLite affinity.

    Matches legacy .data.sql inserts: keep non-numeric strings as strings even
    when the schema declares INT (composite ids like files.id '1696-1'). Never
    invent 0 / defaults on parse failure — that collapses distinct PKs and
    trips UNIQUE. NOT NULL columns never become Python None.
    """
    t = (declared_type or "").upper()
    is_numeric = any(x in t for x in ("INT", "REAL", "FLOA", "DOUB", "BOOL"))

    if value is None:
        # NOT NULL: legacy SQL text emitted ''; keep that for any affinity.
        if not_null:
            return ""
        return None
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    s = str(value)
    if s == "":
        # Do NOT turn '' into NULL for VARCHAR — e.g. events.name NOT NULL.
        if not_null:
            return ""
        if is_numeric:
            return None
        return ""
    if "INT" in t or t == "BOOL":
        low = s.lower()
        if low in ("true", "false"):
            return 1 if low == "true" else 0
        # Strict integer only — avoid int(float(...)) which truncates '1.15'
        # and never map '1696-1' / '4/6' / '31=' to a synthetic int.
        if s.isdigit() or (s[0] in "+-" and s[1:].isdigit()):
            return int(s)
        return s
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        try:
            return float(s)
        except (TypeError, ValueError):
            return s
    return s


def _coerced_key_utf8_expr(
    col: str,
    declared_type: str,
    *,
    keep_sql_nulls: bool = False,
) -> pl.Expr:
    """
    Polars expression mirroring _coerce_sqlite_value for uniqueness checks.

    Result is Utf8 for non-null values (ints normalized to decimal form).
    When keep_sql_nulls=True (nullable UNIQUE columns), Parquet null / empty
    numeric cells stay null so SQLite's \"multiple NULLs are OK\" semantics
    are preserved. When False (PRIMARY KEY / NOT NULL), null/empty → ''.
    """
    t = (declared_type or "").upper()
    s = pl.col(col).cast(pl.Utf8)
    is_null = s.is_null()
    is_empty = s == ""
    if "INT" in t or t == "BOOL":
        low = s.str.to_lowercase()
        int_like = s.str.contains(r"^[+-]?\d+$")
        coerced = (
            pl.when(low == "true")
            .then(pl.lit("1"))
            .when(low == "false")
            .then(pl.lit("0"))
            .when(int_like)
            .then(s.cast(pl.Int64, strict=False).cast(pl.Utf8))
            .otherwise(s)
        )
        if keep_sql_nulls:
            # nullable numeric: null/'' → SQL NULL
            return (
                pl.when(is_null | is_empty)
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(coerced)
                .alias(col)
            )
        return (
            pl.when(is_null | is_empty)
            .then(pl.lit(""))
            .otherwise(coerced)
            .alias(col)
        )
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        as_f = s.cast(pl.Float64, strict=False)
        coerced = (
            pl.when(as_f.is_not_null())
            .then(as_f.cast(pl.Utf8))
            .otherwise(s)
        )
        if keep_sql_nulls:
            return (
                pl.when(is_null | is_empty)
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(coerced)
                .alias(col)
            )
        return (
            pl.when(is_null | is_empty)
            .then(pl.lit(""))
            .otherwise(coerced)
            .alias(col)
        )
    # text
    if keep_sql_nulls:
        return pl.when(is_null).then(pl.lit(None, dtype=pl.Utf8)).otherwise(s).alias(col)
    return pl.when(is_null | is_empty).then(pl.lit("")).otherwise(s).alias(col)


def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return [r[1] for r in rows]


def _table_column_meta(
    conn: sqlite3.Connection, table: str
) -> List[Tuple[str, str, bool]]:
    """Return [(name, declared_type, not_null), ...] in table order."""
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    # cid, name, type, notnull, dflt_value, pk
    return [(r[1], r[2] or "", bool(r[3])) for r in rows]


def _table_column_types(conn: sqlite3.Connection, table: str) -> List[Tuple[str, str]]:
    """Return [(name, declared_type), ...] in table order."""
    return [(n, t) for n, t, _ in _table_column_meta(conn, table)]


def _table_pk_meta(
    conn: sqlite3.Connection, table: str
) -> List[Tuple[str, str, bool]]:
    """Return PK columns as [(name, declared_type, not_null), ...] in PK order."""
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    pks = sorted(
        ((r[5], r[1], r[2] or "", bool(r[3])) for r in rows if r[5]),
        key=lambda x: x[0],
    )
    return [(name, typ, nn) for _, name, typ, nn in pks]


def _unique_column_sets(
    conn: sqlite3.Connection, table: str
) -> List[Tuple[str, List[Tuple[str, str, bool]]]]:
    """
    Return [(label, [(col, declared_type, not_null), ...]), ...] that must be
    unique (PRIMARY KEY and UNIQUE indexes).
    """
    meta_by_col = {n: (t, nn) for n, t, nn in _table_column_meta(conn, table)}
    out: List[Tuple[str, List[Tuple[str, str, bool]]]] = []
    pk = _table_pk_meta(conn, table)
    if pk:
        out.append(("PRIMARY KEY", [(n, t, nn) for n, t, nn in pk]))
    for idx in conn.execute(f'PRAGMA index_list("{table}")').fetchall():
        # seq, name, unique, origin, partial
        if not idx[2]:
            continue
        origin = (idx[3] or "").lower() if len(idx) > 3 else ""
        if origin == "pk":
            continue  # already covered
        cols = [
            r[2]
            for r in conn.execute(f'PRAGMA index_info("{idx[1]}")').fetchall()
        ]
        if cols and all(c in meta_by_col for c in cols):
            out.append(
                (
                    f"UNIQUE:{idx[1]}",
                    [(c, meta_by_col[c][0], meta_by_col[c][1]) for c in cols],
                )
            )
    return out


class ParquetSqlitePreflightError(RuntimeError):
    """Raised when Parquet would violate SQLite constraints after coerce."""


def _preflight_table_parquet(
    table: str,
    pq_path: pathlib.Path,
    conn: sqlite3.Connection,
) -> None:
    """
    Fail fast if post-coerce values would trip NOT NULL / UNIQUE / PRIMARY KEY.

    Runs before any bulk INSERT so a bad coerce cannot waste hours mid-load.
    """
    col_meta = _table_column_meta(conn, table)
    if not col_meta:
        return
    lf = pl.scan_parquet(str(pq_path))
    pq_cols = set(lf.collect_schema().names())
    n = int(lf.select(pl.len()).collect().item())
    if n == 0:
        return

    for label, cols in _unique_column_sets(conn, table):
        present = [(c, t, nn) for c, t, nn in cols if c in pq_cols]
        if len(present) != len(cols):
            missing = [c for c, _, _ in cols if c not in pq_cols]
            raise ParquetSqlitePreflightError(
                f"{table}: {label} columns missing from Parquet: {missing}"
            )
        # PK / NOT NULL unique: null→''. Nullable UNIQUE: keep SQL NULLs
        # (SQLite allows multiple NULLs in a UNIQUE column).
        is_pk = label == "PRIMARY KEY"
        key_exprs = [
            _coerced_key_utf8_expr(
                c, t, keep_sql_nulls=(not is_pk and not nn)
            )
            for c, t, nn in present
        ]
        if len(key_exprs) == 1:
            key = key_exprs[0].alias("_uk")
        else:
            # Composite UNIQUE: SQLite treats a row as distinct if any key
            # part is NULL. Encode null parts as a per-row sentinel via
            # filtering to rows where all parts are non-null.
            key = pl.concat_str(
                [
                    pl.when(e.is_null())
                    .then(pl.lit("\x00"))
                    .otherwise(e)
                    .alias(f"_k{i}")
                    for i, e in enumerate(key_exprs)
                ],
                separator="\x1f",
            ).alias("_uk")
            # Only rows with all-non-null keys can collide.
            all_present = pl.all_horizontal(
                *[e.is_not_null() for e in key_exprs]
            )
            dup_keys = (
                lf.filter(all_present)
                .select(key)
                .group_by("_uk")
                .agg(pl.len().alias("c"))
                .filter(pl.col("c") > 1)
                .sort("c", descending=True)
                .head(5)
                .collect()
            )
            if dup_keys.height:
                samples = [f"{r[0]!r} x{r[1]}" for r in dup_keys.iter_rows()]
                raise ParquetSqlitePreflightError(
                    f"{table}: {label} not unique after coerce. "
                    f"Top collisions: {samples}. "
                    f"Refusing to load — fix coerce or source data."
                )
            continue

        # Single-column UNIQUE / PK
        if is_pk or present[0][2]:
            # NOT NULL: every row participates; null/empty coerced to ''.
            stats = lf.select(
                pl.len().alias("n"),
                key.n_unique().alias("nuniq"),
            ).collect()
            n_rows = int(stats["n"][0])
            n_uniq = int(stats["nuniq"][0])
            if n_uniq != n_rows:
                dup_keys = (
                    lf.select(key)
                    .group_by("_uk")
                    .agg(pl.len().alias("c"))
                    .filter(pl.col("c") > 1)
                    .sort("c", descending=True)
                    .head(5)
                    .collect()
                )
                samples = [f"{r[0]!r} x{r[1]}" for r in dup_keys.iter_rows()]
                raise ParquetSqlitePreflightError(
                    f"{table}: {label} not unique after coerce "
                    f"(rows={n_rows:,}, unique={n_uniq:,}). "
                    f"Top collisions: {samples}. "
                    f"Refusing to load — fix coerce or source data."
                )
        else:
            # Nullable UNIQUE: only non-NULL values must be unique.
            non_null = lf.select(key).filter(pl.col("_uk").is_not_null())
            stats = non_null.select(
                pl.len().alias("n"),
                pl.col("_uk").n_unique().alias("nuniq"),
            ).collect()
            n_nn = int(stats["n"][0])
            n_uniq = int(stats["nuniq"][0])
            if n_nn and n_uniq != n_nn:
                dup_keys = (
                    non_null.group_by("_uk")
                    .agg(pl.len().alias("c"))
                    .filter(pl.col("c") > 1)
                    .sort("c", descending=True)
                    .head(5)
                    .collect()
                )
                samples = [f"{r[0]!r} x{r[1]}" for r in dup_keys.iter_rows()]
                raise ParquetSqlitePreflightError(
                    f"{table}: {label} not unique after coerce among non-NULLs "
                    f"(non_null={n_nn:,}, unique={n_uniq:,}). "
                    f"Top collisions: {samples}. "
                    f"Refusing to load — fix coerce or source data."
                )


def preflight_parquet_dir(
    parquet_dir: pathlib.Path,
    schema_path: pathlib.Path,
) -> None:
    """Validate all table Parquets against schema constraints (in-memory)."""
    schema_text = _schema_sql_without_wal(schema_path.read_text(encoding="utf-8"))
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema_text)
        existing = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet tables in {parquet_dir}")
        t0 = time.time()
        print(
            f"[sqlite] preflight {len(parquet_files)} tables "
            f"(PK/UNIQUE after coerce)...",
            flush=True,
        )
        for i, pq in enumerate(parquet_files, 1):
            table = pq.stem
            if table not in existing:
                continue
            print(f"[sqlite] preflight {i}/{len(parquet_files)} {table}", flush=True)
            _preflight_table_parquet(table, pq, conn)
        print(
            f"[sqlite] preflight OK in {_fmt_duration(time.time() - t0)}",
            flush=True,
        )
    finally:
        conn.close()


def _worker_process_batch(
    json_paths: Sequence[str],
    shard_dir: str,
    batch_id: int,
) -> Tuple[int, int, int, List[str]]:
    """
    Process one batch of JSON files (size ~= flush-every); write shards; free RAM.
    Returns (files_ok, files_err, n_files, error_messages).
    """
    shard_path = pathlib.Path(shard_dir)
    shard_path.mkdir(parents=True, exist_ok=True)

    buffers: Dict[str, List[dict]] = defaultdict(list)
    ok = 0
    err = 0
    errors: List[str] = []

    for path_str in json_paths:
        path = pathlib.Path(path_str)
        try:
            table_rows = _extract_tables_from_json(path)
            for table, rows in table_rows.items():
                if rows:
                    buffers[table].extend(rows)
            ok += 1
        except Exception as e:
            err += 1
            if len(errors) < 5:
                errors.append(f"{path.as_posix()}: {type(e).__name__}: {e}")

    _write_table_shards(buffers, shard_path, batch_id)
    return ok, err, len(json_paths), errors


def _max_process_workers() -> int:
    """Windows ProcessPoolExecutor rejects max_workers > 61."""
    cpu = os.cpu_count() or 8
    soft = max(1, cpu - 2)
    if sys.platform == "win32":
        return min(soft, 61)
    return soft


def _default_workers() -> int:
    return min(DEFAULT_WORKERS, _max_process_workers())


def _batches(items: List[Any], batch_size: int) -> List[List[Any]]:
    if batch_size <= 0:
        return [items] if items else []
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _progress_line(
    phase: str,
    done: int,
    total: int,
    t0: float,
    extra: str = "",
) -> str:
    elapsed = time.time() - t0
    pct = (100.0 * done / total) if total else 100.0
    if done > 0 and elapsed > 0:
        rate = done / elapsed
        eta = (total - done) / rate if rate > 0 else 0.0
        rate_s = f"{rate:.1f}/s"
        eta_s = _fmt_duration(eta)
    else:
        rate_s = "?/s"
        eta_s = "?"
    base = (
        f"[{phase}] {done:,}/{total:,} ({pct:5.1f}%) "
        f"elapsed={_fmt_duration(elapsed)} rate={rate_s} ETA={eta_s}"
    )
    return f"{base} {extra}".rstrip()


def discover_json_files(club_results_dir: pathlib.Path, limit: int = 0) -> List[pathlib.Path]:
    print(f"[discover] scanning {club_results_dir} for *.data.json ...")
    t0 = time.time()
    files = sorted(club_results_dir.rglob("*/details/*.data.json"))
    if limit > 0:
        files = files[:limit]
    print(f"[discover] found {len(files):,} files in {_fmt_duration(time.time() - t0)}")
    return files


def build_parquet_from_json(
    json_files: Sequence[pathlib.Path],
    parquet_dir: pathlib.Path,
    shard_dir: pathlib.Path,
    workers: int,
    flush_every: int = DEFAULT_FLUSH_EVERY,
    keep_shards: bool = False,
) -> List[str]:
    """
    Parallel JSON -> batch Parquet shards -> merged per-table Parquet.
    Returns sorted list of table names written.
    """
    t0 = time.time()
    if shard_dir.exists():
        print(f"[parquet] clearing old shards: {shard_dir}")
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    paths = [str(p) for p in json_files]
    n_files = len(paths)
    workers = max(1, min(workers, _max_process_workers(), n_files or 1))
    flush_every = max(1, flush_every)
    batches = _batches(paths, flush_every)
    n_batches = len(batches)

    print(
        f"[parquet] JSON files={n_files:,} workers={workers} "
        f"flush_every={flush_every:,} batches={n_batches:,}"
    )
    print(f"[parquet] shard_dir={shard_dir}")
    print(
        f"[parquet] peak RAM ~ workers x flush_every files of row-dicts "
        f"(~{workers * flush_every:,} files in flight)"
    )

    total_ok = 0
    total_err = 0
    files_done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker_process_batch, batch, str(shard_dir), i): i
            for i, batch in enumerate(batches)
        }
        batches_done = 0
        for fut in as_completed(futures):
            batch_id = futures[fut]
            try:
                ok, err, n_in_batch, errors = fut.result()
            except Exception as e:
                print(f"[parquet] BATCH FAIL id={batch_id}: {type(e).__name__}: {e}")
                traceback.print_exc()
                raise
            total_ok += ok
            total_err += err
            files_done += n_in_batch
            batches_done += 1
            print(
                _progress_line(
                    "parquet",
                    files_done,
                    n_files,
                    t0,
                    extra=(
                        f"batches={batches_done}/{n_batches} "
                        f"ok={total_ok:,} err={total_err:,} "
                        f"last_batch={batch_id}"
                    ),
                ),
                flush=True,
            )
            for msg in errors:
                print(f"  ERROR {msg}", flush=True)

    # Merge shards per table (streaming sink when possible)
    shard_files = list(shard_dir.glob("*.parquet"))
    tables = sorted({p.name.split("__", 1)[0] for p in shard_files})
    print(
        f"[parquet] merging {len(shard_files):,} shards -> {len(tables)} tables",
        flush=True,
    )
    t_merge = time.time()
    for ti, table in enumerate(tables, 1):
        parts = sorted(shard_dir.glob(f"{table}__*.parquet"))
        if not parts:
            continue
        t_table = time.time()
        lazy = [pl.scan_parquet(str(p)) for p in parts]
        lf = pl.concat(lazy, how="diagonal_relaxed")
        if "id" in lf.collect_schema().names():
            lf = lf.unique(subset=["id"], keep="last")
        out = parquet_dir.joinpath(f"{table}.parquet")
        # Prefer streaming sink to avoid holding the full table in the parent.
        try:
            lf.sink_parquet(str(out), compression="zstd")
            n = pl.scan_parquet(str(out)).select(pl.len()).collect().item()
        except Exception:
            df = lf.collect()
            df.write_parquet(out, compression="zstd")
            n = df.height
        print(
            f"[parquet] merge {ti}/{len(tables)} {out.name}: rows={n:,} "
            f"shards={len(parts)} in {_fmt_duration(time.time() - t_table)}",
            flush=True,
        )

    if not keep_shards:
        print(f"[parquet] removing shard dir {shard_dir}", flush=True)
        shutil.rmtree(shard_dir, ignore_errors=True)

    print(
        _progress_line("parquet", n_files, n_files, t0)
        + f" ok={total_ok:,} err={total_err:,} tables={len(tables)} "
        f"merge={_fmt_duration(time.time() - t_merge)}",
        flush=True,
    )
    return tables


def _schema_sql_without_wal(schema_text: str) -> str:
    lines = []
    for line in schema_text.splitlines():
        if line.strip().upper().startswith("PRAGMA JOURNAL_MODE"):
            continue
        lines.append(line)
    return "\n".join(lines)


def load_sqlite_from_parquet(
    parquet_dir: pathlib.Path,
    db_path: pathlib.Path,
    schema_path: pathlib.Path,
    batch_size: int = 50_000,
) -> None:
    """Recreate SQLite DB from schema, then bulk INSERT each table Parquet."""
    t0 = time.time()
    if not schema_path.is_file():
        raise FileNotFoundError(schema_path)

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet tables in {parquet_dir}")

    # Fail before creating a multi‑GB .building DB if coerce would collapse PKs.
    preflight_parquet_dir(parquet_dir, schema_path)

    if db_path.exists():
        print(f"[sqlite] removing existing {db_path}", flush=True)
        db_path.unlink()

    schema_text = _schema_sql_without_wal(schema_path.read_text(encoding="utf-8"))

    tmp_db = db_path.with_suffix(".sqlite.building")
    if tmp_db.exists():
        print(f"[sqlite] removing leftover {tmp_db}", flush=True)
        tmp_db.unlink()

    print(f"[sqlite] creating schema in {tmp_db}", flush=True)
    conn = sqlite3.connect(str(tmp_db))
    try:
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.execute("PRAGMA journal_mode=OFF;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-1048576;")  # 1 GiB
        conn.execute("PRAGMA locking_mode=EXCLUSIVE;")
        conn.executescript(schema_text)

        existing_tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

        n_tables = len(parquet_files)
        for ti, pq in enumerate(parquet_files, 1):
            table = pq.stem
            if table not in existing_tables:
                print(f"[sqlite] SKIP {pq.name}: no table in schema", flush=True)
                continue

            col_meta = _table_column_meta(conn, table)
            cols = [c for c, _, _ in col_meta]
            if not cols:
                print(f"[sqlite] SKIP {table}: no columns", flush=True)
                continue

            t_table = time.time()
            lf = pl.scan_parquet(str(pq))
            pq_cols = set(lf.collect_schema().names())
            select_exprs = [
                pl.col(c).cast(pl.Utf8) if c in pq_cols else pl.lit(None).cast(pl.Utf8).alias(c)
                for c in cols
            ]
            lf = lf.select(select_exprs)
            n = int(lf.select(pl.len()).collect().item())
            print(
                f"[sqlite] {ti}/{n_tables} loading {table} "
                f"({n:,} rows) from {pq.name} ...",
                flush=True,
            )

            col_list = ",".join(f'"{c}"' for c in cols)
            placeholders = ",".join("?" for _ in cols)
            sql = f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders})'
            declared = [t for _, t, _ in col_meta]
            not_nulls = [nn for _, _, nn in col_meta]
            pk_names = {name for name, _, _ in _table_pk_meta(conn, table)}
            pk_idxs = [i for i, name in enumerate(cols) if name in pk_names]

            inserted = 0
            cur = conn.cursor()
            # Slice from lazy plan so we never hold the full table in RAM.
            for start in range(0, max(n, 1), batch_size):
                if n == 0:
                    break
                batch_df = lf.slice(start, batch_size).collect()
                if batch_df.height == 0:
                    break
                batch = [
                    tuple(
                        _coerce_sqlite_value(
                            declared[i], v, not_null=not_nulls[i]
                        )
                        for i, v in enumerate(row)
                    )
                    for row in batch_df.iter_rows(named=False)
                ]
                try:
                    cur.executemany(sql, batch)
                except sqlite3.IntegrityError as e:
                    # Diagnose the failing batch (should be unreachable after preflight).
                    detail = _diagnose_integrity_batch(
                        table, cols, batch, pk_idxs, str(e)
                    )
                    raise sqlite3.IntegrityError(
                        f"{table} insert failed at rows {start}-{start + len(batch)}: "
                        f"{e}. {detail}"
                    ) from e
                inserted += len(batch)
                if n >= batch_size and (
                    inserted == n or inserted % (batch_size * 10) == 0
                ):
                    print(
                        _progress_line(
                            f"sqlite:{table}",
                            inserted,
                            n,
                            t_table,
                        ),
                        flush=True,
                    )
            conn.commit()
            print(
                f"[sqlite] {table}: inserted={inserted:,}/{n:,} "
                f"in {_fmt_duration(time.time() - t_table)} "
                f"(tables {ti}/{n_tables})",
                flush=True,
            )

        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
    except Exception:
        conn.close()
        conn = None  # type: ignore[assignment]
        if tmp_db.exists():
            print(f"[sqlite] leaving incomplete {tmp_db} for inspection "
                  f"(delete before retry)", flush=True)
        raise
    finally:
        if conn is not None:
            conn.close()

    if db_path.exists():
        db_path.unlink()
    tmp_db.replace(db_path)
    size_gb = db_path.stat().st_size / (1024**3)
    print(
        f"[sqlite] wrote {db_path} size={size_gb:.2f} GB "
        f"elapsed={_fmt_duration(time.time() - t0)}",
        flush=True,
    )


def _diagnose_integrity_batch(
    table: str,
    cols: Sequence[str],
    batch: Sequence[Tuple[Any, ...]],
    pk_idxs: Sequence[int],
    err: str,
) -> str:
    """Summarize likely cause of an IntegrityError within one insert batch."""
    parts: List[str] = []
    if pk_idxs:
        keys = [tuple(row[i] for i in pk_idxs) for row in batch]
        counts = Counter(keys)
        dups = [(k, c) for k, c in counts.items() if c > 1]
        if dups:
            dups.sort(key=lambda x: -x[1])
            pk_names = [cols[i] for i in pk_idxs]
            samples = ", ".join(f"{dict(zip(pk_names, k))!r} x{c}" for k, c in dups[:5])
            parts.append(f"batch PK collisions on {pk_names}: {samples}")
        null_pk = sum(1 for k in keys if any(v is None for v in k))
        if null_pk:
            parts.append(f"{null_pk} batch rows have NULL in PK")
    if "NOT NULL" in err.upper():
        # Find columns that are None in the batch.
        none_cols = [
            cols[i]
            for i in range(len(cols))
            if any(row[i] is None for row in batch)
        ]
        if none_cols:
            parts.append(f"NULL values present in columns: {none_cols[:20]}")
    return " | ".join(parts) if parts else "no batch-level PK/NULL diagnosis"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel club JSON -> Parquet -> SQLite (same schema as legacy path)."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            f"Process pool size (default: {DEFAULT_WORKERS}, "
            f"capped at 61 on Windows)."
        ),
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=DEFAULT_FLUSH_EVERY,
        help=(
            "JSON files per worker batch before writing Parquet shards and "
            f"clearing buffers (default: {DEFAULT_FLUSH_EVERY}). "
            "Lower = less RAM; higher = fewer shard files."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N JSON files (0 = all).",
    )
    parser.add_argument(
        "--parquet-dir",
        type=pathlib.Path,
        default=DEFAULT_PARQUET_DIR,
        help="Directory for per-table Parquet outputs.",
    )
    parser.add_argument(
        "--shard-dir",
        type=pathlib.Path,
        default=DEFAULT_SHARD_DIR,
        help="Scratch directory for worker shards (deleted unless --keep-shards).",
    )
    parser.add_argument(
        "--db",
        type=pathlib.Path,
        default=DEFAULT_DB,
        help="Output SQLite path (acbl_club_results.sqlite).",
    )
    parser.add_argument(
        "--schema",
        type=pathlib.Path,
        default=DEFAULT_SCHEMA,
        help="SQLite schema SQL file (must match legacy).",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Do not rebuild Parquet (with --build-sqlite: load SQLite from existing --parquet-dir).",
    )
    parser.add_argument(
        "--build-sqlite",
        action="store_true",
        help=(
            "Also build acbl_club_results.sqlite from the Parquet tables "
            "(~7h20m for the full corpus). Off by default; 2a/2b read Parquet."
        ),
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Accepted for old command lines; skipping SQLite is now the default.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Only validate existing Parquet against schema PK/UNIQUE after "
            "coerce; do not build Parquet or SQLite."
        ),
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="Keep intermediate worker shard Parquets.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="SQLite executemany / Parquet slice batch size.",
    )
    args = parser.parse_args(argv)
    if args.build_sqlite and args.skip_sqlite:
        parser.error("--build-sqlite and --skip-sqlite contradict each other")
    if not args.workers or args.workers <= 0:
        args.workers = _default_workers()
    else:
        args.workers = max(1, min(args.workers, _max_process_workers()))
    args.flush_every = max(1, args.flush_every)

    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    src_root = pathlib.Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from mlBridge import print_ended, print_started

    t_prog = print_started()
    print("=" * 70)
    print("ACBL Club JSON -> Parquet -> SQLite")
    print("=" * 70)
    print(f"workers:      {args.workers} (Windows ProcessPool cap is 61)")
    print(f"flush_every:  {args.flush_every:,}")
    print(f"parquet_dir:  {args.parquet_dir}")
    print(f"db:           {args.db}")
    print(f"schema:       {args.schema}")

    try:
        if args.preflight_only:
            preflight_parquet_dir(args.parquet_dir, args.schema)
            print_ended(t_prog)
            return 0
        if not args.skip_parquet:
            json_files = discover_json_files(
                acblPath.joinpath("club-results"), args.limit
            )
            if not json_files:
                print("ERROR: no *.data.json files found under club-results/")
                return 1
            build_parquet_from_json(
                json_files,
                args.parquet_dir,
                args.shard_dir,
                workers=args.workers,
                flush_every=args.flush_every,
                keep_shards=args.keep_shards,
            )
        else:
            print(f"[parquet] skipped; using {args.parquet_dir}")

        if args.build_sqlite:
            load_sqlite_from_parquet(
                args.parquet_dir,
                args.db,
                args.schema,
                batch_size=args.batch_size,
            )
        else:
            print("[sqlite] skipped (default; pass --build-sqlite to build it)")
    except Exception:
        traceback.print_exc()
        print_ended(t_prog)
        return 1

    print_ended(t_prog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
