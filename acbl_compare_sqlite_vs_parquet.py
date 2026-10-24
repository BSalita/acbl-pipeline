"""Check that 2a/2b would read the same club data from 1b's parquets as from SQLite.

Stages 2a (acbl_sql_to_hand_records_clean.py) and 2b
(acbl_sql_to_board_results_clean.py) read acbl_club_results.sqlite, which 1b
rebuilds from club_results_parquet/ every run (~7h20m). If every club query
returns identical frames from both sources, 2a/2b can read the parquets and
1b can run with --skip-sqlite.

For each query this script:
  1. runs it through read_sqlite_query (the 2a/2b path) and read_parquet_query
     (same SQL, same casts, parquet views),
  2. compares schemas and row counts,
  3. compares rows as multisets (row hashes), and
  4. joins on the table key and counts mismatches per column, with examples.

Only compare SQLite and parquet from the same 1b run (the script warns if
the parquet files are newer than the SQLite file).

Usage:
    python acbl_compare_sqlite_vs_parquet.py                 # all club tables
    python acbl_compare_sqlite_vs_parquet.py --tables boards sessions
    python acbl_compare_sqlite_vs_parquet.py --list

Exit code 0 when every table matches, 1 otherwise. Tables are read one at a
time; board_results is the largest (~211M rows) and needs well over 100 GB
of RAM for both copies.
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

import polars as pl

from acbl_sqlite_read import read_parquet_query, read_sqlite_query
from acbl_sql_to_board_results_clean import get_club_schemas
from acbl_sql_to_hand_records_clean import get_club_hand_records_schemas

ACBL_PATH = pathlib.Path("e:/bridge/data/acbl")
SQLITE_FILE = ACBL_PATH / "acbl_club_results.sqlite"
PARQUET_DIR = ACBL_PATH / "club_results_parquet"

# Key columns per query (output names). hand_records has no id in its select.
KEYS = {
    "2a.hand_records": ["hand_record_id", "board"],
    "2a.sessions": ["session_id"],
    "2b.events": ["event_id"],
    "2b.board_results": ["board_result_id"],
    "2b.boards": ["board_id"],
    "2b.pair_summaries": ["pair_summary_id"],
    "2b.players": ["player_id"],
    "2b.sessions": ["session_id"],
    "2b.sections": ["section_id"],
}


def _queries() -> dict[str, tuple[str, dict]]:
    out = {}
    for step, (sql_d, schema_d) in (("2a", get_club_hand_records_schemas()), ("2b", get_club_schemas())):
        for table, schema in schema_d.items():
            out[f"{step}.{table}"] = (sql_d[table], schema)
    return out


def _compare(name: str, sql: str, schema: dict, examples: int) -> bool:
    print(f"\n=== {name} ===\n{sql}")
    t = time.time()
    a = read_sqlite_query("sqlite:///" + SQLITE_FILE.as_posix(), sql, schema)
    t_sqlite = time.time() - t
    t = time.time()
    b = read_parquet_query(PARQUET_DIR.as_posix(), sql, schema)
    t_parquet = time.time() - t
    print(f"read sqlite {t_sqlite:.0f}s ({a.height:,} rows), parquet {t_parquet:.0f}s ({b.height:,} rows)")

    ok = True
    if a.schema != b.schema:
        ok = False
        print("SCHEMA DIFFERS")
        for col in sorted(set(a.columns) | set(b.columns)):
            da, db = a.schema.get(col), b.schema.get(col)
            if da != db:
                print(f"  {col}: sqlite={da} parquet={db}")
        common = [c for c in a.columns if c in b.columns and a.schema[c] == b.schema[c]]
        a, b = a.select(common), b.select(common)
    if a.height != b.height:
        ok = False
        print(f"ROW COUNT DIFFERS: sqlite {a.height:,} vs parquet {b.height:,}")

    ha = a.hash_rows().sort()
    hb = b.select(a.columns).hash_rows().sort()
    rows_equal = ha.len() == hb.len() and ha.equals(hb)
    print(f"rows as multiset: {'identical' if rows_equal else 'DIFFER'}")
    del ha, hb
    if rows_equal and ok:
        print("MATCH")
        return True
    ok = False

    keys = [k for k in KEYS.get(name, []) if k in a.columns]
    if not keys:
        print("no key columns to join on; cannot break down differences")
        return False
    for side, df in (("sqlite", a), ("parquet", b)):
        dup = df.height - df.select(keys).n_unique()
        if dup:
            print(f"  note: {dup:,} duplicate keys on the {side} side")
    only_a = a.join(b.select(keys), on=keys, how="anti", nulls_equal=True)
    only_b = b.join(a.select(keys), on=keys, how="anti", nulls_equal=True)
    print(f"keys only in sqlite: {only_a.height:,}; only in parquet: {only_b.height:,}")
    if only_a.height:
        print(only_a.head(examples))
    if only_b.height:
        print(only_b.head(examples))

    value_cols = [c for c in a.columns if c not in keys]
    joined = a.unique(keys, keep="first").join(
        b.unique(keys, keep="first"), on=keys, how="inner", suffix="__pq", nulls_equal=True
    )
    print(f"per-column mismatches over {joined.height:,} shared keys:")
    for col in value_cols:
        diff = joined.filter(~pl.col(col).eq_missing(pl.col(f"{col}__pq")))
        if diff.height:
            print(f"  {col}: {diff.height:,} rows differ, e.g.")
            print(diff.select(keys + [col, f"{col}__pq"]).head(examples))
    return False


def main() -> int:
    queries = _queries()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tables", nargs="*", help="query names or bare table names (default: all)")
    parser.add_argument("--list", action="store_true", help="list query names and exit")
    parser.add_argument("--examples", type=int, default=5, help="example rows to print per difference")
    args = parser.parse_args()
    if args.list:
        for name, (sql, _) in queries.items():
            print(f"{name}: {sql}")
        return 0

    selected = list(queries)
    if args.tables:
        wanted = set(args.tables)
        selected = [n for n in queries if n in wanted or n.split(".", 1)[1] in wanted]
        if not selected:
            print(f"No queries match {args.tables}. Use --list.")
            return 2

    for path in (SQLITE_FILE, PARQUET_DIR):
        if not path.exists():
            print(f"Missing: {path}")
            return 2
    sqlite_mtime = SQLITE_FILE.stat().st_mtime
    newer = [p.name for p in PARQUET_DIR.glob("*.parquet") if p.stat().st_mtime > sqlite_mtime]
    if newer:
        print(f"WARNING: parquet files newer than the SQLite file (not the same 1b run?): {newer}")

    results = {}
    for name in selected:
        sql, schema = queries[name]
        results[name] = _compare(name, sql, schema, args.examples)
        gc.collect()

    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {'MATCH ' if ok else 'DIFFER'}  {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
