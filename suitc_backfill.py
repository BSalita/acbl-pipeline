"""Fill the shared SuitC cache with every holding in the ACBL and FFBridge deals.

The SC_* columns (mlBridge/mlBridgeSuitCLib.py) look up each partnership's suit
holding in E:\\bridge\\data\\suitc\\combos\\ (SUITC_CACHE_DIR). Running this before
a pipeline rebuild means the rebuild only joins. About 800,000 holdings in all;
the cache is saved every 20,000, so the script can be stopped and rerun.

    python suitc_backfill.py --limit 2000        # timing trial
    python suitc_backfill.py --workers 64
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime

import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mlBridge import mlBridgeSuitCLib as sc  # noqa: E402

DEFAULT_SOURCES = (
    pathlib.Path(r"E:\bridge\data\acbl\acbl_club_hand_records_cache_df.parquet"),
    pathlib.Path(r"E:\bridge\data\acbl\acbl_tournament_hand_records_cache_df.parquet"),
    pathlib.Path(r"E:\bridge\data\ffbridge\ffbridge_hand_records_cache.parquet"),
)


def holdings(sources: list[pathlib.Path]) -> pl.DataFrame:
    """Distinct (A, B) partnership holdings over all deals in ``sources``."""
    pbns = pl.concat([pl.scan_parquet(p).select("PBN") for p in sources]).unique().collect()
    print(f"{pbns.height:,} distinct deals", flush=True)
    return sc._suit_holdings(pbns).select("A", "B").unique()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=pathlib.Path, action="append",
                        help="hand-records parquet with a PBN column (repeatable; default: ACBL club, "
                             "ACBL tournament and FFBridge caches)")
    parser.add_argument("--cache-dir", type=pathlib.Path, default=sc.DEFAULT_CACHE_DIR)
    parser.add_argument("--workers", type=int, default=None, help="parallel SuitC runs (default: CPU count)")
    parser.add_argument("--limit", type=int, default=None, help="run at most this many new holdings")
    args = parser.parse_args()

    started = datetime.now()
    print(f"start {started:%Y-%m-%d %H:%M:%S}  SuitC {sc.suitc_exe()}  cache {args.cache_dir}", flush=True)
    sources = args.source or [p for p in DEFAULT_SOURCES if p.is_file()]
    wanted = holdings(sources)
    cache = sc.SuitCCache(args.cache_dir, workers=args.workers)
    done = cache.table().filter(pl.col("Status") != "error").select("A", "B")  # errors are retried
    todo = wanted.join(done, on=["A", "B"], how="anti")
    print(f"{wanted.height:,} distinct holdings; {wanted.height - todo.height:,} cached; {todo.height:,} to run", flush=True)
    if args.limit is not None:
        todo = todo.head(args.limit)
    cache.ensure(todo.rows())
    status = cache.table().group_by("Status").len().sort("Status").rows()
    ended = datetime.now()
    print(f"cache now {cache.table().height:,} holdings; status {status}")
    print(f"end {ended:%Y-%m-%d %H:%M:%S}  elapsed {ended - started}", flush=True)


if __name__ == "__main__":
    main()
