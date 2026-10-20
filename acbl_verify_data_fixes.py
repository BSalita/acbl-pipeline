#!/usr/bin/env python3
"""Verify the October 2026 ACBL data fixes after an acbl_all.bat rebuild.

Checks:
    - Doubled contracts survive in club data (spaced contracts like '5 C x' lost the X).
    - The raw ACBL 'result' / 'tricks_taken' strings are gone; Result and Tricks agree.
    - Player id 0 (missing ACBL number) is stored as null, and Elo does not rate null seats.
    - Pair keys are null with an unknown member and ordered as strings.

Usage:
    python acbl_verify_data_fixes.py [--data-dir e:/bridge/data/acbl] [--since 2026-10-21]

Exits 1 if any check fails. Reads only a few columns of each file (about a minute).
"""

import argparse
import datetime
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Callable

import polars as pl
from tqdm import tqdm

DOUBLED_SHARE_RANGE = (0.025, 0.08)  # ACBL tournament data runs about 4.2%.
SEATS = "NESW"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _scan(path: pathlib.Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def _zero_id(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Utf8).str.contains(r"^0+$").fill_null(False)


def _fresh(path: pathlib.Path, since: datetime.date) -> Check:
    if not path.is_file():
        return Check(f"{path.name} exists", False, "missing")
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    return Check(
        f"{path.name} rebuilt since {since}",
        mtime.date() >= since,
        f"modified {mtime:%Y-%m-%d %H:%M}",
    )


def check_board_results(path: pathlib.Path, *, club: bool) -> list[Check]:
    schema = pl.read_parquet_schema(path)
    checks = [
        Check(
            f"{path.name}: raw result/tricks_taken dropped",
            not {"result", "tricks_taken"} & set(schema),
            f"columns present: {sorted({'result', 'tricks_taken'} & set(schema)) or 'none'}",
        )
    ]
    required = {"BidLvl", "Dbl", "Result", "Tricks", *(f"Player_ID_{s}" for s in SEATS)}
    missing = sorted(required - set(schema))
    if missing:
        checks.append(Check(f"{path.name}: required columns present", False, f"missing {missing}"))
        return checks
    zero_declarer = _zero_id("Declarer_ID").sum() if "Declarer_ID" in schema else pl.lit(0)
    played = pl.col("BidLvl").is_not_null() & (pl.col("BidLvl") > 0)
    stats = _collect(
        _scan(path).select(
            contracts=played.sum(),
            doubled=(played & (pl.col("Dbl").fill_null("") != "")).sum(),
            result_null=(played & pl.col("Result").is_null()).sum(),
            result_mismatch=(
                played
                & pl.col("Result").is_not_null()
                & pl.col("Tricks").is_not_null()
                & (
                    pl.col("Tricks").cast(pl.Int16) - pl.col("BidLvl").cast(pl.Int16) - 6
                    != pl.col("Result").cast(pl.Int16)
                )
            ).sum(),
            **{f"zero_{s}": _zero_id(f"Player_ID_{s}").sum() for s in SEATS},
            **{f"null_{s}": pl.col(f"Player_ID_{s}").is_null().sum() for s in SEATS},
            zero_declarer=zero_declarer,
            rows=pl.len(),
        )
    ).row(0, named=True)
    share = stats["doubled"] / max(stats["contracts"], 1)
    lo, hi = DOUBLED_SHARE_RANGE
    checks.append(Check(
        f"{path.name}: doubled share in {lo:.1%}-{hi:.1%}",
        lo <= share <= hi,
        f"{stats['doubled']:,} of {stats['contracts']:,} contracts ({share:.2%})",
    ))
    checks.append(Check(
        f"{path.name}: Result = Tricks - BidLvl - 6",
        stats["result_mismatch"] == 0,
        f"{stats['result_mismatch']:,} mismatches",
    ))
    if club:
        null_share = stats["result_null"] / max(stats["contracts"], 1)
        checks.append(Check(
            f"{path.name}: Result present for played contracts (< 1% null)",
            null_share < 0.01,
            f"{stats['result_null']:,} null ({null_share:.2%})",
        ))
    zeros = sum(stats[f"zero_{s}"] for s in SEATS) + stats["zero_declarer"]
    nulls = sum(stats[f"null_{s}"] for s in SEATS)
    checks.append(Check(
        f"{path.name}: no player id 0",
        zeros == 0,
        f"{zeros:,} zero ids; {nulls:,} null seat ids of {4 * stats['rows']:,}",
    ))
    return checks


def check_board_elo(path: pathlib.Path) -> list[Check]:
    stats = _collect(
        _scan(path).select(
            **{
                f"rated_null_{s}": (
                    pl.col(f"Player_ID_{s}").is_null() & pl.col(f"Elo_R_{s}").is_not_nan()
                ).sum()
                for s in SEATS
            },
            **{f"zero_{s}": _zero_id(f"Player_ID_{s}").sum() for s in SEATS},
        )
    ).row(0, named=True)
    rated = sum(stats[f"rated_null_{s}"] for s in SEATS)
    zeros = sum(stats[f"zero_{s}"] for s in SEATS)
    return [
        Check(f"{path.name}: no rating for seats without an id", rated == 0, f"{rated:,} rated null seats"),
        Check(f"{path.name}: no player id 0", zeros == 0, f"{zeros:,} zero ids"),
    ]


def check_player_elo(path: pathlib.Path) -> list[Check]:
    stats = _collect(
        _scan(path).select(zero=_zero_id("Player_ID").sum(), null=pl.col("Player_ID").is_null().sum())
    ).row(0, named=True)
    return [Check(
        f"{path.name}: no player id 0 or null",
        stats["zero"] == 0 and stats["null"] == 0,
        f"{stats['zero']:,} zero, {stats['null']:,} null",
    )]


def check_pair_elo(path: pathlib.Path) -> list[Check]:
    parts = pl.col("Pair_IDs").str.split_exact("-", 1)
    left, right = parts.struct.field("field_0"), parts.struct.field("field_1")
    stats = _collect(
        _scan(path).select(
            self_pair=(left == right).sum(),
            zero=(left.str.contains(r"^0+$") | right.str.contains(r"^0+$")).sum(),
            unordered=(left > right).sum(),
            null=pl.col("Pair_IDs").is_null().sum(),
        )
    ).row(0, named=True)
    bad = {k: v for k, v in stats.items() if v}
    return [Check(f"{path.name}: pair keys valid", not bad, str(bad or "self/zero/unordered/null all 0"))]


def check_contracts(path: pathlib.Path) -> list[Check]:
    contract = pl.col("Contract").cast(pl.Utf8)
    stats = _collect(
        _scan(path).select(
            contracts=contract.str.contains(r"^[1-7]").sum(),
            doubled=contract.str.contains(r"^[1-7][CDHSN]X").sum(),
        )
    ).row(0, named=True)
    share = stats["doubled"] / max(stats["contracts"], 1)
    lo, hi = DOUBLED_SHARE_RANGE
    return [Check(
        f"{path.name}: doubled share in {lo:.1%}-{hi:.1%}",
        lo <= share <= hi,
        f"{stats['doubled']:,} of {stats['contracts']:,} ({share:.2%})",
    )]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("e:/bridge/data/acbl"))
    parser.add_argument(
        "--since",
        type=datetime.date.fromisoformat,
        default=datetime.date(2026, 10, 21),
        help="Files older than this predate the fixes (default 2026-10-21).",
    )
    args = parser.parse_args()
    started = datetime.datetime.now()
    t0 = time.perf_counter()
    print(f"start {started:%Y-%m-%d %H:%M:%S}")

    d = args.data_dir
    plan: list[tuple[pathlib.Path, Callable[[pathlib.Path], list[Check]]]] = [
        (d / "acbl_club_board_results_augmented_step1.parquet", lambda p: check_board_results(p, club=True)),
        (d / "acbl_club_board_results_augmented.parquet", lambda p: check_board_results(p, club=True)),
        (d / "acbl_tournament_board_results_augmented.parquet", lambda p: check_board_results(p, club=False)),
        (d / "acbl_club_elo_ratings.parquet", check_board_elo),
        (d / "acbl_tournament_elo_ratings.parquet", check_board_elo),
        (d / "acbl_club_player_elo_ratings.parquet", check_player_elo),
        (d / "acbl_tournament_player_elo_ratings.parquet", check_player_elo),
        (d / "acbl_club_pair_elo_ratings.parquet", check_pair_elo),
        (d / "acbl_tournament_pair_elo_ratings.parquet", check_pair_elo),
        (d / "acbl_club_prediction_data_test_v2.parquet", check_contracts),
    ]
    model_data = d / "acbl_club_model_data.parquet"
    if model_data.is_file():
        plan.append((model_data, check_contracts))

    results: list[Check] = []
    for path, check in tqdm(plan, desc="files", unit="file"):
        fresh = _fresh(path, args.since)
        results.append(fresh)
        if path.is_file():
            results.extend(check(path))

    failed = [r for r in results if not r.passed]
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.detail}")
    ended = datetime.datetime.now()
    print(f"{len(results) - len(failed)} passed, {len(failed)} failed")
    print(f"end {ended:%Y-%m-%d %H:%M:%S} elapsed {datetime.timedelta(seconds=round(time.perf_counter() - t0))}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
