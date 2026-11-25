"""Platinum masterpoints a player had won before a date (ACBL tournament awards).

The awards sidecar (``acbl_tournament_player_awards.parquet``, written by
``elo/acbl_awards.py``) has no dates. Each award is dated by when it became
known:
  - the session's date, when the session is in the tournament board results;
  - otherwise the last session date of that NABC (team events at an NABC),
    or an approximate NABC end date for NABCs before the board results start;
  - otherwise the end of the month in the session id (``yymm...``) for
    regional events.
Values at a date use only awards known strictly before it.

Used by ``acbl_elo_ratings_create.py`` for players' starting ratings. Study:
``elo/ELO_PLATINUM_BENCHMARK.md``; same logic as ``elo/platinum_bench/features.py``.
"""
from __future__ import annotations

import calendar
import pathlib
from datetime import date

import polars as pl

NABC_APPROX_END = {"1": (3, 31), "2": (8, 5), "3": (12, 10)}


def _approx_date(nabc: str | None, session_id: str) -> date | None:
    if nabc:
        yy, n = int(nabc[4:6]), nabc[6:7]
        if n not in NABC_APPROX_END:
            return None
        m, d = NABC_APPROX_END[n]
        return date(2000 + yy, m, d)
    head = session_id[:4]
    if head.isdigit():
        y, m = 2000 + int(head[:2]), int(head[2:4])
        if 1 <= m <= 12:
            return date(y, m, calendar.monthrange(y, m)[1])
    return None


def dated_platinum_awards(awards_path: pathlib.Path, sessions: pl.DataFrame) -> pl.DataFrame:
    """(Player_ID, Date known, cumulative platinum points), one row per player and date.

    ``sessions``: tournament session_id -> Date (from the board results).
    """
    awards = pl.read_parquet(awards_path).filter(
        pl.col("mp_color").str.to_lowercase() == "platinum")
    smap = sessions.select(pl.col("session_id").cast(pl.Utf8), pl.col("Date").alias("sdate")).unique("session_id")
    nabc_end = (
        smap.filter(pl.col("session_id").str.starts_with("NABC"))
        .with_columns(pl.col("session_id").str.extract(r"^(NABC\d+)").alias("nabc"))
        .group_by("nabc").agg(pl.col("sdate").max().alias("nabc_end"))
    )
    a = (
        awards.with_columns(pl.col("session_id").cast(pl.Utf8))
        .join(smap, on="session_id", how="left")
        .with_columns(pl.col("event_id").str.extract(r"^(NABC\d+)").alias("nabc"))
        .join(nabc_end, on="nabc", how="left")
    )
    need = a.filter(pl.col("sdate").is_null() & pl.col("nabc_end").is_null()).select("nabc", "session_id").unique("session_id")
    need = need.with_columns(pl.Series("approx", [_approx_date(n, s) for n, s in need.iter_rows()], dtype=pl.Date))
    a = a.join(need.drop("nabc"), on="session_id", how="left").with_columns(
        pl.coalesce("sdate", "nabc_end", "approx").alias("known"))
    undated = a["known"].null_count()
    if undated:
        raise ValueError(f"{undated} platinum awards have no date; session id format changed?")
    return (
        a.group_by(pl.col("player_id").cast(pl.Utf8).alias("Player_ID"), pl.col("known").alias("Date"))
        .agg(pl.col("mp_won").sum().alias("d"))
        .sort("Player_ID", "Date")
        .with_columns(pl.col("d").cum_sum().over("Player_ID").alias("plat_pts"))
        .select("Player_ID", "Date", "plat_pts")
    )


def platinum_points_before(keys: pl.DataFrame, cum: pl.DataFrame) -> pl.DataFrame:
    """``keys`` (Player_ID Utf8, Date) -> same rows plus ``plat_pts`` won before Date (0 if none)."""
    q = keys.unique(["Player_ID", "Date"]).sort("Date")
    j = q.join_asof(cum.sort("Date"), on="Date", by="Player_ID", strategy="backward", allow_exact_matches=False)
    return j.with_columns(pl.col("plat_pts").fill_null(0.0))
