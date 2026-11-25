#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_elo_ratings_create.py

Takes 6m/1m30s 50m/15m rows by 26->61/24->60 columns. input filesize 43GB->3.5GB/16GB->1.3GB. Output 8.8m/2.8m players 104MB/24MB, 4.4m/1.4m pairs file 23MB/24MB. Uses 128GB memory/pagefile.

Computes Elo ratings for players and pairs from board results.
Creates lookup tables for player and pair Elo ratings.

Player Elo (since 2026-11-25, elo/ELO_PLATINUM_BENCHMARK.md): one rating per
player, fed by club AND tournament results in one date-ordered stream, so
both board-results files are always read. A player starts from
START_CONST + START_PER_LOG_MP * ln(1 + masterpoints)
+ START_PER_LOG_PLATINUM * ln(1 + platinum points won before that day)
instead of 1500, with K 6 and a 6x boost for the first 200 boards. Pair Elo
is unchanged (per side, per direction, K 24). --club / --tournament only
choose which side's files are written.

Previous steps:
    club/tournamentdownload->json->sql->board_results->augmented

Next steps:
    ?

TODO:
    - 
"""

import json
import polars as pl
from collections import defaultdict
import pathlib
import time
import re
from datetime import datetime, timezone

import numpy as np

import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

from acbl_platinum_to_date import dated_platinum_awards, platinum_points_before
from acbl_strata import club_strata_bucket, tournament_strata_bucket

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')

# Player Elo settings, chosen on 2025 and earlier and tested on 2026 in
# elo/ELO_PLATINUM_BENCHMARK.md (club 0.438 -> 0.478, tournament 0.340 ->
# 0.430 section-rank Spearman). Starting weights fitted on pre-2025 ratings.
PLAYER_K_BASE = 6.0
PLAYER_BOOST = 6.0
PLAYER_BOOST_UNTIL_BOARDS = 200
START_CONST = 1455.15
START_PER_LOG_MP = 10.17
START_PER_LOG_PLATINUM = 12.29
TOURNAMENT_AWARDS_FILENAME = 'acbl_tournament_player_awards.parquet'
SIDES = ("club", "tournament")
PLAYER_OUTPUT_COLS = [
    *[f'Elo_R_{s}_Before' for s in 'NESW'], 'Elo_E_NS', 'Elo_E_EW',
    *[f'Elo_R_{s}' for s in 'NESW'], *[f'Elo_N_{s}' for s in 'NESW'],
    'Field_Stdev_NS_Player', 'Field_Stdev_EW_Player', 'Field_Mean_NS_Player', 'Field_Mean_EW_Player',
]


# Default "evidence weight" of the prior in the published (Bayesian-shrunk)
# Elo formula. The Streamlit/API can override at query time.
SHRINKAGE_DEFAULT_PRIOR_SESSIONS = 0  # 0 since 2026-11-25 (the API and app default)

# Persist Elo from the first session. Sample-size eligibility belongs to the
# report's min_sessions filter, while Bayesian shrinkage controls provisional
# volatility. Masking the first nine sessions here made reports count only
# post-threshold sessions and silently applied the minimum twice.
PERSISTED_ELO_MINIMUM_SESSIONS = 1


def _compute_shrinkage_metadata(
    player_elo_lookup: pl.DataFrame,
    pair_elo_lookup: pl.DataFrame,
    *,
    min_sessions_for_prior: int = 10,
) -> dict:
    """Compute global Elo medians used as the Bayesian shrinkage prior.

    The "established" subset (>= ``min_sessions_for_prior`` sessions) is what
    we want to anchor the prior to — picking the median of *every* player
    biases toward newcomers who happen to be near the initial rating.

    Returns a dict suitable for JSON serialization.
    """
    meta: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_sessions_for_prior": int(min_sessions_for_prior),
        "default_prior_sessions": int(SHRINKAGE_DEFAULT_PRIOR_SESSIONS),
        "shrinkage_formula": (
            "published_elo = (sessions * raw_elo + prior_sessions * prior_anchor) / "
            "(sessions + prior_sessions)"
        ),
        "player": {},
        "pair": {},
    }

    # Latest Elo per (Player_ID), then take median over the established subset.
    if not player_elo_lookup.is_empty() and "Elo_R_EventEnd" in player_elo_lookup.columns:
        per_player = (
            player_elo_lookup
            .sort(["Player_ID", "Date", "session_id"])
            .group_by("Player_ID")
            .agg(
                pl.col("Elo_R_EventEnd").last().alias("latest_elo"),
                pl.len().alias("sessions"),
            )
            .with_columns(pl.col("latest_elo").cast(pl.Float64))
        )
        established = per_player.filter(pl.col("sessions") >= min_sessions_for_prior)
        # Drop nulls/NaNs before computing median.
        v = established.select(pl.col("latest_elo").drop_nulls().drop_nans()).get_column("latest_elo")
        if not v.is_empty():
            meta["player"] = {
                "prior_anchor": round(float(v.median()), 2),
                "median": round(float(v.median()), 2),
                "mean": round(float(v.mean()), 2),
                "stdev": round(float(v.std(ddof=0)), 2),
                "n_established": int(len(v)),
                "p25": round(float(v.quantile(0.25, interpolation="linear")), 2),
                "p75": round(float(v.quantile(0.75, interpolation="linear")), 2),
            }

    if not pair_elo_lookup.is_empty() and "Elo_R_EventEnd" in pair_elo_lookup.columns:
        per_pair = (
            pair_elo_lookup
            .sort(["Pair_IDs", "Date", "session_id"])
            .group_by("Pair_IDs")
            .agg(
                pl.col("Elo_R_EventEnd").last().alias("latest_elo"),
                pl.len().alias("sessions"),
            )
            .with_columns(pl.col("latest_elo").cast(pl.Float64))
        )
        established = per_pair.filter(pl.col("sessions") >= min_sessions_for_prior)
        v = established.select(pl.col("latest_elo").drop_nulls().drop_nans()).get_column("latest_elo")
        if not v.is_empty():
            meta["pair"] = {
                "prior_anchor": round(float(v.median()), 2),
                "median": round(float(v.median()), 2),
                "mean": round(float(v.mean()), 2),
                "stdev": round(float(v.std(ddof=0)), 2),
                "n_established": int(len(v)),
                "p25": round(float(v.quantile(0.25, interpolation="linear")), 2),
                "p75": round(float(v.quantile(0.75, interpolation="linear")), 2),
            }
    return meta


def _sanitize_session_mp_top(df: pl.DataFrame) -> pl.DataFrame:
    """Replace per-row MP_Top with the session median when the column exists.

    A single poison row (Thomas: MP_Top=320 vs session 16) inflates the
    section-scale cap and multiplies a corrupt Pct_NS residual.
    """
    if "MP_Top" not in df.columns:
        return df
    print("Replacing per-row MP_Top with session median...")
    return df.with_columns(
        pl.col("MP_Top").median().over("session_id").alias("MP_Top")
    )


def _read_global_stdev_seeds(parquet_path: pathlib.Path) -> dict:
    """Read same-direction global Elo stdev from a previous-run parquet.

    Returns a dict with keys ``pair_ns`` / ``pair_ew`` / ``player_ns`` /
    ``player_ew``; missing columns map to None so the math function falls
    back to its own auto-derived default.
    """
    if not parquet_path.exists():
        return {}
    seeds = {}
    try:
        schema = pl.scan_parquet(parquet_path).collect_schema().names()
    except Exception:
        return {}
    wanted = {
        "pair_ns": "Elo_R_NS_Before",
        "pair_ew": "Elo_R_EW_Before",
        "player_ns": "Elo_R_N_Before",
        "player_ew": "Elo_R_E_Before",
    }
    cols_present = [c for c in wanted.values() if c in schema]
    if not cols_present:
        return {}
    aggs = [
        pl.col(c).drop_nulls().drop_nans().std(ddof=0).alias(c)
        for c in cols_present
    ]
    try:
        row = pl.scan_parquet(parquet_path).select(aggs).collect(engine="streaming").row(0, named=True)
    except Exception:
        return {}
    for key, col in wanted.items():
        v = row.get(col)
        if v is not None and v > 0:
            seeds[key] = float(v)
    return seeds


def _read_pooled_player_seeds(paths: list[pathlib.Path]) -> dict:
    """Player global-stdev seeds for the joint stream: the pooled stdev of the
    previous run's club and tournament player Before columns."""
    out = {}
    for key, col in (("player_ns", "Elo_R_N_Before"), ("player_ew", "Elo_R_E_Before")):
        n_tot, s1, s2 = 0, 0.0, 0.0
        for path in paths:
            if not path.exists() or col not in pl.scan_parquet(path).collect_schema().names():
                continue
            v = pl.col(col).drop_nulls().drop_nans().cast(pl.Float64)
            row = (pl.scan_parquet(path)
                   .select(v.len().alias("n"), v.sum().alias("s1"), (v * v).sum().alias("s2"))
                   .collect(engine="streaming").row(0, named=True))
            n_tot += row["n"]
            s1 += row["s1"] or 0.0
            s2 += row["s2"] or 0.0
        if n_tot:
            mean = s1 / n_tot
            out[key] = float(np.sqrt(max(s2 / n_tot - mean * mean, 0.0)))
    return out


def load_side(club_or_tournament):
    """Board results of one side, prepared for Elo (strata, sorted, MP_Top sanitized)."""
    print(f"Loading {club_or_tournament} board results...")
    acbl_board_results_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    df = pl.read_parquet(acbl_board_results_file, n_rows=0)

    # Select only necessary columns
    regex_cols = [
            'Date', 'is_virtual_game', 'session_id', 'event_id', 'event_name',
            'Club$',  # club number (club only): home-versus-away split for Home_Edge_Z (elo/foresight.py)
            'Round', 'Board', 'Pair_Number_(NS|EW)',
            'Player_ID_[NESW]', 'Player_Name_[NESW]', 'MP_(NS|EW)', 'MP_Top',
            'MasterPoints_[NESW]', 'MasterPoints_(NS|EW)', 'Pct_NS',
            'DD_Tricks_Diff', 'Is_Par_Suit', 'Is_Par_Contract', 'Is_Sacrifice', # must use board results augmented to get these columns
            'Declarer_Direction', 'Declarer_Pair_Direction', 'BidSuit', 'ParContracts',
            'DD_Score_NS', 'DD_Score_EW', 'DD_Score_Declarer',
            'Par_NS', 'Par_EW', 'Par_Declarer',
            'mp_limit',  # event MP ceiling for Strata filter (club mpLimits / tournament mp_limit)
            'mp_color', 'mp_rating',  # tournament platinum identity; unused on club
            # Opening lead and its scores (club only; tournaments record no lead).
            'Direction_OnLead', 'OnLead$', 'Lead$', 'DD_Lead_Cost', 'Lead_Is_DD_Best', 'SD_Lead_',
            # SuitC (3a --suitc): declarer's guess cost and guess-adjusted DD difference.
            'SC_Guess_Declarer$', 'DD_Tricks_Diff_SC$',
        ]
    read_cols = [col for col in df.columns if any(re.match(regex, col) for regex in regex_cols)]

    # load board results data
    t = time.time()
    df = pl.read_parquet(acbl_board_results_file, columns=read_cols)
    print(f"Loaded: {acbl_board_results_filename} shape:{df.shape} "
          f"size:{acbl_board_results_file.stat().st_size} ({time.time()-t:.0f}s)")

    # tournament doesn't have is_virtual_game column
    if 'is_virtual_game' not in df.columns:
        print("Adding is_virtual_game column to df...")
        df = df.with_columns([pl.lit(False).alias('is_virtual_game')])

    if 'mp_limit' not in df.columns:
        raise RuntimeError(
            f"{acbl_board_results_filename} is missing mp_limit. "
            "Rebuild cleaned/augmented board results after pulling mpLimits "
            "(club) or mp_limit (tournament) through acbl_sql_to_board_results_clean.py."
        )

    # Normalize event MP limits to strata_bucket once per session, then broadcast.
    print("Computing strata_bucket from mp_limit...")
    bucket_fn = club_strata_bucket if club_or_tournament == "club" else tournament_strata_bucket
    session_limits = (
        df.select(["session_id", "mp_limit"])
        .unique(subset=["session_id"], keep="first")
    )
    session_buckets = session_limits.with_columns(
        pl.Series(
            "strata_bucket",
            [bucket_fn(v) for v in session_limits.get_column("mp_limit").to_list()],
        )
    )
    df = df.join(session_buckets.select(["session_id", "strata_bucket"]), on="session_id", how="left")
    print(
        "strata_bucket counts:\n",
        df.select(["session_id", "strata_bucket"])
        .unique()
        .group_by("strata_bucket")
        .len()
        .sort("len", descending=True),
    )

    # Sort temporally to calculate Elo
    print("Sorting data temporally...")
    sort_keys = ["Date", "session_id", "Round", "Board"]
    if 'Round' not in df.columns:
        sort_keys.remove('Round')
    df = df.sort(sort_keys)

    return _sanitize_session_mp_top(df)


def add_start_ratings(df: pl.DataFrame, plat_cum: pl.DataFrame) -> pl.DataFrame:
    """Per-seat ``_Elo_Start_{seat}``: the rating a player starts from if this is their first board."""
    keys = pl.concat([
        df.select(pl.col(f"Player_ID_{s}").cast(pl.Utf8).alias("Player_ID"), "Date") for s in "NESW"
    ]).drop_nulls("Player_ID")
    plat = platinum_points_before(keys, plat_cum).rename({"Player_ID": "_pid", "plat_pts": "_plat"})
    for s in "NESW":
        df = (
            df.with_columns(pl.col(f"Player_ID_{s}").cast(pl.Utf8).alias("_pid"))
            .join(plat, on=["_pid", "Date"], how="left", maintain_order="left")
            .with_columns(
                (START_CONST
                 + START_PER_LOG_MP * pl.col(f"MasterPoints_{s}").cast(pl.Float64).log1p()
                 + START_PER_LOG_PLATINUM * pl.col("_plat").fill_null(0.0).log1p()
                 ).alias(f"_Elo_Start_{s}")
            )
            .drop("_pid", "_plat")
        )
    return df


def joint_player_elo(sides: dict, seeds: dict) -> dict:
    """Player Elo over club and tournament together; each side's output columns, row-aligned."""
    parts = []
    for k, (side, df) in enumerate(sides.items()):
        parts.append(df.select(
            pl.lit(k, pl.Int8).alias("_side"),
            pl.int_range(pl.len(), dtype=pl.Int64).alias("_row"),
            "Date",
            (pl.lit(side[0].upper()) + pl.col("session_id").cast(pl.Utf8)).alias("session_id"),
            pl.col("Board").cast(pl.Int32),
            pl.col("Pct_NS").cast(pl.Float32),
            pl.col("MP_Top").cast(pl.Float32),
            pl.col("strata_bucket").cast(pl.Utf8),
            *[pl.col(f"Player_ID_{s}").cast(pl.Utf8) for s in "NESW"],
            *[pl.col(f"_Elo_Start_{s}") for s in "NESW"],
        ))
    joint = pl.concat(parts).sort("Date", maintain_order=True)
    print(f"Joint player stream: {joint.height:,} rows; seeds {seeds}")
    t = time.time()
    out = mlBridgeAugmentLib.compute_player_matchpoint_elo_ratings(
        joint,
        k_base=PLAYER_K_BASE,
        boost=PLAYER_BOOST,
        provisional_boost_until=PLAYER_BOOST_UNTIL_BOARDS,
        merged=True,
        start_rating_cols=tuple(f"_Elo_Start_{s}" for s in "NSEW"),
        minimum_sessions=PERSISTED_ELO_MINIMUM_SESSIONS,
        global_stdev_ns=seeds.get("player_ns"),
        global_stdev_ew=seeds.get("player_ew"),
    )
    print(f"Joint player Elo: {time.time()-t:.0f}s")
    return {
        side: out.filter(pl.col("_side") == k).sort("_row").select(PLAYER_OUTPUT_COLS)
        for k, side in enumerate(sides)
    }


def write_side(club_or_tournament: str, df: pl.DataFrame, player_cols: pl.DataFrame, out_path: pathlib.Path):
    """Pair Elo for one side, attach the joint player Elo, write the Elo files and lookups."""
    seeds = _read_global_stdev_seeds(acblPath.joinpath(f'acbl_{club_or_tournament}_elo_ratings.parquet'))
    print(f"[{club_or_tournament}] pair K-dampening seeds: {seeds}")
    df = df.drop([c for c in df.columns if c.startswith("_Elo_Start_")])
    t = time.time()
    df = mlBridgeAugmentLib.compute_pair_matchpoint_elo_ratings(
        df,
        minimum_sessions=PERSISTED_ELO_MINIMUM_SESSIONS,
        global_stdev_ns=seeds.get("pair_ns"),
        global_stdev_ew=seeds.get("pair_ew"),
    )
    print(f"[{club_or_tournament}] pair Elo: {time.time()-t:.0f}s")
    if player_cols.height != df.height:
        raise RuntimeError(f"player Elo rows {player_cols.height:,} != {club_or_tournament} rows {df.height:,}")
    # Player columns after the pair columns, as compute_matchpoint_elo_ratings
    # did (the player Elo_E_NS/EW replace the pair ones).
    df = df.drop([c for c in player_cols.columns if c in df.columns]).hstack(player_cols)
    df = mlBridgeAugmentLib.compute_event_start_end_elo_columns(df)

    # Save full Elo ratings
    acbl_club_elo_ratings_filename = f'acbl_{club_or_tournament}_elo_ratings.parquet'
    acbl_club_elo_ratings_file = out_path.joinpath(acbl_club_elo_ratings_filename)
    df.write_parquet(acbl_club_elo_ratings_file)
    print(f"Saved {acbl_club_elo_ratings_filename}: shape:{df.shape} size:{acbl_club_elo_ratings_file.stat().st_size}")

    # Create player Elo lookup table
    print("Creating player Elo lookup table...")
    player_dfs = []
    for direction in 'NESW':
        direction_df = df.select([
            pl.col('Date'),
            pl.col('session_id'),
            pl.col(f'Player_ID_{direction}').alias('Player_ID'),
            pl.col(f'Player_Name_{direction}').alias('Player_Name'),
            pl.col(f'MasterPoints_{direction}').alias('MasterPoints'),
            pl.col(f'Elo_R_{direction}_EventStart').alias('Elo_R_EventStart'),
            pl.col(f'Elo_R_{direction}_EventEnd').alias('Elo_R_EventEnd'),
            pl.col('is_virtual_game'),
        ]).filter(pl.col('Player_ID').is_not_null())
        player_dfs.append(direction_df)
    
    all_players_stacked = pl.concat(player_dfs)

    player_elo_lookup = (
        all_players_stacked
        .unique(subset=['Player_ID', 'session_id'], keep='first', maintain_order=True)
        .sort(['Player_ID', 'Date', 'session_id'])
        .with_columns([
            (pl.col('Player_ID').cum_count().over('Player_ID') - 1).alias('Elo_N')
        ])
    )
    
    # Save player Elo lookup
    acbl_club_player_elo_filename = f'acbl_{club_or_tournament}_player_elo_ratings.parquet'
    acbl_club_player_elo_file = out_path.joinpath(acbl_club_player_elo_filename)
    player_elo_lookup.write_parquet(acbl_club_player_elo_file)
    print(f"Saved {acbl_club_player_elo_filename}: shape:{player_elo_lookup.shape} size:{acbl_club_player_elo_file.stat().st_size}")
    
    # Create pair Elo lookup table
    print("Creating pair Elo lookup table...")
    # Extract NS pairs
    ns_pairs = df.select([
        pl.col('Date'),
        pl.col('session_id'),
        mlBridgeAugmentLib.pair_key_expr('Player_ID_N', 'Player_ID_S').alias('Pair_IDs'),
        pl.col('Elo_R_NS_EventStart').alias('Elo_R_EventStart'),
        pl.col('Elo_R_NS_EventEnd').alias('Elo_R_EventEnd'),
        pl.col('^MasterPoints_[NESW]$'),
        pl.col('is_virtual_game'),
    ]).filter(pl.col('Pair_IDs').is_not_null())
    
    # Extract EW pairs
    ew_pairs = df.select([
        pl.col('Date'),
        pl.col('session_id'),
        mlBridgeAugmentLib.pair_key_expr('Player_ID_E', 'Player_ID_W').alias('Pair_IDs'),
        pl.col('Elo_R_EW_EventStart').alias('Elo_R_EventStart'),
        pl.col('Elo_R_EW_EventEnd').alias('Elo_R_EventEnd'),
        pl.col('^MasterPoints_[NESW]$'),
        pl.col('is_virtual_game'),
    ]).filter(pl.col('Pair_IDs').is_not_null())
    
    all_pairs_stacked = pl.concat([ns_pairs, ew_pairs])
    
    pair_elo_lookup = (
        all_pairs_stacked
        .unique(subset=['Pair_IDs', 'session_id'], keep='first', maintain_order=True)
        .sort(['Pair_IDs', 'Date', 'session_id'])
        .with_columns([
            (pl.col('Pair_IDs').cum_count().over('Pair_IDs') - 1).alias('Elo_N')
        ])
    )

    # Save pair Elo lookup
    acbl_club_pair_elo_filename = f'acbl_{club_or_tournament}_pair_elo_ratings.parquet'
    acbl_club_pair_elo_file = out_path.joinpath(acbl_club_pair_elo_filename)
    pair_elo_lookup.write_parquet(acbl_club_pair_elo_file)
    print(f"Saved {acbl_club_pair_elo_filename}: shape:{pair_elo_lookup.shape} size:{acbl_club_pair_elo_file.stat().st_size}")

    # Bayesian shrinkage metadata: written as a sidecar JSON next to the
    # lookup parquets. The API server / Streamlit load this to compute
    # `published_elo` on the fly from the raw aggregates.
    print("Computing shrinkage metadata...")
    shrinkage_meta = _compute_shrinkage_metadata(player_elo_lookup, pair_elo_lookup)
    meta_filename = f'acbl_{club_or_tournament}_elo_shrinkage.json'
    meta_file = out_path.joinpath(meta_filename)
    meta_file.write_text(json.dumps(shrinkage_meta, indent=2), encoding='utf-8')
    print(f"Saved {meta_filename}: player_prior={shrinkage_meta.get('player', {}).get('prior_anchor')} "
          f"pair_prior={shrinkage_meta.get('pair', {}).get('prior_anchor')}")

    return df, player_elo_lookup, pair_elo_lookup


def create_elo_ratings(modes, out_dir=None):
    """Elo for both sides (the player Elo needs both); write the files of ``modes``.

    out_dir: where outputs are written (default: acblPath). Inputs and the
    previous-run seed parquets are always read from acblPath.
    """
    out_path = pathlib.Path(out_dir) if out_dir else acblPath
    out_path.mkdir(parents=True, exist_ok=True)
    sides = {side: load_side(side) for side in SIDES}
    print("Dating platinum awards...")
    plat_cum = dated_platinum_awards(
        acblPath.joinpath(TOURNAMENT_AWARDS_FILENAME),
        sides["tournament"].select("session_id", "Date").unique("session_id"),
    )
    print(f"platinum award dates: {plat_cum.height:,} player-dates, last {plat_cum['Date'].max()}")
    sides = {side: add_start_ratings(df, plat_cum) for side, df in sides.items()}
    seeds = _read_pooled_player_seeds([acblPath.joinpath(f'acbl_{s}_elo_ratings.parquet') for s in SIDES])
    player = joint_player_elo(sides, seeds)
    for side in SIDES:
        df = sides.pop(side)
        if side in modes:
            t = time.time()
            write_side(side, df, player.pop(side), out_path)
            print(f"{side} written in {time.time() - t:.0f}s")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print("-" * 70, "\n")
        del df


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create ACBL Elo ratings.")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    parser.add_argument("--out-dir", default=None, help="Write outputs here instead of e:/bridge/data/acbl")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        return ["club", "tournament"], args.out_dir
    modes = []
    if args.club:
        modes.append("club")
    if args.tournament:
        modes.append("tournament")
    return modes, args.out_dir


if __name__ == "__main__":

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    modes, out_dir = _parse_club_tournament_args()
    create_elo_ratings(modes, out_dir=out_dir)

    print_ended(program_start_time)


