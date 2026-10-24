#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_elo_ratings_create.py

Takes 6m/1m30s 50m/15m rows by 26->61/24->60 columns. input filesize 43GB->3.5GB/16GB->1.3GB. Output 8.8m/2.8m players 104MB/24MB, 4.4m/1.4m pairs file 23MB/24MB. Uses 128GB memory/pagefile.

Computes Elo ratings for players and pairs from board results.
Creates lookup tables for player and pair Elo ratings.

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

import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

from acbl_strata import club_strata_bucket, tournament_strata_bucket

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


# Default "evidence weight" of the prior in the published (Bayesian-shrunk)
# Elo formula. The Streamlit/API can override at query time.
SHRINKAGE_DEFAULT_PRIOR_SESSIONS = 50

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


def create_elo_ratings(club_or_tournament, out_dir=None):
    """Create Elo ratings and lookup tables for players and pairs.

    out_dir: where outputs are written (default: acblPath). Inputs and the
    previous-run seed parquet are always read from acblPath.
    """
    out_path = pathlib.Path(out_dir) if out_dir else acblPath
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Processing {club_or_tournament} Elo ratings...")
    # Read K-dampening global-stdev seeds from a previous run's parquet (if
    # any). First run after the field-relative math change picks up nothing
    # and falls back to STRATUM_DAMPENING_FALLBACK_GLOBAL_STDEV (50); the
    # next run reads the realistic empirical seed (~45 for ACBL today).
    previous_elo_ratings_file = acblPath.joinpath(
        f'acbl_{club_or_tournament}_elo_ratings.parquet'
    )
    seeds = _read_global_stdev_seeds(previous_elo_ratings_file)
    if seeds:
        print(f"K-dampening seeds from previous parquet: {seeds}")
    else:
        print("No K-dampening seeds available; using built-in fallback.")
    
    # Load board results schema
    acbl_board_results_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    df = pl.read_parquet(acbl_board_results_file, n_rows=0)
    print(f"Loaded: {acbl_board_results_filename} shape:{df.shape} size:{acbl_board_results_file.stat().st_size}")
    
    # Select only necessary columns
    regex_cols = [
            'Date', 'is_virtual_game', 'session_id', 'event_id', 'event_name',
            'Round', 'Board', 'Pair_Number_(NS|EW)',
            'Player_ID_[NESW]', 'Player_Name_[NESW]', 'MP_(NS|EW)', 'MP_Top',
            'MasterPoints_[NESW]', 'MasterPoints_(NS|EW)', 'Pct_NS',
            'DD_Tricks_Diff', 'Is_Par_Suit', 'Is_Par_Contract', 'Is_Sacrifice', # must use board results augmented to get these columns
            'Declarer_Direction', 'Declarer_Pair_Direction', 'BidSuit', 'ParContracts',
            'DD_Score_NS', 'DD_Score_EW', 'DD_Score_Declarer',
            'Par_NS', 'Par_EW', 'Par_Declarer',
            'mp_limit',  # event MP ceiling for Strata filter (club mpLimits / tournament mp_limit)
            'mp_color', 'mp_rating',  # tournament platinum identity; unused on club
        ]
    read_cols = [col for col in df.columns if any(re.match(regex, col) for regex in regex_cols)]
    
    # load board results data
    df = pl.read_parquet(acbl_board_results_file, columns=read_cols) #, n_rows=1000000) # temp!!!!!!!
    print(f"Loaded: {acbl_board_results_filename} shape:{df.shape} size:{acbl_board_results_file.stat().st_size}")
    
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

    df = _sanitize_session_mp_top(df)
    
    # Compute Elo ratings
    print("Computing matchpoint Elo ratings...")
    df = mlBridgeAugmentLib.compute_matchpoint_elo_ratings(
        df,
        minimum_sessions=PERSISTED_ELO_MINIMUM_SESSIONS,
        pair_global_stdev_ns=seeds.get("pair_ns"),
        pair_global_stdev_ew=seeds.get("pair_ew"),
        player_global_stdev_ns=seeds.get("player_ns"),
        player_global_stdev_ew=seeds.get("player_ew"),
    )
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
    for club_or_tournament in modes:
        t = time.time()
        create_elo_ratings(club_or_tournament, out_dir=out_dir)
        print(f"{club_or_tournament} processing elapsed time in seconds: {time.time() - t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)


