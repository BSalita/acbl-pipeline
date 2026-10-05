#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_club_results_sql_to_board_results_clean.py

Takes 6m/5m (tournament is slower due to enriching with club MasterPoints) for 185m->134m/24m->22m by 44/44 columns. filesize 3.4GB/584MB. Requires 100GB RAM.

Performs following steps:
    1) Read sqlite db into dataframes.
    2) Clean data.
    3) Avoids creating new columns other than renames.
    4) Create a single dataframe suitable for board/player analysis.
    5) If tournament, enrich with club MasterPoints.
    6) Write dataframe to disk using parquet (fastest).

Requirements:
    pip install adbc-driver-sqlite pyarrow

Previous steps:
    acbl_club_results_download_to_json.ipynb
    acbl_club_results_json_to_sql.ipynb creates sqlite:///acbl_club_results.sqlite

Next steps:
    acbl_club_results_board_results_augment_step1.ipynb
    acbl_club_results_board_results_dicts.ipynb -- might be obsolete?
    acbl_club_results_board_results_augment_step2.ipynb
    acbl_elo_ratings_create.ipynb
    acbl_club_model_data.ipynb
    acbl_predict_torch.ipynb

TODO:
    - Only process PAIRS?
    - Enable del to minimize memory usage?
    - Why is tricks_taken erroring out? Must have a str cell.
"""

import polars as pl
import pathlib
import time
import sys

from acbl_sqlite_read import read_sqlite_query

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
import mlBridge.mlBridgeLib as mlBridgeLib

# override pandas display options
mlBridgeLib.pd_options_display()

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')

# A clean pair id list is null or a JSON array of numbers. Some tournament
# rows are truncated mid-name, e.g. [1590227,Al Ober], and json_decode aborts
# the whole column on the first one.
_PAIR_ID_CLEAN = r'^(?i)\s*(?:null|\[(?:\s*"?[0-9]+"?(?:\s*,\s*"?[0-9]+"?)*)?\s*\])\s*$'
_PAIR_ID_TOKEN = r'#[0-9]+|[0-9]+'


def pair_id_list(column: str) -> pl.Expr:
    """Player-number tokens from a pair id column. Names and broken JSON are skipped."""
    return pl.col(column).str.extract_all(_PAIR_ID_TOKEN).alias(column)


def get_club_schemas():
    """Return SQL selects and schemas for club data."""
    sql_selects_d = {
        # mpLimits aliased to mp_limit so club/tournament Elo share one strata column.
        'events':'SELECT id AS event_id, club_id_number, type AS event_type, board_scoring_method, tb_count, club_session, club_class, is_virtual_game, virtualGameType, mpLimits AS mp_limit FROM events',
        'board_results':'SELECT id AS board_result_id, board_id, round_number, table_number, CAST(ns_pair AS INTEGER) AS ns_pair, CAST(ew_pair AS INTEGER) AS ew_pair, ns_score, ew_score, contract, declarer, ew_match_points, ns_match_points, opening_lead, result, tricks_taken FROM board_results',
        'boards':'SELECT id AS board_id, section_id, board_number FROM boards',
        'pair_summaries':'SELECT id AS pair_summary_id, section_id, CAST(pair_number AS INTEGER) AS pair_number, direction FROM pair_summaries',
        'players':'SELECT id AS player_id, pair_summary_id, CAST(id_number AS INTEGER) AS player_number, name AS player_name, CAST(mp_total AS REAL) AS mp_total FROM players',
        'sessions':'SELECT id AS session_id, event_id, hand_record_id, game_date FROM sessions',
        'sections':'SELECT id AS section_id, session_id, name AS section_name FROM sections'
    }
    
    schema_d = {
        'events':
        {
            'event_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'club_id_number': pl.Int64,
            'event_type': pl.String,
            'board_scoring_method': pl.String,
            'tb_count': pl.Float32,
            'club_session': pl.String,
            'club_class': pl.UInt8,
            'is_virtual_game': pl.Boolean,
            'virtualGameType': pl.UInt8,
            'mp_limit': pl.String,
        },
        'board_results':
        {
            'board_result_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'board_id': pl.Int64, # aka hand_record_id
            'round_number': pl.String,
            'table_number': pl.String,
            'ns_pair': pl.UInt16,
            'ew_pair': pl.UInt16,
            'ns_score': pl.String,
            'ew_score': pl.String,
            'contract': pl.String,
            'declarer': pl.String,
            'ew_match_points': pl.Float32,
            'ns_match_points': pl.Float32,
            'opening_lead': pl.String,
            'result': pl.String,
            'tricks_taken': pl.String,
            #'board_results_add_ons': pl.Int64,
            #'board_results_addons': pl.Int64,
        },
        'boards':
        {
            'board_id': pl.Int64, # aka hand_record_id
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'section_id': pl.Int64,
            'board_number': pl.UInt8,
        },
        'pair_summaries':
        {
            'pair_summary_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'section_id': pl.Int64,
            'pair_number': pl.UInt8,
            'direction': pl.String,
        },
        'players':
        {
            'player_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'pair_summary_id': pl.Int64,
            'player_number': pl.String,
            'player_name': pl.String,
            'mp_total': pl.Float32,
        },
        'sessions':
        {
            'session_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'event_id': pl.Int64,
            'hand_record_id': pl.String,
            'game_date': pl.String,
        },
        'sections':
        {
            'section_id': pl.Int64,
            #'created_at': pl.Datetime,
            #'updated_at': pl.Datetime,
            'session_id': pl.Int64,
            'section_name': pl.String,
        },
    }
    
    return sql_selects_d, schema_d


def get_tournament_schemas():
    """Return SQL selects and schemas for tournament data."""
    sql_selects_d = {
        'event':'SELECT id as event_id, name as event_name, start_date AS game_date, start_time AS game_time, game_type, event_type, mp_limit, mp_color, mp_rating, session_count FROM event',
        'board_results':'SELECT id AS board_result_id, session AS session_id, board_number, orientation, contract, declarer, score, match_points, percentage, pair_number, pair_acbl, pair_names, opponent_pair_number, opponent_pair_names FROM board_results',
        'sections': 'SELECT id AS section_id, session_id, section_label AS section_name, movement_type, scoring_type FROM sections',
    }
    
    schema_d = {
        'board_results':
        {
            "board_result_id": pl.String,
            "session_id": pl.String,
            "board_number": pl.UInt8,
            "orientation": pl.String,
            "contract": pl.String,
            "declarer": pl.String,
            "score": pl.String,
            "match_points": pl.Float32,
            "percentage": pl.Float32,
            "pair_number": pl.Int32,
            "pair_acbl": pl.String,
            "pair_names": pl.String,
            "opponent_pair_number": pl.Int32,
            "opponent_pair_names": pl.String,
        },
        'event':
        {
            'event_id': pl.String,
            'event_name': pl.String,
            'game_date': pl.String,
            'game_time': pl.String,
            'game_type': pl.String,
            'event_type': pl.String,
            'mp_limit': pl.String,
            'mp_color': pl.String,
            'mp_rating': pl.String,
            'session_count': pl.UInt8,
        },
        'sections':
        {
            'section_id': pl.String,
            'session_id': pl.String,
            'section_name': pl.String,
            'movement_type': pl.String,
            'scoring_type': pl.String,
        },
    }
    
    return sql_selects_d, schema_d


def club_board_results_clean():
    """Process club board results."""
    print("Processing club board results...")
    
    sql_selects_d, schema_d = get_club_schemas()
    
    db_connection_string = 'sqlite:///'+acblPath.joinpath(f'acbl_club_results.sqlite').as_posix()
    print(f"Database connection string: {db_connection_string}")
    
    uri = db_connection_string
    dfs_adbc = {}
    for table, schema in schema_d.items():
        print(f"Reading table: {table}")
        dfs_adbc[table] = read_sqlite_query(uri, sql_selects_d[table], schema)
    print(f"Tables read: {list(dfs_adbc.keys())}")
    
    # takes 1m30s/1m
    board_results = dfs_adbc['board_results']
    boards = dfs_adbc['boards']
    brs_df = board_results.join(boards, on='board_id', how='left')
    del board_results, boards
    
    print("Cleaning declarer and scores...")
    brs_df = brs_df.filter(pl.col('declarer').is_null() | pl.col('declarer').is_in(['N','E','S','W']))
    brs_df = brs_df.with_columns(
        pl.when(pl.col('ns_score') == 'PASS').then(pl.lit('0')).otherwise(pl.col('ns_score')).cast(pl.Int16, strict=False).alias('ns_score'),
        pl.when(pl.col('ew_score') == 'PASS').then(pl.lit('0')).otherwise(pl.col('ew_score')).cast(pl.Int16, strict=False).alias('ew_score')
    )
    
    sections = dfs_adbc['sections']
    br_b_sections_df = brs_df.join(sections, on='section_id', how='left')
    del brs_df, sections
    print(f"After sections join: {br_b_sections_df.shape}")
    
    # Process players
    print("Processing players...")
    players = dfs_adbc['players']
    # remove '(swap names)' suffix that's in some player names. Appears to be legacy garbage.
    players = players.with_columns(
        pl.col('player_name').str.replace(r'\s*\(swap names\)\s*$', '', literal=False)
    )
    keep_cols_ns = ['pair_summary_id', 'player_number', 'player_name', 'mp_total']
    keep_cols_ew = ['pair_summary_id', 'player_number', 'player_name', 'mp_total']
    
    player_n = (
        players.unique(subset=['pair_summary_id'], keep='first')
        .select(keep_cols_ns)
        .rename({'player_number': 'player_number_n', 'player_name': 'player_name_n', 'mp_total': 'mp_total_n'})
    )
    player_s = (
        players.unique(subset=['pair_summary_id'], keep='last')
        .select(keep_cols_ns)
        .rename({'player_number': 'player_number_s', 'player_name': 'player_name_s', 'mp_total': 'mp_total_s'})
    )
    player_e = (
        players.unique(subset=['pair_summary_id'], keep='first')
        .select(keep_cols_ew)
        .rename({'player_number': 'player_number_e', 'player_name': 'player_name_e', 'mp_total': 'mp_total_e'})
    )
    player_w = (
        players.unique(subset=['pair_summary_id'], keep='last')
        .select(keep_cols_ew)
        .rename({'player_number': 'player_number_w', 'player_name': 'player_name_w', 'mp_total': 'mp_total_w'})
    )
    
    assert len(set([player_n.height, player_s.height, player_e.height, player_w.height])) == 1
    del players, keep_cols_ns, keep_cols_ew
    
    players_ns = player_n.join(player_s, on='pair_summary_id')
    players_ew = player_e.join(player_w, on='pair_summary_id')
    del player_n, player_s, player_e, player_w
    
    # Join sessions
    print("Joining sessions...")
    sessions = dfs_adbc['sessions']
    br_b_sections_sessions_df = br_b_sections_df.join(sessions, on='session_id', how='left')
    del br_b_sections_df, sessions
    
    # Join events
    print("Joining events...")
    events = dfs_adbc['events']
    br_b_sections_sessions_events_df = br_b_sections_sessions_df.join(events, on='event_id', how='left')
    del br_b_sections_sessions_df, events
    
    # Join pair summaries
    print("Joining pair summaries...")
    pair_summaries = dfs_adbc['pair_summaries']
    
    pairs_ns_df = br_b_sections_sessions_events_df[['section_id', 'ns_pair']].join(
        pair_summaries.filter(pl.col('direction').eq('NS')),
        left_on=('section_id', 'ns_pair'),
        right_on=('section_id', 'pair_number'),
        how='left'
    )
    
    pairs_ew_df = br_b_sections_sessions_events_df[['section_id', 'ew_pair']].join(
        pair_summaries.filter(pl.col('direction').eq('EW')),
        left_on=('section_id', 'ew_pair'),
        right_on=('section_id', 'pair_number'),
        how='left'
    )
    
    br_b_sections_sessions_events_df = br_b_sections_sessions_events_df.with_columns(
        pairs_ns_df['pair_summary_id'].alias('ns_pair_summary_id'),
        pairs_ew_df['pair_summary_id'].alias('ew_pair_summary_id')
    )
    
    br_b_sections_sessions_events_df = br_b_sections_sessions_events_df.drop_nulls(['ns_pair_summary_id', 'ew_pair_summary_id'])
    print(f"After filtering nulls: {br_b_sections_sessions_events_df.shape}")
    
    del pair_summaries, pairs_ns_df, pairs_ew_df
    
    # Join player data
    print("Joining player data...")
    psid_ns = br_b_sections_sessions_events_df[['ns_pair_summary_id']].join(
        players_ns, 
        left_on='ns_pair_summary_id', 
        right_on='pair_summary_id', 
        how='left'
    )
    psid_ew = br_b_sections_sessions_events_df[['ew_pair_summary_id']].join(
        players_ew, 
        left_on='ew_pair_summary_id', 
        right_on='pair_summary_id', 
        how='left'
    )
    
    psid_ns = psid_ns.drop(['ns_pair_summary_id'])
    psid_ew = psid_ew.drop(['ew_pair_summary_id'])
    
    br_b_sections_sessions_events_df = pl.concat(
        [br_b_sections_sessions_events_df, psid_ns, psid_ew], 
        how='horizontal'
    )
    
    del players_ns, players_ew, psid_ns, psid_ew
    
    # Final cleanup
    print("Final cleanup and renaming...")
    brs_df = br_b_sections_sessions_events_df
    brs_df = brs_df.filter(pl.col('game_date').ge('2019-01-01'))
    brs_df = brs_df.rename({
        'game_date': 'Date',
        'board_number': 'Board',
        'round_number': 'Round',
        'contract': 'Contract',
        'declarer': 'Declarer_Direction',
        'ns_score': 'Score_NS',
        'ew_score': 'Score_EW',
        'ns_match_points': 'MP_NS',
        'ew_match_points': 'MP_EW',
        'ns_pair': 'Pair_Number_NS',
        'ew_pair': 'Pair_Number_EW',
        'player_number_n': 'Player_ID_N',
        'player_number_s': 'Player_ID_S',
        'player_number_e': 'Player_ID_E',
        'player_number_w': 'Player_ID_W',
        'player_name_n': 'Player_Name_N',
        'player_name_s': 'Player_Name_S',
        'player_name_e': 'Player_Name_E',
        'player_name_w': 'Player_Name_W',
        'mp_total_n': 'MasterPoints_N',
        'mp_total_s': 'MasterPoints_S',
        'mp_total_e': 'MasterPoints_E',
        'mp_total_w': 'MasterPoints_W',
    })

    acbl_board_results_filename = f'acbl_club_board_results_cleaned.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    brs_df.write_parquet(acbl_board_results_file)
    print(f"Saved {acbl_board_results_filename}: shape:{brs_df.shape}, size:{acbl_board_results_file.stat().st_size}")
    
    return brs_df


def tournament_board_results_clean():
    """Process tournament board results."""
    print("Processing tournament board results...")
    
    sql_selects_d, schema_d = get_tournament_schemas()
    
    db_connection_string = 'sqlite:///'+acblPath.joinpath(f'acbl_tournament_results.sqlite').as_posix()
    print(f"Database connection string: {db_connection_string}")
    
    # Read tables
    uri = db_connection_string
    dfs_adbc = {}
    for table, schema in schema_d.items():
        print(f"Reading table: {table}")
        dfs_adbc[table] = read_sqlite_query(uri, sql_selects_d[table], schema)
    print(f"Tables read: {list(dfs_adbc.keys())}")
    
    # Process board results
    print("Processing board results...")
    brs_df = dfs_adbc['board_results']

    brs_df = brs_df.with_columns(
        pl.when(pl.col('score').eq('PASS'))
        .then(pl.lit('PASS'))
        .otherwise(pl.col('contract'))
        .alias('contract'),
        (pl.col('percentage') / 100).cast(pl.Float32).alias('percentage'),
    )

    # Build NS/EW views
    ns = brs_df.filter(pl.col('orientation') == 'N-S').drop(['orientation'])
    ew = brs_df.filter(pl.col('orientation') == 'E-W').drop(['orientation'])

    # Join without board_result_id or name fields
    brs_df = ns.join(
        ew,
        left_on=['session_id', 'board_number', 'contract', 'declarer', 'pair_number', 'pair_names'], 
        right_on=['session_id', 'board_number', 'contract', 'declarer', 'opponent_pair_number', 'opponent_pair_names'], 
        how='inner',
        suffix='_ew',
    ).drop(['opponent_pair_number', 'opponent_pair_names'])

    for col in brs_df.columns:
        if col.endswith('_ew'):
            brs_df = brs_df.rename({col[:-3]: col.replace('_ew', '_ns')})
    
    brs_df = brs_df.filter(
        pl.col('contract').ne('PASS') | 
        pl.col('match_points_ns').ne(0) | 
        pl.col('percentage_ns').ne(0)
    )
    
    # Process columns
    print("Processing and cleaning columns...")
    for column in ("pair_acbl_ns", "pair_acbl_ew"):
        messy = brs_df.filter(
            pl.col(column).is_not_null() & ~pl.col(column).str.contains(_PAIR_ID_CLEAN)
        ).height
        print(f"Pair ids with non-numeric text in {column}: {messy}")
    brs_df = brs_df.with_columns(
        pl.when(pl.col('declarer').is_null() | pl.col('declarer').eq('')).then(None).otherwise('declarer').replace_strict(mlBridgeLib.Direction_to_NESW_d, return_dtype=pl.Utf8).alias('declarer'),
        pl.when(pl.col('score_ns') == 'PASS').then(0).otherwise(pl.col('score_ns').cast(pl.Int16, strict=False)).alias('score_ns'),
        pl.when(pl.col('score_ew') == 'PASS').then(0).otherwise(pl.col('score_ew').cast(pl.Int16, strict=False)).alias('score_ew'),
        pl.concat_str([
            pl.col('session_id'),
            pl.lit('-'),
            pl.col('board_number').sub(1).cast(pl.String)
        ]).alias('hand_record_id'),
        pl.col('board_result_id_ns').str.split('-').list.slice(0, 4).list.join('-').alias('section_id'),
        pair_id_list("pair_acbl_ns"),
        pl.col("pair_names_ns")
            .str.replace_all(r'^\[|\]$', '')
            .str.replace_all(r'"', '')
            # remove '(swap names)' suffix that's in some player names. Doesn't seem to be any in tournament data.
            #.str.replace(r'\s*\(swap names\)\s*$', '', literal=False)
            .str.split(","),
        pair_id_list("pair_acbl_ew"),
        pl.col("pair_names_ew")
            .str.replace_all(r'^\[|\]$', '')
            .str.replace_all(r'"', '')
            # remove '(swap names)' suffix that's in some player names. Doesn't seem to be any in tournament data.
            #.str.replace(r'\s*\(swap names\)\s*$', '', literal=False)
            .str.split(","),
    )
    
    # Ensure pair lists have exactly 2 elements
    for c in ["pair_acbl_ns", "pair_acbl_ew", "pair_names_ns", "pair_names_ew"]:
        brs_df = brs_df.with_columns(
            pl.when(pl.col(c).list.len() == 0)
            .then(pl.lit([None, None]).cast(pl.List(pl.String)))
            .when(pl.col(c).list.len() == 1)
            .then(pl.concat_list([
                pl.col(c).cast(pl.List(pl.String)),
                pl.lit([None]).cast(pl.List(pl.String))
            ]))
            .otherwise(pl.col(c).cast(pl.List(pl.String)).list.slice(0, 2))
            .alias(c)
        )
    
    # Extract individual players
    brs_df = brs_df.with_columns(
        pl.col('pair_acbl_ns').list.get(0).alias('Player_ID_N'),
        pl.col('pair_acbl_ns').list.get(1).alias('Player_ID_S'),
        pl.col('pair_acbl_ew').list.get(0).alias('Player_ID_E'),
        pl.col('pair_acbl_ew').list.get(1).alias('Player_ID_W'),
        pl.col('pair_names_ns').list.get(0).alias('Player_Name_N'),
        pl.col('pair_names_ns').list.get(1).alias('Player_Name_S'),
        pl.col('pair_names_ew').list.get(0).alias('Player_Name_E'),
        pl.col('pair_names_ew').list.get(1).alias('Player_Name_W'),
    )
    
    # Join event data
    print("Joining event data...")
    event_df = dfs_adbc['event']
    brs_df = brs_df.with_columns(
        pl.col('session_id').str.split('-').list.slice(0, 2).list.join('-').alias('event_id')
    )
    brs_df = brs_df.join(event_df, on='event_id', how='inner')
    
    # Join sections
    print("Joining sections...")
    sections_df = dfs_adbc['sections']
    brs_df = brs_df.join(sections_df, on=['section_id', 'session_id'], how='inner')
    
    # Filter by date
    brs_df = brs_df.filter(pl.col('game_date').ge('2015-02-01'))
    
    # Rename columns
    brs_df = brs_df.rename({
        'game_date': 'Date',
        'board_number': 'Board',
        'contract': 'Contract',
        'declarer': 'Declarer_Direction',
        'score_ns': 'Score_NS',
        'score_ew': 'Score_EW',
        'match_points_ns': 'MP_NS',
        'match_points_ew': 'MP_EW',
        'percentage_ns': 'Pct_NS',
        'percentage_ew': 'Pct_EW',
        'pair_acbl_ns': 'Pair_IDs_NS',
        'pair_acbl_ew': 'Pair_IDs_EW',
        'pair_names_ns': 'Pair_Names_NS',
        'pair_names_ew': 'Pair_Names_EW',
        'pair_number_ns': 'Pair_Number_NS',
        'pair_number_ew': 'Pair_Number_EW',
    })
    
    # Enrich tournament data with club's MasterPoints
    print("Enriching tournament data with club MasterPoints (takes 1 minute)...")
    acbl_board_results_filename = f'acbl_club_board_results_cleaned.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    
    if acbl_board_results_file.exists():
        club_df = pl.read_parquet(acbl_board_results_file)
        print(f"Loaded {acbl_board_results_filename}: shape:{club_df.shape}, size:{acbl_board_results_file.stat().st_size}")
        
        club_players = []
        for pos in "NESW":
            if f"Player_ID_{pos}" in club_df.columns and f"MasterPoints_{pos}" in club_df.columns:
                club_players.append(
                    club_df.select([
                        pl.col(f"Player_ID_{pos}").alias("Player_ID"),
                        pl.col(f"MasterPoints_{pos}").alias("MasterPoints"),
                        pl.col("Date")
                    ]).filter(pl.col("Player_ID").is_not_null())
                )
        
        if club_players:
            # Combine all club player data and get latest MasterPoints per player
            club_players_combined = pl.concat(club_players, how="vertical")
            latest_masterpoints = (
                club_players_combined
                .sort("Date", descending=True)
                .group_by("Player_ID")
                .agg([
                    pl.col("MasterPoints").first().alias("MasterPoints")
                ])
                # Club ids are integers. Tournament ids stay strings because a
                # few are temporary numbers such as #155.
                .with_columns(pl.col("Player_ID").cast(pl.String))
            )
            
            # Join MasterPoints to tournament data for each position
            for pos in "NESW":
                if f"Player_ID_{pos}" in brs_df.columns:
                    brs_df = brs_df.join(
                        latest_masterpoints.select([
                            pl.col("Player_ID"),
                            pl.col("MasterPoints").alias(f"MasterPoints_{pos}")
                        ]),
                        left_on=f"Player_ID_{pos}",
                        right_on="Player_ID",
                        how="left"
                    )
            print("MasterPoints enrichment complete")
        else:
            print("No club player data found for enrichment")
    else:
        print(f"Club data file not found: {acbl_board_results_file}")
        print("Skipping MasterPoints enrichment")
    
    # Save
    acbl_board_results_filename = f'acbl_tournament_board_results_cleaned.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    brs_df.write_parquet(acbl_board_results_file)
    print(f"Saved {acbl_board_results_filename}: shape:{brs_df.shape}, size:{acbl_board_results_file.stat().st_size}")
    
    return brs_df


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Clean ACBL board results from SQLite.")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        return ["club", "tournament"]
    modes = []
    if args.club:
        modes.append("club")
    if args.tournament:
        modes.append("tournament")
    return modes


if __name__ == "__main__":

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    for club_or_tournament in _parse_club_tournament_args():
        t = time.time()
        if club_or_tournament == "club":
            club_board_results_clean()
        elif club_or_tournament == "tournament":
            # Uses existing acbl_club_board_results_cleaned.parquet for MasterPoints enrichment.
            tournament_board_results_clean()
        else:
            raise ValueError(f"Invalid club_or_tournament: {club_or_tournament}")
        print(f"{club_or_tournament} board results clean elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)

