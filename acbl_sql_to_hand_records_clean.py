#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_sql_to_hand_records_clean.py

Takes 53m for 24m->17m/21m->1.7m rows by 10->8/29->27 columns. 481MB/75MB file.
TODO: why are there 6m empty dd/par rows? Why are they being dropped?
TODO: invalid_acbl_pars is broken. Must fix before next run.
TODO: convert hands directly to PBN using parallel processing instead of to hands->brs->pbn

Read acbl_club_results.sqlite file and create hand records df, clean data.
Create ../acbl/acbl_hand_records_cleaned.parquet

Requirements:
    pip install adbc-driver-sqlite pyarrow

Next steps:
    obsolete - acbl_club_results_hand_records_generate_single_dummy_deals.ipynb
    acbl_club_results_hand_records_augment.ipynb
    acbl_tournament_hand_records_augment.ipynb

Previous steps:
    acbl_club_download_to_json.ipynb
    acbl_club_json_to_sql.ipynb creates sqlite:///acbl_club_results.sqlite
    acbl_tournament_events_download_to_json.ipynb
    acbl_tournament_session_json_to_sql.ipynb creates sqlite:///acbl_tournament_results.sqlite
"""

import polars as pl
import pathlib
import time
import re
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
sys.path
import mlBridge.mlBridgeLib as mlBridgeLib


rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


_PAR_CONTRACT_RE = re.compile(r'(\d)([CDHSN])(\**)-(NS|EW|[NSEW])([\+\-]\d)?')


def _parse_acbl_par(value: str) -> tuple | None:
    """Parse ``Par: <score> <contracts>``.

    Tournament all-pass rows are stored as ``0``, which becomes ``Par: 0``.
    Club rows and the other tournament rows are ``Par: +140 3H-NS/3S-NS``.
    Blank pars become ``Par:`` with no score. A leading ``?`` is an unknown
    sign. Both are returned as None so the caller can drop those rows.
    """
    parts = value.split()
    if len(parts) < 2 or parts[0] != 'Par:':
        return None
    try:
        score = int(parts[1])
    except ValueError:
        return None
    if score == 0 and len(parts) == 2:
        return (0, [(0, '', '', '', 0)])
    if len(parts) != 3:
        return None
    if score == 0:
        return (0, [(0, '', '', '', 0)])
    contracts = []
    for contract in parts[2].replace('NT', 'N').split('/'):
        bid = _PAR_CONTRACT_RE.match(contract)
        if bid is None:
            return None
        level, suit, double, direction, result = bid.groups()
        contracts.append((int(level), suit, double, direction, 0 if result is None else int(result)))
    if not contracts:
        return None
    return (score, contracts)


def clean_hand_records(hrs_df):
    # common to club and tournament
    print(f"Initial hrs_df shape: {hrs_df.shape}")

    hrs_renames = {
        'board': 'Board',
        'dealer': 'Dealer',
        'vulnerability': 'Vul',
        'double_dummy_ns': 'ACBL_double_dummy_ns',
        'double_dummy_ew': 'ACBL_double_dummy_ew',
        'par': 'ACBL_Par',
    }
    hrs_df = hrs_df.rename(hrs_renames)

    # takes 2s
    # drop rows having board numbers > 255. doesn't pass the sniff test.
    drop_rows = hrs_df.filter(pl.col('Board') > 255)
    print(f"Dropping {drop_rows.shape[0]} rows with Board > 255")
    if drop_rows.shape[0] > 0:
        print(drop_rows['Board'].value_counts())
    hrs_df = hrs_df.join(drop_rows, on=hrs_df.columns, how='anti') # Thanks to chatgpt for this trick. I had no idea anti-join was a thing.

    print("Dealer value counts:")
    print(hrs_df['Dealer'].value_counts())
    hrs_df = hrs_df.with_columns(pl.col('Dealer').replace_strict(mlBridgeLib.Direction_to_NESW_d, return_dtype=pl.Utf8))
    print("After normalization:")
    print(hrs_df['Dealer'].value_counts())

    # rename types of vuls to a common name
    # todo: use mlBridgeLib vul dict
    print("Vul value counts:")
    print(hrs_df['Vul'].value_counts())
    hrs_df = hrs_df.with_columns(pl.col('Vul').replace_strict(mlBridgeLib.Vulnerability_to_Vul_d,return_dtype=pl.Utf8))
    hrs_df = hrs_df.filter(pl.col('Vul').is_not_null())
    print("After normalization:")
    print(hrs_df['Vul'].value_counts())

    # takes 5s
    # drop rows having board numbers with non-normal vuls. e.g. Board 1 having N-S vul. around 3500 instances.
    drop_rows = pl.Series([v != mlBridgeLib.vul_syms[mlBridgeLib.BoardNumberToVul(bn)] for bn,v in zip(hrs_df['Board'],hrs_df['Vul'])])
    print(f"Dropping {drop_rows.sum()} rows with non-normal vulnerability")
    hrs_df = hrs_df.filter(~drop_rows)
    print(f"After vul filter: {hrs_df.shape}")

    # takes 1s
    hrs_df = hrs_df.with_columns(pl.col('board_record_string').str.replace_all('-', ''))

    # takes 1m30s/8s
    # drop results whose board_record_string contains invalid characters (5492639, 13290144) or wrong number of cards (4754685, 6213531)
    print("Validating board_record_strings...")
    hrs_df = hrs_df.filter(pl.col('board_record_string').map_elements(mlBridgeLib.validate_brs,return_dtype=pl.Boolean))
    print(f"After validation: {hrs_df.shape}")

    # takes 1s
    # drop rows which have empty double dummy or par columns. They're probably all SHUFFLE.
    b1 = hrs_df['ACBL_double_dummy_ns'] != ''
    b2 = hrs_df['ACBL_double_dummy_ew'] != ''
    b3 = hrs_df['ACBL_Par'] != ''
    assert all(b1 == b2) and all(b1 == b3)
    print(f"Dropping {hrs_df.filter(~b1).shape[0]} rows with empty DD/Par columns")
    hrs_df = hrs_df.filter(b1) # drop rows which have empty double dummy or par columns

    # takes 5s
    # clean double dummy columns. rewrite into standarized form.
    # todo: cleaning and validations need to be done in previous notebook
    dd_cols = ['ACBL_double_dummy_ns','ACBL_double_dummy_ew']
    for d in dd_cols:
        assert hrs_df[d].str.starts_with(d[-2:].upper()+':').all()
        hrs_df =  hrs_df.with_columns(pl.col(d).str.replace_all('NT','N').str.replace_all(r'  ',' ')) # unsure why str.replace(r'\s+',' ') doesn't remove double spaces.
        assert not hrs_df[d].str.contains('NT').any()
        b = hrs_df[d].str.count_matches(' ').ne(5) | ~hrs_df[d].str.starts_with(d[-2:].upper()+': ')
        wrong_counts = hrs_df.filter(b)
        print(f"Dropping {wrong_counts.shape[0]} rows with wrong DD format for {d}")
        hrs_df = hrs_df.filter(~b)

    # takes 3m30s/20s
    # rewrite double dummy columns into more usable format
    # put into acbllib.py
    print("Processing double dummy makes...")
    ACBL_DDmakes_l = []
    zdd = zip(hrs_df['ACBL_double_dummy_ns'],hrs_df['ACBL_double_dummy_ew'])
    for nsewdd in zdd:
        tricks = {}
        for dd in nsewdd:
            split_space = dd.split(' ')
            ds = split_space[0][:2].upper()
            assert len(split_space) == 6
            for ddsuit in split_space[1:]:
                found = re.findall(r'(^([CDHSN])(\d+)$)|(^(\d+)([CDHSN])$)|(^([CDHSN])(\d+)\/(\d+)$)|(^(\d+)\/(\d+)([CDHSN])$)',ddsuit)
                assert len(found) == 1
                r = found[0]
                if r[0] != '': # S4 -- direction makes 4 tricks in spades
                    assert r[1] in 'CDHSN' and r[2].isdigit()
                    suit = r[1]
                    levelNE = levelSW = int(r[2])
                elif r[3] != '': # 4S -- direction makes 10 tricks in spades
                    assert r[4].isdigit() and r[5] in 'CDHSN'
                    levelNE = levelSW = int(r[4])+6
                    suit = r[5]
                elif r[6] != '': # S4/5 -- direction (N or E) makes 4 tricks in spades, direction (S or W) makes 5 tricks
                    assert r[7] in 'CDHSN' and r[8].isdigit() and r[9].isdigit()
                    suit = r[7]
                    levelNE = int(r[8])
                    levelSW = int(r[9])
                elif r[10] != '': # S4/5 -- direction (N or E) makes 10 tricks in spades, direction (S or W) makes 11 tricks
                    assert r[11].isdigit() and r[12].isdigit() and r[13] in 'CDHSN'
                    levelNE = int(r[11])+6
                    levelSW = int(r[12])+6
                    suit = r[13]
                else:
                    assert False
                assert ds[0]+suit not in tricks
                tricks[ds[0]+suit] = levelNE
                assert ds[1]+suit not in tricks
                tricks[ds[1]+suit] = levelSW
        assert len(tricks) == 4*5
        t = tuple(tuple(tricks[d+s] for s in 'CDHSN') for d in 'NESW') # controls order of tuple # use dict instead of tuple?
        ACBL_DDmakes_l.append(t)

    hrs_df = hrs_df.with_columns(pl.Series('ACBL_DDmakes', ACBL_DDmakes_l, dtype=pl.Object)) # , strict=False?
    print(f"Added ACBL_DDmakes column")
    hrs_df = hrs_df.drop('ACBL_double_dummy_ns') # remove ACBL_double_dummy_ns as it's often wrong. Use calculated DDmakes instead.
    hrs_df = hrs_df.drop('ACBL_double_dummy_ew') # remove ACBL_double_dummy_ew as it's often wrong. Use calculated DDmakes instead.

    # takes 45s/4.5s
    # Create Par score column in normalized format
    # rename par column. rewrite as list 
    # too complex for one regex, if at all possible, so must iterate and split().
    # todo: eliminate for-loop by using replace() with a list of regex? Or using map/apply?
    print("Processing ACBL Par scores...")
    assert hrs_df['ACBL_Par'].str.starts_with('Par: ').all()
    parsed = [_parse_acbl_par(v) for v in hrs_df['ACBL_Par']]
    keep = [item is not None for item in parsed]
    dropped = len(keep) - sum(keep)
    print(f"Dropping {dropped} rows with unparsed ACBL par")
    if dropped:
        hrs_df = hrs_df.filter(pl.Series('keep_par', keep))
    acbl_par_l = [item for item in parsed if item is not None]

    hrs_df = hrs_df.with_columns(pl.Series('ACBL_Par', acbl_par_l, dtype=pl.Object)) # , strict=False?
    print(f"Updated ACBL_Par column")

    # takes 2s
    # Show Pars which are passed out
    # todo: this should be done using 'Par' column instead of 'ACBL_Par' column. Move this to after 'Par' is created.
    pass_outs = hrs_df.with_columns(pl.col('ACBL_Par').map_elements(lambda x: x[0]==0,return_dtype=pl.Boolean))
    print(f"Pass outs count: {pass_outs.sum()}")

    # create HandRecordBoard column
    hrs_df = hrs_df.with_columns((pl.col("hand_record_id").cast(pl.Utf8) + "_" + pl.col("Board").cast(pl.Utf8).str.zfill(2)).alias("HandRecordBoard"))

    # Get HandRecordBoards with multiple board_record_strings
    # Not valid to have a HandRecordBoard with a different board_record_strings.
    hrbs_brs_nuniques = hrs_df.group_by("HandRecordBoard").agg(pl.col("board_record_string").n_unique().ne(1).alias("nunique"))
    invalid_hrbs = hrbs_brs_nuniques.filter(pl.col('nunique'))

    # Filter out rows with invalid HandRecordBoards
    print(f"Filtering out {invalid_hrbs.shape[0]} HandRecordBoards with inconsistent board_record_strings")
    hrs_df = hrs_df.join(
        invalid_hrbs.select("HandRecordBoard"), 
        on="HandRecordBoard", 
        how="anti"
    )

    # takes 5s
    # todo: what to do with duplicate board_record_string? keep, discard? Keep dups to sustain integrity with board results?

    # Unique count of 'board_record_string'
    unique_brs_count = hrs_df["board_record_string"].n_unique()
    print("unique brs count:", unique_brs_count)

    # Calculate duplicates
    duplicates_df = hrs_df.with_columns(pl.col("board_record_string").is_duplicated().alias("duplicates"))
    keep_false_count = duplicates_df["duplicates"].sum()
    print("duplicate brs count:", keep_false_count)

    # takes 5s
    assert hrs_df['board_record_string'].is_unique().sum() + hrs_df['board_record_string'].is_duplicated().sum() == len(hrs_df)

    # takes 10s
    # experiment with drop_duplicates()
    dups = hrs_df.filter(pl.col('board_record_string').is_duplicated())
    print(f"Total rows: {len(hrs_df)}, Duplicates: {len(dups)}, Unique: {len(hrs_df)-len(dups)}")

    # takes 30m/3m
    # todo: brs_to_pbn() is slow because of Deal(pbn).to_pbn() overhead. Need to re-implement brs_to_pbn() without Deal().
    print("Converting board_record_string to PBN (this may take a while)...")
    hrs_df = hrs_df.with_columns(pl.col('board_record_string').map_elements(mlBridgeLib.brs_to_pbn,return_dtype=pl.String).alias('PBN'))

    hrs_df = hrs_df.drop('board_record_string')
    return hrs_df


def club_hand_records_clean():

    sql_selects_d = {
        'hand_records': 'SELECT board, board_record_string, hand_record_set_id AS hand_record_id, dealer, vulnerability, double_dummy_ew, double_dummy_ns, par FROM hand_records',
        # Must use sessions.id (not event_id) as session_id: board_results/sections reference sessions.id,
        # and since ~2026-04-27 ACBL data has id != event_id, which broke the step2 inner join (dropped all rows after 2026-04-28).
        'sessions':'SELECT id AS session_id, hand_record_id, game_date FROM sessions',
    }

    schema_d = {
        'hand_records': {
            'board': pl.Int16,
            'board_record_string': pl.String,
            'hand_record_id': pl.String,
            'dealer': pl.String,
            'vulnerability': pl.String,
            'double_dummy_ew': pl.String,
            'double_dummy_ns': pl.String,
            'par': pl.String,
        },
        'sessions':
        {
            'session_id': pl.Int64,
            'hand_record_id': pl.String, # using dtype of String because of 'SHUFFLE'.
            'game_date': pl.String, # pl.Datetime, # using String because some dates are not valid DateTime () e.g. 2024-04-29 00:00:00
        },
    }

    # using pathlib to create sqlite path.
    db_connection_string = 'sqlite:///'+acblPath.joinpath(f'acbl_club_results.sqlite').as_posix()
    print(f"Database connection string: {db_connection_string}")

    uri = db_connection_string
    dfs_adbc = {}
    for table,schema in schema_d.items():
        print(f"Reading table:{table}")
        dfs_adbc[table] = read_sqlite_query(uri, sql_selects_d[table], schema)
    print(f"Tables read: {list(dfs_adbc.keys())}")

    hrs_df = dfs_adbc["hand_records"]
    hrs_df = hrs_df.join(dfs_adbc['sessions'], how='left', on='hand_record_id')
    hrs_df = hrs_df.with_columns(pl.col('game_date').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S'))
    drop_rows = hrs_df['game_date'].lt(pl.date(2019,1,1)) # remove 15k rows for dates earlier than 2019. They're the earliest data and suspect.
    print(f"Dropping {hrs_df.filter(drop_rows).shape[0]} rows with dates earlier than 2019")
    hrs_df = hrs_df.filter(~drop_rows)

    hrs_df = clean_hand_records(hrs_df)

    # takes 10s
    acbl_hand_records_cleaned_filename = f'acbl_club_hand_records_cleaned.parquet'
    acbl_hand_records_cleaned_file = acblPath.joinpath(acbl_hand_records_cleaned_filename)
    non_object_columns = [col for col, dtype in zip(hrs_df.columns, hrs_df.dtypes) if dtype != pl.Object] # drop object columns e.g. Hands, Par, DDmakes
    filtered_hrs_df = hrs_df.select(non_object_columns)
    filtered_hrs_df.write_parquet(acbl_hand_records_cleaned_file)
    print(f"Saved {acbl_hand_records_cleaned_filename}: shape:{filtered_hrs_df.shape}, size:{acbl_hand_records_cleaned_file.stat().st_size}")

    return hrs_df


def tournament_hand_records_clean():

    sql_selects_d = {
        'handrecord': 'SELECT id AS hand_record_id, session AS session_id, board_number AS board, north_spades, north_hearts, north_diamonds, north_clubs, east_spades, east_hearts, east_diamonds, east_clubs, south_spades, south_hearts, south_diamonds, south_clubs, west_spades, west_hearts, west_diamonds, west_clubs, double_dummy_north_south AS double_dummy_ns, double_dummy_east_west AS double_dummy_ew, double_dummy_par_score AS par, dealer, vulnerability FROM handrecord',
        'session': 'SELECT id AS session_id, session_number, start_date, start_time FROM session',
    }

    schema_d = {
        'handrecord': {
            "hand_record_id":pl.String,
            "session_id":pl.String,
            "board":pl.UInt32,
            "north_spades":pl.String,
            "north_hearts":pl.String,
            "north_diamonds":pl.String,
            "north_clubs":pl.String,
            "east_spades":pl.String,
            "east_hearts":pl.String,
            "east_diamonds":pl.String,
            "east_clubs":pl.String,
            "south_spades":pl.String,
            "south_hearts":pl.String,
            "south_diamonds":pl.String,
            "south_clubs":pl.String,
            "west_spades":pl.String,
            "west_hearts":pl.String,
            "west_diamonds":pl.String,
            "west_clubs":pl.String,
            "double_dummy_ns":pl.String,
            "double_dummy_ew":pl.String,
            "par":pl.String,
            "dealer":pl.String,
            "vulnerability":pl.String,
        },
        'session':
        {
            'session_id': pl.String,
            #'hand_record_id': pl.String, # raises Int64 to str error.
            'start_date': pl.String,
            'start_time': pl.String,
        },
    }
    # make tournament columns conform to club columns to simplify sharing of code.
    # todo: replace hrs_to_brss in mlBridgeLib with this one.
    def hrs_to_brss(hrs_df,void: str = "", ten: str = "10") -> pl.Expr:
        directions = ["north", "west", "east", "south"]  # NWES
        suits = ["spades", "hearts", "diamonds", "clubs"]
        suit_letters = ["S", "H", "D", "C"]

        parts: list[pl.Expr] = []
        for d in directions:
            for j, s in enumerate(suits):
                parts.append(pl.lit(suit_letters[j]))
                parts.append(pl.col(f"{d}_{s}").cast(pl.Utf8).fill_null(""))

        return (
            pl.concat_str(parts, separator="")
            .str.replace_all(" ", "")
            .str.replace_all("-", void)
            .str.replace_all("10", ten)
        )
        #return hrs_df.with_columns(board_record_string=hrs_to_brss(void="", ten="10"))
        
    # using pathlib to create sqlite path.
    db_connection_string = 'sqlite:///'+acblPath.joinpath(f'acbl_tournament_results.sqlite').as_posix()
    print(f"Database connection string: {db_connection_string}")

    uri = db_connection_string
    dfs_adbc = {}
    for table,schema in schema_d.items():
        print(f"Reading table:{table}")
        dfs_adbc[table] = read_sqlite_query(uri, sql_selects_d[table], schema)
    print(f"Tables read: {list(dfs_adbc.keys())}")
    # usage
    hrs_df = dfs_adbc["handrecord"]
    hrs_df = hrs_df.with_columns(board_record_string=hrs_to_brss(hrs_df, void="", ten="10")) # ugly but works
    hrs_df = hrs_df.join(dfs_adbc["session"], on=['session_id'], how="left")
    hrs_df = hrs_df.with_columns(pl.concat_str(pl.col('start_date'),pl.col('start_time')).str.strptime(pl.Datetime, format='%Y%m%d%H:%M', strict=False).alias('game_date'))
    hrs_df = hrs_df.with_columns(pl.concat_str(pl.lit('NS: '),pl.col('double_dummy_ns')).alias('double_dummy_ns'))
    hrs_df = hrs_df.with_columns(pl.concat_str(pl.lit('EW: '),pl.col('double_dummy_ew')).alias('double_dummy_ew'))
    hrs_df = hrs_df.with_columns(pl.concat_str(pl.lit('Par: '),pl.col('par')).alias('par'))

    hrs_df = clean_hand_records(hrs_df)

    # takes 10s
    acbl_hand_records_cleaned_filename = f'acbl_tournament_hand_records_cleaned.parquet'
    acbl_hand_records_cleaned_file = acblPath.joinpath(acbl_hand_records_cleaned_filename)
    non_object_columns = [col for col, dtype in zip(hrs_df.columns, hrs_df.dtypes) if dtype != pl.Object] # drop object columns e.g. Hands, Par, DDmakes
    filtered_hrs_df = hrs_df.select(non_object_columns)
    filtered_hrs_df.write_parquet(acbl_hand_records_cleaned_file)
    print(f"Saved {acbl_hand_records_cleaned_filename}: shape:{filtered_hrs_df.shape}, size:{acbl_hand_records_cleaned_file.stat().st_size}")

    return hrs_df


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Clean ACBL hand records from SQLite.")
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

    clean_fns = {
        'club': club_hand_records_clean,
        'tournament': tournament_hand_records_clean,
    }
    for club_or_tournament in _parse_club_tournament_args():
        t = time.time()
        clean_fns[club_or_tournament]()
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)

