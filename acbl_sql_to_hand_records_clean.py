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

from acbl_sqlite_read import read_parquet_query

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


# One DD token, e.g. S4, 4S, S4/5 or 4/5S.
_DD_TOKEN = r'(?:[CDHSN]\d+|\d+[CDHSN]|[CDHSN]\d+/\d+|\d+/\d+[CDHSN])'
_RANKS = 'AKQJT98765432'
_BRS_CHARS = '23456789ACDHJKQST' # each appears 4 times in a valid brs (10 written as T)


def _timed(label, t0):
    t = time.time()
    print(f"[time] {label}: {t-t0:.1f}s")
    return t


def _unique_in_threads(fn, s: pl.Series, n: int = 16):
    """Run fn on n slices of s.unique() in threads and concat. Polars releases the GIL, so this uses n cores."""
    from concurrent.futures import ThreadPoolExecutor
    u = s.unique()
    k = (len(u) + n - 1) // n
    with ThreadPoolExecutor(n) as ex:
        return pl.concat(list(ex.map(fn, [u.slice(i*k, k) for i in range(n)])))


def _validate_brs_unique(brs: pl.Series) -> pl.Series:
    """mlBridgeLib.validate_brs for each unique value. Returns the valid values.

    Vectorized check first; values it rejects go through validate_brs too,
    which logs them and has the final say.
    """
    u = brs.unique().to_frame('brs')
    s = pl.col('brs').str.replace_all('10', 'T', literal=True)
    ok = ~pl.col('brs').str.contains('-', literal=True) & ~pl.col('brs').str.contains('T', literal=True) & s.str.len_chars().eq(68)
    for c in _BRS_CHARS:
        ok = ok & s.str.count_matches(c, literal=True).eq(4)
    ok = ok & s.str.slice(0, 17).str.count_matches('[SHDC]').eq(4)
    u = u.with_columns(ok.alias('ok'))
    bad = u.filter(~pl.col('ok'))['brs']
    rescued = [v for v in bad if mlBridgeLib.validate_brs(v)]
    return pl.concat([u.filter(pl.col('ok'))['brs'], pl.Series('brs', rescued, dtype=pl.String)])


def _brs_to_pbn_unique(brs: pl.Series) -> pl.DataFrame:
    """mlBridgeLib.brs_to_pbn for each unique value, as a brs -> PBN table.

    For a complete deal brs_to_pbn only reorders hands to NESW and sorts each
    suit AKQJT..2, which is done here with string ops. Anything else (missing,
    repeated or unknown cards) goes through brs_to_pbn itself.
    """
    u = brs.unique().to_frame('board_record_string')
    g = u.select(pl.col('board_record_string').str.extract_groups('^' + r'S(.*)H(.*)D(.*)C(.*)' * 4).alias('g')).unnest('g')
    # h0..h15 in brs order: hand (N,W,E,S) * 4 + suit (S,H,D,C)
    g = g.select([pl.col(c).str.replace_all('10', 'T', literal=True).alias(f'h{i}') for i, c in enumerate(g.columns)])

    def sort_suit(h): # distinct ranks of h, in AKQJT..2 order
        return pl.concat_str([pl.when(h.str.contains(r, literal=True)).then(pl.lit(r)).otherwise(pl.lit('')) for r in _RANKS])

    # separate columns so Polars runs them in parallel
    g = g.with_columns([sort_suit(pl.col(f'h{i}')).alias(f's{i}') for i in range(16)]
                       + [sort_suit(pl.concat_str([pl.col(f'h{hand*4+j}') for hand in range(4)])).alias(f'u{j}') for j in range(4)])
    n = lambda c: pl.col(c).str.len_chars()
    ok = pl.all_horizontal(
        [n(f's{i}').eq(n(f'h{i}')) for i in range(16)] # no repeated or unknown cards in a suit
        + [pl.sum_horizontal([n(f'h{hand*4+j}') for j in range(4)]).eq(13) for hand in range(4)] # 13 cards per hand
        + [n(f'u{j}').eq(13) & pl.sum_horizontal([n(f'h{hand*4+j}') for hand in range(4)]).eq(13) for j in range(4)]) # each card once
    pbn = pl.concat_str([pl.lit('N:'), pl.concat_str([pl.concat_str([pl.col(f's{hand*4+j}') for j in range(4)], separator='.') for hand in [0, 2, 3, 1]], separator=' ')])
    out = pl.concat([u, g], how='horizontal').select('board_record_string', pl.when(ok.fill_null(False)).then(pbn).alias('PBN'))
    odd = out.filter(pl.col('PBN').is_null())
    if odd.height:
        print(f"brs_to_pbn fallback for {odd.height} odd board_record_strings")
        odd = odd.with_columns(pl.col('board_record_string').map_elements(mlBridgeLib.brs_to_pbn, return_dtype=pl.String).alias('PBN'))
        out = pl.concat([out.filter(pl.col('PBN').is_not_null()), odd])
    return out


def clean_hand_records(hrs_df):
    # common to club and tournament
    print(f"Initial hrs_df shape: {hrs_df.shape}")
    t0 = time.time()

    hrs_renames = {
        'board': 'Board',
        'dealer': 'Dealer',
        'vulnerability': 'Vul',
        'double_dummy_ns': 'ACBL_double_dummy_ns',
        'double_dummy_ew': 'ACBL_double_dummy_ew',
        'par': 'ACBL_Par',
    }
    hrs_df = hrs_df.rename(hrs_renames)

    # drop rows having board numbers > 255. doesn't pass the sniff test.
    # Same rows as the old anti-join on all columns: rows with a null anywhere never matched, so they stay.
    drop_rows = pl.col('Board').gt(255).fill_null(False) & pl.all_horizontal(pl.all().is_not_null())
    n_drop = hrs_df.select(drop_rows.sum()).item()
    print(f"Dropping {n_drop} rows with Board > 255")
    if n_drop > 0:
        print(hrs_df.filter(pl.col('Board') > 255)['Board'].value_counts())
        hrs_df = hrs_df.filter(~drop_rows)
    t0 = _timed('board > 255', t0)

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
    t0 = _timed('dealer/vul normalize', t0)

    # drop rows having board numbers with non-normal vuls. e.g. Board 1 having N-S vul. around 3500 instances.
    # Same as mlBridgeLib.BoardNumberToVul: bn-=1; (bn//4 + (bn&3)) & 3
    assert hrs_df['Board'].null_count() == 0
    bn = pl.col('Board').cast(pl.Int64) - 1
    vul_idx = ((bn // 4) + (bn & 3)) & 3
    drop_rows = pl.col('Vul') != vul_idx.replace_strict(list(range(4)), mlBridgeLib.vul_syms, return_dtype=pl.Utf8)
    print(f"Dropping {hrs_df.select(drop_rows.sum()).item()} rows with non-normal vulnerability")
    hrs_df = hrs_df.filter(~drop_rows)
    print(f"After vul filter: {hrs_df.shape}")
    t0 = _timed('vul filter', t0)

    hrs_df = hrs_df.with_columns(pl.col('board_record_string').str.replace_all('-', ''))

    # drop results whose board_record_string contains invalid characters (5492639, 13290144) or wrong number of cards (4754685, 6213531)
    print("Validating board_record_strings...")
    valid_brs = _unique_in_threads(_validate_brs_unique, hrs_df['board_record_string'])
    hrs_df = hrs_df.filter(pl.col('board_record_string').is_in(valid_brs))
    print(f"After validation: {hrs_df.shape}")
    t0 = _timed('validate brs', t0)

    # drop rows which have empty double dummy or par columns. They're probably all SHUFFLE.
    b1 = hrs_df['ACBL_double_dummy_ns'] != ''
    b2 = hrs_df['ACBL_double_dummy_ew'] != ''
    b3 = hrs_df['ACBL_Par'] != ''
    assert (b1 == b2).fill_null(False).all() and (b1 == b3).fill_null(False).all()
    print(f"Dropping {(~b1).sum()} rows with empty DD/Par columns")
    hrs_df = hrs_df.filter(b1) # drop rows which have empty double dummy or par columns

    # clean double dummy columns. rewrite into standarized form.
    # todo: cleaning and validations need to be done in previous notebook
    dd_cols = ['ACBL_double_dummy_ns','ACBL_double_dummy_ew']
    for d in dd_cols:
        assert hrs_df[d].str.starts_with(d[-2:].upper()+':').all()
        hrs_df =  hrs_df.with_columns(pl.col(d).str.replace_all('NT','N').str.replace_all(r'  ',' ')) # unsure why str.replace(r'\s+',' ') doesn't remove double spaces.
        assert not hrs_df[d].str.contains('NT').any()
        b = hrs_df[d].str.count_matches(' ').ne(5) | ~hrs_df[d].str.starts_with(d[-2:].upper()+': ')
        print(f"Dropping {b.sum()} rows with wrong DD format for {d}")
        hrs_df = hrs_df.filter(~b)
    t0 = _timed('DD format', t0)

    # Check DD columns: 5 tokens like S4, 4S, S4/5, 4/5S, one per strain CDHSN.
    # The old loop built ACBL_DDmakes from these, but it was an Object column that was never written.
    for d in dd_cols:
        dd = hrs_df[d].unique()
        tokens_ok = dd.str.contains('^' + d[-2:].upper() + ': ' + ' '.join([_DD_TOKEN]*5) + r'\n?$')
        strains = dd.str.slice(4).str.replace_all(r'[^CDHSN]', '')
        strains_ok = strains.str.len_chars().eq(5)
        for s in 'CDHSN':
            strains_ok = strains_ok & strains.str.contains(s, literal=True)
        bad = dd.filter(~(tokens_ok & strains_ok))
        assert bad.len() == 0, f"bad {d} values: {bad.head(10).to_list()}"
    hrs_df = hrs_df.drop('ACBL_double_dummy_ns') # remove ACBL_double_dummy_ns as it's often wrong.
    hrs_df = hrs_df.drop('ACBL_double_dummy_ew') # remove ACBL_double_dummy_ew as it's often wrong.
    t0 = _timed('DD check', t0)

    # Par: parse each unique value and drop rows that don't parse.
    # The parsed ACBL_Par was an Object column that was never written, so drop the column after filtering.
    print("Processing ACBL Par scores...")
    assert hrs_df['ACBL_Par'].str.starts_with('Par: ').all()
    par_counts = hrs_df['ACBL_Par'].value_counts()
    parsed = [_parse_acbl_par(v) for v in par_counts['ACBL_Par']]
    good_pars = [v for v, p in zip(par_counts['ACBL_Par'], parsed) if p is not None]
    dropped = sum(n for n, p in zip(par_counts['count'], parsed) if p is None)
    print(f"Dropping {dropped} rows with unparsed ACBL par")
    if dropped:
        hrs_df = hrs_df.filter(pl.col('ACBL_Par').is_in(good_pars))
    print(f"Pass outs count: {sum(n for n, p in zip(par_counts['count'], parsed) if p is not None and p[0] == 0)}")
    hrs_df = hrs_df.drop('ACBL_Par')
    t0 = _timed('par', t0)

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
    t0 = _timed('HandRecordBoard', t0)

    # todo: what to do with duplicate board_record_string? keep, discard? Keep dups to sustain integrity with board results?
    unique_brs_count = hrs_df["board_record_string"].n_unique()
    print("unique brs count:", unique_brs_count)
    n_dups = hrs_df["board_record_string"].is_duplicated().sum()
    print("duplicate brs count:", n_dups)
    print(f"Total rows: {len(hrs_df)}, Duplicates: {n_dups}, Unique: {len(hrs_df)-n_dups}")

    # PBN from each unique board_record_string, joined back in row order.
    print("Converting board_record_string to PBN...")
    pbn_df = _unique_in_threads(_brs_to_pbn_unique, hrs_df['board_record_string'])
    hrs_df = hrs_df.join(pbn_df, on='board_record_string', how='left', maintain_order='left')
    assert hrs_df['PBN'].null_count() == 0

    hrs_df = hrs_df.drop('board_record_string')
    t0 = _timed('PBN', t0)
    return hrs_df


def get_club_hand_records_schemas():
    """Club SQL selects and column types used by club_hand_records_clean."""
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
    return sql_selects_d, schema_d


def club_hand_records_clean():

    sql_selects_d, schema_d = get_club_hand_records_schemas()

    # Read 1b's parquet tables, not acbl_club_results.sqlite (1b now runs
    # with --skip-sqlite). Same SQL and casts; acbl_compare_sqlite_vs_parquet.py
    # showed identical results for every query on 2026-10-24.
    parquet_dir = acblPath.joinpath('club_results_parquet').as_posix()
    print(f"Reading club tables from: {parquet_dir}")

    dfs_adbc = {}
    for table,schema in schema_d.items():
        print(f"Reading table:{table}")
        dfs_adbc[table] = read_parquet_query(parquet_dir, sql_selects_d[table], schema)
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


def get_tournament_hand_records_schemas():
    """Tournament SQL selects and column types used by tournament_hand_records_clean."""
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
    return sql_selects_d, schema_d


def tournament_hand_records_clean():

    sql_selects_d, schema_d = get_tournament_hand_records_schemas()
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
        
    # Read 1e's parquet tables, not acbl_tournament_results.sqlite (1e builds
    # SQLite only with --build-sqlite since 2026-10-24). Same SQL and casts;
    # check with acbl_compare_sqlite_vs_parquet.py --side tournament.
    parquet_dir = acblPath.joinpath('tournament_results_parquet').as_posix()
    print(f"Reading tournament tables from: {parquet_dir}")

    dfs_adbc = {}
    for table,schema in schema_d.items():
        print(f"Reading table:{table}")
        dfs_adbc[table] = read_parquet_query(parquet_dir, sql_selects_d[table], schema)
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

