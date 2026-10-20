#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_club_results_board_results_augment_step1.py

Takes 2m30s-3m30s/30s- for 135m->106m/26m->23m rows 44->50/45->51 columns 3GB/500MB file. 500GB memory/cache.

Perform initial augmentations on board results:
    - Clean and normalize contracts
    - Add vulnerability from board numbers
    - Validate data integrity

Requirements:
    pip install polars numpy

Previous steps:
    acbl_club_sql_to_board_results_clean.py

Next steps:
    acbl_club_results_board_results_dicts.ipynb - obsolete?
    acbl_club_board_results_augment_step2.py
    acbl_elo_ratings_create.py
    acbl_club_model_data.py

TODO:
    - Similar to acbl_to_mldf() in chatlib?
    - Write dropped rows to pkl or sql file
    - Filter out non-pair events?
    - Sanity check when multiple hands don't match par
"""

import polars as pl
import pathlib
import sys
import time

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


def normalize_contracts(brs_df: pl.DataFrame) -> pl.DataFrame:
    """Normalize raw contracts ('4 S x', '4SX', '3 NT') to BidLvl/BidSuit/Dbl and e.g. '4SXN'."""
    brs_df = brs_df.with_columns(
        pl.col('Contract').str.replace_all(' ', '').str.to_uppercase().str.replace('NT', 'N'),
    )
    brs_df = brs_df.with_columns([
        pl.col('Contract').str.extract(r'^(\d)').alias('BidLvl'),
        pl.col('Contract').str.extract(r'^\d([CDHSN])').alias('BidSuit'),
        pl.col('Contract').str.extract(r'^\d[CDHSN](X*)').alias('Dbl')
    ])
    brs_df = brs_df.with_columns(
        pl.when(pl.col('Contract') == 'PASS')
        .then(pl.col('Contract'))
        .otherwise(
            (pl.col('BidLvl') +
             pl.col('BidSuit') +
             pl.col('Dbl') +
             pl.col('Declarer_Direction'))
        )
        .alias('Contract')
    )
    return brs_df.with_columns(pl.col("BidLvl").fill_null(0).cast(pl.UInt8))


def typed_club_results(brs_df: pl.DataFrame) -> pl.DataFrame:
    """Replace ACBL's raw 'result' ('=', '+1', '-2') and 'tricks_taken' strings with Result/Tricks.

    ACBL's values are authoritative, so augmentation must not re-derive them from Score_NS.
    The raw 'result' must not survive: DuckDB is case-insensitive and would resolve 'Result' to it.
    """
    return brs_df.with_columns(
        pl.when(pl.col('result') == '=').then(pl.lit('0')).otherwise(pl.col('result'))
        .cast(pl.Int8, strict=False).alias('Result'),
        pl.col('tricks_taken').cast(pl.UInt8, strict=False).alias('Tricks'),
    ).drop('result', 'tricks_taken')


def augment_board_results_step1(club_or_tournament):
    """Perform step 1 augmentation on board results."""
    print(f"Processing {club_or_tournament} board results augmentation step 1...")

    # Load cleaned board results
    acbl_board_results_filename = f'acbl_{club_or_tournament}_board_results_cleaned.parquet'
    acbl_board_results_file = acblPath.joinpath(acbl_board_results_filename)
    brs_df = pl.read_parquet(acbl_board_results_file)
    print(f"Loaded {acbl_board_results_filename}: shape:{brs_df.shape} size:{acbl_board_results_file.stat().st_size}")
    
    # Drop rows with nulls in required columns
    print("Dropping rows with nulls in required columns...")
    columns_which_must_not_have_nulls = ['hand_record_id', 'Declarer_Direction', 'Contract', 'MP_NS', 'MP_EW']
    brs_df = brs_df.filter(pl.all_horizontal([~pl.col(col).is_null() for col in columns_which_must_not_have_nulls]))
    
    # Clean opening lead (for club data)
    if club_or_tournament == 'club' and 'opening_lead' in brs_df.columns:
        print("Cleaning opening lead...")
        def CleanOpeningLead(x):
            if x is None:
                return None
            x = x.replace('10', 'T')
            if len(x) == 2:
                if x[0] not in mlBridgeLib.CDHS:
                    x = x[1] + x[0]
                if x[0] in mlBridgeLib.CDHS and x[1] in mlBridgeLib.ranked_suit:
                    return x
            return None
        
        brs_df = brs_df.with_columns(
            pl.col('opening_lead').map_elements(CleanOpeningLead, return_dtype=pl.String)
        )
        brs_df = brs_df.filter(pl.col('hand_record_id').str.contains(r'^\d+$')) # obsolete?
        assert (brs_df.group_by("section_id").agg(pl.col("hand_record_id").n_unique().alias('nuniques'))['nuniques'] == 1).all()

    # Create Vul column from board numbers
    # print("Creating vulnerability columns...")
    # brs_df = brs_df.with_columns(
    #     pl.Series('iVul', [mlBridgeLib.BoardNumberToVul(bn) for bn in brs_df['Board']], dtype=pl.UInt8)
    # )
    # brs_df = brs_df.with_columns(
    #     pl.Series('Vul', [mlBridgeLib.vul_syms[iVul] for iVul in brs_df['iVul']], dtype=pl.String)
    # )

    assert (brs_df['Declarer_Direction'].is_null() | brs_df['Declarer_Direction'].is_in(list('NSEW'))).all(), brs_df['Declarer_Direction'].describe()
    assert brs_df['MP_NS'].is_not_null().all() and brs_df['MP_EW'].is_not_null().all()
    assert brs_df['Contract'].is_not_null().all()
    
    print("Normalizing contracts...")
    brs_df = normalize_contracts(brs_df)
    
    # Drop rows with invalid contracts
    drop_rows = pl.col('Contract').ne('PASS') & brs_df['BidSuit'].is_null()
    print(f"Dropping {brs_df.filter(drop_rows).shape[0]} rows with invalid contracts")
    brs_df = brs_df.filter(~drop_rows)
    
    # Validate data
    print("Validating data integrity...")
    assert (brs_df['Contract'].eq('PASS') | brs_df['BidLvl'].is_between(0, 7)).all()
    assert (brs_df['Contract'].eq('PASS') | brs_df['BidSuit'].is_in(list('CDHSN'))).all()
    assert (brs_df['Contract'].eq('PASS') | brs_df['Dbl'].is_in(['', 'X', 'XX'])).all()
    assert brs_df['Declarer_Direction'].is_in(list('NSEW')).all()
    assert brs_df['Board'].ge(1).all()
    
    # drop rows where either ns or ew have negative match points. must be director's adjustment?
    drop_rows = brs_df['MP_NS'].lt(0) | brs_df['MP_EW'].lt(0)
    brs_df = brs_df.filter(~drop_rows)

    # drop rows where ns_match_points + ew_match_points is zero. must be director's adjustment? this causes a divide by zero error or np.inf.
    drop_rows = (brs_df['MP_NS'] + brs_df['MP_EW']) == 0
    brs_df = brs_df.filter(~drop_rows)

    # takes 45s/3s
    # order of changes is important and probably can't be combined.

    rename_cols = {
        #'board_number':'Board',
        'club_id_number':'Club',
        #'declarer': 'Declarer_Direction',
        #'game_date':'Date',
        #'ns_match_points':'match_points_ns',
        #'ew_match_points':'match_points_ew',
        'opening_lead':'Lead',
        #'ns_score':'Score_NS',
        #'ew_score':'Score_EW',
        #'round_number':'Round',
        #'section_id':'Session',
        'table_number':'Table',
    }
    # for d in 'NESW':
    #     rename_cols.update({
    #         'Player_ID_'+d.lower():'Player_ID_'+d.upper(),
    #         'player_name_'+d.lower():'Player_Name_'+d.upper()
    #     })

    for k,v in rename_cols.items():
        if k in brs_df.columns:
            print('renaming:',k,v)
            brs_df = brs_df.rename({k:v})
        else:
            print('column does not exist:',k,v)

    # todo: move these to mlBridgeAugmentLib.py or obsolete?
    brs_df = brs_df.with_columns(pl.col('Board').cast(pl.String).str.zfill(2).alias('sBoard')) # todo: obsolete?
    brs_df = brs_df.with_columns((pl.col('section_name')+(pl.col('Pair_Number_NS').cast(pl.String)).str.zfill(2)).alias('NSPair')) # todo: obsolete?
    brs_df = brs_df.with_columns((pl.col('section_name')+(pl.col('Pair_Number_EW').cast(pl.String)).str.zfill(2)).alias('EWPair')) # todo: obsolete?

    # todo: looks like all of this can be moved/handled in mlBridgeAugmentLib.py
    if club_or_tournament == 'club':
        brs_df = typed_club_results(brs_df)
        #mae = pl.Series('Event',['?' if not isinstance(p, str) else p.split(' ')[1][0] for p in brs_df['club_session']])
        brs_df = brs_df.with_columns(
            pl.col('Date').str.strptime(pl.Date, '%Y-%m-%d %H:%M:%S'), # should Date be pl.Date, pl.Datetime or pl.String?
            pl.col('Club').cast(pl.Int32),
            #pl.col('event_id').cast(pl.String)+mae,
            #(pl.col('Club').cast(pl.String)+'_'+pl.col('event_id').cast(pl.String)+'_'+pl.col('Board')).alias('ClubEventBoard'), # or should this be just ClubEvent?
            #(pl.col('Club').cast(pl.String)+'_'+pl.col('Date').cast(pl.String)+'_'+pl.col('Board')).alias('ClubDateBoard'), # or should this be just ClubDate?
            ##pl.col('hand_record_id').cast(pl.String)+'_'+pl.col('Board'),
            # (pl.col('NSPair')+'_'+pl.col('EWPair')).alias('Pair'),
            # pl.when(pl.col("Player_ID_N") < pl.col("Player_ID_S"))
            #     .then(pl.col("Player_ID_N") + "_" + pl.col("Player_ID_S"))
            #     .otherwise(pl.col("Player_ID_S") + "_" + pl.col("Player_ID_N"))
            #     .alias("PairNS"),
            # pl.when(pl.col("Player_ID_E") < pl.col("Player_ID_W"))
            #     .then(pl.col("Player_ID_E") + "_" + pl.col("Player_ID_W"))
            #     .otherwise(pl.col("Player_ID_W") + "_" + pl.col("Player_ID_E"))
            #     .alias("PairEW"),
            #(pl.col('match_points_ns')/(pl.col('match_points_ns')+pl.col('match_points_ew'))).alias('Pct_NS'),
            #(pl.col('match_points_ew')/(pl.col('match_points_ns')+pl.col('match_points_ew'))).alias('Pct_EW'),
            # defer casting to categorical until after augmentations and data modeling.
            #pl.col('Declarer_Direction').cast(pl.Categorical),
            #pl.col('BidSuit').cast(pl.Categorical),
            #pl.col('Dbl').cast(pl.Categorical),
            pl.col('Round').cast(pl.UInt8),
            pl.col('Table').cast(pl.UInt8),
        )
    elif club_or_tournament == 'tournament':
        brs_df = brs_df.with_columns(
            pl.col('Date').str.strptime(pl.Date, '%Y-%m-%d'), # %H:%M:%S')) # should Date be pl.Date or pl.String?
        )
    else:
        raise ValueError(f"Invalid club_or_tournament: {club_or_tournament}")

    # takes 10s
    # todo: revisit why 'Board' is string. don't like to convert back and forth.
    # todo: doesn't acbl already have a Dealer column?
    # defer casting to categorical until after augmentations and data modeling.
    # obsolete? brs_df = brs_df.with_columns(pl.Series('Dealer',[mlBridgeLib.BoardNumberToDealer(int(b)) for b in brs_df['Board']])) # ,dtype=pl.Categorical))

    # takes 30s/15s for 90m rows and 52 columns. 4GB file.
    acbl_board_results_cleaned_filename = f'acbl_{club_or_tournament}_board_results_augmented_step1.parquet'
    acbl_board_results_cleaned_file = acblPath.joinpath(acbl_board_results_cleaned_filename)
    non_object_columns = [col for col, dtype in zip(brs_df.columns, brs_df.dtypes) if dtype != pl.Object] # todo: better to show pl.Object columns
    filtered_brs_df = brs_df.select(non_object_columns)
    filtered_brs_df.write_parquet(acbl_board_results_cleaned_file)
    print(f"Saved {acbl_board_results_cleaned_filename}: shape:{filtered_brs_df.shape}, size:{acbl_board_results_cleaned_file.stat().st_size}")    

    return brs_df


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Augment ACBL board results (step 1).")
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
        augment_board_results_step1(club_or_tournament)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)


