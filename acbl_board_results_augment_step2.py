#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_club_results_board_results_augment_step2.py

Takes 6h/70m for 106m->?/22m->15m rows and ?/6752 columns. filesize ?GB/16GB. 2TB memory/cache.

Final augmentation step for board results.
Joins with hand records and adds complete augmentation data.

Previous steps:
    acbl_sql_to_hand_records_clean.py
    acbl_hand_records_augment.py
    acbl_sql_to_board_results_clean.py
    acbl_board_results_augment_step1.py

Next steps:
    acbl_club_model_data.py

TODO:
1. why is 'section_name_right' appearing?
2. investigate source of out-of-bounds percentage values (see _validate_percentage_columns)
"""

import polars as pl
import pyarrow.parquet as pq
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
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


PCT_VALIDATION_COLS = [
    'Pct_NS',
    'Pct_EW',
    'Declarer_Pct',
    'DD_Score_Pct_NS',
    'DD_Score_Pct_EW',
    'Par_Pct_NS',
    'Par_Pct_EW',
    'MP_DD_Pct_Declarer',
    'MP_Par_Pct_Declarer',
    'MP_EV_Pct_Declarer',
    'MP_EV_Max_Pct_Declarer',
    'DD_Pct_Max_NS',
    'DD_Pct_Max_EW',
    'EV_Pct_Max_NS',
    'EV_Pct_Max_EW',
]


def _validate_percentage_columns(df: pl.DataFrame, label: str):
    """Warn about percentage columns with out-of-bounds values [0,1]."""
    for col in PCT_VALIDATION_COLS:
        if col not in df.columns:
            print(f"  WARNING ({label}): validation column '{col}' not found -- skipping")
            continue
        oob = df.filter(~pl.col(col).is_between(0, 1) & pl.col(col).is_not_null())
        if len(oob) > 0:
            print(f"  WARNING ({label}): {len(oob)} rows have '{col}' outside [0,1]")


def augment_board_results_step2(club_or_tournament):
    """Perform final augmentation on board results by joining with hand records."""
    print(f"Processing {club_or_tournament} board results augmentation step 2...")
    
    # Load step 1 augmented board results
    acbl_board_results_step1_filename = f'acbl_{club_or_tournament}_board_results_augmented_step1.parquet'
    acbl_board_results_step1_file = acblPath.joinpath(acbl_board_results_step1_filename)
    brs_df = pl.read_parquet(acbl_board_results_step1_file)
    print(f"Loaded {acbl_board_results_step1_filename}: shape:{brs_df.shape}, size:{acbl_board_results_step1_file.stat().st_size}")
    
    # takes 1m-5m/10s-1m10s Time seems to vary with vaguries of cache?
    #print("Sorting data temporally...")
    #sort_keys = [c for c in ["Date", "session_id", "Round", "Board"] if c in brs_df.columns]
    #brs_df = brs_df.sort(sort_keys)  # sort temporially for elo and just good in general.

    # obsoleted by joining with elo ratings file created in acbl_elo_ratings_create.py?
    # takes 9m30s/5m30s
    # brs_df = mlBridgeAugmentLib.compute_matchpoint_elo_ratings(brs_df)

    # Load augmented hand records
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)

    # Not implemented:To make joining with brs_df tractable, only load columns which are needed by board results augmentations.
    # filter_cols = ['HandRecordBoard', 'Dealer', 'Vul', 'PBN']
    # filter_cols += ['^Vul_(NS|EW)$', '^SL(_Max)?_(NS|EW)(_[SHDC])?$', '^DD_([NESW]|NS|EW)_[SHDCN]$', 'DD_Score_[1-7][CDHSN]_([NESW]|NS|EW)$', '^Par_(NS|EW)$']
    # filter_cols += [r'^Probs_(NS|EW)_[NESW]_[SHDCN]_\d+$', '^EV_(NS|EW)(_[NESW])?(_[SHDCN])?(_[1-7])?_(NV|V)(_Max)?(_Col)?$']
    # cols, all_cols = get_parquet_columns_filtered(acbl_hand_records_augmented_file, filter_cols)

    # print(f"Reading {len(cols)} of {len(all_cols)} columns.")
    # hrs_df = pl.read_parquet(acbl_hand_records_augmented_file, columns=cols)
    # print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    hrs_df = pl.read_parquet(acbl_hand_records_augmented_file)
    print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")
    print(hrs_df)

    print(set(brs_df.columns).intersection(set(hrs_df.columns)))

    # Decide whether to batch by year or process all at once based on total row count
    acbl_board_results_augmented_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_augmented_file = acblPath.joinpath(acbl_board_results_augmented_filename)

    if club_or_tournament == 'club':
        #hrs_df = hrs_df.with_columns(pl.col('hand_record_id').cast(pl.Int64)) # todo: temp until previous step is corrected. issue is 'SHUFFLE' in hand_record_id for session.
        years = (
            brs_df
            .select(pl.col('Date').dt.year().alias('year'))
            .drop_nulls()
            .unique()
            .sort('year')
            .to_series()
            .to_list()
        )
        print(f"processing years: {years}")

        writer = None
        prev_columns = None
        total_rows_written = 0
        for y in years:
            t = time.time()
            part_brs_df = brs_df.filter(pl.col('Date').dt.year() == y) # or group_by()?
            print(f"\nProcessing year {y}: part_brs_df shape:{part_brs_df.shape}")
            part_brs_df = part_brs_df.join(hrs_df, left_on=['hand_record_id','session_id','Board'], right_on=['hand_record_id','session_id','Board'], how='inner')
            print(part_brs_df)
            #assert part_brs_df.select('^.*_right$').is_empty(), part_brs_df.select('^.*_right$')
            if part_brs_df.is_empty():
                print(f"No data for year {y}")
                continue
            part_brs_df = mlBridgeAugmentLib.AllBoardResultsAugmentations(part_brs_df).perform_all_board_results_augmentations()
            non_object_columns = [col for col, dtype in zip(part_brs_df.columns, part_brs_df.dtypes) if dtype != pl.Object]
            part_brs_df = part_brs_df.select(non_object_columns)
            _validate_percentage_columns(part_brs_df, f"club year {y}")
            print(part_brs_df)
            arrow_table = part_brs_df.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(acbl_board_results_augmented_file, arrow_table.schema, compression="zstd")
                prev_columns = part_brs_df.columns
            assert part_brs_df.columns == prev_columns, f"columns changed from {prev_columns} to {part_brs_df.columns}"
            writer.write_table(arrow_table, row_group_size=512_000)
            total_rows_written += len(part_brs_df)
            print(f"wrote {y} to {acbl_board_results_augmented_filename}")
            print(f"year {y} elapsed time in seconds: {time.time()-t}")
        if writer is not None:
            writer.close()
            print(f"Saved {acbl_board_results_augmented_filename}: {total_rows_written} total rows, size:{acbl_board_results_augmented_file.stat().st_size}")
        try:
            del part_brs_df, arrow_table
        except NameError:
            pass
    elif club_or_tournament == 'tournament':
        brs_df = brs_df.join(hrs_df, on=['hand_record_id','session_id','Board'], how='inner')
        assert brs_df.select('^.*_right$').is_empty(), brs_df.select('^.*_right$')
        if brs_df.is_empty():
            print("No data to process.")
        else:
            brs_df = mlBridgeAugmentLib.AllBoardResultsAugmentations(brs_df).perform_all_board_results_augmentations()
            non_object_columns = [col for col, dtype in zip(brs_df.columns, brs_df.dtypes) if dtype != pl.Object]
            brs_df = brs_df.select(non_object_columns)
            _validate_percentage_columns(brs_df, "tournament")
            brs_df.write_parquet(acbl_board_results_augmented_file)
            print(f"Saved {acbl_board_results_augmented_filename}: shape:{brs_df.shape}, size:{acbl_board_results_augmented_file.stat().st_size}")
    else:
        raise ValueError(f"Invalid club_or_tournament: {club_or_tournament}")

    if club_or_tournament == "club":
        from acbl_recent_update import augmented_max_date, prune_recent

        removed = prune_recent(
            pathlib.Path(r"e:/bridge/data/acbl/recent"),
            augmented_max_date(acbl_board_results_augmented_file),
        )
        print(f"Dropped {removed} recent rows now covered by the augmented parquet")

    return


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Augment ACBL board results (step 2).")
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
        augment_board_results_step2(club_or_tournament)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)


