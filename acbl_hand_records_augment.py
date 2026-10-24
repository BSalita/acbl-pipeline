#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_results_hand_records_augment.py

Takes 1h/5m minimum (no SD calculations) for 17m->7m/17m->17m rows by /5923 columns. Uses 300GB (1.2TB?) memory/cache for processing.
Recommend 2TB PCIe 5.0 SSD for pagefile.

Augment hand records df with DD/SD analysis and probabilities.

Skips a side when its inputs are unchanged since the last run. The fingerprint
covers the cleaned input rows, this side's cache rows, the code, the polars
version and the parameters. It is kept in
acbl_{side}_hand_records_augmented_manifest.json next to the outputs.
Use --force to always recompute.

Previous steps:
    acbl_sql_to_hand_records_clean.py

Next steps:
    acbl_board_results_augment_step1.py
    acbl_board_results_augment_step2.py
    acbl_model_data.py

TODO:
    - Rename acbl_club_hand_records_cache_df.parquet to acbl_hand_records_cache_df.parquet
    - Obsolete most logic with calls to mlBridgeAugmentLib
    - Re-implement acbl_cgd
    - Implement matchpoint dict
"""

import polars as pl
import pathlib
from collections import defaultdict
import hashlib
import json
import os
import sys
import time

import numpy as np

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib
import mlBridge.mlBridgeLib as mlBridgeLib
import mlBridge.dds_ddss as dds_ddss

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')

# note: not tournament option. club and tournament share one cache file.
CACHE_FILENAME = 'acbl_club_hand_records_cache_df.parquet'

# Everything below changes the outputs, so it is part of the fingerprint.
PARAMS = {
    'sd_productions': 10,
    'max_dd_adds': None, # todo: should max_(dd|sd)_adds of None be 0 or all?
    'max_sd_adds': None,
    'sample_size': 100000,
    'sample_seed': 0, # fixed seed so _small and _narrow are the same on every run
    'narrow_exclude_regex': r'^(C_|CT_.*_.*_|HB_)',
}
MANIFEST_VERSION = 1


def _output_files(acbl_path, club_or_tournament):
    return {
        'augmented': acbl_path.joinpath(f'acbl_{club_or_tournament}_hand_records_augmented.parquet'),
        'small': acbl_path.joinpath(f'acbl_{club_or_tournament}_hand_records_augmented_small.parquet'),
        'narrow': acbl_path.joinpath(f'acbl_{club_or_tournament}_hand_records_augmented_narrow.parquet'),
    }


def _file_stat(path):
    st = path.stat()
    return {'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def _hash_rows(df, ordered=True):
    """128-bit content hash of all rows (two 64-bit row hashes). Unordered sorts the row hashes first."""
    h0 = df.hash_rows(seed=0).to_numpy()
    h1 = df.hash_rows(seed=1).to_numpy()
    if not ordered:
        idx = np.lexsort((h1, h0))
        h0, h1 = h0[idx], h1[idx]
    b = hashlib.blake2b(digest_size=16)
    b.update(json.dumps([(n, str(t)) for n, t in df.schema.items()]).encode())
    b.update(str(df.height).encode())
    b.update(np.ascontiguousarray(np.stack([h0, h1], axis=1)).tobytes())
    return b.hexdigest()


def _cache_rows_hash(cache, side_pbns):
    """Hash of the cache rows whose PBN is in this side. Row order is ignored.
    Joins on PBN only so legacy rows with null Dealer/Vul are included too."""
    lf = cache.lazy() if isinstance(cache, pl.DataFrame) else cache
    rows = lf.join(side_pbns.lazy(), on='PBN', how='semi').collect()
    return _hash_rows(rows, ordered=False), rows.height


def _code_fingerprint():
    """Hashes of the code that determines the outputs, plus package versions."""
    modules = {
        'acbl_hand_records_augment': pathlib.Path(__file__),
        'mlBridgeAugmentLib': pathlib.Path(mlBridgeAugmentLib.__file__),
        'mlBridgeLib': pathlib.Path(mlBridgeLib.__file__),
        'dds_ddss': pathlib.Path(dds_ddss.__file__),
    }
    code = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in modules.items()}
    versions = {
        'polars': pl.__version__,
        'numpy': np.__version__,
        'endplay': mlBridgeAugmentLib.endplay.__version__,
        'ddss_available': bool(dds_ddss.DDSS_AVAILABLE),
    }
    return {'code': code, 'versions': versions}


def _read_manifest(manifest_file):
    try:
        return json.loads(manifest_file.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _write_manifest(manifest_file, manifest):
    tmp = manifest_file.with_name(manifest_file.name + '.tmp')
    tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    os.replace(tmp, manifest_file)


def _check_skip(manifest, fingerprint, hrs_df, outputs, cache_file):
    """Returns (skip, reason, updated_manifest). updated_manifest is set when only the cache stat needs refreshing."""
    if manifest is None:
        return False, 'no manifest', None
    if manifest.get('manifest_version') != MANIFEST_VERSION:
        return False, 'manifest version changed', None
    for name, path in outputs.items():
        if not path.exists():
            return False, f'{path.name} is missing', None
        if _file_stat(path) != manifest.get('outputs', {}).get(name):
            return False, f'{path.name} changed since the last run', None
    old = manifest.get('fingerprint', {})
    for key in ('code', 'versions', 'params'):
        if old.get(key) != fingerprint[key]:
            changed = sorted(k for k in set(old.get(key, {})) | set(fingerprint[key]) if old.get(key, {}).get(k) != fingerprint[key].get(k))
            return False, f'{key} changed: {changed}', None
    if old.get('input') != fingerprint['input']:
        if manifest.get('input_unordered') == _hash_rows(hrs_df, ordered=False):
            return False, 'cleaned input has the same rows in a different order', None
        return False, 'cleaned input changed', None
    # The cache rows hash is only recomputed when the cache file was rewritten.
    cache_stat = _file_stat(cache_file)
    if cache_stat == manifest.get('cache_file'):
        return True, 'inputs, code and parameters unchanged; cache file not rewritten', None
    t = time.time()
    cache_hash, cache_rows = _cache_rows_hash(pl.scan_parquet(cache_file), hrs_df.select('PBN'))
    print(f"Hashed {cache_rows} cache rows for this side in {time.time()-t:.1f}s")
    if cache_hash != old.get('cache_rows'):
        return False, 'cache rows for this side changed', None
    updated = dict(manifest, cache_file=cache_stat)
    return True, 'inputs, code and parameters unchanged; cache rewritten but rows for this side unchanged', updated


def augment_hand_records(club_or_tournament='club', acbl_path=None, force=False):
    """Augment hand records with DD analysis and probabilities."""
    print(f"Processing {club_or_tournament} hand records augmentation...")
    acbl_path = acblPath if acbl_path is None else pathlib.Path(acbl_path)
    outputs = _output_files(acbl_path, club_or_tournament)
    manifest_file = acbl_path.joinpath(f'acbl_{club_or_tournament}_hand_records_augmented_manifest.json')
    acbl_club_hand_records_cache_df_filename = CACHE_FILENAME
    acbl_club_hand_records_cache_df_file = acbl_path.joinpath(acbl_club_hand_records_cache_df_filename)

    # Load cleaned hand records
    acbl_hand_records_cleaned_filename = f'acbl_{club_or_tournament}_hand_records_cleaned.parquet'
    acbl_hand_records_cleaned_file = acbl_path.joinpath(acbl_hand_records_cleaned_filename)
    hrs_df = pl.read_parquet(acbl_hand_records_cleaned_file)
    print(f"Loaded {acbl_hand_records_cleaned_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_cleaned_file.stat().st_size}")

    # Fingerprint of everything the outputs depend on.
    t = time.time()
    fingerprint = {
        **_code_fingerprint(),
        'params': PARAMS,
        'input': _hash_rows(hrs_df),
    }
    if force:
        skip, reason, updated = False, '--force', None
    else:
        skip, reason, updated = _check_skip(_read_manifest(manifest_file), fingerprint, hrs_df,
                                            outputs, acbl_club_hand_records_cache_df_file)
    print(f"Fingerprint check took {time.time()-t:.1f}s")
    if skip:
        if updated is not None:
            _write_manifest(manifest_file, updated)
        print(f"Skipping {club_or_tournament}: {reason}. Outputs kept as is.")
        return
    print(f"Recomputing {club_or_tournament}: {reason}")
    # Drop the old manifest so a failed run can't be mistaken for a finished one.
    manifest_file.unlink(missing_ok=True)

    # Show columns with nulls
    null_counts = [(s.name, s[0]) for s in hrs_df.null_count() if s[0] > 0]
    if null_counts:
        print(f"Columns with nulls: {null_counts}")

    # Load or create cache
    hrs_cache_df = pl.read_parquet(acbl_club_hand_records_cache_df_file)
    print(f"Loaded {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")

    assert hrs_df['PBN'].null_count() == 0, hrs_df['PBN'].null_count()
    assert hrs_df['Dealer'].null_count() == 0, hrs_df['Dealer'].null_count()
    assert hrs_df['Vul'].null_count() == 0, hrs_df['Vul'].null_count()
    assert hrs_cache_df['PBN'].null_count() == 0, hrs_cache_df['PBN'].null_count()
    #assert hrs_cache_df['Dealer'].null_count() == 0, hrs_cache_df['Dealer'].null_count()
    #assert hrs_cache_df['Vul'].null_count() == 0, hrs_cache_df['Vul'].null_count()
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN','Dealer','Vul']).height, "hrs_cache_df has duplicate combinations of PBN+Dealer+Vul"
    assert hrs_cache_df.select(pl.col(pl.Float64)).width == 0, hrs_cache_df.select(pl.col(pl.Float64)).columns

    # takes 1h per 10000 sd calculations using 10 trials per pbn.
    if False: # only needed if > 50000 sd calculations
        max_sd_adds = 50000 # len(hrs_df)
        for i in range(0, len(hrs_df), max_sd_adds):
            batch_df = hrs_df.slice(i, max_sd_adds)

            hand_record_augmenter = mlBridgeAugmentLib.AllHandRecordAugmentations(
                batch_df,
                hrs_cache_df,
                sd_productions=10,
                max_dd_adds=None,
                max_sd_adds=max_sd_adds
            )
            _, hrs_cache_df = hand_record_augmenter.perform_all_hand_record_augmentations()

        # Save cache between batches
        hrs_cache_df.write_parquet(acbl_club_hand_records_cache_df_file)
        print(f"Batch {i//max_sd_adds + 1} of {len(hrs_df)//max_sd_adds} completed")
        print(f"Saved {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")

    # takes 64m/8m minimum (2m for 100 new SD productions, 3m for EVs, ?m for joins, 2m for best contracts, 11m total for DD/SD).
    # todo: is progress bar missing in  create sd calculations?
    loaded_cache_df = hrs_cache_df
    side_pbns = hrs_df.select('PBN')
    input_unordered = _hash_rows(hrs_df, ordered=False)
    all_augmentations = mlBridgeAugmentLib.AllHandRecordAugmentations(hrs_df,hrs_cache_df,sd_productions=PARAMS['sd_productions'],max_dd_adds=PARAMS['max_dd_adds'],max_sd_adds=PARAMS['max_sd_adds'],cache_file_path=acbl_club_hand_records_cache_df_file)
    hrs_df, hrs_cache_df = all_augmentations.perform_all_hand_record_augmentations()

    assert hrs_df['PBN'].null_count() == 0, hrs_df['PBN'].null_count()
    assert hrs_df['Dealer'].null_count() == 0, hrs_df['Dealer'].null_count()
    assert hrs_df['Vul'].null_count() == 0, hrs_df['Vul'].null_count()
    assert hrs_cache_df['PBN'].null_count() == 0, hrs_cache_df['PBN'].null_count()
    #assert hrs_cache_df['Dealer'].null_count() == 0, hrs_cache_df['Dealer'].null_count() # doesn't work if dd and sd do not track each other????
    #assert hrs_cache_df['Vul'].null_count() == 0, hrs_cache_df['Vul'].null_count()
    assert hrs_cache_df.height == hrs_cache_df.unique(subset=['PBN','Dealer','Vul']).height, "hrs_cache_df has duplicate combinations of PBN+Dealer+Vul"
    assert hrs_cache_df.select(pl.col(pl.Float64)).width == 0, hrs_cache_df.select(pl.col(pl.Float64)).columns

    # takes 1m to write 9m rows by 587 columns. 3.1GB file.
    # The lib returns the same DataFrame object when nothing was added or migrated.
    if hrs_cache_df is loaded_cache_df:
        print(f"{acbl_club_hand_records_cache_df_filename} unchanged: not rewritten")
    else:
        hrs_cache_df.write_parquet(acbl_club_hand_records_cache_df_file)
        print(f"Saved {acbl_club_hand_records_cache_df_filename}: shape:{hrs_cache_df.shape} size:{acbl_club_hand_records_cache_df_file.stat().st_size}")
    del loaded_cache_df
    t = time.time()
    fingerprint['cache_rows'], cache_rows = _cache_rows_hash(hrs_cache_df, side_pbns)
    print(f"Hashed {cache_rows} cache rows for this side in {time.time()-t:.1f}s")
    del hrs_cache_df

    assert all([col for col, dtype in zip(hrs_df.columns, hrs_df.dtypes) if dtype == pl.Object])

    # takes 20m?/45s for 17m/1.65m rows x 5904/5893 columns. 55GB/3.5GB file.
    acbl_hand_records_augmented_file = outputs['augmented']
    hrs_df.write_parquet(acbl_hand_records_augmented_file)
    print(f"Saved {acbl_hand_records_augmented_file.name}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    # takes 16m?/5s for 16m/7m->100K rows x 5904/5893 columns. 375MB file.
    # save hrs_df after sampling rows
    sample_size = PARAMS['sample_size']
    acbl_hand_records_augmented_file = outputs['small']
    hrs_df = hrs_df if sample_size >= len(hrs_df) else hrs_df.sample(sample_size, seed=PARAMS['sample_seed'])
    hrs_df.write_parquet(acbl_hand_records_augmented_file)
    print(f"Saved {acbl_hand_records_augmented_file.name}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    # takes 10s/5s for 100K rows x 5664/5546 columns. 372MB/280MB file.
    # saving hrs_df after filtering out column names and using sampled rows
    acbl_hand_records_augmented_file = outputs['narrow']
    # filter out columns starting with C_ or CT_.*_.*_ or HB_
    hrs_df = hrs_df.select(~pl.selectors.matches(PARAMS['narrow_exclude_regex'])) # todo: probably should be done using the more commonly use pl.exclude()
    hrs_df.write_parquet(acbl_hand_records_augmented_file) # note: regex is not including CT_.*_.* e.g. CT_NS_S. Should it?
    print(f"Saved {acbl_hand_records_augmented_file.name}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    # Written last, so it only exists after all outputs are written.
    _write_manifest(manifest_file, {
        'manifest_version': MANIFEST_VERSION,
        'side': club_or_tournament,
        'written': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        'fingerprint': fingerprint,
        'input_unordered': input_unordered,
        'cache_file': _file_stat(acbl_club_hand_records_cache_df_file),
        'outputs': {name: _file_stat(path) for name, path in outputs.items()},
    })
    print(f"Saved {manifest_file.name}")

    return

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Augment ACBL hand records with DD/SD analysis.")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    parser.add_argument("--force", action="store_true", help="Recompute even if inputs are unchanged")
    parser.add_argument("--data-dir", default=str(acblPath), help=f"Folder with the input, cache and output files (default {acblPath})")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        args.modes = ["club", "tournament"]
    else:
        args.modes = [m for m in ("club", "tournament") if getattr(args, m)]
    return args


if __name__ == "__main__":

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    args = _parse_args()
    for club_or_tournament in args.modes:
        t = time.time()
        augment_hand_records(club_or_tournament, acbl_path=args.data_dir, force=args.force)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)
