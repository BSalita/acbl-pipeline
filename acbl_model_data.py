#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_model_data.py

Creates final model-ready dataset from augmented board results.
Year-chunking refactor (2026-04-20, second pass) -- previously the streaming
sink_parquet path itself blew up to 600 GB committed memory on club before
writing a single byte, mirroring the original eager-mode pagefile thrash.

Why year-chunking:
    The "streaming" Polars pipeline (scan_parquet -> select 6772 cols -> join
    -> sink_parquet) does NOT actually stream for a schema this wide. Even
    with broadcast inner-join on the small hrs side, the engine pre-allocates
    intermediates on the order of (rows * cols * dtype_bytes) before any
    output bytes appear. Killed PID 10896 at 600 GB committed before Windows
    locked up.
    Bounding the plan to one calendar year at a time caps the row count at
    ~6-8M (instead of 58.8M for club) and reduces the worst-case memory by
    the same factor. Each year sinks to a shard, then a final streaming
    concat merges them. Resume-safe: a shard is skipped only when the source
    window still has the row count stored in manifest.json (source_rows).
    Older shards, which lack that field, are skipped only when their Date
    max still equals the source window.

Architecture:
    1. Eager schema-validation prelude (unchanged): build 0-row joined schema,
       compute columns_matching_regex, save the *_model_data_d.pkl metadata.
    2. Eager small-side load: hrs deal_features (5 cols x 1.8M-19M rows = small).
    3. Per-year streaming sink: for each calendar year in the brs Date range,
       scan_parquet(brs).filter(Date in [y, y+1)) -> project -> inner-join
       hrs.lazy() -> sink_parquet(shard, maintain_order=False, zstd:3).
    4. Output: shards stay in shards_{mode}_model_data/ plus a manifest.json
       (default since 2026-08-16). Downstream consumers scan the shard glob;
       benchmarks showed monthly shards beat a single merged file on every
       access pattern (file-level Date pruning: 1.4s vs 674s for a 50-col
       one-year read) and the single-writer merge recompressed ~86 GB of
       6,778-col data at ~2 MiB/s (~12 h). Pass --merge-shards only if a
       single canonical *_model_data.parquet is really needed.

CLI:
    --club, --tournament         (default: both)
    --chunk-years / --no-chunk-years  (default: chunk; recommended)
    --start-year / --end-year    (default: source min/max year, inclusive)
    --merge-shards / --no-merge-shards  (default: NO merge; keep shards + manifest)
    --keep-shards                (default: False; with --merge-shards, delete shards after)

Wall-clock baselines (dev box: 192 GB RAM, ~40 cores, NVMe E:):
    tournament: ~15 min   (16.73M rows x 6786 cols -> 132 monthly shards,
                           15.3 GB; measured 889 s on 2026-08-16)
    club:       ~60-75 min cold (69.45M rows x 6778 cols -> 96 monthly
                           shards, 86 GB; ~60 min measured 2026-04-20 at
                           58.84M rows -- grows with data)
    resume:     ~1 min per mode when all shards validate (club measured
                           54 s on 2026-08-16)
    Logs: logs/05a_model_data_noshardmerge_*.log. The removed single-file
    merge added ~80 min (2026-07, pyarrow) to ~12 h (2026-08, at 86 GB).

Previous steps:
    acbl_board_results_augment_step2.py
    acbl_elo_ratings_create.py

Next steps:
    acbl_prediction_data.py (or acbl_predict_torch.py for legacy torch path)
"""

import polars as pl
import pathlib
import re
from collections import defaultdict
import pickle
import psutil
import sys
import time
import gc

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
_s = str(_SRC_DIR)
if _s not in sys.path:
    sys.path.append(_s)
import mlBridge
from mlBridge.mlBridgeAiLib import features_enum

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')


# todo: put somewhere, into mlBridgeAiLib?
def show_estimated_memory_usage(df):

    estimated_size_df = df.estimated_size()

    # Per-column bytes and MB
    estimated_size_per_col = (
        pl.DataFrame({
            "column": df.columns,
            "bytes": [df[c].estimated_size() for c in df.columns],
        })
        .with_columns((pl.col("bytes") / (1024**2)).round(2).alias("MB"))
        .sort("bytes", descending=True)
    )

    estimated_size_per_dtype = (
        pl.DataFrame({
            "dtype": [str(df[c].dtype) for c in df.columns],
            "bytes": [df[c].estimated_size() for c in df.columns],
        })
        .group_by("dtype")
        .agg(pl.col("bytes").sum().alias("bytes"))
        .with_columns((pl.col("bytes") / (1024**2)).round(2).alias("MB"))
        .sort("bytes", descending=True)
    )

    return estimated_size_df, estimated_size_per_col, estimated_size_per_dtype


def show_memory_usage(label=''):
    proc = psutil.Process()
    mem = proc.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"[Memory{' - ' + label if label else ''}] "
          f"Process: RSS={mem.rss / (1024**3):.1f}GB VMS={mem.vms / (1024**3):.1f}GB | "
          f"System: used={vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f}GB ({vm.percent}%) | "
          f"Pagefile: used={swap.used / (1024**3):.1f}/{swap.total / (1024**3):.1f}GB ({swap.percent}%)")


def _build_join_plan(brs_lf, *, hrs_df, brs_read_cols, join_keys):
    """Build the canonical brs.select(cols).join(hrs.lazy(), inner) plan
    on top of an arbitrary brs LazyFrame (e.g. unfiltered scan, or scan
    with year filter).

    Any Float64 columns are downcast to Float32 in the plan. The dtype
    conversions computed for the *_model_data_d.pkl metadata only touch the
    0-row schema frame, never the streamed data, so a source regenerated
    with f64 columns (e.g. tournament MP_NS/MP_EW, 2026-08) would otherwise
    trip _validate_plan_schema's no-Float64 assertion."""
    lf = (
        brs_lf.select(brs_read_cols)
        .join(hrs_df.lazy(), on=join_keys, how='inner')
    )
    f64_cols = [c for c, dt in lf.collect_schema().items() if dt == pl.Float64]
    if f64_cols:
        lf = lf.with_columns([pl.col(c).cast(pl.Float32) for c in f64_cols])
    return lf


def _validate_plan_schema(lf, *, label):
    """Common pre-flight assertions on a built lazy plan: no Float64
    surprises (would 2x output size), no _x/_y/_right/_left collisions
    (indicates a bad/forgotten join rename)."""
    plan_schema = lf.collect_schema()
    f64_cols = [c for c, dt in plan_schema.items() if dt == pl.Float64]
    assert not f64_cols, (
        f"[{label}] Float64 columns in plan (would expand size 2x): {f64_cols}"
    )
    suffix_cols = [
        c for c in plan_schema.names()
        if re.match(r'^.*_(x|y|right|left)$', c)
    ]
    assert not suffix_cols, (
        f"[{label}] Suffix-collision columns in plan: {suffix_cols}"
    )
    return plan_schema


# Polars' sink_parquet defaults to 262K rows per row group. For our 6,772-column
# schema that's 262K * 6772 * ~8 B = ~14 GB *per row group buffer*, and the
# streaming engine appears to allocate several in parallel before flushing
# (observed: 745 GB committed memory on a single-year 2024 club shard at the
# default size, before any output bytes were written). Capping the row group
# at 16K rows brings the per-buffer footprint to ~870 MB and made club
# sinks finish in bounded memory. Keep the on-disk row groups small but not
# tiny -- too small wastes Parquet metadata overhead and downstream scan
# parallelism.
SINK_ROW_GROUP_SIZE = 16384  # rows per parquet row group (was 262144 default)
SINK_DATA_PAGE_SIZE = 1024 * 1024  # 1 MiB pages (was 1 MiB default)


def _sink_parquet_safe(lf, path, *, row_group_size=SINK_ROW_GROUP_SIZE,
                       engine='streaming'):
    """Wrapper around sink_parquet with our memory-safe defaults baked in
    (small row groups, zstd:3, no order maintenance, new streaming engine).

    engine='streaming' explicitly selects Polars' v2 streaming engine
    (added in 1.x). The default engine='auto' was observed to silently
    choose an in-memory path for our 6,772-column schema, leading to
    197+ GB committed memory before any output bytes were written
    (PID 6740, 2026-04-20). The v2 streaming engine processes data in
    bounded morsels (~16K rows by default) and never materializes the
    full plan."""
    lf.sink_parquet(
        path,
        compression='zstd',
        compression_level=3,
        maintain_order=False,
        row_group_size=row_group_size,
        data_page_size=SINK_DATA_PAGE_SIZE,
        engine=engine,
    )


def _stream_concat_shards(shards, out_path, *, label):
    """Concatenate per-shard parquets into a single parquet, streaming one row
    group at a time via pyarrow.

    We deliberately do NOT use polars' ``pl.concat([scan...]).sink_parquet``
    here: on our ~6,772-column schema the streaming engine intermittently aborts
    a 96-way concat+sink with a misleading ``parquet: File out of specification:
    ... Data corruption detected`` even though every shard reads back perfectly
    on its own (verified 2026-07-08). pyarrow copies row groups verbatim with
    bounded memory (one row group in flight) and is robust at this width."""
    import pyarrow.parquet as pq

    print(f"Merging {len(shards)} {label} shards -> {out_path.name}")
    show_memory_usage(f'before merge {label}')
    t = time.time()

    tmp_path = out_path.with_name(out_path.name + '.merge.tmp')
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    total_rows = 0
    try:
        for s in shards:
            pf = pq.ParquetFile(str(s))
            try:
                if pf.metadata.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(tmp_path), pf.schema_arrow,
                        compression='zstd', compression_level=3,
                    )
                for i in range(pf.num_row_groups):
                    tbl = pf.read_row_group(i)
                    writer.write_table(tbl)
                    total_rows += tbl.num_rows
            finally:
                pf.close()
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # All shards were empty; nothing written.
        if tmp_path.exists():
            tmp_path.unlink()
        print(f"  {label}: all shards empty; no merged file written")
        show_memory_usage(f'after merge {label}')
        return 0

    if out_path.exists():
        out_path.unlink()
    tmp_path.rename(out_path)

    elapsed = time.time() - t
    size = out_path.stat().st_size
    print(f"  {label} merged in {elapsed:.1f}s; rows={total_rows:,}; "
          f"size={size/(1024**3):.2f}GB")
    show_memory_usage(f'after merge {label}')
    return size


def _write_shard_manifest(shard_dir, shards, *, label, source_by_file=None):
    """Write manifest.json describing the complete shard set.

    Consumers assert every listed file still exists (and optionally check
    row totals) before scanning the shard glob, so a manually deleted or
    half-written shard fails fast instead of silently dropping months.
    Row counts are read from parquet footers only (~1 s for ~100 shards).
    source_by_file records the pre-join board-results row count for each
    window so the next run can rebuild a month that gained or lost rows.
    The inner join means shard row count is not that source count.
    """
    import json
    import pyarrow.parquet as pq
    from datetime import datetime, timezone

    entries = []
    total_rows = 0
    for s in shards:
        if not s.exists():
            continue
        n = pq.ParquetFile(str(s)).metadata.num_rows
        entry = {'file': s.name, 'rows': n, 'bytes': s.stat().st_size}
        extra = (source_by_file or {}).get(s.name)
        if extra is not None:
            entry['source_rows'] = int(extra['source_rows'])
            if extra.get('source_date_max'):
                entry['source_date_max'] = extra['source_date_max']
        entries.append(entry)
        total_rows += n
    manifest = {
        'label': label,
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'shard_count': len(entries),
        'total_rows': total_rows,
        'shards': entries,
    }
    path = shard_dir.joinpath('manifest.json')
    path.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f"Wrote {path}: {len(entries)} shards, {total_rows:,} rows")
    return manifest


def _as_date(value):
    """Normalize polars/pyarrow date stats to datetime.date."""
    from datetime import date, datetime
    if value is None:
        return None
    if hasattr(value, 'as_py'):
        value = value.as_py()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return value


def _read_recorded_source_rows(shard_dir):
    """filename -> source_rows from the previous manifest, if that field exists."""
    import json
    path = shard_dir.joinpath('manifest.json')
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    recorded = {}
    for entry in data.get('shards', []):
        if 'source_rows' in entry and entry.get('file'):
            recorded[entry['file']] = int(entry['source_rows'])
    return recorded


def _parquet_num_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


def _parquet_date_max(path):
    """Date max from parquet column statistics, falling back to a column scan."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(str(path))
    names = list(pf.schema_arrow.names)
    if 'Date' not in names:
        return None
    idx = names.index('Date')
    mx = None
    for rg_i in range(pf.metadata.num_row_groups):
        stats = pf.metadata.row_group(rg_i).column(idx).statistics
        if stats is None or not stats.has_min_max:
            mx = None
            break
        v = _as_date(stats.max)
        if v is None:
            mx = None
            break
        if mx is None or v > mx:
            mx = v
    else:
        return mx
    value = pl.scan_parquet(path).select(pl.col('Date').max()).collect().item()
    return _as_date(value)


def _source_window_stats(source_file, windows):
    """One Date scan. Map each [start, end) window to (row_count, date_max)."""
    from dateutil.relativedelta import relativedelta
    monthly = (
        pl.scan_parquet(source_file)
        .select(pl.col('Date').cast(pl.Date))
        .filter(pl.col('Date').is_not_null())
        .group_by(pl.col('Date').dt.truncate('1mo').alias('month'))
        .agg(
            pl.len().alias('n'),
            pl.col('Date').max().alias('mx'),
        )
        .collect()
    )
    by_month = {}
    for row in monthly.iter_rows(named=True):
        month = _as_date(row['month'])
        by_month[month] = (int(row['n']), _as_date(row['mx']))
    stats = {}
    for w_start, w_end in windows:
        n = 0
        mx = None
        cur = w_start
        while cur < w_end:
            got = by_month.get(cur)
            if got:
                month_n, month_mx = got
                n += month_n
                if month_mx is not None and (mx is None or month_mx > mx):
                    mx = month_mx
            cur = cur + relativedelta(months=1)
        stats[(w_start, w_end)] = (n, mx)
    return stats


def _model_shard_is_current(shard, src_n, src_mx, recorded_source_n, shard_label):
    """True when the shard can be skipped. Unreadable or stale shards are deleted."""
    try:
        pl.read_parquet_schema(shard)
        shard_n = _parquet_num_rows(shard)
    except Exception as e:
        print(f"[{shard_label}] shard exists but unreadable ({e}); "
              f"deleting and re-creating")
        shard.unlink()
        return False
    if src_n == 0:
        if shard_n == 0:
            print(f"[{shard_label}] shard matches empty source; skipping")
            return True
        print(f"[{shard_label}] source window is empty but shard has "
              f"{shard_n:,} rows; rebuilding")
        shard.unlink()
        return False
    if recorded_source_n is not None:
        if recorded_source_n == src_n:
            print(f"[{shard_label}] shard matches source ({src_n:,} rows, "
                  f"max {src_mx}); skipping")
            return True
        print(f"[{shard_label}] source rows {recorded_source_n:,} -> {src_n:,} "
              f"(max {src_mx}); rebuilding")
        shard.unlink()
        return False
    if shard_n == 0:
        print(f"[{shard_label}] shard is empty but source has {src_n:,} rows "
              f"(max {src_mx}); rebuilding")
        shard.unlink()
        return False
    shard_mx = _parquet_date_max(shard)
    if shard_mx == src_mx:
        print(f"[{shard_label}] shard Date max {shard_mx} matches source "
              f"({src_n:,} rows); skipping")
        return True
    print(f"[{shard_label}] shard Date max {shard_mx} != source {src_mx} "
          f"({src_n:,} rows); rebuilding")
    shard.unlink()
    return False


def create_model_data(
    club_or_tournament='club',
    *,
    chunk_years=True,
    start_year=None,
    end_year=None,
    merge_shards=False,
    keep_shards=False,
    months_per_chunk=1,
):
    """Create final model-ready dataset."""
    print(f"Processing {club_or_tournament} model data creation...")
    show_memory_usage('start')

    # columns not mentioned will block them from becoming features for training. They will still be in the saved dataframe but not available to a model.
    # '# no matches:' means the match string doesn't have any matches. This may be due to deprecation, or waiting implementation of a desirable feature.
    # todo: need to uncommented columns replacing with boolean of whether to include or not. Issue is that casts have to be applied.
    # todo: continue reviewing for unhelpful/redundant columns from becoming training features. remove all string columns?
    # todo: game state numbering valid after renumbering?

    # known by board design. might not be universally same across bridge cultures. usually invariant board info.
    invariant_board_features = {
        'Board':pl.UInt8,
        'sBoard':pl.String,
        'Dealer':pl.Categorical, # N, E, S, W
        'iVul':pl.UInt8, # 0-3 # pl.Categorical?
        'Vul_(NS|EW)':pl.Boolean,
        'Vul':pl.Categorical, # None, NS, EW, Both
    }

    # deal features are knowable/derived from hand record.
    # todo?: 1 hand is knownable by player holding hand. 4 hands are knownable by all players after play ends.
    deal_features = {
        # todo: how to express that each player knows their own hand, but not the others?
        'PBN':pl.String,
        'C_[NESW][CDHS][2-9TJQKA]':pl.Boolean,
        'CT_[NESW]_[CDHSN]':pl.Categorical,
        'CT_[NESW]_[CDHSN]_.*':pl.Boolean, # wildcard
        # no matches: 'CT_(NS|EW)_[CDHSN]':pl.Categorical,
        'CT_(NS|EW)_[CDHSN]_.*':pl.Boolean, # wildcard
        'DD_[NESW]_[CDHSN]':pl.UInt8,
        'DD_(NS|EW)_[CDHSN]':pl.UInt8,
        'DD_Score_[1-7][CDHSN]_([NESW]|NS|EW)':pl.Int16,
        'DP_[NESW]':pl.UInt8,
        'DP_[NESW]_[CDHS]':pl.UInt8,
        'DP_(NS|EW)':pl.UInt8,
        'DP_(NS|EW)_[CDHS]':pl.UInt8,
        'hand_record_id':pl.Int64, # todo: or String?
        'board_result_id':pl.String,
        'HandRecordBoard':pl.String,
        #'HB_[NESW]':pl.String, # can't see it being useful for training. redundant with other features. Maybe useful for searching?
        'HCP_[NESW]':pl.UInt8,
        'HCP_[NESW]_[CDHS]':pl.UInt8,
        'HCP_(NS|EW)':pl.UInt8,
        'HCP_(NS|EW)_[CDHS]':pl.UInt8,
        # no matches: 'LoTT_SL':pl.UInt8,
        # no matches: 'LoTT_DD':pl.UInt8,
        #'LoTT_Tricks':pl.UInt8,
        #'LoTT_Suit_Length':pl.UInt8,
        #'LoTT_Variance':pl.Int8,
        'LoTT_(NS|EW)':pl.UInt8,
        # no matches: 'LoTT_SL_(NS|EW)_[CDHS]':pl.UInt8,
        # no matches: 'LoTT_SL_(NS|EW)':pl.UInt8,
        # no matches: 'LoTT_DD_(NS|EW)':pl.UInt8,
        # no matches: 'LoTT_DD_(NS|EW)_[CDHSN]':pl.UInt8,
        'LTC_[NESW]':pl.UInt8,
        'LTC_[NESW]_[CDHS]':pl.UInt8,
        'LTC_(NS|EW)':pl.UInt8,
        'LTC_(NS|EW)_[CDHS]':pl.UInt8,
        'Par_(NS|EW)':pl.Int16,
        'QL_[NESW]':pl.UInt8,
        'QL_[NESW]_[CDHS]':pl.UInt8,
        'QL_(NS|EW)':pl.UInt8,
        'QL_(NS|EW)_[CDHS]':pl.UInt8,
        'QT_[NESW]':pl.Float32,
        'QT_[NESW]_[CDHS]':pl.Float32,
        'QT_(NS|EW)':pl.Float32,
        'QT_(NS|EW)_[CDHS]':pl.Float32,
        'SL_[NESW]_[CDHS]':pl.UInt8,
        'SL_(NS|EW)_[CDHS]':pl.UInt8,
        'SL_Max_(NS|EW)':pl.Categorical, # or string
        # no matches: 'SL_[NESW]_[CDHS]_SJ':pl.Categorical, # todo: reimplement
        'SL_[NESW]_ML_SJ':pl.Categorical, # todo: reimplement
        r'Probs_(NS|EW)_[NESW]_[CDHSN]_\d+':pl.Float32,
        'EV_Max':pl.Float32,
        'EV_Max_Col':pl.String,
        'EV_Max_(NS|EW)':pl.Float32,
        'EV_Max_Col_(NS|EW)':pl.String,
        # no matches: 'EV_Max_(NS|EW)_[NESW]':pl.Float32,
        'EV_(NS|EW)_[NESW]_Max':pl.Float32,
        'EV_(NS|EW)_[NESW]_Max_Col':pl.String,
        'EV_(NS|EW)_[NESW]_(NV|V)_Max':pl.Float32,
        'EV_(NS|EW)_[NESW]_(NV|V)_Max_Col':pl.String,
        'EV_(NS|EW)_(NV|V)_Max':pl.Float32,
        'EV_(NS|EW)_(NV|V)_Max_Col':pl.String,
        'EV_(NS|EW)_[NESW]_[CDHSN]_[1-7]':pl.Float32,
        'EV_(NS|EW)_[NESW]_[CDHSN]_Max':pl.Float32,
        'EV_(NS|EW)_[NESW]_[CDHSN]_Max_Col':pl.String,
        'EV_(NS|EW)_[NESW]_[CDHSN]_[1-7]_(NV|V)':pl.Float32,
        r'EV_(NS|EW)_[NESW]_[CDHSN]_[1-7]_(NV|V)_\d+_-?\d+':pl.Float32,
        'EV_(NS|EW)_[NESW]_[CDHSN]_(NV|V)_Max':pl.Float32,
        'EV_(NS|EW)_[NESW]_[CDHSN]_(NV|V)_Max_Col':pl.String,
        'EV_(NV|V)_Max':pl.Float32,
        'EV_(NV|V)_Max_Col':pl.String,
        'DD_Score_[CDHSN]_[NESW]_Max':pl.Int16,
        'DD_Score_[CDHSN]_(NS|EW)_Max':pl.Int16,
        'Hand_[NESW]':pl.String,
        'Hands':pl.List(pl.List(pl.String)),
        'Suit_[NESW]_[SHDC]':pl.String,
        'SL_[NESW]_CDHS':pl.String,
        'SL_[NESW]_CDHS_SJ':pl.Categorical,
        'SL_[NESW]_ML':pl.String,
        'SL_[NESW]_ML_I':pl.String,
        'SL_[NESW]_ML_I_SJ':pl.Categorical,
        'SL_Max_(NS|EW)_Col':pl.String,
        'Total_Points_[NESW]_[SHDC]':pl.UInt8,
        'Total_Points_[NESW]':pl.UInt8,
        'Total_Points_(NS|EW)':pl.UInt8,
        'Biddable_[NESW]_[SHDC]':pl.Boolean,
        'Rebiddable_[NESW]_[SHDC]':pl.Boolean,
        'Twice_Rebiddable_[NESW]_[SHDC]':pl.Boolean,
        'Strong_Rebiddable_[NESW]_[SHDC]':pl.Boolean,
        'Solid_[NESW]_[SHDC]':pl.Boolean,
        'At_Best_Partial_Stop_In_[NESW]_[SHDC]':pl.Boolean,
        'Partial_Stop_In_[NESW]_[SHDC]':pl.Boolean,
        'Likely_Stop_In_[NESW]_[SHDC]':pl.Boolean,
        'Stop_In_[NESW]_[SHDC]':pl.Boolean,
        'At_Best_Stop_In_[NESW]_[SHDC]':pl.Boolean,
        'Two_Stops_In_[NESW]_[SHDC]':pl.Boolean,
        'Forcing_One_Round':pl.Boolean,
        'Opponents_Cannot_Play_Undoubled_Below_2N':pl.Boolean,
        'Forcing_To_2N':pl.Boolean,
        'Forcing_To_3N':pl.Boolean,
        'Balanced_[NESW]':pl.Boolean,
        'ParScore':pl.Int16,
        'ParContracts':pl.List(pl.Struct({'Level': pl.String, 'Strain': pl.String, 'Doubled': pl.String, 'Pair_Direction': pl.String, 'Result': pl.Int16})),
        'ParNumber':pl.Int8,
        'Probs_Trials':pl.Int64,
    }

    # known when event is sanctioned.(?)
    event_features = {
        #'ClubEvent':pl.String,
        'Club':pl.Categorical,
        #'ClubDate':pl.String,
        'Date':pl.Datetime,
        #'Event':pl.Categorical, # todo: doesn't exist. why?
        'event_id':pl.Utf8, # can't convert. has alphanumerics.
        #'iDate':pl.Int64, # todo: doesn't exist. why?
        #'start_date':pl.Datetime, # todo: doesn't exist. why?
        'event_type':pl.Categorical, # PAIRS, TEAMS, INDIVIDUAL?
        #'game_type':pl.Categorical, # tournament only
        #'masterpoints_limit':pl.Int32, # tournament only
        #'masterpoints_color':pl.Categorical, # tournament only
        #'masterpoints_rating':pl.Categorical, # tournament only
    }

    # known when player registers
    players_features = {
        'Player_ID_[NESW]':pl.Categorical, # pl.Categorical,
        'Player_Name_[NESW]':pl.String, # block from training
        #'Player_Names_(NS|EW)':pl.List(pl.String), # block from training # s/b Pair_Names_(NS|EW)
        'Pair_IDs_(NS|EW)':pl.List(pl.String),
        'Pair_Names_(NS|EW)':pl.List(pl.String),
        'Elo_E_Pair_(NS|EW)':pl.Float32, # ok to be included in players_features? # should be as of Date.
        'Elo_E_Players_[NESW]':pl.Float32, # ok to be included in players_features? # should be as of Date.
        'Elo_N_[NESW]':pl.UInt16, # should be as of Date.
        'Elo_N_(NS|EW)':pl.UInt16, # should be as of Date.
        'Elo_R_Player_[NESW]_EventStart':pl.Float32, # should be as of Date.
        'Elo_R_Pair_(NS|EW)_EventStart':pl.Float32, # should be as of Date.
        'MasterPoints_[NESW]':pl.Float32, # should be as of Date.
        #'MasterPoints_(NS|EW)':pl.Float32, # deprecated - todo: use MasterPoints_Sum_(NS|EW) instead
        # no matches 'MasterPoints_Sum_(NS|EW)':pl.Float32,
        # no matches 'MasterPoints_Geo_(NS|EW)':pl.Float32,
        #'Player_Names_(NS|EW)':pl.String, # todo: deprecate replace by Pair_Names_(NS|EW)
    }

    # known when players are assigned to tables and before game starts
    session_section_features = {
        #'board_id':pl.UInt8, # todo: doesn't exist. why?
        # no matches: 'Session':pl.Int64, # todo: pl.Categorical,
        'session_id':pl.String,
        'section_id':pl.String,
        #'section_name':pl.String, # todo: doesn't exist. why?
        'board_scoring_method':pl.Categorical, # IMP, MATCHPOINTS, etc. # todo: drop this after asserting all are MATCHPOINTS
        'club_session':pl.Categorical,
        #'ClubDateBoard':pl.String,
        #'ClubEventBoard':pl.String,
        'game_date':pl.String, # todo: redundant with 'Date'?
        'group_id':pl.String,
        #'HandRecord':pl.Int64, # todo: doesn't exist. why?
        #'HandRecordBoard':pl.String, # todo: Use hand_record_id instead?
        #'Pair(NS|EW)':pl.String, # todo? rename or drop? acbl number _ acbl number
        #'(NS|EW)Pair':pl.String, # todo? rename or drop? acbl number _ acbl number
        #'(ns|ew)_pair_summary_id':pl.Int32,
        'Pair_Number_(NS|EW)':pl.UInt8,
        'Round':pl.UInt8,
        'Table':pl.UInt8,
        'tb_count':pl.UInt8,
        'MP_Top':pl.UInt32, # todo: rename to MatchPoints_Top
    }

    if club_or_tournament == 'club':
        session_section_features['section_name'] = pl.String
        session_section_features['club_class'] = pl.UInt8
        session_section_features['is_virtual_game'] = pl.Boolean
        session_section_features['virtualGameType'] = pl.UInt8
    elif club_or_tournament == 'tournament':
        session_section_features['section_name'] = pl.String
        session_section_features['session_number'] = pl.UInt8
        session_section_features['start_date'] = pl.String
        session_section_features['start_time'] = pl.String
        session_section_features['(north|east|south|west)_(spades|hearts|diamonds|clubs)'] = pl.String

    final_contract_features = {
        #'auction':pl.String, # todo: check dtype
        'BidLvl':pl.UInt8,
        'BidSuit':pl.Categorical,
        'Contract':pl.Categorical, # todo: try again with creating a Contract Categorical with all possible contracts. Otherwise, have to cast to String if y name.
        'ContractType':pl.Categorical,
        'Dbl':pl.Categorical,
        'Declarer':pl.String, # todo: make a Categorical? rename to Declarer_ID?
        #'Declarer.*':pl.String, # todo
        #'iDeclarer':pl.Int32, # todo? pl.Categorical,
        'Declarer_Direction':pl.Categorical,
        r'Prob_Taking_\d+':pl.Float32,
        # no matches: 'Declarer_MP':pl.Float32,
        #'Declarer_Name':pl.String,
        #'Declarer_Pair':pl.String, # todo? acbl number _ acbl number
        'Declarer_Pair_Direction':pl.Categorical,
        'Defender_Pair_Direction':pl.Categorical,
        #'iOnLead':pl.Int32,
        'OnLead':pl.String, # todo: obsolete?
        # no matches: 'OnLead_MP':pl.Float32,
        #'iDummy':pl.Int32,
        'Dummy':pl.String, # todo: obsolete?
        # no matches: 'Dummy_MP':pl.Float32,
        'NotOnLead':pl.String, # todo: obsolete?
        # no matches: 'NotOnLead_MP':pl.Float32,
        #'iNotOnLead':pl.Int32,
        'Vul_Declarer':pl.Categorical,
        'Declarer_ID':pl.String,
        'Declarer_Name':pl.String,
        'Direction_OnLead':pl.Categorical,
        'Direction_Dummy':pl.Categorical,
        'Direction_NotOnLead':pl.Categorical,
        'LHO_Direction':pl.Categorical,
        'Dummy_Direction':pl.Categorical,
        'RHO_Direction':pl.Categorical,
        'Declarer_Rating':pl.Float32,
        'Defender_OnLead_Rating':pl.Float32,
        'Defender_NotOnLead_Rating':pl.Float32,
        'DD_Tricks':pl.UInt8,
        'DD_Tricks_Dummy':pl.UInt8,
        'DD_Score_Refs':pl.Int16,
        'DD_Score_Declarer':pl.Int16,
        'DD_Score_Max_Declarer':pl.Int16,
        'DD_Pct_Max_(NS|EW)':pl.Float32,
        'Par_Declarer':pl.Int16,
        'Par_Contract_(NS|EW)':pl.Int8,
        'Is_Par_Suit':pl.Boolean,
        'Is_Par_Contract':pl.Boolean,
        'Is_Sacrifice_Opportunity':pl.Boolean,
        'Is_Sacrifice':pl.Boolean,
        'EV_Max_Declarer':pl.Float32,
        'EV_Max_Col_Declarer':pl.String,
        'EV_Score_Col_Declarer':pl.Float32,
        'EV_Score_Declarer':pl.Float32,
        'EV_Score_(NS|EW)':pl.Float32,
        'MP_DD_Score_(NS|EW|Declarer)':pl.Int16,
        'MP_DD_Score_Max_(NS|EW)':pl.Int16,
        'MP_Par_Declarer':pl.Int16,
        'MP_EV_Score_Declarer':pl.Float32,
        'MP_Par_Pct_Declarer':pl.Float32,
        'MP_EV_Max_Declarer':pl.Float32,
        'MP_DD_Pct_Declarer':pl.Float32,
        'MP_EV_Pct_Declarer':pl.Float32,
        'MP_EV_Max_Pct_Declarer':pl.Float32,
        # no matches: 'Declarer_MP_Pair_Sum':pl.Float32,
        # no matches: 'Declarer_MP_Pair_Geo':pl.Float32,
        # no matches: 'Defender_Pair':pl.String, # todo? acbl number _ acbl number
        # no matches: 'Defender_MP_Pair_Sum':pl.Float32,
        # no matches: 'Defender_MP_Pair_Geo':pl.Float32,
        'DD_Score_(NS|EW)':pl.Int16,
        # no matches: 'Declarer_DD_Tricks':pl.UInt8,
        # no matches: 'Declarer_DD_Score':pl.Int16,
        # no matches: 'Declarer_ParScore':pl.Int16,
        #'Declarer_Vul':pl.Categorical, # None, NS, EW, Both
        # no matches: 'Declarer_EV_Contract_Max':pl.Categorical, # todo: Categorical of type Contract
        #'Declarer_EV_Probs':pl.Array(pl.Float32, shape=(14,)), # todo: use it or drop it?
        # no matches: 'Declarer_EV_Score':pl.Float32,
        #'Declarer_EV_Scores':pl.Array(pl.Int16, shape=(14,)), # todo: use it or drop it?
        # no matches: 'Declarer_EV_Score_Max':pl.Float32,
        #'Declarer_EV_Score_L[1-7]':pl.Array(pl.Float32, shape=(14,)), # todo: use it or drop it?
        #'Declarer_EV_EVs_L[1-7]':pl.Array(pl.Float32, shape=(14,)), # todo: use it or drop it?
        #'Declarer_Tricks_DD_Diff':pl.Int8, # obsoleted by DD_Tricks_Diff
    }

    # known during play
    opening_lead_features = {
        'Lead':pl.Categorical,
        #'Opening_Lead_Suit':pl.Categorical, # todo: same as BidSuit category
        #'Opening_Lead_Rank':pl.UInt8,
        #'Opening_Lead_Convention':pl.Categorical, # todo: 4th longest and strongest, high from doubleton, etc.
    }

    # known after lead is made
    dummy_features = {
        # todo: none of these are in the df
        #'Dummy_HCP':pl.UInt8,
        #'Dummy_QT':pl.Float32, # todo: could be string or ordred categorical or int*10 (200,150,100,50,etc.)
        #'Dummy_SL_[CDHS]':pl.UInt8,
        #'Dummy_DP':pl.UInt8,
        #'Dummy_C_[2-9TJQKA][NESW]':pl.Boolean,
        #'Dummy_SL_CDHS_J':pl.Categorical,
        #'Dummy_SL_ML_SJ':pl.Categorical,
    }

    # known as each card is played
    play_features = {
        #{cards_played:pl.Categorical, # todo:
        #{declarer_trick_count:pl.Categorical, # todo:
        #{defender_trick_count:pl.Categorical, # todo:
    }

    # known after play concludes
    board_results_features = {
        #'HandRecordBoardScore':pl.String,
        #'board_result_id':pl.Int64,
        'Result':pl.Int8,
        'Tricks':pl.UInt8,
        'Score':pl.Int16, # todo: deprecate in favor of Score_Declarer? Need to look at acbl, bridgewebs, ffbridge, pbn, etc.
        'Score_Declarer':pl.Int16,
        'Score_(NS|EW)':pl.Int16,
        # todo: rename Declarer_* to *_Declarer
        # no matches: 'Declarer_Score':pl.Int16,
        'HandRecordBoardScore':pl.Int16,
        # no matches: 'LoTT_Diff':pl.Int8,
        'Computed_Score_Declarer':pl.Int16,
        'Computed_Score_Declarer2':pl.Int16,
        'OverTricks':pl.Int8,
        'JustMade':pl.Int8,
        'UnderTricks':pl.Int8,
        'Defender_Par_GE':pl.Float32,
        'DD_Tricks_Diff':pl.Int8,
        'DD_Pct_Max_Diff_(NS|EW)':pl.Float32,
        'DD_EV_Pct_Max_Diff_(NS|EW)':pl.Float32,
        # no matches: 'EV_Max_Diff':pl.Float32,
        #'DD_Score_Diff':pl.Int16, # todo: implement
    }

    # known after all tables have played board
    # todo: rename MP_ to distinguish between matchpoint and masterpoints.
    # must have MP_ or Pct_
    matchpoint_features = {
        #'match_points_(ns|ew)':pl.Float32, # todo: rename, drop
        #'MatchPoints_(NS|EW)':pl.Float32, # todo: rename, drop
        #'ParScore_MPs':pl.Float32, # todo: missing, why?

        # DD columns
        'DD_Score_Pct_(NS|EW)':pl.Float32,
        # no matches: 'DD_Score_Pct_(NS|EW)_Max':pl.Float32,
        # Declarer columns
        # todo: rename Declarer_* to *_Declarer
        # no matches: 'Declarer_Score_DD_Diff':pl.Int16,
        # no matches: 'Declarer_EV_Score_Diff':pl.Float32,
        # no matches: 'Declarer_EV_Score_Max_Diff':pl.Float32,
        # no matches: 'Declarer_DD_Pct':pl.Float32,
        'Declarer_Pct':pl.Float32,
        # no matches: 'Declarer_Pct_DD_Diff':pl.Float32,
        # no matches: 'Declarer_ParScore_Pct':pl.Float32,
        # no matches: 'Declarer_ParScore_DD_Diff':pl.Int16,
        # no matches: 'Declarer_EV_Pct':pl.Float32,
        # no matches: 'Declarer_EV_Pct_Max':pl.Float32,
        # no matches: 'Declarer_EV_Pct_Max_Diff':pl.Float32,
        # no matches: 'Declarer_EV_ParScore_Pct_Diff':pl.Float32,
        # no matches: 'Declarer_EV_ParScore_Pct_Max_Diff':pl.Float32,
        # Elo columns
        'Elo_Delta_After':pl.Float32,
        'Elo_Delta_Before':pl.Float32,
        'Elo_R_[NESW]':pl.Float32,
        'Elo_R_(NS|EW)':pl.Float32,
        'Elo_R_[NESW]_Before':pl.Float32,
        'Elo_R_(NS|EW)_Before':pl.Float32,
        'Elo_R_Player_[NESW]_EventEnd':pl.Float32,
        'Elo_R_Pair_(NS|EW)_EventEnd':pl.Float32,
        # EV columns
        'EV_Pct_Max_(NS|EW)':pl.Float32,
        'EV_Pct_Max_Diff_(NS|EW)':pl.Float32,
        'EV_Score_Pct_(NS|EW)':pl.Float32,
        # no matches: 'EV_Par_Pct_Diff_(NS|EW)':pl.Float32,
        # no matches: 'EV_Par_Pct_Max_Diff_(NS|EW)':pl.Float32,
        'EV_Max_Diff_(NS|EW)':pl.Float32,
        # MP columns
        #'MatchPoints_(NS|EW)':pl.Float32,
        'MP_(NS|EW)':pl.Float32,
        'MP_DD_Score_[1-7][CDHSN]_([NESW]|NS|EW)':pl.Float32,
        'MP_EV_(NS|EW)_[NESW]_[CDHSN]_[1-7]_(NV|V)':pl.Float32,
        'MP_EV_Score_(NS|EW)':pl.Float32,
        'MP_Par_(NS|EW)':pl.Float32,
        #'MP_DD_Score_(NS|EW)':pl.Float32,
        # no matches: 'MP_DD_Score_Pct_(NS|EW)':pl.Float32,
        # no matches: 'MP_DD_Score_(NS|EW)_Max':pl.Float32,
        'MP_EV_Max_(NS|EW)':pl.Float32,
        'MP_Par_(NS|EW)':pl.Float32,
        # Par columns
        'Par_Pct_(NS|EW)':pl.Float32,
        'Par_Diff_(NS|EW)':pl.Float32,
        # Pct columns
        'Pct_(NS|EW)':pl.Float32,
    }

    # known after all tables have played board
    rank_features = {
        #'OverallRanking_(NS|EW)':pl.Float32, # todo: implement
        #'OverallRankingLimit_(NS|EW)':pl.Float32, # todo: implement
        #'OverallRank':pl.Float32, # includes both NS and EW # todo: implement
        #'SectionRank':pl.Float32, # includes both NS and EW # todo: implement
        #'SectionRanking_(NS|EW)':pl.Float32, # todo: implement
        #'SectionRankingLimit_(NS|EW)':pl.Float32, # todo: implement
    }

    # todo: Implement some naming scheme for game_state value. Create an enum in mlBridgeAi?
    # game_state value is used by models to determine which features to use for training. e.g. game_state in [range(5)] specifies pre-game knowable info + hand info +  deal (hand record info).
    features_d = {
        'board': {'game_state': features_enum.board_game_state, 'features':invariant_board_features},
        'deal': {'game_state': features_enum.deal_game_state, 'features':deal_features},
        'event': {'game_state': features_enum.event_game_state, 'features':event_features},
        'players': {'game_state': features_enum.players_game_state, 'features':players_features},
        'session': {'game_state': features_enum.session_game_state, 'features':session_section_features},
        'final_contract': {'game_state': features_enum.final_contract_game_state, 'features':final_contract_features},
        'opening_lead': {'game_state': features_enum.opening_lead_game_state, 'features':opening_lead_features},
        'dummy': {'game_state': features_enum.dummy_game_state, 'features':dummy_features},
        'play': {'game_state': features_enum.play_game_state, 'features':play_features},
        'board_results': {'game_state': features_enum.board_results_game_state, 'features':board_results_features},
        'matchpoint': {'game_state': features_enum.matchpoint_game_state, 'features':matchpoint_features},
        'rank': {'game_state': features_enum.rank_game_state, 'features':rank_features},
    }
    
    # takes 1s for 0 rows.
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)
    hrs0_df = pl.read_parquet(acbl_hand_records_augmented_file,n_rows=0)
    print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs0_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    assert hrs0_df.select(pl.col(r'^.*_(x|y|right|left)$')).width == 0, hrs0_df.select(pl.col(r'^.*_(x|y|right|left)$')).columns
    
        # takes 1s to read 0 rows x 1787 columns. 25GB file.
    acbl_board_results_augmented_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_augmented_file = acblPath.joinpath(acbl_board_results_augmented_filename)
    brs0_df = pl.read_parquet(acbl_board_results_augmented_file,n_rows=0)
    print(f"Loaded {acbl_board_results_augmented_filename}: shape:{brs0_df.shape} size:{acbl_board_results_augmented_file.stat().st_size}")

    # todo: remove these _right columns in previous step
    brs0_df = brs0_df.select(pl.exclude('hand_record_id_right', 'session_id_right', 'Board_right', 'section_name_right', 'iVul_right'))
    brs0_df = brs0_df.select(pl.exclude('MP_DD_Score_NS_right', 'MP_DD_Score_EW_right', 'MP_Par_NS_right', 'MP_Par_EW_right'))
    
    # takes 1s
    assert brs0_df.select(pl.col(r'^.*_(x|y|right|left)$')).width == 0, brs0_df.select(pl.col(r'^.*_(x|y|right|left)$')).columns

    intersected_columns = set(hrs0_df.columns).intersection(brs0_df.columns)
    print(f"Columns shared between hrs and brs: {len(intersected_columns)}")

    candidate_hrs0_df_columns = set(hrs0_df.columns)-intersected_columns
    print(f"Candidate hrs-only columns: {len(candidate_hrs0_df_columns)}")

    deal_features_in_hrs0 = list(set(col for pattern in deal_features for col in candidate_hrs0_df_columns if re.match('^'+pattern+'$', col)))
    deal_features_to_join = deal_features_in_hrs0 + ['HandRecordBoard','Dealer','Vul','hand_record_id','Board']
    print(f"Deal features to join from hrs: {len(deal_features_to_join)}")

    deal_features_in_brs0 = [col for pattern in deal_features for col in brs0_df.columns if re.match('^'+pattern+'$', col)]
    print(f"Deal features already in brs: {len(deal_features_in_brs0)}")

    # join brs_df and hrs_df schema (0 rows). Only want the intersection of both dataframes so using how='inner'.
    combined0_df = brs0_df.join(hrs0_df[deal_features_to_join],on=['HandRecordBoard','Dealer','Vul','hand_record_id','Board'],how='inner') # todo: add event_id after dtype reconcile
    print(f"Combined schema: {combined0_df.shape[1]} columns")

    assert combined0_df.select(pl.col(r'^.*_(x|y|right|left)$')).width == 0

    regex_not_matching_any_column = []
    columns_matching_regex = {}
    columns_matching_multiple_regex = defaultdict(list)
    matches_per_feature_type = defaultdict(int)
    for feature_type, features in features_d.items():
        for regex_feature,dtype in features['features'].items():
            feature_cols = [col for col in combined0_df.columns if re.match('^'+regex_feature+'$', col)]
            # feature_cols = brs_df.select(pl.col('^'+regex_feature+'$')).first().columns # select is hopelessly slow because it generates data. first() doesn't help.
            if len(feature_cols) == 0:
                regex_not_matching_any_column.append((regex_feature, feature_type))
                continue
            matches_per_feature_type[feature_type] += len(feature_cols)
            for feature in feature_cols:
                if feature in columns_matching_regex:
                    columns_matching_multiple_regex[feature].append((features['game_state'], regex_feature, feature_type))
                columns_matching_regex[feature] = (features['game_state'], dtype, regex_feature, feature_type)

    print(f"Columns matching a regex: {len(columns_matching_regex)}")
    for ft, cnt in matches_per_feature_type.items():
        print(f"  {ft}: {cnt} columns")

    conversions = 0
    for k, v in columns_matching_regex.items():
        if combined0_df[k].dtype != v[1]:
            combined0_df = combined0_df.with_columns(pl.col(k).cast(v[1]))
            conversions += 1
    print(f"  dtype conversions applied: {conversions}")
    
    # todo: probably should remove from brs_df and hrs_df directly instead of after join. requires splitting into two drop lists.
    # list of deprecated or no longer useful features.
    if club_or_tournament == 'club':
        shouldnt_exist_features = [
            'board_id',
            'result',
            'tricks_taken',
            #'session_id',
            #'section_name',
            'ns_pair_summary_id',
            'ew_pair_summary_id',
            'NSPair',
            'EWPair',
            'scores_l',
            'mp_limit',  # event MP ceiling; used by Elo strata, not ML features
        ]
    elif club_or_tournament == 'tournament':
        shouldnt_exist_features = [
            #'board_id',
            #'result',
            #'tricks_taken',
            #'session_id',
            #'section_name',
            #'ns_pair_summary_id',
            #'ew_pair_summary_id',
            'NSPair',
            'EWPair',
            'scores_l',
            # todo: revisit these. They are not in the tournament data but are in the club data?
            'board_result_id_ns',
            #'MP_NS',
            'board_result_id_ew',
            #'MP_EW',
            'event_name',
            'game_time',
            'game_type',
            'mp_limit',
            'mp_color',
            'mp_rating',
            'session_count',
            'movement_type',
            'scoring_type',
        ]
        # todo: remove
        shouldnt_exist_features_club_only = [
            # todo: in club but not in tournament
            'board_result_id',
            'Club',
            #'MasterPoints_N',
            #'MasterPoints_E',
            #'MasterPoints_S',
            #'MasterPoints_W',
            'session',
            'Round',
            'Table',
            'tb_count',
            'Lead',
            'HandRecordBoardScore',
            #'MatchPoints_NS',
            #'MatchPoints_EW',
        ]
    else:
        raise ValueError(f"Invalid club_or_tournament: {club_or_tournament}")
    assert [col for col in shouldnt_exist_features if col not in combined0_df.columns] == [], [col for col in shouldnt_exist_features if col not in combined0_df.columns]
    combined0_df = combined0_df.drop(shouldnt_exist_features, strict=False) # strict=False allows column names which don't exist.
    print(f"After dropping deprecated features: {combined0_df.shape[1]} columns")

    assert len(columns_matching_multiple_regex) == 0, f"Columns matching multiple regex: {columns_matching_multiple_regex}"

    columns_not_matching_any_regex = [col for col in combined0_df.columns if col not in columns_matching_regex.keys()]
    if columns_not_matching_any_regex:
        sample = columns_not_matching_any_regex[:20]
        msg = f"{len(columns_not_matching_any_regex)} columns not matching any regex (first 20): {sample}"
        assert len(columns_not_matching_any_regex) == 0, msg

    if regex_not_matching_any_column:
        print(f"Regex patterns not matching any column: {len(regex_not_matching_any_column)}")

    dtypes_to_exclude = [pl.Array(pl.Float32,shape=(14,)),pl.Array(pl.Int16,shape=(14,))] # exclude all pl.Array dtypes. todo: is there a simple way to exclude all arrays?
    array_cols = combined0_df.select(pl.col(dtypes_to_exclude)).columns
    if array_cols:
        print(f"Excluding {len(array_cols)} Array columns: {array_cols}")
    combined0_df = combined0_df.select(pl.exclude(dtypes_to_exclude))
    print(f"Final schema: {combined0_df.shape[1]} columns, {len(deal_features_to_join)} deal features to join")

    # takes 0s
    # todo: move game_status lists to mlBridgeAi? That way there's no need for the acbl_club_model_data_d.pkl file.
    acbl_club_model_data_d_filename = f"acbl_{club_or_tournament}_model_data_d.pkl"
    acbl_club_model_data_d_file = acblPath.joinpath(acbl_club_model_data_d_filename)
    with open(acbl_club_model_data_d_file, 'wb') as f:
        pickle.dump(columns_matching_regex,f)
    print(f"Saved {acbl_club_model_data_d_filename}: size:{acbl_club_model_data_d_file.stat().st_size}")

    # ============================================================
    # STREAMING SECTION (refactored 2026-04-20; was eager, OOM'd club)
    # ============================================================
    # Step A: load hrs deal-features eagerly -- 5 cols x 1.8M (tournament)
    # or 19M (club) rows is small enough (~1-2 GB) to broadcast as the
    # build side of a streaming hash-join.
    acbl_hand_records_augmented_filename = f'acbl_{club_or_tournament}_hand_records_augmented.parquet'
    acbl_hand_records_augmented_file = acblPath.joinpath(acbl_hand_records_augmented_filename)
    hrs_columns = deal_features_to_join
    hrs_df = pl.read_parquet(acbl_hand_records_augmented_file, columns=hrs_columns)
    print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")
    show_memory_usage('after loading hrs_df (eager small side)')

    # Step B: determine brs projection from schema (no data load).
    # Project out List/Array/Object/Struct columns AT SCAN TIME so they're
    # never even read off disk -- saves substantial I/O for club.
    acbl_board_results_augmented_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_augmented_file = acblPath.joinpath(acbl_board_results_augmented_filename)
    brs_schema_full = pl.read_parquet_schema(acbl_board_results_augmented_file)
    target_cols = set(brs_schema_full.keys()).intersection(combined0_df.columns)
    complex_cols = sorted([
        c for c in target_cols
        if isinstance(brs_schema_full[c], (pl.List, pl.Array, pl.Object, pl.Struct))
    ])
    if complex_cols:
        print(f"Excluding {len(complex_cols)} complex-dtype columns at scan: {complex_cols}")
    brs_read_cols = sorted(target_cols - set(complex_cols))
    print(f"brs streaming projection: {len(brs_read_cols)} columns "
          f"(of {len(brs_schema_full)} in source)")

    # Sanity: keys we'll join on must be in the projection.
    join_keys = ['HandRecordBoard', 'Dealer', 'Vul', 'hand_record_id', 'Board']
    missing_keys = [k for k in join_keys if k not in brs_read_cols]
    assert not missing_keys, f"Join keys missing from brs projection: {missing_keys}"

    # Step C: pre-flight on the full (unfiltered) plan to validate the schema
    # once. Cheap -- collect_schema() never materializes data.
    full_plan = _build_join_plan(
        pl.scan_parquet(acbl_board_results_augmented_file),
        hrs_df=hrs_df,
        brs_read_cols=brs_read_cols,
        join_keys=join_keys,
    )
    plan_schema = _validate_plan_schema(full_plan, label='full')
    print(f"Final lazy schema: {len(plan_schema)} columns")

    acbl_club_model_data_filename = f"acbl_{club_or_tournament}_model_data.parquet"
    acbl_club_model_data_file = acblPath.joinpath(acbl_club_model_data_filename)

    # Stale 0-byte canonical files block the post-write summary scan and are
    # always wrong (a previous run died mid-write). Clear them before we start
    # so the rest of the function can use file-existence as a success signal.
    if (
        acbl_club_model_data_file.exists()
        and acbl_club_model_data_file.stat().st_size == 0
    ):
        print(f"Removing stale 0-byte {acbl_club_model_data_filename}")
        acbl_club_model_data_file.unlink()

    if not chunk_years:
        # Legacy single-pass streaming sink. Known to OOM for club (60 GB
        # board-results parquet) -- 600 GB committed memory observed before
        # any output bytes. Kept behind --no-chunk-years for tournament
        # smoke tests where the smaller schema sometimes fits.
        show_memory_usage('before streaming sink')
        t_sink = time.time()
        _sink_parquet_safe(full_plan, acbl_club_model_data_file)
        sink_elapsed = time.time() - t_sink
        out_size = acbl_club_model_data_file.stat().st_size
        print(f"Streaming sink complete in {sink_elapsed:.1f}s; "
              f"output size {out_size / (1024**3):.2f} GB")
        show_memory_usage('after streaming sink')
    else:
        # Month-chunking path with eager collect + eager write_parquet.
        #
        # Why NOT sink_parquet, even with year chunks and engine='streaming':
        #   For a 6,772-column schema with a 5-column composite hash-join key
        #   (HandRecordBoard, Dealer, Vul, hand_record_id, Board), Polars'
        #   sink_parquet pre-allocates per-column row-group buffers AND a
        #   wide-keyed hash-join state, which together blew past 700 GB
        #   committed memory on a single year of club data (PIDs 10896,
        #   29500, 6740, 9588 -- all killed before producing output bytes).
        #   Year-chunking + sink_parquet did NOT help. The new v2 streaming
        #   engine (engine='streaming') did NOT help either.
        #
        # Why month chunks + eager:
        #   1 month of club ~= 700K rows. After projection+join that's
        #   ~700K * 6,772 cells ~= 38 GB uncompressed in memory at peak,
        #   well under the 189 GB box limit. Eager write_parquet writes
        #   from an already-materialized DataFrame and never has the
        #   per-column buffer pre-allocation pathology that sink_parquet
        #   does for wide schemas.
        #
        # Final merge across ~84 month shards still uses sink (via
        # _stream_concat_shards) because the merge plan is just
        # scan -> concat -> sink with NO joins -- the wide hash-table
        # pathology doesn't apply.
        date_range = (
            pl.scan_parquet(acbl_board_results_augmented_file)
            .select(
                pl.col('Date').min().alias('mn'),
                pl.col('Date').max().alias('mx'),
            )
            .collect()
        )
        d_min, d_max = date_range['mn'][0], date_range['mx'][0]
        src_y_min, src_y_max = d_min.year, d_max.year
        y_lo = start_year if start_year is not None else src_y_min
        y_hi = end_year if end_year is not None else src_y_max

        # Enumerate (year, month_start, months_in_chunk) windows covering
        # [y_lo-01-01 .. (y_hi+1)-01-01). Default months_per_chunk=1 means
        # one shard per calendar month.
        from datetime import date as _date
        from dateutil.relativedelta import relativedelta as _rd
        windows = []
        cur = _date(y_lo, 1, 1)
        end_excl = _date(y_hi + 1, 1, 1)
        while cur < end_excl:
            nxt = cur + _rd(months=months_per_chunk)
            if nxt > end_excl:
                nxt = end_excl
            windows.append((cur, nxt))
            cur = nxt
        print(
            f"Source Date range: [{d_min} .. {d_max}]; "
            f"processing years {y_lo}..{y_hi} inclusive in "
            f"{len(windows)} chunks of {months_per_chunk} month(s) each"
        )
        shard_dir = acblPath.joinpath(f'shards_{club_or_tournament}_model_data')
        shard_dir.mkdir(exist_ok=True)
        print(f"Shard dir: {shard_dir}")
        recorded_source_rows = _read_recorded_source_rows(shard_dir)
        print("Counting source rows per month (Date column only)...")
        t_stats = time.time()
        source_stats = _source_window_stats(acbl_board_results_augmented_file, windows)
        print(f"Source window stats in {time.time() - t_stats:.1f}s")
        source_by_file = {}

        shards = []
        for w_start, w_end in windows:
            # Naming convention: shard files keyed by inclusive start date
            # so resume across runs with different chunk sizes works only
            # if the chunk boundaries align. For mismatched chunk sizes the
            # validate-and-skip step below will simply re-create.
            shard_label = f'{w_start.isoformat()}_{w_end.isoformat()}'
            shard = shard_dir.joinpath(f'window={shard_label}.parquet')
            shards.append(shard)
            src_n, src_mx = source_stats[(w_start, w_end)]
            source_by_file[shard.name] = {
                'source_rows': src_n,
                'source_date_max': None if src_mx is None else src_mx.isoformat(),
            }
            if shard.exists() and _model_shard_is_current(
                shard, src_n, src_mx, recorded_source_rows.get(shard.name), shard_label,
            ):
                continue

            d_start_lit = pl.lit(w_start.isoformat()).str.to_date()
            d_end_lit = pl.lit(w_end.isoformat()).str.to_date()
            shard_plan = _build_join_plan(
                pl.scan_parquet(acbl_board_results_augmented_file).filter(
                    (pl.col('Date') >= d_start_lit) & (pl.col('Date') < d_end_lit),
                ),
                hrs_df=hrs_df,
                brs_read_cols=brs_read_cols,
                join_keys=join_keys,
            )
            print(f"[{shard_label}] window=[{w_start} .. {w_end}) "
                  f"collecting + writing shard (eager path)...")
            show_memory_usage(f'before {shard_label}')
            t_w = time.time()
            # engine='streaming' on collect tells Polars to use streaming
            # internally for the scan/filter/select/join pipeline, but
            # still hand back a single materialized DataFrame at the end.
            # For our chunk size (~700K rows post-filter) this fits in
            # ~38 GB peak RSS without invoking sink_parquet's pathological
            # per-column buffer allocation.
            df = shard_plan.collect(engine='streaming')
            n_rows_w = df.height
            print(f"[{shard_label}] collected: {n_rows_w:,} rows; writing parquet...")
            df.write_parquet(
                shard,
                compression='zstd',
                compression_level=3,
                row_group_size=SINK_ROW_GROUP_SIZE,
                data_page_size=SINK_DATA_PAGE_SIZE,
            )
            del df
            gc.collect()
            elapsed_w = time.time() - t_w
            size_w = shard.stat().st_size
            print(f"[{shard_label}] done in {elapsed_w:.1f}s; "
                  f"rows={n_rows_w:,}; size={size_w/(1024**3):.3f}GB")
            show_memory_usage(f'after {shard_label}')
            print()

        # Free the broadcast side BEFORE the merge so the merge plan only
        # has the shard scan + concat + sink in memory.
        del hrs_df
        gc.collect()

        # Manifest: lets consumers (acbl_prediction_data.py etc.) verify the
        # shard set is complete before scanning the glob. Row counts come
        # from parquet footers only (cheap), so this also covers shards that
        # were skip-resumed from an earlier run.
        _write_shard_manifest(
            shard_dir, shards, label=club_or_tournament, source_by_file=source_by_file,
        )

        if merge_shards:
            non_empty_shards = [s for s in shards if s.stat().st_size > 0]
            if not non_empty_shards:
                print("No non-empty shards produced; skipping merge.")
            else:
                _stream_concat_shards(
                    non_empty_shards,
                    acbl_club_model_data_file,
                    label=club_or_tournament,
                )
                if not keep_shards:
                    print(f"Deleting {len(shards)} shards (use --keep-shards to retain)")
                    for s in shards:
                        if s.exists():
                            s.unlink()
                    manifest = shard_dir.joinpath('manifest.json')
                    if manifest.exists():
                        manifest.unlink()
                    try:
                        shard_dir.rmdir()
                    except OSError:
                        pass  # non-empty (e.g. user added files) -- leave it
        else:
            print(f"Shards + manifest retained at {shard_dir} "
                  f"(default; pass --merge-shards for a single file)")

    # Defensive cleanup if not already done by the chunked path.
    try:
        del hrs_df
    except (NameError, UnboundLocalError):
        pass
    gc.collect()

    # Step E: cheap post-write summary (lazy, never materializes data).
    # With the default shard output, summarize from the manifest; only scan
    # the canonical merged file when --merge-shards actually produced one.
    shard_manifest = acblPath.joinpath(
        f'shards_{club_or_tournament}_model_data', 'manifest.json'
    )
    if (merge_shards or not chunk_years) and (
        acbl_club_model_data_file.exists()
        and acbl_club_model_data_file.stat().st_size > 0
    ):
        out_size = acbl_club_model_data_file.stat().st_size
        n_rows = pl.scan_parquet(acbl_club_model_data_file).select(pl.len()).collect()[0, 0]
        n_cols_out = len(pl.read_parquet_schema(acbl_club_model_data_file))
        print(f"Saved {acbl_club_model_data_filename}: shape:({n_rows:,}, {n_cols_out}), size:{out_size}")
    elif shard_manifest.exists():
        import json
        m = json.loads(shard_manifest.read_text(encoding='utf-8'))
        total_bytes = sum(e['bytes'] for e in m['shards'])
        print(f"Saved {m['shard_count']} shards in {shard_manifest.parent.name}: "
              f"rows={m['total_rows']:,}, size={total_bytes:,}")
    else:
        print(f"(no merged file or shard manifest produced; check shard dir)")

    return None  # streaming mode -- no materialized DataFrame to return


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create ACBL model data.")
    parser.add_argument("--club", action="store_true",
                        help="Process club data only")
    parser.add_argument("--tournament", action="store_true",
                        help="Process tournament data only")
    parser.add_argument("--chunk-years", dest="chunk_years",
                        action="store_true", default=True,
                        help="Process one calendar year at a time (default).")
    parser.add_argument("--no-chunk-years", dest="chunk_years",
                        action="store_false",
                        help="Disable year chunking; do a single streaming "
                             "sink. Known to OOM for club -- only safe for "
                             "tournament smoke tests.")
    parser.add_argument("--start-year", type=int, default=None,
                        help="First year (inclusive) to process. Default: source min.")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Last year (inclusive) to process. Default: source max.")
    parser.add_argument("--months-per-chunk", type=int, default=1,
                        help="Months per shard. Default 1 (~700K rows/chunk "
                             "for club, ~38 GB peak RSS). Increase to 3 or 12 "
                             "for tournament where rows/month are smaller.")
    parser.add_argument("--merge-shards", dest="merge_shards",
                        action="store_true", default=False,
                        help="After per-window sinks, concat shards into a "
                             "single *_model_data.parquet. Very slow for the "
                             "6,778-col schema (~12 h for club) and downstream "
                             "reads get slower, not faster -- only use if a "
                             "single file is required.")
    parser.add_argument("--no-merge-shards", dest="merge_shards",
                        action="store_false",
                        help="Skip the merge step; keep shards + manifest.json "
                             "for consumers to scan (default).")
    parser.add_argument("--keep-shards", action="store_true", default=False,
                        help="With --merge-shards, retain per-year shards "
                             "instead of deleting them after the merge.")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        modes = ["club", "tournament"]
    else:
        modes = []
        if args.club:
            modes.append("club")
        if args.tournament:
            modes.append("tournament")
    return modes, args


if __name__ == "__main__":

    # Defense-in-depth: ensure UTF-8 stdout/stderr even if PYTHONUTF8 isn't
    # set in the parent env. Prior crash (2026-04-19) was UnicodeEncodeError
    # on cp1252 default when printing a Polars DataFrame __repr__.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    modes, args = _parse_club_tournament_args()
    print("=" * 70)
    print(f"  modes         = {modes}")
    print(f"  chunk_years   = {args.chunk_years} "
          f"(start_year={args.start_year}, end_year={args.end_year}, "
          f"months_per_chunk={args.months_per_chunk})")
    print(f"  merge_shards  = {args.merge_shards} (keep_shards={args.keep_shards})")
    print("=" * 70)

    for club_or_tournament in modes:
        t = time.time()
        create_model_data(
            club_or_tournament,
            chunk_years=args.chunk_years,
            start_year=args.start_year,
            end_year=args.end_year,
            merge_shards=args.merge_shards,
            keep_shards=args.keep_shards,
            months_per_chunk=args.months_per_chunk,
        )
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)


