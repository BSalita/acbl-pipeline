#!/usr/bin/env python3
"""Create ACBL prediction data (train/test split).

Streaming, full-volume, fixed-temporal-cutoff version (refactored 2026-04).

Why streaming:
    Eager mode at full data scale (51.7M club / 14.8M tournament rows) needs
    ~2.8 TB RAM+pagefile (per the original file header comment). Streaming mode
    (Polars LazyFrame + sink_parquet, with pl.Enum casts from
    mlBridge.CATEGORICAL_SCHEMAS instead of pl.Categorical) keeps the per-batch
    intermediates bounded.

Wall-clock baselines (dev box: 192 GB RAM, ~40 cores, NVMe E:):
    tournament: ~30-60 min total
                (~16M rows -> 41 GB train + 1 GB test)
    club:       ~7.3 h total, of which ~3 h was a single anomalous year (2022)
                (55.4M train + 3.8M test rows -> 165 GB train + 10 GB test)
                See "KNOWN ISSUE (2026-04-21)" block below for the per-year
                breakdown and what we tried to root-cause it.
    Measured 2026-04-20 -> 2026-04-21 (logs/04_prediction_data_club.log),
    reading the single merged model_data parquet at 59.5M club rows.
    Estimate at current volume (69.4M club rows, 2026-08-16, scaling the
    non-anomalous ~0.26 ms/row sink rate): club ~5 h (+2.5 h if the 2022
    anomaly repeats), tournament ~0.5-1 h. Since 2026-08-16 the source is
    the monthly shard dir (resolve_model_data_source): per-year scans prune
    to ~12 shard files instead of scanning the whole ~86 GB merged file,
    which may shave another 10-30% off the scan side -- remeasure on next run.

Why fixed-date cutoff (was: quantile-based 90th percentile):
    Quantile-based splits can leak across seasons (a single hot-streak month in
    the train tail leaks information about the test head). Fixed-date is the
    standard temporal-generalization split. Default cutoff = 2026-01-01.

Why no row cap (was: hardcoded 2_000_000):
    The source parquets are sorted oldest-first, so n_rows=2M was silently
    training only on 2018-2020 data while discarding 2021-2026 entirely.
    Default is now no cap; use --max-rows for smoke testing.

Why pl.Enum (was: pl.Categorical):
    pl.Categorical needs a global string dictionary scan, which forces
    materialization and breaks streaming. pl.Enum has a fixed pre-declared
    category list, casts per-batch, and is streaming-safe. Schemas are pulled
    from mlBridge.CATEGORICAL_SCHEMAS (Dealer=4 vals, Declarer_Direction=5,
    Contract=421).

CLI:
    --club, --tournament         (default: both)
    --max-rows N                 (default: None = no cap)
    --output-suffix _v2          (default: '_v2'; preserves existing files)
    --test-cutoff YYYY-MM-DD     (default: 2026-01-01)
    --chunk-years / --no-chunk-years  (default: chunk; see below)
    --start-year / --end-year    (default: source min/max year, inclusive)
    --merge-shards / --no-merge-shards  (default: merge into final files)
    --keep-shards                (default: False; with merge, delete shards after)

Why chunk by year (the default):
    The lazy plan keeps ~6,000 wide columns through 6 chained left joins. Even
    in streaming mode, processing 50M rows Ã— 6k cols at once builds enormous
    per-batch state (the original eager-mode peak was 2.8 TB of pagefile).
    Filtering the source to one year at a time (~6M rows) cuts peak intermediate
    state ~8x and makes streaming actually fit. Per-year shards are written, then
    optionally stream-concat'd into the final two parquets. The concat pass is a
    clean pipeline (no joins/casts) and streams cheaply.

KNOWN ISSUE (2026-04-21): one full-club run completed in 7.3 hours, but
    the 2022 shard took 3 hours by itself for reasons that remain unexplained.
    Empirical per-year sink times on a clean full club run:
        2019/train:  1.2M rows /  4.46 GB shard /   137 s   (~25 KB/row)
        2020/train:  4.0M rows / 15.08 GB shard /  2168 s   (4 ms/row)
        2021/train:  3.5M rows / 13.46 GB shard /  1216 s   (~normal)
        2022/train:  7.9M rows / 30.68 GB shard / 10797 s   ANOMALY: ~5x slow
        2023/train: 11.6M rows / 35.25 GB shard /  2724 s   (~normal)
        2024/train: 13.1M rows / 31.77 GB shard /  2931 s   (~normal)
        2025/train: 14.1M rows / 34.45 GB shard /  2896 s   (~normal)
        2026/test:   3.8M rows /  9.98 GB shard /  1321 s   (~normal)
        train merge: 7 shards -> 164.98 GB / 2013 s
        test merge:  1 shard  ->   9.98 GB /   62 s
    Per-year sink rate (excl. 2022): ~4-5 ms/row consistently. The 2022 sink
    spent ~9000 s slower than that rate predicts. Peak pagefile was 76 GB
    during the 2022 sink, peak private commit was 443 GB during 2024 (paged
    out), but the system never OOMed and recovered between years
    (pagefile after 2024: 51 GB, after 2025: 52 GB).

    Things ruled out as causes:
      - Row-explosion in elo joins. Verified `(Date, session_id, Player_ID)`
        and `(Date, session_id, Pair_IDs)` have 0 duplicates in both
        acbl_club_player_elo_ratings.parquet and acbl_club_pair_elo_ratings.parquet.
      - Real per-year row imbalance. 2022 IS 2.3x bigger than 2021 (post-COVID
        club recovery), but 2023/2024/2025 are bigger still and finished
        normally.

    Open hypothesis for 2022's slowness: transient OS-level paging contention
    that resolved itself, NOT a deterministic per-year memory leak. A repeat
    run is the only way to confirm.

    Recommended workaround until root-caused: for one-shot full-club re-runs,
    process each year in a fresh Python process (resume logic skips written
    shards). Each fresh process starts with ~5 GB RSS / 2 GB pagefile, so
    even if the leak hypothesis is real it can never accumulate:
        python acbl_prediction_data.py --club --start-year 2019 --end-year 2019
        python acbl_prediction_data.py --club --start-year 2020 --end-year 2020
        ... etc, then a final unflagged invocation triggers the merge.

    Permanent fix (TODO -- see TODO.md "Per-year subprocess isolation"):
    wrap the year loop in subprocess spawns OR switch from sink_parquet to
    eager collect+write_parquet per year (the same pattern that fixed
    acbl_model_data.py --club). Lower priority now that one full run is
    known to complete unattended in ~7 hours.

    Tournament has ~16M total rows and finishes well inside the box's
    headroom; this issue does not apply.

Predict targets: Declarer_Direction, Contract, Pct_NS
"""

import argparse
import datetime as _dt
import gc
import pathlib
import pickle
import re
import sys
import time
from typing import Optional

import polars as pl
import psutil

_SRC_DIR = pathlib.Path(__file__).resolve().parent.parent
_MLBRIDGE = _SRC_DIR / 'mlBridge'
if not _MLBRIDGE.is_dir():
    raise FileNotFoundError(f'mlBridge not found at {_MLBRIDGE}')
for _p in (_SRC_DIR, _MLBRIDGE):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.append(_s)
import mlBridge  # noqa: E402

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')
savedModelsPath = acblPath.joinpath('SavedModels')


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def show_memory(label: str = '') -> None:
    proc = psutil.Process()
    mem = proc.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    tag = f' - {label}' if label else ''
    print(
        f"[Memory{tag}] Process RSS={mem.rss / (1024**3):.1f}GB | "
        f"System used={vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f}GB ({vm.percent}%) | "
        f"Pagefile used={swap.used / (1024**3):.1f}/{swap.total / (1024**3):.1f}GB ({swap.percent}%)"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ACBL prediction data (streaming).")
    parser.add_argument("--club", action="store_true", help="Process club data")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap input row count (default: no cap; for smoke testing)")
    parser.add_argument("--recent", action="store_true",
                        help="With --max-rows: take most-recent N rows from the tail "
                             "(source is sorted oldest-first). Without --recent, takes head (oldest).")
    parser.add_argument("--output-suffix", type=str, default="_v2",
                        help="Suffix for output filenames (default: '_v2'; pass '' to overwrite)")
    parser.add_argument("--test-cutoff", type=str, default="2026-01-01",
                        help="Test set start date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--chunk-years", dest="chunk_years", action="store_true", default=True,
                        help="Process one calendar year at a time (default; lower memory)")
    parser.add_argument("--no-chunk-years", dest="chunk_years", action="store_false",
                        help="Disable per-year chunking; build one big lazy plan instead")
    parser.add_argument("--start-year", type=int, default=None,
                        help="First calendar year to process (default: source min year)")
    parser.add_argument("--end-year", type=int, default=None,
                        help="Last calendar year to process, inclusive (default: source max year)")
    parser.add_argument("--merge-shards", dest="merge_shards", action="store_true", default=True,
                        help="Stream-concat per-year shards into final two parquets (default)")
    parser.add_argument("--no-merge-shards", dest="merge_shards", action="store_false",
                        help="Leave per-year shards on disk; do not produce merged final files")
    parser.add_argument("--keep-shards", action="store_true",
                        help="With --merge-shards, do not delete per-year shards after merging")
    args = parser.parse_args()

    if not args.club and not args.tournament:
        args.modes = ["club", "tournament"]
    else:
        args.modes = []
        if args.club:
            args.modes.append("club")
        if args.tournament:
            args.modes.append("tournament")

    args.test_cutoff_date = _dt.date.fromisoformat(args.test_cutoff)
    return args


# ----------------------------------------------------------------------------
# Column dropping (regex patterns; preserved verbatim from the original script)
# ----------------------------------------------------------------------------

DROP_PATTERNS = [
    r'^Pair_IDs_(NS|EW)$',
    r'^CT_[NESW]_[CDHSN]$',         # Contract type features
    r'^EV_.*Max_Col.*$',            # Expected value features
    r'^SL_[NESW]_ML_SJ$',           # cardinality issues; downstream schema mismatch
    r'^HandRecordBoard$',
    r'^Opponent_Pair_Direction$',
    r'^PBN$',
    r'^Player_ID_[NESW]$',
    r'^Player_Name_[NESW]$',
    r'^Vul$',
    r'^board_scoring_method$',
    r'^club_session$',
    r'^event_type$',
    r'^hand_record_id$',
    r'^sBoard$',
    r'^event_id$',                  # tournament only
    r'^section_id$',                # tournament only
    r'^session_id$',                # tournament only
    r'^SL_[NESW]_CDHS_SJ$',
    r'^SL_[NESW]_ML_I_SJ$',
    r'^SL_Max_(NS|EW)_Col$',
    r'^C_[EWNS][CDHS]',             # Card columns (e.g. C_ED4, C_ECA)
    r'^Suit_[NESW]_[CDHS]$',        # Suit distribution columns
    r'^Hand_[NESW]$',               # Hand string columns
    r'^(north|south|east|west)_(spades|hearts|diamonds|clubs)$',
    r'^start_(time|date)$',
]
_compiled_drop = [re.compile(p) for p in DROP_PATTERNS]

STRING_COLUMNS_KEEP = ['section_name']

# Columns that exist in model_data but cause downstream issues; dropped at read.
EXTRA_PROJECTION_DROPS = {
    'Pair_IDs_NS', 'Pair_IDs_EW', 'Pair_Names_EW', 'Pair_Names_NS',
    'SL_S_ML_I', 'SL_W_ML', 'SL_N_CDHS', 'SL_E_ML', 'SL_W_CDHS',
    'SL_S_ML', 'SL_W_ML_I', 'SL_N_ML_I', 'SL_E_CDHS', 'SL_S_CDHS',
    'SL_N_ML', 'SL_E_ML_I', 'ParContracts', 'Hands',
}


def resolve_model_data_source(
    acblPath: pathlib.Path,
    club_or_tournament: str,
) -> tuple[list[pathlib.Path], str, int]:
    """Locate the model-data source: shard directory (preferred) or merged file.

    acbl_model_data.py defaults to --no-merge-shards (2026-08-16): it leaves
    monthly shards + manifest.json in shards_{mode}_model_data/. Scanning the
    shard list is faster than a merged file for every downstream pattern
    (file-level Date pruning), so we prefer it whenever it exists.

    Returns (paths, description, total_bytes). paths is a list usable with
    pl.scan_parquet(). When manifest.json exists, the shard set is validated
    against it: missing shards or unlisted extras (e.g. leftovers from a run
    with a different --months-per-chunk) are hard errors, because both would
    silently drop or double-count rows.
    """
    import json

    shard_dir = acblPath.joinpath(f"shards_{club_or_tournament}_model_data")
    shard_files = (
        sorted(shard_dir.glob('window=*.parquet')) if shard_dir.is_dir() else []
    )
    if shard_files:
        manifest_path = shard_dir.joinpath('manifest.json')
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text(encoding='utf-8'))
            listed = [shard_dir.joinpath(e['file']) for e in m['shards']]
            missing = [p.name for p in listed if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{manifest_path} lists {len(missing)} missing shard(s), "
                    f"e.g. {missing[:3]} -- rerun acbl_model_data.py"
                )
            extras = sorted(
                {p.name for p in shard_files} - {p.name for p in listed}
            )
            if extras:
                raise RuntimeError(
                    f"{shard_dir} contains {len(extras)} shard(s) not in "
                    f"manifest.json, e.g. {extras[:3]} -- stale windows from a "
                    f"different chunk size? Delete them or rerun acbl_model_data.py"
                )
            shard_files = listed
        else:
            print(f"WARNING: {shard_dir} has no manifest.json (incomplete "
                  f"acbl_model_data.py run?); using glob of "
                  f"{len(shard_files)} shards unchecked")
        total = sum(p.stat().st_size for p in shard_files)
        desc = f"{shard_dir.name} ({len(shard_files)} shards)"
        return shard_files, desc, total

    merged = acblPath.joinpath(f"acbl_{club_or_tournament}_model_data.parquet")
    if not merged.exists():
        raise FileNotFoundError(
            f"Neither shard dir {shard_dir} nor merged file {merged} exists; "
            f"run acbl_model_data.py first"
        )
    return [merged], merged.name, merged.stat().st_size


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _build_joined_plan(
    base_lf: pl.LazyFrame,
    *,
    player_elo: pl.DataFrame,
    pair_elo: pl.DataFrame,
) -> pl.LazyFrame:
    """Apply the player/pair Elo joins, Pair_IDs derivation, section_name fill,
    Enum casts, and string-column drops to the given base LazyFrame.

    Pure function: no I/O, no global state besides mlBridge.CATEGORICAL_SCHEMAS
    and the module-level _compiled_drop / STRING_COLUMNS_KEEP."""
    lf = base_lf.with_columns([
        pl.col(f"Player_ID_{direction}").cast(pl.String, strict=False)
        for direction in "NESW"
    ])

    for direction in "NESW":
        rhs = (
            player_elo.lazy()
            .select([
                pl.col("Player_ID").cast(pl.String, strict=False),
                "session_id",
                "Elo_N",
                "Elo_R_EventStart",
            ])
            .rename({
                "Player_ID": f"Player_ID_{direction}",
                "Elo_N": f"Elo_N_{direction}",
                "Elo_R_EventStart": f"Elo_R_{direction}_EventStart",
            })
        )
        lf = lf.join(rhs, on=[f"Player_ID_{direction}", "session_id"], how="left")

    lf = lf.with_columns(
        Pair_IDs_NS=pl.concat_str([
            pl.min_horizontal(pl.col("Player_ID_N").cast(pl.Utf8), pl.col("Player_ID_S").cast(pl.Utf8)),
            pl.lit("-"),
            pl.max_horizontal(pl.col("Player_ID_N").cast(pl.Utf8), pl.col("Player_ID_S").cast(pl.Utf8)),
        ]),
        Pair_IDs_EW=pl.concat_str([
            pl.min_horizontal(pl.col("Player_ID_E").cast(pl.Utf8), pl.col("Player_ID_W").cast(pl.Utf8)),
            pl.lit("-"),
            pl.max_horizontal(pl.col("Player_ID_E").cast(pl.Utf8), pl.col("Player_ID_W").cast(pl.Utf8)),
        ]),
    )

    for direction in ["NS", "EW"]:
        rhs = (
            pair_elo.lazy()
            .select(["Pair_IDs", "session_id", "Elo_N", "Elo_R_EventStart"])
            .rename({
                "Pair_IDs": f"Pair_IDs_{direction}",
                "Elo_N": f"Elo_N_{direction}",
                "Elo_R_EventStart": f"Elo_R_{direction}_EventStart",
            })
        )
        lf = lf.join(rhs, on=[f"Pair_IDs_{direction}", "session_id"], how="left")

    plan_schema = lf.collect_schema()
    if 'section_name' not in plan_schema.names():
        lf = lf.with_columns(pl.lit(None).cast(pl.String).alias('section_name'))

    cat_columns = ['Contract', 'Declarer_Direction', 'Dealer']
    plan_schema = lf.collect_schema()
    if 'Round' in plan_schema.names() and 'Round' in mlBridge.CATEGORICAL_SCHEMAS:
        cat_columns.append('Round')
    cat_exprs = []
    for col in cat_columns:
        if col not in plan_schema.names():
            continue
        enum = mlBridge.CATEGORICAL_SCHEMAS.get(col)
        if enum is None:
            cat_exprs.append(pl.col(col).cast(pl.Categorical))
        else:
            cat_exprs.append(pl.col(col).cast(enum))
    if cat_exprs:
        lf = lf.with_columns(cat_exprs)

    plan_schema = lf.collect_schema()
    string_cols = [c for c, dt in plan_schema.items() if dt == pl.String]
    drop_cols = [
        c for c in string_cols
        if any(p.match(c) for p in _compiled_drop) and c not in STRING_COLUMNS_KEEP
    ]
    if drop_cols:
        lf = lf.drop(drop_cols)

    return lf


def _add_split_cols(lf: pl.LazyFrame, split: str) -> pl.LazyFrame:
    """Add is_train_set / is_val_set / is_test_set indicator columns and
    canonicalise Date to pl.Date."""
    return lf.with_columns(
        pl.col('Date').cast(pl.Date),
        pl.lit(split == 'train').alias('is_train_set'),
        pl.lit(False).alias('is_val_set'),
        pl.lit(split == 'test').alias('is_test_set'),
    )


def _stream_concat_shards(shards: list, out_path: pathlib.Path, label: str) -> None:
    """Stream-concat a list of parquet shards into a single output parquet.
    Pure pass-through pipeline (no joins/casts), so it streams cheaply."""
    if not shards:
        print(f"  No {label} shards to merge; skipping {out_path.name}")
        return
    print(f"\nMerging {len(shards)} {label} shards -> {out_path.name}")
    show_memory(f'before merge {label}')
    t = time.time()
    (
        pl.concat([pl.scan_parquet(s) for s in shards], how='vertical_relaxed')
        .sink_parquet(out_path, compression='zstd', compression_level=3, maintain_order=False)
    )
    print(f"  {label} merged in {time.time() - t:.1f}s; "
          f"size={out_path.stat().st_size / (1024**3):.2f}GB")
    show_memory(f'after merge {label}')


_TEXT_ID_COLUMNS = ("Player_ID_N", "Player_ID_E", "Player_ID_S", "Player_ID_W", "Declarer_ID")
_SHARD_SCHEMA_CACHE: dict[str, pl.Schema] = {}


def _shard_schema(path) -> pl.Schema:
    key = str(path)
    schema = _SHARD_SCHEMA_CACHE.get(key)
    if schema is None:
        schema = pl.scan_parquet(path).collect_schema()
        _SHARD_SCHEMA_CACHE[key] = schema
    return schema


def _scan_source_shards(src_paths, columns: list[str]) -> pl.LazyFrame:
    """Scan model-data shards that do not all share one schema.

    Newer months add columns such as Is_Sacrifice_Opportunity, and some of
    those files store player ids as integers. Group identical projections,
    cast the id columns to text, and fill a column the file does not have
    with null.
    """
    paths = [src_paths] if isinstance(src_paths, (str, pathlib.Path)) else list(src_paths)
    union: dict[str, pl.DataType] = {}
    groups: dict[tuple, list] = {}
    for path in paths:
        schema = _shard_schema(path)
        for name, dtype in schema.items():
            union.setdefault(name, dtype)
        signature = tuple(
            (name, str(schema[name]) if name in schema else None) for name in columns
        )
        groups.setdefault(signature, []).append((path, schema))

    frames = []
    for group in groups.values():
        schema = group[0][1]
        exprs = []
        for name in columns:
            if name not in schema:
                exprs.append(pl.lit(None).cast(union[name]).alias(name))
            elif name in _TEXT_ID_COLUMNS and schema[name] != pl.String:
                exprs.append(pl.col(name).cast(pl.String, strict=False).alias(name))
            else:
                exprs.append(pl.col(name))
        frames.append(
            pl.scan_parquet(
                [path for path, _schema in group],
                extra_columns="ignore",
                missing_columns="insert",
            ).select(exprs)
        )
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="vertical_relaxed")


def _prediction_source_counts(src_paths, test_cutoff):
    """One Date scan. Returns (date_min, date_max, {(year, is_test): row_count}).

    is_test matches the train/test windows in the year loop: a year entirely
    before test_cutoff is train, entirely on or after it is test, and a year
    that contains the cutoff is split into those two groups.
    """
    grouped = (
        _scan_source_shards(src_paths, ["Date"])
        .select(pl.col('Date').cast(pl.Date))
        .filter(pl.col('Date').is_not_null())
        .group_by(
            pl.col('Date').dt.year().alias('year'),
            (pl.col('Date') >= test_cutoff).alias('is_test'),
        )
        .agg(
            pl.len().alias('n'),
            pl.col('Date').min().alias('mn'),
            pl.col('Date').max().alias('mx'),
        )
        .collect()
    )
    if grouped.is_empty():
        raise RuntimeError("Source model data has no Date values")
    lookup = {}
    for row in grouped.iter_rows(named=True):
        lookup[(int(row['year']), bool(row['is_test']))] = int(row['n'])
    src_min = min(grouped['mn'].to_list())
    src_max = max(grouped['mx'].to_list())
    if isinstance(src_min, _dt.datetime):
        src_min = src_min.date()
    if isinstance(src_max, _dt.datetime):
        src_max = src_max.date()
    return src_min, src_max, lookup


def _parquet_num_rows(path) -> int:
    import pyarrow.parquet as pq
    return pq.ParquetFile(str(path)).metadata.num_rows


def prepare_prediction_data(
    club_or_tournament: str,
    *,
    max_rows: Optional[int] = None,
    recent: bool = False,
    output_suffix: str = "_v2",
    test_cutoff: _dt.date = _dt.date(2026, 1, 1),
    chunk_years: bool = True,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    merge_shards: bool = True,
    keep_shards: bool = False,
) -> None:
    t0 = time.time()
    print(f"\n{'=' * 70}")
    print(f"Processing {club_or_tournament} prediction data")
    print(f"  max_rows      = {max_rows} ({'most-recent (tail)' if recent else 'oldest (head)'})")
    print(f"  output_suffix = '{output_suffix}'")
    print(f"  test_cutoff   = {test_cutoff} (test = Date >= cutoff, train = Date < cutoff)")
    print(f"  chunk_years   = {chunk_years}"
          f"{f' (start_year={start_year}, end_year={end_year})' if chunk_years else ''}")
    print(f"  merge_shards  = {merge_shards} (keep_shards={keep_shards})")
    print(f"{'=' * 70}")
    show_memory('start')

    # --- 1. Load game-state column metadata --------------------------------
    gs_pkl = acblPath.joinpath(f"acbl_{club_or_tournament}_model_data_d.pkl")
    with open(gs_pkl, 'rb') as f:
        game_state_columns = pickle.load(f)
    print(f"Loaded {gs_pkl.name}: {len(game_state_columns)} columns")

    # game_state=5 = pre-board-results info (we want everything KNOWN before play)
    y_names = ['Declarer_Direction', 'Contract', 'Pct_NS']
    requested_game_state = list(range(5))
    read_cols = set(y_names + [
        k for k, v in game_state_columns.items()
        if v[0].value[0] in requested_game_state
    ])
    if 'board_result_id' in read_cols:
        read_cols.discard('board_result_id')
    read_cols -= EXTRA_PROJECTION_DROPS
    print(f"Game state levels: {requested_game_state}, read_cols: {len(read_cols)}")

    # --- 2. Validate read_cols against source schema (no data load) -------
    src_paths, src_desc, src_bytes = resolve_model_data_source(acblPath, club_or_tournament)
    src_schema = pl.scan_parquet(src_paths).collect_schema()
    print(f"Source: {src_desc} size={src_bytes / (1024**3):.2f}GB cols={len(src_schema)}")

    missing = read_cols - set(src_schema.keys())
    assert not missing, f"columns in read_cols but not in source: {sorted(missing)}"

    # Confirm there are no Float64 columns we need to handle (source should
    # already be Float32; if not, we'd need a lazy cast which we currently skip).
    f64_in_read = [c for c in read_cols if src_schema[c] == pl.Float64]
    assert not f64_in_read, f"Float64 columns in projection (would need lazy cast): {f64_in_read}"

    # --- 3. Load Elo lookup tables eagerly (small, ~200MB total) ----------
    player_elo_path = acblPath.joinpath(f"acbl_{club_or_tournament}_player_elo_ratings.parquet")
    player_elo = pl.read_parquet(player_elo_path)
    print(f"Loaded player_elo: shape={player_elo.shape} size={player_elo_path.stat().st_size / (1024**2):.1f}MB")

    pair_elo_path = acblPath.joinpath(f"acbl_{club_or_tournament}_pair_elo_ratings.parquet")
    pair_elo = pl.read_parquet(pair_elo_path)
    print(f"Loaded pair_elo:   shape={pair_elo.shape} size={pair_elo_path.stat().st_size / (1024**2):.1f}MB")

    show_memory('after lookups')

    # --- 4. Disk space check ----------------------------------------------
    import shutil, errno
    minimum_required_gb = 100
    free_gb = shutil.disk_usage(savedModelsPath.parent).free / (1024**3)
    print(f"Free space on {savedModelsPath.parent}: {free_gb:.2f} GB")
    if free_gb < minimum_required_gb:
        raise OSError(errno.ENOSPC, f"Not enough disk space: {free_gb:.1f}<{minimum_required_gb}")

    # --- 5. Schema introspection (build a 1-row plan once for diagnostics) -
    # Used to count final columns and verify no leftover String/List/Object/
    # Struct columns. Pure schema walk; no data materialised.
    diag_lf = _build_joined_plan(
        _scan_source_shards(src_paths, sorted(read_cols)).head(0),
        player_elo=player_elo, pair_elo=pair_elo,
    )
    diag_schema = diag_lf.collect_schema()
    leftover_strings = [
        c for c, dt in diag_schema.items()
        if dt == pl.String and c not in STRING_COLUMNS_KEEP
    ]
    assert not leftover_strings, f"Unexpected string columns remain: {leftover_strings}"
    bad_dtypes = [
        c for c, dt in diag_schema.items()
        if isinstance(dt, (pl.List, pl.Object, pl.Struct))
    ]
    assert not bad_dtypes, f"Unsupported dtype columns: {bad_dtypes}"
    n_cols_final = len(diag_schema) + 3  # +3 for is_train/is_val/is_test_set
    print(f"Final schema (per shard): {n_cols_final} columns "
          f"(joined+cast+dropped, plus 3 split flags)")

    # --- 6. Final output paths --------------------------------------------
    train_path = acblPath.joinpath(
        f"acbl_{club_or_tournament}_prediction_data_train{output_suffix}.parquet"
    )
    test_path = acblPath.joinpath(
        f"acbl_{club_or_tournament}_prediction_data_test{output_suffix}.parquet"
    )

    # ======================================================================
    # PATH A: Single-pass (no chunking). Kept as a fallback / sanity-check
    # path; this is what blows memory on full data.
    # ======================================================================
    if not chunk_years:
        print("\nBuilding single lazy plan (no year chunking)...")
        base = _scan_source_shards(src_paths, sorted(read_cols))
        if max_rows is not None:
            if recent:
                total_rows = _scan_source_shards(src_paths, ["Date"]).select(pl.len()).collect()[0, 0]
                offset = max(0, total_rows - max_rows)
                actual_take = min(max_rows, total_rows)
                base = base.slice(offset, actual_take)
                print(f"  Applied row cap: {actual_take:,} (most-recent of {total_rows:,}, offset={offset:,})")
            else:
                base = base.head(max_rows)
                print(f"  Applied row cap: {max_rows:,} (oldest)")

        joined = _build_joined_plan(base, player_elo=player_elo, pair_elo=pair_elo)
        cutoff_expr = pl.lit(test_cutoff).cast(pl.Date)
        train_lf = _add_split_cols(joined.filter(pl.col('Date').cast(pl.Date) < cutoff_expr), 'train')
        test_lf  = _add_split_cols(joined.filter(pl.col('Date').cast(pl.Date) >= cutoff_expr), 'test')

        print(f"\nStreaming TRAIN -> {train_path.name}")
        show_memory('before train sink')
        t_train = time.time()
        train_lf.sink_parquet(train_path, compression='zstd', compression_level=3, maintain_order=False)
        print(f"  TRAIN done in {time.time() - t_train:.1f}s; "
              f"size={train_path.stat().st_size / (1024**3):.2f}GB")
        show_memory('after train sink')
        gc.collect()

        print(f"\nStreaming TEST  -> {test_path.name}")
        show_memory('before test sink')
        t_test = time.time()
        test_lf.sink_parquet(test_path, compression='zstd', compression_level=3, maintain_order=False)
        print(f"  TEST done in {time.time() - t_test:.1f}s; "
              f"size={test_path.stat().st_size / (1024**3):.2f}GB")
        show_memory('after test sink')
        gc.collect()

    # ======================================================================
    # PATH B: Per-year chunking (default). One year at a time, write shards.
    # ======================================================================
    else:
        if max_rows is not None:
            print(f"  WARNING: --max-rows is ignored in chunk-years mode. "
                  f"Use --start-year / --end-year to limit the date range.")

        # Year range plus per-split row counts from one Date scan. The Elo
        # joins are left joins on unique keys, so a year shard's footer row
        # count equals this source count. A readable shard whose count still
        # matches is unchanged; a month that gained rows will not.
        print("Counting source rows per year/split (Date column only)...")
        t_stats = time.time()
        src_min, src_max, source_counts = _prediction_source_counts(src_paths, test_cutoff)
        print(f"Source window stats in {time.time() - t_stats:.1f}s")
        year_lo = max(start_year, src_min.year) if start_year is not None else src_min.year
        year_hi = min(end_year,   src_max.year) if end_year   is not None else src_max.year
        print(f"Source Date range: [{src_min} .. {src_max}]; "
              f"processing years {year_lo}..{year_hi} inclusive")

        shard_dir = acblPath.joinpath(f"shards_{club_or_tournament}{output_suffix}")
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"Shard dir: {shard_dir}")

        train_shards: list[pathlib.Path] = []
        test_shards:  list[pathlib.Path] = []

        for yr in range(year_lo, year_hi + 1):
            y_start = _dt.date(yr, 1, 1)
            y_end   = _dt.date(yr + 1, 1, 1)

            # Decide which split(s) this year contributes to
            if y_end <= test_cutoff:
                splits = [('train', y_start, y_end)]
            elif y_start >= test_cutoff:
                splits = [('test',  y_start, y_end)]
            else:
                splits = [('train', y_start, test_cutoff),
                          ('test',  test_cutoff, y_end)]

            for split, s_start, s_end in splits:
                shard = shard_dir.joinpath(f"{split}_year={yr}.parquet")
                src_n = source_counts.get((yr, split == 'test'), 0)
                if shard.exists():
                    # A readable parquet is not enough: the previous resume
                    # kept 2026 shards whose Date max was still 2026-08-12
                    # after the source had moved on. Rebuild unless the
                    # footer row count still equals the source window.
                    try:
                        pl.read_parquet_schema(shard)
                        shard_n = _parquet_num_rows(shard)
                    except Exception as e:
                        print(f"\n[{yr}/{split}] partial/invalid shard found "
                              f"({shard.name}: {e}); re-creating")
                        try:
                            shard.unlink()
                        except OSError:
                            pass
                    else:
                        if shard_n == src_n:
                            print(f"\n[{yr}/{split}] shard matches source "
                                  f"({src_n:,} rows), skipping: {shard.name}")
                            (train_shards if split == 'train' else test_shards).append(shard)
                            continue
                        print(f"\n[{yr}/{split}] shard has {shard_n:,} rows but "
                              f"source window has {src_n:,}; re-creating")
                        try:
                            shard.unlink()
                        except OSError:
                            pass

                print(f"\n[{yr}/{split}] window=[{s_start} .. {s_end}) sinking shard...")
                show_memory(f'before {yr}/{split}')
                t = time.time()

                base = (
                    _scan_source_shards(src_paths, sorted(read_cols))
                    .filter(
                        (pl.col('Date').cast(pl.Date) >= s_start)
                        & (pl.col('Date').cast(pl.Date) < s_end)
                    )
                )
                joined = _build_joined_plan(base, player_elo=player_elo, pair_elo=pair_elo)
                lf = _add_split_cols(joined, split)
                lf.sink_parquet(shard, compression='zstd', compression_level=3,
                                maintain_order=False)

                size_gb = shard.stat().st_size / (1024**3)
                print(f"[{yr}/{split}] done in {time.time() - t:.1f}s; size={size_gb:.2f}GB")
                show_memory(f'after {yr}/{split}')
                (train_shards if split == 'train' else test_shards).append(shard)
                gc.collect()

        if merge_shards:
            _stream_concat_shards(train_shards, train_path, 'train')
            _stream_concat_shards(test_shards,  test_path,  'test')
            if not keep_shards:
                print(f"\nDeleting {len(train_shards) + len(test_shards)} shards "
                      f"(use --keep-shards to retain)")
                for s in train_shards + test_shards:
                    try:
                        s.unlink()
                    except OSError as e:
                        print(f"  WARN: failed to delete {s.name}: {e}")
                try:
                    shard_dir.rmdir()
                except OSError:
                    pass  # not empty (e.g., other suffix runs); leave alone
        else:
            print(f"\n--no-merge-shards: leaving {len(train_shards)} train + "
                  f"{len(test_shards)} test shards in {shard_dir}")
            return  # skip post-write summary; shards are the deliverable

    # --- 10. Post-write summary (cheap: scan + collect just shape/dates) --
    # Either split may be empty depending on date range vs cutoff (e.g.,
    # --start-year/--end-year that lies entirely on one side of test_cutoff).
    def _summary(path: pathlib.Path):
        if not path.exists():
            return None
        return (
            pl.scan_parquet(path)
            .select(
                pl.len().alias('n_rows'),
                pl.col('Date').min().alias('date_min'),
                pl.col('Date').max().alias('date_max'),
            )
            .collect()
        )

    train_summary = _summary(train_path)
    test_summary = _summary(test_path)
    n_train = train_summary['n_rows'][0] if train_summary is not None else 0
    n_test  = test_summary['n_rows'][0]  if test_summary  is not None else 0
    total = n_train + n_test
    pct = (lambda n: (n / total * 100) if total else 0.0)
    print(f"\nSummary for {club_or_tournament}:")
    if train_summary is not None:
        print(f"  TRAIN: {n_train:,} rows ({pct(n_train):.2f}%) "
              f"dates [{train_summary['date_min'][0]} .. {train_summary['date_max'][0]}]")
    else:
        print(f"  TRAIN: (no file written)")
    if test_summary is not None:
        print(f"  TEST:  {n_test:,} rows ({pct(n_test):.2f}%) "
              f"dates [{test_summary['date_min'][0]} .. {test_summary['date_max'][0]}]")
    else:
        print(f"  TEST:  (no file written)")
    print(f"  TOTAL: {total:,} rows, {n_cols_final} columns")
    print(f"  Wall time: {time.time() - t0:.1f}s")
    print(f"{'-' * 70}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    args = _parse_args()
    mlBridge.pd_options_display()
    savedModelsPath.mkdir(parents=True, exist_ok=True)

    from mlBridge import print_started, print_ended
    prog_start = print_started()

    for mode in args.modes:
        prepare_prediction_data(
            mode,
            max_rows=args.max_rows,
            recent=args.recent,
            output_suffix=args.output_suffix,
            test_cutoff=args.test_cutoff_date,
            chunk_years=args.chunk_years,
            start_year=args.start_year,
            end_year=args.end_year,
            merge_shards=args.merge_shards,
            keep_shards=args.keep_shards,
        )

    print_ended(prog_start)
