#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_club_results_board_results_augment_step2.py

Takes 6h/70m for 106m->?/22m->15m rows and ?/6752 columns. filesize ?GB/16GB. 2TB memory/cache.

Final augmentation step for board results.
Joins with hand records and adds complete augmentation data.

Club results are computed per year and cached as one parquet shard per year in
acbl_club_board_results_augmented_years/. A year is recomputed only when its step1
rows, its joined hand records, the code or the input schemas change. The final
parquet is then assembled from the shards by copying their row groups byte for byte.
Use --rebuild-all to recompute every year.

Previous steps:
    acbl_sql_to_hand_records_clean.py
    acbl_hand_records_augment.py
    acbl_sql_to_board_results_clean.py
    acbl_board_results_augment_step1.py

Next steps:
    acbl_club_model_data.py

TODO:
1. why is 'section_name_right' appearing?
2. club MP_Top is tables-in-frame - 1, so boards with missing rows get inflated
   Pct_NS/Pct_EW. Derive it from ACBL's matchpoints instead (see RESULTS.md,
   2026-10-20 caveats).
"""

import hashlib
import inspect
import json
import os
import polars as pl
import pyarrow
import pyarrow.parquet as pq
import pathlib
import struct
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
from acbl_sql_to_board_results_clean import drop_out_of_range_pct

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')
recentPath = acblPath.joinpath('recent')

JOIN_KEYS = ['hand_record_id', 'session_id', 'Board']
ROW_GROUP_SIZE = 512_000
COMPRESSION = 'zstd'
CACHE_VERSION = 1


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


# ---------------------------------------------------------------------------
# Minimal thrift compact protocol codec, used to merge parquet footers.
# A struct is a list of [field_id, type, value]; lists are (elem_type, [values]).
# Re-encoding a parsed footer gives back the same bytes.
# ---------------------------------------------------------------------------

_T_TRUE, _T_FALSE, _T_BYTE, _T_I16, _T_I32, _T_I64, _T_DOUBLE, _T_BINARY, _T_LIST, _T_SET, _T_MAP, _T_STRUCT = range(1, 13)


class _ThriftReader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def byte(self) -> int:
        b = self.buf[self.pos]
        self.pos += 1
        return b

    def varint(self) -> int:
        shift = result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, t: int):
        if t in (_T_TRUE, _T_FALSE):
            # only reached for list elements; struct bools live in the type nibble
            return self.byte()
        if t == _T_BYTE:
            return self.byte()
        if t in (_T_I16, _T_I32, _T_I64):
            return self.zigzag()
        if t == _T_DOUBLE:
            v = self.buf[self.pos:self.pos + 8]
            self.pos += 8
            return v
        if t == _T_BINARY:
            n = self.varint()
            v = self.buf[self.pos:self.pos + n]
            self.pos += n
            return v
        if t in (_T_LIST, _T_SET):
            h = self.byte()
            n, et = h >> 4, h & 0x0F
            if n == 15:
                n = self.varint()
            return (et, [self.value(et) for _ in range(n)])
        if t == _T_MAP:
            n = self.varint()
            if n == 0:
                return (0, [])
            kv = self.byte()
            kt, vt = kv >> 4, kv & 0x0F
            return (kv, [(self.value(kt), self.value(vt)) for _ in range(n)])
        if t == _T_STRUCT:
            return self.struct()
        raise ValueError(f'unknown thrift type {t}')

    def struct(self) -> list:
        fields = []
        last = 0
        while True:
            h = self.byte()
            if h == 0:
                return fields
            t, delta = h & 0x0F, h >> 4
            fid = last + delta if delta else self.zigzag()
            last = fid
            if t in (_T_TRUE, _T_FALSE):
                fields.append([fid, t, t == _T_TRUE])
            else:
                fields.append([fid, t, self.value(t)])


class _ThriftWriter:
    def __init__(self):
        self.out = bytearray()

    def varint(self, n: int):
        while True:
            if n < 0x80:
                self.out.append(n)
                return
            self.out.append((n & 0x7F) | 0x80)
            n >>= 7

    def zigzag(self, n: int):
        self.varint((n << 1) ^ (n >> 63))

    def value(self, t: int, v):
        if t in (_T_TRUE, _T_FALSE, _T_BYTE):
            self.out.append(v)
        elif t in (_T_I16, _T_I32, _T_I64):
            self.zigzag(v)
        elif t == _T_DOUBLE:
            self.out += v
        elif t == _T_BINARY:
            self.varint(len(v))
            self.out += v
        elif t in (_T_LIST, _T_SET):
            et, items = v
            if len(items) < 15:
                self.out.append((len(items) << 4) | et)
            else:
                self.out.append(0xF0 | et)
                self.varint(len(items))
            for item in items:
                self.value(et, item)
        elif t == _T_MAP:
            kv, items = v
            self.varint(len(items))
            if items:
                self.out.append(kv)
                for k, val in items:
                    self.value(kv >> 4, k)
                    self.value(kv & 0x0F, val)
        elif t == _T_STRUCT:
            self.struct(v)
        else:
            raise ValueError(f'unknown thrift type {t}')

    def struct(self, fields: list):
        last = 0
        for fid, t, v in fields:
            wt = (_T_TRUE if v else _T_FALSE) if t in (_T_TRUE, _T_FALSE) else t
            if 0 < fid - last <= 15:
                self.out.append(((fid - last) << 4) | wt)
            else:
                self.out.append(wt)
                self.zigzag(fid)
            last = fid
            if t not in (_T_TRUE, _T_FALSE):
                self.value(t, v)
        self.out.append(0)


def _field(fields: list, fid: int):
    for f in fields:
        if f[0] == fid:
            return f
    return None


def _shift(fields: list, fid: int, delta: int):
    f = _field(fields, fid)
    if f is not None:
        f[2] += delta


def _read_footer(path: pathlib.Path):
    """Return (file size, footer start offset, parsed FileMetaData)."""
    size = path.stat().st_size
    with open(path, 'rb') as f:
        if f.read(4) != b'PAR1':
            raise ValueError(f'{path}: not a parquet file')
        f.seek(size - 8)
        tail = f.read(8)
        if tail[4:] != b'PAR1':
            raise ValueError(f'{path}: bad parquet footer (encrypted?)')
        footer_len = struct.unpack('<i', tail[:4])[0]
        start = size - 8 - footer_len
        f.seek(start)
        buf = f.read(footer_len)
    r = _ThriftReader(buf)
    meta = r.struct()
    if r.pos != footer_len:
        raise ValueError(f'{path}: footer parse stopped at {r.pos} of {footer_len}')
    w = _ThriftWriter()
    w.struct(meta)
    if bytes(w.out) != buf:
        raise ValueError(f'{path}: footer did not round trip')
    return size, start, meta


def concat_parquet_files(paths: list, out_file: pathlib.Path, chunk_bytes: int = 64 << 20) -> int:
    """Concatenate parquet files with identical schemas into one file without re-encoding.

    Row groups are copied byte for byte; only footer offsets are shifted. Files with page
    indexes, bloom filters or encryption are refused since those hold absolute offsets.
    Returns the total row count.
    """
    first_schema = first_kv = None
    row_groups = []
    total_rows = 0
    pos = 4
    plan = []
    for p in paths:
        size, start, meta = _read_footer(p)
        schema, kv = _field(meta, 2), _field(meta, 5)
        if first_schema is None:
            first_meta, first_schema, first_kv = meta, schema, kv
        elif schema != first_schema or kv != first_kv:
            raise ValueError(f'{p}: schema differs from {paths[0]}')
        if _field(meta, 8) is not None:
            raise ValueError(f'{p}: encrypted parquet not supported')
        delta = pos - 4
        rgs = _field(meta, 4)
        for rg in (rgs[2][1] if rgs else []):
            for cc in _field(rg, 1)[2][1]:
                for fid in (1, 4, 6, 8, 9):
                    if _field(cc, fid) is not None:
                        raise ValueError(f'{p}: column chunk field {fid} (path/page index/crypto) not supported')
                # Deprecated ColumnChunk.file_offset; Arrow writes 0, keep 0 as is.
                if _field(cc, 2) is not None and _field(cc, 2)[2]:
                    _shift(cc, 2, delta)
                md = _field(cc, 3)[2]
                if _field(md, 14) is not None:
                    raise ValueError(f'{p}: bloom filters not supported')
                for fid in (9, 10, 11):
                    _shift(md, fid, delta)
            _shift(rg, 5, delta)
            ordinal = _field(rg, 7)
            if ordinal is not None:
                ordinal[2] = len(row_groups)
            row_groups.append(rg)
        total_rows += _field(meta, 3)[2]
        plan.append((p, start))
        pos += start - 4

    meta = [list(f) for f in first_meta]
    _field(meta, 3)[2] = total_rows
    rg_field = _field(meta, 4)
    if rg_field is None:
        raise ValueError(f'{paths[0]}: no row_groups field')
    rg_field[2] = (_T_STRUCT, row_groups)
    w = _ThriftWriter()
    w.struct(meta)
    footer = bytes(w.out)

    with open(out_file, 'wb') as out:
        out.write(b'PAR1')
        for p, start in plan:
            with open(p, 'rb') as f:
                f.seek(4)
                left = start - 4
                while left:
                    buf = f.read(min(chunk_bytes, left))
                    if not buf:
                        raise IOError(f'{p}: short read')
                    out.write(buf)
                    left -= len(buf)
        out.write(footer)
        out.write(struct.pack('<i', len(footer)))
        out.write(b'PAR1')
    return total_rows


# ---------------------------------------------------------------------------
# Club per-year cache
# ---------------------------------------------------------------------------

def _sha(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        b = part if isinstance(part, (bytes, bytearray)) else str(part).encode('utf-8')
        h.update(len(b).to_bytes(8, 'little'))
        h.update(b)
    return h.hexdigest()


def _code_fingerprint() -> str:
    """Hash of the code that shapes a year's output, plus library versions."""
    import mlBridge.mlBridgeLib as mlBridgeLib
    import numpy
    parts = []
    for f in (__file__, mlBridgeAugmentLib.__file__, mlBridgeLib.__file__):
        p = pathlib.Path(f).resolve()
        parts += [p.name, p.read_bytes()]
    parts.append(inspect.getsource(drop_out_of_range_pct))
    parts += [pl.__version__, pyarrow.__version__, numpy.__version__]
    parts += [JOIN_KEYS, ROW_GROUP_SIZE, COMPRESSION, CACHE_VERSION]
    return _sha(*parts)


def _schema_str(schema) -> str:
    return repr([(name, str(dtype)) for name, dtype in schema.items()])


def _hash_series_bytes(s: pl.Series) -> bytes:
    return s.to_numpy().tobytes()


def _hrs_row_hashes(hrs_file: pathlib.Path, chunk_rows: int = 500_000) -> pl.DataFrame:
    """Join keys, file row number and a full-row hash for each hand record.

    Reads the file in row slices so the full table is never in memory. Eager hash_rows
    per slice is ~5x faster than a streamed struct hash on ~6k columns.
    """
    lf = pl.scan_parquet(hrs_file)
    n = lf.select(pl.len()).collect().item()
    parts = []
    for offset in range(0, n, chunk_rows):
        chunk = lf.slice(offset, chunk_rows).collect()
        parts.append(chunk.select(JOIN_KEYS).with_columns(chunk.hash_rows(seed=0).alias('_h')))
        del chunk
    if not parts:
        return pl.DataFrame(schema={'_rn': pl.UInt32, **{k: v for k, v in lf.collect_schema().items() if k in JOIN_KEYS}, '_h': pl.UInt64})
    return pl.concat(parts).with_row_index('_rn')


def _year_fingerprint(common: str, part_brs_df: pl.DataFrame, hrs_hashes: pl.DataFrame) -> str:
    keys = part_brs_df.select(JOIN_KEYS).unique()
    hrs_part = hrs_hashes.join(keys, on=JOIN_KEYS, how='semi').sort('_rn')
    return _sha(
        common,
        part_brs_df.height,
        _hash_series_bytes(part_brs_df.hash_rows(seed=0)),
        hrs_part.height,
        _hash_series_bytes(hrs_part['_rn']),
        _hash_series_bytes(hrs_part['_h']),
    )


def _write_json_atomic(obj, path: pathlib.Path):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def _load_manifest(path: pathlib.Path) -> dict:
    try:
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('version') == CACHE_VERSION:
            m.setdefault('years', {})
            return m
    except (OSError, ValueError):
        pass
    return {'version': CACHE_VERSION, 'years': {}}


def _num_rows(path: pathlib.Path) -> int:
    # Explicit handle: on Windows pyarrow can keep a path open, which blocks a later rename.
    with open(path, 'rb') as f:
        return pq.ParquetFile(f).metadata.num_rows


def _shard_ok(entry: dict, cache_dir: pathlib.Path, fingerprint: str) -> bool:
    if not entry or entry.get('fingerprint') != fingerprint:
        return False
    if entry.get('rows') == 0 and entry.get('file') is None:
        return True
    try:
        p = cache_dir / entry['file']
        return _num_rows(p) == entry['rows'] and p.stat().st_size == entry['size']
    except (OSError, KeyError, TypeError, ValueError, pyarrow.ArrowException):
        return False


def _augment_club_year(part_brs_df: pl.DataFrame, hrs_df: pl.DataFrame, y) -> pl.DataFrame:
    """Same steps as before the cache was added; returns None if the join is empty."""
    part_brs_df = part_brs_df.join(hrs_df, left_on=JOIN_KEYS, right_on=JOIN_KEYS, how='inner')
    print(part_brs_df)
    #assert part_brs_df.select('^.*_right$').is_empty(), part_brs_df.select('^.*_right$')
    if part_brs_df.is_empty():
        return None
    part_brs_df = mlBridgeAugmentLib.AllBoardResultsAugmentations(part_brs_df).perform_all_board_results_augmentations()
    non_object_columns = [col for col, dtype in zip(part_brs_df.columns, part_brs_df.dtypes) if dtype != pl.Object]
    part_brs_df = part_brs_df.select(non_object_columns)
    part_brs_df = drop_out_of_range_pct(part_brs_df, f"club year {y}")
    _validate_percentage_columns(part_brs_df, f"club year {y}")
    print(part_brs_df)
    return part_brs_df


def _write_shard(df: pl.DataFrame, path: pathlib.Path):
    """Write with the final file's settings; temp file + rename so a crash leaves no half shard."""
    tmp = path.with_name(path.name + '.tmp')
    arrow_table = df.to_arrow()
    writer = pq.ParquetWriter(tmp, arrow_table.schema, compression=COMPRESSION)
    try:
        writer.write_table(arrow_table, row_group_size=ROW_GROUP_SIZE)
    finally:
        writer.close()
    os.replace(tmp, path)


def _augment_club_by_year(brs_df: pl.DataFrame, hrs_file: pathlib.Path, out_file: pathlib.Path, rebuild_all: bool):
    cache_dir = out_file.with_name(out_file.stem + '_years')
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = cache_dir / 'manifest.json'
    manifest = _load_manifest(manifest_file)

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

    t = time.time()
    hrs_schema = pl.read_parquet_schema(hrs_file)
    common = _sha(_code_fingerprint(), _schema_str(brs_df.schema), _schema_str(hrs_schema))
    hrs_hashes = _hrs_row_hashes(hrs_file)
    print(f"hashed {len(hrs_hashes)} hand records in {time.time()-t:.1f}s")

    # Fingerprint every year and decide which ones to recompute.
    plan = []
    for y in years:
        t = time.time()
        part_brs_df = brs_df.filter(pl.col('Date').dt.year() == y)
        fp = _year_fingerprint(common, part_brs_df, hrs_hashes)
        skip = not rebuild_all and _shard_ok(manifest['years'].get(str(y)), cache_dir, fp)
        plan.append((y, fp, skip))
        print(f"year {y}: {'unchanged, reuse shard' if skip else 'changed, recompute'} (fingerprint {time.time()-t:.1f}s)")
    del hrs_hashes

    hrs_df = None
    if not all(skip for _, _, skip in plan):
        hrs_df = pl.read_parquet(hrs_file)
        print(f"Loaded {hrs_file.name}: shape:{hrs_df.shape}, size:{hrs_file.stat().st_size}")
        print(hrs_df)
        print(set(brs_df.columns).intersection(set(hrs_df.columns)))

    for y, fp, skip in plan:
        if skip:
            print(f"year {y}: skipped, {manifest['years'][str(y)]['rows']} rows from cache")
            continue
        t = time.time()
        part_brs_df = brs_df.filter(pl.col('Date').dt.year() == y) # or group_by()?
        print(f"\nProcessing year {y}: part_brs_df shape:{part_brs_df.shape}")
        part_brs_df = _augment_club_year(part_brs_df, hrs_df, y)
        shard_name = f'year_{y}.parquet'
        if part_brs_df is None:
            print(f"No data for year {y}")
            (cache_dir / shard_name).unlink(missing_ok=True)
            entry = {'fingerprint': fp, 'rows': 0, 'file': None}
        else:
            _write_shard(part_brs_df, cache_dir / shard_name)
            entry = {'fingerprint': fp, 'rows': len(part_brs_df), 'file': shard_name,
                     'size': (cache_dir / shard_name).stat().st_size, 'columns': len(part_brs_df.columns)}
        entry['seconds'] = round(time.time() - t, 1)
        manifest['years'][str(y)] = entry
        _write_json_atomic(manifest, manifest_file)
        print(f"year {y}: recomputed, {entry['rows']} rows, elapsed time in seconds: {time.time()-t}")
        part_brs_df = None
    del hrs_df

    # Forget years that no longer exist in step1.
    for key in [k for k in manifest['years'] if int(k) not in years]:
        stale = manifest['years'].pop(key)
        if stale.get('file'):
            (cache_dir / stale['file']).unlink(missing_ok=True)
        print(f"year {key}: no longer in step1, dropped from cache")
    _write_json_atomic(manifest, manifest_file)

    shards = [cache_dir / manifest['years'][str(y)]['file'] for y, _, _ in plan if manifest['years'][str(y)]['file']]
    if not shards:
        return

    # Columns must not change across years.
    prev_columns = None
    for p in shards:
        with open(p, 'rb') as f:
            columns = pq.ParquetFile(f).schema_arrow.names
        if prev_columns is None:
            prev_columns = columns
        assert columns == prev_columns, f"columns changed from {prev_columns} to {columns} in {p.name}"

    # Assemble the final file, unless it already matches these shards.
    assembled_fp = _sha(*[f'{y}:{fp}' for y, fp, _ in plan])
    prev = manifest.get('assembled') or {}
    try:
        st = out_file.stat()
        up_to_date = (prev.get('fingerprint') == assembled_fp and prev.get('size') == st.st_size
                      and prev.get('mtime_ns') == st.st_mtime_ns)
    except OSError:
        up_to_date = False
    if up_to_date and not rebuild_all:
        print(f"{out_file.name} already matches the year shards, not rewritten")
        return

    t = time.time()
    tmp = out_file.with_name(out_file.name + '.tmp')
    try:
        total_rows = concat_parquet_files(shards, tmp)
        # Cheap read check: decode one column across all row groups.
        with open(tmp, 'rb') as f:
            assert pq.ParquetFile(f).read(columns=[prev_columns[0]]).num_rows == total_rows
        os.replace(tmp, out_file)
    finally:
        tmp.unlink(missing_ok=True)
    st = out_file.stat()
    manifest['assembled'] = {'fingerprint': assembled_fp, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    _write_json_atomic(manifest, manifest_file)
    print(f"Saved {out_file.name}: {total_rows} total rows, size:{st.st_size}, assembled from {len(shards)} shards in {time.time()-t:.1f}s")


def augment_board_results_step2(club_or_tournament, data_path=None, recent_path=None, rebuild_all=False):
    """Perform final augmentation on board results by joining with hand records."""
    print(f"Processing {club_or_tournament} board results augmentation step 2...")
    data_path = acblPath if data_path is None else pathlib.Path(data_path)
    recent_path = recentPath if recent_path is None else pathlib.Path(recent_path)

    # Load step 1 augmented board results
    acbl_board_results_step1_filename = f'acbl_{club_or_tournament}_board_results_augmented_step1.parquet'
    acbl_board_results_step1_file = data_path.joinpath(acbl_board_results_step1_filename)
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
    acbl_hand_records_augmented_file = data_path.joinpath(acbl_hand_records_augmented_filename)

    # Not implemented:To make joining with brs_df tractable, only load columns which are needed by board results augmentations.
    # filter_cols = ['HandRecordBoard', 'Dealer', 'Vul', 'PBN']
    # filter_cols += ['^Vul_(NS|EW)$', '^SL(_Max)?_(NS|EW)(_[SHDC])?$', '^DD_([NESW]|NS|EW)_[SHDCN]$', 'DD_Score_[1-7][CDHSN]_([NESW]|NS|EW)$', '^Par_(NS|EW)$']
    # filter_cols += [r'^Probs_(NS|EW)_[NESW]_[SHDCN]_\d+$', '^EV_(NS|EW)(_[NESW])?(_[SHDCN])?(_[1-7])?_(NV|V)(_Max)?(_Col)?$']
    # cols, all_cols = get_parquet_columns_filtered(acbl_hand_records_augmented_file, filter_cols)

    # print(f"Reading {len(cols)} of {len(all_cols)} columns.")
    # hrs_df = pl.read_parquet(acbl_hand_records_augmented_file, columns=cols)
    # print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")

    acbl_board_results_augmented_filename = f'acbl_{club_or_tournament}_board_results_augmented.parquet'
    acbl_board_results_augmented_file = data_path.joinpath(acbl_board_results_augmented_filename)

    if club_or_tournament == 'club':
        #hrs_df = hrs_df.with_columns(pl.col('hand_record_id').cast(pl.Int64)) # todo: temp until previous step is corrected. issue is 'SHUFFLE' in hand_record_id for session.
        # Hand records are loaded inside, and only when some year must be recomputed.
        _augment_club_by_year(brs_df, acbl_hand_records_augmented_file, acbl_board_results_augmented_file, rebuild_all)
    elif club_or_tournament == 'tournament':
        hrs_df = pl.read_parquet(acbl_hand_records_augmented_file)
        print(f"Loaded {acbl_hand_records_augmented_filename}: shape:{hrs_df.shape}, size:{acbl_hand_records_augmented_file.stat().st_size}")
        print(hrs_df)

        print(set(brs_df.columns).intersection(set(hrs_df.columns)))

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
            recent_path,
            augmented_max_date(acbl_board_results_augmented_file),
        )
        print(f"Dropped {removed} recent rows now covered by the augmented parquet")

    return


def _parse_club_tournament_args():
    import argparse
    parser = argparse.ArgumentParser(description="Augment ACBL board results (step 2).")
    parser.add_argument("--club", action="store_true", help="Process club data only")
    parser.add_argument("--tournament", action="store_true", help="Process tournament data only")
    parser.add_argument("--rebuild-all", action="store_true", help="Club: recompute every year, ignoring the year cache")
    args = parser.parse_args()
    if not args.club and not args.tournament:
        return ["club", "tournament"], args.rebuild_all
    modes = []
    if args.club:
        modes.append("club")
    if args.tournament:
        modes.append("tournament")
    return modes, args.rebuild_all


if __name__ == "__main__":

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    modes, rebuild_all = _parse_club_tournament_args()
    for club_or_tournament in modes:
        t = time.time()
        augment_board_results_step2(club_or_tournament, rebuild_all=rebuild_all)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)
