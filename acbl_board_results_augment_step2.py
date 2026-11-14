#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acbl_club_results_board_results_augment_step2.py

Takes 6h/70m for 106m->?/22m->15m rows and ?/6752 columns. filesize ?GB/16GB. 2TB memory/cache.

Final augmentation step for board results.
Joins with hand records and adds complete augmentation data.

Results are computed in blocks of sessions and cached as one parquet shard per block in
acbl_{club,tournament}_board_results_augmented_blocks/. A block is recomputed only when
its step1 rows, its hand records, the code or the input schemas change. Use
--rebuild-all to recompute every block.

Club: a block is SESSION_BLOCK session ids. The hand records file is in session_id
order (step 2a sorts it, step 3a keeps the order), so each block reads only its own
hand records and the whole file is never in memory.

Tournament: session ids are strings such as 1502054-2005-1 (sanction, event, session),
so a block is the sessions sharing a prefix (TOURNAMENT_BLOCK_KEY): one month of
sanctions, or one year of NABCs. All sessions of a tournament are in one block. The
hand records (about 2M rows) are not in session order in the file; they are read once
and sorted in memory.

Every grouped calculation in the board results augmentations stays inside one session,
except the three player ratings (RATING_COLS), which average over all rows. Those are
left out of the shards and computed over all blocks when the final file is assembled.
The final parquet copies the shards' row groups byte for byte and appends the rating
columns to each row group.

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

import base64
import copy
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
CACHE_VERSION = 2
# Session ids per cached block. Fixed-width blocks keep their boundaries when new
# sessions arrive, so only the newest block changes on a routine run.
SESSION_BLOCK = 25_000
# Tournament block key: the leading part of the session id. Sanction ids start with
# YYMM (1502054-...), NABC ids with letters and a two-digit year (NABC163-...). Ids that
# match neither fall back to their first four characters. A month's block changes only
# when that month gets new sessions.
TOURNAMENT_BLOCK_KEY = r'^(\d{4}|[A-Z]+\d{2})'
# Row order inside a block, and so in the final file. Columns a side does not have are skipped.
BLOCK_SORT_KEYS = ['session_id', 'section_id', 'Board', 'Round', 'Table', 'board_result_id']
# Averages over all of a player's rows (mlBridgeAugmentLib.add_player_ratings), so they
# cannot be computed block by block. RATING_INPUT_COLS is what the averages are made from.
RATING_COLS = ['Declarer_Rating', 'Defender_OnLead_Rating', 'Defender_NotOnLead_Rating']
RATING_INPUT_COLS = ['Declarer_ID', 'OnLead', 'NotOnLead', 'DD_Tricks_Diff', 'Defender_Par_GE']


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
    cols = []
    for col in PCT_VALIDATION_COLS:
        if col not in df.columns:
            print(f"  WARNING ({label}): validation column '{col}' not found -- skipping")
        else:
            cols.append(col)
    if not cols:
        return
    # Counted in one pass over these columns. Filtering the whole ~6,800-column frame once
    # per column took about 2 minutes on a 2.3M-row block.
    counts = df.select([(~pl.col(c).is_between(0, 1) & pl.col(c).is_not_null()).sum().alias(c) for c in cols]).row(0)
    for col, n in zip(cols, counts):
        if n > 0:
            print(f"  WARNING ({label}): {n} rows have '{col}' outside [0,1]")


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


def _row_groups(meta: list) -> list:
    rgs = _field(meta, 4)
    return rgs[2][1] if rgs else []


def _first_page_offset(md: list) -> int:
    """Where a column chunk starts: its dictionary page when it has one, else its first data page."""
    dictionary = _field(md, 11)
    if dictionary is not None and dictionary[2] > 0:
        return dictionary[2]
    return _field(md, 9)[2]


def _row_group_spans(meta: list, footer_start: int, p) -> list:
    """Byte span (start, end) of every row group: from its first column chunk to the end of
    its last. Row groups must follow one another without overlapping. Anything stored after
    the last row group (a page index) lies outside every span and is not copied."""
    spans = []
    for rg in _row_groups(meta):
        chunks = []
        for cc in _field(rg, 1)[2][1]:
            md = _field(cc, 3)[2]
            first = _first_page_offset(md)
            chunks.append((first, first + _field(md, 7)[2]))
        start, end = min(c[0] for c in chunks), max(c[1] for c in chunks)
        if start < (spans[-1][1] if spans else 4) or end > footer_start:
            raise ValueError(f'{p}: row groups overlap or run into the footer')
        spans.append((start, end))
    return spans


def _relocate_row_group(rg: list, delta: int, p):
    """Shift every absolute offset in a row group's metadata by delta bytes.

    The page index (ColumnChunk fields 4-7) is dropped: it holds absolute page offsets in a
    part of the file that is not copied. Polars writes one; readers work without it.
    """
    for cc in _field(rg, 1)[2][1]:
        cc[:] = [f for f in cc if f[0] not in (4, 5, 6, 7)]
        for fid in (1, 8, 9):
            if _field(cc, fid) is not None:
                raise ValueError(f'{p}: column chunk field {fid} (path/crypto) not supported')
        # Deprecated ColumnChunk.file_offset; Arrow writes 0, keep 0 as is.
        if _field(cc, 2) is not None and _field(cc, 2)[2]:
            _shift(cc, 2, delta)
        md = _field(cc, 3)[2]
        if _field(md, 14) is not None:
            raise ValueError(f'{p}: bloom filters not supported')
        for fid in (9, 10, 11):
            f = _field(md, fid)
            if f is not None and f[2] > 0:  # 0 means "no such page"
                f[2] += delta
    _shift(rg, 5, delta)


def _arrow_schema_entry(meta: list):
    """The key/value struct holding pyarrow's own copy of the schema, or None."""
    kv = _field(meta, 5)
    for entry in (kv[2][1] if kv else []):
        if bytes(_field(entry, 1)[2]) == b'ARROW:schema':
            return entry
    return None


def _decode_arrow_schema(entry: list) -> pyarrow.Schema:
    return pyarrow.ipc.read_schema(pyarrow.BufferReader(base64.b64decode(bytes(_field(entry, 2)[2]))))


def _append_side_schema(meta: list, side_meta: list, side_path):
    """Add the side file's columns to the end of meta's schema. Side columns must be flat."""
    schema, side = _field(meta, 2)[2][1], _field(side_meta, 2)[2][1]
    side_children = _field(side[0], 5)[2]
    if len(side) - 1 != side_children:
        raise ValueError(f'{side_path}: nested columns not supported in a side file')
    names = {bytes(_field(e, 4)[2]) for e in schema[1:]}
    for e in side[1:]:
        if bytes(_field(e, 4)[2]) in names:
            raise ValueError(f'{side_path}: column {bytes(_field(e, 4)[2]).decode()} is already in the main file')
    _field(schema[0], 5)[2] += side_children
    schema.extend(side[1:])
    # One ColumnOrder per leaf column.
    orders, side_orders = _field(meta, 7), _field(side_meta, 7)
    if (orders is None) != (side_orders is None):
        raise ValueError(f'{side_path}: column_orders present in only one of the files')
    if orders is not None:
        orders[2][1].extend(side_orders[2][1])
    # pyarrow stores its schema a second time; readers that use it must see the new columns.
    entry, side_entry = _arrow_schema_entry(meta), _arrow_schema_entry(side_meta)
    if entry is not None:
        if side_entry is None:
            raise ValueError(f'{side_path}: no ARROW:schema entry')
        main_schema, side_schema = _decode_arrow_schema(entry), _decode_arrow_schema(side_entry)
        merged = pyarrow.schema(list(main_schema) + list(side_schema), metadata=main_schema.metadata)
        _field(entry, 2)[2] = base64.b64encode(merged.serialize().to_pybytes())


def concat_parquet_files(paths: list, out_file: pathlib.Path, side_paths: list = None, chunk_bytes: int = 64 << 20) -> int:
    """Concatenate parquet files with identical schemas into one file without re-encoding.

    Row groups are copied byte for byte; only footer offsets are shifted. Files with bloom
    filters or encryption are refused since those hold absolute offsets. A page index is
    left out of the result.

    side_paths gives one more parquet file per input file, holding extra columns for the
    same rows in the same row groups. Each side row group is copied right after its main
    row group and its columns are appended to that row group, so the output has the main
    columns followed by the side columns.
    Returns the total row count.
    """
    if side_paths is not None and len(side_paths) != len(paths):
        raise ValueError(f'{len(side_paths)} side files for {len(paths)} files')
    first_meta = first_schema = first_kv = None
    first_side_meta = first_side_schema = first_side_kv = None
    row_groups = []
    segments = []  # [path, offset, length], merged when contiguous
    total_rows = 0
    pos = 4

    def take(p, rg, span):
        nonlocal pos
        start, end = span
        _relocate_row_group(rg, pos - start, p)
        if segments and segments[-1][0] == p and segments[-1][1] + segments[-1][2] == start:
            segments[-1][2] += end - start
        else:
            segments.append([p, start, end - start])
        pos += end - start

    for i, p in enumerate(paths):
        _, start, meta = _read_footer(p)
        schema, kv = _field(meta, 2), _field(meta, 5)
        if first_schema is None:
            first_meta, first_schema, first_kv = meta, schema, kv
        elif schema != first_schema or kv != first_kv:
            raise ValueError(f'{p}: schema differs from {paths[0]}')
        if _field(meta, 8) is not None:
            raise ValueError(f'{p}: encrypted parquet not supported')
        groups, spans = _row_groups(meta), _row_group_spans(meta, start, p)
        side_groups = side_spans = sp = None
        if side_paths is not None:
            sp = side_paths[i]
            _, side_start, side_meta = _read_footer(sp)
            side_schema, side_kv = _field(side_meta, 2), _field(side_meta, 5)
            if first_side_schema is None:
                first_side_meta, first_side_schema, first_side_kv = side_meta, side_schema, side_kv
            elif side_schema != first_side_schema or side_kv != first_side_kv:
                raise ValueError(f'{sp}: schema differs from {side_paths[0]}')
            if _field(side_meta, 8) is not None:
                raise ValueError(f'{sp}: encrypted parquet not supported')
            side_groups, side_spans = _row_groups(side_meta), _row_group_spans(side_meta, side_start, sp)
            if [_field(rg, 3)[2] for rg in side_groups] != [_field(rg, 3)[2] for rg in groups]:
                raise ValueError(f'{sp}: row groups do not match {p}')
        for k, rg in enumerate(groups):
            take(p, rg, spans[k])
            if side_groups is not None:
                side_rg = side_groups[k]
                take(sp, side_rg, side_spans[k])
                _field(rg, 1)[2][1].extend(_field(side_rg, 1)[2][1])
                _field(rg, 2)[2] += _field(side_rg, 2)[2]
                compressed, side_compressed = _field(rg, 6), _field(side_rg, 6)
                if (compressed is None) != (side_compressed is None):
                    raise ValueError(f'{sp}: total_compressed_size present in only one of the files')
                if compressed is not None:
                    compressed[2] += side_compressed[2]
            ordinal = _field(rg, 7)
            if ordinal is not None:
                ordinal[2] = len(row_groups)
            row_groups.append(rg)
        total_rows += _field(meta, 3)[2]

    # Deep copy: the schema and key/value fields are edited below when side columns are added.
    meta = copy.deepcopy([f for f in first_meta if f[0] != 4])
    rg_at = next((i for i, f in enumerate(first_meta) if f[0] == 4), None)
    if rg_at is None:
        raise ValueError(f'{paths[0]}: no row_groups field')
    meta.insert(rg_at, [4, _T_LIST, (_T_STRUCT, row_groups)])
    _field(meta, 3)[2] = total_rows
    if side_paths is not None:
        _append_side_schema(meta, first_side_meta, side_paths[0])
    w = _ThriftWriter()
    w.struct(meta)
    footer = bytes(w.out)

    with open(out_file, 'wb') as out:
        out.write(b'PAR1')
        for p, offset, left in segments:
            with open(p, 'rb') as f:
                f.seek(offset)
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
# Block cache: one shard per block of sessions
# ---------------------------------------------------------------------------

def _sha(*parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        b = part if isinstance(part, (bytes, bytearray)) else str(part).encode('utf-8')
        h.update(len(b).to_bytes(8, 'little'))
        h.update(b)
    return h.hexdigest()


def _code_fingerprint() -> str:
    """Hash of the code that shapes a block's output, plus library versions."""
    import mlBridge.mlBridgeLib as mlBridgeLib
    import numpy
    parts = []
    for f in (__file__, mlBridgeAugmentLib.__file__, mlBridgeLib.__file__):
        p = pathlib.Path(f).resolve()
        parts += [p.name, p.read_bytes()]
    parts.append(inspect.getsource(drop_out_of_range_pct))
    parts += [pl.__version__, pyarrow.__version__, numpy.__version__]
    parts += [JOIN_KEYS, ROW_GROUP_SIZE, COMPRESSION, CACHE_VERSION]
    parts += [SESSION_BLOCK, TOURNAMENT_BLOCK_KEY, BLOCK_SORT_KEYS, RATING_COLS, RATING_INPUT_COLS]
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


def _block_starts(session_ids: pl.Series) -> list:
    """First session id of every block that holds at least one of these sessions, ascending."""
    return (session_ids // SESSION_BLOCK * SESSION_BLOCK).unique().sort().to_list()


def _slice_bounds(sorted_ids: pl.Series, lo: int, hi: int) -> tuple:
    """(offset, length) of the rows with lo <= id < hi in an ascending series."""
    a = int(sorted_ids.search_sorted(lo, side='left'))
    b = int(sorted_ids.search_sorted(hi, side='left'))
    return a, b - a


def _club_blocks(brs_ids: pl.Series, hrs_ids: pl.Series) -> list:
    """[(block key, step1 slice, hand records slice)] for ascending integer session ids."""
    return [(lo, _slice_bounds(brs_ids, lo, lo + SESSION_BLOCK), _slice_bounds(hrs_ids, lo, lo + SESSION_BLOCK))
            for lo in _block_starts(brs_ids)]


def _tournament_block_keys(session_ids: pl.Series) -> pl.Series:
    return session_ids.str.extract(TOURNAMENT_BLOCK_KEY, 1).fill_null(session_ids.str.slice(0, 4))


def _key_runs(keys: pl.Series) -> list:
    """[(key, (offset, length))] in order of appearance; every key's rows must be adjacent."""
    runs = (pl.DataFrame({'_k': keys}).with_row_index('_at')
            .group_by('_k', maintain_order=True)
            .agg(pl.col('_at').first().alias('_first'), pl.col('_at').last().alias('_last'), pl.len().alias('_n')))
    split = runs.filter(pl.col('_last') - pl.col('_first') + 1 != pl.col('_n'))
    if split.height:
        raise RuntimeError(f"rows of block {split['_k'][0]!r} are not adjacent: the session ids are not sorted")
    return [(k, (int(first), int(n))) for k, first, _, n in runs.iter_rows()]


def _tournament_blocks(brs_ids: pl.Series, hrs_ids: pl.Series) -> list:
    """[(block key, step1 slice, hand records slice)] for string session ids, both sorted.

    A block with no hand records gets an empty slice.
    """
    hrs_runs = dict(_key_runs(_tournament_block_keys(hrs_ids)))
    return [(key, brs_at, hrs_runs.get(key, (0, 0))) for key, brs_at in _key_runs(_tournament_block_keys(brs_ids))]


def _block_fingerprint(common: str, part_brs_df: pl.DataFrame, hrs_row_hashes: pl.Series) -> str:
    """Covers the block's step1 rows and its hand records, both in file order.

    Row positions in the hand records file are left out, so sessions added to an
    earlier block do not change this one.
    """
    return _sha(
        common,
        part_brs_df.height,
        _hash_series_bytes(part_brs_df.hash_rows(seed=0)),
        hrs_row_hashes.len(),
        _hash_series_bytes(hrs_row_hashes),
    )


def _write_json_atomic(obj, path: pathlib.Path):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding='utf-8')
    os.replace(tmp, path)


def _load_manifest(path: pathlib.Path) -> dict:
    try:
        m = json.loads(path.read_text(encoding='utf-8'))
        if m.get('version') == CACHE_VERSION:
            m.setdefault('blocks', {})
            return m
    except (OSError, ValueError):
        pass
    return {'version': CACHE_VERSION, 'blocks': {}}


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
        return (_num_rows(p) == entry['rows'] and p.stat().st_size == entry['size']
                and _num_rows(cache_dir / entry['inputs']) == entry['rows'])
    except (OSError, KeyError, TypeError, ValueError, pyarrow.ArrowException):
        return False


def _augment_block(side: str, part_brs_df: pl.DataFrame, hrs_df: pl.DataFrame, label: str, sd_lead=None) -> pl.DataFrame:
    """Join one block of step1 rows to its hand records and augment; returns None if the join is empty.

    Rows keep the order of part_brs_df. The result still holds RATING_COLS as averages
    over this block only; the caller leaves them out of the shard.
    """
    part_brs_df = part_brs_df.join(hrs_df, left_on=JOIN_KEYS, right_on=JOIN_KEYS, how='inner', maintain_order='left')
    if side == 'tournament':
        # todo: club has 'section_name_right'; see TODO 1 at the top.
        assert part_brs_df.select('^.*_right$').is_empty(), part_brs_df.select('^.*_right$')
    if part_brs_df.is_empty():
        return None
    part_brs_df = mlBridgeAugmentLib.AllBoardResultsAugmentations(part_brs_df, sd_lead=sd_lead).perform_all_board_results_augmentations()
    if sd_lead is not None:
        sd_lead.flush()  # keep solved positions even if a later block fails
    non_object_columns = [col for col, dtype in zip(part_brs_df.columns, part_brs_df.dtypes) if dtype != pl.Object]
    part_brs_df = part_brs_df.select(non_object_columns)
    if side == 'club':
        part_brs_df = drop_out_of_range_pct(part_brs_df, label)
    _validate_percentage_columns(part_brs_df, label)
    return part_brs_df


def _write_shard(df: pl.DataFrame, path: pathlib.Path):
    """Write with the final file's settings; temp file + rename so a crash leaves no half shard.

    Polars encodes the columns in parallel. PyArrow encodes them one after another: on
    2026-11-13 one 512,000-row group of 6,804 columns took 261 s with PyArrow and 5 s with Polars.
    """
    tmp = path.with_name(path.name + '.tmp')
    df.write_parquet(tmp, compression=COMPRESSION, row_group_size=ROW_GROUP_SIZE)
    os.replace(tmp, path)


def _sd_lead_fingerprint(sd_lead) -> str:
    return 'sd_lead=off' if sd_lead is None else f'sd_lead={sd_lead.samples}:{sd_lead.mode}'


def _block_name(key) -> str:
    """File name stem of a block: club keys are first session ids, tournament keys are id prefixes."""
    return f'block_{key:08d}' if isinstance(key, int) else f'block_{key}'


def _block_label(side: str, key) -> str:
    if side == 'club':
        return f"club sessions {key}-{key + SESSION_BLOCK - 1}"
    return f"tournament sessions {key}"


def _write_rating_files(cache_dir: pathlib.Path, entries: list) -> list:
    """Compute RATING_COLS over all blocks and write them as one side file per shard.

    entries are the manifest entries of the shards, in final file order. Each side file
    holds its shard's rows in its shard's row groups, as concat_parquet_files needs.
    The averages are over the rows kept in the shards, so rows removed by
    drop_out_of_range_pct do not count.
    """
    inputs = pl.concat([pl.read_parquet(cache_dir / e['inputs']) for e in entries])
    ratings = mlBridgeAugmentLib.add_player_ratings(inputs).select(RATING_COLS)
    del inputs
    side_files = []
    offset = 0
    for e in entries:
        with open(cache_dir / e['file'], 'rb') as f:
            md = pq.ParquetFile(f).metadata
            group_rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        table = ratings.slice(offset, e['rows']).to_arrow()
        assert sum(group_rows) == table.num_rows == e['rows'], f"{e['file']}: {sum(group_rows)} rows in file, {table.num_rows} ratings, {e['rows']} in manifest"
        path = cache_dir / e['file'].replace('.parquet', '.ratings.parquet')
        tmp = path.with_name(path.name + '.tmp')
        writer = pq.ParquetWriter(tmp, table.schema, compression=COMPRESSION)
        try:
            at = 0
            for n in group_rows:
                writer.write_table(table.slice(at, n), row_group_size=n)
                at += n
        finally:
            writer.close()
        os.replace(tmp, path)
        side_files.append(path)
        offset += e['rows']
    return side_files


def _augment_by_block(side: str, brs_df: pl.DataFrame, hrs_file: pathlib.Path, out_file: pathlib.Path, rebuild_all: bool, sd_lead=None):
    cache_dir = out_file.with_name(out_file.stem + '_blocks')
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = cache_dir / 'manifest.json'
    manifest = _load_manifest(manifest_file)

    # Session order makes every block one slice of step1, and gives the final file its row order.
    t = time.time()
    brs_df = brs_df.sort([c for c in BLOCK_SORT_KEYS if c in brs_df.columns], maintain_order=True)
    brs_ids = brs_df['session_id']
    print(f"sorted {len(brs_df)} step1 rows in {time.time()-t:.1f}s")

    t = time.time()
    hrs_schema = pl.read_parquet_schema(hrs_file)
    # Turning the SD lead score on or off (or changing its samples) recomputes every block.
    common = _sha(_code_fingerprint(), _schema_str(brs_df.schema), _schema_str(hrs_schema), _sd_lead_fingerprint(sd_lead))
    if side == 'club':
        # Too big to hold: hashed in slices here, read one block at a time below.
        all_hrs_df = None
        hrs_hashes = _hrs_row_hashes(hrs_file)
        if not hrs_hashes['session_id'].is_sorted():
            raise RuntimeError(
                f"{hrs_file.name} is not in session_id order, so a block's hand records cannot be read on their own. "
                "Rerun acbl_sql_to_hand_records_clean.py (step 2a), then acbl_hand_records_augment.py (step 3a).")
        blocks = _club_blocks(brs_ids, hrs_hashes['session_id'])
    else:
        # About 2M rows, not in session order in the file: held in memory, sorted here.
        # The join keys are unique, so the order (and each block's fingerprint) does not
        # depend on the file's row order.
        all_hrs_df = pl.read_parquet(hrs_file).sort(JOIN_KEYS)
        hrs_hashes = all_hrs_df.select(JOIN_KEYS).with_columns(all_hrs_df.hash_rows(seed=0).alias('_h'))
        blocks = _tournament_blocks(brs_ids, hrs_hashes['session_id'])
    print(f"hashed {len(hrs_hashes)} hand records in {time.time()-t:.1f}s; {len(blocks)} blocks")

    # Fingerprint every block and decide which ones to recompute.
    t = time.time()
    plan = []
    for lo, brs_at, hrs_at in blocks:
        fp = _block_fingerprint(common, brs_df.slice(*brs_at), hrs_hashes['_h'].slice(*hrs_at))
        skip = not rebuild_all and _shard_ok(manifest['blocks'].get(str(lo)), cache_dir, fp)
        plan.append((lo, fp, skip, brs_at, hrs_at))
    del hrs_hashes, blocks
    todo = sum(not skip for _, _, skip, _, _ in plan)
    print(f"{len(plan) - todo} blocks unchanged, {todo} to recompute (fingerprints {time.time()-t:.1f}s)")

    hrs_lf = pl.scan_parquet(hrs_file)
    for n, (lo, fp, skip, brs_at, hrs_at) in enumerate(plan, 1):
        if skip:
            continue
        t = time.time()
        label = _block_label(side, lo)
        part_brs_df = brs_df.slice(*brs_at)
        hrs_df = hrs_lf.slice(*hrs_at).collect() if all_hrs_df is None else all_hrs_df.slice(*hrs_at)
        print(f"\nBlock {n} of {len(plan)}, {label}: {part_brs_df.height} step1 rows, {hrs_df.height} hand records")
        part_brs_df = _augment_block(side, part_brs_df, hrs_df, label, sd_lead)
        del hrs_df
        shard_name = f'{_block_name(lo)}.parquet'
        inputs_name = f'{_block_name(lo)}.ratings_in.parquet'
        if part_brs_df is None:
            print(f"No data for {label}")
            (cache_dir / shard_name).unlink(missing_ok=True)
            (cache_dir / inputs_name).unlink(missing_ok=True)
            entry = {'fingerprint': fp, 'rows': 0, 'file': None}
        else:
            _write_shard(part_brs_df.select(RATING_INPUT_COLS), cache_dir / inputs_name)
            part_brs_df = part_brs_df.drop(RATING_COLS)
            _write_shard(part_brs_df, cache_dir / shard_name)
            entry = {'fingerprint': fp, 'rows': len(part_brs_df), 'file': shard_name, 'inputs': inputs_name,
                     'size': (cache_dir / shard_name).stat().st_size, 'columns': len(part_brs_df.columns)}
        entry['seconds'] = round(time.time() - t, 1)
        manifest['blocks'][str(lo)] = entry
        _write_json_atomic(manifest, manifest_file)
        print(f"block {lo}: recomputed, {entry['rows']} rows, elapsed time in seconds: {time.time()-t:.1f}")
        part_brs_df = None
    del brs_df, all_hrs_df

    # Forget blocks that no longer exist in step1.
    current = {str(lo) for lo, _, _, _, _ in plan}
    for key in [k for k in manifest['blocks'] if k not in current]:
        stale = manifest['blocks'].pop(key)
        for suffix in ('.parquet', '.ratings_in.parquet', '.ratings.parquet'):
            (cache_dir / f"{_block_name(int(key) if side == 'club' else key)}{suffix}").unlink(missing_ok=True)
        print(f"block {key}: no longer in step1, dropped from cache ({stale.get('rows')} rows)")
    _write_json_atomic(manifest, manifest_file)

    entries = [manifest['blocks'][str(lo)] for lo, _, _, _, _ in plan if manifest['blocks'][str(lo)]['file']]
    if not entries:
        return
    shards = [cache_dir / e['file'] for e in entries]

    # Columns must not change across blocks.
    prev_columns = None
    for p in shards:
        with open(p, 'rb') as f:
            columns = pq.ParquetFile(f).schema_arrow.names
        if prev_columns is None:
            prev_columns = columns
        assert columns == prev_columns, f"columns changed from {prev_columns} to {columns} in {p.name}"

    # Assemble the final file, unless it already matches these shards.
    assembled_fp = _sha(*[f'{lo}:{fp}' for lo, fp, _, _, _ in plan])
    prev = manifest.get('assembled') or {}
    try:
        st = out_file.stat()
        up_to_date = (prev.get('fingerprint') == assembled_fp and prev.get('size') == st.st_size
                      and prev.get('mtime_ns') == st.st_mtime_ns)
    except OSError:
        up_to_date = False
    if up_to_date and not rebuild_all:
        print(f"{out_file.name} already matches the block shards, not rewritten")
        return

    # The ratings average over every block, so they are rewritten whenever any block changed.
    t = time.time()
    side_files = _write_rating_files(cache_dir, entries)
    print(f"player ratings over {sum(e['rows'] for e in entries)} rows written in {time.time()-t:.1f}s")

    t = time.time()
    tmp = out_file.with_name(out_file.name + '.tmp')
    try:
        total_rows = concat_parquet_files(shards, tmp, side_paths=side_files)
        # Cheap read check: decode one shard column and the rating columns across all row groups.
        with open(tmp, 'rb') as f:
            assert pq.ParquetFile(f).read(columns=[prev_columns[0]] + RATING_COLS).num_rows == total_rows
        os.replace(tmp, out_file)
    finally:
        tmp.unlink(missing_ok=True)
    st = out_file.stat()
    manifest['assembled'] = {'fingerprint': assembled_fp, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}
    _write_json_atomic(manifest, manifest_file)
    print(f"Saved {out_file.name}: {total_rows} total rows, size:{st.st_size}, assembled from {len(shards)} shards in {time.time()-t:.1f}s")


def augment_board_results_step2(club_or_tournament, data_path=None, recent_path=None, rebuild_all=False,
                                sd_lead_samples=0, sd_lead_mode='contract'):
    """Perform final augmentation on board results by joining with hand records.

    sd_lead_samples > 0 adds the single-dummy lead score (SD_Lead_* columns) to
    boards with an opening lead. Solved positions are kept in
    acbl_{club|tournament}_sd_lead_cache/, so later runs only solve new ones.
    """
    print(f"Processing {club_or_tournament} board results augmentation step 2...")
    data_path = acblPath if data_path is None else pathlib.Path(data_path)
    recent_path = recentPath if recent_path is None else pathlib.Path(recent_path)
    sd_lead = None
    if sd_lead_samples > 0:
        sd_lead = mlBridgeAugmentLib.SDLeadCache(
            sd_lead_samples, sd_lead_mode, data_path / f'acbl_{club_or_tournament}_sd_lead_cache')
        print(f"SD lead score on: {sd_lead_samples} samples, {sd_lead_mode} mode, cache {sd_lead.directory}")

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

    if club_or_tournament not in ('club', 'tournament'):
        raise ValueError(f"Invalid club_or_tournament: {club_or_tournament}")
    #hrs_df = hrs_df.with_columns(pl.col('hand_record_id').cast(pl.Int64)) # todo: temp until previous step is corrected. issue is 'SHUFFLE' in hand_record_id for session.
    # Hand records are read inside: club one block of sessions at a time, tournament once.
    _augment_by_block(club_or_tournament, brs_df, acbl_hand_records_augmented_file, acbl_board_results_augmented_file, rebuild_all, sd_lead)

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
    parser.add_argument("--rebuild-all", action="store_true", help="Recompute every block of sessions, ignoring the cache")
    parser.add_argument("--sd-lead-samples", type=int, default=0,
                        help="Add the single-dummy lead score with this many layouts per lead (0 = off; 20 recommended). "
                             "First club run: about 33 h at 20.")
    parser.add_argument("--sd-lead-mode", choices=mlBridgeAugmentLib.SD_LEAD_MODES, default="contract",
                        help="contract: layouts keep declaring side HCP (+/-2) and trump length; hand: leader's hand only")
    args = parser.parse_args()
    if args.sd_lead_samples < 0:
        parser.error("--sd-lead-samples must be 0 or more")
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

    from mlBridge import print_started, print_ended
    program_start_time = print_started()

    modes, args = _parse_club_tournament_args()
    for club_or_tournament in modes:
        t = time.time()
        augment_board_results_step2(club_or_tournament, rebuild_all=args.rebuild_all,
                                    sd_lead_samples=args.sd_lead_samples, sd_lead_mode=args.sd_lead_mode)
        print(f"{club_or_tournament} elapsed time in seconds: {time.time()-t}")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("-" * 70, "\n")

    print_ended(program_start_time)
