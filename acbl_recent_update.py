"""Download recent ACBL club games and upsert them into the recent parquet store.

The store lives beside the historical files, under ``e:/bridge/data/acbl/recent``.
Each normalized table is one parquet. Rows carry ``event_id`` and ``game_date``
so a later run can drop sessions that stage 1b has already absorbed.

``--every`` chooses the schedule window and how many clubs are visited:

- hour: yesterday through today, then the next slice of clubs (default 200)
- day: the last 2 days, every club
- week: the last 8 days, every club
- quarter: the last 95 days, every club

The download start is also the day after the newest ``game_date`` in the
stage-1b sessions parquet, when that day is earlier than the schedule window.
Re-runs skip session JSON that is already on disk and refresh club listing
pages so a new game is visible.

Schedule examples (Task Scheduler, start in this directory)::

    acbl_recent.bat hour
    acbl_recent.bat day
    acbl_recent.bat week
    acbl_recent.bat quarter
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence

import polars as pl

from acbl_club_download_to_json import (
    Forbidden403Error,
    get_all_clubs,
    process_club_sessions,
)
from acbl_club_json_to_parquet_sql import _extract_tables_from_json

LOOKBACK_DAYS = {"hour": 2, "day": 2, "week": 8, "quarter": 95}
HOUR_CLUB_BATCH = 200
DEFAULT_RECENT_DIR = pathlib.Path(r"e:/bridge/data/acbl/recent")
DEFAULT_ARCHIVE = pathlib.Path(r"e:/bridge/data/acbl/club-results")
DEFAULT_SESSIONS = pathlib.Path(
    r"e:/bridge/data/acbl/club_results_parquet/sessions.parquet"
)
DEFAULT_PARQUET_DIR = DEFAULT_SESSIONS.parent
DEFAULT_AUGMENTED = pathlib.Path(
    r"e:/bridge/data/acbl/acbl_club_board_results_augmented.parquet"
)


def coverage_start(
    historical_max: Optional[date], every: str, today: date
) -> date:
    """First game date this run should download and keep.

    A stale historical file widens the window back to the day after its max
    date. A current file keeps the schedule overlap (two days for hour/day).
    """
    lookback = today - timedelta(days=LOOKBACK_DAYS[every] - 1)
    if historical_max is None:
        return lookback
    return min(historical_max + timedelta(days=1), lookback)


def _date_from_stat(value: object) -> Optional[date]:
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value)[:10])


def historical_max_date(path: pathlib.Path) -> Optional[date]:
    """Newest game_date in the stage-1b sessions parquet, if that file exists."""
    if not path.is_file():
        return None
    value = (
        pl.scan_parquet(path).select(pl.col("game_date").max()).collect().item()
    )
    return _date_from_stat(value)


def augmented_max_date(path: pathlib.Path) -> Optional[date]:
    """Newest Date in the augmented board-results parquet, if that file exists."""
    if not path.is_file():
        return None
    schema = pl.scan_parquet(path).collect_schema()
    if "Date" not in schema.names():
        return None
    value = pl.scan_parquet(path).select(pl.col("Date").max()).collect().item()
    return _date_from_stat(value)


def next_club_batch(
    club_ids: Sequence[int], cursor: Optional[int], batch: int
) -> tuple[List[int], Optional[int]]:
    """Return the next club slice and the cursor to save after it finishes.

    ``batch <= 0`` or a batch that covers every club returns the full list and
    a null cursor. Otherwise the slice continues after ``cursor`` and wraps.
    """
    ids = list(club_ids)
    if not ids:
        return [], None
    if batch <= 0 or batch >= len(ids):
        return ids, None
    start = 0
    if cursor is not None:
        for index, club_id in enumerate(ids):
            if club_id > cursor:
                start = index
                break
        else:
            start = 0
    chosen = [ids[(start + offset) % len(ids)] for offset in range(batch)]
    return chosen, chosen[-1]


def _parse_game_date(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def stamped_tables(path: pathlib.Path) -> Dict[str, List[dict]]:
    """Normalized tables for one session JSON, tagged with its event id and date."""
    tables = _extract_tables_from_json(path)
    events = tables.get("events") or []
    if not events:
        return {}
    event_id = str(events[0].get("id") or "").strip()
    if not event_id:
        return {}
    game_date = ""
    for row in tables.get("sessions") or []:
        game_date = _parse_game_date(row.get("game_date"))
        if game_date:
            break
    if not game_date:
        game_date = _parse_game_date(events[0].get("start_date"))
    stamped: Dict[str, List[dict]] = {}
    for name, rows in tables.items():
        stamped[name] = [
            {**row, "event_id": event_id, "game_date": game_date} for row in rows
        ]
    return stamped


def _align_concat(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    return pl.concat(list(frames), how="diagonal_relaxed")


def _atomic_parquet(frame: pl.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary, compression="zstd")
    temporary.replace(path)


def upsert_tables(
    recent_dir: pathlib.Path,
    extracted: Iterable[Dict[str, List[dict]]],
    *,
    historical_max: Optional[date],
) -> int:
    """Replace rows for the incoming events and drop sessions already in history.

    Returns the number of event ids still stored.
    """
    incoming: Dict[str, List[dict]] = {}
    event_ids: set[str] = set()
    for tables in extracted:
        for name, rows in tables.items():
            incoming.setdefault(name, []).extend(rows)
            for row in rows:
                if row.get("event_id"):
                    event_ids.add(str(row["event_id"]))
    recent_dir.mkdir(parents=True, exist_ok=True)
    names = set(incoming)
    for path in recent_dir.glob("*.parquet"):
        names.add(path.stem)
    cutoff = historical_max.isoformat() if historical_max is not None else None
    for name in sorted(names):
        path = recent_dir / f"{name}.parquet"
        frames: List[pl.DataFrame] = []
        if path.is_file():
            old = pl.read_parquet(path)
            if event_ids and "event_id" in old.columns:
                old = old.filter(~pl.col("event_id").is_in(list(event_ids)))
            frames.append(old)
        rows = incoming.get(name) or []
        if rows:
            frames.append(
                pl.DataFrame(
                    [{key: (None if value is None else str(value)) for key, value in row.items()} for row in rows],
                    infer_schema_length=len(rows),
                )
            )
        if not frames:
            continue
        frame = _align_concat(frames)
        if cutoff is not None and "game_date" in frame.columns:
            frame = frame.filter(
                (pl.col("game_date") == "") | (pl.col("game_date") > cutoff)
            )
        if frame.is_empty():
            if path.is_file():
                path.unlink()
            continue
        _atomic_parquet(frame, path)
    events_path = recent_dir / "events.parquet"
    if not events_path.is_file():
        return 0
    return pl.scan_parquet(events_path).select(pl.len()).collect().item()


def scan_with_recent(
    historical: Optional[pathlib.Path],
    table: str,
    recent_dir: pathlib.Path = DEFAULT_RECENT_DIR,
) -> Optional[pl.LazyFrame]:
    """Scan a stage-1b table with recent rows of the same name overlaid on ``id``."""
    frames: List[pl.LazyFrame] = []
    if historical is not None and historical.is_file():
        frames.append(pl.scan_parquet(historical))
    recent = recent_dir / f"{table}.parquet"
    if recent.is_file():
        frames.append(pl.scan_parquet(recent))
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    historical_lf, recent_lf = frames
    hist_names = set(historical_lf.collect_schema().names())
    recent_names = set(recent_lf.collect_schema().names())
    if "id" in hist_names and "id" in recent_names:
        historical_lf = historical_lf.join(
            recent_lf.select(pl.col("id").cast(pl.Utf8)),
            left_on=pl.col("id").cast(pl.Utf8),
            right_on="id",
            how="anti",
        )
    return pl.concat([historical_lf, recent_lf], how="diagonal_relaxed")


def remember_session_json(
    path: pathlib.Path,
    recent_dir: pathlib.Path = DEFAULT_RECENT_DIR,
) -> int:
    """Upsert one downloaded session JSON into the recent store."""
    tables = stamped_tables(path)
    if not tables:
        return 0
    return upsert_tables(recent_dir, [tables], historical_max=None)


def prune_recent(
    recent_dir: pathlib.Path,
    absorbed_through: Optional[date],
) -> int:
    """Drop recent rows whose game_date is already in the augmented parquet."""
    if absorbed_through is None or not recent_dir.is_dir():
        return 0
    cutoff = absorbed_through.isoformat()
    removed = 0
    for path in recent_dir.glob("*.parquet"):
        frame = pl.read_parquet(path)
        if "game_date" not in frame.columns:
            continue
        kept = frame.filter(
            (pl.col("game_date") == "") | (pl.col("game_date") > cutoff)
        )
        removed += frame.height - kept.height
        if kept.height == frame.height:
            continue
        if kept.is_empty():
            path.unlink()
        else:
            _atomic_parquet(kept, path)
    return removed


def _ids(frame: pl.DataFrame, column: str) -> List[str]:
    if frame.is_empty() or column not in frame.columns:
        return []
    return frame[column].cast(pl.Utf8).drop_nulls().unique().to_list()


def _stamp_gap(
    frame: pl.DataFrame, event_dates: pl.DataFrame
) -> pl.DataFrame:
    stamped = frame.with_columns(pl.col("event_id").cast(pl.Utf8))
    return stamped.join(event_dates, on="event_id", how="left").with_columns(
        pl.col("game_date").cast(pl.Utf8)
    )


def backfill_stage1b(
    parquet_dir: pathlib.Path,
    recent_dir: pathlib.Path,
    absorbed_through: Optional[date],
) -> int:
    """Copy stage-1b rows newer than the augmented parquet into the recent store.

    The scheduled download skips JSON that stage 1b already saved. Those games
    still have to stay in the recent store until stage 3c absorbs them.
    """
    sessions_path = parquet_dir / "sessions.parquet"
    if absorbed_through is None or not sessions_path.is_file():
        return 0
    cutoff = absorbed_through.isoformat()
    gap = (
        pl.scan_parquet(sessions_path)
        .select(
            pl.col("id").cast(pl.Utf8).alias("session_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8).str.slice(0, 10).alias("game_date"),
        )
        .filter(pl.col("game_date") > cutoff)
        .unique(subset=["event_id"])
        .collect()
    )
    if gap.is_empty():
        return 0
    events_path = recent_dir / "events.parquet"
    if events_path.is_file():
        have = pl.read_parquet(events_path, columns=["event_id"]).select(
            pl.col("event_id").cast(pl.Utf8)
        )
        gap = gap.join(have, on="event_id", how="anti")
    if gap.is_empty():
        return 0
    event_ids = gap["event_id"].to_list()
    event_dates = gap.select("event_id", "game_date")
    session_ids = gap["session_id"].to_list()

    def _take(name: str, column: str, values: Sequence[str]) -> pl.DataFrame:
        path = parquet_dir / f"{name}.parquet"
        if not path.is_file() or not values:
            return pl.DataFrame()
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col(column).cast(pl.Utf8).is_in(values))
            .collect()
        )
        if "event_id" not in frame.columns or "game_date" in frame.columns:
            if "event_id" in frame.columns:
                return frame.with_columns(pl.col("event_id").cast(pl.Utf8))
            return frame
        return _stamp_gap(frame, event_dates)

    sessions = _take("sessions", "event_id", event_ids)
    if sessions.is_empty():
        return 0
    if "game_date" in sessions.columns:
        sessions = sessions.with_columns(
            pl.col("game_date").cast(pl.Utf8).str.slice(0, 10)
        )
    events = _take("events", "id", event_ids)
    if not events.is_empty() and "event_id" not in events.columns:
        events = events.with_columns(pl.col("id").cast(pl.Utf8).alias("event_id"))
        events = events.join(event_dates, on="event_id", how="left")
    sections = _take("sections", "session_id", session_ids)
    section_ids = _ids(sections, "id")
    if sections.height and "event_id" not in sections.columns:
        link = sessions.select(
            pl.col("id").cast(pl.Utf8).alias("session_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8),
        )
        sections = sections.with_columns(pl.col("session_id").cast(pl.Utf8)).join(
            link, on="session_id", how="left"
        )
    boards = _take("boards", "section_id", section_ids)
    board_ids = _ids(boards, "id")
    if boards.height and "event_id" not in boards.columns and sections.height:
        link = sections.select(
            pl.col("id").cast(pl.Utf8).alias("section_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8),
        )
        boards = boards.with_columns(pl.col("section_id").cast(pl.Utf8)).join(
            link, on="section_id", how="left"
        )
    board_results = _take("board_results", "board_id", board_ids)
    if board_results.height and "event_id" not in board_results.columns and boards.height:
        link = boards.select(
            pl.col("id").cast(pl.Utf8).alias("board_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8),
        )
        board_results = board_results.with_columns(
            pl.col("board_id").cast(pl.Utf8)
        ).join(link, on="board_id", how="left")
    pairs = _take("pair_summaries", "section_id", section_ids)
    if pairs.height and "event_id" not in pairs.columns and sections.height:
        link = sections.select(
            pl.col("id").cast(pl.Utf8).alias("section_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8),
        )
        pairs = pairs.with_columns(pl.col("section_id").cast(pl.Utf8)).join(
            link, on="section_id", how="left"
        )
    pair_ids = _ids(pairs, "id")
    players = _take("players", "pair_summary_id", pair_ids)
    if players.height and "event_id" not in players.columns and pairs.height:
        link = pairs.select(
            pl.col("id").cast(pl.Utf8).alias("pair_summary_id"),
            pl.col("event_id").cast(pl.Utf8),
            pl.col("game_date").cast(pl.Utf8),
        )
        players = players.with_columns(pl.col("pair_summary_id").cast(pl.Utf8)).join(
            link, on="pair_summary_id", how="left"
        )
    tables = {
        "events": events,
        "sessions": sessions,
        "sections": sections,
        "boards": boards,
        "board_results": board_results,
        "pair_summaries": pairs,
        "players": players,
    }
    _upsert_frames(recent_dir, tables, absorbed_through)
    return gap.height


def _upsert_frames(
    recent_dir: pathlib.Path,
    tables: Dict[str, pl.DataFrame],
    absorbed_through: Optional[date],
) -> None:
    recent_dir.mkdir(parents=True, exist_ok=True)
    cutoff = absorbed_through.isoformat() if absorbed_through is not None else None
    for name, incoming in tables.items():
        if incoming.is_empty():
            continue
        path = recent_dir / f"{name}.parquet"
        incoming = incoming.with_columns(
            [pl.col(column).cast(pl.Utf8) for column in incoming.columns]
        )
        frames = [incoming]
        event_ids: List[str] = []
        if "event_id" in incoming.columns:
            event_ids = incoming["event_id"].drop_nulls().unique().to_list()
        if path.is_file():
            old = pl.read_parquet(path)
            if event_ids and "event_id" in old.columns:
                old = old.filter(~pl.col("event_id").cast(pl.Utf8).is_in(event_ids))
            frames.insert(0, old)
        frame = _align_concat(frames)
        if cutoff is not None and "game_date" in frame.columns:
            frame = frame.filter(
                (pl.col("game_date") == "") | (pl.col("game_date") > cutoff)
            )
        if frame.is_empty():
            if path.is_file():
                path.unlink()
            continue
        _atomic_parquet(frame, path)


def files_to_ingest(
    club_ids: Sequence[int], archive: pathlib.Path, coverage: date
) -> List[pathlib.Path]:
    """Session JSON files written on or after ``coverage`` for these clubs."""
    found: List[pathlib.Path] = []
    for club_id in club_ids:
        details = archive / str(club_id) / "details"
        if not details.is_dir():
            continue
        for path in details.glob("*.data.json"):
            modified = date.fromtimestamp(path.stat().st_mtime)
            if modified >= coverage:
                found.append(path)
    return found


def _read_cursor(path: pathlib.Path) -> Optional[int]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if text.isdigit():
        return int(text)
    return None


def _write_manifest(
    recent_dir: pathlib.Path,
    *,
    every: str,
    coverage: date,
    historical_max: Optional[date],
    clubs: Sequence[int],
    event_count: int,
) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "every": every,
        "coverage_start": coverage.isoformat(),
        "historical_max": historical_max.isoformat() if historical_max else None,
        "clubs_visited": len(clubs),
        "events": event_count,
    }
    (recent_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--every",
        choices=tuple(LOOKBACK_DAYS),
        required=True,
        help="Schedule cadence. hour visits a slice of clubs; the others visit all.",
    )
    parser.add_argument(
        "--today",
        type=date.fromisoformat,
        default=date.today(),
        help="Override today's date (YYYY-MM-DD), for a backfill window.",
    )
    parser.add_argument("--recent-dir", type=pathlib.Path, default=DEFAULT_RECENT_DIR)
    parser.add_argument("--archive", type=pathlib.Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--sessions-parquet", type=pathlib.Path, default=DEFAULT_SESSIONS)
    parser.add_argument("--augmented-parquet", type=pathlib.Path, default=DEFAULT_AUGMENTED)
    parser.add_argument(
        "--club-batch",
        type=int,
        default=None,
        help="Clubs to visit this run. Default: 200 for hour, all clubs otherwise.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=2,
        help="Seconds between session downloads (default: 2).",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only upsert JSON already saved under the archive.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    sessions_max = historical_max_date(args.sessions_parquet)
    absorbed_through = augmented_max_date(args.augmented_parquet)
    coverage = coverage_start(sessions_max, args.every, args.today)
    print(
        f"[recent] every={args.every} coverage_start={coverage.isoformat()} "
        f"sessions_max={sessions_max.isoformat() if sessions_max else 'none'} "
        f"augmented_max={absorbed_through.isoformat() if absorbed_through else 'none'}",
        flush=True,
    )
    filled = backfill_stage1b(args.sessions_parquet.parent, args.recent_dir, absorbed_through)
    print(f"[recent] backfilled {filled} stage-1b event(s) newer than augmented", flush=True)
    batch = args.club_batch
    if batch is None:
        batch = HOUR_CLUB_BATCH if args.every == "hour" else 0
    cursor_path = args.recent_dir / "club_cursor.txt"
    if args.no_download:
        club_ids = [
            int(path.name)
            for path in args.archive.iterdir()
            if path.is_dir() and path.name.isdigit()
        ] if args.archive.is_dir() else []
        club_ids.sort()
    else:
        raw_ids = get_all_clubs()
        if not raw_ids:
            print("[recent] no clubs returned", flush=True)
            return 1
        club_ids = sorted(int(club_id) for club_id in raw_ids if str(club_id).isdigit())
    clubs, next_cursor = next_club_batch(
        club_ids, _read_cursor(cursor_path), batch
    )
    print(f"[recent] visiting {len(clubs)} club(s)", flush=True)
    if not args.no_download:
        for index, club_id in enumerate(clubs, start=1):
            print(f"[recent] club {index}/{len(clubs)} {club_id}", flush=True)
            try:
                process_club_sessions(
                    str(club_id),
                    args.archive,
                    coverage.isoformat(),
                    None,
                    None,
                    True,
                    args.sleep,
                )
            except Forbidden403Error as exc:
                print(f"[recent] challenge did not clear: {exc}", flush=True)
                return 1
            except Exception as exc:
                print(f"[recent] ERROR club {club_id}: {exc}", flush=True)
        if next_cursor is None:
            if cursor_path.is_file():
                cursor_path.unlink()
        else:
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            cursor_path.write_text(str(next_cursor), encoding="utf-8")
    paths = files_to_ingest(clubs, args.archive, coverage)
    print(f"[recent] ingesting {len(paths)} session file(s)", flush=True)
    extracted = []
    for path in paths:
        try:
            tables = stamped_tables(path)
        except Exception as exc:
            print(f"[recent] ERROR {path.name}: {exc}", flush=True)
            continue
        if tables:
            extracted.append(tables)
    event_count = upsert_tables(
        args.recent_dir, extracted, historical_max=absorbed_through
    )
    _write_manifest(
        args.recent_dir,
        every=args.every,
        coverage=coverage,
        historical_max=absorbed_through,
        clubs=clubs,
        event_count=event_count,
    )
    print(f"[recent] events in recent store: {event_count}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
