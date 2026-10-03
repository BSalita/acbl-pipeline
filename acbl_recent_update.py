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


def historical_max_date(path: pathlib.Path) -> Optional[date]:
    """Newest game_date in the stage-1b sessions parquet, if that file exists."""
    if not path.is_file():
        return None
    value = (
        pl.scan_parquet(path).select(pl.col("game_date").max()).collect().item()
    )
    if value is None or str(value).strip() == "":
        return None
    return date.fromisoformat(str(value)[:10])


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
    historical_max = historical_max_date(args.sessions_parquet)
    coverage = coverage_start(historical_max, args.every, args.today)
    print(
        f"[recent] every={args.every} coverage_start={coverage.isoformat()} "
        f"historical_max={historical_max.isoformat() if historical_max else 'none'}",
        flush=True,
    )
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
        args.recent_dir, extracted, historical_max=historical_max
    )
    _write_manifest(
        args.recent_dir,
        every=args.every,
        coverage=coverage,
        historical_max=historical_max,
        clubs=clubs,
        event_count=event_count,
    )
    print(f"[recent] events in recent store: {event_count}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
