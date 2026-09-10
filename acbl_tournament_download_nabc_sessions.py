"""Download missing NABC tournament sessions using canonical event ids.

NABC live.acbl.org session ids are ``NABC262-OSHL-1``, not the accounting
sanction form ``2607001-OSHL-1``. Platinum events are fetched first so the
Elo filter becomes usable before the remaining Gold/Red NABC sessions finish.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from acbl_tournament_download_sessions_using_sanctioned_events import (
    acblPath,
    audit_session_artifacts,
    canonical_event_id,
    download_tournament_sessions,
    expired_api_key_message,
)


def _session_exists(sessions_dir: Path, session_id: str) -> bool:
    return (sessions_dir / f"{session_id}.session.json").is_file() or (
        sessions_dir / f"{session_id}.session.sql"
    ).is_file()


def nabc_sessions_from_sanctions(
    events_dir: Path,
    *,
    platinum_only: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (platinum_session_ids, other_nabc_session_ids), newest first."""
    platinum: list[str] = []
    other: list[str] = []
    for fp in sorted(events_dir.glob("NABC*.sanction.json"), reverse=True):
        event = json.loads(fp.read_text(encoding="utf-8"))
        event_id = canonical_event_id(event)
        session_count = event.get("session_count")
        if not event_id or session_count is None:
            raise ValueError(f"invalid NABC sanction file: {fp}")
        try:
            count = int(session_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid session_count in {fp}") from exc
        ids = [f"{event_id}-{n}" for n in range(1, count + 1)]
        if str(event.get("mp_color") or "").strip().lower() == "platinum":
            platinum.extend(ids)
        elif not platinum_only:
            other.extend(ids)
    return platinum, other


def main() -> int:
    load_dotenv()
    started = datetime.now()
    print("start", started, flush=True)

    parser = argparse.ArgumentParser(
        description="Download missing NABC sessions (platinum first)"
    )
    parser.add_argument("--events-dir", default="tournaments/events")
    parser.add_argument("--sessions-dir", default="tournaments/sessions")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--platinum-only",
        action="store_true",
        help="Download only mp_color=Platinum NABC sessions",
    )
    args = parser.parse_args()

    api_key = os.getenv("ACBL_API_KEY")
    if not api_key:
        print("ERROR: ACBL_API_KEY environment variable not set", flush=True)
        return 1
    expiry_msg = expired_api_key_message(api_key)
    if expiry_msg:
        print(expiry_msg, flush=True)
        return 1

    events_dir = acblPath.joinpath(args.events_dir)
    sessions_dir = acblPath.joinpath(args.sessions_dir)
    if not events_dir.exists():
        print(f"ERROR: events directory does not exist: {events_dir}", flush=True)
        return 1
    sessions_dir.mkdir(parents=True, exist_ok=True)

    platinum_ids, other_ids = nabc_sessions_from_sanctions(
        events_dir, platinum_only=args.platinum_only
    )
    queue = [
        ("platinum", platinum_ids),
        ("other-nabc", other_ids),
    ]
    totals = {"written": 0, "skipped": 0, "errors": 0, "unavailable": 0, "aborted": False}

    print("=" * 70, flush=True)
    print("ACBL NABC session download (canonical event ids)", flush=True)
    print("=" * 70, flush=True)
    print(f"Events dir:   {events_dir}", flush=True)
    print(f"Sessions dir: {sessions_dir}", flush=True)
    print(f"Timeout:      {args.timeout}s  sleep:{args.sleep}s", flush=True)
    print(
        f"Queued:       platinum={len(platinum_ids)} other={len(other_ids)}",
        flush=True,
    )
    print(flush=True)

    for label, session_ids in queue:
        if not session_ids:
            continue
        missing = [sid for sid in session_ids if not _session_exists(sessions_dir, sid)]
        print(
            f"--- {label}: {len(session_ids)} sessions, "
            f"{len(missing)} missing, {len(session_ids) - len(missing)} already on disk ---",
            flush=True,
        )
        if not missing:
            continue
        stats = download_tournament_sessions(
            session_ids=missing,
            api_key=api_key,
            output_dir=sessions_dir,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            sleep_seconds=args.sleep,
        )
        still_missing = audit_session_artifacts(
            session_ids,
            sessions_dir,
            unavailable_ids=stats.unavailable_ids,
        )
        totals["written"] += stats.written
        totals["skipped"] += stats.skipped
        totals["errors"] += stats.errors
        totals["unavailable"] += stats.unavailable
        totals["aborted"] = totals["aborted"] or stats.aborted
        print(
            f"--- {label} done: written={stats.written} skipped={stats.skipped} "
            f"errors={stats.errors} unavailable={stats.unavailable} "
            f"still_missing={len(still_missing)} ---",
            flush=True,
        )

    elapsed = (datetime.now() - started).total_seconds()
    print(flush=True)
    print("=" * 70, flush=True)
    print(
        f"COMPLETE: written:{totals['written']} skipped:{totals['skipped']} "
        f"errors:{totals['errors']} unavailable:{totals['unavailable']}",
        flush=True,
    )
    print(f"elapsed {elapsed:.1f}s", flush=True)
    print("=" * 70, flush=True)
    if totals["aborted"] or totals["errors"]:
        print("FAILED: hard NABC download errors remain.", flush=True)
        return 1
    if totals["unavailable"]:
        print(
            f"OK with {totals['unavailable']} unavailable session(s) "
            f"(HTTP 400/404). They will be retried on the next run.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
