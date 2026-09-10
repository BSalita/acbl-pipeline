"""
ACBL Tournament Sessions Downloader (from sanctioned events)

Script version of `acbl_tournament_download_sessions_using_sanctioned_events.ipynb`.

Reads `tournaments/events/*.sanction.json` files (downloaded previously) and
derives session ids from the event API's canonical ``id``:

  <event_id>-<session_number>

NABC live sessions use ``NABC262-OSHL-1``, not the accounting-sanction form
``2607001-OSHL-1``.

Then downloads each session payload from:
  https://api.acbl.org/v1/tournament/session?id=<session_id>&full_monty=1

and writes:
  tournaments/sessions/<session_id>.session.json

Behavior:
  - Skips if ``<session_id>.session.json`` or ``.session.sql`` already exists
  - 400/404 are unavailable (cancelled, unpublished, no boards) and do not
    fail the run; they are retried on the next invocation
  - Timeouts and 429/5xx are retried; exhausted attempts fail the run
  - 401/403 abort immediately (invalid or expired ACBL_API_KEY)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import requests
from dotenv import load_dotenv

rootPath = pathlib.Path("e:/bridge/data")
acblPath = rootPath.joinpath("acbl")

DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_CONNECT_TIMEOUT_SECONDS = 15
UNAVAILABLE_HTTP_STATUSES = frozenset({400, 404})
AUTH_HTTP_STATUSES = frozenset({401, 403})
JWT_EXPIRY_LEEWAY_SECONDS = 60


class AcblApiAuthError(RuntimeError):
    """ACBL API rejected the bearer token (HTTP 401/403 or expired JWT)."""


def _iter_files_sorted(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    return sorted(paths, key=lambda p: p.as_posix())


def _load_sanction_event_files(events_dir: pathlib.Path) -> list[pathlib.Path]:
    # Ignore filesystem/editor artifacts such as the historical
    # ``.sanction.json`` placeholder; real event files always have a visible id.
    return _iter_files_sorted(
        path
        for path in events_dir.rglob("*.sanction.json")
        if not path.name.startswith(".")
    )


def jwt_expiry_unix(api_key: str) -> int | None:
    """Return the JWT ``exp`` claim, or None if the key is not a JWT."""
    parts = (api_key or "").strip().split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        exp = payload.get("exp")
        return int(exp) if exp is not None else None
    except (ValueError, TypeError, json.JSONDecodeError, OSError):
        return None


def expired_api_key_message(
    api_key: str,
    *,
    now: float | None = None,
) -> str | None:
    """If ``api_key`` is an expired JWT, return a user-facing error."""
    exp = jwt_expiry_unix(api_key)
    if exp is None:
        return None
    current = time.time() if now is None else now
    if exp + JWT_EXPIRY_LEEWAY_SECONDS >= current:
        return None
    expired_at = datetime.fromtimestamp(exp, tz=timezone.utc).isoformat()
    return (
        f"ERROR: ACBL_API_KEY JWT expired at {expired_at}. "
        "Get a new key at https://api.acbl.org and re-run."
    )


def canonical_event_id(event: Mapping[str, Any]) -> str:
    """Return the session-id prefix for a sanctioned-event payload.

    Prefer the event API's ``id``. Some older regional payloads store a
    truncated id that is just ``-<event_code>`` (e.g. ``-0810``); rebuild
    those from ``sanction`` + ``event_code``. Do not rebuild well-formed
    ids — NABC live ids are ``NABC262-OSHL``, not ``2607001-OSHL``.
    """
    event_id = str(event.get("id") or "").strip()
    sanction = str(event.get("sanction") or "").strip()
    event_code = str(event.get("event_code") or "").strip()
    if event_id.startswith("-") or not event_id:
        if sanction and event_code:
            return f"{sanction}-{event_code}"
        return ""
    return event_id


def build_session_ids_from_sanctioned_events(events_dir: pathlib.Path) -> list[str]:
    """
    Read `*.sanction.json` files and return unique session ids (sorted desc).

    Use the event API's canonical ``id`` as the session-id prefix. Rebuilding
    the prefix from ``sanction`` + ``event_code`` is wrong for NABCs: for
    example, the 2026 Oshlag event id is ``NABC262-OSHL`` while its accounting
    sanction is ``2607001``. The session endpoint expects
    ``NABC262-OSHL-<n>``.
    """
    event_files = _load_sanction_event_files(events_dir)
    sessions: list[str] = []
    invalid_files: list[str] = []

    for fp in event_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                event = json.load(f)
        except (OSError, json.JSONDecodeError):
            invalid_files.append(fp.as_posix())
            continue

        event_id = canonical_event_id(event)
        session_count = event.get("session_count")
        if not event_id or session_count is None:
            invalid_files.append(fp.as_posix())
            continue

        try:
            sc = int(session_count)
        except (TypeError, ValueError):
            invalid_files.append(fp.as_posix())
            continue

        for c in range(1, sc + 1):
            sessions.append(f"{event_id}-{c}")

    if invalid_files:
        preview = "\n  ".join(invalid_files[:20])
        raise ValueError(
            f"{len(invalid_files)} invalid sanctioned-event files; first entries:\n  {preview}"
        )

    # Notebook sorts reverse to start with newest-ish ids first.
    return sorted(set(sessions), reverse=True)


def session_request_timeout(timeout_seconds: int) -> tuple[float, float]:
    """Connect/read timeout pair. Large NABC full_monty payloads need a long read."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    read = float(timeout_seconds)
    connect = min(float(DEFAULT_CONNECT_TIMEOUT_SECONDS), read)
    return (connect, read)


def is_unavailable_http(status_code: int) -> bool:
    """True for permanent-for-this-run HTTP statuses (unpublished / no boards)."""
    return status_code in UNAVAILABLE_HTTP_STATUSES


@dataclass
class DownloadStats:
    written: int = 0
    skipped: int = 0
    errors: int = 0
    unavailable: int = 0
    aborted: bool = False
    unavailable_ids: list[str] = field(default_factory=list)

    def failed(self) -> bool:
        return self.aborted or self.errors > 0


def download_tournament_sessions(
    session_ids: list[str],
    api_key: str,
    output_dir: pathlib.Path,
    full_monty: int = 1,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = 4,
    sleep_seconds: float = 0.0,
    starting_nfile: int = 0,
    ending_nfile: int = 0,
    skip_if_json_exists: bool = True,
    skip_if_sql_exists: bool = True,
    delete_partial_json_on_error: bool = True,
) -> DownloadStats:
    """Download tournament session JSON payloads."""
    headers = {
        "Accept": "application/json",
        "Authorization": "Bearer " + api_key,
        # User-Agent is required (May 2025+).
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    base_url = "https://api.acbl.org/v1/tournament/session"
    output_dir.mkdir(parents=True, exist_ok=True)

    if ending_nfile == 0:
        ending_nfile = len(session_ids)
    filtered = session_ids[starting_nfile:ending_nfile]
    total = len(filtered)

    stats = DownloadStats()
    request_timeout = session_request_timeout(timeout)

    except_count = 0
    start_run = time.time()

    for idx, session_id in enumerate(filtered):
        file_json = output_dir / f"{session_id}.session.json"
        file_sql = output_dir / f"{session_id}.session.sql"

        if skip_if_sql_exists and file_sql.exists():
            print(f"{idx}/{total}: File exists: {file_sql}: skipping")
            stats.skipped += 1
            continue
        if skip_if_json_exists and file_json.exists():
            print(f"{idx}/{total}: File exists: {file_json}: skipping")
            stats.skipped += 1
            continue

        query = {"id": session_id, "full_monty": full_monty}
        url = base_url + "?" + urllib.parse.urlencode(query)
        t0 = time.time()
        print(f"{idx}/{total} url:{url}")

        if sleep_seconds > 0 and idx > 0:
            time.sleep(sleep_seconds)

        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.get(url, headers=headers, timeout=request_timeout)
            except KeyboardInterrupt:
                raise
            except requests.RequestException as ex:
                if attempt == max_attempts:
                    print(
                        f"ERROR after {max_attempts} attempts: "
                        f"{type(ex).__name__}: {ex}"
                    )
                    break
                wait_seconds = min(60, 2 ** (attempt - 1))
                print(
                    f"Transient request failure ({attempt}/{max_attempts}): "
                    f"{type(ex).__name__}; retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue

            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempt == max_attempts:
                    break
                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(60, 2 ** (attempt - 1))
                )
                print(
                    f"Transient HTTP {response.status_code} "
                    f"({attempt}/{max_attempts}); retrying in {wait_seconds}s"
                )
                time.sleep(wait_seconds)
                continue
            break

        if response is None:
            stats.errors += 1
            except_count += 1
            if except_count > 5:
                print("Consecutive request failures exceeded 5; stopping")
                stats.aborted = True
                break
            continue
        except_count = 0

        if response.status_code in AUTH_HTTP_STATUSES:
            stats.errors += 1
            stats.aborted = True
            print(
                f"ERROR: HTTP {response.status_code} unauthorized/forbidden "
                f"for {session_id}. ACBL_API_KEY is invalid or expired; stopping."
            )
            expiry_msg = expired_api_key_message(api_key)
            if expiry_msg:
                print(expiry_msg)
            break

        if is_unavailable_http(response.status_code):
            stats.unavailable += 1
            stats.unavailable_ids.append(session_id)
            print(
                f"UNAVAILABLE (HTTP {response.status_code}): {session_id} "
                f"(will retry next run)"
            )
            continue

        if response.status_code != 200:
            stats.errors += 1
            print(f"ERROR: HTTP {response.status_code}: url:{url}")
            continue

        try:
            json_response = response.json()
            json_pretty = json.dumps(json_response, indent=4)
            print(
                f"{idx}/{total} rate:{round(time.time() - t0, 2)} "
                f"elapsed:{round(time.time() - start_run, 1)} url:{url}"
            )
            print(f"{idx}/{total}: Writing:{file_json} size:{len(json_pretty)}")
            with open(file_json, "w", encoding="utf-8") as f:
                f.write(json_pretty)
            stats.written += 1
        except KeyboardInterrupt:
            raise
        except Exception:
            stats.errors += 1
            if delete_partial_json_on_error:
                file_json.unlink(missing_ok=True)
            print("ERROR writing response JSON")
            print(response.text[:500])
            continue

    return stats


def audit_session_artifacts(
    session_ids: list[str],
    output_dir: pathlib.Path,
    unavailable_ids: Iterable[str] | None = None,
) -> list[str]:
    """Write an audit manifest and return expected sessions with no JSON or SQL."""
    missing = [
        session_id
        for session_id in session_ids
        if not (output_dir / f"{session_id}.session.json").is_file()
        and not (output_dir / f"{session_id}.session.sql").is_file()
    ]
    unavailable = [sid for sid in missing if sid in set(unavailable_ids or [])]
    unresolved = [sid for sid in missing if sid not in set(unavailable)]
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "expected_sessions": len(session_ids),
        "complete_sessions": len(session_ids) - len(missing),
        "missing_sessions": missing,
        "unavailable_sessions": unavailable,
        "unresolved_sessions": unresolved,
    }
    audit_path = output_dir / "_session_download_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(
        f"Audit: expected:{len(session_ids)} complete:{len(session_ids) - len(missing)} "
        f"unavailable:{len(unavailable)} unresolved:{len(unresolved)} -> {audit_path}"
    )
    return missing


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Download ACBL tournament sessions using tournaments/events/*.sanction.json"
    )
    parser.add_argument(
        "--events-dir",
        default="tournaments/events",
        help=f"Input subdirectory relative to {acblPath} (default: tournaments/events)",
    )
    parser.add_argument(
        "--sessions-dir",
        default="tournaments/sessions",
        help=f"Output subdirectory relative to {acblPath} (default: tournaments/sessions)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="ACBL API key (default: from ACBL_API_KEY env var)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            f"Read timeout seconds; connect timeout is "
            f"{DEFAULT_CONNECT_TIMEOUT_SECONDS}s "
            f"(default: {DEFAULT_TIMEOUT_SECONDS})"
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
        help="Attempts for transient request/HTTP failures (default: 4)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests (default: 0.0)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting session index (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=0,
        help="Ending session index, 0 means all (default: 0)",
    )
    parser.add_argument(
        "--full-monty",
        type=int,
        default=1,
        help="full_monty query parameter (default: 1)",
    )
    parser.add_argument(
        "--no-skip-existing-json",
        action="store_false",
        dest="skip_if_json_exists",
        help="Do not skip when .session.json exists",
    )
    parser.set_defaults(skip_if_json_exists=True)
    parser.add_argument(
        "--no-skip-existing-sql",
        action="store_false",
        dest="skip_if_sql_exists",
        help="Do not skip when .session.sql exists",
    )
    parser.set_defaults(skip_if_sql_exists=True)
    parser.add_argument(
        "--session-id",
        action="append",
        default=[],
        help=(
            "Download one exact session id; repeat for multiple ids. "
            "When supplied, sanctioned-event discovery is skipped."
        ),
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ACBL_API_KEY")
    if not api_key:
        print("ERROR: ACBL_API_KEY environment variable not set")
        return 1
    expiry_msg = expired_api_key_message(api_key)
    if expiry_msg:
        print(expiry_msg)
        return 1

    events_dir = acblPath.joinpath(args.events_dir)
    sessions_dir = acblPath.joinpath(args.sessions_dir)

    if not events_dir.exists():
        print(f"ERROR: events directory does not exist: {events_dir}")
        return 1

    print("=" * 70)
    print("ACBL Tournament Sessions Downloader (from sanctioned events)")
    print("=" * 70)
    print(f"Events dir:   {events_dir}")
    print(f"Sessions dir: {sessions_dir}")
    print(f"Start/end:    {args.start}/{args.end or 'all'}")
    print(f"Timeout:      {session_request_timeout(args.timeout)} (connect, read)")
    print(f"Sleep:        {args.sleep}s")
    print()

    session_ids = (
        sorted(set(args.session_id), reverse=True)
        if args.session_id
        else build_session_ids_from_sanctioned_events(events_dir)
    )
    if args.session_id:
        print(f"Using {len(session_ids):,} explicitly requested session ids")
    else:
        print(f"Derived {len(session_ids):,} unique sessions from sanctioned events")

    stats = download_tournament_sessions(
        session_ids=session_ids,
        api_key=api_key,
        output_dir=sessions_dir,
        full_monty=args.full_monty,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        sleep_seconds=args.sleep,
        starting_nfile=args.start,
        ending_nfile=args.end,
        skip_if_json_exists=args.skip_if_json_exists,
        skip_if_sql_exists=args.skip_if_sql_exists,
    )
    selected_end = args.end or len(session_ids)
    selected_ids = session_ids[args.start:selected_end]
    missing = audit_session_artifacts(
        selected_ids,
        sessions_dir,
        unavailable_ids=stats.unavailable_ids,
    )

    print()
    print("=" * 70)
    print(
        f"COMPLETE: written:{stats.written} skipped:{stats.skipped} "
        f"errors:{stats.errors} unavailable:{stats.unavailable} "
        f"missing:{len(missing)}"
    )
    print("=" * 70)
    if stats.failed():
        print(
            "FAILED: hard download errors remain; "
            "downstream rebuild is unsafe."
        )
        return 1
    if stats.unavailable:
        print(
            f"OK with {stats.unavailable} unavailable session(s) "
            f"(HTTP 400/404). They will be retried on the next run."
        )
    return 0


if __name__ == "__main__":
    from mlBridge import print_started, print_ended
    program_start_time = print_started()
    try:
        sys.exit(main())
    finally:
        print_ended(program_start_time)

