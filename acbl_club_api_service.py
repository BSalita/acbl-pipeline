"""Streamlit-free core for the unified ACBL results API.

Cache-first reads of club-results/<club_id>/details/<session_id>.data.json,
with Playwright fetches of my.acbl.org when the cache misses or refresh=True.
Tournament history and sessions use the official ACBL API. Fully augmented
postmortems resolve from historical monoliths, then an API-owned parquet cache,
then a headless live build.
Does not import the downloader CLI (it uses a process-global browser and
os._exit). Playwright helpers come from mlBridge.mlBridgeAcblLib.
"""

from __future__ import annotations

import io
import json
import math
import os
import pathlib
import re
import sys
import threading
import time
import unicodedata
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import duckdb
import pandas as pd
import polars as pl
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from rapidfuzz import fuzz

_APP_DIR = pathlib.Path(__file__).resolve().parent
_SRC_DIR = _APP_DIR.parent
load_dotenv(_SRC_DIR / "postmortem-acbl" / ".env")
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mlBridge.mlBridgeAcblLib import (  # noqa: E402
    _goto_with_diagnostics,
    _run_in_thread_with_new_loop,
    create_acbl_browser_context,
    create_club_dfs,
    extract_json_from_var_data,
    get_club_results_details_data_playwright,
    parse_acbl_events_from_html,
    resolve_acbl_browser_profile_dir,
)
from mlBridge.mlBridgeAcblPostmortemLib import (  # noqa: E402
    build_club_postmortem,
    build_tournament_postmortem,
)

ACBL_ORIGIN = "https://my.acbl.org"
DEFAULT_ROW_LIMIT = 200
MAX_ROW_LIMIT = 2000
NAV_SLEEP_SECONDS = 2.0
# Live pagination depth when parquet history already covers the past: only the
# boundary gap (games newer than the last stage-1b run) needs scraping.
GAP_SCRAPE_LIMIT = 50
CLOUDFLARE_HINT = (
    "my.acbl.org is behind Cloudflare. Warm the persistent Chrome profile "
    "(ACBL_BROWSER_PROFILE_DIR) with solve_acbl_postmortem.ps1 or "
    "python acbl_solve_challenge.py, then retry. Or open the page in a "
    "normal browser and POST the HTML to /handoff."
)
# A page the user's browser loaded. Prefer it over a headless refetch so a
# Cloudflare block does not throw away the listing they just retrieved.
HANDOFF_TTL_S = 6 * 60 * 60
HANDOFF_MAX_CHARS = 12_000_000
_HANDOFF_URL_RE = re.compile(
    r"^https://my\.acbl\.org/club-results/(?:(?P<kind>my-results|details)/)?(?P<id>\d+)/?$",
    re.IGNORECASE,
)

_CHATBOT_CACHE = _SRC_DIR / "postmortem-acbl" / "club-results"
_LOCAL_CACHE = _APP_DIR / "club-results"


def _default_cache_dir() -> pathlib.Path:
    env = os.environ.get("ACBL_CLUB_CACHE_DIR")
    if env:
        return pathlib.Path(env)
    if _CHATBOT_CACHE.is_dir():
        return _CHATBOT_CACHE
    return _LOCAL_CACHE


CACHE_DIR = _default_cache_dir()

# Master archive maintained by acbl_all.bat stage 1a (acbl_club_download_to_json.py):
# same <club>/details/<session>.data.json layout, ~1.3M sessions. Live scrapes are
# written here so stage 1a skips them later. Never enumerated wholesale; access is
# by direct path or per-club glob only.
ARCHIVE_CACHE_DIR = pathlib.Path(
    os.environ.get("ACBL_CLUB_ARCHIVE_DIR", "e:/bridge/data/acbl/club-results")
)
# Stage 1b outputs (acbl_club_json_to_sql.py): five normalized relationship
# tables support listings/lookups. Historical postmortems use the cleaned and
# fully augmented Stage 3c monolith instead of rebuilding from raw entities.
PARQUET_DIR = pathlib.Path(
    os.environ.get("ACBL_CLUB_PARQUET_DIR", "e:/bridge/data/acbl/club_results_parquet")
)
_AUGMENTED_FILENAME = "acbl_club_board_results_augmented.parquet"
_TOURNAMENT_AUGMENTED_FILENAME = (
    "acbl_tournament_board_results_augmented.parquet")


def _default_augmented_parquet_file() -> pathlib.Path:
    env = os.environ.get("ACBL_CLUB_AUGMENTED_PARQUET")
    if env:
        return pathlib.Path(env)
    deployed = PARQUET_DIR / _AUGMENTED_FILENAME
    if deployed.is_file():
        return deployed
    return pathlib.Path("e:/bridge/data/acbl") / _AUGMENTED_FILENAME


AUGMENTED_PARQUET_FILE = _default_augmented_parquet_file()


def _default_tournament_augmented_parquet_file() -> pathlib.Path:
    env = os.environ.get("ACBL_TOURNAMENT_AUGMENTED_PARQUET")
    if env:
        return pathlib.Path(env)
    deployed = PARQUET_DIR / _TOURNAMENT_AUGMENTED_FILENAME
    if deployed.is_file():
        return deployed
    return pathlib.Path("e:/bridge/data/acbl") / _TOURNAMENT_AUGMENTED_FILENAME


TOURNAMENT_AUGMENTED_PARQUET_FILE = (
    _default_tournament_augmented_parquet_file())
POSTMORTEM_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "ACBL_POSTMORTEM_API_CACHE_DIR",
        str(CACHE_DIR / "_postmortems"),
    )
)
TOURNAMENT_CACHE_DIR = pathlib.Path(
    os.environ.get(
        "ACBL_TOURNAMENT_CACHE_DIR",
        str(CACHE_DIR / "_tournaments"),
    )
)
ACBL_API_KEY = os.environ.get("ACBL_API_KEY", "").strip()
SINGLE_DUMMY_SAMPLE_COUNT = int(
    os.environ.get("ACBL_SINGLE_DUMMY_SAMPLE_COUNT", "40"))

# Ensure the cache dir exists up front: CACHE_ROOTS is fixed at import, and a
# missing dir would otherwise be excluded from reads even after writes create it.
CACHE_DIR.mkdir(parents=True, exist_ok=True)
POSTMORTEM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TOURNAMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_ROOTS: List[pathlib.Path] = [
    root for root in (CACHE_DIR, ARCHIVE_CACHE_DIR) if root.is_dir()
] or [CACHE_DIR]
# Write scrapes into the archive so stage 1a skips them -- but only when it is
# writable. In production the archive is a read-only mount of the OneDrive
# "recent slice" (see src/acbl-pipeline/u.bat), so writes go to the local cache instead.
WRITE_CACHE_DIR = (
    ARCHIVE_CACHE_DIR
    if ARCHIVE_CACHE_DIR.is_dir() and os.access(str(ARCHIVE_CACHE_DIR), os.W_OK)
    else CACHE_DIR
)

_PLAYER_INFO_CANDIDATES = (
    os.environ.get("ACBL_PLAYER_INFO_PARQUET"),
    str(_SRC_DIR / "bridgestats-acbl" / "data" / "acbl_player_info.parquet"),
    str(_APP_DIR / "data" / "acbl_player_info.parquet"),
)

_SCRAPE_LOCK = threading.Lock()
_SESSION_DF_CACHE: Dict[Tuple[str, float], Dict[str, pl.DataFrame]] = {}
_SESSION_DF_LOCK = threading.Lock()
_SESSION_DF_MAX = 8
_AUGMENTED_SESSION_CACHE: Dict[Tuple[str, float], bytes] = {}
_AUGMENTED_SESSION_LOCK = threading.Lock()
_AUGMENTED_SESSION_MAX = 8
_POSTMORTEM_BUILD_LOCK = threading.Lock()
_POSTMORTEM_CACHE_VERSION = 2
_TOURNAMENT_PARQUET_HITS = 0
_TOURNAMENT_PARQUET_MISSES = 0
_LAST_TOURNAMENT_API_SUCCESS: Optional[str] = None
_LAST_TOURNAMENT_PARQUET_QUERY: Optional[Dict[str, Any]] = None
_RECENT_TOURNAMENT_PARQUET_SESSIONS: List[str] = []
_CLUB_PARQUET_HITS = 0
_CLUB_PARQUET_MISSES = 0
_LAST_CLUB_PARQUET_QUERY: Optional[Dict[str, Any]] = None
_RECENT_CLUB_PARQUET_SESSIONS: List[str] = []
_CLUB_METRICS_FILE = POSTMORTEM_CACHE_DIR / "_club_parquet_metrics.json"
_CLUB_METRICS_LOCK = threading.Lock()


class ClubApiError(Exception):
    """Structured error for the FastAPI layer."""

    def __init__(self, detail: str, status_code: int = 400, hint: Optional[str] = None):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.hint = hint


def clamp_limit(limit: Optional[int], default: int = DEFAULT_ROW_LIMIT) -> int:
    if limit is None:
        return default
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, MAX_ROW_LIMIT))


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _as_pl(df: Any) -> pl.DataFrame:
    if isinstance(df, pl.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise ClubApiError(f"Unsupported dataframe type: {type(df)!r}", status_code=500)


def dataframe_to_table(
    df: Any,
    limit: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a frame to {columns, rows, row_count, truncated, meta}."""
    cap = clamp_limit(limit)
    frame = _as_pl(df)
    truncated = frame.height > cap
    frame = frame.head(cap)
    rows = [{k: _jsonable(v) for k, v in rec.items()} for rec in frame.to_dicts()]
    return {
        "columns": list(frame.columns),
        "rows": rows,
        "row_count": frame.height,
        "truncated": truncated,
        "meta": _jsonable(meta or {}),
    }


def rows_to_table(
    rows: List[Dict[str, Any]],
    limit: Optional[int] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cap = clamp_limit(limit)
    truncated = len(rows) > cap
    sliced = rows[:cap]
    columns: List[str] = []
    seen = set()
    for row in sliced:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return {
        "columns": columns,
        "rows": [{k: _jsonable(v) for k, v in row.items()} for row in sliced],
        "row_count": len(sliced),
        "truncated": truncated,
        "meta": _jsonable(meta or {}),
    }


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_mtime_iso(path: pathlib.Path) -> Optional[str]:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    except OSError:
        return None


def _newest_mtime_iso(paths: Iterable[pathlib.Path]) -> Optional[str]:
    newest: Optional[float] = None
    for path in paths:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return datetime.fromtimestamp(newest).isoformat(timespec="seconds")


def _parse_date_any(value: Any) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def _parquet_file(name: str) -> Optional[pathlib.Path]:
    path = PARQUET_DIR / f"{name}.parquet"
    return path if path.is_file() else None


def _parquet_scan(name: str) -> Optional[pl.LazyFrame]:
    path = _parquet_file(name)
    if path is None:
        return None
    return pl.scan_parquet(path)


def _collect_retry(lazy: pl.LazyFrame) -> Optional[pl.DataFrame]:
    """Collect with one retry: stage 1b rewrites the parquets in place, so a
    scan can transiently fail mid-replace.

    Streaming engine: identical results, but chunked execution roughly halves
    peak memory on the 5-table join queries (measured 2.7 GB vs 4.1 GB worst
    case), which matters inside the memory-capped acbl-club-api container."""
    for attempt in (0, 1):
        try:
            return lazy.collect(engine="streaming")
        except Exception:
            if attempt:
                return None
            time.sleep(0.5)
    return None


def _parquet_updated_iso() -> Optional[str]:
    path = _parquet_file("events")
    return _file_mtime_iso(path) if path else None


def _session_cache_path(club_id: str, session_id: str) -> pathlib.Path:
    return WRITE_CACHE_DIR / str(club_id) / "details" / f"{session_id}.data.json"


def _club_of_session_from_parquet(session_id: str) -> Optional[str]:
    events = _parquet_scan("events")
    if events is None:
        return None
    df = _collect_retry(
        events.filter(pl.col("id") == pl.lit(str(session_id))).select("club_id_number").head(1)
    )
    if df is None or df.is_empty():
        return None
    club = df[0, "club_id_number"]
    return str(club) if club else None


def find_session_cache(session_id: str) -> Optional[pathlib.Path]:
    sid = str(session_id)
    # Direct path via the events parquet (session -> club), avoiding globs
    # over the 2,883-club archive.
    club = _club_of_session_from_parquet(sid)
    if club:
        for root in CACHE_ROOTS:
            path = root / club / "details" / f"{sid}.data.json"
            if path.is_file():
                return path
    # Fallback glob per root, e.g. sessions newer than the last stage-1b run
    # that a live fetch already cached. Local root first (it is tiny).
    for root in CACHE_ROOTS:
        hits = list(root.glob(f"*/details/{sid}.data.json"))
        if hits:
            return hits[0]
    return None


def _iter_session_cache_files() -> Iterable[pathlib.Path]:
    """Session JSONs in the local cache only. The 1.3M-file archive is
    deliberately excluded: enumerate it never, access it by direct path."""
    if not CACHE_DIR.is_dir():
        return []
    return CACHE_DIR.glob("*/details/*.data.json")


def cached_session_count() -> int:
    return sum(1 for _ in _iter_session_cache_files())


def _player_info_path() -> Optional[pathlib.Path]:
    for candidate in _PLAYER_INFO_CANDIDATES:
        if not candidate:
            continue
        path = pathlib.Path(candidate)
        if path.is_file():
            return path
    return None


_DATASET_INFO_TTL_S = 15.0
_dataset_info_cache: Optional[Tuple[float, Dict[str, Any]]] = None


def dataset_info() -> Dict[str, Any]:
    global _dataset_info_cache
    now = time.monotonic()
    if _dataset_info_cache is not None and now - _dataset_info_cache[0] < _DATASET_INFO_TTL_S:
        return _dataset_info_cache[1]
    profile = resolve_acbl_browser_profile_dir()
    parquet = _player_info_path()
    clubs = sorted(
        p.name
        for p in CACHE_DIR.iterdir()
        if p.is_dir() and p.name.isdigit()
    ) if CACHE_DIR.is_dir() else []
    archive_clubs = (
        sum(1 for p in ARCHIVE_CACHE_DIR.iterdir() if p.is_dir() and p.name.isdigit())
        if ARCHIVE_CACHE_DIR.is_dir()
        else 0
    )
    info = {
        "cache_dir": str(CACHE_DIR),
        "cached_sessions": cached_session_count(),
        "cached_clubs": clubs,
        "archive_dir": str(ARCHIVE_CACHE_DIR) if ARCHIVE_CACHE_DIR.is_dir() else None,
        "archive_clubs": archive_clubs,
        "parquet_dir": str(PARQUET_DIR) if PARQUET_DIR.is_dir() else None,
        "parquet_updated_at": _parquet_updated_iso(),
        "club_augmented_parquet": (
            str(AUGMENTED_PARQUET_FILE)
            if AUGMENTED_PARQUET_FILE.is_file() else None
        ),
        "tournament_augmented_parquet": (
            str(TOURNAMENT_AUGMENTED_PARQUET_FILE)
            if TOURNAMENT_AUGMENTED_PARQUET_FILE.is_file() else None
        ),
        "postmortem_cache_dir": str(POSTMORTEM_CACHE_DIR),
        "cached_postmortems": sum(
            1 for _ in POSTMORTEM_CACHE_DIR.glob("*.parquet")),
        "tournament_api_configured": bool(ACBL_API_KEY),
        "write_cache_dir": str(WRITE_CACHE_DIR),
        "chrome_profile": str(profile) if profile else None,
        "player_info_parquet": str(parquet) if parquet else None,
        "note": (
            "Club and tournament postmortems resolve from historical augmented "
            "parquets, then the API parquet cache, then a headless live build. "
            "Both Streamlit and MCP clients use this API."
        ),
    }
    _dataset_info_cache = (now, info)
    return info


def club_dataset_info() -> Dict[str, Any]:
    """Club-specific parquet and cache availability."""
    path = AUGMENTED_PARQUET_FILE
    metrics = _club_metrics_snapshot()
    cached_files = list(POSTMORTEM_CACHE_DIR.glob("*.parquet"))
    cached_ids = sorted({
        item.stem.rsplit("-", 1)[0]
        for item in cached_files
        if item.stem.rsplit("-", 1)[0].isdigit()
    })
    return {
        "club_augmented_parquet": str(path) if path.is_file() else None,
        "club_parquet_size_bytes": path.stat().st_size if path.is_file() else None,
        "club_parquet_updated_at": (
            _file_mtime_iso(path) if path.is_file() else None
        ),
        "club_parquet_hits": metrics["hits"],
        "club_parquet_misses": metrics["misses"],
        "last_club_parquet_query": metrics["last_query"],
        "historical_parquet_sessions_recently_served": metrics[
            "recent_sessions"],
        "generated_club_postmortem_cache_count": len(cached_ids),
        "generated_club_postmortem_cache_ids": cached_ids,
        "raw_json_cached_sessions": cached_session_count(),
        "archive_dir": (
            str(ARCHIVE_CACHE_DIR) if ARCHIVE_CACHE_DIR.is_dir() else None
        ),
        "note": (
            "Historical session tables and postmortems are projected from the "
            "club augmented parquet. Raw JSON is only used for recent sessions "
            "that are absent from that parquet."
        ),
    }


def _club_metrics_snapshot() -> Dict[str, Any]:
    fallback = {
        "hits": _CLUB_PARQUET_HITS,
        "misses": _CLUB_PARQUET_MISSES,
        "last_query": _LAST_CLUB_PARQUET_QUERY,
        "recent_sessions": list(_RECENT_CLUB_PARQUET_SESSIONS),
    }
    if not _CLUB_METRICS_FILE.is_file():
        return fallback
    try:
        stored = json.loads(_CLUB_METRICS_FILE.read_text(encoding="utf-8"))
        return {
            "hits": int(stored.get("hits", fallback["hits"])),
            "misses": int(stored.get("misses", fallback["misses"])),
            "last_query": stored.get("last_query"),
            "recent_sessions": list(stored.get("recent_sessions") or []),
        }
    except (OSError, ValueError, TypeError):
        return fallback


def _record_club_parquet_query(session_id: str, hit: bool) -> None:
    global _CLUB_PARQUET_HITS, _CLUB_PARQUET_MISSES
    global _LAST_CLUB_PARQUET_QUERY
    with _CLUB_METRICS_LOCK:
        stored = _club_metrics_snapshot()
        if hit:
            _CLUB_PARQUET_HITS += 1
            stored["hits"] += 1
            sid = str(session_id)
            recent = stored["recent_sessions"]
            if sid in recent:
                recent.remove(sid)
            recent.insert(0, sid)
            del recent[20:]
            _RECENT_CLUB_PARQUET_SESSIONS[:] = recent
        else:
            _CLUB_PARQUET_MISSES += 1
            stored["misses"] += 1
        query = {
            "session_id": str(session_id),
            "hit": hit,
            "queried_at": _now_iso(),
        }
        _LAST_CLUB_PARQUET_QUERY = query
        stored["last_query"] = query
        temp_file = _CLUB_METRICS_FILE.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(stored, indent=2), encoding="utf-8")
        os.replace(temp_file, _CLUB_METRICS_FILE)


def tournament_dataset_info() -> Dict[str, Any]:
    """Tournament-specific parquet, cache, and live API status."""
    path = TOURNAMENT_AUGMENTED_PARQUET_FILE
    cache_files = list(TOURNAMENT_CACHE_DIR.glob("*.json"))
    session_caches = [
        item for item in cache_files if item.name.endswith(".session.json")
    ]
    listing_caches = [
        item for item in cache_files if item.name.endswith(".sessions.json")
    ]
    postmortem_caches = [
        item for item in POSTMORTEM_CACHE_DIR.glob("*.parquet")
        if "-" in item.stem.rsplit("-", 1)[0]
    ]
    latest_cache = max(
        (_file_mtime_iso(item) for item in cache_files),
        default=None,
    )
    return {
        "tournament_augmented_parquet": str(path) if path.is_file() else None,
        "tournament_parquet_size_bytes": (
            path.stat().st_size if path.is_file() else None
        ),
        "tournament_parquet_updated_at": (
            _file_mtime_iso(path) if path.is_file() else None
        ),
        "tournament_parquet_hits": _TOURNAMENT_PARQUET_HITS,
        "tournament_parquet_misses": _TOURNAMENT_PARQUET_MISSES,
        "last_tournament_parquet_query": _LAST_TOURNAMENT_PARQUET_QUERY,
        "recent_tournament_parquet_sessions": list(
            _RECENT_TOURNAMENT_PARQUET_SESSIONS),
        "cached_tournament_sessions": len(session_caches),
        "cached_tournament_player_listings": len(listing_caches),
        "cached_tournament_postmortems": len(postmortem_caches),
        "last_tournament_cache_update": latest_cache,
        "last_live_api_success": (
            _LAST_TOURNAMENT_API_SUCCESS or latest_cache
        ),
        "tournament_api_configured": bool(ACBL_API_KEY),
        "note": (
            "Tournament MCP reads historical augmented parquet first. "
            "Official ACBL API access is limited to uncached pair sessions; "
            "team and knockout sessions are reported as having no board results."
        ),
    }


def _record_tournament_parquet_query(session_id: str, hit: bool) -> None:
    global _TOURNAMENT_PARQUET_HITS, _TOURNAMENT_PARQUET_MISSES
    global _LAST_TOURNAMENT_PARQUET_QUERY
    if hit:
        _TOURNAMENT_PARQUET_HITS += 1
        sid = str(session_id)
        if sid in _RECENT_TOURNAMENT_PARQUET_SESSIONS:
            _RECENT_TOURNAMENT_PARQUET_SESSIONS.remove(sid)
        _RECENT_TOURNAMENT_PARQUET_SESSIONS.insert(0, sid)
        del _RECENT_TOURNAMENT_PARQUET_SESSIONS[20:]
    else:
        _TOURNAMENT_PARQUET_MISSES += 1
    _LAST_TOURNAMENT_PARQUET_QUERY = {
        "session_id": str(session_id),
        "hit": hit,
        "queried_at": _now_iso(),
    }


def _save_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _playwright_paginated_html(
    url: str,
    limit: int = 0,
    count_items: Optional[Callable[[str], int]] = None,
) -> Tuple[str, bool]:
    """Fetch url and follow DataTables 'Next' until last page or limit items.

    Returns (html, complete). complete is False when pagination stopped early
    because the item limit was reached, so callers can mark caches partial.
    """

    def _sync() -> Tuple[str, bool]:
        from playwright.sync_api import sync_playwright

        pages_html: List[str] = []
        complete = True
        with sync_playwright() as p:
            browser, context = create_acbl_browser_context(p, headless=True)
            page = context.new_page()
            try:
                response = _goto_with_diagnostics(page, url, verbose=False)
                if response is None or response.status != 200:
                    status = getattr(response, "status", None)
                    raise ClubApiError(
                        f"Failed to load {url} (status {status})",
                        status_code=502,
                        hint=CLOUDFLARE_HINT,
                    )
                page_num = 1
                while True:
                    try:
                        page.wait_for_selector(".dataTables_paginate, table", timeout=5000)
                    except Exception:
                        pass
                    html = page.content()
                    pages_html.append(html)
                    if limit > 0 and count_items is not None:
                        combined = "\n".join(pages_html)
                        if count_items(combined) >= limit:
                            complete = False
                            break
                    next_clicked = False
                    selectors = [
                        "a.paginate_button.next",
                        "#DataTables_Table_0_next",
                        'a.page-link:has-text("Next")',
                        'a[rel="next"]',
                        "a.next:not(.disabled)",
                    ]
                    for selector in selectors:
                        try:
                            next_button = page.locator(selector).first
                            if next_button.count() == 0 or not next_button.is_visible():
                                continue
                            class_attr = next_button.get_attribute("class") or ""
                            if "disabled" in class_attr:
                                continue
                            next_button.click()
                            page.wait_for_load_state("networkidle", timeout=60000)
                            time.sleep(0.5)
                            page_num += 1
                            next_clicked = True
                            break
                        except Exception:
                            continue
                    if not next_clicked:
                        break
            finally:
                browser.close()
        return "\n".join(pages_html), complete

    try:
        with _SCRAPE_LOCK:
            time.sleep(NAV_SLEEP_SECONDS)
            return _run_in_thread_with_new_loop(_sync)
    except ClubApiError:
        raise
    except Exception as exc:
        raise ClubApiError(
            f"Playwright fetch failed for {url}: {exc}",
            status_code=502,
            hint=CLOUDFLARE_HINT,
        ) from exc


def _count_detail_links(html: str) -> int:
    return len(set(re.findall(r"/club-results/details/(\d+)", html)))


def _handoff_is_fresh(path: pathlib.Path) -> bool:
    if not path.is_file():
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age <= HANDOFF_TTL_S


def _fresh_handoff_html(path: pathlib.Path) -> Optional[str]:
    if not _handoff_is_fresh(path):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if len(text) < 200:
        return None
    return text


def _club_handoff_path(club_id: str) -> pathlib.Path:
    return CACHE_DIR / str(club_id) / f"{club_id}.handoff.html"


def _player_handoff_path(player_id: str) -> pathlib.Path:
    return CACHE_DIR / "_players" / f"{player_id}.handoff.html"


def _session_handoff_marker(session_id: str) -> pathlib.Path:
    return CACHE_DIR / "_handoff" / f"{session_id}.session"


def _fresh_session_handoff_path(session_id: str) -> Optional[pathlib.Path]:
    marker = _session_handoff_marker(session_id)
    if not _handoff_is_fresh(marker):
        return None
    try:
        club_id = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not club_id:
        return None
    path = CACHE_DIR / club_id / "details" / f"{session_id}.data.json"
    return path if path.is_file() else None


def html_is_cloudflare_challenge(html: str) -> bool:
    """True when the body is the interstitial, not a loaded results page."""
    sample = html[:12000].lower()
    return any(
        marker in sample
        for marker in (
            "just a moment",
            "checking your browser",
            "cf-challenge",
            "_cf_chl_opt",
        )
    )


def classify_handoff_url(url: str) -> Tuple[str, str, str]:
    """Return kind, id, canonical URL for a club, player, or session page."""
    raw = str(url or "").strip().split("#", 1)[0].split("?", 1)[0]
    match = _HANDOFF_URL_RE.match(raw)
    if not match:
        raise ClubApiError(
            "url must be a my.acbl.org club, player, or session results page",
            status_code=400,
        )
    ident = match.group("id")
    token = (match.group("kind") or "").lower()
    if token == "my-results":
        kind = "player"
        canonical = f"{ACBL_ORIGIN}/club-results/my-results/{ident}"
    elif token == "details":
        kind = "session"
        canonical = f"{ACBL_ORIGIN}/club-results/details/{ident}"
    else:
        kind = "club"
        canonical = f"{ACBL_ORIGIN}/club-results/{ident}"
    return kind, ident, canonical


def session_details_from_html(html: str) -> Optional[Dict[str, Any]]:
    """Session JSON embedded in a details page, without launching a browser."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in (
        "result-details-combined-section",
        "result-details",
        "team-result-details",
    ):
        node = soup.find(tag)
        if node is None:
            continue
        raw = node.get("v-bind:data")
        if isinstance(raw, str) and raw:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    parsed = extract_json_from_var_data(html)
    return parsed if isinstance(parsed, dict) else None


def _club_games_from_handoff(
    club_id: str,
    parquet_rows: List[Dict[str, Any]],
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    cap: int,
) -> Optional[Dict[str, Any]]:
    html = _fresh_handoff_html(_club_handoff_path(club_id))
    if html is None:
        return None
    info, live_rows = parse_club_page(html, club_id)
    if not live_rows:
        return None
    for row in live_rows:
        row["listing_source"] = "browser handoff"
    rows = _filter_games_by_date(
        _merge_game_rows(live_rows, parquet_rows), start_date, end_date
    )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "source_url": f"{ACBL_ORIGIN}/club-results/{club_id}",
            "source": "browser-handoff",
            "cached": False,
            "complete": bool(parquet_rows),
            "fetched_at": _now_iso(),
            "club": info,
        },
    )


def _player_games_from_handoff(
    player_id: str,
    parquet_rows: List[Dict[str, Any]],
    cap: int,
) -> Optional[Dict[str, Any]]:
    html = _fresh_handoff_html(_player_handoff_path(player_id))
    if html is None:
        return None
    live_rows = parse_player_games_html(html, player_id)
    if not live_rows:
        return None
    rows = _merge_game_rows(live_rows, parquet_rows)
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "source_url": f"{ACBL_ORIGIN}/club-results/my-results/{player_id}",
            "source": "browser-handoff",
            "cached": False,
            "complete": bool(parquet_rows),
            "fetched_at": _now_iso(),
        },
    )


def ingest_browser_handoff(url: str, html: str) -> Dict[str, Any]:
    """Cache a results page loaded in the user's browser."""
    if not isinstance(html, str) or not html.strip():
        raise ClubApiError("html is empty", status_code=400)
    if len(html) > HANDOFF_MAX_CHARS:
        raise ClubApiError("html is too large", status_code=413)
    kind, ident, canonical = classify_handoff_url(url)
    if html_is_cloudflare_challenge(html):
        raise ClubApiError(
            "That page is still the Cloudflare challenge.",
            status_code=409,
            hint=(
                "Wait until the games or results are visible, then send "
                "the page again."
            ),
        )
    if kind == "club":
        return _ingest_club_handoff(ident, canonical, html)
    if kind == "player":
        return _ingest_player_handoff(ident, canonical, html)
    return _ingest_session_handoff(ident, canonical, html)


def _ingest_club_handoff(club_id: str, url: str, html: str) -> Dict[str, Any]:
    info, rows = parse_club_page(html, club_id)
    if not rows:
        raise ClubApiError(
            f"No club games found in the page for club {club_id}",
            status_code=422,
            hint="Send the page after the game table is visible.",
        )
    for row in rows:
        row["listing_source"] = "browser handoff"
    path = _club_handoff_path(club_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    _save_json(
        CACHE_DIR / club_id / f"{club_id}.games.json",
        {"info": info, "games": rows, "complete": False, "handoff": True},
    )
    name = info.get("club_name") or club_id
    return {
        "ok": True,
        "kind": "club",
        "url": url,
        "id": club_id,
        "rows": len(rows),
        "club_name": info.get("club_name"),
        "message": f"Saved {len(rows)} games for {name}. Ask again to use them.",
    }


def _ingest_player_handoff(player_id: str, url: str, html: str) -> Dict[str, Any]:
    rows = parse_player_games_html(html, player_id)
    if not rows:
        raise ClubApiError(
            f"No club games found in the page for player {player_id}",
            status_code=422,
            hint="Send the page after the game table is visible.",
        )
    path = _player_handoff_path(player_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    _save_json(
        CACHE_DIR / "_players" / f"{player_id}.games.json",
        {"games": rows, "complete": False, "handoff": True},
    )
    return {
        "ok": True,
        "kind": "player",
        "url": url,
        "id": player_id,
        "rows": len(rows),
        "message": (
            f"Saved {len(rows)} games for player {player_id}. Ask again to use them."
        ),
    }


def _ingest_session_handoff(session_id: str, url: str, html: str) -> Dict[str, Any]:
    data = session_details_from_html(html)
    if not data:
        raise ClubApiError(
            f"No session results were embedded in the page for {session_id}",
            status_code=422,
            hint="Send the page after the board results are visible.",
        )
    club_id = str(data.get("club_id_number") or data.get("club_id") or "unknown")
    path = CACHE_DIR / club_id / "details" / f"{session_id}.data.json"
    _save_json(path, data)
    marker = _session_handoff_marker(session_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(club_id, encoding="utf-8")
    label = data.get("event_name") or data.get("name") or session_id
    return {
        "ok": True,
        "kind": "session",
        "url": url,
        "id": session_id,
        "club_id": club_id,
        "rows": 1,
        "message": f"Saved session {session_id} ({label}). Ask again to use it.",
    }


def parse_club_page(html: str, club_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Club header plus game rows from a club-results HTML listing."""
    soup = BeautifulSoup(html, "html.parser")
    info: Dict[str, Any] = {"club_id": str(club_id)}
    club_info = soup.find("div", class_="col-md-8")
    if club_info:
        h1 = club_info.find("h1")
        h5 = club_info.find("h5")
        link = club_info.find("a")
        if h1:
            info["club_name"] = h1.get_text(strip=True)
        if h5:
            info["location"] = h5.get_text(strip=True)
        if link and link.get("href"):
            info["website"] = link["href"]

    rows: List[Dict[str, Any]] = []
    seen = set()
    for anchor in soup.find_all("a", href=re.compile(r"/club-results/details/\d+$")):
        href = anchor.get("href") or ""
        session_id = href.rstrip("/").split("/")[-1]
        if not session_id or session_id in seen:
            continue
        seen.add(session_id)
        tr = anchor.find_parent("tr")
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")] if tr else []
        date_text = cells[0] if cells else None
        rows.append(
            {
                "session_id": session_id,
                "date": date_text,
                "event_name": anchor.get_text(strip=True) or (cells[1] if len(cells) > 1 else None),
                "club_id": str(club_id),
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
            }
        )
    return info, rows


def parse_player_games_html(html: str, player_id: str) -> List[Dict[str, Any]]:
    parsed = parse_acbl_events_from_html(html, f"{ACBL_ORIGIN}/club-results/my-results/{player_id}")
    rows: List[Dict[str, Any]] = []
    soup = BeautifulSoup(html, "html.parser")
    by_id = {}
    for anchor in soup.find_all("a", href=re.compile(r"/club-results/details/\d+$")):
        sid = anchor["href"].rstrip("/").split("/")[-1]
        tr = anchor.find_parent("tr")
        tds = tr.find_all("td") if tr else []
        texts = [td.get_text(" ", strip=True) for td in tds]
        club_id = None
        club_name = None
        if tr:
            club_link = tr.find("a", href=re.compile(r"/club-results/\d+$"))
            if club_link and club_link.get("href"):
                m = re.search(r"/club-results/(\d+)$", club_link["href"])
                if m:
                    club_id = m.group(1)
                club_name = club_link.get_text(strip=True)
        # my-results row cells: Date, Club Name, Event, Session, Last Updated,
        # Score, mps, Color, Links, Personal Scores
        by_id[sid] = {
            "date": texts[0] if texts else None,
            "club_name": club_name or (texts[1] if len(texts) > 1 else None),
            "event": texts[2] if len(texts) > 2 else None,
            "session": texts[3] if len(texts) > 3 else None,
            "score": texts[5] if len(texts) > 5 else None,
            "club_id": club_id,
        }
    for event_id, (_url, detail_url, msg, club_id) in parsed.items():
        extra = by_id.get(str(event_id), {})
        rows.append(
            {
                "session_id": str(event_id),
                "date": extra.get("date"),
                "club_id": extra.get("club_id") or club_id,
                "club_name": extra.get("club_name"),
                "event": extra.get("event") or msg,
                "session": extra.get("session"),
                "score": extra.get("score"),
                "details_url": detail_url,
                "player_id": str(player_id),
                "listing_source": "downloaded from ACBL web",
            }
        )
    return rows


def parse_club_directory_html(html: str) -> List[Dict[str, Any]]:
    match = re.search(r"let\s+address\s*=\s*(\[.*?\]);", html, re.DOTALL)
    if match:
        try:
            address_data = json.loads(match.group(1))
            rows = []
            for item in address_data:
                if not isinstance(item, dict):
                    continue
                if item.get("type") and item.get("type") != "club":
                    continue
                if not item.get("club_id"):
                    continue
                rows.append(item)
            if rows:
                return rows
        except json.JSONDecodeError:
            pass
    match = re.search(r"clubs:JSON\.stringify\(\[([\d,\s]+)\]", html)
    if match:
        return [{"club_id": cid.strip()} for cid in match.group(1).split(",") if cid.strip()]
    return []


def _filter_games_by_date(
    rows: List[Dict[str, Any]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> List[Dict[str, Any]]:
    if not start_date and not end_date:
        return rows

    start = _parse_date_any(start_date)
    end = _parse_date_any(end_date)
    out = []
    for row in rows:
        parsed = _parse_date_any(row.get("date") or row.get("start_date"))
        if parsed is None:
            out.append(row)
            continue
        if start and parsed < start:
            continue
        if end and parsed > end:
            continue
        out.append(row)
    return out


def _games_from_cached_details(club_id: str) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for root in CACHE_ROOTS:
        details_dir = root / str(club_id) / "details"
        if not details_dir.is_dir():
            continue
        for path in details_dir.glob("*.data.json"):
            session_id = path.name.replace(".data.json", "")
            if session_id in seen:
                continue
            try:
                data = _load_json(path)
            except Exception:
                continue
            session_id = str(data.get("id") or session_id)
            seen.add(session_id)
            rows.append(
                {
                    "session_id": session_id,
                    "date": data.get("start_date"),
                    "event_name": data.get("name"),
                    "club_id": str(data.get("club_id_number") or club_id),
                    "club_name": data.get("club_name"),
                    "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
                    "listing_source": (
                        "archive JSON data file"
                        if root == ARCHIVE_CACHE_DIR
                        else "cached downloaded JSON data file"
                    ),
                }
            )
    rows.sort(key=_date_sort_key, reverse=True)
    return rows


def _date_sort_key(row: Dict[str, Any]) -> Tuple[date, str]:
    text = str(row.get("date") or "")
    return (_parse_date_any(text) or date.min, text)


def _player_games_from_cached_details(player_id: str) -> List[Dict[str, Any]]:
    pid = str(player_id)
    rows = []
    seen = set()
    for path in _iter_session_cache_files():
        try:
            data = _load_json(path)
        except Exception:
            continue
        if not _session_has_player(data, pid):
            continue
        session_id = str(data.get("id") or path.name.replace(".data.json", ""))
        if session_id in seen:
            continue
        seen.add(session_id)
        rows.append(
            {
                "session_id": session_id,
                "date": data.get("start_date"),
                "club_id": str(data.get("club_id_number") or path.parent.parent.name),
                "club_name": data.get("club_name"),
                "event": data.get("name"),
                "score": None,
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
                "player_id": pid,
                "listing_source": "cached downloaded JSON data file",
            }
        )
    rows.sort(key=_date_sort_key, reverse=True)
    return rows


def _session_has_player(data: Dict[str, Any], player_id: str) -> bool:
    pid = str(player_id)
    for session in data.get("sessions") or []:
        for section in session.get("sections") or []:
            for pair in section.get("pair_summaries") or []:
                for player in pair.get("players") or []:
                    if str(player.get("id_number") or "") == pid:
                        return True
    return False


def _extract_players_from_session(data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    sid = str(session_id or data.get("id") or "")
    club_id = str(data.get("club_id_number") or "")
    rows = []
    seen = set()
    for session in data.get("sessions") or []:
        for section in session.get("sections") or []:
            section_name = section.get("name")
            for pair in section.get("pair_summaries") or []:
                direction = pair.get("direction")
                pair_number = pair.get("pair_number")
                for player in pair.get("players") or []:
                    number = str(player.get("id_number") or "")
                    name = player.get("name")
                    key = (number, name, sid)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "player_number": number,
                            "player_name": name,
                            "session_id": sid,
                            "club_id": club_id,
                            "club_name": data.get("club_name"),
                            "section": section_name,
                            "direction": direction,
                            "pair_number": pair_number,
                            "city": player.get("city"),
                            "state": player.get("state"),
                        }
                    )
    return rows


def _club_games_from_parquet(club_id: str) -> List[Dict[str, Any]]:
    events = _parquet_scan("events")
    if events is None:
        return []
    df = _collect_retry(
        events.filter(pl.col("club_id_number") == pl.lit(str(club_id))).select(
            "id", "start_date", "name", "club_name", "type", "board_scoring_method"
        )
    )
    if df is None or df.is_empty():
        return []
    rows = [
        {
            "session_id": rec["id"],
            "date": rec["start_date"],
            "event_name": rec["name"],
            "club_id": str(club_id),
            "club_name": rec["club_name"],
            "event_type": rec["type"],
            "board_scoring_method": rec["board_scoring_method"],
            "details_url": f"{ACBL_ORIGIN}/club-results/details/{rec['id']}",
            "listing_source": "historical parquet",
        }
        for rec in df.to_dicts()
        if rec.get("id")
    ]
    # Event ids increase over time; start_date strings (MM/DD/YYYY) do not sort.
    rows.sort(key=lambda r: int(r["session_id"]), reverse=True)
    return rows


def _merge_game_rows(
    fresh: List[Dict[str, Any]], history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Fresh (live-scraped) rows first, then history rows they don't cover."""
    seen = {str(r.get("session_id")) for r in fresh}
    return fresh + [r for r in history if str(r.get("session_id")) not in seen]


def club_games(
    club_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    cid = str(club_id).strip()
    if not cid.isdigit():
        raise ClubApiError("club_id must be a numeric ACBL club number", status_code=400)
    cap = clamp_limit(limit)
    source_url = f"{ACBL_ORIGIN}/club-results/{cid}"
    html_cache = CACHE_DIR / cid / f"{cid}.html"
    games_cache = CACHE_DIR / cid / f"{cid}.games.json"
    cached = True
    complete = False
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    info: Dict[str, Any] = {"club_id": cid}

    parquet_rows = _club_games_from_parquet(cid)
    handed = _club_games_from_handoff(
        cid,
        parquet_rows,
        start_date=start_date,
        end_date=end_date,
        cap=cap,
    )
    if handed is not None:
        return handed
    if parquet_rows:
        rows = parquet_rows
        info["club_name"] = parquet_rows[0].get("club_name")
        source = "parquet"
        complete = True  # full history through the last stage-1b run
        fetched_at = _parquet_updated_iso()
    elif not refresh:
        if games_cache.is_file():
            payload = _load_json(games_cache)
            info = payload.get("info") or info
            rows = payload.get("games") or []
            for row in rows:
                row["listing_source"] = "cached game-list JSON (originally web)"
            complete = bool(payload.get("complete"))
            fetched_at = _file_mtime_iso(games_cache)
        elif html_cache.is_file() and html_cache.stat().st_size > 2048:
            # Legacy HTML cache; completeness unknown, so treat as partial.
            info, rows = parse_club_page(html_cache.read_text(encoding="utf-8"), cid)
            for row in rows:
                row["listing_source"] = "cached club-page HTML"
            fetched_at = _file_mtime_iso(html_cache)
        else:
            rows = _games_from_cached_details(cid)
            if rows:
                info["club_name"] = rows[0].get("club_name")
                fetched_at = _newest_mtime_iso(
                    p for root in CACHE_ROOTS for p in (root / cid / "details").glob("*.data.json")
                )

    # Scrape when asked to refresh, when nothing local answers, or when a
    # partial cache cannot satisfy a request for more rows than it holds.
    if refresh or not rows or (not complete and len(rows) < cap):
        scrape_limit = min(cap, GAP_SCRAPE_LIMIT) if parquet_rows else cap
        try:
            html, live_complete = _playwright_paginated_html(
                source_url, limit=scrape_limit, count_items=_count_detail_links
            )
        except ClubApiError:
            if not rows:
                raise
            # Scrape failed (e.g. Cloudflare); serve what the caches have.
        else:
            html_cache.parent.mkdir(parents=True, exist_ok=True)
            html_cache.write_text(html, encoding="utf-8")
            info, live_rows = parse_club_page(html, cid)
            for row in live_rows:
                row["listing_source"] = "downloaded from ACBL web"
            _save_json(games_cache, {"info": info, "games": live_rows, "complete": live_complete})
            # Parquet history extends a limit-truncated live listing.
            rows = _merge_game_rows(live_rows, parquet_rows)
            complete = live_complete or bool(parquet_rows)
            cached = False
            source = "live+parquet" if parquet_rows else "live"
            fetched_at = _now_iso()

    rows = _filter_games_by_date(rows, start_date, end_date)
    meta = {
        "source_url": source_url,
        "source": source,
        "cached": cached,
        "complete": complete,
        "fetched_at": fetched_at,
        "club": info,
    }
    if rows:
        # rows_to_table: parquet and live rows have slightly different keys.
        return rows_to_table(rows, limit=cap, meta=meta)
    return dataframe_to_table(
        pl.DataFrame(
            schema={"session_id": pl.Utf8, "date": pl.Utf8, "event_name": pl.Utf8, "club_id": pl.Utf8, "details_url": pl.Utf8}
        ),
        limit=cap,
        meta=meta,
    )


def club_info(club_id: str, refresh: bool = False) -> Dict[str, Any]:
    table = club_games(club_id, limit=1, refresh=refresh)
    inner_meta = table.get("meta") or {}
    info = inner_meta.get("club") or {"club_id": str(club_id)}
    if not info.get("club_name"):
        cached = _games_from_cached_details(str(club_id))
        if cached:
            info["club_name"] = cached[0].get("club_name")
    rows = [info]
    return rows_to_table(
        rows,
        limit=1,
        meta={
            "source_url": f"{ACBL_ORIGIN}/club-results/{club_id}",
            "cached": inner_meta.get("cached"),
            "fetched_at": inner_meta.get("fetched_at"),
        },
    )


def _player_games_from_parquet(player_id: str) -> List[Dict[str, Any]]:
    """Full club-game history for a player from the stage-1b parquets:
    players -> pair_summaries -> sections -> sessions -> events."""
    players = _parquet_scan("players")
    pairs = _parquet_scan("pair_summaries")
    sections = _parquet_scan("sections")
    sessions = _parquet_scan("sessions")
    events = _parquet_scan("events")
    if any(lf is None for lf in (players, pairs, sections, sessions, events)):
        return []
    lazy = (
        players.filter(pl.col("id_number") == pl.lit(str(player_id)))
        .select("pair_summary_id")
        .join(
            pairs.select(pl.col("id").alias("pair_summary_id"), "section_id", "percentage"),
            on="pair_summary_id",
        )
        .join(
            sections.select(pl.col("id").alias("section_id"), "session_id"),
            on="section_id",
        )
        .join(
            sessions.select(pl.col("id").alias("session_id"), "event_id", "game_date"),
            on="session_id",
        )
        .join(
            events.select(
                pl.col("id").alias("event_id"),
                pl.col("name").alias("event_name"),
                "club_id_number",
                "club_name",
            ),
            on="event_id",
        )
        .unique(subset="event_id", keep="first")
    )
    df = _collect_retry(lazy)
    if df is None or df.is_empty():
        return []
    rows = []
    for rec in df.to_dicts():
        eid = rec.get("event_id")
        if not eid:
            continue
        pct = rec.get("percentage")
        rows.append(
            {
                "session_id": str(eid),
                "date": str(rec.get("game_date") or "")[:10] or None,
                "club_id": rec.get("club_id_number"),
                "club_name": rec.get("club_name"),
                "event": rec.get("event_name"),
                "score": f"{pct}%" if pct not in (None, "") else None,
                "details_url": f"{ACBL_ORIGIN}/club-results/details/{eid}",
                "player_id": str(player_id),
                "listing_source": "historical parquet",
            }
        )
    rows.sort(key=lambda r: int(r["session_id"]), reverse=True)
    return rows


def player_games(
    player_id: str,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    pid = str(player_id).strip()
    if not pid.isdigit():
        raise ClubApiError("player_id must be a numeric ACBL player number", status_code=400)
    cap = clamp_limit(limit)
    source_url = f"{ACBL_ORIGIN}/club-results/my-results/{pid}"
    cache_file = CACHE_DIR / "_players" / f"{pid}.games.json"
    cached = True
    complete = False
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []
    refresh_error: Optional[str] = None
    refresh_attempts: List[Dict[str, Any]] = []

    parquet_rows = _player_games_from_parquet(pid)
    handed = _player_games_from_handoff(pid, parquet_rows, cap)
    if handed is not None:
        return handed
    if parquet_rows:
        rows = parquet_rows
        source = "parquet"
        complete = True  # full history through the last stage-1b run
        fetched_at = _parquet_updated_iso()
    elif not refresh:
        if cache_file.is_file():
            payload = _load_json(cache_file)
            if isinstance(payload, dict):
                rows = payload.get("games") or []
                complete = bool(payload.get("complete"))
            else:
                # Legacy cache: bare list, completeness unknown.
                rows = payload or []
            for row in rows:
                row["listing_source"] = "cached game-list JSON (originally web)"
            fetched_at = _file_mtime_iso(cache_file)
        if not rows:
            rows = _player_games_from_cached_details(pid)
            if rows:
                fetched_at = _newest_mtime_iso(_iter_session_cache_files())

    # Scrape when asked to refresh, when nothing local answers, or when a
    # partial cache cannot satisfy a request for more rows than it holds.
    if refresh or not rows or (not complete and len(rows) < cap):
        scrape_limit = min(cap, GAP_SCRAPE_LIMIT) if parquet_rows else cap
        html: Optional[str] = None
        live_complete = False
        last_error: Optional[ClubApiError] = None
        # A headed browser/profile startup can fail transiently even when the
        # Cloudflare clearance is valid. A stale fallback is especially harmful
        # here because the caller uses the first row as "latest game".
        attempts = 2 if refresh else 1
        for attempt in range(attempts):
            try:
                html, live_complete = _playwright_paginated_html(
                    source_url, limit=scrape_limit, count_items=_count_detail_links
                )
                refresh_attempts.append(
                    {"attempt": attempt + 1, "status": "success"}
                )
                last_error = None
                break
            except ClubApiError as exc:
                refresh_attempts.append(
                    {
                        "attempt": attempt + 1,
                        "status": "error",
                        "status_code": exc.status_code,
                        "detail": exc.detail,
                    }
                )
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(1.0)
        if html is None:
            if not rows and last_error is not None:
                raise last_error
            # Serve history, but report that the requested refresh failed.
            refresh_error = last_error.detail if last_error is not None else "Unknown scrape failure"
        else:
            live_rows = parse_player_games_html(html, pid)
            if live_rows:
                _save_json(cache_file, {"games": live_rows, "complete": live_complete})
                # Parquet history extends a limit-truncated live listing.
                rows = _merge_game_rows(live_rows, parquet_rows)
                complete = live_complete or bool(parquet_rows)
                cached = False
                source = "live+parquet" if parquet_rows else "live"
                fetched_at = _now_iso()

    if not rows:
        raise ClubApiError(
            f"No club games found for player {pid}",
            status_code=404,
            hint="Confirm the ACBL number, or generate cache by loading a postmortem for that player.",
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "source_url": source_url,
            "source": source,
            "cached": cached,
            "complete": complete,
            "fetched_at": fetched_at,
            "refresh_failed": refresh_error is not None,
            "refresh_error": refresh_error,
            "refresh_attempts": refresh_attempts,
        },
    )


def _fetch_session_json(
    session_id: str,
    refresh: bool = False,
    allow_live: bool = True,
) -> Tuple[Dict[str, Any], bool, pathlib.Path]:
    sid = str(session_id).strip()
    if not sid.isdigit():
        raise ClubApiError("session_id must be a numeric club event id", status_code=400)
    handoff_path = _fresh_session_handoff_path(sid)
    if handoff_path is not None:
        return _load_json(handoff_path), True, handoff_path
    cached_path = find_session_cache(sid)
    if cached_path is not None and not refresh:
        return _load_json(cached_path), True, cached_path
    if not allow_live:
        raise ClubApiError(
            f"Raw tables for session {sid} are absent from normalized session parquets and JSON archive",
            status_code=404,
            hint=(
                "Use the club postmortem boards, schema, or SQL tools for "
                "historical sessions in the augmented parquet."
            ),
        )

    url = f"{ACBL_ORIGIN}/club-results/details/{sid}"
    try:
        with _SCRAPE_LOCK:
            time.sleep(NAV_SLEEP_SECONDS)
            data = get_club_results_details_data_playwright(url, headless=True, verbose=False)
    except Exception as exc:
        raise ClubApiError(
            f"Failed to fetch session {sid}: {exc}",
            status_code=502,
            hint=CLOUDFLARE_HINT,
        ) from exc
    if not data:
        raise ClubApiError(
            f"No details JSON for session {sid} (team events are not supported)",
            status_code=404,
            hint="Pair matchpoint games embed var data = {...} on the details page.",
        )
    club_id = str(data.get("club_id_number") or data.get("club_id") or "unknown")
    path = _session_cache_path(club_id, sid)
    _save_json(path, data)
    return data, False, path


def session_raw(session_id: str, refresh: bool = False) -> Dict[str, Any]:
    """Verbatim details JSON for one session.

    Unlike the table endpoints this preserves the nested structure, which the
    postmortem app needs for create_club_dfs / merge_clean_augment_club_dfs.
    """
    data, _cached, _path = _fetch_session_json(session_id, refresh=refresh)
    return data


def session_raw_with_source(
    session_id: str, refresh: bool = False
) -> Tuple[Dict[str, Any], str]:
    """Raw session JSON plus a human-readable provenance label."""
    data, cached, path = _fetch_session_json(session_id, refresh=refresh)
    if not cached:
        source = "downloaded from ACBL web"
    elif path.is_relative_to(ARCHIVE_CACHE_DIR):
        source = "archive JSON data file"
    else:
        source = "cached downloaded JSON data file"
    return data, source


def _session_kind(session_id: str) -> str:
    return "club" if str(session_id).isdigit() else "tournament"


def _historical_augmented_parquet(
    session_id: str,
) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    sid = str(session_id)
    kind = _session_kind(sid)
    path = (
        AUGMENTED_PARQUET_FILE
        if kind == "club"
        else TOURNAMENT_AUGMENTED_PARQUET_FILE
    )
    if not path.is_file():
        return None

    key = (f"{path}:{sid}", path.stat().st_mtime)
    with _AUGMENTED_SESSION_LOCK:
        cached = _AUGMENTED_SESSION_CACHE.get(key)
    if cached is None:
        lazy = pl.scan_parquet(path)
        id_column = "event_id" if kind == "club" else "session_id"
        schema = lazy.collect_schema()
        if id_column not in schema.names():
            raise ClubApiError(
                f"{path.name} does not contain {id_column}",
                status_code=500,
            )
        # Preserve Parquet predicate pushdown. Casting the 81 GiB monolith's
        # event_id to String forced a full scan before applying the filter.
        id_value: Any = int(sid) if schema[id_column].is_integer() else sid
        frame = _collect_retry(
            lazy.filter(pl.col(id_column) == id_value))
        if frame is None:
            raise ClubApiError(
                f"Could not read historical postmortem {session_id}",
                status_code=503,
                hint="The augmented parquet may be updating; retry shortly.",
            )
        if frame.is_empty():
            if kind == "tournament":
                _record_tournament_parquet_query(sid, False)
            else:
                _record_club_parquet_query(sid, False)
            return None
        output = io.BytesIO()
        frame.write_parquet(output, compression="zstd")
        cached = output.getvalue()
        with _AUGMENTED_SESSION_LOCK:
            if len(_AUGMENTED_SESSION_CACHE) >= _AUGMENTED_SESSION_MAX:
                _AUGMENTED_SESSION_CACHE.pop(
                    next(iter(_AUGMENTED_SESSION_CACHE)))
            _AUGMENTED_SESSION_CACHE[key] = cached
    if kind == "tournament":
        _record_tournament_parquet_query(sid, True)
    else:
        _record_club_parquet_query(sid, True)
    return cached, {
        "source": f"historical {kind} augmented parquet",
        "source_tier": "historical",
        "source_file": path.name,
        "session_id": sid,
        "kind": kind,
        "fetched_at": _file_mtime_iso(path),
    }


def _postmortem_cache_file(session_id: str, player_id: str) -> pathlib.Path:
    safe_session = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))
    safe_player = re.sub(r"[^A-Za-z0-9_.-]", "_", str(player_id))
    return POSTMORTEM_CACHE_DIR / f"{safe_session}-{safe_player}.parquet"


def _validated_postmortem_cache(
    cache_file: pathlib.Path, player_id: str, session_id: str
) -> Optional[bytes]:
    """Normalize legacy float IDs and reject caches for the wrong player."""
    try:
        frame = pl.read_parquet(cache_file)
    except Exception:
        cache_file.unlink(missing_ok=True)
        return None
    if _session_kind(session_id) == "tournament":
        if "_api_cache_version" not in frame.columns or not (
            (
                frame.get_column("_api_cache_version").drop_nulls()
                == _POSTMORTEM_CACHE_VERSION
            ).all()
        ):
            # Version 1 tournament caches rotated already-given-first names
            # (Robert Salita -> Salita Robert). Rebuild once from the official
            # API rather than attempting an ambiguous name heuristic.
            cache_file.unlink(missing_ok=True)
            return None
    player_columns = [
        name for name in (f"Player_ID_{seat}" for seat in "NESW")
        if name in frame.columns
    ]
    if not player_columns:
        cache_file.unlink(missing_ok=True)
        return None
    has_legacy_float_ids = any(
        frame.filter(
            pl.col(name).cast(pl.String).str.ends_with(".0")
        ).height > 0
        for name in player_columns
    )
    frame = frame.with_columns(
        pl.col(name)
        .cast(pl.String)
        .str.strip_chars()
        .str.replace(r"\.0$", "")
        .alias(name)
        for name in player_columns
    )
    pid = str(player_id).strip()
    if not any(
        frame.filter(pl.col(name) == pid).height > 0
        for name in player_columns
    ):
        cache_file.unlink(missing_ok=True)
        return None
    if has_legacy_float_ids:
        temp_file = cache_file.with_suffix(
            f".{threading.get_ident()}.validated.tmp.parquet")
        frame.write_parquet(temp_file, compression="zstd")
        os.replace(temp_file, cache_file)
    return cache_file.read_bytes()


def _require_tournament_api_key() -> str:
    if not ACBL_API_KEY:
        raise ClubApiError(
            "The ACBL tournament API bearer token is not configured",
            status_code=503,
            hint="Set ACBL_API_KEY on the unified ACBL API process.",
        )
    return ACBL_API_KEY


def _tournament_api_get(
    url: str, params: Optional[Dict[str, Any]] = None
) -> requests.Response:
    global _LAST_TOURNAMENT_API_SUCCESS
    response = requests.get(
        url,
        params=params,
        headers={
            "accept": "application/json",
            "Authorization": f"Bearer {_require_tournament_api_key()}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        timeout=60,
    )
    if not response.ok:
        try:
            payload = response.json()
            detail = (
                payload.get("detail")
                or payload.get("message")
                or payload.get("error")
                or "; ".join(payload.get("messages") or [])
                or response.text[:500]
            )
        except ValueError:
            detail = response.text[:500]
        raise ClubApiError(
            f"ACBL tournament API returned HTTP {response.status_code}: {detail}",
            status_code=(
                response.status_code
                if 400 <= response.status_code < 500
                else 502
            ),
            hint="The official ACBL tournament API rejected this request.",
        )
    _LAST_TOURNAMENT_API_SUCCESS = _now_iso()
    return response


def tournament_session_data(
    session_id: str, refresh: bool = False
) -> Tuple[Dict[str, Any], str]:
    sid = str(session_id)
    cache_file = TOURNAMENT_CACHE_DIR / f"{sid}.session.json"
    if cache_file.is_file() and not refresh:
        return _load_json(cache_file), "tournament API JSON cache"
    response = _tournament_api_get(
        "https://api.acbl.org/v1/tournament/session",
        {"id": sid, "full_monty": 1},
    )
    data = response.json()
    _save_json(cache_file, data)
    return data, "live ACBL tournament API"


def historical_club_player_sessions(
    player_id: str, limit: Optional[int] = None
) -> Dict[str, Any]:
    """Club sessions for a player that are guaranteed present in the monolith."""
    pid = str(player_id).strip()
    if not pid.isdigit():
        raise ClubApiError(
            "player_id must be a numeric ACBL player number", status_code=400)
    path = AUGMENTED_PARQUET_FILE
    if not path.is_file():
        raise ClubApiError(
            "Club augmented parquet is unavailable", status_code=503)
    lazy = pl.scan_parquet(path)
    names = set(lazy.collect_schema().names())
    player_columns = [
        f"Player_ID_{seat}" for seat in "NESW"
        if f"Player_ID_{seat}" in names
    ]
    if "event_id" not in names or not player_columns:
        raise ClubApiError(
            "Club augmented parquet lacks session/player columns",
            status_code=500,
        )
    seat_matches = {
        seat: pl.col(f"Player_ID_{seat}").cast(pl.String) == pid
        for seat in "NESW"
        if f"Player_ID_{seat}" in names
    }
    player_match = pl.any_horizontal(list(seat_matches.values()))
    pair_score = (
        pl.when(
            seat_matches.get("N", pl.lit(False))
            | seat_matches.get("S", pl.lit(False))
        ).then(pl.col("Pct_NS"))
        .when(
            seat_matches.get("E", pl.lit(False))
            | seat_matches.get("W", pl.lit(False))
        ).then(pl.col("Pct_EW"))
        .otherwise(None)
    )
    expressions = [
        pl.col("event_id").cast(pl.String).alias("session_id"),
        pl.col("Date") if "Date" in names else pl.lit(None).alias("Date"),
        (
            pl.col("club_name")
            if "club_name" in names else pl.lit(None).alias("club_name")
        ),
        (
            pl.col("event_name")
            if "event_name" in names else pl.lit(None).alias("event_name")
        ),
        pair_score.alias("score"),
    ]
    frame = _collect_retry(
        lazy.filter(player_match)
        .select(expressions)
        .group_by("session_id")
        .agg(
            pl.col("Date").first(),
            pl.col("club_name").first(),
            pl.col("event_name").first(),
            (pl.col("score").mean() * 100).alias("score"),
        )
        .sort("Date", descending=True)
        .limit(clamp_limit(limit))
    )
    if frame is None:
        raise ClubApiError(
            "Could not read historical club postmortem sessions",
            status_code=503,
        )
    rows = [
        {
            "session_id": row["session_id"],
            "date": _jsonable(row.get("Date")),
            "club_name": row.get("club_name"),
            "event": row.get("event_name"),
            "session": None,
            "score": (
                f"{float(row['score']):.2f}%"
                if row.get("score") is not None else None
            ),
            "details_url": (
                f"{ACBL_ORIGIN}/club-results/details/{row['session_id']}"
            ),
            "listing_source": "historical club augmented parquet",
            "boards": True,
        }
        for row in frame.to_dicts()
    ]
    return rows_to_table(
        rows,
        limit=clamp_limit(limit),
        meta={
            "source": "historical club augmented parquet",
            "source_file": path.name,
            "fetched_at": _file_mtime_iso(path),
        },
    )


def _tournament_results_url(session_id: str) -> Optional[str]:
    """Public web2 recap. live.acbl.org NABC pages require login."""
    session = str(session_id).strip()
    if not session:
        return None
    nabc = re.match(r"^(NABC\d+)", session, re.IGNORECASE)
    if nabc:
        return (
            "https://web2.acbl.org/tournaments/Results/NABC/"
            f"{nabc.group(1).upper()}.HTM"
        )
    sanction = re.match(r"^(\d{7})(?:-|$)", session)
    if sanction:
        code = sanction.group(1)
        return (
            "https://web2.acbl.org/tournaments/results/"
            f"20{code[:2]}/{code[2:4]}/{code}.htm"
        )
    return None


def _historical_tournament_player_sessions(
    player_id: str,
) -> List[Dict[str, Any]]:
    path = TOURNAMENT_AUGMENTED_PARQUET_FILE
    if not path.is_file():
        return []
    pid = str(player_id)
    lazy = pl.scan_parquet(path)
    names = set(lazy.collect_schema().names())
    player_columns = [
        f"Player_ID_{seat}" for seat in "NESW"
        if f"Player_ID_{seat}" in names
    ]
    if not player_columns or "session_id" not in names:
        return []
    is_ns = (
        (pl.col("Player_ID_N").cast(pl.String) == pid)
        | (pl.col("Player_ID_S").cast(pl.String) == pid)
    )
    is_ew = (
        (pl.col("Player_ID_E").cast(pl.String) == pid)
        | (pl.col("Player_ID_W").cast(pl.String) == pid)
    )
    expressions = [
        pl.col("session_id"),
        pl.col("Date") if "Date" in names else pl.lit(None).alias("Date"),
        (
            pl.col("event_name")
            if "event_name" in names
            else pl.lit(None).alias("event_name")
        ),
        (
            pl.when(is_ns).then(pl.col("Pct_NS"))
            .when(is_ew).then(pl.col("Pct_EW"))
            .otherwise(None).alias("score")
        ),
    ]
    frame = _collect_retry(
        lazy.filter(is_ns | is_ew)
        .select(expressions)
        .group_by("session_id")
        .agg(
            pl.col("Date").first(),
            pl.col("event_name").first(),
            pl.col("score").mean(),
        )
        .sort("Date", descending=True)
    )
    if frame is None:
        return []
    return [
        {
            "session_id": row["session_id"],
            "date": _jsonable(row.get("Date")),
            "tournament_name": None,
            "event_name": row.get("event_name"),
            "session": None,
            "score": (
                f"{float(row['score']) * 100:.2f}%"
                if row.get("score") is not None else None
            ),
            "event_type": "Pairs",
            "boards": True,
            "unavailable_reason": None,
            "details_url": _tournament_results_url(str(row["session_id"])),
            "listing_source": "historical tournament augmented parquet",
        }
        for row in frame.to_dicts()
    ]


def _live_tournament_player_sessions(
    player_id: str,
    *,
    listing_source: str = "live ACBL tournament API",
) -> List[Dict[str, Any]]:
    url: Optional[str] = (
        "https://api.acbl.org/v1/tournament/player/history_query")
    params: Optional[Dict[str, Any]] = {
        "acbl_number": str(player_id),
        "page": 1,
        "page_size": 50,
        "start_date": "1900-01-01",
    }
    rows: List[Dict[str, Any]] = []
    while url:
        payload = _tournament_api_get(url, params).json()
        params = None
        for item in payload.get("data") or []:
            sid = str(item.get("session_id") or "")
            if not sid:
                continue
            event = item.get("event") or {}
            event_type = str(
                event.get("game_type")
                or item.get("score_score_type")
                or ""
            ).strip()
            has_boards = event_type.lower() in {
                "pairs", "matchpoints", "imp pairs", "board-a-match"
            }
            rows.append(
                {
                    "session_id": sid,
                    "date": item.get("date"),
                    "tournament_name": item.get("score_tournament_name"),
                    "event_name": item.get("score_event_name"),
                    "session": item.get("score_session_time_description"),
                    "score": item.get("percentage") if has_boards else None,
                    "event_type": event_type or None,
                    "boards": has_boards,
                    "unavailable_reason": (
                        None if has_boards else "no_board_results"
                    ),
                    "details_url": _tournament_results_url(sid),
                    "listing_source": listing_source,
                }
            )
        url = payload.get("next_page_url")
    return rows


def tournament_player_sessions(
    player_id: str,
    limit: Optional[int] = None,
    refresh: bool = False,
    tournament_name: Optional[str] = None,
) -> Dict[str, Any]:
    pid = str(player_id).strip()
    if not pid.isdigit():
        raise ClubApiError(
            "player_id must be a numeric ACBL player number", status_code=400)
    cap = clamp_limit(limit)
    historical = _historical_tournament_player_sessions(pid)
    cache_file = TOURNAMENT_CACHE_DIR / f"{pid}.sessions.json"
    live: List[Dict[str, Any]] = []
    live_error: Optional[str] = None
    if refresh:
        try:
            live = _live_tournament_player_sessions(pid)
            _save_json(cache_file, live)
        except (ClubApiError, requests.RequestException) as exc:
            live_error = str(exc)
            if cache_file.is_file():
                live = _load_json(cache_file)
    elif cache_file.is_file():
        live = _load_json(cache_file)
        for row in live:
            row["listing_source"] = "tournament API cache"
            if "boards" not in row:
                label = " ".join(
                    str(row.get(key) or "")
                    for key in ("event_type", "event_name")
                ).lower()
                row["boards"] = not any(
                    token in label
                    for token in ("swiss", "knockout", " ko", "teams")
                )
                row["unavailable_reason"] = (
                    None if row["boards"] else "no_board_results"
                )
                if not row["boards"]:
                    row["score"] = None
    if not historical and not live and not refresh:
        try:
            live = _live_tournament_player_sessions(pid)
            _save_json(cache_file, live)
        except (ClubApiError, requests.RequestException) as exc:
            live_error = str(exc)
    merged = {str(row["session_id"]): row for row in live}
    # Historical rows are known to have board-level postmortems and must not
    # be relabeled as live merely because the listing API also returned them.
    # The official player-history percentage is the published session result;
    # averaging board percentages from the augmented rows is only an estimate.
    for row in historical:
        sid = str(row["session_id"])
        official = merged.get(sid)
        if official is not None:
            row = {
                **row,
                "score": official.get("score"),
                "score_source": "official ACBL tournament player history",
                "parquet_board_average": row.get("score"),
                "tournament_name": official.get("tournament_name"),
                "session": official.get("session"),
                "event_type": official.get("event_type") or row.get("event_type"),
                "unavailable_reason": None,
            }
        else:
            row = {
                **row,
                "score_source": "mean board percentage from historical parquet",
            }
        merged[sid] = row
    rows = sorted(
        merged.values(),
        key=lambda row: (str(row.get("date") or ""), str(row["session_id"])),
        reverse=True,
    )
    name_query = str(tournament_name or "").strip()
    if name_query:
        matched_rows = []
        for row in rows:
            candidate = " ".join(
                str(row.get(key) or "")
                for key in ("tournament_name", "event_name", "event_type")
            )
            score = float(fuzz.WRatio(name_query, candidate))
            if score >= 55.0:
                matched_rows.append({**row, "match_score": round(score, 1)})
        rows = sorted(
            matched_rows,
            key=lambda row: (
                -float(row["match_score"]),
                str(row.get("date") or ""),
            ),
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "source": "live+historical" if live and historical else (
                "live/cache" if live else "historical parquet"),
            "refresh_failed": live_error is not None,
            "refresh_error": live_error,
            "fetched_at": _now_iso() if live else _file_mtime_iso(
                TOURNAMENT_AUGMENTED_PARQUET_FILE),
        },
    )


def _cached_tournament_listing(
    player_id: str, session_id: str
) -> Optional[Dict[str, Any]]:
    cache_file = TOURNAMENT_CACHE_DIR / f"{player_id}.sessions.json"
    if not cache_file.is_file():
        return None
    for row in _load_json(cache_file):
        if str(row.get("session_id")) == str(session_id):
            return row
    return None


def _with_tournament_listing_meta(
    meta: Dict[str, Any], player_id: str, session_id: str
) -> Dict[str, Any]:
    if _session_kind(session_id) != "tournament":
        return meta
    listing = _cached_tournament_listing(player_id, session_id)
    if listing is None:
        return meta
    return {
        **meta,
        "official_session_score": listing.get("score"),
        "score_source": "official ACBL tournament player history",
        "tournament_name": listing.get("tournament_name"),
        "event_name": listing.get("event_name"),
        "session": listing.get("session"),
    }


def session_augmented_parquet(
    session_id: str,
    player_id: Optional[str] = None,
    refresh: bool = False,
    allow_build: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    """Resolve a complete postmortem without involving Streamlit."""
    sid = str(session_id)
    historical = _historical_augmented_parquet(sid)
    if historical is not None:
        return historical

    pid = str(player_id or "").strip()
    if not pid:
        raise ClubApiError(
            f"Session {sid} is absent from historical augmented parquet",
            status_code=404,
            hint="Pass player_id so the API can build and cache a recent session.",
        )
    cache_file = _postmortem_cache_file(sid, pid)
    if cache_file.is_file() and not refresh:
        cached = _validated_postmortem_cache(cache_file, pid, sid)
        if cached is not None:
            return cached, {
                "source": "API-generated postmortem parquet cache",
                "source_tier": "cache",
                "source_file": cache_file.name,
                "session_id": sid,
                "player_id": pid,
                "kind": _session_kind(sid),
                "fetched_at": _file_mtime_iso(cache_file),
            }
    if not allow_build:
        raise ClubApiError(
            f"Session {sid} is absent from historical augmented parquet and API cache",
            status_code=404,
            hint="MCP postmortem requests do not scrape or build sessions live.",
        )

    with _POSTMORTEM_BUILD_LOCK:
        if cache_file.is_file() and not refresh:
            cached = _validated_postmortem_cache(cache_file, pid, sid)
            if cached is not None:
                return cached, {
                    "source": "API-generated postmortem parquet cache",
                    "source_tier": "cache",
                    "source_file": cache_file.name,
                    "session_id": sid,
                    "player_id": pid,
                    "kind": _session_kind(sid),
                    "fetched_at": _file_mtime_iso(cache_file),
                }
        try:
            if _session_kind(sid) == "club":
                frames, raw_meta = session_dataframes(sid, refresh=refresh)
                frame = build_club_postmortem(
                    frames,
                    pid,
                    single_dummy_sample_count=SINGLE_DUMMY_SAMPLE_COUNT,
                )
                live_source = raw_meta.get("source")
            else:
                listing = _cached_tournament_listing(pid, sid)
                if listing is not None:
                    event_type = str(
                        listing.get("event_type")
                        or listing.get("event_name")
                        or "team/knockout"
                    )
                    label = event_type.lower()
                    has_boards = listing.get("boards")
                    if has_boards is False or any(
                        token in label
                        for token in ("swiss", "knockout", " ko", "teams")
                    ):
                        raise ClubApiError(
                            (
                                f"Tournament session {sid} has no board results "
                                f"(event_type={event_type})"
                            ),
                            status_code=422,
                            hint="unavailable_reason=no_board_results",
                        )
                data, live_source = tournament_session_data(
                    sid, refresh=refresh)
                frame = build_tournament_postmortem(
                    data,
                    pid,
                    single_dummy_sample_count=SINGLE_DUMMY_SAMPLE_COUNT,
                )
        except ClubApiError:
            raise
        except Exception as exc:
            raise ClubApiError(
                f"Could not build postmortem for session {sid}: {exc}",
                status_code=422,
            ) from exc
        frame = frame.with_columns(
            pl.lit(_POSTMORTEM_CACHE_VERSION).alias("_api_cache_version"))
        temp_file = cache_file.with_suffix(".tmp.parquet")
        frame.write_parquet(temp_file, compression="zstd")
        os.replace(temp_file, cache_file)
        return cache_file.read_bytes(), {
            "source": f"headless API build from {live_source}",
            "source_tier": "live",
            "source_file": cache_file.name,
            "session_id": sid,
            "player_id": pid,
            "kind": _session_kind(sid),
            "fetched_at": _now_iso(),
        }


_POSTMORTEM_SEATS = (
    ("North", "NS", "S", "EW"),
    ("South", "NS", "N", "EW"),
    ("East", "EW", "W", "NS"),
    ("West", "EW", "E", "NS"),
)
_POSTMORTEM_BOARD_COLUMNS = [
    "Board", "Contract", "Declarer_Direction", "Declarer_ID", "Declarer_Name",
    "Result", "Tricks", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
    "MP_NS", "MP_EW", "MP_Top", "Par_NS",
    "Pair_Number_NS", "Pair_Number_EW", "PBN",
]
_POSTMORTEM_CONTEXT_COLUMNS = [
    "section_name", "Date", "Declarer_ID", "Declarer_Direction",
    "result", "tricks_taken",
    "Pair_Number_NS", "Pair_Number_EW",
    *(f"Player_ID_{seat}" for seat in "NESW"),
    *(f"Player_Name_{seat}" for seat in "NESW"),
]
_POSTMORTEM_DERIVED_DTYPES = {
    "Opponent_Pair_Direction": "String",
    "My_Section": "Boolean",
    "Our_Section": "Boolean",
    "My_Pair": "Boolean",
    "Our_Pair": "Boolean",
    "Boards_I_Played": "Boolean",
    "Boards_We_Played": "Boolean",
    "Our_Boards": "Boolean",
    "Boards_I_Declared": "Boolean",
    "Boards_Partner_Declared": "Boolean",
    "Boards_Opponent_Declared": "Boolean",
    "Boards_We_Declared": "Boolean",
}


def _historical_postmortem_lazy(
    session_id: str,
) -> Optional[Tuple[pl.LazyFrame, Dict[str, Any]]]:
    """Return a predicate-pushed historical session scan without collecting it."""
    sid = str(session_id)
    kind = _session_kind(sid)
    path = (
        AUGMENTED_PARQUET_FILE
        if kind == "club"
        else TOURNAMENT_AUGMENTED_PARQUET_FILE
    )
    if not path.is_file():
        return None
    lazy = pl.scan_parquet(path)
    id_column = "event_id" if kind == "club" else "session_id"
    schema = lazy.collect_schema()
    if id_column not in schema.names():
        raise ClubApiError(
            f"{path.name} does not contain {id_column}", status_code=500)
    id_value: Any = int(sid) if schema[id_column].is_integer() else sid
    return lazy.filter(pl.col(id_column) == id_value), {
        "source": f"historical {kind} augmented parquet",
        "source_tier": "historical",
        "source_file": path.name,
        "session_id": sid,
        "kind": kind,
        "fetched_at": _file_mtime_iso(path),
    }


def _given_name_first(value: Any) -> Any:
    if not isinstance(value, str) or "," not in value:
        return value
    last, first = value.split(",", 1)
    return f"{first.strip()} {last.strip()}".strip()


def _normalize_postmortem_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize API presentation fields without changing the monolith."""
    name_columns = [
        name for name in [
            *(f"Player_Name_{seat}" for seat in "NESW"),
            "Declarer_Name",
        ]
        if name in frame.columns
    ]
    expressions: List[pl.Expr] = [
        pl.col(name)
        .map_elements(_given_name_first, return_dtype=pl.String)
        .alias(name)
        for name in name_columns
    ]
    if "tricks_taken" in frame.columns and "Tricks" in frame.columns:
        raw_tricks = pl.col("tricks_taken").cast(pl.Int16, strict=False)
        expressions.append(
            pl.when(raw_tricks.is_not_null())
            .then(raw_tricks)
            .otherwise(pl.col("Tricks"))
            .alias("Tricks")
        )
    if "result" in frame.columns and "Result" in frame.columns:
        raw_result = pl.col("result").cast(pl.Int16, strict=False)
        expressions.append(
            pl.when(raw_result.is_not_null())
            .then(raw_result)
            .otherwise(pl.col("Result"))
            .alias("Result")
        )
    return frame.with_columns(expressions) if expressions else frame


def _scale_percentage_columns(frame: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for name in frame.columns:
        lowered = name.lower()
        if not (
            lowered.startswith("pct_")
            or lowered.endswith("_pct")
            or "percentage" in lowered
        ):
            continue
        column = frame.get_column(name)
        if not column.dtype.is_numeric():
            continue
        maximum = column.max()
        if maximum is not None and float(maximum) <= 1.0:
            expressions.append((pl.col(name) * 100).alias(name))
    return frame.with_columns(expressions) if expressions else frame


def _personalize_postmortem(
    frame: pl.DataFrame, player_id: str
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    pid = str(player_id).strip()
    for player_direction, pair_direction, partner_direction, opponent_pair_direction in _POSTMORTEM_SEATS:
        seat = player_direction[0]
        player_col = f"Player_ID_{seat}"
        if player_col not in frame.columns:
            continue
        rows = frame.filter(
            pl.col(player_col).cast(pl.String).str.strip_chars() == pid)
        if rows.is_empty():
            continue
        section_name = rows["section_name"][0]
        pair_number = rows[f"Pair_Number_{pair_direction}"][0]
        partner_id = rows[f"Player_ID_{partner_direction}"][0]
        frame = frame.with_columns(
            pl.lit(opponent_pair_direction).alias(
                "Opponent_Pair_Direction"),
            (pl.col("section_name") == section_name).alias("My_Section"),
        ).with_columns(
            pl.col("My_Section").alias("Our_Section"),
            (
                pl.col("My_Section")
                & (pl.col(f"Pair_Number_{pair_direction}") == pair_number)
            ).alias("My_Pair"),
        ).with_columns(
            pl.col("My_Pair").alias("Our_Pair"),
            pl.col("My_Pair").alias("Boards_I_Played"),
            pl.col("My_Pair").alias("Boards_We_Played"),
            pl.col("My_Pair").alias("Our_Boards"),
            (
                pl.col("My_Pair")
                & (pl.col("Declarer_ID").cast(pl.String) == pid)
            ).alias("Boards_I_Declared"),
            (
                pl.col("My_Pair")
                & (
                    pl.col("Declarer_ID").cast(pl.String)
                    == str(partner_id)
                )
            ).alias("Boards_Partner_Declared"),
            (
                pl.col("My_Pair")
                & pl.col("Declarer_Direction").is_in(
                    list(opponent_pair_direction))
            ).alias("Boards_Opponent_Declared"),
        ).with_columns(
            (
                pl.col("Boards_I_Declared")
                | pl.col("Boards_Partner_Declared")
            ).alias("Boards_We_Declared"),
        )
        return frame, {
            "player_id": pid,
            "player_name": rows[f"Player_Name_{seat}"][0],
            "player_direction": player_direction,
            "partner_id": partner_id,
            "partner_name": rows[f"Player_Name_{partner_direction}"][0],
            "partner_direction": partner_direction,
            "pair_direction": pair_direction,
            "opponent_pair_direction": opponent_pair_direction,
            "section_name": section_name,
            "pair_number": pair_number,
            "game_date": (
                str(frame["Date"].first())
                if "Date" in frame.columns else None
            ),
        }
    raise ClubApiError(
        f"Player {pid} was not found in session", status_code=404)


def postmortem_dataframe(
    session_id: str,
    player_id: str,
    refresh: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    payload, source_meta = session_augmented_parquet(
        session_id,
        player_id=player_id,
        refresh=refresh,
        allow_build=_session_kind(session_id) == "tournament",
    )
    frame = _normalize_postmortem_frame(
        pl.read_parquet(io.BytesIO(payload)))
    frame, player_meta = _personalize_postmortem(frame, player_id)
    meta = _with_tournament_listing_meta(
        {**source_meta, **player_meta}, player_id, session_id)
    return frame, meta


def postmortem_boards(
    session_id: str,
    player_id: str,
    only_my_boards: bool = True,
    columns: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    wanted = (
        [name.strip() for name in columns.split(",") if name.strip()]
        if columns else _POSTMORTEM_BOARD_COLUMNS
    )
    historical = _historical_postmortem_lazy(session_id)
    if historical is not None:
        lazy, source_meta = historical
        names = lazy.collect_schema().names()
        selected_for_scan = list(dict.fromkeys(
            name for name in [*wanted, *_POSTMORTEM_CONTEXT_COLUMNS]
            if name in names
        ))
        frame = _collect_retry(lazy.select(selected_for_scan))
        if frame is None:
            raise ClubApiError(
                f"Could not read historical postmortem {session_id}",
                status_code=503,
            )
        if frame.is_empty():
            frame, meta = postmortem_dataframe(
                session_id, player_id, refresh=refresh)
        else:
            frame = _normalize_postmortem_frame(frame)
            if source_meta.get("kind") == "tournament":
                _record_tournament_parquet_query(session_id, True)
            else:
                _record_club_parquet_query(session_id, True)
            frame, player_meta = _personalize_postmortem(frame, player_id)
            meta = _with_tournament_listing_meta(
                {**source_meta, **player_meta}, player_id, session_id)
    else:
        frame, meta = postmortem_dataframe(
            session_id, player_id, refresh=refresh)
    if only_my_boards:
        frame = frame.filter(pl.col("Boards_I_Played"))
    if not frame.is_empty():
        percent_columns = [
            name for name in ("Pct_NS", "Pct_EW")
            if name in frame.columns
            and frame.get_column(name).max() is not None
            and float(frame.get_column(name).max()) <= 1.0
        ]
        if percent_columns:
            frame = frame.with_columns(
                (pl.col(name) * 100).alias(name)
                for name in percent_columns
            )
    meta = {**meta, "percentage_scale": "0-100"}
    missing = [name for name in wanted if name not in frame.columns]
    selected = [name for name in wanted if name in frame.columns]
    if not selected:
        raise ClubApiError(
            f"None of the requested columns exist: {', '.join(missing)}",
            status_code=400,
        )
    if "Board" in frame.columns:
        frame = frame.sort("Board")
    result = dataframe_to_table(
        frame.select(selected), limit=limit, meta=meta)
    result["missing_columns"] = missing
    return result


def _postmortem_sql_macros(sql: str, meta: Dict[str, Any]) -> str:
    for macro, key in (
        ("{Player_Direction}", "player_direction"),
        ("{Partner_Direction}", "partner_direction"),
        ("{Pair_Direction}", "pair_direction"),
        ("{Opponent_Pair_Direction}", "opponent_pair_direction"),
    ):
        if meta.get(key) is not None:
            sql = sql.replace(macro, str(meta[key]))
    return sql


def postmortem_sql(
    session_id: str,
    player_id: str,
    sql: str,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    if not sql or not sql.strip():
        raise ClubApiError("sql is required", status_code=400)
    frame, meta = postmortem_dataframe(
        session_id, player_id, refresh=refresh)
    query = _postmortem_sql_macros(sql.strip().rstrip(";"), meta)
    if "from self" not in query.lower():
        query = f"FROM self {query}"
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        con.register("self", frame)
        result = con.execute(query).pl()
    except Exception as exc:
        raise ClubApiError(
            f"Postmortem SQL failed: {exc}", status_code=400) from exc
    finally:
        con.close()
    result = _scale_percentage_columns(result)
    table = dataframe_to_table(result, limit=limit, meta=meta)
    table["meta"]["percentage_scale"] = "0-100"
    table["sql"] = query
    return table


def postmortem_schema(
    session_id: str,
    player_id: str,
    pattern: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    historical = _historical_postmortem_lazy(session_id)
    if historical is not None:
        lazy, source_meta = historical
        schema = lazy.collect_schema()
        context_columns = [
            name for name in _POSTMORTEM_CONTEXT_COLUMNS
            if name in schema.names()
        ]
        context = _collect_retry(lazy.select(context_columns))
        if context is None:
            raise ClubApiError(
                f"Could not read historical postmortem {session_id}",
                status_code=503,
            )
        if context.is_empty():
            frame, meta = postmortem_dataframe(
                session_id, player_id, refresh=refresh)
            dtypes = dict(zip(
                frame.columns, (str(dtype) for dtype in frame.dtypes)))
        else:
            context = _normalize_postmortem_frame(context)
            if source_meta.get("kind") == "tournament":
                _record_tournament_parquet_query(session_id, True)
            else:
                _record_club_parquet_query(session_id, True)
            _context, player_meta = _personalize_postmortem(
                context, player_id)
            meta = _with_tournament_listing_meta(
                {**source_meta, **player_meta}, player_id, session_id)
            dtypes = {
                name: str(dtype)
                for name, dtype in zip(schema.names(), schema.dtypes())
            }
            dtypes.update(_POSTMORTEM_DERIVED_DTYPES)
    else:
        frame, meta = postmortem_dataframe(
            session_id, player_id, refresh=refresh)
        dtypes = dict(zip(
            frame.columns, (str(dtype) for dtype in frame.dtypes)))
    names = sorted(dtypes)
    if pattern:
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            raise ClubApiError(
                f"Invalid schema pattern: {exc}", status_code=400) from exc
        names = [name for name in names if regex.search(name)]
    cap = clamp_limit(limit, default=200)
    matched = len(names)
    names = names[:cap]
    return {
        "total_columns": len(dtypes),
        "matched_columns": matched,
        "truncated": matched > cap,
        "columns": {name: dtypes[name] for name in names},
        "meta": _jsonable(meta),
    }


def _standings_df(dfs: Dict[str, Any]) -> pl.DataFrame:
    pairs = _as_pl(dfs["pair_summaries"])
    players = _as_pl(dfs["players"])
    if "id" in pairs.columns and "pair_summary_id" not in pairs.columns:
        pairs = pairs.rename({"id": "pair_summary_id"})
    player_cols = [c for c in ("id_number", "name", "mp_total", "city", "state") if c in players.columns]
    aggs = []
    if "id_number" in player_cols:
        aggs.extend(
            [
                pl.first("id_number").alias("player_number_1"),
                pl.last("id_number").alias("player_number_2"),
            ]
        )
    if "name" in player_cols:
        aggs.extend(
            [
                pl.first("name").alias("player_name_1"),
                pl.last("name").alias("player_name_2"),
            ]
        )
    for col in ("mp_total", "city", "state"):
        if col in player_cols:
            aggs.extend(
                [
                    pl.first(col).alias(f"{col}_1"),
                    pl.last(col).alias(f"{col}_2"),
                ]
            )
    grouped = players.group_by("pair_summary_id").agg(aggs) if aggs else players
    return pairs.join(grouped, on="pair_summary_id", how="left")


_NESTED_SESSION_TABLES = {
    "hand_records": ["sessions", "hand_records"],
    "strat_place": ["sessions", "sections", "pair_summaries", "strat_place"],
    "sections": ["sessions", "sections"],
    "boards": ["sessions", "sections", "boards"],
    "pair_summaries": ["sessions", "sections", "pair_summaries"],
    "players": ["sessions", "sections", "pair_summaries", "players"],
    "board_results": ["sessions", "sections", "boards", "board_results"],
}

_PARQUET_SESSION_TABLES = (
    "events",
    "club",
    "sessions",
    "sections",
    "boards",
    "board_results",
    "pair_summaries",
    "players",
    "hand_records",
)

# Stage-1b stores every value as Utf8. Restore only the scalar types used by
# create_club_dfs/merge_clean_augment_club_dfs so joins behave like raw JSON.
_PARQUET_INT_COLUMNS = {
    "events": {"id", "club_id_number"},
    "club": {"id"},
    "sessions": {"id", "event_id", "number"},
    "sections": {"id", "session_id"},
    "boards": {"id", "section_id", "board_number"},
    "board_results": {
        "id", "board_id", "round_number", "table_number", "ns_pair", "ew_pair"
    },
    "pair_summaries": {"id", "section_id", "pair_number"},
    "players": {"id", "pair_summary_id"},
    "hand_records": {"id", "board", "hand_record_set_id"},
}
_PARQUET_FLOAT_COLUMNS = {
    "events": {"tb_count"},
    "board_results": {"ew_match_points", "ns_match_points"},
    "pair_summaries": {
        "score", "percentage", "adjustment", "handicap", "raw_score"
    },
    "players": {"mp_total"},
}


def _parquet_session_cache_key(session_id: str) -> Optional[Tuple[str, float]]:
    files = [_parquet_file(name) for name in _PARQUET_SESSION_TABLES]
    if any(path is None for path in files):
        return None
    return (
        f"parquet:{session_id}",
        max(path.stat().st_mtime for path in files if path is not None),
    )


def _restore_parquet_types(table: str, frame: pl.DataFrame) -> pl.DataFrame:
    expressions = []
    for col in _PARQUET_INT_COLUMNS.get(table, set()):
        if col in frame.columns:
            expressions.append(pl.col(col).cast(pl.Int64, strict=False))
    for col in _PARQUET_FLOAT_COLUMNS.get(table, set()):
        if col in frame.columns:
            expressions.append(pl.col(col).cast(pl.Float64, strict=False))
    return frame.with_columns(expressions) if expressions else frame


def _collect_parquet_rows(
    table: str, column: str, values: Iterable[Any]
) -> Optional[pl.DataFrame]:
    lazy = _parquet_scan(table)
    wanted = [str(value) for value in values if value not in (None, "")]
    if lazy is None or not wanted or column not in lazy.collect_schema().names():
        return None
    frame = _collect_retry(
        lazy.filter(pl.col(column).cast(pl.Utf8).is_in(wanted))
    )
    return _restore_parquet_types(table, frame) if frame is not None else None


def session_frames_from_parquet(
    session_id: str,
) -> Optional[Dict[str, pl.DataFrame]]:
    """Build create_club_dfs-compatible frames from normalized stage-1b data.

    ``session_id`` is the details-page/event id. Predicate pushdown keeps the
    multi-GB board and hand tables out of memory except for this one event.
    """
    if any(_parquet_file(name) is None for name in _PARQUET_SESSION_TABLES):
        return None
    sid = str(session_id)
    event = _collect_parquet_rows("events", "id", [sid])
    sessions = _collect_parquet_rows("sessions", "event_id", [sid])
    if event is None or event.is_empty() or sessions is None or sessions.is_empty():
        return None

    session_ids = sessions.get_column("id").drop_nulls().to_list()
    sections = _collect_parquet_rows("sections", "session_id", session_ids)
    if sections is None or sections.is_empty():
        return None
    section_ids = sections.get_column("id").drop_nulls().to_list()

    boards = _collect_parquet_rows("boards", "section_id", section_ids)
    pairs = _collect_parquet_rows("pair_summaries", "section_id", section_ids)
    if boards is None or boards.is_empty() or pairs is None or pairs.is_empty():
        return None
    board_ids = boards.get_column("id").drop_nulls().to_list()
    pair_ids = pairs.get_column("id").drop_nulls().to_list()
    board_results = _collect_parquet_rows("board_results", "board_id", board_ids)
    players = _collect_parquet_rows("players", "pair_summary_id", pair_ids)

    hand_set_ids = [
        value
        for value in sessions.get_column("hand_record_id").drop_nulls().to_list()
        if str(value).isdigit()
    ]
    hand_records = _collect_parquet_rows(
        "hand_records", "hand_record_set_id", hand_set_ids
    )
    club_ids = event.get_column("club_id_number").drop_nulls().to_list()
    club = _collect_parquet_rows("club", "id", club_ids)
    if any(
        frame is None or frame.is_empty()
        for frame in (board_results, players, hand_records, club)
    ):
        return None

    # create_club_dfs flattens hand_records.points into these four columns.
    # They are unused by the merge, but its compatibility drop expects them.
    if "points" in hand_records.columns:
        hand_records = hand_records.drop("points")
    hand_records = hand_records.with_columns(
        pl.lit(None, dtype=pl.Utf8).alias(f"points.{direction}")
        for direction in ("N", "E", "S", "W")
    )

    return {
        "event": event,
        "club": club,
        "sessions": sessions.sort("id"),
        "sections": sections.sort("id"),
        "boards": boards.sort(["section_id", "board_number"]),
        "board_results": board_results.sort(["board_id", "id"]),
        "pair_summaries": pairs.sort(["section_id", "id"]),
        "players": players.sort(["pair_summary_id", "id"]),
        "hand_records": hand_records.sort(["hand_record_set_id", "board"]),
        "strat_place": pl.DataFrame(),
    }


def _drop_nested_columns(frame: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for col in frame.columns:
        sample = frame[col].iloc[0] if len(frame) else None
        if isinstance(sample, (dict, list)):
            continue
        keep.append(col)
    return frame[keep] if keep else frame.iloc[:, 0:0]


def _normalize_path(data: Dict[str, Any], path: List[str]) -> pl.DataFrame:
    try:
        frame = pd.json_normalize(data, path)
    except Exception:
        return pl.DataFrame()
    if frame.empty:
        return pl.from_pandas(frame)
    frame = _drop_nested_columns(frame)
    try:
        return pl.from_pandas(frame)
    except Exception:
        return pl.from_pandas(frame.astype("string"))


def session_frames_from_json(data: Dict[str, Any]) -> Dict[str, pl.DataFrame]:
    """Build the same tables as create_club_dfs, without its nested-scalar crash."""
    try:
        raw = create_club_dfs(data)
        return {name: _as_pl(frame) for name, frame in raw.items()}
    except Exception:
        pass

    dfs: Dict[str, pl.DataFrame] = {}
    scalars = {k: v for k, v in data.items() if not isinstance(v, (dict, list))}
    dfs["event"] = pl.from_pandas(pd.DataFrame([scalars]).astype("string"))
    club = data.get("club")
    if isinstance(club, dict):
        club_df = pd.json_normalize(club)
        dfs["club"] = pl.from_pandas(_drop_nested_columns(club_df).astype("string"))
    sessions = data.get("sessions")
    if isinstance(sessions, list) and sessions:
        sess_df = pd.json_normalize(sessions, max_level=0)
        dfs["sessions"] = pl.from_pandas(_drop_nested_columns(sess_df).astype("string"))
    for name, path in _NESTED_SESSION_TABLES.items():
        dfs[name] = _normalize_path(data, path)
    if "board_results" not in dfs or dfs["board_results"].is_empty():
        raise ClubApiError(
            "Could not build tables for this session",
            status_code=422,
            hint="Session must be a pair game with embedded details JSON (Mitchell-style club games).",
        )
    return dfs


def session_frames_from_augmented_parquet(
    session_id: str,
) -> Optional[Tuple[Dict[str, pl.DataFrame], Dict[str, Any]]]:
    """Project useful historical session tables from the augmented monolith."""
    historical = _historical_postmortem_lazy(session_id)
    if historical is None:
        return None
    lazy, source_meta = historical
    if source_meta.get("kind") != "club":
        return None
    available = lazy.collect_schema().names()
    wanted = [
        "event_id", "session_id", "Date", "club_id", "club_name",
        "event_name", "section_name", "Board", "Contract", "Result", "result",
        "Tricks", "tricks_taken", "Declarer_Direction", "Declarer_ID",
        "Declarer_Name", "Score_NS", "Score_EW", "Pct_NS", "Pct_EW",
        "MP_NS", "MP_EW", "MP_Top", "Pair_Number_NS", "Pair_Number_EW",
        "PBN", "Vul", "Vul_NS", "Vul_EW", "DD_Tricks",
        *(f"Player_ID_{seat}" for seat in "NESW"),
        *(f"Player_Name_{seat}" for seat in "NESW"),
    ]
    selected = [name for name in wanted if name in available]
    frame = _collect_retry(lazy.select(selected))
    if frame is None:
        raise ClubApiError(
            f"Could not read historical session {session_id}",
            status_code=503,
        )
    if frame.is_empty():
        _record_club_parquet_query(session_id, False)
        return None
    _record_club_parquet_query(session_id, True)
    frame = _normalize_postmortem_frame(frame)

    event_columns = [
        name for name in (
            "event_id", "session_id", "Date", "club_id", "club_name",
            "event_name",
        )
        if name in frame.columns
    ]
    event = frame.select(event_columns).unique(maintain_order=True).head(1)
    sections = frame.select([
        name for name in ("event_id", "session_id", "section_name")
        if name in frame.columns
    ]).unique(maintain_order=True)
    boards = frame.select([
        name for name in ("event_id", "session_id", "section_name", "Board")
        if name in frame.columns
    ]).unique(maintain_order=True).sort(
        [name for name in ("section_name", "Board") if name in frame.columns]
    )

    pair_frames: List[pl.DataFrame] = []
    for direction, seats in (("NS", "NS"), ("EW", "EW")):
        required = [
            "section_name", f"Pair_Number_{direction}",
            *(f"Player_ID_{seat}" for seat in seats),
            *(f"Player_Name_{seat}" for seat in seats),
        ]
        if all(name in frame.columns for name in required):
            percentage = f"Pct_{direction}"
            pair_frames.append(
                frame.group_by(required, maintain_order=True).agg(
                    (pl.col(percentage).mean() * 100).alias("percentage")
                    if percentage in frame.columns
                    else pl.lit(None).alias("percentage")
                ).with_columns(
                    pl.lit(direction).alias("pair_direction")
                ).rename({
                    f"Pair_Number_{direction}": "pair_number",
                    f"Player_ID_{seats[0]}": "player_id_1",
                    f"Player_ID_{seats[1]}": "player_id_2",
                    f"Player_Name_{seats[0]}": "player_name_1",
                    f"Player_Name_{seats[1]}": "player_name_2",
                })
            )
    pair_summaries = (
        pl.concat(pair_frames, how="diagonal_relaxed")
        if pair_frames else pl.DataFrame()
    )
    hand_records = (
        frame.select(["Board", "PBN"]).unique(maintain_order=True)
        if {"Board", "PBN"}.issubset(frame.columns)
        else pl.DataFrame()
    )
    dfs = {
        "event": event,
        "sections": sections,
        "boards": boards,
        "board_results": frame,
        "pair_summaries": pair_summaries,
        "standings": pair_summaries,
        "hand_records": hand_records,
        "postmortem": frame,
    }
    return dfs, {
        **source_meta,
        "source": "historical augmented parquet projection",
        "cached": True,
        "table_semantics": (
            "Projected analytical tables; not the original raw JSON schema."
        ),
        "percentage_scale": "0-100 for projected pair summaries/standings",
    }


def session_dataframes(
    session_id: str,
    refresh: bool = False,
    allow_live: bool = True,
) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Any]]:
    if not refresh:
        parquet_key = _parquet_session_cache_key(str(session_id))
        with _SESSION_DF_LOCK:
            parquet_dfs = (
                _SESSION_DF_CACHE.get(parquet_key)
                if parquet_key is not None
                else None
            )
        if parquet_dfs is None:
            parquet_dfs = session_frames_from_parquet(session_id)
            if parquet_dfs is not None:
                try:
                    parquet_dfs["standings"] = _standings_df(parquet_dfs)
                except Exception:
                    pass
                if parquet_key is not None:
                    with _SESSION_DF_LOCK:
                        if len(_SESSION_DF_CACHE) >= _SESSION_DF_MAX:
                            _SESSION_DF_CACHE.pop(next(iter(_SESSION_DF_CACHE)))
                        _SESSION_DF_CACHE[parquet_key] = parquet_dfs
        if parquet_dfs is not None:
            event = parquet_dfs["event"].row(0, named=True)
            return parquet_dfs, {
                "source_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
                "source": "historical parquet",
                "cached": True,
                "fetched_at": _parquet_updated_iso(),
                "session_id": str(session_id),
                "club_id": str(event.get("club_id_number") or ""),
                "club_name": event.get("club_name"),
                "event_name": event.get("name"),
                "start_date": event.get("start_date"),
                "cache_file": None,
            }
        augmented = session_frames_from_augmented_parquet(session_id)
        if augmented is not None:
            return augmented

    data, cached, path = _fetch_session_json(
        session_id, refresh=refresh, allow_live=allow_live)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    key = (str(path), mtime)
    with _SESSION_DF_LOCK:
        if key in _SESSION_DF_CACHE and not refresh:
            dfs = _SESSION_DF_CACHE[key]
        else:
            try:
                dfs = session_frames_from_json(data)
            except ClubApiError:
                raise
            except Exception as exc:
                raise ClubApiError(
                    f"Could not build tables for session {session_id}: {exc}",
                    status_code=422,
                    hint="Session must be a pair game with embedded details JSON (Mitchell-style club games).",
                ) from exc
            try:
                dfs["standings"] = _standings_df(dfs)
            except Exception:
                pass
            if len(_SESSION_DF_CACHE) >= _SESSION_DF_MAX:
                _SESSION_DF_CACHE.pop(next(iter(_SESSION_DF_CACHE)))
            _SESSION_DF_CACHE[key] = dfs
    if not cached:
        session_source = "live"
    elif ARCHIVE_CACHE_DIR.is_dir() and str(path).startswith(str(ARCHIVE_CACHE_DIR)):
        session_source = "archive"
    else:
        session_source = "cache"
    meta = {
        "source_url": f"{ACBL_ORIGIN}/club-results/details/{session_id}",
        "source": session_source,
        "cached": cached,
        "fetched_at": _file_mtime_iso(path) if cached else _now_iso(),
        "session_id": str(session_id),
        "club_id": str(data.get("club_id_number") or ""),
        "club_name": data.get("club_name"),
        "event_name": data.get("name"),
        "start_date": data.get("start_date"),
        "cache_file": str(path),
    }
    return dfs, meta


def session_frames_payload(session_id: str, refresh: bool = False) -> Dict[str, Any]:
    """JSON transport for create_club_dfs-compatible session frames."""
    dfs, meta = session_dataframes(session_id, refresh=refresh)
    return {
        "tables": {
            name: [
                {key: _jsonable(value) for key, value in row.items()}
                for row in frame.to_dicts()
            ]
            for name, frame in dfs.items()
        },
        "meta": _jsonable(meta),
    }


def session_tables(
    session_id: str,
    refresh: bool = False,
    allow_live: bool = True,
) -> Dict[str, Any]:
    dfs, meta = session_dataframes(
        session_id, refresh=refresh, allow_live=allow_live)
    rows = [
        {
            "table": name,
            "row_count": frame.height,
            "columns": ", ".join(frame.columns),
            "column_count": len(frame.columns),
        }
        for name, frame in dfs.items()
    ]
    return rows_to_table(rows, limit=MAX_ROW_LIMIT, meta=meta)


def session_results(
    session_id: str,
    table: str = "board_results",
    columns: Optional[str] = None,
    limit: Optional[int] = None,
    refresh: bool = False,
    allow_live: bool = True,
) -> Dict[str, Any]:
    dfs, meta = session_dataframes(
        session_id, refresh=refresh, allow_live=allow_live)
    name = (table or "board_results").strip()
    if name not in dfs:
        raise ClubApiError(
            f"Unknown table '{name}' for session {session_id}",
            status_code=404,
            hint=f"Valid tables: {', '.join(sorted(dfs))}",
        )
    frame = dfs[name]
    if columns:
        wanted = [c.strip() for c in columns.split(",") if c.strip()]
        missing = [c for c in wanted if c not in frame.columns]
        if missing:
            raise ClubApiError(
                f"Unknown columns: {', '.join(missing)}",
                status_code=400,
                hint=f"Available columns: {', '.join(frame.columns)}",
            )
        frame = frame.select(wanted)
    meta = dict(meta)
    meta["table"] = name
    return dataframe_to_table(frame, limit=limit, meta=meta)


def session_sql(
    session_id: str,
    sql: str,
    limit: Optional[int] = None,
    refresh: bool = False,
    allow_live: bool = True,
) -> Dict[str, Any]:
    if not sql or not sql.strip():
        raise ClubApiError("sql is required", status_code=400)
    dfs, meta = session_dataframes(
        session_id, refresh=refresh, allow_live=allow_live)
    cap = clamp_limit(limit)
    query = sql.strip().rstrip(";")
    con = duckdb.connect(config={"enable_external_access": "false"})
    try:
        for name, frame in dfs.items():
            con.register(name, frame)
        result = con.execute(query).pl()
    except Exception as exc:
        raise ClubApiError(
            f"SQL failed: {exc}",
            status_code=400,
            hint=f"Registered tables: {', '.join(sorted(dfs))}",
        ) from exc
    finally:
        con.close()
    meta = dict(meta)
    meta["sql"] = query
    table = dataframe_to_table(result, limit=cap, meta=meta)
    table["sql"] = query
    return table


def club_list(query: Optional[str] = None, limit: Optional[int] = None, refresh: bool = False) -> Dict[str, Any]:
    cap = clamp_limit(limit)
    cache_file = CACHE_DIR / "_clubs.json"
    source_url = f"{ACBL_ORIGIN}/club-results"
    cached = True
    source = "cache"
    fetched_at: Optional[str] = None
    rows: List[Dict[str, Any]] = []

    if not refresh:
        events = _parquet_scan("events")
        if events is not None:
            df = _collect_retry(
                events.group_by("club_id_number")
                .agg(
                    pl.col("club_name").last().alias("club_name"),
                    pl.len().alias("events"),
                )
                .sort("club_id_number")
            )
            if df is not None and not df.is_empty():
                rows = [
                    {
                        "club_id": rec.get("club_id_number"),
                        "club_name": rec.get("club_name"),
                        "events": rec.get("events"),
                        "type": "club",
                    }
                    for rec in df.to_dicts()
                    if rec.get("club_id_number")
                ]
                source = "parquet"
                fetched_at = _parquet_updated_iso()
        if not rows and cache_file.is_file():
            rows = _load_json(cache_file)
            fetched_at = _file_mtime_iso(cache_file)
        if not rows and CACHE_DIR.is_dir():
            for club_dir in sorted(CACHE_DIR.iterdir()):
                if not (club_dir.is_dir() and club_dir.name.isdigit()):
                    continue
                games = _games_from_cached_details(club_dir.name)
                rows.append(
                    {
                        "club_id": club_dir.name,
                        "club_name": games[0].get("club_name") if games else None,
                        "cached_sessions": len(games),
                        "type": "club",
                    }
                )
            if rows:
                fetched_at = _newest_mtime_iso(_iter_session_cache_files())

    if refresh or not rows:
        html, _complete = _playwright_paginated_html(source_url, limit=0)
        rows = parse_club_directory_html(html)
        if not rows:
            raise ClubApiError(
                "Could not parse club directory from my.acbl.org/club-results",
                status_code=502,
                hint=CLOUDFLARE_HINT,
            )
        _save_json(cache_file, rows)
        cached = False
        source = "live"
        fetched_at = _now_iso()

    q = (query or "").strip().lower()
    if q:
        filtered = []
        for row in rows:
            blob = " ".join(str(v) for v in row.values() if v is not None).lower()
            score = 100.0 if q in blob else float(fuzz.WRatio(q, blob))
            if score >= 55.0:
                matched = dict(row)
                matched["match_score"] = round(score, 1)
                filtered.append(matched)
        rows = sorted(
            filtered,
            key=lambda row: (
                -float(row["match_score"]),
                str(row.get("club_name") or ""),
            ),
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={"source_url": source_url, "source": source, "cached": cached, "fetched_at": fetched_at},
    )


def _normalize_fuzzy_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _fuzzy_text_score(candidate: object, query: object) -> float:
    haystack = _normalize_fuzzy_text(candidate)
    needle = _normalize_fuzzy_text(query)
    if not haystack or not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    scores = [SequenceMatcher(None, needle, haystack).ratio()]
    tokens = haystack.split()
    width = max(1, len(needle.split()))
    for start in range(len(tokens)):
        window = " ".join(tokens[start : start + width])
        scores.append(SequenceMatcher(None, needle, window).ratio())
    return max(scores)


def _rank_player_name_rows(
    rows: List[Dict[str, Any]],
    query: str,
    *,
    min_score: float = 0.72,
    drop_below: bool = False,
) -> List[Dict[str, Any]]:
    ranked: List[tuple[float, int, int, Dict[str, Any]]] = []
    needle = _normalize_fuzzy_text(query)
    for row in rows:
        name = str(row.get("player_name") or "")
        score = _fuzzy_text_score(name, query)
        last = _normalize_fuzzy_text(name).split()[-1] if name else ""
        if drop_below and score < min_score and needle not in last:
            continue
        ranked.append(
            (
                score,
                1 if last == needle else 0,
                1 if needle and last.startswith(needle) else 0,
                {**row, "match_score": round(score * 100.0, 1)},
            )
        )
    ranked.sort(key=lambda item: (-item[0], -item[1], -item[2], str(item[3].get("player_name") or "")))
    return [item[3] for item in ranked]


def _player_lookup_from_parquet(
    query: str, by_number: bool, club_id: Optional[str], limit: int
) -> List[Dict[str, Any]]:
    """Name <-> number lookup against the stage-1b players parquet
    (151k players, all club sessions), optionally scoped to one club."""
    players = _parquet_scan("players")
    if players is None:
        return []
    if by_number:
        base = players.filter(pl.col("id_number") == pl.lit(query))
    else:
        tokens = [token for token in query.lower().replace(",", " ").split() if token]
        name_expr = pl.lit(True)
        for token in tokens:
            name_expr &= (
                pl.col("name")
                .str.to_lowercase()
                .str.contains(token, literal=True)
            )
        base = players.filter(name_expr)
    base = base.select("id_number", "name", "city", "state", "mp_total", "pair_summary_id", "transaction_date")
    if club_id:
        pairs = _parquet_scan("pair_summaries")
        sections = _parquet_scan("sections")
        sessions = _parquet_scan("sessions")
        events = _parquet_scan("events")
        if any(lf is None for lf in (pairs, sections, sessions, events)):
            return []
        base = (
            base.join(pairs.select(pl.col("id").alias("pair_summary_id"), "section_id"), on="pair_summary_id")
            .join(sections.select(pl.col("id").alias("section_id"), "session_id"), on="section_id")
            .join(sessions.select(pl.col("id").alias("session_id"), "event_id"), on="session_id")
            .join(events.select(pl.col("id").alias("event_id"), "club_id_number"), on="event_id")
            .filter(pl.col("club_id_number") == pl.lit(str(club_id)))
        )
    lazy = (
        base.sort("transaction_date")
        .group_by("id_number")
        .agg(
            pl.col("name").last(),
            pl.col("city").last(),
            pl.col("state").last(),
            pl.col("mp_total").last(),
            pl.len().alias("club_sessions"),
        )
        .sort("name")
        .head(limit)
    )
    df = _collect_retry(lazy)
    if df is None or df.is_empty():
        if by_number or club_id:
            return []
        return _fuzzy_player_lookup_from_parquet(query, limit)
    rows = [
        {
            "player_number": rec.get("id_number"),
            "player_name": _given_name_first(rec.get("name")),
            "city": rec.get("city"),
            "state": rec.get("state"),
            "mp_total": rec.get("mp_total"),
            "club_sessions": rec.get("club_sessions"),
            "club_id": str(club_id) if club_id else None,
            "source": "club_results_parquet",
        }
        for rec in df.to_dicts()
    ]
    if by_number:
        return rows
    return _rank_player_name_rows(rows, query)


def _fuzzy_player_lookup_from_parquet(
    query: str, limit: int
) -> List[Dict[str, Any]]:
    """Return typo-tolerant name matches from a bounded candidate set."""
    players = _parquet_scan("players")
    if players is None:
        return []
    tokens = [
        token for token in query.lower().replace(",", " ").split()
        if token
    ]
    candidate_expr = pl.lit(False)
    for token in tokens:
        fragments = (
            {token[index:index + 3] for index in range(len(token) - 2)}
            if len(token) >= 3 else {token}
        )
        for fragment in fragments:
            candidate_expr |= (
                pl.col("name")
                .str.to_lowercase()
                .str.contains(fragment, literal=True)
            )
    lazy = (
        players.filter(candidate_expr)
        .select(
            "id_number", "name", "city", "state", "mp_total",
            "transaction_date",
        )
        .sort("transaction_date")
        .group_by("id_number")
        .agg(
            pl.col("name").last(),
            pl.col("city").last(),
            pl.col("state").last(),
            pl.col("mp_total").last(),
            pl.len().alias("club_sessions"),
        )
    )
    frame = _collect_retry(lazy)
    if frame is None or frame.is_empty():
        return []
    rows = []
    for rec in frame.to_dicts():
        player_number = str(rec.get("id_number") or "").strip()
        if not player_number.isdigit():
            continue
        rows.append({
            "player_number": player_number,
            "player_name": _given_name_first(rec.get("name")),
            "city": rec.get("city"),
            "state": rec.get("state"),
            "mp_total": rec.get("mp_total"),
            "club_sessions": rec.get("club_sessions"),
            "club_id": None,
            "source": "club_results_parquet_fuzzy",
        })
    return _rank_player_name_rows(rows, query, drop_below=True)[:limit]


def _lookup_from_parquet(query: str, by_number: bool, limit: int) -> List[Dict[str, Any]]:
    path = _player_info_path()
    if path is None:
        return []
    df = pl.read_parquet(path)
    number_col = next((c for c in ("acbl_number", "player_number", "id_number") if c in df.columns), None)
    name_cols = [c for c in ("player_name", "name", "last_name", "first_name") if c in df.columns]
    if by_number and number_col:
        df = df.filter(pl.col(number_col).cast(pl.Utf8) == query)
    elif name_cols:
        expr = None
        for col in name_cols:
            piece = pl.col(col).cast(pl.Utf8).str.to_lowercase().str.contains(query.lower(), literal=True)
            expr = piece if expr is None else (expr | piece)
        df = df.filter(expr)
    else:
        return []
    keep = [c for c in df.columns if not str(c).startswith("mp_")]
    df = df.select(keep).head(limit)
    rows = []
    for rec in df.to_dicts():
        rec["source"] = "parquet"
        rows.append(rec)
    return rows


def player_lookup(
    query: str,
    session_id: Optional[str] = None,
    club_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    q = (query or "").strip()
    if not q:
        raise ClubApiError("q is required (player name or ACBL number)", status_code=400)
    cap = clamp_limit(limit)
    by_number = q.isdigit()
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    cached = True

    if session_id:
        data, cached, _path = _fetch_session_json(session_id, refresh=False)
        rows.extend(_extract_players_from_session(data, session_id))
        sources.append(f"session:{session_id}{' (cache)' if cached else ''}")
    else:
        parquet_rows = _player_lookup_from_parquet(q, by_number=by_number, club_id=club_id, limit=cap)
        if parquet_rows:
            rows.extend(parquet_rows)
            sources.append("club_results_parquet")
        else:
            files: List[pathlib.Path]
            if club_id:
                files = [
                    p
                    for root in CACHE_ROOTS
                    for p in (root / str(club_id) / "details").glob("*.data.json")
                ]
            else:
                files = list(_iter_session_cache_files())
            for path in files:
                try:
                    data = _load_json(path)
                except Exception:
                    continue
                sid = str(data.get("id") or path.name.replace(".data.json", ""))
                rows.extend(_extract_players_from_session(data, sid))
            sources.append("session-cache")

    if by_number:
        rows = [r for r in rows if str(r.get("player_number") or "") == q]
    elif session_id:
        tokens = [token for token in q.lower().replace(",", " ").split() if token]
        rows = [
            r for r in rows
            if all(
                token in str(r.get("player_name") or "").lower()
                for token in tokens
            )
        ]

    if not rows:
        parquet_rows = _lookup_from_parquet(q, by_number=by_number, limit=cap)
        if parquet_rows:
            sources.append("player_info_parquet")
            for rec in parquet_rows:
                number = str(rec.get("acbl_number") or rec.get("player_number") or rec.get("id_number") or "")
                name = rec.get("player_name") or rec.get("name") or rec.get("last_name")
                rows.append(
                    {
                        "player_number": number,
                        "player_name": name,
                        "session_id": None,
                        "club_id": rec.get("club"),
                        "source": "parquet",
                    }
                )

    if not rows:
        raise ClubApiError(
            f"No players matching '{q}'",
            status_code=404,
            hint="Pass session_id for a game roster, club_id to scan that club's cached sessions, or configure ACBL_PLAYER_INFO_PARQUET.",
        )
    return rows_to_table(
        rows,
        limit=cap,
        meta={
            "query": q,
            "by_number": by_number,
            "sources": sources,
            "cached": cached,
            "fetched_at": _now_iso(),
        },
    )
