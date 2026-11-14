"""
Find the shortest download delay that does not get "403 Forbidden" from ACBL.

This is a live test: it sends real requests to ACBL for hours and is meant to
be started by hand. It defines no test_* functions, so pytest and unittest
discovery collect nothing from it.

Targets (--target):
  club        acbl_club_download_to_json.py. --sleep is the minimum time
              between session page requests to my.acbl.org. 5 s is safe and
              2 s was blocked after ~120-200 requests (2026-11-10), so the
              search is a binary chop over 2.1 .. 4.9 s.
              Only session pages are requested here. A real run also loads
              one listing page per club. Listing pages count toward ACBL's
              limit (403 on 2026-11-12 after 121 listing pages about 2 s
              apart), so the downloader applies the same delay to them.
  tournament  acbl_tournament_download_sessions_using_sanctioned_events.py,
              its live.acbl.org page path (the one in use while ACBL_API_KEY
              is expired). --sleep is the pause after each scorecard page only
              (none after a recap or hand record page). Binary chop over
              0.0 .. 4.9 s; 0 is tried first only while the default is 0.

How a delay is judged:
  Each trial calls the downloader's own code (same Chrome session, same rate
  limiter, same 403 handling) with a new empty output folder. No session is
  skipped as "already downloaded", so every session costs a request, as in a
  first full download. Session ids come from the files already on disk; the
  saved results under e:/bridge/data/acbl are not touched.

  A trial passes when --requests page requests finish with no 403. It fails
  at the first 403; the downloader's own code then waits until ACBL serves
  pages again. Trials are separated by --cooldown seconds so one trial's
  requests do not count against the next.

  The recommended default is the lowest passing delay plus --margin, and it
  must also pass a longer run of --confirm-requests requests. If that run is
  blocked, 0.5 s is added and it is confirmed again.

Usage (acbl-pipeline venv; do not run while a download is using Chrome):
  python test_acbl_download_delay.py --target club
  python test_acbl_download_delay.py --target club --apply     # also rewrite the default
  python test_acbl_download_delay.py --target tournament
  python test_acbl_download_delay.py --target club --resume    # reuse finished trials
  python test_acbl_download_delay.py --target club --simulate 3.3   # no requests; checks the search

Trials are logged to logs/acbl_download_delay_<target>.json after each one.
Each 403 is saved to logs/acbl_download_delay_403_<time>.txt: the response
headers, the body as the server sent it and the page as Chrome rendered it,
in case ACBL says there what tripped the block or how long it lasts.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable, Iterator

HERE = pathlib.Path(__file__).resolve().parent
STEP = 0.1
# After a blocked confirmation run, retry this much slower.
CONFIRM_STEP = 0.5
# This many failed fetches in a row with no 403 means something else is wrong
# (network down, Chrome closed); stop instead of calling it a pass.
MAX_CONSECUTIVE_ERRORS = 10
# Longest wait for a 403 this test finds on its own to clear.
BLOCK_CLEAR_LIMIT_S = 30 * 60
DEFAULT_RE = re.compile(r"(?m)^DEFAULT_SLEEP_SECONDS = ([0-9.]+)")
# Headers of a 403 worth showing on the console (all are saved to the file).
NOTABLE_HEADER_RE = re.compile(r"retry|rate|limit|timing|mitigat|^cf-|^server$|^date$", re.I)


@dataclass(frozen=True)
class Target:
    name: str
    source: str      # file that holds DEFAULT_SLEEP_SECONDS
    low: float
    high: float
    low_first: bool  # low is the current default: try it before searching


TARGETS = {
    "club": Target("club", "acbl_club_download_to_json.py", 2.1, 4.9, False),
    "tournament": Target(
        "tournament",
        "acbl_tournament_download_sessions_using_sanctioned_events.py",
        0.0,
        4.9,
        True,
    ),
}


@dataclass
class Trial:
    stage: str               # "search" or "confirm"
    delay: float
    requests: int            # page requests made (up to and including a blocked one)
    blocked: bool
    cleared_s: float | None  # how long the 403 lasted; None if it never cleared
    errors: int              # fetches that failed for a reason other than 403
    mean_interval_s: float   # measured time between request starts
    max_gap_s: float         # longest pause between request starts
    elapsed_s: float
    capture: str | None = None  # file holding the saved 403 response


class StopTest(Exception):
    """The test cannot continue (a block that does not clear, repeated errors)."""


def _finish_trial(delay: float, starts: list[float], watch: "_BlockWatch", errors: int, began: float) -> Trial:
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    return Trial(
        stage="",
        delay=delay,
        requests=len(starts),
        blocked=watch.blocked,
        cleared_s=round(watch.cleared_s, 1) if watch.cleared_s is not None else None,
        errors=errors,
        mean_interval_s=round(sum(gaps) / len(gaps), 2) if gaps else 0.0,
        max_gap_s=round(max(gaps), 1) if gaps else 0.0,
        elapsed_s=round(time.monotonic() - began, 1),
        capture=watch.capture,
    )


# --- Search ------------------------------------------------------------------

def delay_grid(low: float, high: float) -> list[float]:
    return [round(low + i * STEP, 1) for i in range(int(round((high - low) / STEP)) + 1)]


def find_lowest_passing(candidates: list[float], passes: Callable[[float], bool], low_first: bool = False) -> float | None:
    """Binary chop for the lowest delay that passes. Assumes a longer delay
    never does worse than a shorter one. None when even the longest fails."""
    lo, hi = 0, len(candidates) - 1
    best = None
    if low_first:
        if passes(candidates[0]):
            return candidates[0]
        lo = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if passes(candidates[mid]):
            best = candidates[mid]
            hi = mid - 1
        else:
            lo = mid + 1
    return best


def settle_recommendation(best: float | None, current: float, high: float, margin: float, confirm: Callable[[float], bool] | None) -> float | None:
    """Turn the lowest passing delay into the default to use.

    Adds the margin (unless the lowest passing delay is today's default) and
    confirms it with the longer run, backing off CONFIRM_STEP per blocked
    confirmation. None when nothing useful held up: no gain over a default
    above the range, or no pass inside the range.
    """
    if best is None:
        return None
    rec = best if best == current else round(best + margin, 1)

    def worth_trying(delay: float) -> bool:
        if current > high:
            return delay < current  # must beat today's default
        return delay <= round(high + margin, 1)

    while worth_trying(rec):
        if confirm is None or confirm(rec):
            return rec
        rec = round(rec + CONFIRM_STEP, 1)
    return None


# --- Default in the downloader source ----------------------------------------

def read_default(source: pathlib.Path) -> float:
    with open(source, encoding="utf-8", newline="") as f:
        match = DEFAULT_RE.search(f.read())
    if not match:
        raise SystemExit(f"ERROR: no 'DEFAULT_SLEEP_SECONDS = <number>' line in {source}")
    return float(match.group(1))


def apply_default(source: pathlib.Path, value: float) -> None:
    """Rewrite the one DEFAULT_SLEEP_SECONDS line. Line endings are kept."""
    with open(source, encoding="utf-8", newline="") as f:
        text = f.read()
    new_text, count = DEFAULT_RE.subn(f"DEFAULT_SLEEP_SECONDS = {value}", text)
    if count != 1:
        raise SystemExit(f"ERROR: expected one DEFAULT_SLEEP_SECONDS line in {source}, found {count}")
    with open(source, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)


# --- Trial log ---------------------------------------------------------------

class TrialLog:
    def __init__(self, path: pathlib.Path | None, resume: bool, settings: dict):
        self.path = path
        self.settings = settings
        self.trials: list[dict] = []
        self.result: dict | None = None
        if path is not None and resume and path.is_file():
            self.trials = json.loads(path.read_text(encoding="utf-8")).get("trials", [])
            print(f"Resuming with {len(self.trials)} finished trial(s) from {path}")

    def known(self, delay: float, requests: int) -> bool | None:
        """Outcome already on record for this delay, or None. A block at any
        length is a fail; a pass only counts if it ran at least this long."""
        same = [t for t in self.trials if t["delay"] == delay]
        if any(t["blocked"] for t in same):
            return False
        if any(t["requests"] >= requests for t in same):
            return True
        return None

    def add(self, trial: Trial) -> None:
        self.trials.append(asdict(trial))
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved": datetime.now().isoformat(timespec="seconds"),
            "settings": self.settings,
            "trials": self.trials,
            "result": self.result,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# --- Probers: run one trial at one delay -------------------------------------

class SimProber:
    """No requests: a delay is blocked when it is below ``threshold``."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def run(self, delay: float, requests: int) -> Trial:
        blocked = delay < self.threshold
        return Trial("", delay, requests // 2 if blocked else requests, blocked, 0.0 if blocked else None, 0, delay, delay, 0.0)

    def close(self) -> None:
        pass


class _BlockWatch:
    """Notes every plain "403 Forbidden" the downloader waits out, and how
    long it lasted. Also checks the page itself, for a 403 on a code path
    that does not reach the downloader's wait."""

    def __init__(self, club):
        self.club = club
        self.blocked = False
        self.cleared_s: float | None = None
        self.capture: str | None = None
        self.last_403 = None  # latest page-navigation response with status 403
        original = club._wait_out_plain_403
        launch = club._launch_browser_context

        def watched(page, deadline):
            if not self.blocked:
                self.capture = self._save_block(page)
            self.blocked = True
            started = time.time()
            original(page, deadline)  # raises if the block does not clear
            self.cleared_s = time.time() - started

        def launch_and_listen(playwright):
            context = launch(playwright)
            context.on("response", self._on_response)
            return context

        club._wait_out_plain_403 = watched
        club._launch_browser_context = launch_and_listen

    def reset(self) -> None:
        self.blocked = False
        self.cleared_s = None
        self.capture = None
        self.last_403 = None

    def _on_response(self, response) -> None:
        # Runs inside Playwright's event dispatch: only read plain attributes.
        try:
            if response.status == 403 and response.request.is_navigation_request():
                self.last_403 = response
        except Exception:
            pass

    def _save_block(self, page) -> str | None:
        """Save the 403 exactly as received, before the downloader reloads
        the page. Cookies are left out. Returns the file path."""
        parts = []
        notable = []
        response = self.last_403
        if response is not None:
            try:
                headers = {k: v for k, v in response.all_headers().items() if k.lower() != "set-cookie"}
                parts.append(f"URL: {response.url}\nStatus: {response.status}")
                parts.append("--- response headers ---\n" + "\n".join(f"{k}: {v}" for k, v in sorted(headers.items())))
                notable = [f"{k}: {v}" for k, v in sorted(headers.items()) if NOTABLE_HEADER_RE.search(k)]
            except Exception as exc:
                parts.append(f"(response headers not readable: {exc})")
            try:
                parts.append("--- body as sent ---\n" + response.text())
            except Exception as exc:
                parts.append(f"(response body not readable: {exc})")
        else:
            parts.append("(no 403 navigation response was seen; rendered page only)")
        try:
            parts.append("--- page as rendered ---\n" + page.content())
        except Exception as exc:
            parts.append(f"(page not readable: {exc})")
        path = HERE / "logs" / f"acbl_download_delay_403_{datetime.now():%Y%m%d_%H%M%S}.txt"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"  could not save the 403 response ({exc})")
            return None
        print(f"  403 response saved to {path}")
        for line in notable:
            print(f"    {line}")
        return str(path)

    def check_page(self) -> None:
        page = self.club._PAGE
        if page is None:
            return
        try:
            content = page.content()
        except Exception:
            return
        if not self.club._is_plain_403(page, content):
            return
        try:
            self.club._wait_out_plain_403(page, time.time() + BLOCK_CLEAR_LIMIT_S)
        except self.club.Forbidden403Error as exc:
            print(f"  {exc}")


def _club_session_ids(root: pathlib.Path, start_club: int) -> Iterator[tuple[str, str]]:
    """(club id, session id) for sessions already on disk: clubs ascending,
    newest session first, the order of a full download."""
    clubs = sorted((int(d.name) for d in root.iterdir() if d.is_dir() and d.name.isdigit()))
    for club in clubs:
        if club < start_club:
            continue
        details = root / str(club) / "details"
        if not details.is_dir():
            continue
        ids = [p.name.split(".")[0] for p in details.glob("*.data.json")]
        for session_id in sorted((i for i in ids if i.isdigit()), key=int, reverse=True):
            yield str(club), session_id


class ClubProber:
    """Session pages from my.acbl.org through process_single_session."""

    def __init__(self, start_club: int):
        import acbl_club_download_to_json as club

        self.club = club
        self.watch = _BlockWatch(club)
        self.sessions = _club_session_ids(club.acblPath / "club-results", start_club)
        self.out = pathlib.Path(tempfile.mkdtemp(prefix="acbl_delay_test_club_"))
        print(f"Trial output (deleted at the end): {self.out}")

    def run(self, delay: float, requests: int) -> Trial:
        club = self.club
        self.watch.reset()
        club._last_request_start = None  # first request of a trial does not wait
        starts: list[float] = []
        errors = streak = 0
        began = time.monotonic()
        for n in range(1, requests + 1):
            item = next(self.sessions, None)
            if item is None:
                raise StopTest("ran out of saved club sessions to request")
            club_id, session_id = item
            try:
                status = club.process_single_session(
                    {"id": session_id, "name": f"delay test {delay}s"},
                    self.out,
                    n,
                    requests,
                    club_id=club_id,
                    sleep_seconds=delay,
                )[1]
            except club.Forbidden403Error:
                if not self.watch.blocked:
                    raise
                status = "blocked"
            starts.append(club._last_request_start or time.monotonic())
            if not self.watch.blocked and status != "processed":
                self.watch.check_page()
            if self.watch.blocked:
                break
            if status == "processed":
                streak = 0
                continue
            errors += 1
            streak += 1
            if streak >= MAX_CONSECUTIVE_ERRORS:
                raise StopTest(f"{streak} session fetches in a row failed without a 403")
        return _finish_trial(delay, starts, self.watch, errors, began)

    def close(self) -> None:
        self.club._shutdown_browser()
        shutil.rmtree(self.out, ignore_errors=True)


class TournamentProber:
    """Recap, hand record and scorecard pages from live.acbl.org through
    download_sessions_from_web."""

    def __init__(self):
        import acbl_club_download_to_json as club
        import acbl_tournament_live_download as live

        self.club = club
        self.live = live
        self.watch = _BlockWatch(club)
        # Raised from the fetch hook to end a trial; the downloader lets
        # Forbidden403Error through and stops.
        self.trial_over = type("TrialOver", (club.Forbidden403Error,), {})
        saved = (club.acblPath / "tournaments" / "sessions").glob("*.session.json")
        self.sessions = iter(sorted((p.name[: -len(".session.json")] for p in saved), reverse=True))
        self.out = pathlib.Path(tempfile.mkdtemp(prefix="acbl_delay_test_tournament_"))
        print(f"Trial output (deleted at the end): {self.out}")

    def run(self, delay: float, requests: int) -> Trial:
        self.watch.reset()
        starts: list[float] = []
        state = {"errors": 0, "streak": 0, "stop": ""}

        def fetch(url: str) -> str:
            if self.watch.blocked or state["stop"] or len(starts) >= requests:
                raise self.trial_over("trial over")
            starts.append(time.monotonic())
            try:
                page_html = self.live._default_fetch(url)
            except self.club.Forbidden403Error as exc:
                if not self.watch.blocked:
                    state["stop"] = f"page check did not clear: {exc}"
                raise
            except Exception:
                state["errors"] += 1
                state["streak"] += 1
                if state["streak"] >= MAX_CONSECUTIVE_ERRORS:
                    state["stop"] = f"{state['streak']} page fetches in a row failed without a 403"
                raise
            state["streak"] = 0
            return page_html

        began = time.monotonic()
        while not self.watch.blocked and not state["stop"] and len(starts) < requests:
            session_id = next(self.sessions, None)
            if session_id is None:
                raise StopTest("ran out of saved tournament sessions to request")
            self.live.download_sessions_from_web([session_id], self.out, sleep_seconds=delay, fetch=fetch)
        if state["stop"] and not self.watch.blocked:
            raise StopTest(state["stop"])
        return _finish_trial(delay, starts, self.watch, state["errors"], began)

    def close(self) -> None:
        self.club._shutdown_browser()
        shutil.rmtree(self.out, ignore_errors=True)


# --- Main --------------------------------------------------------------------

def _describe(trial: Trial) -> str:
    if trial.blocked:
        lasted = "did not clear" if trial.cleared_s is None else f"cleared after {trial.cleared_s:.0f}s"
        return f"BLOCKED at request {trial.requests} (403 {lasted})"
    return f"pass: {trial.requests} requests, no 403"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=sorted(TARGETS), default="club")
    parser.add_argument("--low", type=float, help="shortest delay to try (default: 2.1 club, 0.0 tournament)")
    parser.add_argument("--high", type=float, help="longest delay to try (default: 4.9)")
    parser.add_argument("--requests", type=int, default=400, help="page requests per search trial (default: 400)")
    parser.add_argument("--confirm-requests", type=int, default=1000, help="page requests in the confirmation run; 0 skips it (default: 1000)")
    parser.add_argument("--margin", type=float, default=0.2, help="seconds added to the lowest passing delay (default: 0.2)")
    parser.add_argument("--cooldown", type=float, default=300.0, help="idle seconds between trials (default: 300)")
    parser.add_argument("--start-club", type=int, default=0, help="club target: take sessions from clubs >= this number")
    parser.add_argument("--resume", action="store_true", help="reuse the trials already in the log")
    parser.add_argument("--apply", action="store_true", help="write the result into the downloader's DEFAULT_SLEEP_SECONDS")
    parser.add_argument("--simulate", type=float, metavar="SECONDS", help="no requests: pretend delays below SECONDS get a 403")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    target = TARGETS[args.target]
    low = target.low if args.low is None else args.low
    high = target.high if args.high is None else args.high
    if not 0 <= low <= high or args.requests < 1:
        parser.error("need 0 <= --low <= --high and --requests >= 1")
    candidates = delay_grid(low, high)
    source = HERE / target.source
    current = read_default(source)
    simulated = args.simulate is not None
    low_first = target.low_first and candidates[0] == current

    steps = math.ceil(math.log2(len(candidates) + 1))
    mid = (low + high) / 2
    hours = (steps * (args.requests * mid + args.cooldown) + args.confirm_requests * mid + args.cooldown) / 3600
    print("=" * 70)
    print(f"ACBL download delay test: {target.name}")
    print("=" * 70)
    print(f"Downloader:      {target.source} (current default {current}s)")
    print(f"Delays tried:    {candidates[0]}s .. {candidates[-1]}s in {STEP}s steps" + (", current default first" if low_first else ""))
    print(f"Search trial:    {args.requests} requests; confirmation: {args.confirm_requests} requests")
    print(f"Margin/cooldown: {args.margin}s / {args.cooldown:.0f}s")
    if simulated:
        print(f"SIMULATION: no requests are made; delays below {args.simulate}s are treated as blocked")
    else:
        print(f"Up to {steps} search trials + confirmation: roughly {hours:.1f} h, more for each 403 that has to clear")
    print()

    log_path = None if simulated else HERE / "logs" / f"acbl_download_delay_{target.name}.json"
    log = TrialLog(log_path, args.resume, {k: v for k, v in vars(args).items() if k not in ("resume", "apply")})
    if simulated:
        prober = SimProber(args.simulate)
    elif target.name == "club":
        prober = ClubProber(args.start_club)
    else:
        prober = TournamentProber()

    ran_a_trial = False

    def passes(delay: float, requests: int, stage: str) -> bool:
        nonlocal ran_a_trial
        known = log.known(delay, requests)
        if known is not None:
            print(f"[{stage}] {delay}s: {'pass' if known else 'BLOCKED'} (from the log)")
            return known
        if ran_a_trial and not simulated:
            print(f"Cooling down {args.cooldown:.0f}s before the next trial...")
            time.sleep(args.cooldown)
        ran_a_trial = True
        print(f"[{stage}] Trying {delay}s for {requests} requests...")
        trial = prober.run(delay, requests)
        trial.stage = stage
        log.add(trial)
        print(f"[{stage}] {delay}s: {_describe(trial)}; measured {trial.mean_interval_s}s between requests, longest gap {trial.max_gap_s}s, {trial.errors} other errors")
        if trial.blocked and trial.cleared_s is None:
            raise StopTest(f"the 403 at {delay}s did not clear; wait, then rerun with --resume")
        return not trial.blocked

    best = recommended = None
    try:
        best = find_lowest_passing(candidates, lambda d: passes(d, args.requests, "search"), low_first)
        confirm = (lambda d: passes(d, args.confirm_requests, "confirm")) if args.confirm_requests > 0 else None
        recommended = settle_recommendation(best, current, high, args.margin, confirm)
    except StopTest as exc:
        print(f"\nSTOPPED: {exc}")
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted. Finished trials are in the log; rerun with --resume.")
        return 130
    finally:
        prober.close()

    log.result = {"lowest_passing": best, "recommended": recommended, "previous_default": current}
    log.save()

    print()
    print("=" * 70)
    print(f"{'delay':>6} {'stage':>8} {'requests':>9} {'interval':>9}  result")
    for t in sorted(log.trials, key=lambda t: t["delay"]):
        print(f"{t['delay']:>5}s {t['stage']:>8} {t['requests']:>9} {t['mean_interval_s']:>8}s  {'BLOCKED' if t['blocked'] else 'pass'}")
    print("=" * 70)
    print(f"Lowest delay that passed {args.requests} requests: {best if best is not None else 'none'}")
    if recommended is None:
        if current > high:
            print(f"No delay below the current default held up. Leave DEFAULT_SLEEP_SECONDS = {current}.")
            return 0
        print(f"No delay up to {high}s avoided a 403. Rerun with a larger --high.")
        return 1
    if recommended == current:
        print(f"The current default {current}s passed. Nothing to change.")
        return 0
    print(f"Recommended default: {recommended}s (was {current}s)")
    if simulated:
        print("Simulation only: nothing was measured and nothing is changed.")
    elif args.apply:
        apply_default(source, recommended)
        print(f"Updated DEFAULT_SLEEP_SECONDS in {source}")
    else:
        print(f"To use it: set DEFAULT_SLEEP_SECONDS = {recommended} in {target.source}, or rerun with --resume --apply.")
    if log_path is not None:
        print(f"Trial log: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
