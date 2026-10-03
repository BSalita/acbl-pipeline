"""
ACBL Club Downloader
Downloads and extracts ACBL club session data from my.acbl.org
"""
import pathlib
import requests
import re
import json
import sys
import os
import time
from datetime import datetime, date
from typing import Optional, List

rootPath = pathlib.Path('e:/bridge/data')
acblPath = rootPath.joinpath('acbl')

# Persistent browser profile. Cloudflare's managed challenge on my.acbl.org can
# only be cleared by a real, non-headless Chrome. Keeping a real-Chrome
# user-data-dir lets the cf_clearance cookie survive between runs.
# IMPORTANT: the window must stay ON-SCREEN while a challenge is active —
# Turnstile / managed challenges often never complete when the window is
# minimized or moved off-screen (the previous -32000,-32000 setup hung forever).
PROFILE_DIR = acblPath.joinpath('playwright_profile')

# Init script applied to every new document. Playwright sets navigator.webdriver
# which Cloudflare treats as an automation hard-fail.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
"""

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Install with: pip install playwright && playwright install")

class Forbidden403Error(RuntimeError):
    """Raised when ACBL returns HTTP 403 (rate-limit / blocked)."""


def _launch_browser_context(p):
    """
    Launch a persistent, real-Chrome browser context capable of passing the
    Cloudflare managed challenge that now fronts my.acbl.org.

    Notes:
      - channel="chrome" uses the locally installed Google Chrome (genuine TLS
        + JS fingerprint). Headless Chrome is blocked; we run headed and keep
        the window on-screen so Turnstile can complete (auto or via a click).
      - ignore_default_args removes Playwright's --enable-automation flag.
      - We deliberately do NOT override the user agent: a real Chrome UA paired
        with mismatched client hints is itself a bot signal.
      - launch_persistent_context returns a BrowserContext directly; close the
        context to shut the browser down. The persistent user-data-dir keeps
        the cf_clearance cookie warm across runs.
    """
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        channel="chrome",
        headless=False,
        # Playwright defaults chromium_sandbox=False, which injects
        # --no-sandbox; system Chrome then shows an "unsupported command
        # line flag" infobar that itself looks bot-like to Cloudflare.
        chromium_sandbox=True,
        ignore_default_args=["--enable-automation", "--no-sandbox"],
        # Keep args minimal: recent system Chrome flags --disable-blink-features
        # (and --no-sandbox) as "unsupported" infobars, which also hurt CF scores.
        # Stealth is handled via ignore_default_args + _STEALTH_INIT_SCRIPT.
        args=[
            '--no-first-run',
            '--no-default-browser-check',
            '--window-position=80,80',
            '--window-size=1280,900',
        ],
        viewport={'width': 1280, 'height': 900},
    )


def _is_aws_waf_challenge(content: str) -> bool:
    """True for the AWS WAF interstitial in front of live.acbl.org."""
    if not content:
        return False
    return any(marker in content for marker in ("gokuProps", "challenge-container", "awswaf.com"))


def _page_has_acbl_payload(content: str) -> bool:
    """True when the HTML looks like a real ACBL results page, not a challenge interstitial."""
    if not content or _is_aws_waf_challenge(content):
        return False
    # Club details pages embed session JSON; list pages expose the Vue/app shell.
    # live.acbl.org renders tournament HTML (no ``var data =``).
    return (
        "var data =" in content
        or ('id="app"' in content and "/club-results/" in content)
        or 'href="/club-results/details/' in content
        or 'id="events"' in content
        or 'class="tournaments"' in content
        or "tablesorter scorecard" in content
        or 'id="board-' in content
        or "/ajax/tourn-list/" in content
    )


def _is_cloudflare_challenge(page, content: str) -> bool:
    """
    Detect an active Cloudflare or AWS WAF interstitial.

    Do NOT treat bare 'cf-chl' / 'challenge-platform' substrings as sufficient
    once real ACBL payload is present — cleared pages can still mention those
    strings in leftover scripts and would otherwise loop forever.
    A WAF page (gokuProps / challenge-container) is still a challenge even when
    its title is empty and the Cloudflare markers are absent.
    """
    if _is_aws_waf_challenge(content):
        return True
    if _page_has_acbl_payload(content):
        return False
    title = ""
    try:
        title = (page.title() or "").lower()
    except Exception:
        pass
    if any(s in title for s in ("just a moment", "checking your browser", "attention required", "verify you are human")):
        return True
    if not content:
        return True
    markers = (
        "cf-turnstile",
        "challenge-platform",
        "cf-chl-bypass",
        "cdn-cgi/challenge-platform",
        "checking if the site connection is secure",
    )
    return any(m in content for m in markers)


def _bring_page_window_forward(page) -> None:
    """Best-effort: put the Chrome window on-screen so Turnstile can be solved."""
    try:
        page.evaluate(
            """() => {
                try { window.moveTo(80, 80); } catch (e) {}
                try { window.resizeTo(1280, 900); } catch (e) {}
                try { window.focus(); } catch (e) {}
            }"""
        )
    except Exception:
        pass
    try:
        # CDP Browser.bringToFront is more reliable than window.focus alone.
        session = page.context.new_cdp_session(page)
        session.send("Browser.bringToFront")
    except Exception:
        pass


def _try_click_turnstile(page) -> None:
    """Best-effort click on a visible Turnstile checkbox / widget body."""
    try:
        for frame in page.frames:
            url = (frame.url or "").lower()
            if "turnstile" not in url and "challenges.cloudflare.com" not in url:
                continue
            for sel in ("input[type='checkbox']", "body"):
                try:
                    loc = frame.locator(sel)
                    if loc.count() > 0:
                        loc.first.click(timeout=1500, force=True)
                        return
                except Exception:
                    continue
    except Exception:
        pass
    try:
        page.frame_locator(
            "iframe[src*='turnstile'], iframe[src*='challenges.cloudflare']"
        ).locator("body").first.click(timeout=1500, force=True)
    except Exception:
        pass


# Someone often is not at the machine when Turnstile appears. Three minutes
# expired before the check could be clicked (2026-10-03, title stayed
# 'ACBL Live'). Keep waiting, and reload: a refresh often replaces the
# interstitial once the clearance cookie is set.
CHALLENGE_TIMEOUT_MS = 6 * 60 * 60 * 1000
CHALLENGE_RELOAD_S = 30


def _wait_through_cloudflare(
    page,
    timeout_ms: int = CHALLENGE_TIMEOUT_MS,
    reload_after_s: float = CHALLENGE_RELOAD_S,
) -> bool:
    """
    After navigating, wait out a Cloudflare managed/JS challenge interstitial.

    Real Chrome usually clears the challenge automatically within a few seconds.
    Interactive Turnstile needs a visible window (and sometimes a human click);
    we bring the window forward and print instructions. Every
    ``reload_after_s`` the page is refreshed, which often dismisses a stuck
    challenge without a click. The wait defaults to six hours so an unattended
    run is still sitting on the check when someone returns.

    Returns:
        True once real page content is loaded.

    Raises:
        Forbidden403Error if the challenge never clears within timeout_ms.
    """
    deadline = time.time() + timeout_ms / 1000.0
    started = time.time()
    announced = False
    last_status = 0.0
    last_reload = time.time()
    clicked = False
    while time.time() < deadline:
        content = ""
        try:
            content = page.content()
        except Exception:
            pass

        if content and not _is_cloudflare_challenge(page, content):
            try:
                page.wait_for_load_state('domcontentloaded', timeout=10000)
            except Exception:
                pass
            if announced:
                print("  Cloudflare/WAF challenge cleared.")
            return True

        if not announced:
            print("  Cloudflare/WAF challenge detected; waiting for it to clear...")
            print("  -> Chrome window should be visible. If you see 'Verify you are human', click it.")
            print(f"  -> Refreshing every {int(reload_after_s)}s. A reload often clears this without a click.")
            print(f"  -> This wait lasts up to {int(timeout_ms / 3600000)}h, so it is still here when you come back.")
            print(f"  -> Profile: {PROFILE_DIR}")
            _bring_page_window_forward(page)
            announced = True
            last_reload = time.time()

        if not clicked:
            _try_click_turnstile(page)
            clicked = True

        now = time.time()
        if announced and reload_after_s >= 0 and now - last_reload >= reload_after_s:
            print("  refreshing the page; a reload often clears the challenge")
            try:
                page.reload(wait_until="domcontentloaded", timeout=90000)
            except Exception as exc:
                print(f"  refresh failed ({exc}); still waiting")
            last_reload = time.time()
            clicked = False
            _try_click_turnstile(page)
            continue

        if now - last_status >= 15:
            title = ""
            try:
                title = page.title() or ""
            except Exception:
                pass
            elapsed_m = int((now - started) / 60)
            print(f"  ... still waiting ({elapsed_m}m elapsed); title={title!r}")
            _bring_page_window_forward(page)
            _try_click_turnstile(page)
            last_status = now

        page.wait_for_timeout(1000)

    raise Forbidden403Error(
        "Cloudflare challenge did not clear within timeout. "
        "Complete the check in the Chrome window, or delete "
        f"{PROFILE_DIR} and retry."
    )


# --- Shared browser singleton -------------------------------------------------
# Launching (and closing) a full Chrome per HTTP GET is extremely slow. Instead
# we keep ONE persistent context + page alive for the whole run and reuse it for
# every navigation. This also keeps the Cloudflare cf_clearance cookie warm.
_PW = None        # playwright instance from sync_playwright().start()
_CONTEXT = None   # persistent BrowserContext
_PAGE = None      # reusable Page


def _get_browser_page():
    """
    Return a shared, reusable Playwright Page, launching the persistent
    real-Chrome context on first use and reusing it for every later request.
    """
    global _PW, _CONTEXT, _PAGE
    if _PAGE is not None:
        return _PAGE
    _PW = sync_playwright().start()
    _CONTEXT = _launch_browser_context(_PW)
    try:
        _CONTEXT.add_init_script(_STEALTH_INIT_SCRIPT)
    except Exception:
        pass
    # Reuse an existing tab from the persistent profile when present; otherwise
    # open one. Warm the origin so cf_clearance can be established before the
    # first details URL (helps when the profile is cold).
    _PAGE = _CONTEXT.pages[0] if _CONTEXT.pages else _CONTEXT.new_page()
    try:
        if "my.acbl.org" not in (_PAGE.url or ""):
            print("  Warming Chrome session at https://my.acbl.org/ ...")
            _PAGE.goto("https://my.acbl.org/", wait_until="domcontentloaded", timeout=60000)
            _wait_through_cloudflare(_PAGE)
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"  Warning: session warm-up failed ({e}); continuing anyway")
    return _PAGE


def _reset_browser_page() -> None:
    """Drop the shared tab after a cancelled navigation and open a fresh one.

    ``net::ERR_ABORTED`` leaves the shared page mid-navigation. The next
    ``goto`` on that same tab usually aborts too. A new tab keeps the
    persistent profile and its Cloudflare cookie.
    """
    global _PAGE
    page = _PAGE
    _PAGE = None
    try:
        if page is not None:
            page.close()
    except Exception:
        pass
    if _CONTEXT is None:
        return
    try:
        _PAGE = _CONTEXT.new_page()
    except Exception as exc:
        print(f"  could not open a new tab ({exc}); restarting Chrome")
        _shutdown_browser()


def _transient_navigation_error(exc: BaseException) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in ("ERR_", "net::", "Timeout", "timeout", "Target closed", "frame was detached")
    )


def fetch_acbl_page(url: str, timeout_ms: int = 180000) -> str:
    """
    Open ``url`` in the shared real-Chrome session and return its HTML.

    Club and live.acbl.org pages both go through this. The persistent profile
    keeps the Cloudflare clearance cookie warm, and the waiter holds the
    on-screen window until a Cloudflare or AWS WAF interstitial finishes.
    A cancelled navigation (``net::ERR_ABORTED``) retries on a fresh tab.
    """
    last_error: Exception | None = None
    for attempt in range(1, 4):
        page = _get_browser_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=90000)
            _wait_through_cloudflare(page)
            for _ in range(8):
                try:
                    return page.content()
                except Exception as exc:
                    last_error = exc
                    page.wait_for_timeout(400)
            raise Forbidden403Error(f"Could not read page content for {url}: {last_error}")
        except Forbidden403Error:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= 3 or not _transient_navigation_error(exc):
                raise
            print(f"  navigation failed for {url} ({exc}); retry {attempt}/3")
            _reset_browser_page()
            time.sleep(2 * attempt)
    raise Forbidden403Error(f"Could not open {url}: {last_error}")


def shutdown_acbl_browser() -> None:
    """Close the shared Chrome session started by :func:`fetch_acbl_page`."""
    _shutdown_browser()


def _shutdown_browser() -> None:
    """Close the shared browser context and stop Playwright (best effort)."""
    global _PW, _CONTEXT, _PAGE
    try:
        if _CONTEXT is not None:
            _CONTEXT.close()
    except Exception:
        pass
    try:
        if _PW is not None:
            _PW.stop()
    except Exception:
        pass
    _PW = _CONTEXT = _PAGE = None


def _countdown_abort_window(seconds: int, message: str) -> bool:
    """
    Give the user a short Ctrl+C window.

    Returns:
        True if user did NOT abort (safe to proceed), False if aborted.
    """
    print(message)
    try:
        for remaining in range(seconds, 0, -1):
            print(f"  -> Proceeding in {remaining} second(s)... (Ctrl+C to cancel)")
            time.sleep(1)
        return True
    except KeyboardInterrupt:
        print("\nAborted by user. Skipping this step.\n")
        return False


def remove_club_html_files_on_startup(output_dir: pathlib.Path) -> int:
    """
    Remove all <club-number>/<club-number>.html files under output_dir.
    Gives the user 10 seconds to abort before deleting.

    Returns:
        Number of files deleted.
    """
    try:
        if not output_dir.exists():
            return 0

        candidates: list[pathlib.Path] = []
        for club_dir in output_dir.iterdir():
            if not club_dir.is_dir():
                continue
            club_number = club_dir.name
            if not club_number.isdigit():
                continue
            html_path = club_dir / f"{club_number}.html"
            if html_path.exists() and html_path.is_file():
                candidates.append(html_path)

        if not candidates:
            return 0

        print("=" * 70)
        print("Startup cleanup: deleting cached club HTML files")
        print("=" * 70)
        print(f"Target directory: {output_dir}")
        print(f"Matched files: {len(candidates)}")
        print("Pattern: <club-number>/<club-number>.html")

        if not _countdown_abort_window(
            10,
            "\nAbout to delete matched files.\n",
        ):
            return 0

        deleted = 0
        for p in candidates:
            try:
                p.unlink()
                deleted += 1
            except Exception as e:
                print(f"WARNING: Failed to delete {p}: {e}")

        print(f"\nDeleted {deleted} file(s).\n")
        return deleted
    except Exception as e:
        print(f"WARNING: Startup cleanup failed: {e}")
        return 0


def extract_embedded_json(url: str) -> Optional[dict]:
    """
    Fetch ACBL results page and extract embedded JSON data
    
    Args:
        url: ACBL results page URL
        
    Returns:
        Extracted JSON data as dictionary, or None if failed
    """
    # Set up headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        print(f"Response status: {response.status_code}")
        print(f"Content length: {len(response.text):,} bytes")
        
        # Extract embedded JSON
        pattern = r'var data = (\{.*?\});'
        match = re.search(pattern, response.text, re.DOTALL)
        
        if not match:
            print("ERROR: Could not find 'var data = {...}' in page")
            return None
        
        # Clean up and parse JSON
        json_str = match.group(1)
        json_str = json_str.replace(r'\/', '/')
        
        data = json.loads(json_str)
        print(f"Successfully extracted JSON data")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: HTTP request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return None


def extract_with_playwright(url: str) -> Optional[dict]:
    """
    Fetch ACBL results page using Playwright and extract embedded JSON data
    
    Args:
        url: ACBL results page URL
        
    Returns:
        Extracted JSON data as dictionary, or None if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright is not available")
        return None
    
    try:
        print(f"Fetching with Playwright: {url}")
        
        # Reuse the shared persistent page (launched once, kept alive for the run)
        page = _get_browser_page()
        
        # Navigate to page
        response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
        status = response.status if response else None
        print(f"Response status: {status}")
        
        # The initial response may be a Cloudflare 403 challenge page; wait
        # for it to clear (raises Forbidden403Error if it never does).
        _wait_through_cloudflare(page)
        
        # Wait for content to load
        page.wait_for_selector('body', timeout=10000)
        
        # Get page content
        content = page.content()
        
        # Extract embedded JSON
        pattern = r'var data = (\{.*?\});'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("ERROR: Could not find 'var data = {...}' in page")
            return None
        
        # Clean up and parse JSON
        json_str = match.group(1)
        json_str = json_str.replace(r'\/', '/')
        
        data = json.loads(json_str)
        
        return data
    
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected in extract_with_playwright")
        _shutdown_browser()
        os._exit(130)
            
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"ERROR: Playwright extraction failed: {e}")
        return None


def extract_from_file(html_file: str) -> Optional[dict]:
    """
    Extract embedded JSON from a local HTML file
    
    Args:
        html_file: Path to HTML file
        
    Returns:
        Extracted JSON data as dictionary, or None if failed
    """
    try:
        print(f"Reading from file: {html_file}")
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"File size: {len(content):,} bytes")
        
        # Extract embedded JSON
        pattern = r'var data = (\{.*?\});'
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            print("ERROR: Could not find 'var data = {...}' in file")
            return None
        
        # Clean up and parse JSON
        json_str = match.group(1)
        json_str = json_str.replace(r'\/', '/')
        
        data = json.loads(json_str)
        print(f"Successfully extracted JSON data")
        
        return data
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {html_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return None


def get_club_sessions(club_number: str, output_dir: pathlib.Path, refresh: bool = False, start_date: Optional[str] = None) -> Optional[tuple[List[dict], str, pathlib.Path]]:
    """
    Get list of all sessions for a club
    
    Args:
        club_number: Club number
        output_dir: Base output directory
        refresh: If True, force refresh even if cached file exists
        start_date: If provided, stop pagination when encountering sessions before this date (YYYY-MM-DD)
        
    Returns:
        List of session dictionaries with id and name, or None if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright is required for club session listing")
        return None
    
    try:
        url = f"https://my.acbl.org/club-results/{club_number}"
        
        # Create club directory
        club_dir = pathlib.Path(output_dir) / club_number
        club_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if club HTML already exists
        club_html_file = club_dir / f"{club_number}.html"
        
        # Check if file exists and is valid
        file_is_valid = False
        if club_html_file.exists() and not refresh:
            file_size = club_html_file.stat().st_size
            if file_size < 2048:  # Less than 2KB is likely invalid
                print(f"WARNING: Cached club page {club_html_file} is too small ({file_size} bytes < 2KB)")
                print(f"Removing invalid file and re-fetching...")
                club_html_file.unlink()  # Delete the invalid file
            else:
                file_is_valid = True
        
        if file_is_valid:
            print(f"Using cached club page: {club_html_file}")
            # Read from cached file
            with open(club_html_file, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"Fetching club sessions from: {url}")
            
            try:
                # Reuse the shared persistent page (launched once for the run)
                page = _get_browser_page()
                
                # Navigate to club results page
                response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
                
                print(f"Response status: {response.status if response else None}")
                
                # The initial response may be a Cloudflare 403 challenge
                # page; wait for it to clear (raises Forbidden403Error if it
                # never does).
                _wait_through_cloudflare(page)
                
                page.wait_for_selector('body', timeout=10000)
                
                # Collect all session links across all pages
                # Stop early if we encounter sessions we already have (sorted by date)
                all_content = []
                page_num = 1
                details_dir = pathlib.Path(output_dir) / club_number / "details"
                
                while True:
                    print(f"  Fetching page {page_num}...")
                    
                    # Wait for content to load
                    time.sleep(2)
                    
                    # Get current page content
                    content = page.content()
                    all_content.append(content)
                    
                    # Check if any sessions on this page already exist (optimization)
                    # Since sessions are sorted by date, if we find existing sessions, we can stop
                    if details_dir.exists():
                        page_pattern = r'href="/club-results/details/(\d+)"'
                        page_sessions = re.findall(page_pattern, content)
                        existing_count = 0
                        for session_id in page_sessions:
                            session_file = details_dir / f"{session_id}.data.json"
                            if session_file.exists():
                                existing_count += 1
                        
                        if existing_count > 0:
                            print(f"  Found {existing_count} existing sessions on page {page_num}")
                            print(f"  Stopping pagination (sessions sorted by date, rest already downloaded)")
                            break
                    
                    # Date-based pagination optimization using Unix timestamps from HTML
                    # Check if any session on this page is before start_date
                    if start_date:
                        # Extract Unix timestamps from data-sort attributes in <td> tags
                        # Pattern: <td data-sort="1574035200">11/18/2019</td>
                        timestamp_pattern = r'<td data-sort="(\d+)">'
                        timestamps = re.findall(timestamp_pattern, content)
                        
                        if timestamps:
                            # Convert start_date to Unix timestamp for comparison
                            try:
                                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                                start_timestamp = int(start_dt.timestamp())
                                
                                # Check if the oldest (last) timestamp on this page is before start_date
                                oldest_timestamp = int(timestamps[-1]) if timestamps else float('inf')
                                
                                if oldest_timestamp < start_timestamp:
                                    print(f"  Found session(s) before start_date {start_date} on page {page_num}")
                                    print(f"  Stopping pagination (remaining pages contain older sessions)")
                                    break
                            except Exception as e:
                                # If date parsing fails, continue pagination
                                pass
                    
                    # Check if there's a "Next" button and if it's enabled
                    try:
                        # Look for pagination - common patterns:
                        # 1. Next button/link with class containing "next"
                        # 2. Page number links
                        next_button = page.query_selector('a[rel="next"], button[rel="next"], a.next:not(.disabled), button.next:not(.disabled), .pagination a[aria-label="Next"]:not(.disabled)')
                        
                        if next_button:
                            # Check if button is actually clickable (not disabled)
                            is_disabled = next_button.get_attribute('aria-disabled')
                            class_attr = next_button.get_attribute('class')
                            
                            if is_disabled == 'true' or (class_attr and 'disabled' in class_attr):
                                print(f"  Reached last page (page {page_num})")
                                break
                            
                            # Click next and wait for navigation
                            print(f"  Navigating to page {page_num + 1}...")
                            next_button.click()
                            page.wait_for_load_state('domcontentloaded', timeout=10000)
                            page_num += 1
                        else:
                            # No next button found - single page or last page
                            print(f"  No more pages (total: {page_num})")
                            break
                            
                    except Exception as e:
                        # No pagination or error - assume single page
                        print(f"  Pagination check complete (total: {page_num} pages)")
                        break
                
                # Combine all page content and cache to disk
                combined_content = '\n'.join(all_content)
                content = combined_content

                with open(club_html_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"Cached club page: {club_html_file}")
            
            except KeyboardInterrupt:
                print("\n\nKeyboardInterrupt detected in get_club_sessions")
                _shutdown_browser()
                os._exit(130)
        
        # Parse session links from all pages
        # Look for links like: /club-results/details/{session_id}
        pattern = r'href="/club-results/details/(\d+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, content)
        
        sessions = []
        seen_ids = set()
        
        for session_id, session_name in matches:
            if session_id not in seen_ids:
                sessions.append({
                    'id': session_id,
                    'name': session_name.strip(),
                    'club_id': club_number
                })
                seen_ids.add(session_id)
        
        print(f"Found {len(sessions)} sessions for club {club_number}")
        return sessions, content, club_html_file
        
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"ERROR: Failed to get club sessions: {e}")
        return None


def get_player_sessions(player_id: str) -> Optional[List[dict]]:
    """
    Get list of all sessions for a player
    
    Args:
        player_id: Player ID (ACBL number)
        
    Returns:
        List of session dictionaries with id and name, or None if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright is required for player session listing")
        return None
    
    try:
        url = f"https://my.acbl.org/club-results/my-results/{player_id}"
        print(f"Fetching player sessions from: {url}")
        
        # Reuse the shared persistent page (launched once for the run)
        page = _get_browser_page()
        
        # Navigate to player results page
        response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
        
        print(f"Response status: {response.status if response else None}")
        
        # The initial response may be a Cloudflare 403 challenge page; wait
        # for it to clear (raises Forbidden403Error if it never does).
        _wait_through_cloudflare(page)
        
        page.wait_for_selector('body', timeout=10000)
        
        # Collect all session links across all pages
        all_content = []
        page_num = 1
        
        while True:
            print(f"  Fetching page {page_num}...")
            
            # Wait for content to load
            time.sleep(2)
            
            # Get current page content
            content = page.content()
            all_content.append(content)
            
            # Check if there's a "Next" button and if it's enabled
            try:
                # Look for pagination - common patterns
                next_button = page.query_selector('a[rel="next"], button[rel="next"], a.next:not(.disabled), button.next:not(.disabled), .pagination a[aria-label="Next"]:not(.disabled)')
                
                if next_button:
                    # Check if button is actually clickable (not disabled)
                    is_disabled = next_button.get_attribute('aria-disabled')
                    class_attr = next_button.get_attribute('class')
                    
                    if is_disabled == 'true' or (class_attr and 'disabled' in class_attr):
                        print(f"  Reached last page (page {page_num})")
                        break
                    
                    # Click next and wait for navigation
                    print(f"  Navigating to page {page_num + 1}...")
                    next_button.click()
                    page.wait_for_load_state('domcontentloaded', timeout=10000)
                    page_num += 1
                else:
                    # No next button found - single page or last page
                    print(f"  No more pages (total: {page_num})")
                    break
                    
            except Exception as e:
                # No pagination or error - assume single page
                print(f"  Pagination check complete (total: {page_num} pages)")
                break
        
        # Combine all page content
        combined_content = '\n'.join(all_content)
        
        # Parse session links from all pages
        # Look for links like: /club-results/details/{session_id}
        pattern = r'href="/club-results/details/(\d+)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, combined_content)
        
        sessions = []
        seen_ids = set()
        
        for session_id, session_name in matches:
            if session_id not in seen_ids:
                sessions.append({
                    'id': session_id,
                    'name': session_name.strip()
                })
                seen_ids.add(session_id)
        
        print(f"Found {len(sessions)} sessions for player {player_id}")
        return sessions
    
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected in get_player_sessions")
        _shutdown_browser()
        os._exit(130)
            
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"ERROR: Failed to get player sessions: {e}")
        return None


def get_all_clubs() -> Optional[List[str]]:
    """
    Get list of all club IDs from ACBL main club results page
    
    Returns:
        List of club ID strings, or None if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright is required for getting club list")
        return None
    
    try:
        url = "https://my.acbl.org/club-results"
        print(f"Fetching all clubs from: {url}")
        
        # Reuse the shared persistent page (launched once for the run)
        page = _get_browser_page()
        
        # Navigate to main club results page
        response = page.goto(url, wait_until='domcontentloaded', timeout=60000)
        
        print(f"Response status: {response.status if response else None}")
        
        # The initial response may be a Cloudflare 403 challenge page; wait
        # for it to clear (raises Forbidden403Error if it never does).
        _wait_through_cloudflare(page)
        
        page.wait_for_selector('body', timeout=10000)
        
        # Get page content
        content = page.content()
        
        # Try new format first: let address = [{"club_id":"100040",...},...]
        pattern_new = r'let\s+address\s*=\s*(\[.*?\]);'
        match = re.search(pattern_new, content, re.DOTALL)
        
        if match:
            try:
                address_data = json.loads(match.group(1))
                # Extract club_id from each object, filter for type="club"
                club_ids = [
                    item['club_id'] for item in address_data 
                    if isinstance(item, dict) and item.get('type') == 'club' and item.get('club_id')
                ]
                if club_ids:
                    print(f"Found {len(club_ids)} clubs (new format)")
                    return club_ids
            except json.JSONDecodeError as e:
                print(f"WARNING: Failed to parse address JSON: {e}")
        
        # Fall back to old format: clubs:JSON.stringify([100040,100123,...])
        pattern_old = r'clubs:JSON\.stringify\(\[([\d,\s]+)\]'
        match = re.search(pattern_old, content)
        
        if match:
            clubs_str = match.group(1)
            club_ids = [club_id.strip() for club_id in clubs_str.split(',')]
            print(f"Found {len(club_ids)} clubs (old format)")
            return club_ids
        
        print("ERROR: Could not find clubs array in page (tried both formats)")
        return None  # Explicit return to satisfy type checker
    
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt detected in get_all_clubs")
        _shutdown_browser()
        os._exit(130)
            
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"ERROR: Failed to get club list: {e}")
        return None


def process_single_session(session: dict, output_dir: str, session_num: int, total_sessions: int, club_id: Optional[str] = None, refresh: bool = False, sleep_seconds: int = 2, start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple:
    """
    Process a single session
    
    Args:
        session: Session dict with 'id' and 'name'
        output_dir: Directory to save JSON files
        session_num: Current session number (for display)
        total_sessions: Total number of sessions (for display)
        club_id: Optional club ID for organizing files
        refresh: Not used for sessions (only affects club HTML)
        sleep_seconds: Seconds to sleep between requests (default: 2)
        start_date: Filter sessions on/after this date (YYYY-MM-DD) - stops processing if session is before this
        end_date: Filter sessions on/before this date (YYYY-MM-DD)
        
    Returns:
        Tuple of (session_id, status) where status is:
            'processed' - successfully extracted and saved
            'skipped' - already exists or filtered out
            'failed' - extraction failed
            'stop' - stop processing (date before start_date)
            'skip_club' - skip entire club (first session failed to extract)
    """
    session_id = session['id']
    session_name = session['name']
    
    # Determine club ID from session data if available
    if not club_id and 'club_id' in session:
        club_id = session['club_id']
    
    # For player/session modes without club_id, we need to fetch first to determine club_id
    # Check common locations for existing file
    possible_locations = []
    
    if club_id:
        # We know the club ID, use standard location
        details_dir = pathlib.Path(output_dir) / club_id / "details"
        output_file = details_dir / f"{session_id}.data.json"
        possible_locations.append(output_file)
    else:
        # Check if file exists in any club directory (from previous downloads)
        base_dir = pathlib.Path(output_dir)
        if base_dir.exists():
            for club_dir in base_dir.iterdir():
                if club_dir.is_dir() and club_dir.name.isdigit():
                    details_dir = club_dir / "details"
                    session_file = details_dir / f"{session_id}.data.json"
                    if session_file.exists():
                        print(f"[{session_num}/{total_sessions}] SKIP - {session_id} ({session_name}) - already exists at {session_file}")
                        return (session_id, 'skipped')
    
    # Check if file exists at known location
    if club_id and output_file.exists():
        print(f"[{session_num}/{total_sessions}] SKIP - {session_id} ({session_name}) - already exists")
        return (session_id, 'skipped')
    
    print(f"[{session_num}/{total_sessions}] Processing {session_id} ({session_name})...")
    
    # Sleep BEFORE making request to rate limit properly
    # Skip sleep for first session
    if session_num > 1:
        print(f"  -> Sleeping {sleep_seconds} seconds before request...")
        time.sleep(sleep_seconds)
    
    # Construct URL and extract JSON
    url = f"https://my.acbl.org/club-results/details/{session_id}"
    
    try:
        data = extract_with_playwright(url)
        
        if data:
            # Check date filters AFTER fetching (to avoid fetching outside date range)
            session_date = data.get('start_date')
            if session_date and (start_date or end_date):
                try:
                    # Parse ACBL date format (MM/DD/YYYY)
                    dt = datetime.strptime(session_date, '%m/%d/%Y')
                    session_date_comparable = dt.strftime('%Y-%m-%d')
                    
                    # Check if session is BEFORE start_date - STOP processing (sessions sorted by date, newest first)
                    if start_date and session_date_comparable < start_date:
                        print(f"  -> Session date {session_date} is before start_date {start_date}")
                        print(f"  -> STOPPING: All remaining sessions are older than start_date")
                        return (session_id, 'stop')
                    
                    # Check if session is AFTER end_date - skip this session but continue
                    if end_date and session_date_comparable > end_date:
                        print(f"  -> Session date {session_date} is after end_date {end_date}, skipping")
                        return (session_id, 'skipped')
                        
                except Exception as e:
                    print(f"  -> WARNING: Could not parse session date '{session_date}': {e}")
            
            # Extract club_id from JSON if not provided
            if not club_id:
                club_id = str(data.get('club_id_number', ''))
                if not club_id:
                    print(f"  -> WARNING: No club_id_number found in JSON, cannot organize properly")
                    return (session_id, 'failed')
            
            # Now save to proper location: club-results/<club_id>/details/<session_id>.data.json
            details_dir = pathlib.Path(output_dir) / club_id / "details"
            details_dir.mkdir(parents=True, exist_ok=True)
            output_file = details_dir / f"{session_id}.data.json"
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Get file size
            file_size = output_file.stat().st_size
            print(f"  -> Saved [{session_num}/{total_sessions}]: {output_file} ({file_size:,} bytes)")
            
            return (session_id, 'processed')
        else:
            print(f"  -> FAILED to extract data")
            # If this is the first session and extraction failed, skip the entire club
            if session_num == 1:
                print(f"  -> First session failed - skipping entire club")
                return (session_id, 'skip_club')
            return (session_id, 'failed')
            
    except Forbidden403Error:
        raise
    except Exception as e:
        print(f"  -> ERROR: {e}")
        # If this is the first session and extraction failed, skip the entire club
        if session_num == 1:
            print(f"  -> First session failed - skipping entire club")
            return (session_id, 'skip_club')
        return (session_id, 'failed')


def filter_sessions(sessions: List[dict], start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
    """
    Filter sessions by date range and limit
    
    Args:
        sessions: List of session dictionaries with 'id' and 'name'
        start_date: Include sessions on/after this date (YYYY-MM-DD)
        end_date: Include sessions on/before this date (YYYY-MM-DD)
        limit: Maximum number of sessions to return
        
    Returns:
        Filtered list of sessions
    """
    filtered = sessions
    
    # Apply date filters (session names often contain dates, but we'll need to fetch to get actual dates)
    # For now, we'll apply limit only and note that date filtering requires fetching session data
    
    # Apply limit
    if limit is not None and limit > 0:
        filtered = filtered[:limit]
    
    return filtered


def filter_sessions_by_date(sessions: List[dict], start_date: Optional[str] = None, end_date: Optional[str] = None, output_dir: str = "club-results") -> List[dict]:
    """
    Filter sessions by actual session date from cached files only (no fetching)
    
    Note: This only filters sessions that have cached data. Sessions without cached data
    will be included and date-checked during processing.
    
    Args:
        sessions: List of session dictionaries with 'id' and 'name'
        start_date: Include sessions on/after this date (YYYY-MM-DD)
        end_date: Include sessions on/before this date (YYYY-MM-DD)
        output_dir: Directory where cached files are stored
        
    Returns:
        Filtered list of sessions that match date criteria or have no cached date
    """
    if not start_date and not end_date:
        return sessions
    
    filtered = []
    
    for session in sessions:
        session_id = session['id']
        club_id = session.get('club_id')
        
        # Determine file location based on club organization
        if club_id:
            session_file = pathlib.Path(output_dir) / club_id / "details" / f"{session_id}.data.json"
        else:
            session_file = pathlib.Path(output_dir) / session_id / f"{session_id}.data.json"
        
        # Check if we have cached data
        session_date = None
        if session_file.exists():
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_date = data.get('start_date')
            except:
                pass
        
        # Apply date filter only if we have cached date info
        if session_date:
            # Convert date format (MM/DD/YYYY to YYYY-MM-DD for comparison)
            try:
                # Parse ACBL date format (MM/DD/YYYY)
                dt = datetime.strptime(session_date, '%m/%d/%Y')
                session_date_comparable = dt.strftime('%Y-%m-%d')
                
                # Check date range - skip if outside range
                if start_date and session_date_comparable < start_date:
                    continue
                if end_date and session_date_comparable > end_date:
                    continue
            except:
                # If date parsing fails, include the session
                pass
        
        # Include session (either no cached date, or date is in range)
        filtered.append(session)
    
    return filtered


def process_club_sessions(club_number: str, output_dir: pathlib.Path, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None, refresh: bool = False, sleep_seconds: int = 2) -> int:
    """
    Process all sessions for a club and extract JSON for each
    
    Args:
        club_number: Club number
        output_dir: Directory to save JSON files
        start_date: Filter sessions on/after this date (YYYY-MM-DD)
        end_date: Filter sessions on/before this date (YYYY-MM-DD)
        limit: Maximum number of sessions to process
        refresh: If True, re-fetch club HTML to get latest session list
        sleep_seconds: Seconds to sleep between requests (default: 2)
        
    Returns:
        Number of sessions successfully processed
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of sessions (creates club directory; HTML marker is written at end on success)
    # Pass start_date to enable smart pagination (stops when encountering old sessions)
    club_sessions_result = get_club_sessions(club_number, output_dir, refresh, start_date)
    
    if not club_sessions_result:
        print("No sessions found or error occurred")
        return 0

    sessions, club_html_content, club_html_file = club_sessions_result
    
    print(f"\nFound {len(sessions)} total sessions")
    
    # Apply date filtering if specified
    if start_date or end_date:
        print(f"Filtering by date range: {start_date or 'any'} to {end_date or 'any'}...")
        sessions = filter_sessions_by_date(sessions, start_date, end_date, output_dir)
        print(f"After date filter: {len(sessions)} sessions")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        print(f"Applying limit: {limit} sessions")
        sessions = sessions[:limit]
    
    print(f"\nProcessing {len(sessions)} sessions...")
    print("=" * 70)
    
    # Process sessions sequentially (no concurrent requests due to rate limiting)
    results = []
    for i, session in enumerate(sessions):
        result = process_single_session(session, output_dir, i + 1, len(sessions), club_number, refresh, sleep_seconds, start_date, end_date)
        results.append(result)
        
        # Check if we should skip the entire club (first session failed to extract)
        if result[1] == 'skip_club':
            print(f"\nSkipping entire club - first session failed to extract data")
            break
        
        # Check if we should stop processing (session date before start_date)
        if result[1] == 'stop':
            print(f"\nStopping session processing - reached sessions older than start_date")
            break
    
    # Count results
    processed = sum(1 for _, status in results if status == 'processed')
    skipped = sum(1 for _, status in results if status == 'skipped')
    failed = sum(1 for _, status in results if status == 'failed')
    stopped = sum(1 for _, status in results if status == 'stop')
    club_skipped = sum(1 for _, status in results if status == 'skip_club')
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total sessions attempted: {len(results)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing/filtered): {skipped}")
    print(f"  Failed: {failed}")
    if stopped > 0:
        print(f"  Stopped early: {stopped} (date before start_date)")
    if club_skipped > 0:
        print(f"  Club skipped: first session failed to extract")
    
    return processed


def process_player_sessions(player_id: str, output_dir: pathlib.Path, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None, refresh: bool = False, sleep_seconds: int = 2) -> int:
    """
    Process all sessions for a player and extract JSON for each
    
    Args:
        player_id: Player ID (ACBL number)
        output_dir: Directory to save JSON files
        start_date: Filter sessions on/after this date (YYYY-MM-DD)
        end_date: Filter sessions on/before this date (YYYY-MM-DD)
        limit: Maximum number of sessions to process
        refresh: Not used for player mode (sessions always skip existing files)
        sleep_seconds: Seconds to sleep between requests (default: 2)
        
    Returns:
        Number of sessions successfully processed
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of sessions
    sessions = get_player_sessions(player_id)
    
    if not sessions:
        print("No sessions found or error occurred")
        return 0
    
    print(f"\nFound {len(sessions)} total sessions")
    
    # Apply date filtering if specified
    if start_date or end_date:
        print(f"Filtering by date range: {start_date or 'any'} to {end_date or 'any'}...")
        sessions = filter_sessions_by_date(sessions, start_date, end_date, output_dir)
        print(f"After date filter: {len(sessions)} sessions")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        print(f"Applying limit: {limit} sessions")
        sessions = sessions[:limit]
    
    print(f"\nProcessing {len(sessions)} sessions...")
    print("=" * 70)
    
    # Process sessions sequentially (no concurrent requests due to rate limiting)
    results = []
    for i, session in enumerate(sessions):
        result = process_single_session(session, output_dir, i + 1, len(sessions), None, refresh, sleep_seconds, start_date, end_date)
        results.append(result)
        
        # Check if we should stop processing (session date before start_date)
        if result[1] == 'stop':
            print(f"\nStopping session processing - reached sessions older than start_date")
            break
    
    # Count results
    processed = sum(1 for _, status in results if status == 'processed')
    skipped = sum(1 for _, status in results if status == 'skipped')
    failed = sum(1 for _, status in results if status == 'failed')
    stopped = sum(1 for _, status in results if status == 'stop')
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Total sessions attempted: {len(results)}")
    print(f"  Processed: {processed}")
    print(f"  Skipped (existing/filtered): {skipped}")
    print(f"  Failed: {failed}")
    if stopped > 0:
        print(f"  Stopped early: {stopped} (date before start_date)")
    
    return processed


def process_all_clubs(output_dir: pathlib.Path, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None, refresh: bool = False, sleep_seconds: int = 2, starting_club: Optional[int] = None, skip_if_html_exists: bool = False, sleep_403_seconds: int = 300) -> int:
    """
    Process all clubs from ACBL and extract sessions for each
    
    Args:
        output_dir: Directory to save JSON files
        start_date: Filter sessions on/after this date (YYYY-MM-DD)
        end_date: Filter sessions on/before this date (YYYY-MM-DD)
        limit: Maximum number of sessions per club to process
        refresh: If True, re-fetch club HTML pages
        sleep_seconds: Seconds to sleep between requests (default: 2)
        
    Returns:
        Total number of sessions processed across all clubs
    """
    # Get list of all clubs
    while True:
        try:
            club_ids = get_all_clubs()
            break
        except Forbidden403Error as e:
            print(f"\n403 Forbidden while fetching club list: {e}")
            print(f"Waiting {sleep_403_seconds} seconds, then retrying...")
            time.sleep(sleep_403_seconds)
    
    if not club_ids:
        print("No clubs found or error occurred")
        return 0

    # Ensure deterministic ascending order and apply optional starting-club filter.
    # Club IDs are expected to be numeric strings.
    numeric_clubs: list[int] = []
    for cid in club_ids:
        if isinstance(cid, str) and cid.isdigit():
            numeric_clubs.append(int(cid))
    numeric_clubs.sort()

    total_all_clubs = len(numeric_clubs)
    club_to_position = {club: idx for idx, club in enumerate(numeric_clubs, start=1)}

    if starting_club is not None:
        numeric_clubs = [c for c in numeric_clubs if c >= starting_club]
    
    print(f"\nProcessing {len(numeric_clubs)} clubs...")
    print("=" * 70)
    
    total_processed = 0
    
    for i, club_int in enumerate(numeric_clubs, 1):
        club_id = str(club_int)
        # Show progress as "<nth in full list>/<total clubs>" (requested behavior).
        pos = club_to_position.get(club_int, i)
        club_dir = pathlib.Path(output_dir) / club_id
        details_dir = club_dir / "details"
        club_has_data = details_dir.exists() and any(details_dir.glob("*.data.json"))
        if skip_if_html_exists and club_has_data:
            print(f"\n[Club {pos}/{total_all_clubs}] Skipping club {club_id} (club already exists)")
            continue

        print(f"\n[Club {pos}/{total_all_clubs}] Processing club {club_id}...")
        
        while True:
            try:
                processed = process_club_sessions(club_id, output_dir, start_date, end_date, limit, refresh, sleep_seconds)
                total_processed += processed
                break
            except Forbidden403Error as e:
                print(f"\n403 Forbidden while processing club {club_id}: {e}")
                print(f"Waiting {sleep_403_seconds} seconds, then reprocessing this club...")
                time.sleep(sleep_403_seconds)
                continue
            except Exception as e:
                print(f"ERROR processing club {club_id}: {e}")
                break
    
    print("\n" + "=" * 70)
    print(f"ALL CLUBS SUMMARY:")
    print(f"  Total clubs processed: {len(numeric_clubs)}")
    print(f"  Total sessions extracted: {total_processed}")
    print("=" * 70)
    
    return total_processed


def main():
    """Main entry point"""
    # Default session ID
    session_id = None
    html_file = None
    club_number = None
    player_id = None
    all_clubs = True   # Process all clubs (default mode)
    output_dir = "club-results"  # Default subdirectory under acblPath
    start_date = None  # Filter: sessions on or after this date (YYYY-MM-DD)
    end_date = None    # Filter: sessions on or before this date (YYYY-MM-DD), defaults to today
    limit = None       # Limit number of sessions to fetch (None = unlimited)
    end_date_specified = False  # Track if user explicitly set end_date
    refresh = True     # Re-fetch club HTML to get latest sessions (default); --no-refresh to use cache
    sleep_seconds = 2  # Seconds to sleep between requests
    sleep_403_seconds = 300  # Seconds to wait on HTTP 403 before retry (default: 5 minutes)
    starting_club = None  # In all-clubs mode, start at this club number (inclusive)
    skip_if_html_exists = False  # Default: process all clubs; individual sessions skip if data.json exists
    remove_all_html_files = False  # Default: keep <club>/<club>.html cache on startup
    
    # Show usage if --help
    if '--help' in sys.argv or '-h' in sys.argv:
        print("=" * 70)
        print("ACBL Club Downloader - Usage")
        print("=" * 70)
        print()
        print("Download and extract ACBL club session data")
        print()
        print("Usage:")
        print("  python acbl_club_downloader.py                        # All clubs (default)")
        print("  python acbl_club_downloader.py --session <session_id>")
        print("  python acbl_club_downloader.py --club <club_number>")
        print("  python acbl_club_downloader.py --player <player_id>")
        print("  python acbl_club_downloader.py <file.html>")
        print()
        print("Examples:")
        print("  # Fetch single session")
        print("  python acbl_club_downloader.py --session 993420")
        print()
        print("  # Extract ALL sessions for a club")
        print("  python acbl_club_downloader.py --club 267096")
        print()
        print("  # Extract ALL sessions for a player")
        print("  python acbl_club_downloader.py --player 2663279")
        print()
        print("  # Extract ALL clubs (default mode - warning: very large operation)")
        print("  python acbl_club_downloader.py --limit 5")
        print()
        print("  # Resume all-clubs mode from a specific club number")
        print("  python acbl_club_downloader.py --starting-club 200000")
        print()
        print("  # Skip clubs already downloaded (default behavior)")
        print("  python acbl_club_downloader.py --skip-if-html-exists")
        print()
        print("  # Extract sessions with filters")
        print("  python acbl_club_downloader.py --club 267096 --start-date 2024-01-01")
        print("  python acbl_club_downloader.py --player 2663279 --limit 5")
        print("  python acbl_club_downloader.py --start-date 2024-09-01 --limit 1")
        print()
        print("  # Customize delay between requests (faster/slower)")
        print("  python acbl_club_downloader.py --club 267096 --sleep 5   # 5 seconds")
        print("  python acbl_club_downloader.py --club 267096 --sleep 30  # 30 seconds")
        print()
        print("  # Extract from local HTML file")
        print("  python acbl_club_downloader.py acbl-results-993420.html")
        print()
        print("Options:")
        print("  --session <id>         Extract a single session by ID")
        print("  --club <number>        Extract all sessions for a club")
        print("  --player <id>          Extract all sessions for a player")
        print("  --start-date <date>    Filter sessions on/after date (YYYY-MM-DD) (default: first available date)")
        print("  --end-date <date>      Filter sessions on/before date (YYYY-MM-DD) (default: today)")
        print("  --limit <n>            Limit number of sessions to fetch per club (default: unlimited)")
        print("  --sleep <seconds>      Delay between session requests (default: 2 seconds)")
        print("  --sleep-403 <seconds>  Wait time before retry on 403 (default: 300 seconds / 5 minutes)")
        print("  --starting-club <n>    In all-clubs mode, start at club >= n (default: none)")
        print("  --skip-if-html-exists  Skip clubs where <club>/details/*.data.json already exists")
        print("  --no-skip-if-html-exists  Process all clubs; individual sessions skip if data.json exists (default)")
        print("  --remove-all-html-files     Delete <club>/<club>.html files on startup")
        print("  --no-remove-all-html-files  Keep <club>/<club>.html cache on startup (default)")
        print("  --output-dir <path>    Output directory (default: club-results under acblPath)")
        print("  --refresh              Re-fetch club HTML pages to get latest sessions (default)")
        print("  --no-refresh           Use cached club HTML files instead of re-fetching")
        print("  --help, -h             Show this help message")
        print()
        print("Note: Default mode is ALL clubs; --session, --club, --player override this")
        print("      Sessions are processed sequentially with delays to avoid rate limiting")
        print("      Default delay between session requests is 2 seconds (configurable with --sleep)")
        print("      On 403 Forbidden, program will wait and retry (see --sleep-403)")
        print()
        return 0
    
    # Parse command-line arguments
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg == '--session':
            if i + 1 < len(sys.argv):
                session_id = sys.argv[i + 1]
                all_clubs = False  # Disable default all-clubs mode
                i += 2
            else:
                print("ERROR: --session requires a session ID")
                return 1
        elif arg == '--club':
            if i + 1 < len(sys.argv):
                club_number = sys.argv[i + 1]
                all_clubs = False  # Disable default all-clubs mode
                i += 2
            else:
                print("ERROR: --club requires a club number")
                return 1
        elif arg == '--player':
            if i + 1 < len(sys.argv):
                player_id = sys.argv[i + 1]
                all_clubs = False  # Disable default all-clubs mode
                i += 2
            else:
                print("ERROR: --player requires a player ID")
                return 1
        elif arg == '--start-date':
            if i + 1 < len(sys.argv):
                start_date = sys.argv[i + 1]
                i += 2
            else:
                print("ERROR: --start-date requires a date (YYYY-MM-DD)")
                return 1
        elif arg == '--end-date':
            if i + 1 < len(sys.argv):
                end_date = sys.argv[i + 1]
                end_date_specified = True
                i += 2
            else:
                print("ERROR: --end-date requires a date (YYYY-MM-DD)")
                return 1
        elif arg == '--limit':
            if i + 1 < len(sys.argv):
                try:
                    limit = int(sys.argv[i + 1])
                    i += 2
                except ValueError:
                    print("ERROR: --limit requires an integer")
                    return 1
            else:
                print("ERROR: --limit requires a number")
                return 1
        elif arg == '--sleep':
            if i + 1 < len(sys.argv):
                try:
                    sleep_seconds = int(sys.argv[i + 1])
                    if sleep_seconds < 0:
                        print("ERROR: --sleep requires a positive integer")
                        return 1
                    i += 2
                except ValueError:
                    print("ERROR: --sleep requires an integer")
                    return 1
            else:
                print("ERROR: --sleep requires a number of seconds")
                return 1
        elif arg == '--sleep-403':
            if i + 1 < len(sys.argv):
                try:
                    sleep_403_seconds = int(sys.argv[i + 1])
                    if sleep_403_seconds < 0:
                        print("ERROR: --sleep-403 requires a non-negative integer")
                        return 1
                    i += 2
                except ValueError:
                    print("ERROR: --sleep-403 requires an integer number of seconds")
                    return 1
            else:
                print("ERROR: --sleep-403 requires a number of seconds")
                return 1
        elif arg == '--starting-club':
            if i + 1 < len(sys.argv):
                try:
                    starting_club = int(sys.argv[i + 1])
                    if starting_club < 0:
                        print("ERROR: --starting-club requires a non-negative integer")
                        return 1
                    i += 2
                except ValueError:
                    print("ERROR: --starting-club requires an integer")
                    return 1
            else:
                print("ERROR: --starting-club requires a club number")
                return 1
        elif arg == '--skip-if-html-exists':
            skip_if_html_exists = True
            i += 1
        elif arg == '--no-skip-if-html-exists':
            skip_if_html_exists = False
            i += 1
        elif arg == '--remove-all-html-files':
            remove_all_html_files = True
            i += 1
        elif arg == '--no-remove-all-html-files':
            remove_all_html_files = False
            i += 1
        elif arg == '--output-dir':
            if i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
                i += 2
            else:
                print("ERROR: --output-dir requires a path (relative to acblPath or absolute)")
                return 1
        elif arg == '--refresh':
            refresh = True
            i += 1
        elif arg == '--no-refresh':
            refresh = False
            i += 1
        elif arg.endswith('.html'):
            html_file = arg
            all_clubs = False  # Disable default --all mode
            # Extract session ID from filename if possible
            match = re.search(r'(\d+)', html_file)
            if match:
                session_id = match.group(1)
            i += 1
        elif arg.startswith('-'):
            # Unknown flag
            i += 1
        else:
            # Backward compatibility: assume it's a session ID if no mode specified
            if not club_number and not player_id and not html_file:
                session_id = arg
            i += 1
    
    # Check for mutually exclusive modes
    mode_count = sum([bool(club_number), bool(player_id), bool(html_file), bool(all_clubs)])
    if mode_count > 1:
        print("ERROR: --club, --player, and file modes are mutually exclusive")
        print("Please specify only one mode.")
        return 1
    
    # Set default end_date to today if start_date is specified but end_date isn't
    if start_date and not end_date_specified:
        end_date = date.today().strftime('%Y-%m-%d')
    
    print("=" * 70)
    print("ACBL Club Downloader")
    print("=" * 70)
    
    # Normalize output_dir to a pathlib.Path; if relative, make relative to acblPath
    if isinstance(output_dir, str):
        out_path = pathlib.Path(output_dir)
        if not out_path.is_absolute():
            output_dir = acblPath.joinpath(out_path)
        else:
            output_dir = out_path

    # Startup cleanup: remove all <club>/<club>.html under the output directory
    if remove_all_html_files:
        remove_club_html_files_on_startup(pathlib.Path(output_dir))

    # All clubs mode - extract sessions from all clubs
    if all_clubs:
        if starting_club is not None:
            print(f"Mode: ALL CLUBS (starting at club >= {starting_club})")
        else:
            print(f"Mode: ALL CLUBS")
        if start_date or end_date or limit or refresh:
            filters = []
            if start_date:
                filters.append(f"start>={start_date}")
            if end_date:
                end_date_label = f"end<={end_date}"
                if not end_date_specified:
                    end_date_label += " (today)"
                filters.append(end_date_label)
            if limit:
                filters.append(f"limit={limit} per club")
            if refresh:
                filters.append("refresh=true")
            print(f"Filters: {', '.join(filters)}")
        print()
        
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright is required for all-clubs mode")
            print("Install with: pip install playwright && playwright install")
            return 1
        
        if skip_if_html_exists:
            print("Option: --skip-if-html-exists enabled (default)")
        print()

        processed = process_all_clubs(
            output_dir,
            start_date,
            end_date,
            limit,
            refresh,
            sleep_seconds,
            starting_club=starting_club,
            skip_if_html_exists=skip_if_html_exists,
            sleep_403_seconds=sleep_403_seconds,
        )
        return 0 if processed > 0 else 1
    
    # Club mode - extract all sessions
    if club_number:
        print(f"Mode: Club crawl")
        print(f"Club number: {club_number}")
        if skip_if_html_exists:
            club_dir = pathlib.Path(output_dir) / club_number
            details_dir = club_dir / "details"
            club_has_data = details_dir.exists() and any(details_dir.glob("*.data.json"))
            if club_has_data:
                print(f"Skip: club already exists: {details_dir}")
                return 0
        if start_date or end_date or limit or refresh:
            filters = []
            if start_date:
                filters.append(f"start>={start_date}")
            if end_date:
                end_date_label = f"end<={end_date}"
                if not end_date_specified:
                    end_date_label += " (today)"
                filters.append(end_date_label)
            if limit:
                filters.append(f"limit={limit}")
            if refresh:
                filters.append("refresh=true")
            print(f"Filters: {', '.join(filters)}")
        print()
        
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright is required for club mode")
            print("Install with: pip install playwright && playwright install")
            return 1
        
        while True:
            try:
                processed = process_club_sessions(club_number, output_dir, start_date, end_date, limit, refresh, sleep_seconds)
                break
            except Forbidden403Error as e:
                print(f"\n403 Forbidden while processing club {club_number}: {e}")
                print(f"Waiting {sleep_403_seconds} seconds, then reprocessing this club...")
                time.sleep(sleep_403_seconds)
                continue
        return 0 if processed > 0 else 1
    
    # Player mode - extract all sessions for a player
    if player_id:
        print(f"Mode: Player history crawl")
        print(f"Player ID: {player_id}")
        if start_date or end_date or limit or refresh:
            filters = []
            if start_date:
                filters.append(f"start>={start_date}")
            if end_date:
                end_date_label = f"end<={end_date}"
                if not end_date_specified:
                    end_date_label += " (today)"
                filters.append(end_date_label)
            if limit:
                filters.append(f"limit={limit}")
            if refresh:
                filters.append("refresh=true")
            print(f"Filters: {', '.join(filters)}")
        print()
        
        if not PLAYWRIGHT_AVAILABLE:
            print("ERROR: Playwright is required for player mode")
            print("Install with: pip install playwright && playwright install")
            return 1
        
        while True:
            try:
                processed = process_player_sessions(player_id, output_dir, start_date, end_date, limit, refresh, sleep_seconds)
                break
            except Forbidden403Error as e:
                print(f"\n403 Forbidden while processing player {player_id}: {e}")
                print(f"Waiting {sleep_403_seconds} seconds, then retrying...")
                time.sleep(sleep_403_seconds)
                continue
        return 0 if processed > 0 else 1
    
    # Single session mode
    print(f"Mode: Single session")
    print(f"Session ID: {session_id}")
    print()
    
    # Check if file exists in any club directory first
    base_dir = acblPath.joinpath(output_dir)
    existing_file = None
    if base_dir.exists():
        for club_dir in base_dir.iterdir():
            if club_dir.is_dir() and club_dir.name.isdigit():
                details_dir = club_dir / "details"
                session_file = details_dir / f"{session_id}.data.json"
                if session_file.exists():
                    existing_file = session_file
                    break
    
    if existing_file:
        print(f"File already exists: {existing_file}")
        print("Skipping extraction.")
        return 0
    
    # Construct URL
    url = f"https://my.acbl.org/club-results/details/{session_id}"
    
    # Extract JSON - try file first, then Playwright
    data = None
    if html_file:
        print(f"Extracting from local file: {html_file}")
        data = extract_from_file(html_file)
    else:
        while True:
            try:
                print(f"Fetching with Playwright: {url}")
                data = extract_with_playwright(url)
                break
            except Forbidden403Error as e:
                print(f"\n403 Forbidden while fetching session {session_id}: {e}")
                print(f"Waiting {sleep_403_seconds} seconds, then retrying...")
                time.sleep(sleep_403_seconds)
                continue
    
    if data:
        # Extract club_id from JSON to organize properly
        club_id = str(data.get('club_id_number', ''))
        
        if not club_id:
            print("WARNING: No club_id_number found in JSON")
            print("Cannot organize file properly - extraction failed")
            return 1
        
        # Save to club-results/<club_id>/details/<session_id>.data.json
        details_dir = acblPath.joinpath(output_dir) / club_id / "details"
        details_dir.mkdir(parents=True, exist_ok=True)
        output_file = details_dir / f"{session_id}.data.json"
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully extracted JSON data")
        print(f"  -> Saved to {output_file}")
        print(f"File size: {len(json.dumps(data, indent=2)):,} bytes")
        print()
        print("Data Summary:")
        print(f"  Event: {data.get('name', 'N/A')}")
        print(f"  Club: {data.get('club_name', 'N/A')}")
        print(f"  Date: {data.get('start_date', 'N/A')}")
        print(f"  Type: {data.get('type', 'N/A')}")
        print(f"  Scoring: {data.get('board_scoring_method', 'N/A')}")
        
        # Show structure
        if 'sessions' in data and data['sessions']:
            session = data['sessions'][0]
            print(f"\nSessions: {len(data['sessions'])}")
            
            if 'sections' in session and session['sections']:
                section = session['sections'][0]
                print(f"Sections: {len(session['sections'])}")
                
                if 'boards' in section and section['boards']:
                    print(f"Boards: {len(section['boards'])}")
                    
                    board = section['boards'][0]
                    if 'board_results' in board and board['board_results']:
                        total_results = sum(len(b.get('board_results', [])) for b in section['boards'])
                        print(f"Total board results: {total_results}")
                
                if 'pair_summaries' in section and section['pair_summaries']:
                    print(f"Pair summaries: {len(section['pair_summaries'])}")
        
        print()
        print("Top-level keys available:")
        for i, key in enumerate(list(data.keys())[:15]):
            print(f"  - {key}")
        if len(data) > 15:
            print(f"  ... and {len(data) - 15} more")
        
        return 0
    else:
        print()
        print("=" * 70)
        print("EXTRACTION FAILED")
        print("=" * 70)
        if not html_file and not PLAYWRIGHT_AVAILABLE:
            print("Note: Playwright is not available.")
            print("Install with: pip install playwright && playwright install")
        return 1


if __name__ == "__main__":
    from mlBridge import print_started, print_ended
    program_start_time = print_started()
    try:
        main()
        _shutdown_browser()
        print_ended(program_start_time)
        sys.exit(0) # todo: is this still necessary? was used to force exit on http 403 error (bypass playwright async on ACBL rate limiting)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user (Ctrl+C)")
        _shutdown_browser()
        print_ended(program_start_time)
        print("Exiting...")
        os._exit(130)  # Standard exit code for SIGINT

