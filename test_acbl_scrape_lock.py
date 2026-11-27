"""A live fetch stuck on ACBL's Cloudflare check must not hold up other requests.

The challenge waiter can wait hours for a human with the scrape lock held; a
player-games refresh then answers from history after SCRAPE_LOCK_WAIT_S.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import acbl_club_api_service as svc

HISTORY = [
    {
        "session_id": "100",
        "date": "2020-01-01",
        "club_name": "Old Club",
        "details_url": "https://my.acbl.org/club-results/details/100",
    }
]


class ScrapeLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patches = [
            patch.object(svc, "CACHE_DIR", Path(self._tmp.name)),
            patch.object(svc, "SCRAPE_LOCK_WAIT_S", 0.2),
            patch.object(svc, "NAV_SLEEP_SECONDS", 0.0),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_busy_browser_raises_busy_error(self) -> None:
        self.assertTrue(svc._SCRAPE_LOCK.acquire(timeout=1))
        try:
            started = time.monotonic()
            with self.assertRaises(svc.ScrapeBusyError) as caught:
                svc._playwright_paginated_html("https://my.acbl.org/club-results/my-results/1")
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertEqual(caught.exception.status_code, 503)
        finally:
            svc._SCRAPE_LOCK.release()

    def test_player_refresh_answers_from_history_while_browser_busy(self) -> None:
        self.assertTrue(svc._SCRAPE_LOCK.acquire(timeout=1))
        try:
            with (
                patch.object(svc, "_player_games_from_parquet", return_value=[dict(r) for r in HISTORY]),
                patch.object(svc, "_player_games_from_handoff", return_value=None),
            ):
                started = time.monotonic()
                table = svc.player_games("2663279", limit=3, refresh=True)
                elapsed = time.monotonic() - started
        finally:
            svc._SCRAPE_LOCK.release()
        self.assertLess(elapsed, 2.0)
        meta = table["meta"]
        self.assertTrue(meta["refresh_failed"])
        self.assertEqual(len(meta["refresh_attempts"]), 1)  # no second wait
        self.assertEqual(meta["refresh_attempts"][0]["status_code"], 503)
        self.assertEqual(table["rows"][0]["session_id"], "100")

    def test_slow_fetch_stops_the_wait_but_keeps_the_browser_until_it_ends(self) -> None:
        def slow(_fn, *_args, **_kwargs):
            time.sleep(1.0)
            return "<html></html>", True

        with (
            patch.object(svc, "SCRAPE_RESULT_WAIT_S", 0.2),
            patch.object(svc, "_run_in_thread_with_new_loop", side_effect=slow),
        ):
            started = time.monotonic()
            with self.assertRaises(svc.ScrapeBusyError):
                svc._playwright_paginated_html("https://my.acbl.org/club-results/my-results/1")
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertTrue(svc._SCRAPE_LOCK.locked())  # the fetch still has the browser
            self.assertTrue(svc._SCRAPE_LOCK.acquire(timeout=3))  # and lets it go when it ends
            svc._SCRAPE_LOCK.release()

    def test_fast_fetch_returns_its_result_and_errors(self) -> None:
        with patch.object(svc, "_run_in_thread_with_new_loop", return_value=("<html>x</html>", True)):
            self.assertEqual(svc._playwright_paginated_html("https://x"), ("<html>x</html>", True))
        with patch.object(svc, "_run_in_thread_with_new_loop", side_effect=RuntimeError("boom")):
            with self.assertRaises(svc.ClubApiError) as caught:
                svc._playwright_paginated_html("https://x")
            self.assertEqual(caught.exception.status_code, 502)
        self.assertFalse(svc._SCRAPE_LOCK.locked())

    def test_lock_is_released_after_a_fetch(self) -> None:
        with svc._scrape_slot():
            self.assertTrue(svc._SCRAPE_LOCK.locked())
        self.assertFalse(svc._SCRAPE_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
