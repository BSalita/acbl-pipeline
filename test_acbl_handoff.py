"""A browser-loaded ACBL page is cached and served instead of Playwright."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import acbl_club_api_service as svc


def _pad(html: str) -> str:
    return f"<!-- {'x' * 240} -->\n{html}"


CLUB_HTML = _pad(
    """
<html><body>
<div class="col-md-8"><h1>Test Club</h1></div>
<table><tr>
<td>09/26/2026</td>
<td><a href="/club-results/details/1513830">Friday</a></td>
</tr></table>
</body></html>
"""
)

PLAYER_HTML = _pad(
    """
<html><body><table><tr>
<td>09/26/2026</td>
<td><a href="/club-results/108571">Test Club</a></td>
<td><a href="/club-results/details/1513830">Friday</a></td>
<td>Open</td>
<td></td>
<td>62%</td>
</tr></table></body></html>
"""
)

SESSION_HTML = """
<html><body>
<result-details v-bind:data='{"club_id_number":"267096","event_name":"Friday Open","boards":[]}'></result-details>
</body></html>
"""


class AcblBrowserHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self._cache_patch = patch.object(svc, "CACHE_DIR", self.cache)
        self._cache_patch.start()

    def tearDown(self) -> None:
        self._cache_patch.stop()
        self._tmp.cleanup()

    def test_challenge_page_is_rejected(self) -> None:
        html = "<html><title>Just a moment...</title><p>Checking your browser</p></html>"
        with self.assertRaises(svc.ClubApiError) as caught:
            svc.ingest_browser_handoff("https://my.acbl.org/club-results/108571", html)
        self.assertEqual(caught.exception.status_code, 409)

    def test_club_page_is_served_without_playwright(self) -> None:
        saved = svc.ingest_browser_handoff(
            "https://my.acbl.org/club-results/108571", CLUB_HTML
        )
        self.assertEqual(saved["rows"], 1)
        self.assertEqual(saved["kind"], "club")

        def scrape(*_args, **_kwargs):
            raise AssertionError("playwright")

        with (
            patch.object(svc, "_club_games_from_parquet", return_value=[]),
            patch.object(svc, "_playwright_paginated_html", side_effect=scrape),
        ):
            table = svc.club_games("108571", refresh=True)
        self.assertEqual(table["meta"]["source"], "browser-handoff")
        self.assertEqual(table["rows"][0]["session_id"], "1513830")
        self.assertEqual(table["meta"]["club"]["club_name"], "Test Club")

    def test_fresh_club_page_leads_parquet_history(self) -> None:
        svc.ingest_browser_handoff("https://my.acbl.org/club-results/108571", CLUB_HTML)
        history = [
            {
                "session_id": "100",
                "date": "01/01/2020",
                "event_name": "Old",
                "club_id": "108571",
                "details_url": "https://my.acbl.org/club-results/details/100",
            }
        ]
        with patch.object(svc, "_club_games_from_parquet", return_value=history):
            table = svc.club_games("108571")
        self.assertEqual(table["meta"]["source"], "browser-handoff")
        self.assertEqual(
            [row["session_id"] for row in table["rows"]],
            ["1513830", "100"],
        )

    def test_stale_club_page_does_not_override_parquet(self) -> None:
        svc.ingest_browser_handoff("https://my.acbl.org/club-results/108571", CLUB_HTML)
        html_path = self.cache / "108571" / "108571.handoff.html"
        old = time.time() - svc.HANDOFF_TTL_S - 60
        os.utime(html_path, (old, old))
        history = [
            {
                "session_id": "100",
                "date": "01/01/2020",
                "event_name": "Old",
                "club_id": "108571",
            }
        ]
        with patch.object(svc, "_club_games_from_parquet", return_value=history):
            table = svc.club_games("108571")
        self.assertEqual(table["meta"]["source"], "parquet")
        self.assertEqual(table["rows"][0]["session_id"], "100")

    def test_player_page_is_served_without_playwright(self) -> None:
        saved = svc.ingest_browser_handoff(
            "https://my.acbl.org/club-results/my-results/2663279",
            PLAYER_HTML,
        )
        self.assertEqual(saved["kind"], "player")
        self.assertEqual(saved["rows"], 1)

        def scrape(*_args, **_kwargs):
            raise AssertionError("playwright")

        with (
            patch.object(svc, "_player_games_from_parquet", return_value=[]),
            patch.object(svc, "_playwright_paginated_html", side_effect=scrape),
        ):
            table = svc.player_games("2663279", refresh=True)
        self.assertEqual(table["meta"]["source"], "browser-handoff")
        self.assertEqual(table["rows"][0]["session_id"], "1513830")

    def test_session_page_skips_a_live_refetch(self) -> None:
        saved = svc.ingest_browser_handoff(
            "https://my.acbl.org/club-results/details/993420",
            SESSION_HTML,
        )
        self.assertEqual(saved["kind"], "session")
        self.assertEqual(saved["club_id"], "267096")

        def scrape(*_args, **_kwargs):
            raise AssertionError("playwright")

        with patch.object(
            svc, "get_club_results_details_data_playwright", side_effect=scrape
        ):
            data, cached, path = svc._fetch_session_json("993420", refresh=True)
        self.assertTrue(cached)
        self.assertEqual(data["event_name"], "Friday Open")
        self.assertEqual(path.name, "993420.data.json")

    def test_session_page_lets_a_club_postmortem_build(self) -> None:
        seen = {}

        def augmented(session_id, player_id=None, refresh=False, allow_build=True):
            seen["allow_build"] = allow_build
            raise svc.ClubApiError("stop", status_code=418)

        with patch.object(svc, "session_augmented_parquet", side_effect=augmented):
            with self.assertRaises(svc.ClubApiError):
                svc.postmortem_dataframe("993420", "2663279")
            self.assertFalse(seen["allow_build"])
            svc.ingest_browser_handoff(
                "https://my.acbl.org/club-results/details/993420",
                SESSION_HTML,
            )
            with self.assertRaises(svc.ClubApiError):
                svc.postmortem_dataframe("993420", "2663279")
        self.assertTrue(seen["allow_build"])


class HistoricalSessionMissTests(unittest.TestCase):
    def test_absent_session_is_scanned_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            monolith = Path(tmp) / "augmented.parquet"
            monolith.write_bytes(b"x")
            with (
                patch.object(svc, "AUGMENTED_PARQUET_FILE", monolith),
                patch.object(svc, "_AUGMENTED_SESSION_MISSES", {}),
                patch.object(svc, "_record_club_parquet_query"),
                patch.object(
                    svc, "_scan_historical_session", return_value=None
                ) as scan,
            ):
                self.assertIsNone(svc._historical_augmented_parquet("777"))
                self.assertIsNone(svc._historical_augmented_parquet("777"))
                self.assertIsNone(svc._historical_postmortem_lazy("777"))
        self.assertEqual(scan.call_count, 1)
