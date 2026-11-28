"""X-ACBL-History-Only: 1 (sent by the MCP's backends) never reaches my.acbl.org or the tournament API."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import acbl_club_api_server as server
import acbl_club_api_service as svc

HISTORY = [
    {
        "session_id": "100",
        "date": "2020-01-01",
        "club_name": "Old Club",
        "details_url": "https://my.acbl.org/club-results/details/100",
    }
]
HEADER = {"X-ACBL-History-Only": "1"}


def _no_live(*_args, **_kwargs):
    raise AssertionError("live fetch in a history-only request")


class HistoryOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._patches = [
            patch.object(svc, "CACHE_DIR", Path(self._tmp.name)),
            patch.object(svc, "_player_games_from_parquet", side_effect=lambda pid: [dict(r) for r in HISTORY]),
            patch.object(svc, "_player_games_from_handoff", return_value=None),
        ]
        for p in self._patches:
            p.start()
        self.client = TestClient(server.app)

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()

    def test_refresh_is_answered_from_history(self) -> None:
        with patch.object(svc, "_run_in_thread_with_new_loop", side_effect=_no_live):
            answer = self.client.get("/players/2663279/games", params={"limit": 3, "refresh": True},
                                     headers=HEADER).json()
        self.assertEqual(answer["rows"][0]["session_id"], "100")
        self.assertFalse(answer["meta"]["refresh_failed"])
        self.assertEqual(answer["meta"]["refresh_attempts"], [])
        self.assertFalse(svc.HISTORY_ONLY.get())  # reset after the request

    def test_without_the_header_refresh_still_goes_live(self) -> None:
        calls = []

        def live(_fn, *_args, **_kwargs):
            calls.append(1)
            return "<html></html>", True

        with patch.object(svc, "NAV_SLEEP_SECONDS", 0.0), \
                patch.object(svc, "_run_in_thread_with_new_loop", side_effect=live):
            self.client.get("/players/2663279/games", params={"limit": 3, "refresh": True})
        self.assertEqual(len(calls), 1)

    def test_live_helpers_refuse_inside_a_history_only_request(self) -> None:
        token = svc.HISTORY_ONLY.set(True)
        try:
            with self.assertRaises(svc.HistoryOnlyError) as caught:
                svc._playwright_paginated_html("https://my.acbl.org/club-results/my-results/1")
            self.assertEqual(caught.exception.status_code, 404)
            with self.assertRaises(svc.HistoryOnlyError):
                with svc._scrape_slot():
                    pass
            with self.assertRaises(svc.HistoryOnlyError):
                svc._tournament_api_get("https://api.acbl.org/v1/tournament/session")
        finally:
            svc.HISTORY_ONLY.reset(token)
        self.assertFalse(svc._SCRAPE_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
