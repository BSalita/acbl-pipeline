"""ACBL will not renew the tournament token: after a 401 the API is not called again."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import acbl_club_api_service as svc


def _response(status: int) -> MagicMock:
    response = MagicMock()
    response.ok = status < 400
    response.status_code = status
    response.json.return_value = {"message": "Unauthenticated."}
    return response


class TournamentApiRejectedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._patches = [
            patch.object(svc, "_TOURNAMENT_API_REJECTED", None),
            patch.object(svc, "ACBL_API_KEY", "test-token"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()

    def test_after_a_401_the_api_is_not_called_again(self) -> None:
        with patch.object(svc.requests, "get", return_value=_response(401)) as get:
            with self.assertRaises(svc.ClubApiError) as first:
                svc._tournament_api_get("https://api.acbl.org/v1/tournament/player/history_query")
            self.assertEqual(first.exception.status_code, 401)
            with self.assertRaises(svc.ClubApiError) as second:
                svc._tournament_api_get("https://api.acbl.org/v1/tournament/player/history_query")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(second.exception.status_code, 503)
        self.assertIn("token was rejected (HTTP 401", second.exception.detail)

    def test_other_errors_do_not_disable_the_api(self) -> None:
        with patch.object(svc.requests, "get", return_value=_response(500)) as get:
            for _ in range(2):
                with self.assertRaises(svc.ClubApiError):
                    svc._tournament_api_get("https://api.acbl.org/v1/tournament/session")
        self.assertEqual(get.call_count, 2)
        self.assertIsNone(svc._TOURNAMENT_API_REJECTED)

    def test_player_sessions_report_history_only(self) -> None:
        with patch.object(svc, "_TOURNAMENT_API_REJECTED", "its token was rejected (HTTP 401)"), \
                patch.object(svc, "_historical_tournament_player_sessions", return_value=[]):
            table = svc.tournament_player_sessions("2663279", limit=3, refresh=True)
        meta = table["meta"]
        self.assertTrue(meta["refresh_failed"])
        self.assertFalse(meta["live_api_available"])


if __name__ == "__main__":
    unittest.main()
