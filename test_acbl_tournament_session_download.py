from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

import acbl_elo_ratings_create
from acbl_tournament_download_sessions_using_sanctioned_events import (
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    DownloadStats,
    audit_session_artifacts,
    build_session_ids_from_sanctioned_events,
    canonical_event_id,
    download_tournament_sessions,
    sessions_still_needed,
    expired_api_key_message,
    is_unavailable_http,
    jwt_expiry_unix,
    api_key_rejection_reason,
    session_request_timeout,
)


class TournamentSessionDiscoveryTests(unittest.TestCase):
    def test_uses_canonical_event_id_for_nabc_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            event = {
                "id": "NABC262-OSHL",
                "sanction": "2607001",
                "event_code": "OSHL",
                "session_count": 4,
            }
            (events_dir / "NABC262-OSHL.sanction.json").write_text(
                json.dumps(event),
                encoding="utf-8",
            )

            self.assertEqual(
                build_session_ids_from_sanctioned_events(events_dir),
                [
                    "NABC262-OSHL-4",
                    "NABC262-OSHL-3",
                    "NABC262-OSHL-2",
                    "NABC262-OSHL-1",
                ],
            )

    def test_rebuilds_truncated_hyphen_event_ids(self) -> None:
        event = {
            "id": "-0810",
            "sanction": "1803020",
            "event_code": "0810",
            "session_count": 3,
        }
        self.assertEqual(canonical_event_id(event), "1803020-0810")
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            (events_dir / "-0810.sanction.json").write_text(
                json.dumps(event),
                encoding="utf-8",
            )
            self.assertEqual(
                build_session_ids_from_sanctioned_events(events_dir),
                ["1803020-0810-3", "1803020-0810-2", "1803020-0810-1"],
            )

    def test_expired_jwt_is_detected(self) -> None:
        payload = json.dumps({"exp": 1_000_000_000}).encode()
        padded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        token = f"eyJhbGciOiJub25lIn0.{padded}.sig"
        self.assertEqual(jwt_expiry_unix(token), 1_000_000_000)
        self.assertIsNotNone(
            expired_api_key_message(token, now=1_000_000_000 + 120)
        )
        self.assertIsNone(expired_api_key_message(token, now=1_000_000_000))
        self.assertIsNone(expired_api_key_message("not-a-jwt"))
        self.assertIsNotNone(api_key_rejection_reason(None))
        self.assertIsNotNone(api_key_rejection_reason("  "))
        self.assertIsNotNone(api_key_rejection_reason(token, now=1_000_000_000 + 120))
        self.assertIsNone(api_key_rejection_reason("not-a-jwt"))

    def test_audit_reports_only_sessions_without_json_or_sql(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            (sessions_dir / "event-1.session.json").write_text("{}", encoding="utf-8")
            (sessions_dir / "event-2.session.sql").write_text("-- ok", encoding="utf-8")

            missing = audit_session_artifacts(
                ["event-1", "event-2", "event-3"],
                sessions_dir,
            )

            self.assertEqual(missing, ["event-3"])
            audit = json.loads(
                (sessions_dir / "_session_download_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["expected_sessions"], 3)
            self.assertEqual(audit["complete_sessions"], 2)
            self.assertEqual(audit["unavailable_sessions"], [])
            self.assertEqual(audit["unresolved_sessions"], ["event-3"])

    def test_audit_separates_unavailable_from_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sessions_dir = Path(tmp)
            missing = audit_session_artifacts(
                ["ok", "gone", "later"],
                sessions_dir,
                unavailable_ids=["gone"],
            )
            self.assertEqual(missing, ["ok", "gone", "later"])
            audit = json.loads(
                (sessions_dir / "_session_download_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(audit["unavailable_sessions"], ["gone"])
            self.assertEqual(audit["unresolved_sessions"], ["ok", "later"])

    def test_session_request_timeout_splits_connect_and_read(self) -> None:
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 90)
        self.assertEqual(
            session_request_timeout(90),
            (float(DEFAULT_CONNECT_TIMEOUT_SECONDS), 90.0),
        )
        self.assertEqual(session_request_timeout(10), (10.0, 10.0))
        with self.assertRaises(ValueError):
            session_request_timeout(0)

    def test_unavailable_http_is_only_400_and_404(self) -> None:
        self.assertTrue(is_unavailable_http(400))
        self.assertTrue(is_unavailable_http(404))
        self.assertFalse(is_unavailable_http(429))
        self.assertFalse(is_unavailable_http(500))
        self.assertFalse(is_unavailable_http(200))

    def test_download_stats_fail_only_on_hard_errors(self) -> None:
        self.assertFalse(DownloadStats(unavailable=12).failed())
        self.assertTrue(DownloadStats(errors=1).failed())
        self.assertTrue(DownloadStats(aborted=True).failed())

    def test_download_treats_404_as_unavailable_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            response = Mock()
            response.status_code = 404
            response.headers = {}
            with patch(
                "acbl_tournament_download_sessions_using_sanctioned_events.requests.get",
                return_value=response,
            ) as get:
                stats = download_tournament_sessions(
                    session_ids=["NABC262-SPIN-1"],
                    api_key="test-key",
                    output_dir=output_dir,
                    max_attempts=1,
                )
            get.assert_called_once()
            _args, kwargs = get.call_args
            self.assertEqual(
                kwargs["timeout"],
                session_request_timeout(DEFAULT_TIMEOUT_SECONDS),
            )
            self.assertEqual(stats.unavailable, 1)
            self.assertEqual(stats.errors, 0)
            self.assertEqual(stats.unavailable_ids, ["NABC262-SPIN-1"])
            self.assertFalse(stats.failed())
            self.assertFalse((output_dir / "NABC262-SPIN-1.session.json").exists())

    def test_download_401_aborts_without_trying_remaining_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            response = Mock()
            response.status_code = 401
            response.headers = {}
            with patch(
                "acbl_tournament_download_sessions_using_sanctioned_events.requests.get",
                return_value=response,
            ) as get:
                stats = download_tournament_sessions(
                    session_ids=["2608313-3-1", "2608313-3-2"],
                    api_key="test-key",
                    output_dir=output_dir,
                    max_attempts=4,
                )
            get.assert_called_once()
            self.assertEqual(stats.errors, 1)
            self.assertTrue(stats.aborted)
            self.assertTrue(stats.auth_rejected)
            self.assertTrue(stats.failed())

    def test_download_exhausted_timeout_is_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with patch(
                "acbl_tournament_download_sessions_using_sanctioned_events.requests.get",
                side_effect=requests.Timeout("read timed out"),
            ):
                stats = download_tournament_sessions(
                    session_ids=["NABC262-WERN-1"],
                    api_key="test-key",
                    output_dir=output_dir,
                    max_attempts=1,
                )
            self.assertEqual(stats.errors, 1)
            self.assertEqual(stats.unavailable, 0)
            self.assertTrue(stats.failed())

    def test_sessions_still_needed_leaves_saved_files_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "2604346-1001-1.session.json").write_text("{}", encoding="utf-8")
            (output_dir / "2604346-1001-2.session.sql").write_text("select 1", encoding="utf-8")
            pending, saved = sessions_still_needed(
                ["2604346-1001-1", "2604346-1001-2", "2604346-1001-3"],
                output_dir,
            )
        self.assertEqual(saved, 2)
        self.assertEqual(pending, ["2604346-1001-3"])

    def test_acbl_builder_persists_elo_from_first_session(self) -> None:
        self.assertEqual(
            acbl_elo_ratings_create.PERSISTED_ELO_MINIMUM_SESSIONS,
            1,
        )


class WebTeamSessionSkipTests(unittest.TestCase):
    """Team sessions the live.acbl.org path cannot save are skipped only on evidence."""

    EVENTS = {
        "2605102-201B": ("Swiss", "2026-05-20"),
        "2610350-0101": ("Teams", "2026-10-31"),
        "2605102-191A": ("Pairs", "2026-05-19"),
        "2605102-NODT": ("KO", ""),
    }

    def test_skips_only_team_sessions_checked_after_the_event_settled(self) -> None:
        from acbl_tournament_download_sessions_using_sanctioned_events import web_team_sessions_to_skip

        checked = {
            "2605102-201B-1": "2026-11-12",  # team, checked months later: skip
            "2610350-0101-1": "2026-11-12",  # team, checked 12 days after the event: ask again
            "2605102-191A-1": "2026-11-12",  # pairs: never skipped
            "2605102-NODT-1": "2026-11-12",  # no event date: ask again
            "9999999-XXXX-1": "2026-11-12",  # unknown event: ask again
        }
        ids = list(checked) + ["2605102-201B-2"]  # same team event, never checked
        self.assertEqual(
            web_team_sessions_to_skip(ids, self.EVENTS, checked),
            {"2605102-201B-1"},
        )

    def test_checked_list_round_trip_and_unreadable_file(self) -> None:
        from acbl_tournament_download_sessions_using_sanctioned_events import (
            WEB_NO_SCORECARDS_FILE,
            load_web_no_scorecards,
            save_web_no_scorecards,
        )

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(load_web_no_scorecards(folder), {})
            save_web_no_scorecards(folder, {"2605102-201B-1": "2026-11-12"})
            self.assertEqual(load_web_no_scorecards(folder), {"2605102-201B-1": "2026-11-12"})
            (folder / WEB_NO_SCORECARDS_FILE).write_text("not json", encoding="utf-8")
            self.assertEqual(load_web_no_scorecards(folder), {})

    def test_event_info_is_collected_with_the_session_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_dir = Path(tmp)
            event = {
                "id": "2605102-201B",
                "session_count": 2,
                "game_type": "Swiss",
                "start_date": "2026-05-20",
            }
            (events_dir / "2605102-201B.sanction.json").write_text(json.dumps(event), encoding="utf-8")
            info: dict[str, tuple[str, str]] = {}
            ids = build_session_ids_from_sanctioned_events(events_dir, info)
        self.assertEqual(ids, ["2605102-201B-2", "2605102-201B-1"])
        self.assertEqual(info, {"2605102-201B": ("Swiss", "2026-05-20")})

    def test_no_scorecards_is_remembered_only_for_a_real_results_page(self) -> None:
        from acbl_tournament_live_download import download_sessions_from_web

        recap = "<html><head><title>ACBL Live</title></head><body><h2>Swiss Teams</h2></body></html>"
        pages = {
            "/event/2605102/201B/1/recap": recap,
            "/event/2605102/201B/2/recap": "<html><title>502</title>Bad Gateway</html>",
            "/event/2605102/201B/3/recap": "<html><title>Something else</title></html>",
        }
        ids = ["2605102-201B-1", "2605102-201B-2", "2605102-201B-3"]
        stats = DownloadStats()
        with tempfile.TemporaryDirectory() as tmp:
            returned = download_sessions_from_web(ids, Path(tmp), fetch=pages.__getitem__, stats=stats)
        self.assertIs(returned, stats)
        self.assertEqual(stats.unavailable, 3)
        self.assertEqual(stats.no_scorecard_ids, ["2605102-201B-1"])

    # Trimmed from https://web3.acbl.org/event/1803023/1/1/scores/A/N/14 (2026-11-23):
    # a real results page whose scorecard table has no board rows.
    EMPTY_SCORECARD = (
        "<html><head><title>ACBL Live</title></head><body>"
        '<table class="tablesorter scorecard table"><thead><tr><th></th><th>Board</th>'
        "<th>Contract</th><th>By</th><th>Plus</th><th>Minus</th><th>Matchpoints</th><th>%</th>"
        "</tr></thead><tbody>\n\t\t\t\t\t\t\t\t\t</tbody></table></body></html>"
    )

    def test_empty_scorecard_page_is_recognized(self) -> None:
        from acbl_tournament_live_download import _looks_like_empty_scorecard

        self.assertTrue(_looks_like_empty_scorecard(self.EMPTY_SCORECARD))
        with_row = self.EMPTY_SCORECARD.replace("<tbody>", '<tbody><tr><td><a href="?board_num=1">1</a></td></tr>')
        self.assertFalse(_looks_like_empty_scorecard(with_row))
        self.assertFalse(_looks_like_empty_scorecard(self.EMPTY_SCORECARD.replace("ACBL Live", "Login")))
        self.assertFalse(_looks_like_empty_scorecard("<html><title>ACBL Live</title><body>no table</body></html>"))

    def test_no_board_rows_is_remembered_only_when_every_scorecard_table_is_empty(self) -> None:
        from acbl_tournament_live_download import download_sessions_from_web

        def recap(session: str) -> str:
            return (
                "<html><head><title>ACBL Live</title></head><body><h2>Morning 299er Pairs</h2>"
                f'<a href="/event/1803023/{session}/1/scores/A/N/14">Scores</a></body></html>'
            )

        pages = {
            "/event/1803023/1/1/recap": recap("1"),
            "/event/1803023/1/1/scores/A/N/14": self.EMPTY_SCORECARD,
            "/event/1803023/2/1/recap": recap("2"),
            # Not a results page (for example a check page): may be temporary, so asked again.
            "/event/1803023/2/1/scores/A/N/14": "<html><title>Just a moment...</title></html>",
        }
        ids = ["1803023-1-1", "1803023-2-1"]
        stats = DownloadStats()
        with tempfile.TemporaryDirectory() as tmp:
            download_sessions_from_web(ids, Path(tmp), fetch=pages.__getitem__, stats=stats)
        self.assertEqual(stats.unavailable, 2)
        self.assertEqual(stats.no_board_rows_ids, ["1803023-1-1"])
        self.assertEqual(stats.no_scorecard_ids, [])

    def test_no_board_rows_sessions_are_skipped_for_any_game_type_once_settled(self) -> None:
        from acbl_tournament_download_sessions_using_sanctioned_events import (
            WEB_NO_BOARD_ROWS_FILE,
            load_web_no_scorecards,
            save_web_no_scorecards,
            web_settled_sessions_to_skip,
        )

        checked = {
            "2605102-191A-1": "2026-11-23",  # pairs, checked months later: skip
            "2605102-201B-1": "2026-11-23",  # team, checked months later: skip
            "2610350-0101-1": "2026-11-12",  # checked 12 days after the event: ask again
            "9999999-XXXX-1": "2026-11-23",  # unknown event: ask again
        }
        self.assertEqual(
            web_settled_sessions_to_skip(list(checked) + ["2605102-191A-2"], self.EVENTS, checked),
            {"2605102-191A-1", "2605102-201B-1"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_web_no_scorecards(folder, checked, WEB_NO_BOARD_ROWS_FILE)
            self.assertEqual(load_web_no_scorecards(folder, WEB_NO_BOARD_ROWS_FILE), checked)
            self.assertEqual(load_web_no_scorecards(folder), {})  # the team list is a separate file
            note = json.loads((folder / WEB_NO_BOARD_ROWS_FILE).read_text(encoding="utf-8"))["note"]
            self.assertIn("empty board tables", note)


if __name__ == "__main__":
    unittest.main()
