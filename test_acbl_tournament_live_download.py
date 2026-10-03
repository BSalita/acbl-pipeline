from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from acbl_club_download_to_json import _is_cloudflare_challenge, _wait_through_cloudflare
from acbl_tournament_live_download import (
    build_session_payload,
    download_sessions_from_web,
    parse_events_page,
    parse_handrecords_page,
    parse_scorecard,
    parse_tournament_list,
    split_session_id,
)

EVENTS_HTML = """
<table id="events">
<tr role="row"><th>Date</th></tr>
<tr role="row" class="odd">
  <td class="sorting_1">04/11/2026</td><td>Saturday</td>
  <td class="sorting_2">Open Swiss Teams</td><td class="sorting_3">10:00 am</td><td></td>
  <td class="links"><a href="/event/2604346/1101/2/recap">Recaps</a></td>
</tr>
<tr role="row" class="even">
  <td>04/10/2026</td><td>Friday</td><td>Open Pairs</td><td>10:00 am</td><td></td>
  <td class="links">
    <a href="/event/2604346/1001/1/summary">Summary</a>
    <a href="/handrecords/2604346/04101000">Hands</a>
  </td>
</tr>
<tr role="row">
  <td>04/10/2026</td><td>Friday</td><td>Open Pairs</td><td>3:00 pm</td><td></td>
  <td><a href="/event/2604346/1001/2/recap">Recaps</a></td>
</tr>
</table>
"""

HAND_HTML = """
<div id="board-1" class="board">
  <div class="board-data"><span>Dlr: N</span><span>Vul: None</span></div>
  <div class="hand">
    <span class=""><span class="spades symbol"></span> J 9 8 6 3 2</span>
    <span class=""><span class="hearts symbol"></span> 3</span>
    <span class=""><span class="diams symbol"></span> 10 5</span>
    <span class=""><span class="clubs symbol"></span> A J 10 6</span>
  </div>
  <div class="hand middle">
    <div class="inner-slice left">
      <span class=""><span class="spades symbol"></span> Q</span>
      <span class=""><span class="hearts symbol"></span> J 10</span>
      <span class=""><span class="diams symbol"></span> A K Q 9 8 6 4 2</span>
      <span class=""><span class="clubs symbol"></span> 8 3</span>
    </div>
    <div class="inner-slice right">
      <span class=""><span class="spades symbol"></span> A 10 4</span>
      <span class=""><span class="hearts symbol"></span> A K 8 4</span>
      <span class=""><span class="diams symbol"></span> J 7 3</span>
      <span class=""><span class="clubs symbol"></span> 9 5 2</span>
    </div>
  </div>
  <div class="hand">
    <span class=""><span class="spades symbol"></span> K 7 5</span>
    <span class=""><span class="hearts symbol"></span> Q 9 7 6 5 2</span>
    <span class=""><span class="diams symbol"></span> \u2014</span>
    <span class=""><span class="clubs symbol"></span> K Q 7 4</span>
  </div>
  <div class="double-dummy">
    <span>NS:  3<span class="clubs symbol"></span>  2<span class="hearts symbol"></span>  4<span class="spades symbol"></span>
      <div class="reverse"><span class="diams symbol"></span>2</div>
      <div class="reverse">NT2</div></span>
    <span>EW:  5<span class="diams symbol"></span> 3NT
      <div class="reverse"><span class="clubs symbol"></span>4/3</div>
      <div class="reverse"><span class="hearts symbol"></span>5</div>
      <div class="reverse"><span class="spades symbol"></span>3</div></span>
  </div>
  <div class="par-score"><span>-100 5S*-NS-1</span></div>
  <div class="bid-section"></div>
</div>
"""

RECAP_HTML = """
<h1>Apr 10, 2026 - Friday 10:00 am</h1>
<h2>Open Pairs Scores</h2>
<a href="/handrecords/2604346/04101000">Hands</a>
<a href="/event/2604346/1001/1/scores/A/N/1">NS</a>
<a href="/event/2604346/1001/1/scores/A/E/7">EW</a>
"""

SCORE_NS = """
<h4 title="3402819 | 4692195"><span class="orange-text">1NS</span> - Susan McCoy &amp; Ingeborg Dickerson</h4>
<table class="tablesorter scorecard">
<tr>
  <td><a href="https://www.bridgebase.com/tools/handviewer.html?b=15&amp;p={Optimal Contract: -100 5%21Sx-NS-1}">Play</a></td>
  <td><a href="/event/2604346/1001/1/board-detail/A?board_num=15">15</a></td>
  <td> 4<span class="spades symbol contract"></span>x</td>
  <td>S</td><td>790</td><td></td><td>11.5</td><td>96</td>
  <td class="vs" title="5006279 | 3391809">7 - Daryl Sallaz - Tony Cardoni</td>
</tr>
</table>
"""

SCORE_EW = """
<h4 title="5006279 | 3391809"><span>7EW</span> - Daryl Sallaz &amp; Tony Cardoni</h4>
<table class="tablesorter scorecard">
<tr>
  <td><a href="https://www.bridgebase.com/tools/handviewer.html?b=15">Play</a></td>
  <td><a href="/event/2604346/1001/1/board-detail/A?board_num=15">15</a></td>
  <td> 4<span class="spades symbol contract"></span>x</td>
  <td>S</td><td></td><td>-790</td><td>0.5</td><td>4</td>
  <td class="vs" title="3402819 | 4692195">1 - Susan McCoy - Ingeborg Dickerson</td>
</tr>
</table>
"""


class LiveTournamentParseTests(unittest.TestCase):
    def test_split_session_id(self) -> None:
        self.assertEqual(split_session_id("2604346-1001-1"), ("2604346", "1001", 1))
        self.assertEqual(split_session_id("NABC262-OSHL-2"), ("NABC262", "OSHL", 2))

    def test_events_page_groups_sessions(self) -> None:
        events = {event["id"]: event for event in parse_events_page(EVENTS_HTML, "2604346")}
        self.assertEqual(events["2604346-1001"]["session_count"], 2)
        self.assertEqual(events["2604346-1001"]["start_date"], "2026-04-10")
        self.assertEqual(events["2604346-1001"]["game_type"], "Pairs")
        self.assertEqual(events["2604346-1101"]["game_type"], "Teams")
        self.assertEqual(events["2604346-1101"]["session_count"], 2)

    def test_hand_and_scorecard_match_api_fields(self) -> None:
        hand = parse_handrecords_page(HAND_HTML, box_number="04101000")[0]
        self.assertEqual(hand["north_spades"], "J 9 8 6 3 2")
        self.assertEqual(hand["south_diamonds"], "-----")
        self.assertEqual(hand["dealer"], "North")
        self.assertEqual(hand["vulnerability"], "None")
        self.assertEqual(hand["double_dummy_north_south"], "3C D2 2H 4S NT2")
        self.assertEqual(hand["double_dummy_east_west"], "C4/3 5D H5 S3 3NT")

        card = parse_scorecard(SCORE_NS, page_url="/event/2604346/1001/1/scores/A/N/1")
        row = card["rows"][0]
        self.assertEqual(row["contract"], "4Sx")
        self.assertEqual(row["declarer"], "S")
        self.assertEqual(row["score"], 790)
        self.assertEqual(row["orientation"], "N-S")
        self.assertEqual(row["pair_acbl"], [3402819, 4692195])
        self.assertEqual(row["opponent_pair_names"], ["Daryl Sallaz", "Tony Cardoni"])

    def test_session_payload_has_both_orientations(self) -> None:
        payload = build_session_payload(
            "2604346-1001-1",
            RECAP_HTML,
            HAND_HTML,
            [
                ("/event/2604346/1001/1/scores/A/N/1", SCORE_NS),
                ("/event/2604346/1001/1/scores/A/E/7", SCORE_EW),
            ],
            tournament_name="Ontario Sectional",
        )
        assert payload is not None
        self.assertEqual(payload["id"], "2604346-1001-1")
        self.assertEqual(payload["start_date"], "20260410")
        self.assertEqual(payload["handrecord"][0]["double_dummy_par_score"], "-100 5S*-NS-1")
        boards = payload["sections"][0]["board_results"]
        self.assertEqual(sorted(row["orientation"] for row in boards), ["E-W", "N-S"])
        self.assertEqual(boards[0]["contract"], "4Sx")

    def test_webpage_download_skips_404_and_writes_json(self) -> None:
        pages = {
            "/event/2604346/1001/1/recap": RECAP_HTML,
            "/handrecords/2604346/04101000": HAND_HTML,
            "/event/2604346/1001/1/scores/A/N/1": SCORE_NS,
            "/event/2604346/1001/1/scores/A/E/7": SCORE_EW,
            "/event/2604346/1001/2/recap": "<html><title>404 Not Found</title>A 404 error occurred</html>",
        }

        def fetch(url: str) -> str:
            path = url
            if "://" in url:
                path = "/" + url.split("/", 3)[-1]
            return pages[path]

        with tempfile.TemporaryDirectory() as tmp:
            stats = download_sessions_from_web(
                ["2604346-1001-2", "2604346-1001-1"],
                Path(tmp),
                fetch=fetch,
            )
            written = json.loads((Path(tmp) / "2604346-1001-1.session.json").read_text(encoding="utf-8"))
        self.assertEqual(stats.written, 1)
        self.assertEqual(stats.unavailable, 1)
        self.assertEqual(stats.errors, 0)
        self.assertFalse(stats.failed())
        self.assertEqual(written["sections"][0]["board_results"][0]["match_points"], 0.5)

    def test_aborted_navigation_skips_session_and_continues(self) -> None:
        pages = {
            "/event/2604346/1001/1/recap": RECAP_HTML,
            "/handrecords/2604346/04101000": HAND_HTML,
            "/event/2604346/1001/1/scores/A/N/1": SCORE_NS,
            "/event/2604346/1001/2/recap": "<html><title>404 Not Found</title>A 404 error occurred</html>",
        }

        def fetch(url: str) -> str:
            if "scores/" in url and url.endswith("/E/7"):
                raise RuntimeError(
                    "Page.goto: net::ERR_ABORTED at "
                    "https://live.acbl.org/event/2604346/1001/1/scores/A/E/7"
                )
            return pages[url]

        with tempfile.TemporaryDirectory() as tmp:
            stats = download_sessions_from_web(
                ["2604346-1001-1", "2604346-1001-2"],
                Path(tmp),
                fetch=fetch,
            )
            self.assertFalse((Path(tmp) / "2604346-1001-1.session.json").exists())
        self.assertEqual(stats.written, 0)
        self.assertEqual(stats.unavailable, 2)
        self.assertEqual(stats.errors, 0)
        self.assertFalse(stats.aborted)
        self.assertFalse(stats.failed())

    def test_five_consecutive_page_failures_stop_the_run(self) -> None:
        def fetch(url: str) -> str:
            raise RuntimeError("Page.goto: net::ERR_ABORTED")

        ids = [f"2610104-29GP-{n}" for n in range(1, 8)]
        with tempfile.TemporaryDirectory() as tmp:
            stats = download_sessions_from_web(ids, Path(tmp), fetch=fetch)
        self.assertEqual(stats.unavailable, 5)
        self.assertTrue(stats.aborted)
        self.assertTrue(stats.failed())
        self.assertEqual(stats.errors, 0)

    def test_tournament_list_links(self) -> None:
        html = (
            '<a href="/events/2609385"><b>Charleston, SC</b>'
            '<span class="tourn-date"><i>Sep 26, 2026 - Sep 27, 2026</i></span></a>'
            '<a data-page="3"></a>'
        )
        found, pages = parse_tournament_list(html)
        self.assertEqual(found[0]["sanction"], "2609385")
        self.assertEqual(pages, 3)

    def test_reload_clears_a_stuck_challenge(self) -> None:
        class Page:
            def __init__(self) -> None:
                self.reloads = 0

            def content(self) -> str:
                if self.reloads:
                    return '<html><table id="events"></table><h1>ACBL Live</h1></html>'
                return "<html>cf-turnstile challenge-platform</html>"

            def title(self) -> str:
                return "ACBL Live"

            def reload(self, **_kwargs) -> None:
                self.reloads += 1

            def wait_for_timeout(self, _ms: int) -> None:
                return None

            def wait_for_load_state(self, *_args, **_kwargs) -> None:
                return None

        page = Page()
        self.assertTrue(_wait_through_cloudflare(page, timeout_ms=60_000, reload_after_s=0))
        self.assertEqual(page.reloads, 1)

    def test_waf_interstitial_is_still_a_challenge(self) -> None:
        page = SimpleNamespace(title=lambda: "")
        waf = "<html><script>var gokuProps = {};</script><div id='challenge-container'></div></html>"
        self.assertTrue(_is_cloudflare_challenge(page, waf))
        live = '<html><table id="events"></table><h1>ACBL Live</h1></html>'
        self.assertFalse(_is_cloudflare_challenge(page, live))
        club = '<html><script>var data = {"x": 1};</script></html>'
        self.assertFalse(_is_cloudflare_challenge(page, club))


class SavedLivePageTests(unittest.TestCase):
    def test_probed_events_hands_and_scorecard(self) -> None:
        temp = Path(__import__("os").environ.get("TEMP", ""))
        events = temp / "live_events_cleared2.html"
        hands = temp / "live_hr.html"
        score = temp / "live_score.html"
        if not (events.is_file() and hands.is_file() and score.is_file()):
            self.skipTest("saved live.acbl.org probes are not in TEMP")
        parsed = {event["id"]: event for event in parse_events_page(events.read_text(encoding="utf-8"), "2604346")}
        self.assertGreaterEqual(parsed["2604346-1001"]["session_count"], 1)
        self.assertIn("2604346-1101", parsed)
        board = parse_handrecords_page(hands.read_text(encoding="utf-8"), box_number="04101000")[0]
        self.assertEqual(board["board_number"], 1)
        self.assertEqual(board["south_diamonds"], "-----")
        self.assertEqual(board["double_dummy_north_south"], "3C D2 2H 4S NT2")
        card = parse_scorecard(score.read_text(encoding="utf-8"))
        row = next(item for item in card["rows"] if item["board_number"] == 15)
        self.assertEqual(row["contract"], "4Sx")
        self.assertEqual(row["score"], 790)
        self.assertEqual(row["pair_names"][0], "Susan McCoy")


if __name__ == "__main__":
    unittest.main()
