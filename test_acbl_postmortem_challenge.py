"""Recent-game scrapes use the same Cloudflare waiter as the downloaders."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mlBridge.mlBridgeAcblLib import _goto_with_diagnostics, wait_through_acbl_challenge
from acbl_club_download_to_json import Forbidden403Error


class _Page:
    def __init__(self, failures=0):
        self.failures = failures
        self.gotos = 0

    def goto(self, url, wait_until=None, timeout=None):
        self.gotos += 1
        if self.gotos <= self.failures:
            raise RuntimeError("net::ERR_ABORTED at " + url)
        return SimpleResponse()


class SimpleResponse:
    status = 403


class ChallengeNavigationTests(unittest.TestCase):
    def test_goto_waits_through_cloudflare(self):
        seen = []

        def wait(page, timeout_ms=None, reload_after_s=30):
            seen.append((page, timeout_ms, reload_after_s))
            return True

        import mlBridge.mlBridgeAcblLib as lib

        original = lib._pipeline_challenge_api
        lib._pipeline_challenge_api = lambda: (Forbidden403Error, lambda exc: False, wait)
        try:
            page = _Page()
            response = _goto_with_diagnostics(page, "https://my.acbl.org/club-results/details/1", verbose=False)
        finally:
            lib._pipeline_challenge_api = original
        self.assertIsInstance(response, SimpleResponse)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], page)
        self.assertIsNone(seen[0][1])

    def test_goto_retries_aborted_navigation(self):
        import mlBridge.mlBridgeAcblLib as lib

        original = lib._pipeline_challenge_api
        lib._pipeline_challenge_api = lambda: (
            Forbidden403Error,
            lambda exc: "ERR_ABORTED" in str(exc),
            lambda page, timeout_ms=None, reload_after_s=30: True,
        )
        try:
            page = _Page(failures=2)
            _goto_with_diagnostics(page, "https://my.acbl.org/x", verbose=False)
        finally:
            lib._pipeline_challenge_api = original
        self.assertEqual(page.gotos, 3)

    def test_wait_helper_calls_pipeline_waiter(self):
        import mlBridge.mlBridgeAcblLib as lib

        seen = {}

        def wait(page, timeout_ms=None, reload_after_s=30):
            seen["timeout_ms"] = timeout_ms
            seen["reload"] = reload_after_s
            return True

        original = lib._pipeline_challenge_api
        lib._pipeline_challenge_api = lambda: (Forbidden403Error, lambda exc: False, wait)
        try:
            wait_through_acbl_challenge(object(), timeout_ms=1000)
        finally:
            lib._pipeline_challenge_api = original
        self.assertEqual(seen["timeout_ms"], 1000)
        self.assertEqual(seen["reload"], 30)


if __name__ == "__main__":
    unittest.main()
