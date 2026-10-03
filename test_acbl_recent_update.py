"""Schedule window, club cursor, and recent-store upsert."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

from acbl_recent_update import (
    coverage_start,
    files_to_ingest,
    next_club_batch,
    stamped_tables,
    upsert_tables,
)


class CoverageTests(unittest.TestCase):
    def test_stale_history_opens_the_gap(self) -> None:
        start = coverage_start(date(2026, 9, 9), "hour", date(2026, 10, 3))
        self.assertEqual(start, date(2026, 9, 10))

    def test_current_history_keeps_the_hour_overlap(self) -> None:
        start = coverage_start(date(2026, 10, 2), "hour", date(2026, 10, 3))
        self.assertEqual(start, date(2026, 10, 2))

    def test_quarter_reaches_back_when_history_is_current(self) -> None:
        start = coverage_start(date(2026, 10, 2), "quarter", date(2026, 10, 3))
        self.assertEqual(start, date(2026, 7, 1))


class ClubBatchTests(unittest.TestCase):
    def test_hour_slice_wraps(self) -> None:
        first, cursor = next_club_batch([10, 20, 30, 40], None, 3)
        self.assertEqual(first, [10, 20, 30])
        self.assertEqual(cursor, 30)
        second, cursor = next_club_batch([10, 20, 30, 40], cursor, 3)
        self.assertEqual(second, [40, 10, 20])
        self.assertEqual(cursor, 20)

    def test_zero_batch_visits_every_club(self) -> None:
        clubs, cursor = next_club_batch([1, 2], 1, 0)
        self.assertEqual(clubs, [1, 2])
        self.assertIsNone(cursor)


class UpsertTests(unittest.TestCase):
    def test_replace_and_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = root / "99.data.json"
            session.write_text(
                json.dumps(
                    {
                        "id": 99,
                        "start_date": "10/02/2026",
                        "name": "Open",
                        "sessions": [{"id": 5, "game_date": "2026-10-02"}],
                    }
                ),
                encoding="utf-8",
            )
            tables = stamped_tables(session)
            recent = root / "recent"
            kept = upsert_tables(
                recent, [tables], historical_max=date(2026, 10, 1)
            )
            self.assertEqual(kept, 1)
            events = pl.read_parquet(recent / "events.parquet")
            self.assertEqual(events["event_id"].to_list(), ["99"])
            self.assertEqual(events["game_date"].to_list(), ["2026-10-02"])

            replacement = root / "99b.data.json"
            replacement.write_text(
                json.dumps(
                    {
                        "id": 99,
                        "start_date": "09/01/2026",
                        "name": "Old",
                        "sessions": [{"id": 5, "game_date": "2026-09-01"}],
                    }
                ),
                encoding="utf-8",
            )
            kept = upsert_tables(
                recent,
                [stamped_tables(replacement)],
                historical_max=date(2026, 10, 1),
            )
            self.assertEqual(kept, 0)
            self.assertFalse((recent / "events.parquet").exists())

    def test_ingest_uses_file_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details = root / "100" / "details"
            details.mkdir(parents=True)
            path = details / "1.data.json"
            path.write_text("{}", encoding="utf-8")
            found = files_to_ingest([100], root, date.today())
            self.assertEqual(found, [path])
            self.assertEqual(
                files_to_ingest([100], root, date.today().replace(year=date.today().year + 1)),
                [],
            )


if __name__ == "__main__":
    unittest.main()
