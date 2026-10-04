import sqlite3
import tempfile
import unittest
from pathlib import Path

import polars as pl

from acbl_sqlite_read import read_sqlite_query


class SqliteReadTests(unittest.TestCase):
    def test_casts_bools_and_keeps_text_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.sqlite"
            con = sqlite3.connect(path)
            con.execute(
                "CREATE TABLE events (id INTEGER, is_virtual_game INTEGER, hand_record_id TEXT)"
            )
            con.executemany(
                "INSERT INTO events VALUES (?, ?, ?)",
                [(1, 0, "10"), (2, 1, "SHUFFLE")],
            )
            con.commit()
            con.close()

            frame = read_sqlite_query(
                "sqlite:///" + path.as_posix(),
                "SELECT id AS event_id, is_virtual_game, hand_record_id FROM events",
                {
                    "event_id": pl.Int64,
                    "is_virtual_game": pl.Boolean,
                    "hand_record_id": pl.String,
                },
            )
            self.assertEqual(frame["event_id"].to_list(), [1, 2])
            self.assertEqual(frame["is_virtual_game"].to_list(), [False, True])
            self.assertEqual(frame["hand_record_id"].to_list(), ["10", "SHUFFLE"])


if __name__ == "__main__":
    unittest.main()
