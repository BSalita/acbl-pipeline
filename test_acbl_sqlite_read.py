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

    def test_sqlite_cast_keeps_a_leading_integer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.sqlite"
            con = sqlite3.connect(path)
            con.execute("CREATE TABLE board_results (ns_pair TEXT, ew_pair TEXT, mp_total TEXT)")
            con.executemany(
                "INSERT INTO board_results VALUES (?, ?, ?)",
                [("2-NS", "15", "12.5x"), ("NS", None, None), ("", "3", "nope")],
            )
            con.commit()
            con.close()

            frame = read_sqlite_query(
                "sqlite:///" + path.as_posix(),
                "SELECT CAST(ns_pair AS INTEGER) AS ns_pair, "
                "CAST(ew_pair AS INTEGER) AS ew_pair, "
                "CAST(mp_total AS REAL) AS mp_total FROM board_results",
                {"ns_pair": pl.UInt16, "ew_pair": pl.UInt16, "mp_total": pl.Float32},
            )
            self.assertEqual(frame["ns_pair"].to_list(), [2, 0, 0])
            self.assertEqual(frame["ew_pair"].to_list(), [15, None, 3])
            self.assertEqual(frame["mp_total"].to_list(), [12.5, None, 0.0])


if __name__ == "__main__":
    unittest.main()
