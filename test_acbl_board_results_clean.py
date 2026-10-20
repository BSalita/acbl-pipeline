import sqlite3
import unittest

import polars as pl

from acbl_sql_to_board_results_clean import drop_out_of_range_pct, get_club_schemas, pair_id_list


class PairIdListTests(unittest.TestCase):
    def test_keeps_numbers_and_skips_truncated_names(self) -> None:
        frame = pl.DataFrame(
            {
                "pair_acbl": [
                    "[3402819, 4692195]",
                    "[1590227,Al Ober]",
                    '["Linda F","Annette"]',
                    '["#155","7022131"]',
                    "[9121560,#320]",
                    "[]",
                    None,
                ]
            }
        )
        parsed = frame.select(pair_id_list("pair_acbl"))["pair_acbl"].to_list()
        self.assertEqual(
            parsed,
            [
                ["3402819", "4692195"],
                ["1590227"],
                [],
                ["#155", "7022131"],
                ["9121560", "#320"],
                [],
                None,
            ],
        )

    def test_zero_player_number_keeps_its_seat_as_null(self) -> None:
        frame = pl.DataFrame({"pair_acbl": ["[0, 4692195]", "[3402819,0]", '["000","7"]']})
        parsed = frame.select(pair_id_list("pair_acbl"))["pair_acbl"].to_list()
        self.assertEqual(parsed, [[None, "4692195"], ["3402819", None], [None, "7"]])


class ClubPlayersSqlTests(unittest.TestCase):
    def test_missing_acbl_number_is_null_not_zero(self) -> None:
        sql_selects_d, _schema_d = get_club_schemas()
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE players (id INTEGER, pair_summary_id INTEGER, id_number TEXT, name TEXT, mp_total TEXT)"
        )
        con.executemany(
            "INSERT INTO players VALUES (?, ?, ?, ?, ?)",
            [(1, 10, "2663279", "A", "1.5"), (2, 10, "", "B", "0"), (3, 11, "0", "C", None)],
        )
        numbers = [row[2] for row in con.execute(sql_selects_d["players"] + " ORDER BY id")]
        self.assertEqual(numbers, [2663279, None, None])


class DropOutOfRangePctTests(unittest.TestCase):
    def test_drops_rows_outside_unit_interval_and_keeps_nulls(self) -> None:
        frame = pl.DataFrame(
            {
                "Pct_NS": [0.0, 1.0, 0.5, 9.99, 1.06, 0.05, None, -0.01],
                "Pct_EW": [1.0, 0.0, 0.5, 9.99, 0.2, 2.45, None, 1.0],
            },
            schema={"Pct_NS": pl.Float32, "Pct_EW": pl.Float32},
        )
        kept = drop_out_of_range_pct(frame, "test")
        self.assertEqual(
            kept.rows(),
            [(0.0, 1.0), (1.0, 0.0), (0.5, 0.5), (None, None)],
        )


if __name__ == "__main__":
    unittest.main()
