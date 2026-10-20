import unittest

import polars as pl

from acbl_sql_to_board_results_clean import drop_out_of_range_pct, pair_id_list


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
