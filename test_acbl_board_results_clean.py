import unittest

import polars as pl

from acbl_sql_to_board_results_clean import pair_id_list


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


if __name__ == "__main__":
    unittest.main()
