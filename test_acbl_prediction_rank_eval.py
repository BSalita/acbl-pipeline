import unittest

import polars as pl

import acbl_prediction_rank_eval as ev


def _section(boards: int = 8) -> pl.DataFrame:
    rows = []
    # NS tables 1-4 and EW tables 1-4 meet every board. Actual NS skill rises
    # with table number and averages 50%; the prediction agrees but sits 0.1
    # higher.
    for board in range(1, boards + 1):
        for table in range(1, 5):
            rows.append(
                {
                    "session_id": 1,
                    "section_id": 10,
                    "Board": board,
                    "Pair_Number_NS": table,
                    "Pair_Number_EW": table,
                    "Pct_NS": 0.25 + 0.1 * table,
                    "pred": 0.35 + 0.1 * table,
                }
            )
    return pl.DataFrame(rows)


class SectionRankTests(unittest.TestCase):
    def test_section_centering_keeps_ranks_and_removes_the_offset(self):
        frame = _section().with_columns(ev.center("pred", ev.SECTION).alias("centered"))
        quality = ev.section_rank_quality(frame, ["pred", "centered"])
        self.assertEqual(quality["section_directions"], 2)
        self.assertAlmostEqual(quality["pred"]["mean_spearman"], 1.0)
        self.assertAlmostEqual(quality["centered"]["mean_spearman"], 1.0)
        self.assertAlmostEqual(quality["pred"]["pair_mae"], 0.1, places=6)
        self.assertAlmostEqual(quality["centered"]["pair_mae"], 0.0, places=6)

    def test_pairs_with_too_few_boards_are_left_out(self):
        with self.assertRaises(ValueError):
            ev.section_rank_quality(_section(boards=ev.MIN_PAIR_BOARDS - 1), ["pred"])


if __name__ == "__main__":
    unittest.main()
