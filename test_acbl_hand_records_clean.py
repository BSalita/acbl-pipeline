import unittest

import polars as pl

from acbl_sql_to_hand_records_clean import _parse_acbl_par, sort_by_session


class ParParseTests(unittest.TestCase):
    def test_tournament_all_pass_has_no_contract(self) -> None:
        self.assertEqual(_parse_acbl_par("Par: 0"), (0, [(0, "", "", "", 0)]))

    def test_signed_score_and_slash_contracts(self) -> None:
        self.assertEqual(
            _parse_acbl_par("Par: +140 3H-NS/3S-NS"),
            (140, [(3, "H", "", "NS", 0), (3, "S", "", "NS", 0)]),
        )

    def test_blank_and_unknown_sign_are_dropped(self) -> None:
        self.assertIsNone(_parse_acbl_par("Par: "))
        self.assertIsNone(_parse_acbl_par("Par: ?90 1NT-SW"))

    def test_notrump_doubled_off_one(self) -> None:
        self.assertEqual(
            _parse_acbl_par("Par: -100 5S*-NS-1"),
            (-100, [(5, "S", "*", "NS", -1)]),
        )
        self.assertEqual(
            _parse_acbl_par("Par: -90 1NT-EW/2C-EW"),
            (-90, [(1, "N", "", "EW", 0), (2, "C", "", "EW", 0)]),
        )


class SortBySessionTests(unittest.TestCase):
    ROWS = {
        "Board": [2, 1, 1, 1, 3],
        "hand_record_id": ["9", "9", "7", "7", "100"],
        "session_id": [20, 20, 20, 20, 3],
        "PBN": ["d", "c", "b", "a", "e"],
    }

    def test_orders_by_session_then_board_then_the_rest(self) -> None:
        out = sort_by_session(pl.DataFrame(self.ROWS))
        self.assertEqual(out.columns, list(self.ROWS))
        self.assertEqual(out["session_id"].to_list(), [3, 20, 20, 20, 20])
        self.assertEqual(out["Board"].to_list(), [3, 1, 1, 1, 2])
        # Three rows share session 20, board 1: hand_record_id, then PBN, decide.
        self.assertEqual(out["PBN"].to_list(), ["e", "a", "b", "c", "d"])

    def test_same_result_whatever_the_input_order(self) -> None:
        frame = pl.DataFrame(self.ROWS)
        expected = sort_by_session(frame)
        for seed in range(5):
            shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
            self.assertTrue(sort_by_session(shuffled).equals(expected))


if __name__ == "__main__":
    unittest.main()
