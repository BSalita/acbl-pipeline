import unittest

import polars as pl

from acbl_sql_to_hand_records_clean import _parse_acbl_par, clean_hand_records, sort_by_session


class NewFormHandRecordTests(unittest.TestCase):
    """Club hand records from about March 2026 on: spaced cards, ten as T, no ACBL DD/Par."""

    # Raw row of hand-record set 1107741, board 19.
    SPACED = ("S 7 6 5H K 9 5 3D K 6 5C A Q 4S A K Q 8 2H 6D Q J 8C J 8 7 6"
              "S T 9 4 3H Q 8D 9 7 4C T 9 5 3S JH A J T 7 4 2D A T 3 2C K 2")
    OLDER = SPACED.replace(" ", "").replace("T", "10")

    def _clean(self, rows):
        frame = pl.DataFrame(rows, schema={
            "board": pl.Int16, "board_record_string": pl.String, "hand_record_id": pl.String,
            "dealer": pl.String, "vulnerability": pl.String, "double_dummy_ew": pl.String,
            "double_dummy_ns": pl.String, "par": pl.String,
        }, orient="row")
        return clean_hand_records(frame)

    def test_spaced_row_without_dd_is_kept_with_its_pbn(self) -> None:
        out = self._clean([(19, self.SPACED, "1107741", "S", "E-W", "", "", "")])
        self.assertEqual(out["PBN"].to_list(),
                         ["N:765.K953.K65.AQ4 T943.Q8.974.T953 J.AJT742.AT32.K2 AKQ82.6.QJ8.J876"])
        self.assertNotIn("_no_acbl_dd", out.columns)

    def test_older_form_row_without_dd_is_still_dropped(self) -> None:
        out = self._clean([(19, self.OLDER, "1107741", "S", "E-W", "", "", "")])
        self.assertEqual(out.height, 0)


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
