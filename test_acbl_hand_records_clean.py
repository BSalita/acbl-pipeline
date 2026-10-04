import unittest

from acbl_sql_to_hand_records_clean import _parse_acbl_par


class ParParseTests(unittest.TestCase):
    def test_tournament_all_pass_has_no_contract(self) -> None:
        self.assertEqual(_parse_acbl_par("Par: 0"), (0, [(0, "", "", "", 0)]))

    def test_signed_score_and_slash_contracts(self) -> None:
        self.assertEqual(
            _parse_acbl_par("Par: +140 3H-NS/3S-NS"),
            (140, [(3, "H", "", "NS", 0), (3, "S", "", "NS", 0)]),
        )

    def test_notrump_doubled_off_one(self) -> None:
        self.assertEqual(
            _parse_acbl_par("Par: -100 5S*-NS-1"),
            (-100, [(5, "S", "*", "NS", -1)]),
        )
        self.assertEqual(
            _parse_acbl_par("Par: -90 1NT-EW/2C-EW"),
            (-90, [(1, "N", "", "EW", 0), (2, "C", "", "EW", 0)]),
        )


if __name__ == "__main__":
    unittest.main()
