import unittest

import polars as pl

from acbl_board_results_augment_step1 import normalize_contracts, typed_club_results


class NormalizeContractsTests(unittest.TestCase):
    def test_keeps_doubles_in_spaced_and_tight_formats(self) -> None:
        frame = pl.DataFrame(
            {
                "Contract": ["5 C x", "4 S xx", "3 NT", "4SX", "2 H ", "PASS"],
                "Declarer_Direction": ["S", "N", "E", "W", "N", "N"],
            }
        )
        out = normalize_contracts(frame)
        self.assertEqual(
            out["Contract"].to_list(), ["5CXS", "4SXXN", "3NE", "4SXW", "2HN", "PASS"]
        )
        self.assertEqual(out["Dbl"].to_list(), ["X", "XX", "", "X", "", None])
        self.assertEqual(out["BidLvl"].to_list(), [5, 4, 3, 4, 2, 0])


class TypedClubResultsTests(unittest.TestCase):
    def test_parses_acbl_result_strings_and_drops_raw_columns(self) -> None:
        frame = pl.DataFrame(
            {
                "result": ["=", "+1", "-3", "", None],
                "tricks_taken": ["10", "9", "8", "", None],
            }
        )
        out = typed_club_results(frame)
        self.assertEqual(out.columns, ["Result", "Tricks"])
        self.assertEqual(out["Result"].to_list(), [0, 1, -3, None, None])
        self.assertEqual(out["Tricks"].to_list(), [10, 9, 8, None, None])
        self.assertEqual(out.schema["Result"], pl.Int8)
        self.assertEqual(out.schema["Tricks"], pl.UInt8)


class PostmortemFrameTests(unittest.TestCase):
    def test_raw_result_strings_are_merged_then_dropped(self) -> None:
        from acbl_club_api_service import _normalize_postmortem_frame

        frame = pl.DataFrame(
            {
                "Result": pl.Series([0, 1, -1], dtype=pl.Int8),
                "result": ["=", "+1", "-1"],
                "Tricks": pl.Series([9, 10, 8], dtype=pl.Int16),
                "tricks_taken": ["9", "10", "8"],
            }
        )
        out = _normalize_postmortem_frame(frame)
        self.assertEqual(out.columns, ["Result", "Tricks"])
        self.assertEqual(out["Result"].to_list(), [0, 1, -1])


if __name__ == "__main__":
    unittest.main()
