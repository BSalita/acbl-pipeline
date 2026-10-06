import unittest
import tempfile
from datetime import date
from pathlib import Path

import polars as pl

from acbl_prediction_data import _build_joined_plan, _scan_source_shards


class EloJoinTests(unittest.TestCase):
    def test_integer_elo_ids_join_string_player_ids(self) -> None:
        base = pl.LazyFrame(
            {
                "Player_ID_N": ["5798205"],
                "Player_ID_E": ["8"],
                "Player_ID_S": ["2"],
                "Player_ID_W": ["#155"],
                "session_id": [10],
            }
        )
        player_elo = pl.DataFrame(
            {
                "Player_ID": pl.Series([5798205, 8, 2], dtype=pl.Int32),
                "session_id": [10, 10, 10],
                "Elo_N": [1500.0, 1600.0, 1700.0],
                "Elo_R_EventStart": [1400.0, 1500.0, 1600.0],
            }
        )
        pair_elo = pl.DataFrame(
            {
                "Pair_IDs": pl.Series([], dtype=pl.String),
                "session_id": pl.Series([], dtype=pl.Int64),
                "Elo_N": pl.Series([], dtype=pl.Float64),
                "Elo_R_EventStart": pl.Series([], dtype=pl.Float64),
            }
        )
        out = _build_joined_plan(base, player_elo=player_elo, pair_elo=pair_elo).collect()
        self.assertEqual(out["Elo_N_N"].to_list(), [1500.0])
        self.assertEqual(out["Elo_N_E"].to_list(), [1600.0])
        self.assertEqual(out["Elo_N_W"].to_list(), [None])


class ShardScanTests(unittest.TestCase):
    def test_integer_ids_and_extra_columns_scan_together(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.parquet"
            new = Path(tmp) / "new.parquet"
            pl.DataFrame(
                {
                    "Date": [date(2020, 1, 1)],
                    "Player_ID_N": ["111"],
                }
            ).write_parquet(old)
            pl.DataFrame(
                {
                    "Date": [date(2026, 8, 1)],
                    "Player_ID_N": pl.Series([5798205], dtype=pl.Int32),
                    "Is_Sacrifice_Opportunity": [True],
                }
            ).write_parquet(new)
            out = (
                _scan_source_shards(
                    [old, new],
                    ["Date", "Player_ID_N", "Is_Sacrifice_Opportunity"],
                )
                .collect()
                .sort("Date")
            )
            self.assertEqual(out["Player_ID_N"].dtype, pl.String)
            self.assertEqual(out["Player_ID_N"].to_list(), ["111", "5798205"])
            self.assertEqual(out["Is_Sacrifice_Opportunity"].to_list(), [None, True])

    def test_mixed_session_number_casts_to_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.parquet"
            new = Path(tmp) / "new.parquet"
            common = {
                "Player_ID_N": ["1"],
                "Player_ID_E": ["2"],
                "Player_ID_S": ["3"],
                "Player_ID_W": ["4"],
                "session_id": ["10"],
            }
            pl.DataFrame({**common, "session_number": pl.Series([1], dtype=pl.Int64)}).write_parquet(old)
            pl.DataFrame({**common, "session_number": ["2"]}).write_parquet(new)
            player_elo = pl.DataFrame(
                {
                    "Player_ID": pl.Series([], dtype=pl.String),
                    "session_id": pl.Series([], dtype=pl.String),
                    "Elo_N": pl.Series([], dtype=pl.Float64),
                    "Elo_R_EventStart": pl.Series([], dtype=pl.Float64),
                }
            )
            pair_elo = pl.DataFrame(
                {
                    "Pair_IDs": pl.Series([], dtype=pl.String),
                    "session_id": pl.Series([], dtype=pl.String),
                    "Elo_N": pl.Series([], dtype=pl.Float64),
                    "Elo_R_EventStart": pl.Series([], dtype=pl.Float64),
                }
            )
            out = (
                _build_joined_plan(
                    _scan_source_shards([old, new], ["session_number", *common]),
                    player_elo=player_elo,
                    pair_elo=pair_elo,
                )
                .collect()
                .sort("session_number")
            )
            self.assertEqual(out["session_number"].dtype, pl.UInt8)
            self.assertEqual(out["session_number"].to_list(), [1, 2])


if __name__ == "__main__":
    unittest.main()
