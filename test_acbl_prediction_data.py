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


class TournamentIdTests(unittest.TestCase):
    def test_text_section_ids_reach_the_output(self) -> None:
        base = pl.LazyFrame(
            {
                **{f"Player_ID_{seat}": [str(i)] for i, seat in enumerate("NESW", 1)},
                "event_id": ["2503355-2901"],
                "session_id": ["2503355-2901-2"],
                "section_id": ["2503355-2901-2-0"],
                "hand_record_id": ["x"],
            }
        )
        empty_elo = {
            "session_id": pl.Series([], dtype=pl.String),
            "Elo_N": pl.Series([], dtype=pl.Float64),
            "Elo_R_EventStart": pl.Series([], dtype=pl.Float64),
        }
        out = _build_joined_plan(
            base,
            player_elo=pl.DataFrame({"Player_ID": pl.Series([], dtype=pl.String), **empty_elo}),
            pair_elo=pl.DataFrame({"Pair_IDs": pl.Series([], dtype=pl.String), **empty_elo}),
        ).collect()
        self.assertEqual(out["section_id"].to_list(), ["2503355-2901-2-0"])
        self.assertEqual(out["session_id"].to_list(), ["2503355-2901-2"])
        self.assertEqual(out["event_id"].to_list(), ["2503355-2901"])
        self.assertNotIn("hand_record_id", out.columns)


class SchemaBlacklistTests(unittest.TestCase):
    def test_ids_and_dates_are_not_model_inputs(self) -> None:
        from mlBridge.mlBridgeAiLib import generate_and_save_schema

        frame = pl.DataFrame(
            {
                "Pct_NS": pl.Series([0.4, 0.6, 0.5], dtype=pl.Float32),
                "HCP_N": pl.Series([10, 12, 8], dtype=pl.UInt8),
                "club_class": pl.Series([3, 4, 4], dtype=pl.UInt8),
                "Club": pl.Series([1, 2, 3], dtype=pl.Int32),
                "section_id": pl.Series([7, 8, 9], dtype=pl.Int64),
                "session_id": ["a", "b", "c"],
                "event_id": pl.Series([1, 2, 3], dtype=pl.Int64),
                "Date": [date(2025, 1, 1)] * 3,
                "game_date": pl.Series([date(2025, 1, 1)] * 3).cast(pl.Datetime("us")),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            schema = generate_and_save_schema(frame, Path(tmp), "m", "Pct_NS", y_range=None)
        self.assertEqual(sorted(schema["feature_dtypes"]), ["HCP_N", "club_class"])


if __name__ == "__main__":
    unittest.main()
