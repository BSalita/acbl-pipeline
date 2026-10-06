import unittest

import polars as pl

from acbl_prediction_data import _build_joined_plan


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


if __name__ == "__main__":
    unittest.main()
