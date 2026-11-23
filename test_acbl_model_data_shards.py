"""Step 5a shard reuse: a shard is kept only when it was built from the same sources."""
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path

import polars as pl

import acbl_model_data as m


class ShardSourceIdTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.brs = self.dir / "brs.parquet"
        self.hrs = self.dir / "hrs.parquet"
        for f in (self.brs, self.hrs):
            f.write_bytes(b"x")
        self.shard = self.dir / "window=2019-01-01_2019-02-01.parquet"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_shard(self) -> None:
        pl.DataFrame({"Date": [date(2019, 1, 5), date(2019, 1, 9)]}).write_parquet(self.shard)

    def _id(self, columns=("A", "B")) -> str:
        return m._source_id(brs_file=self.brs, hrs_file=self.hrs, columns=list(columns))

    def test_id_changes_with_a_rewritten_source_or_other_columns(self) -> None:
        first = self._id()
        self.assertEqual(first, self._id(("B", "A")))  # order does not matter
        self.assertNotEqual(first, self._id(("A", "B", "SC_ET_NS_S")))
        st = self.brs.stat()
        os.utime(self.brs, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(first, self._id())

    def test_shard_with_the_same_rows_but_another_source_is_rebuilt(self) -> None:
        self._write_shard()
        sid = self._id()
        self.assertTrue(m._model_shard_is_current(self.shard, 10, date(2019, 1, 9), 10, "w",
                                                  source_id=sid, recorded_source_id=sid))
        self.assertTrue(self.shard.exists())
        self.assertFalse(m._model_shard_is_current(self.shard, 10, date(2019, 1, 9), 10, "w",
                                                   source_id=sid, recorded_source_id="older"))
        self.assertFalse(self.shard.exists())

    def test_shard_from_a_manifest_without_source_id_is_rebuilt(self) -> None:
        self._write_shard()
        self.assertFalse(m._model_shard_is_current(self.shard, 10, date(2019, 1, 9), 10, "w",
                                                   source_id=self._id(), recorded_source_id=None))

    def test_manifest_round_trip_keeps_the_source_id(self) -> None:
        self._write_shard()
        sid = self._id()
        m._write_shard_manifest(self.dir, [self.shard], label="club", source_by_file={
            self.shard.name: {"source_rows": 10, "source_date_max": "2019-01-09", "source_id": sid},
        })
        self.assertEqual(m._read_recorded_source_ids(self.dir), {self.shard.name: sid})
        self.assertEqual(m._read_recorded_source_rows(self.dir), {self.shard.name: 10})


if __name__ == "__main__":
    unittest.main()
