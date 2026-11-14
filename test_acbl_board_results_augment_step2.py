import datetime
import pathlib
import tempfile
import unittest
from unittest import mock

import duckdb
import polars as pl
import pyarrow.parquet as pq

import acbl_board_results_augment_step2 as step2
import mlBridge.mlBridgeAugmentLib as mlBridgeAugmentLib


def _main_frame(start: int, rows: int) -> pl.DataFrame:
    ids = list(range(start, start + rows))
    return pl.DataFrame(
        {
            "session_id": pl.Series(ids, dtype=pl.Int64),
            "Board": pl.Series([i % 36 + 1 for i in ids], dtype=pl.UInt8),
            "Contract": [None if i % 5 == 0 else f"{i % 7 + 1}N" for i in ids],
            "Date": [datetime.date(2024, 1, 1) + datetime.timedelta(days=i % 300) for i in ids],
            "Pct_NS": pl.Series([i / 1000 for i in ids], dtype=pl.Float32),
            "Flags": [[i, i + 1] if i % 3 else [] for i in ids],
        }
    )


def _side_frame(start: int, rows: int) -> pl.DataFrame:
    ids = list(range(start, start + rows))
    return pl.DataFrame(
        {
            "Declarer_Rating": pl.Series([None if i % 4 == 0 else i * 0.5 for i in ids], dtype=pl.Float32),
            "Defender_OnLead_Rating": pl.Series([i * 0.25 for i in ids], dtype=pl.Float32),
        }
    )


class ConcatParquetFilesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        patcher = mock.patch.object(step2, "ROW_GROUP_SIZE", 4)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, frame: pl.DataFrame) -> pathlib.Path:
        path = self.dir / name
        step2._write_shard(frame, path)
        return path

    def _write_side(self, name: str, frame: pl.DataFrame, main: pathlib.Path) -> pathlib.Path:
        # Same row groups as the main file, as _write_rating_files writes them.
        md = pq.ParquetFile(main).metadata
        path = self.dir / name
        table = frame.to_arrow()
        writer = pq.ParquetWriter(path, table.schema, compression=step2.COMPRESSION)
        at = 0
        for i in range(md.num_row_groups):
            n = md.row_group(i).num_rows
            writer.write_table(table.slice(at, n), row_group_size=n)
            at += n
        writer.close()
        return path

    def test_plain_concat_matches_the_frames(self) -> None:
        frames = [_main_frame(0, 10), _main_frame(10, 3), _main_frame(13, 8)]
        paths = [self._write(f"m{i}.parquet", f) for i, f in enumerate(frames)]
        out = self.dir / "out.parquet"
        self.assertEqual(step2.concat_parquet_files(paths, out), 21)
        expected = pl.concat(frames)
        self.assertTrue(pl.read_parquet(out).equals(expected))
        self.assertEqual(pq.ParquetFile(out).metadata.num_row_groups, 3 + 1 + 2)

    def test_side_columns_are_appended_to_every_row_group(self) -> None:
        sizes = [(0, 10), (10, 3), (13, 8)]
        mains = [_main_frame(*s) for s in sizes]
        sides = [_side_frame(*s) for s in sizes]
        paths = [self._write(f"m{i}.parquet", f) for i, f in enumerate(mains)]
        side_paths = [self._write_side(f"s{i}.parquet", f, paths[i]) for i, f in enumerate(sides)]
        out = self.dir / "out.parquet"
        self.assertEqual(step2.concat_parquet_files(paths, out, side_paths=side_paths), 21)
        expected = pl.concat([m.hstack(s) for m, s in zip(mains, sides)])

        # Three readers, since each finds columns in the footer its own way.
        self.assertTrue(pl.read_parquet(out).equals(expected))
        self.assertTrue(pl.from_arrow(pq.read_table(out)).equals(expected))
        via_duckdb = duckdb.sql(f"SELECT * FROM read_parquet('{out.as_posix()}')").pl()
        self.assertEqual(via_duckdb.columns, expected.columns)
        self.assertEqual(via_duckdb["Declarer_Rating"].to_list(), expected["Declarer_Rating"].to_list())
        self.assertEqual(via_duckdb["Contract"].to_list(), expected["Contract"].to_list())

        # Reading only side columns, only main columns, and one row group.
        self.assertTrue(pl.read_parquet(out, columns=["Defender_OnLead_Rating"]).equals(expected.select("Defender_OnLead_Rating")))
        self.assertTrue(pl.read_parquet(out, columns=["Flags", "Date"]).equals(expected.select("Flags", "Date")))
        md = pq.ParquetFile(out).metadata
        self.assertEqual(md.num_row_groups, 6)
        self.assertEqual(md.num_columns, pq.ParquetFile(paths[0]).metadata.num_columns + 2)
        self.assertTrue(pl.from_arrow(pq.ParquetFile(out).read_row_group(3)).equals(expected.slice(10, 3)))
        self.assertEqual(pl.read_parquet_schema(out)["Declarer_Rating"], pl.Float32)

    def test_row_groups_follow_one_another_without_gaps_or_overlap(self) -> None:
        mains = [_main_frame(0, 9), _main_frame(9, 6)]
        sides = [_side_frame(0, 9), _side_frame(9, 6)]
        paths = [self._write(f"m{i}.parquet", f) for i, f in enumerate(mains)]
        side_paths = [self._write_side(f"s{i}.parquet", f, paths[i]) for i, f in enumerate(sides)]
        out = self.dir / "out.parquet"
        step2.concat_parquet_files(paths, out, side_paths=side_paths)
        md = pq.ParquetFile(out).metadata
        end = 4
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            chunks = []
            for j in range(rg.num_columns):
                col = rg.column(j)
                first = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
                chunks.append((first, first + col.total_compressed_size))
            # Each row group starts where the one before it ended, and its column chunks
            # do not overlap. (Polars leaves a few bytes of its own between chunks.)
            self.assertEqual(min(c[0] for c in chunks), end)
            chunks.sort()
            for (_, a_end), (b_start, _) in zip(chunks, chunks[1:]):
                self.assertLessEqual(a_end, b_start)
            end = max(c[1] for c in chunks)
        self.assertEqual(md.row_group(0).column(0).has_offset_index, False)

    def test_side_file_with_other_row_groups_is_refused(self) -> None:
        main = self._write("m.parquet", _main_frame(0, 10))
        side = self.dir / "s.parquet"
        pq.write_table(_side_frame(0, 10).to_arrow(), side, row_group_size=5, compression=step2.COMPRESSION)
        with self.assertRaisesRegex(ValueError, "row groups do not match"):
            step2.concat_parquet_files([main], self.dir / "out.parquet", side_paths=[side])

    def test_side_column_already_in_main_is_refused(self) -> None:
        main = self._write("m.parquet", _main_frame(0, 4))
        side = self._write_side("s.parquet", _main_frame(0, 4).select("Pct_NS"), main)
        with self.assertRaisesRegex(ValueError, "already in the main file"):
            step2.concat_parquet_files([main], self.dir / "out.parquet", side_paths=[side])


class BlockTests(unittest.TestCase):
    def test_block_starts_and_slices(self) -> None:
        with mock.patch.object(step2, "SESSION_BLOCK", 100):
            ids = pl.Series([5, 5, 99, 100, 250, 250, 299, 1000], dtype=pl.Int64)
            self.assertEqual(step2._block_starts(ids), [0, 100, 200, 1000])
            self.assertEqual(step2._slice_bounds(ids, 0, 100), (0, 3))
            self.assertEqual(step2._slice_bounds(ids, 100, 200), (3, 1))
            self.assertEqual(step2._slice_bounds(ids, 200, 300), (4, 3))
            self.assertEqual(step2._slice_bounds(ids, 300, 400), (7, 0))
            self.assertEqual(step2._slice_bounds(ids, 1000, 1100), (7, 1))

    def test_club_blocks_pair_step1_and_hand_record_slices(self) -> None:
        with mock.patch.object(step2, "SESSION_BLOCK", 100):
            brs_ids = pl.Series([5, 5, 99, 250, 1000], dtype=pl.Int64)
            hrs_ids = pl.Series([5, 99, 100, 1000, 1000], dtype=pl.Int64)
            self.assertEqual(
                step2._club_blocks(brs_ids, hrs_ids),
                [(0, (0, 3), (0, 2)), (200, (3, 1), (3, 0)), (1000, (4, 1), (3, 2))],
            )

    def test_tournament_block_keys(self) -> None:
        ids = pl.Series(["1502054-2005-1", "2611003-01OP-2", "NABC163-SS02-1", "YNABC21-17Y-2", "ab-1", "x"])
        self.assertEqual(
            step2._tournament_block_keys(ids).to_list(),
            ["1502", "2611", "NABC16", "YNABC21", "ab-1", "x"],
        )

    def test_tournament_blocks_pair_step1_and_hand_record_slices(self) -> None:
        brs_ids = pl.Series(["1502054-2005-1", "1502054-2005-2", "1503001-01-1", "NABC163-SS02-1", "NABC171-01-1"])
        # No hand records for 1502; an extra month (1601) that step1 does not have.
        hrs_ids = pl.Series(["1503001-01-1", "1503001-01-1", "1601001-01-1", "NABC163-SS02-1", "NABC171-01-1"])
        self.assertEqual(
            step2._tournament_blocks(brs_ids, hrs_ids),
            [("1502", (0, 2), (0, 0)), ("1503", (2, 1), (0, 2)), ("NABC16", (3, 1), (3, 1)), ("NABC17", (4, 1), (4, 1))],
        )

    def test_key_runs_refuses_unsorted_ids(self) -> None:
        with self.assertRaises(RuntimeError):
            step2._key_runs(pl.Series(["1502", "1503", "1502"]))

    def test_block_names(self) -> None:
        self.assertEqual(step2._block_name(25000), "block_00025000")
        self.assertEqual(step2._block_name("NABC16"), "block_NABC16")

    def test_fingerprint_ignores_where_the_block_sits_in_the_file(self) -> None:
        rows = pl.DataFrame({"session_id": [7, 8], "Board": [1, 2]})
        hashes = pl.Series([11, 22, 33, 44, 55], dtype=pl.UInt64)
        a = step2._block_fingerprint("c", rows, hashes.slice(1, 2))
        b = step2._block_fingerprint("c", rows, pl.Series([22, 33], dtype=pl.UInt64))
        self.assertEqual(a, b)
        self.assertNotEqual(a, step2._block_fingerprint("c", rows, hashes.slice(2, 2)))
        self.assertNotEqual(a, step2._block_fingerprint("c", rows.reverse(), hashes.slice(1, 2)))
        self.assertNotEqual(a, step2._block_fingerprint("d", rows, hashes.slice(1, 2)))


class ValidatePercentageTests(unittest.TestCase):
    def test_counts_out_of_range_rows_and_ignores_nulls(self) -> None:
        import contextlib
        import io

        frame = pl.DataFrame({"Pct_NS": [0.0, 1.0, 1.5, None, -0.1], "Pct_EW": [0.5, None, 0.2, 1.0, 0.0]})
        with mock.patch.object(step2, "PCT_VALIDATION_COLS", ["Pct_NS", "Pct_EW", "Missing_Col"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                step2._validate_percentage_columns(frame, "t")
        lines = buf.getvalue().splitlines()
        self.assertEqual(
            lines,
            [
                "  WARNING (t): validation column 'Missing_Col' not found -- skipping",
                "  WARNING (t): 2 rows have 'Pct_NS' outside [0,1]",
            ],
        )


class RatingFilesTests(unittest.TestCase):
    def test_ratings_average_over_all_blocks(self) -> None:
        inputs = [
            pl.DataFrame(
                {
                    "Declarer_ID": ["a", "b", "a", "c", "b", "a", "a"],
                    "OnLead": ["x", "y", "x", "x", "y", "z", "z"],
                    "NotOnLead": ["p", "p", "q", "q", "p", "q", "p"],
                    "DD_Tricks_Diff": pl.Series([1, -1, 0, 2, None, -2, 1], dtype=pl.Int8),
                    "Defender_Par_GE": [True, False, True, True, False, None, True],
                }
            ),
            pl.DataFrame(
                {
                    "Declarer_ID": ["a", "d", "b"],
                    "OnLead": ["y", "y", "x"],
                    "NotOnLead": ["p", "r", "q"],
                    "DD_Tricks_Diff": pl.Series([3, 0, -3], dtype=pl.Int8),
                    "Defender_Par_GE": [False, True, False],
                }
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(step2, "ROW_GROUP_SIZE", 3):
            cache_dir = pathlib.Path(tmp)
            entries = []
            start = 0
            for i, frame in enumerate(inputs):
                name = step2._block_name(i * step2.SESSION_BLOCK)
                step2._write_shard(_main_frame(start, frame.height), cache_dir / f"{name}.parquet")
                step2._write_shard(frame, cache_dir / f"{name}.ratings_in.parquet")
                entries.append({"file": f"{name}.parquet", "inputs": f"{name}.ratings_in.parquet", "rows": frame.height})
                start += frame.height
            side_files = step2._write_rating_files(cache_dir, entries)
            out = cache_dir / "out.parquet"
            shards = [cache_dir / e["file"] for e in entries]
            self.assertEqual(step2.concat_parquet_files(shards, out, side_paths=side_files), 10)
            got = pl.read_parquet(out)
            # Row groups of the side files follow the shards: 3+3+1 and 3.
            self.assertEqual([pq.ParquetFile(f).metadata.num_row_groups for f in side_files], [3, 1])

        expected = mlBridgeAugmentLib.add_player_ratings(pl.concat(inputs)).select(step2.RATING_COLS)
        self.assertEqual(got.columns, _main_frame(0, 1).columns + step2.RATING_COLS)
        self.assertTrue(got.select(step2.RATING_COLS).equals(expected))
        self.assertTrue(got.drop(step2.RATING_COLS).equals(_main_frame(0, 10)))
        # Player "a" declares in both blocks: one average over all five of their rows.
        a_rows = got.filter(pl.concat(inputs)["Declarer_ID"] == "a")["Declarer_Rating"].to_list()
        self.assertEqual(len(set(a_rows)), 1)
        self.assertAlmostEqual(a_rows[0], (1 + 0 - 2 + 1 + 3) / 5, places=6)


if __name__ == "__main__":
    unittest.main()
