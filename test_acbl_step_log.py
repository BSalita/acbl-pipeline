"""Tests for the Polars spill settings acbl_step_log.py gives each step."""

import os
import shutil
import tempfile
import unittest

import acbl_step_log


class SpillSettingsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.folder = os.path.join(self.tmp, "polars-spill")
        self.free_gb = shutil.disk_usage(self.tmp).free / 2**30

    def test_sets_folder_and_limit_from_free_space(self):
        added, message = acbl_step_log.spill_settings({}, spill_dirs=(self.folder,), min_free_gb=0, reserve_gb=0)
        self.assertEqual(added["POLARS_OOC_SPILL_DIR"], self.folder)
        self.assertTrue(os.path.isdir(self.folder))
        # Free space moves a little between the two readings.
        self.assertAlmostEqual(int(added["POLARS_OOC_DISK_BUDGET_MB"]) / 1024, self.free_gb, delta=1 + self.free_gb * 0.05)
        self.assertIn(self.folder, message)

    def test_reserve_is_kept_free(self):
        reserve = self.free_gb / 2
        added, _ = acbl_step_log.spill_settings({}, spill_dirs=(self.folder,), min_free_gb=0, reserve_gb=reserve)
        self.assertAlmostEqual(int(added["POLARS_OOC_DISK_BUDGET_MB"]) / 1024, self.free_gb - reserve, delta=1 + self.free_gb * 0.05)

    def test_existing_settings_are_left_alone(self):
        added, message = acbl_step_log.spill_settings({"POLARS_OOC_SPILL_DIR": r"X:\mine"}, spill_dirs=(self.folder,), min_free_gb=0)
        self.assertEqual(added, {})
        self.assertIn(r"X:\mine", message)
        self.assertFalse(os.path.exists(self.folder))

        added, message = acbl_step_log.spill_settings({"POLARS_OOC_DISK_BUDGET_MB": "2048"}, spill_dirs=(self.folder,), min_free_gb=0)
        self.assertEqual(added, {"POLARS_OOC_SPILL_DIR": self.folder})
        self.assertIn("spill limit 2 GB", message)

    def test_drive_without_enough_room_or_missing_is_skipped(self):
        missing = "?:\\no-such-drive\\spill"
        added, message = acbl_step_log.spill_settings({}, spill_dirs=(missing, self.folder), min_free_gb=self.free_gb * 10)
        self.assertEqual(added, {})
        self.assertIn("WARNING", message)

        added, _ = acbl_step_log.spill_settings({}, spill_dirs=(missing, self.folder), min_free_gb=0, reserve_gb=0)
        self.assertEqual(added["POLARS_OOC_SPILL_DIR"], self.folder)


if __name__ == "__main__":
    unittest.main()
