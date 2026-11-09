"""Run from the repository root: python -m unittest discover"""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sweep_storage as storage
from plot import summarize_file


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "L3_tau1.csv"
        self.count = 1000
        rows = []
        for sample in range(1, 4):
            for step in range(1, self.count + 1):
                distance = sample + .2 * np.sin(step / 20)
                rows.append([3, 1, sample, sample + 6, step, (step - 1) * .01,
                             distance, distance / 3])
        self.raw = np.array(rows)
        with self.path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(storage.RAW_HEADER)
            writer.writerows(rows)

    def tearDown(self):
        self.directory.cleanup()

    def test_compaction_preserves_plot_means_and_uncertainty(self):
        expected = summarize_file((self.path, 3))
        with patch.object(storage, "CHUNK_ROWS", 137):
            old, new = storage.compact_file(self.path)
        self.assertLess(new, old / 5)
        np.testing.assert_allclose(summarize_file((self.path, 3)), expected, rtol=1e-13)
        self.assertEqual(storage.completed_means(self.path, 3, 1, 7, 1, 1, 1000, 3, .01), 3)
        data = np.loadtxt(self.path, delimiter=",", skiprows=1)
        np.testing.assert_allclose([data[data[:, 2] == s, 7].sum() for s in (1, 2, 3)], 1000)

    def test_atomic_failure_keeps_original(self):
        original = self.path.read_bytes()
        # A partial last sample must not replace the source with incomplete means.
        lines = original.splitlines(keepends=True)
        self.path.write_bytes(b"".join(lines[:-10]))
        incomplete = self.path.read_bytes()
        with self.assertRaises(ValueError):
            storage.compact_file(self.path)
        self.assertEqual(self.path.read_bytes(), incomplete)
        self.assertFalse(self.path.with_name(self.path.name + ".compact.tmp").exists())

    def test_resume_discards_only_incomplete_run(self):
        storage.compact_file(self.path)
        lines = self.path.read_bytes().splitlines(keepends=True)
        expected = b"".join([lines[0]] + [r for r in lines[1:] if float(r.split(b",")[2]) <= 2])
        self.path.write_bytes(expected + b"3,1,3,9,")
        self.assertEqual(storage.completed_means(self.path, 3, 1, 7, 1, 1, 1000, 3, .01), 2)
        self.assertEqual(self.path.read_bytes(), expected)

    def test_resume_of_partial_first_raw_sample(self):
        with self.path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(storage.RAW_HEADER)
            writer.writerows(self.raw[:40])
        self.assertEqual(storage.completed_samples(self.path, 3, 1, 7, 1, 1, 1000, 3, .01), 0)
        self.assertEqual(self.path.read_text().strip().split(","), storage.HEADER)


if __name__ == "__main__":
    unittest.main()
