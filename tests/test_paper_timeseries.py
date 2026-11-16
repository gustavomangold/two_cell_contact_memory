"""Time-series observations must preserve the paper trajectory and resume safely."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from demixing import snapshot_demixing
from paper_engine import DEFAULTS, ROOT, build_engine, run_sample, source_hash
from paper_timeseries import RECORDER, record_sample, recorder_hash


class PaperTimeSeriesTests(unittest.TestCase):
    def test_observation_preserves_initial_final_demixing_and_rng(self):
        original_hash = source_hash()
        settings = {**DEFAULTS, "duration": .002, "time_points": 5,
                    "source_hash": original_hash, "recorder": RECORDER,
                    "recorder_hash": recorder_hash()}
        engine = build_engine()
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            for number in [4, 32]:
                with self.subTest(N=number):
                    reference = work / f"reference_{number}.npz"
                    path = work / f"series_{number}.npz"
                    run_sample((number, 360000., .001, 1, 7, settings, engine, reference))
                    record_sample((number, 360000., .001, 1, 7, settings, path, None))
                    with np.load(reference) as old, np.load(path) as new:
                        np.testing.assert_array_equal(new["final_rng_state"], old["final_rng_state"])
                        np.testing.assert_allclose(new["demixing"][-1], old["final_metrics"], rtol=0, atol=0, equal_nan=True)
                        np.testing.assert_allclose(new["demixing"][0],
                            snapshot_demixing(old["initial_positions"], box_size=float(old["L"])),
                            rtol=0, atol=0, equal_nan=True)
                        self.assertEqual(new["demixing"].shape, (5, 6))
                        self.assertEqual(new["times"][0], 0.)
                        self.assertEqual(new["times"][-1], float(old["elapsed_time"]))
                        self.assertTrue((np.diff(new["times"]) > 0).all())
        self.assertEqual(source_hash(), original_hash)

    def test_cli_records_resumes_and_plots_selected_taus(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data, figures = work / "data", work / "figures"
            command = [sys.executable, str(ROOT / "src/paper_timeseries.py"),
                       "--sizes", "4", "--tau", ".001", ".01", "1",
                       "--duration", ".0002", "--time-points", "5", "--samples", "2",
                       "--workers", "2", "--output", str(data)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            paths = sorted(data.glob("N*/sample_*.npz"))
            self.assertEqual(len(paths), 6)
            timestamps = {p:p.stat().st_mtime_ns for p in paths}
            resumed = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(resumed.stdout.count("2/2 saved"), 3)
            self.assertEqual(timestamps, {p:p.stat().st_mtime_ns for p in paths})
            subprocess.run([sys.executable, str(ROOT / "src/paper_plot.py"), "--time-series",
                            "--data-dir", str(data), "--output", str(figures)],
                           check=True, capture_output=True, text=True)
            self.assertTrue((figures / "paper_demixing_time_N4_D360000.png").is_file())
            with (figures / "paper_demixing_time.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 15)
            self.assertEqual({float(r["tau"]) for r in rows}, {.001, .01, 1.})
            self.assertTrue(all(0 <= int(r["runs"]) <= 2 for r in rows))
            # Never present an ordinary sweep's final states as a full time series.
            ordinary = work / "ordinary"
            ordinary.mkdir()
            (ordinary / "settings.json").write_text(json.dumps(DEFAULTS))
            bad = subprocess.run([sys.executable, str(ROOT / "src/paper_plot.py"),
                                  "--time-series", "--data-dir", str(ordinary)],
                                 capture_output=True, text=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("only final demixing snapshots", bad.stderr)


if __name__ == "__main__":
    unittest.main()
