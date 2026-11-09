"""Small end-to-end checks of parallel sweeps and resume behavior."""

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sweep
import two_disks as simulation


class SweepTests(unittest.TestCase):
    def test_tau_grids(self):
        self.assertEqual(sweep.tau_grid([.25], [(0, .5, .125)]), [0, .125, .25, .375, .5])
        self.assertEqual(sweep.tau_grid(None, [(0, .25, .075)]), [0, .075, .15, .225])
        self.assertEqual(sweep.tau_grid([1e-5, 1e-4], None), [1e-5, 1e-4])
        for ranges in [[(0, 1, 0)], [(1, 0, .1)], [(0, float("nan"), 1)]]:
            with self.assertRaises(ValueError):
                sweep.tau_grid(None, ranges)

    def test_parallel_resume_and_recorded_means(self):
        settings = {name: value for name, value in vars(simulation).items()
                    if name.isupper() and isinstance(value, (int, float))}
        settings.update(DT=.001, DURATION=.1, STEADY_STATE_STEPS=50,
                        STEP_SEPARATION=7, SAVE_EVERY=20, SPEED=20.,
                        REORIENT_ON_CONTACT=True)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            sweep.run_sweep([3], [.25], 3, 2, folder, settings)
            path = folder / "L3_tau0.25.csv"
            original = path.read_bytes()
            sweep.run_sweep([3], [.25], 3, 2, folder, settings)
            self.assertEqual(path.read_bytes(), original)

            # An interrupted final run is redone; earlier bytes stay intact.
            with path.open("ab") as file:
                file.write(b"3,0.25,4,10,")
            sweep.run_sweep([3], [.25], 4, 2, folder, settings)
            self.assertTrue(path.read_bytes().startswith(original))
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            for sample in range(1, 5):
                with patch.multiple(simulation, **{**settings, "ARENA_DIAMETER": 3.,
                                                  "PERSISTENCE_TIME": .25,
                                                  "SEED": settings["SEED"] + sample - 1}):
                    distances = simulation.simulate()[3]
                saved = data[data[:, 2] == sample]
                self.assertEqual(saved[:, 7].sum(), len(distances))
                np.testing.assert_allclose(np.average(saved[:, 8], weights=saved[:, 7]), distances.mean())
                np.testing.assert_allclose(saved[:, 6], (saved[:, 5] - 1) * settings["DT"])

            complete = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Settings changed"):
                sweep.run_sweep([3], [.25], 4, 2, folder, {**settings, "DT": .0001})
            self.assertEqual(path.read_bytes(), complete)


if __name__ == "__main__":
    unittest.main()
