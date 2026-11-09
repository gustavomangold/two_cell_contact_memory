"""Check that the compiled model reproduces the NumPy reference."""

import unittest
from unittest.mock import patch

import numpy as np
import two_disks as simulation


class AccelerationTests(unittest.TestCase):
    def check_results(self, expected, actual):
        for reference, result in zip(expected, actual):
            self.assertEqual(reference.dtype, result.dtype)
            np.testing.assert_array_equal(reference, result)

    def test_numpy_fallback(self):
        with patch.multiple(simulation, DURATION=.1, STEADY_STATE_STEPS=3,
                            _simulate_kernel=None):
            self.check_results(simulation._simulate_python(), simulation.simulate())

    def test_contact_turning_can_be_disabled(self):
        with patch.multiple(simulation, DURATION=1.2, DT=.001, STEADY_STATE_STEPS=0,
                            SPEED=1., REPULSION=0., WALL_REPULSION=0.,
                            TRANSLATIONAL_DIFFUSION=0., ANGULAR_DIFFUSION=0.,
                            PERSISTENCE_TIME=0., SAVE_EVERY=1):
            with patch.object(simulation, "REORIENT_ON_CONTACT", False):
                no_turn = simulation.simulate()[1]
            with patch.object(simulation, "REORIENT_ON_CONTACT", True):
                turn = simulation.simulate()[1]
            self.assertGreater(no_turn[-1, 0, 0], 0.)
            self.assertLess(turn[-1, 0, 0], -.5)

    def test_translational_diffusion(self):
        with patch.multiple(simulation, DURATION=.001, DT=.001, STEADY_STATE_STEPS=0,
                            SPEED=0., REPULSION=0., WALL_REPULSION=0.,
                            TRANSLATIONAL_DIFFUSION=.001, ANGULAR_DIFFUSION=0.,
                            SAVE_EVERY=1):
            rng = np.random.default_rng(simulation.SEED)
            expected = np.array([[-1., 0.], [1., 0.]])
            expected += np.sqrt(2 * .001 * .001) * rng.normal(size=(2, 2))
            np.testing.assert_array_equal(simulation.simulate()[1][-1], expected)

    @unittest.skipIf(simulation._simulate_kernel is None, "Numba is optional")
    def test_compiled_matches_reference(self):
        cases = [
            {},
            dict(SEED=8, ARENA_DIAMETER=2., SPEED=20., REPULSION=5., PERSISTENCE_TIME=0.),
            dict(SEED=9, ARENA_DIAMETER=4., PERSISTENCE_TIME=.25),
            dict(ANGULAR_DIFFUSION=0, PERSISTENCE_TIME=.1, DURATION=3.),
            dict(ANGULAR_DIFFUSION=360, REPULSION=1., DT=.0001, DURATION=.1),
            dict(DT=.003, DURATION=.021, STEADY_STATE_STEPS=13,
                 STEP_SEPARATION=4, SAVE_EVERY=6),
        ]
        for case in cases:
            with self.subTest(case=case):
                settings = dict(DURATION=1., STEADY_STATE_STEPS=0)
                settings.update(case)
                with patch.multiple(simulation, **settings):
                    self.check_results(simulation._simulate_python(), simulation.simulate())

    @unittest.skipIf(simulation._simulate_kernel is None, "Numba is optional")
    def test_generator_stream(self):
        rng = np.random.default_rng(7)
        reference = np.random.default_rng(7)
        simulation._simulate_kernel(rng, 5., 1., 30., 10., 2., .001,
                                    .1, 3, 5, 20, 1)
        reference.normal(size=(103, 2))
        np.testing.assert_array_equal(rng.normal(size=20), reference.normal(size=20))


if __name__ == "__main__":
    unittest.main()
