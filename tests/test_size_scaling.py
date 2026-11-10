"""Size-study mechanics, normalization and regression against the original pair."""

import unittest
from unittest.mock import patch

import numpy as np
import two_disks as simulation
from contact_stats import make_recorder, finish
from many_disks import HAS_NUMBA, _simulate_python, initial_state, make_model, normalize, simulate_kernel
from sweep_size_scaling import arena_size
from sweep_storage import window_edges


def settings(**changes):
    result = {name: value for name, value in vars(simulation).items()
              if name.isupper() and isinstance(value, (int, float))}
    result.update(REORIENT_ON_CONTACT=False, DT=.001, DURATION=.15,
                  STEADY_STATE_STEPS=13, SAVE_EVERY=7, SPEED=20.)
    result.update(changes)
    return result


class SizeScalingTests(unittest.TestCase):
    def test_requested_sizes_and_optional_density_control(self):
        sizes = [2, 4, 8, 16, 32]
        np.testing.assert_array_equal([arena_size(n) for n in sizes], [3, 6, 12, 24, 48])
        np.testing.assert_allclose([n / arena_size(n, scaling="constant-density") ** 2 for n in sizes], 2/9)

    def test_initialization_and_original_rng_stream(self):
        rng, reference = np.random.default_rng(7), np.random.default_rng(7)
        positions, angles = initial_state(2, 3., rng)
        np.testing.assert_array_equal(positions, [[-1, 0], [1, 0]])
        np.testing.assert_array_equal(angles, [0, np.pi])
        np.testing.assert_array_equal(rng.normal(size=10), reference.normal(size=10))
        for n in [4, 8, 16, 32]:
            arena = arena_size(n, scaling="constant-density")
            positions, _ = initial_state(n, arena, np.random.default_rng(7))
            self.assertTrue((np.linalg.norm(positions, axis=1) <= arena / 2 - .5).all())
            for i in range(n):
                self.assertTrue((np.linalg.norm(positions[i + 1:] - positions[i], axis=1) >= 1).all())

    def test_pair_trajectory_and_contact_fraction_match_existing_simulation(self):
        for tau, angular in [(0., 0.), (.01, 0.), (.2, 10.), (.001, 360.)]:
            with self.subTest(tau=tau, angular=angular):
                values = settings(ARENA_DIAMETER=3., PERSISTENCE_TIME=tau, ANGULAR_DIFFUSION=angular)
                with patch.multiple(simulation, **values):
                    recorder = make_recorder(values, max_lag=.01)
                    expected = simulation.simulate(recorder)
                    expected_stats = finish(recorder)["summary"]
                rng = np.random.default_rng(values["SEED"])
                positions, angles = initial_state(2, 3., rng)
                model = make_model(values, 3., tau)
                result = simulate_kernel(rng, positions, angles, model, window_edges(model.steps), record=True)
                np.testing.assert_array_equal(result[1], expected[0])
                np.testing.assert_array_equal(result[2], expected[1])
                np.testing.assert_array_equal(result[3][:, 0], expected[2])
                metrics = normalize(result[0].sum(axis=0), 2, model.steps)
                for index, name in [(0, "contact_fraction"), (3, "persistent_fraction"),
                                    (6, "wall_fraction_green"), (7, "wall_fraction_red")]:
                    self.assertEqual(metrics[index], expected_stats[name])

    def test_same_type_repulsion_does_not_activate_memory(self):
        values = settings(SPEED=0., ANGULAR_DIFFUSION=0., TRANSLATIONAL_DIFFUSION=0.,
                          DURATION=.001, STEADY_STATE_STEPS=0)
        positions = np.array([[-.4, 0], [.4, 0], [8., 0], [10., 0]])
        model = make_model(values, 30., .1)
        result = simulate_kernel(np.random.default_rng(7), positions, np.zeros(4), model, window_edges(1))
        self.assertLess(result[4][0, 0], positions[0, 0])
        self.assertGreater(result[4][1, 0], positions[1, 0])
        np.testing.assert_array_equal(result[6], 0)
        metrics = normalize(result[0].sum(axis=0), 4, 1)
        self.assertEqual(metrics[0], 0)
        self.assertEqual(metrics[4], 1)
        self.assertEqual(metrics[5], .5)

    def test_multiple_unlike_contacts_use_one_green_timer_and_any_contact_fraction(self):
        values = settings(SPEED=0., ANGULAR_DIFFUSION=0., TRANSLATIONAL_DIFFUSION=0.,
                          DURATION=.001, STEADY_STATE_STEPS=0)
        positions = np.array([[0., 0], [8., 0], [.4, 0], [-.4, 0]])
        model = make_model(values, 30., .1)
        result = simulate_kernel(np.random.default_rng(7), positions, np.zeros(4), model, window_edges(1))
        np.testing.assert_allclose(result[6], [.1, 0])
        metrics = normalize(result[0].sum(axis=0), 4, 1)
        self.assertEqual(metrics[0], .5)  # One of two greens has unlike contact.
        self.assertEqual(metrics[1], 1.)  # Both reds touch the same green.
        self.assertEqual(metrics[2], 1.)  # Two unlike contacts, divided by two greens.

    def test_neighbor_list_preserves_forces_and_random_stream(self):
        values = settings(DT=.005, DURATION=.5, TRANSLATIONAL_DIFFUSION=.3, ANGULAR_DIFFUSION=10.)
        model = make_model(values, 6., .1)
        results = []
        for use_list in (False, True):
            rng = np.random.default_rng(9)
            positions, angles = initial_state(8, 6., rng)
            result = simulate_kernel(rng, positions, angles, model, window_edges(model.steps),
                                     record=True, neighbor_list=use_list)
            results.append((result, rng.normal(size=12)))
        for expected, actual in zip(results[0][0], results[1][0]):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(results[1][1], results[0][1])

    @unittest.skipUnless(HAS_NUMBA, "Numba is optional")
    def test_compiled_matches_readable_general_loop(self):
        model = make_model(settings(DURATION=.03), 6., .01)
        results = []
        for kernel in (_simulate_python, simulate_kernel):
            rng = np.random.default_rng(11)
            positions, angles = initial_state(4, 6., rng)
            results.append(kernel(rng, positions, angles, model, window_edges(model.steps), record=True))
        for expected, actual in zip(*results):
            np.testing.assert_array_equal(actual, expected)

    def test_no_neighbors_has_undefined_demixing(self):
        raw = np.zeros(8)
        metrics = normalize(raw, 4, 100)
        self.assertTrue(np.isnan(metrics[4]))
        self.assertEqual(metrics[5], 0)


if __name__ == "__main__":
    unittest.main()
