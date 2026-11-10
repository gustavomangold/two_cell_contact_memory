"""Post-separation measurements on trajectories with known endpoints."""

import unittest
from unittest.mock import patch

import numpy as np
import two_disks as simulation
from contact_stats import make_recorder, _observe_python, finish
from plot_post_separation import selected_measurements


def settings(**changes):
    values = {name: value for name, value in vars(simulation).items()
              if name.isupper() and isinstance(value, (int, float))}
    values.update(ARENA_DIAMETER=6., DT=1., DURATION=10., STEADY_STATE_STEPS=0,
                  PERSISTENCE_TIME=3.)
    values.update(changes)
    return values


class PostSeparationTests(unittest.TestCase):
    def scripted(self, window=2., tau=3.):
        distances = [.9, 1.1, 1.4, 1.2, 1.3, .9, 1.1, .9, 1.1, 1.5]
        memory = [3., 3., 2., 1., 0., 3., 1., 5., 2., 1.] if tau else [0.] * 10
        recorder = make_recorder(settings(PERSISTENCE_TIME=tau), max_lag=1.,
                                 post_separation_time=window)
        for index, distance in enumerate(distances):
            _observe_python(recorder, index + 1, np.array([[distance, 0.], [0., 0.]]),
                            np.zeros(2), memory[index], bool(tau and index in (5, 7)))
        return finish(recorder), recorder

    def test_expiry_recontact_and_end_censoring(self):
        result, _ = self.scripted()
        events = result["post_separation"]
        np.testing.assert_array_equal(events[:, 0], [1, 6, 8])
        np.testing.assert_array_equal(events[:, 1], [3, 1, 1])
        np.testing.assert_array_equal(events[:, 2], [3, 1, 2])
        np.testing.assert_allclose(events[:, 3], [.2, .2, .4])
        np.testing.assert_allclose(events[:, 4], [.3, .2, .4])
        np.testing.assert_allclose(events[:, 5], [.3, 0, .4])
        np.testing.assert_array_equal(events[:, 7], [0, 1, 2])
        # A fresh memory duration of 5 at recontact must not extend the old trip.
        self.assertEqual(events[1, 1], 1)
        summary = result["summary"]
        self.assertEqual(summary["post_separation_active_fraction"], 1)
        self.assertEqual(summary["post_separation_complete_events"], 2)
        self.assertAlmostEqual(summary["post_separation_net_displacement"], .2)
        self.assertEqual(summary["post_separation_censored_fraction"], 1 / 3)

    def test_fixed_window_includes_recontact_and_logs_unfinished_windows(self):
        result, _ = self.scripted()
        fixed = result["fixed_after_separation"]
        np.testing.assert_array_equal(fixed[:, 0], [1, 6, 8])
        np.testing.assert_array_equal(fixed[:, 1], [2, 2, 1])
        np.testing.assert_allclose(fixed[:, 3], [.1, 0, .4])
        np.testing.assert_array_equal(fixed[:, 7], [0, 1, 0])
        np.testing.assert_array_equal(fixed[:, 10], [1, 1, 0])
        self.assertAlmostEqual(result["summary"]["fixed_separation_net_displacement"], .05)

    def test_no_memory_has_undefined_conditional_distance_but_a_fixed_control(self):
        result, _ = self.scripted(tau=0.)
        self.assertEqual(len(result["post_separation"]), 0)
        self.assertEqual(result["summary"]["post_separation_active_fraction"], 0)
        self.assertTrue(np.isnan(result["summary"]["post_separation_net_displacement"]))
        self.assertGreater(result["summary"]["fixed_separation_net_displacement"], 0)
        self.assertEqual(len(result["fixed_after_separation"]), 3)

    def test_overlapping_windows_survive_ring_buffer_wraparound(self):
        recorder = make_recorder(settings(), max_lag=1., post_separation_time=5.)
        for index, distance in enumerate([.9, 1.1, .9, 1.2, .9, 1.3, .9, 1.4, .9, 1.5]):
            _observe_python(recorder, index + 1, np.array([[distance, 0.], [0., 0.]]), np.zeros(2), 3., False)
        fixed = finish(recorder)["fixed_after_separation"]
        np.testing.assert_array_equal(fixed[:, 0], [1, 3, 5, 7, 9])
        np.testing.assert_array_equal(fixed[:, 1], [5, 5, 4, 2, 0])
        np.testing.assert_array_equal(fixed[:, 10], [1, 1, 0, 0, 0])

    def test_long_window_is_bounded_by_observation_length(self):
        result, recorder = self.scripted(window=20.)
        self.assertLessEqual(len(recorder.history), 10)
        np.testing.assert_array_equal(result["fixed_after_separation"][:, 10], 0)
        np.testing.assert_array_equal(result["fixed_after_separation"][:, 1], [8, 3, 1])

    def test_separation_waits_for_hysteresis_threshold(self):
        recorder = make_recorder(settings(DURATION=4.), max_lag=1., post_separation_time=1.)
        for index, distance in enumerate([.9, 1.005, 1.02, 1.1]):
            _observe_python(recorder, index + 1, np.array([[distance, 0.], [0., 0.]]), np.zeros(2),
                            [3., 2., 1., 0.][index], False)
        events = finish(recorder)["post_separation"]
        self.assertEqual(events[0, 0], 2)
        self.assertAlmostEqual(events[0, 3], .08)

    def test_wall_visit_and_group_filtering(self):
        recorder = make_recorder(settings(ARENA_DIAMETER=4., DURATION=4.),
                                 max_lag=1., post_separation_time=2.)
        frames = [(.9, 0.), (1.2, .1), (1.6, .2), (1.3, .2)]
        for index, (green, red) in enumerate(frames):
            _observe_python(recorder, index + 1, np.array([[green, 0.], [red, 0.]]),
                            np.zeros(2), [3., 2., 1., 0.][index], False)
        result = finish(recorder)
        event = result["post_separation"][0]
        self.assertEqual(event[8], 0)  # Starts away from the wall.
        self.assertEqual(event[9], 1)  # Reaches it later.
        self.assertAlmostEqual(event[3], .1)
        self.assertAlmostEqual(event[6], 0.)  # Green moved, but the pair gap did not grow.
        self.assertEqual(result["fixed_after_separation"][0, 8], 1)
        metrics, counts, _, _ = selected_measurements(result, "away", wall_free=True)
        self.assertEqual(metrics["post_separation_active_fraction"], 1.)
        self.assertTrue(np.isnan(metrics["post_separation_net_displacement"]))
        self.assertTrue(np.isnan(metrics["fixed_separation_net_displacement"]))

    def test_observer_preserves_dynamics_and_old_measurements(self):
        for tau in (0., .02):
            values = settings(DT=.001, DURATION=.1, STEADY_STATE_STEPS=3,
                              SPEED=20., ANGULAR_DIFFUSION=0., TRANSLATIONAL_DIFFUSION=0.,
                              REORIENT_ON_CONTACT=True, PERSISTENCE_TIME=tau)
            with patch.multiple(simulation, **values):
                old_recorder = make_recorder(values, max_lag=.01)
                reference = simulation._simulate_python(old_recorder)
                old_stats = finish(old_recorder)
                recorder = make_recorder(values, max_lag=.01, post_separation_time=.03)
                measured = simulation._simulate_python(recorder)
                for expected, actual in zip(reference, measured):
                    np.testing.assert_array_equal(actual, expected)
                result = finish(recorder)
                for field in ("events", "traces", "responses", "lags"):
                    np.testing.assert_allclose(result[field], old_stats[field], atol=1e-12)
                for name, value in old_stats["summary"].items():
                    np.testing.assert_allclose(result["summary"][name], value)
                if simulation._simulate_kernel is not None:
                    recorder = make_recorder(values, max_lag=.01, post_separation_time=.03)
                    compiled = simulation.simulate(recorder)
                    for expected, actual in zip(reference, compiled):
                        np.testing.assert_array_equal(actual, expected)
                    compiled_stats = finish(recorder)
                    for field in ("post_separation", "fixed_after_separation"):
                        np.testing.assert_allclose(compiled_stats[field], result[field], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
