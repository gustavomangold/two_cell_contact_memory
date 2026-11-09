"""Known trajectories check event timing; instrumented runs must preserve physics."""

import unittest
from unittest.mock import patch

import numpy as np
import two_disks as simulation
from contact_stats import make_recorder, _observe_python, finish
from plot_contact_stats import survival, mean_sem, response_curve


def settings(**overrides):
    values = {name: value for name, value in vars(simulation).items()
              if name.isupper() and isinstance(value, (int, float))}
    values.update(DT=1., DURATION=8., STEADY_STATE_STEPS=0,
                  ARENA_DIAMETER=4., PERSISTENCE_TIME=2.)
    values.update(overrides)
    return values


class ContactStatsTests(unittest.TestCase):
    def scripted(self, tau=2.):
        distances = [1.2, .9, 1.005, 1.02, 1.1, .95, .9, 1.2]
        memory = [0., 2., 1., 0., 0., 2., 2., 1.] if tau else [0.] * 8
        recorder = make_recorder(settings(PERSISTENCE_TIME=tau), max_lag=3., lag_points=3)
        for index, distance in enumerate(distances):
            positions = np.array([[distance, 0.], [0., 0.]])
            _observe_python(recorder, index + 1, positions, np.array([np.pi, 0.]),
                            memory[index], bool(tau and index in (1, 5, 6)))
        return finish(recorder), distances, memory

    def test_hysteresis_durations_censoring_and_occupancy(self):
        result, distances, memory = self.scripted()
        events = result["events"]
        np.testing.assert_array_equal(events[:, 0], [1, 0, 1, 0, 1])
        np.testing.assert_array_equal(events[:, 3], [1, 2, 2, 2, 1])
        np.testing.assert_array_equal(events[:, 4], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(events[:, 5], [0, 0, 0, 0, 1])
        np.testing.assert_array_equal(events[events[:, 0] == 1, 6], [0, 0, 1])
        summary = result["summary"]
        self.assertEqual(summary["contact_fraction"], 3 / 8)
        self.assertEqual(summary["hysteresis_contact_fraction"], 4 / 8)
        self.assertEqual(summary["persistent_fraction"], 5 / 8)
        self.assertEqual(summary["activations"], 3)
        self.assertEqual(summary["renewal_fraction"], 1 / 3)
        self.assertEqual(summary["contacts_per_time"], 2 / 8)
        self.assertEqual(summary["wall_fraction_green"], 0)
        self.assertAlmostEqual(summary["mean_wall_clearance_green"], 1.5 - np.mean(distances))
        np.testing.assert_array_equal(result["traces"][:, 7], memory)
        np.testing.assert_allclose(result["traces"][:, 5], np.pi)
        np.testing.assert_allclose(result["traces"][:, 11], 1.5 - np.array(distances))
        np.testing.assert_array_equal(result["traces"][:, 8], np.array(distances) < 1.)

    def test_activation_motion_uses_every_event_and_finite_followup(self):
        result, distances, _ = self.scripted()
        values, counts = response_curve(result["responses"], 0, "away")
        for lag_index, lag in enumerate(result["lags"].astype(int)):
            changes = [distances[start + lag] - distances[start]
                       for start in (1, 5, 6) if start + lag < len(distances)]
            self.assertEqual(counts[lag_index], len(changes))
            self.assertAlmostEqual(values[lag_index, 0], np.mean(changes))
            self.assertAlmostEqual(values[lag_index, 2], np.mean(changes))
            self.assertAlmostEqual(values[lag_index, 5], -1.)
        self.assertEqual(result["responses"][1:].sum(), 0.)

    def test_zero_tau_uses_new_contacts_without_fake_activations(self):
        result, _, _ = self.scripted(tau=0.)
        self.assertEqual(result["summary"]["activations"], 0)
        self.assertTrue(np.isnan(result["summary"]["renewal_fraction"]))
        self.assertEqual(result["responses"][0, 0, 0, 0], 2)

    def test_outward_normal_signs(self):
        recorder = make_recorder(settings(DURATION=3.), max_lag=1.)
        frames = [np.array([[1.2, 0.], [0., 0.]]), np.array([[.9, 0.], [0., 0.]]),
                  np.array([[1.1, 0.], [-.1, 0.]])]
        for index, positions in enumerate(frames):
            _observe_python(recorder, index + 1, positions, np.zeros(2), 1., index == 1)
        result = finish(recorder)
        values, counts = response_curve(result["responses"], 2, "away")
        np.testing.assert_allclose(values[1, :3], [.2, .1, .3])

    def test_warmup_crossing_is_left_censored(self):
        recorder = make_recorder(settings(STEADY_STATE_STEPS=2, DURATION=3.), max_lag=1.)
        for index, distance in enumerate([.9, .9, .9, 1.005, 1.1]):
            _observe_python(recorder, index + 1, np.array([[distance, 0.], [0., 0.]]), np.zeros(2), 0., False)
        result = finish(recorder)
        contact = result["events"][result["events"][:, 0] == 0]
        self.assertEqual(contact[0, 4], 1)
        self.assertTrue(np.isnan(survival(contact, np.array([0., 1., 2.]))).all())
        self.assertEqual(result["summary"]["contact_fraction"], 1 / 3)

    def test_wall_force_fraction_and_activation_location(self):
        for strength in (0., 30.):
            recorder = make_recorder(settings(DURATION=2., WALL_REPULSION=strength), max_lag=1.)
            _observe_python(recorder, 1, np.array([[1.7, 0.], [0., 0.]]), np.array([np.pi, 0.]), 0., False)
            _observe_python(recorder, 2, np.array([[1.6, 0.], [.7, 0.]]), np.array([np.pi, 0.]), 1., True)
            result = finish(recorder)
            self.assertEqual(result["summary"]["wall_fraction_green"], float(strength > 0))
            self.assertEqual(result["summary"]["wall_fraction_red"], 0.)
            self.assertEqual(result["responses"][0, 1, 0, 0], 1.)
            self.assertEqual(result["responses"][0, 0].sum(), 0.)

    def test_reorientation_records_the_propulsion_heading_after_turning(self):
        values = settings(DT=.001, DURATION=.06, SPEED=20., STEADY_STATE_STEPS=0,
                          REORIENT_ON_CONTACT=True, PERSISTENCE_TIME=.02,
                          ANGULAR_DIFFUSION=0., TRANSLATIONAL_DIFFUSION=0.)
        with patch.multiple(simulation, **values):
            reference = simulation.simulate()
            recorder = make_recorder(values, max_lag=.01)
            measured = simulation.simulate(recorder)
            for expected, actual in zip(reference, measured):
                np.testing.assert_array_equal(actual, expected)
        result = finish(recorder)
        self.assertGreater(result["responses"][2, :, 0, 0].sum(), 0)
        self.assertEqual(result["responses"][:2].sum(), 0.)

    def test_survival_uses_right_censoring(self):
        events = np.zeros((3, 12))
        events[:, 3] = [1., 2., 3.]
        events[:, 5] = [0., 1., 0.]
        np.testing.assert_allclose(survival(events, np.array([0., 1., 2., 3.])), [1, 2/3, 2/3, 0])
        self.assertTrue(np.isnan(survival(events, np.array([4.]))[0]))

    def test_uncertainty_is_across_seeds_and_handles_empty_bins(self):
        mean, sem, n = mean_sem([[1., np.nan], [3., np.nan]])
        self.assertEqual(mean[0], 2.)
        self.assertEqual(sem[0], 1.)
        self.assertTrue(np.isnan(mean[1]))
        self.assertEqual(n[1], 0)

    def test_recording_preserves_trajectory_and_matches_backends(self):
        for tau in (0., .001, .02):
            with self.subTest(tau=tau):
                values = settings(DT=.001, DURATION=.15, STEADY_STATE_STEPS=7,
                                  SPEED=20., PERSISTENCE_TIME=tau, SAVE_EVERY=2, STEP_SEPARATION=2)
                with patch.multiple(simulation, **values):
                    reference = simulation._simulate_python()
                    python_recorder = make_recorder(values, max_lag=.03)
                    measured = simulation._simulate_python(python_recorder)
                    for expected, actual in zip(reference, measured):
                        np.testing.assert_array_equal(actual, expected)
                    python_stats = finish(python_recorder)
                    if simulation._simulate_kernel is not None:
                        recorder = make_recorder(values, max_lag=.03)
                        compiled = simulation.simulate(recorder)
                        for expected, actual in zip(reference, compiled):
                            np.testing.assert_array_equal(actual, expected)
                        compiled_stats = finish(recorder)
                        for field in ("events", "traces", "responses", "lags"):
                            np.testing.assert_allclose(compiled_stats[field], python_stats[field], atol=1e-12)
                        for name in python_stats["summary"]:
                            np.testing.assert_allclose(compiled_stats["summary"][name], python_stats["summary"][name])


if __name__ == "__main__":
    unittest.main()
