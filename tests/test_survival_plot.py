"""A survival curve must not rise when short-follow-up runs drop out."""

import unittest

import numpy as np
from plot_contact_stats import ensemble_survival


def episodes(durations, censored):
    values = np.zeros((len(durations), 12))
    values[:, 3] = durations
    values[:, 5] = censored
    return values


class SurvivalPlotTests(unittest.TestCase):
    def test_fixed_episode_population_prevents_rising_tail(self):
        # The old per-time averaging changed from (0 + 1)/2 to 1 after t=1.
        runs = [episodes([1.], [0.]), episodes([5.], [1.])]
        times = np.array([0., 1., 2., 5.])
        mean, sem, followup = ensemble_survival(runs, times, bootstrap=40)
        np.testing.assert_allclose(mean, [1., .5, .5, .5])
        self.assertTrue((np.diff(mean) <= 0).all())
        np.testing.assert_array_equal(followup, [2, 2, 1, 1])
        self.assertTrue(np.isfinite(sem).all())

    def test_no_eligible_events_remain_undefined(self):
        left_censored = episodes([4.], [1.])
        left_censored[:, 4] = 1
        mean, sem, count = ensemble_survival([left_censored], np.array([0., 1.]))
        self.assertTrue(np.isnan(mean).all())
        self.assertTrue(np.isnan(sem).all())
        np.testing.assert_array_equal(count, 0)


if __name__ == "__main__":
    unittest.main()
