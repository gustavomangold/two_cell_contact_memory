"""Paper Eq. (4): cell weighting, type labels, isolated cells and cutoff."""

import unittest

import numpy as np

from demixing import snapshot_demixing


class DemixingTests(unittest.TestCase):
    def test_separated_same_type_clusters(self):
        positions = [[0, 0], [.5, 0], [5, 0], [5.5, 0]]
        np.testing.assert_array_equal(snapshot_demixing(positions), np.ones(6))

    def test_alternating_chain_uses_mean_of_cell_ratios_not_ratio_of_totals(self):
        # In spatial order: G,G,R,G,R,R. Degrees: 1,2,2,2,2,1.
        positions = np.array([[0, 0], [1, 0], [3, 0], [2, 0], [4, 0], [5, 0]])
        result = snapshot_demixing(positions)
        np.testing.assert_allclose(result, [.5, .5, .5, 1, 1, 1])
        self.assertNotAlmostEqual(result[0], 4/10)  # Edge-weighted ratio differs.

    def test_isolated_cells_excluded_and_types_kept_in_original_order(self):
        # The isolated green occurs before the non-isolated green.
        positions = [[10, 0], [0, 0], [.9, 0], [1.8, 0]]
        np.testing.assert_allclose(snapshot_demixing(positions), [.5, 0, .75, .75, .5, 1])

    def test_undefined_population_does_not_become_perfectly_demixed(self):
        result = snapshot_demixing([[10, 0], [20, 0], [0, 0], [.5, 0]])
        self.assertEqual(result[0], 1)
        self.assertTrue(np.isnan(result[1]))
        np.testing.assert_array_equal(result[2:], [1, .5, 0, 1])
        result = snapshot_demixing([[0, 0], [5, 0]])
        self.assertTrue(np.isnan(result[:3]).all())
        np.testing.assert_array_equal(result[3:], 0)

    def test_cutoff_is_strict_and_reference_cell_is_excluded(self):
        result = snapshot_demixing([[0, 0], [1.1, 0]])
        self.assertTrue(np.isnan(result[:3]).all())
        np.testing.assert_array_equal(snapshot_demixing([[0, 0], [1.099, 0]]), [0, 0, 0, 1, 1, 1])

    def test_invalid_snapshot_rejected(self):
        for positions in [[], [[0, 0]], [[0, 0], [np.nan, 0]], [[0], [1]]]:
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                snapshot_demixing(positions)


if __name__ == "__main__":
    unittest.main()
