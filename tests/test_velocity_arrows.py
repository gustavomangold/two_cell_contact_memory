import unittest

import numpy as np
from velocity_arrows import motion_velocities, arrow_components


class VelocityArrowTests(unittest.TestCase):
    def test_constant_velocity_with_irregular_frames(self):
        times = np.array([0., .1, .4, 1.])
        velocity = np.array([[1., 2.], [-2., 0.]])
        tracks = times[:, None, None] * velocity[None, :, :]
        np.testing.assert_allclose(motion_velocities(times, tracks),
                                   np.broadcast_to(velocity, tracks.shape))

    def test_directions_and_length_cap(self):
        velocity = np.array([[3., 4.], [-1., 2.], [0., 0.]])
        u, v = arrow_components(velocity)
        arrows = np.column_stack((u.filled(0), v.filled(0)))
        cross = arrows[:, 0] * velocity[:, 1] - arrows[:, 1] * velocity[:, 0]
        np.testing.assert_allclose(cross, 0., atol=1e-15)
        self.assertTrue((np.sum(arrows[:2] * velocity[:2], axis=1) > 0).all())
        self.assertTrue((np.linalg.norm(arrows, axis=1) <= .65 + 1e-12).all())
        self.assertTrue(u.mask[-1] and v.mask[-1])

    def test_single_frame_and_repeated_times(self):
        np.testing.assert_array_equal(motion_velocities([0], np.zeros((1, 2, 2))), 0)
        with self.assertRaises(ValueError):
            motion_velocities([0, 0], np.zeros((2, 2, 2)))


if __name__ == "__main__":
    unittest.main()
