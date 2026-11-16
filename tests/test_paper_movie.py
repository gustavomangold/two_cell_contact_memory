"""GIF recording must preserve paper dynamics and periodic displacement."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from paper_engine import DEFAULTS, build_engine, run_sample, source_hash
from paper_movie import build_movie_engine, record_movie


class PaperMovieTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="paper_movie_test_")
        cls.root = Path(cls.directory.name)
        cls.original_hash = source_hash()
        try:
            cls.engine = build_engine()
            build_movie_engine()
        except RuntimeError as error:
            cls.directory.cleanup()
            raise unittest.SkipTest(str(error))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_recording_preserves_all_final_states_and_rng(self):
        settings = {**DEFAULTS,"duration":.002}
        for number in [4,32]:
            with self.subTest(N=number):
                work = self.root / f"N{number}"
                work.mkdir()
                path = work / "sample_00001.npz"
                run_sample((number,360000.,.001,1,7,settings,self.engine,path))
                movie = record_movie(number,360000.,.001,7,settings,0.,5,100,work/"movie.log")
                with np.load(path) as data:
                    np.testing.assert_array_equal(movie["tracks"][0],data["initial_positions"])
                    np.testing.assert_array_equal(movie["angles"][0],data["initial_angles"])
                    np.testing.assert_array_equal(movie["tracks"][-1],data["final_positions"])
                    np.testing.assert_array_equal(movie["angles"][-1],data["final_angles"])
                    np.testing.assert_array_equal(movie["colors"][-1],data["final_colors"])
                    np.testing.assert_array_equal(movie["memory"][-1,:number//2],data["final_memory"])
                    np.testing.assert_array_equal(movie["rng_state"],data["final_rng_state"])
                self.assertTrue((np.diff(movie["times"])>0).all())
                self.assertEqual(movie["times"][-1],201 * settings["dt"])
        self.assertEqual(source_hash(),self.original_hash)

    def test_velocity_recording_handles_multiple_periodic_crossings(self):
        settings = {**DEFAULTS,"duration":.02}
        movie = record_movie(4,0.,.001,7,settings,0.,5,2,self.root/"crossings.log")
        self.assertEqual(len(movie["times"]),2)
        np.testing.assert_allclose(movie["unwrapped"] % movie["L"],movie["tracks"],atol=1e-10)
        displacement = movie["unwrapped"][-1] - movie["unwrapped"][0]
        self.assertTrue((np.abs(displacement)>movie["L"]).any())
        self.assertEqual(source_hash(),self.original_hash)


if __name__ == "__main__":
    unittest.main()
