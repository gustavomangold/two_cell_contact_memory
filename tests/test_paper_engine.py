"""Compare the compact driver against the authors' original complete executable."""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from demixing import snapshot_demixing
from paper_engine import SOURCE, DEFAULTS, build_engine, parameters, run_sample


class PaperDefinitionTests(unittest.TestCase):
    def test_vendored_source_is_unmodified(self):
        provenance = json.loads((SOURCE / "provenance.json").read_text())
        for name, digest in provenance["files"].items():
            self.assertEqual(hashlib.sha256((SOURCE / "upstream" / name).read_bytes()).hexdigest(), digest)

    def test_paper_parameters_and_noise_amplitude(self):
        params = parameters(4000, 360000., .002, DEFAULTS)
        self.assertEqual(params["greenD"], 600.)
        self.assertEqual(params["redD"], 600.)
        self.assertEqual(params["greenPersistentD"], 0.)
        self.assertEqual(params["stepLimit"], 500000)
        self.assertEqual(params["stepDuration"], 1e-5)
        self.assertEqual(params["Pe"], 1000.)
        self.assertEqual(params["turnAround"], 0.)
        self.assertEqual(params["potentialRange"], 1.)

    def test_demixing_uses_periodic_neighbors(self):
        positions = np.array([[.1, 0], [5, 5], [9.9, 0], [7, 7]])
        periodic = snapshot_demixing(positions, box_size=10.)
        np.testing.assert_allclose(periodic, [0, 0, 0, .5, .5, .5])
        self.assertTrue(np.isnan(snapshot_demixing(positions)[:3]).all())


@unittest.skipUnless(shutil.which("cc"), "C compiler required for upstream comparison")
class OriginalEngineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="paper_reference_")
        cls.root = Path(cls.directory.name)
        try:
            cls.engine = build_engine()
        except RuntimeError as error:
            cls.directory.cleanup()
            raise unittest.SkipTest(str(error))
        reference = cls.root / "reference"
        shutil.copytree(SOURCE / "upstream", reference)
        config = reference / "agent_simulation_config.h"
        config.write_text(config.read_text().replace("#define SEED time(NULL);",
                          '#define SEED strtoull(getenv("PAPER_SEED"), NULL, 10);'))
        # Add read-only full-precision state dumps to upstream main, no dynamics changes.
        dump = '''{
            FILE *dump=fopen("NAME.bin","wb");
            for(int i=0;i<nParticles;i++) {
                double v[]={particles[i].r.x,particles[i].r.y,particles[i].theta,
                            particles[i].decayTimer,particles[i].color};
                fwrite(v,sizeof(double),COUNT,dump);
            }
            EXTRA
            fclose(dump);
        }'''
        main = reference / "main.c"
        text = main.read_text()
        text = text.replace("SetUpJob();", "SetUpJob();\n" + dump.replace("NAME", "initial").replace("COUNT", "3").replace("EXTRA", ""))
        text = text.replace("cleanup();", dump.replace("NAME", "final").replace("COUNT", "5")
                            .replace("EXTRA", "fwrite(rng.s,sizeof(uint64_t),4,dump);") + "\ncleanup();")
        main.write_text(text)
        cls.reference = reference / "original"
        subprocess.run(["cc", "-O3", "-std=gnu11", "main.c", "cell_simulation.c", "xoshiro_rng.c",
                        "-lgsl", "-lgslcblas", "-lm", "-o", str(cls.reference)],
                       cwd=reference, check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_positions_angles_memory_colors_and_rng_are_bitwise_identical(self):
        settings = {**DEFAULTS, "duration": .002}
        for number, tau in [(32, 0.), (32, .00003), (64, .001), (128, .002)]:
            with self.subTest(N=number, tau=tau):
                work = self.root / f"N{number}_tau{tau}"
                work.mkdir()
                params = parameters(number, 360000., tau, settings)
                (work / "parameters").write_text("".join(f"{k} : {v:.17g}\n" for k,v in params.items()))
                subprocess.run([str(self.reference), "parameters"], cwd=work,
                               env={**os.environ, "PAPER_SEED": "7"}, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
                path = work / "sample_00001.npz"
                run_sample((number, 360000., tau, 1, 7, settings, self.engine, path))
                with np.load(path) as data:
                    initial = np.fromfile(work / "initial.bin").reshape(number, 3)
                    with (work / "final.bin").open("rb") as file:
                        final = np.fromfile(file, count=5 * number).reshape(number, 5)
                        rng_state = np.fromfile(file, dtype=np.uint64, count=4)
                    np.testing.assert_array_equal(data["initial_positions"], initial[:, :2])
                    np.testing.assert_array_equal(data["initial_angles"], initial[:, 2])
                    np.testing.assert_array_equal(data["final_positions"], final[:, :2])
                    np.testing.assert_array_equal(data["final_angles"], final[:, 2])
                    np.testing.assert_array_equal(data["final_memory"], final[:number//2, 3])
                    np.testing.assert_array_equal(data["final_colors"], final[:, 4])
                    np.testing.assert_array_equal(data["final_rng_state"], rng_state)
                    np.testing.assert_array_equal(data["window_edges"][-1], params["stepLimit"] + 1)
                    self.assertAlmostEqual(float(data["final_time"]), .002)
                    self.assertAlmostEqual(float(data["elapsed_time"]), .00201)


if __name__ == "__main__":
    unittest.main()
