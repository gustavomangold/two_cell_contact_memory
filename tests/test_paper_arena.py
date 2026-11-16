"""The arena changes confinement, not the paper's particle update rules."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from demixing import snapshot_demixing
from paper_engine import (ROOT, DEFAULTS, source_hash, prepare_source, parameters,
                          build_engine, run_sample, domain_size)
from paper_arena import prepare_arena_source, arena_source_hash


PROBE = r'''
int main(int argc, char **argv) {
    SetParameters(argc, argv);
    AllocArrays();
    InitialiseInternalStates();
    for (int i=0; i<nParticles; i++) {
        particles[i].r.x = 3.9 + .2*(i/2);
        particles[i].r.y = 4.1 + .5*(i%2);
        particles[i].theta = .1*(i+1);
    }
    int mode = atoi(getenv("PROBE_MODE"));
    if (mode == 0) {
        for (int step=0; step<200; step++) {
            simulationTime = step * stepDuration;
            SingleStep(step);
        }
    } else {
        particles[0].r = (VecR){.1,4};
        particles[1].r = (VecR){4,1.4};
        particles[2].r = (VecR){7.9,4};
        particles[3].r = (VecR){4,6};
        if (mode == 1) particles[0].r = (VecR){7.6,4};
        if (mode == 1) particles[2].r = (VecR){2,4};
        BuildNeighbourList();
        #ifdef ARENA_PROBE
        if (mode == 2) ArenaParticleInteractions(); else ComputeInteractions();
        #else
        ComputeInteractions();
        #endif
        if (mode == 2) UpdateInternalStates();
        if (mode == 1) {
            /* The minimizer's wall energy must differentiate to its gradient. */
            gsl_vector *v=gsl_vector_alloc(2*nParticles), *g=gsl_vector_alloc(2*nParticles);
            for (int i=0;i<nParticles;i++) {
                gsl_vector_set(v,2*i,particles[i].r.x);
                gsl_vector_set(v,2*i+1,particles[i].r.y);
            }
            double params[]={1.};
            dfHarmonicPotential(v,params,g);
            for (int d=0;d<2*nParticles;d++) {
                double original=gsl_vector_get(v,d), h=1e-6;
                gsl_vector_set(v,d,original+h); double plus=fHarmonicPotential(v,params);
                gsl_vector_set(v,d,original-h); double minus=fHarmonicPotential(v,params);
                gsl_vector_set(v,d,original);
                if (fabs((plus-minus)/(2*h)-gsl_vector_get(g,d))>1e-5) return 3;
            }
            gsl_vector_free(v); gsl_vector_free(g);
        }
    }
    FILE *file=fopen("probe.bin","wb");
    for (int i=0;i<nParticles;i++) {
        double values[]={particles[i].r.x,particles[i].r.y,particles[i].theta,
                         particles[i].D,particles[i].decayTimer,particles[i].color,
                         particles[i].force.x,particles[i].force.y};
        fwrite(values,sizeof(double),8,file);
    }
    fwrite(rng.s,sizeof(uint64_t),4,file);
    fclose(file);
    return 0;
}
'''


class PaperArenaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="paper_arena_tests_")
        cls.root = Path(cls.directory.name)
        cls.original_hash = source_hash()
        cls.probes = {}
        for arena in [False, True]:
            folder = cls.root / ("arena" if arena else "periodic")
            folder.mkdir()
            source = folder / "source"
            (prepare_arena_source if arena else prepare_source)(source)
            driver = source / "driver.c"
            prelude = driver.read_text().split("int main(int argc, char **argv) {")[0]
            declarations = "void ArenaParticleInteractions(void);\n" if arena else ""
            driver.write_text(prelude + declarations + PROBE)
            command = ["cc", "-O3", "-std=gnu11"]
            if arena:
                command.append("-DARENA_PROBE")
            command += ["source/driver.c", "source/upstream/cell_simulation.c", "source/upstream/xoshiro_rng.c"]
            if arena:
                command.append("source/arena.c")
            command += ["-lgsl", "-lgslcblas", "-lm", "-o", "probe"]
            subprocess.run(command, cwd=folder, check=True, capture_output=True, text=True)
            cls.probes[arena] = folder

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def probe(self, arena, mode, tau=.001, wall=30000.):
        folder = self.probes[arena]
        # The same 8x8 bounding grid lets us compare the identical interior pairs.
        settings = {**DEFAULTS, "duration": .002, "packing_fraction": 4*(1. if arena else .785398)/64}
        params = parameters(4, 360000., tau, settings)
        (folder / "parameters").write_text("".join(f"{k} : {v:.17g}\n" for k,v in params.items()))
        subprocess.run([str(folder / "probe"), "parameters"], cwd=folder, check=True,
                       env={**os.environ, "PAPER_SEED":"7", "PROBE_MODE":str(mode), "PAPER_WALL_STIFFNESS":str(wall)},
                       capture_output=True, text=True)
        with (folder / "probe.bin").open("rb") as file:
            state = np.fromfile(file, dtype=np.float64, count=32).reshape(4,8)
            rng = np.fromfile(file, dtype=np.uint64)
        return state, rng

    def test_interior_particle_dynamics_and_rng_are_identical(self):
        for tau in [0., .001, .01]:
            with self.subTest(tau=tau):
                periodic, rng_periodic = self.probe(False, 0, tau)
                arena, rng_arena = self.probe(True, 0, tau)
                np.testing.assert_array_equal(arena, periodic)
                np.testing.assert_array_equal(rng_arena, rng_periodic)

    def test_wall_is_a_soft_spring_with_configurable_strength(self):
        state, _ = self.probe(True, 1)
        np.testing.assert_allclose(state[:,6:], [[-3000.,0],[0,0],[0,0],[0,0]], atol=1e-9)
        self.assertEqual(state[0,0],7.6)  # No projection onto the wall.
        weaker, _ = self.probe(True, 1, wall=10000.)
        np.testing.assert_allclose(weaker[0,6], -1000., atol=1e-9)

    def test_opposite_edges_are_not_neighbors_or_memory_triggers(self):
        periodic, _ = self.probe(False, 2)
        arena, _ = self.probe(True, 2)
        self.assertGreater(abs(periodic[0,6]), 0)
        self.assertEqual(periodic[0,5], 2)
        self.assertEqual(arena[0,6], 0)
        self.assertEqual(arena[0,5], 1)

    def test_packing_fraction_initialization_and_nonperiodic_demixing(self):
        settings = {**DEFAULTS, "boundary":"circular-arena", "wall_stiffness":30000.,
                    "duration":.002, "arena_source_hash":arena_source_hash()}
        engine = build_engine(boundary="circular-arena")
        for number in [2, 4, 32]:
            with self.subTest(N=number):
                path = self.root / f"sample_N{number}.npz"
                run_sample((number, 360000., .001, 1, 7, settings, engine, path))
                with np.load(path) as data:
                    length = float(data["L"])
                    self.assertEqual(length,domain_size(number,.4,"circular-arena"))
                    self.assertAlmostEqual(number/length**2,.4)
                    radius = np.linalg.norm(data["initial_positions"]-length/2,axis=1)
                    self.assertLessEqual(radius.max(),(length-1)/2+1e-4)
                    np.testing.assert_allclose(data["final_metrics"], snapshot_demixing(data["final_positions"]), equal_nan=True)
        self.assertEqual(source_hash(),self.original_hash)

    def test_constant_density_cli_routes_to_paper_arena_and_resumes(self):
        output = self.root / "sweep"
        command = [sys.executable, str(ROOT/"src/sweep_size_scaling.py"), "--scaling", "constant-density",
                   "--sizes", "4", "32", "--tau", ".001", "--duration", ".0002", "--samples", "2", "--workers", "2", "--output", str(output)]
        subprocess.run(command,check=True,capture_output=True,text=True)
        settings=json.loads((output/"settings.json").read_text())
        self.assertEqual(settings["boundary"],"circular-arena")
        self.assertEqual(settings["packing_fraction"],.4)
        self.assertEqual(settings["D_plus"],0.)
        self.assertEqual(settings["units"],"paper")
        paths=list(output.glob("N*/sample_*.npz"))
        self.assertEqual(len(paths),4)
        timestamps={p:p.stat().st_mtime_ns for p in paths}
        resumed=subprocess.run(command,check=True,capture_output=True,text=True)
        self.assertEqual(resumed.stdout.count("2/2 saved"),2)
        self.assertEqual(timestamps,{p:p.stat().st_mtime_ns for p in paths})
        figures=self.root/"figures"
        subprocess.run([sys.executable,str(ROOT/"src/plot_size_scaling.py"),"--data-dir",str(output),"--output",str(figures)],check=True,capture_output=True,text=True)
        self.assertTrue((figures/"paper_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
