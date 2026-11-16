"""One regression suite for the consolidated engine and three entry points."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import engine


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="clean_simulation_tests_")
        cls.work = Path(cls.directory.name)
        # The original implementation is the oracle, kept in the cleanup archive.
        with tarfile.open(engine.ROOT/"archive/code_before_cleanup_2026-11-16.tar.gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.startswith("src/"):
                    destination = cls.work/"original"/member.name
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(archive.extractfile(member).read())
        script = r'''
from pathlib import Path
import json,sys
from paper_engine import DEFAULTS,build_engine,run_sample,source_hash,domain_size,neighbor_method
from paper_arena import arena_source_hash
from paper_timeseries import record_sample,recorder_hash
root=Path(sys.argv[1])
for boundary in ['periodic-square','circular-arena']:
    settings={**DEFAULTS,'boundary':boundary,'duration':.002,'source_hash':source_hash()}
    if boundary=='circular-arena':settings.update(wall_stiffness=30000.,arena_source_hash=arena_source_hash())
    folder=root/boundary;folder.mkdir(parents=True)
    (folder/'settings.json').write_text(json.dumps(settings))
    executable=build_engine(boundary=boundary)
    for number in [4,32]:
        for tau in [0.,.001,.01]:
            point=folder/f'N{number}_phi0.4_D360000_tau{tau:.12g}'
            point.mkdir()
            metadata=dict(N=number,L=domain_size(number,.4,boundary),D=360000.,tau=tau,settings=settings,
                          neighbor_method=neighbor_method(number,.4,boundary))
            (point/'settings.json').write_text(json.dumps(metadata))
            run_sample((number,360000.,tau,1,7,settings,executable,point/'sample_00001.npz'))
    if boundary=='periodic-square':
        settings={**settings,'recorder':'demixing-time-series-v1','recorder_hash':recorder_hash(),'time_points':5}
        folder=root/'old_time_series';folder.mkdir()
        (folder/'settings.json').write_text(json.dumps(settings))
        point=folder/'N32_phi0.4_D360000_tau0.001';point.mkdir()
        (point/'settings.json').write_text(json.dumps(dict(N=32,D=360000.,tau=.001,settings=settings)))
        record_sample((32,360000.,.001,1,7,settings,point/'sample_00001.npz',None))
'''
        oracle = cls.work/"oracle.py"
        oracle.write_text(script)
        result = subprocess.run([sys.executable,str(oracle),str(cls.work/"reference")],capture_output=True,text=True,
                                env={**os.environ,"PYTHONPATH":str(cls.work/"original/src")})
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def cli(self, name, *arguments):
        result = subprocess.run([sys.executable,str(engine.ROOT/"src"/name),*map(str,arguments)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return result.stdout

    def test_archive_and_four_active_python_files(self):
        self.assertEqual({p.name for p in (engine.ROOT/"src").glob("*.py")},{"engine.py","run.py","plot.py","make_gif.py"})
        with tarfile.open(engine.ROOT/"archive/code_before_cleanup_2026-11-16.tar.gz") as archive:
            manifest = json.loads(archive.extractfile("SHA256.json").read())
            for name,digest in manifest.items():
                self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(),digest)

    def test_generated_c_and_measurements_match_original(self):
        self.assertEqual(engine.measurement_hash(),engine.REFERENCE["measurement_hash"])
        for boundary in engine.BOUNDARIES.values():
            self.assertEqual(engine.model_hash(boundary),engine.REFERENCE["engine_hash"][boundary])
            for frames in [False,True]:
                key = boundary+("_frames" if frames else "")
                self.assertEqual(engine.prepared_hash(boundary,frames),engine.REFERENCE["prepared"][key])

    def test_all_endpoint_states_rng_and_measurements_are_bitwise_identical(self):
        names = ["initial_positions","initial_angles","final_positions","final_angles","final_memory","final_colors",
                 "final_rng_state","metrics","window_sums","window_edges","final_metrics","final_time","elapsed_time"]
        for boundary in engine.BOUNDARIES.values():
            executable = engine.build_engine(boundary)
            settings = engine.make_settings(boundary,1e-5,.002,.4,7)
            for number in [4,32]:
                for tau in [0.,.001,.01]:
                    with self.subTest(boundary=boundary,N=number,tau=tau):
                        reference = self.work/"reference"/boundary/f"N{number}_phi0.4_D360000_tau{tau:.12g}/sample_00001.npz"
                        path = self.work/f"new_{boundary}_{number}_{tau}.npz"
                        engine.run_sample((number,360000.,tau,1,7,settings,executable,path,settings["engine_hash"]))
                        with np.load(reference) as old,np.load(path) as new:
                            for name in names:
                                np.testing.assert_array_equal(new[name],old[name],err_msg=name)

    def test_resumes_both_old_sweeps_and_old_time_series(self):
        for short,boundary in engine.BOUNDARIES.items():
            folder = self.work/"reference"/boundary
            paths = list(folder.glob("N*/sample_*.npz"))
            before = {p:p.stat().st_mtime_ns for p in paths}
            output = self.cli("run.py","--boundary",short,"--sizes",4,32,"--tau",0,.001,.01,
                              "--duration",.002,"--samples",1,"--output",folder,"--dry-run")
            self.assertIn("0 missing samples",output)
            self.assertEqual(before,{p:p.stat().st_mtime_ns for p in paths})
        output = self.cli("run.py","--boundary","pbc","--sizes",32,"--tau",.001,"--duration",.002,
                          "--samples",1,"--time-points",5,"--output",self.work/"reference/old_time_series","--dry-run")
        self.assertIn("0 missing samples",output)

    def test_recording_does_not_change_particle_dynamics(self):
        for boundary in engine.BOUNDARIES.values():
            settings = engine.make_settings(boundary,1e-5,.002,.4,7)
            movie = engine.record_motion(32,360000.,.001,7,settings,0.,5,100,self.work/f"{boundary}.log")
            reference = self.work/"reference"/boundary/"N32_phi0.4_D360000_tau0.001/sample_00001.npz"
            with np.load(reference) as data:
                for key,field in [("tracks","final_positions"),("angles","final_angles"),("colors","final_colors")]:
                    np.testing.assert_array_equal(movie[key][-1],data[field])
                np.testing.assert_array_equal(movie["rng_state"],data["final_rng_state"])
                np.testing.assert_array_equal(movie["memory"][-1,:16],data["final_memory"])
            self.assertTrue((np.diff(movie["times"])>0).all())
            expected = movie["unwrapped"]%movie["L"] if boundary=="periodic-square" else movie["unwrapped"]
            np.testing.assert_allclose(expected,movie["tracks"],atol=1e-10)

    def test_new_runner_resumes_and_rejects_changed_physics(self):
        folder = self.work/"new_sweep"
        arguments = ["--boundary","arena","--sizes",4,32,"--tau",0,.001,.01,
                     "--samples",2,"--workers",2,"--duration",.0002,"--output",folder]
        self.cli("run.py",*arguments)
        paths = list(folder.glob("N*/sample_*.npz"))
        self.assertEqual(len(paths),12)
        before = {p:p.stat().st_mtime_ns for p in paths}
        self.cli("run.py",*arguments)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in paths})
        wrong = subprocess.run([sys.executable,str(engine.ROOT/"src/run.py"),*map(str,arguments),"--dt","5e-6"],capture_output=True,text=True)
        self.assertNotEqual(wrong.returncode,0)
        self.assertIn("incompatible settings",wrong.stderr)
        figures = self.work/"sweep_figures"
        self.cli("plot.py","--data-dir",folder,"--output",figures)
        self.assertTrue((figures/"demixing_index_vs_tau.png").is_file())
        with (figures/"plot_means.csv").open() as file: rows=list(csv.DictReader(file))
        self.assertEqual(len(rows),6)

    def test_one_runner_records_time_series_for_both_boundaries(self):
        for short in engine.BOUNDARIES:
            folder = self.work/f"series_{short}"
            self.cli("run.py","--boundary",short,"--sizes",32,"--tau",.001,.01,1,
                     "--time-points",5,"--samples",2,"--workers",2,"--duration",.0002,"--output",folder)
            self.assertEqual(len(list(folder.glob("N*/sample_*.npz"))),6)
            figures = self.work/f"time_figures_{short}"
            self.cli("plot.py","--data-dir",folder,"--output",figures)
            self.assertTrue((figures/"demixing_index_time_N32_D360000.png").is_file())
            with (figures/"plot_means.csv").open() as file: rows=list(csv.DictReader(file))
            self.assertEqual(len(rows),15)

    def test_one_gif_script_renders_both_boundaries(self):
        from PIL import Image
        for short in engine.BOUNDARIES:
            path = self.work/f"{short}.gif"
            self.cli("make_gif.py","--boundary",short,"--N",4,"--tau",.001,"--start",0,
                     "--duration",.002,"--save-every",5,"--max-frames",5,"--output",path)
            with Image.open(path) as image:self.assertEqual(image.n_frames,5)
            self.assertTrue(path.with_suffix(".png").is_file())

    def test_simulation_imports_need_no_matplotlib_pillow_or_numba(self):
        code = '''import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'matplotlib','PIL','numba'}:raise ImportError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import engine,run
'''
        result = subprocess.run([sys.executable,"-c",code],capture_output=True,text=True,
                                env={**os.environ,"PYTHONPATH":str(engine.ROOT/"src")})
        self.assertEqual(result.returncode,0,result.stderr)

    def test_legacy_reader_uses_final_demixing_not_old_time_average(self):
        import plot
        root = self.work/"legacy_plot_data"
        folder = root/"N4_L10_tau1"
        folder.mkdir(parents=True)
        (folder/"settings.json").write_text(json.dumps(dict(N=4,L=10.,tau=1.,packing_fraction=.04)))
        positions = np.array([[0.,0.],[.5,0.],[3.,0.],[3.5,0.]])
        np.savez(folder/"sample_00001.npz",metric_names=np.array(["demixing_index"]),
                 metrics=np.array([.2]),final_positions=positions)
        rows = plot.load_sweep(root,"demixing_index",None,None)
        self.assertEqual(rows[0]["mean"],1.)


if __name__ == "__main__":
    unittest.main()
