"""Build and run the paper's original C/GSL engine; all times are paper units."""

import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

import numpy as np
from demixing import FINAL_METRICS, snapshot_demixing
from sweep_storage import window_edges


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent / "paper"
REVISION = "fa2989f7841ea4c8aea0d5691ed0710ee9432da4"
METRICS = ("contact_fraction", "red_contact_fraction", "persistent_fraction")
DEFAULTS = dict(dt=1e-5, duration=5., packing_fraction=.4, Pe=1000., k=1000.,
                D_plus=0., seed=7, units="paper", boundary="periodic-square",
                implementation="upstream-c", revision=REVISION, schema=1)


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(SOURCE).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_engine(gsl_prefix=None):
    """Compile once. Only the clock-based seed is replaced in a build copy."""
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = []
    if gsl_prefix:
        prefix = Path(gsl_prefix).expanduser().resolve()
        flags = [f"-I{prefix / 'include'}", f"-L{prefix / 'lib'}", f"-Wl,-rpath,{prefix / 'lib'}"]
    identity = hashlib.sha256(json.dumps([source_hash(), compiler, flags]).encode()).hexdigest()[:20]
    cache = ROOT / ".paper-build"
    executable = cache / identity / "paper_engine"
    if executable.is_file():
        return executable
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compile_", dir=cache) as directory:
        build = Path(directory)
        shutil.copytree(SOURCE, build / "source")
        config = build / "source/upstream/agent_simulation_config.h"
        text = config.read_text()
        old = "#define SEED time(NULL);"
        if text.count(old) != 1:
            raise ValueError("Unexpected upstream seed definition")
        config.write_text(text.replace(old, '#define SEED strtoull(getenv("PAPER_SEED"), NULL, 10);'))
        command = [*compiler, "-O3", "-std=gnu11", *flags, "source/driver.c",
                   "source/upstream/cell_simulation.c", "source/upstream/xoshiro_rng.c",
                   "-lgsl", "-lgslcblas", "-lm", "-o", "paper_engine"]
        try:
            result = subprocess.run(command, cwd=build, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError("A C compiler and GSL development files are required; see docs/paper.md") from error
        if result.returncode:
            raise RuntimeError(f"Paper engine compilation failed. A C compiler and GSL are required.\n{result.stderr}")
        executable.parent.mkdir(exist_ok=True)
        shutil.copy2(build / "paper_engine", build / "ready")
        (build / "ready").replace(executable)
    return executable


def parameters(number, diffusion, tau, settings):
    """The original engine takes sqrt(D), not the diffusion coefficient D."""
    steps = round(settings["duration"] / settings["dt"])
    interval = math.gcd(steps, max(1, round(.25 / settings["dt"]))) * settings["dt"]
    return dict(stepLimit=steps,
                stepDuration=settings["dt"], skipSteps=0,
                measurementInterval=interval, nGreenParticles=number // 2,
                nRedParticles=number // 2, areaFraction=settings["packing_fraction"],
                redD=np.sqrt(diffusion), greenD=np.sqrt(diffusion),
                greenPersistentD=np.sqrt(settings["D_plus"]), k=settings["k"], tau=tau,
                Pe=settings["Pe"], potentialRange=1., LennardJones=0, turnAround=0.,
                redRedAdhesionMult=1., greenGreenAdhesionMutl=1., redGreenAdhesionMult=1.)


def run_sample(task):
    number, diffusion, tau, sample, seed, settings, executable, path = task
    path = Path(path)
    params = parameters(number, diffusion, tau, settings)
    total = params["stepLimit"] + 1  # Original inclusive loop.
    edges = window_edges(total)
    with tempfile.TemporaryDirectory(prefix="paper_sample_") as directory:
        work = Path(directory)
        (work / "parameters").write_text("".join(f"{name} : {value:.17g}\n" for name, value in params.items()))
        (work / "edges.txt").write_text(f"{len(edges) - 1}\n" + " ".join(map(str, edges)) + "\n")
        log = path.with_suffix(".log")
        with log.open("w") as stream:
            result = subprocess.run([str(executable), "parameters"], cwd=work,
                                    env={**os.environ, "PAPER_SEED": str(seed)}, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Paper engine failed for sample {sample}; see {log}")
        with (work / "result.bin").open("rb") as file:
            initial = np.fromfile(file, dtype=np.float64, count=3 * number).reshape(number, 3)
            final = np.fromfile(file, dtype=np.float64, count=5 * number).reshape(number, 5)
            sums = np.fromfile(file, dtype=np.float64, count=3 * (len(edges) - 1)).reshape(-1, 3)
            rng_state = np.fromfile(file, dtype=np.uint64, count=4)
            if file.read(1) or len(rng_state) != 4:
                raise RuntimeError("Unexpected paper-engine output")
        # Preserve the original engine's rounded pi/4 constant in box sizing.
        length = np.sqrt(number * .785398 / settings["packing_fraction"])
        metrics = sums.sum(axis=0) / (total * (number // 2))
        final_metrics = snapshot_demixing(final[:, :2], box_size=length)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, N=number, L=length, D=diffusion, tau=tau, sample=sample, seed=seed,
                                metric_names=np.array(METRICS), metrics=metrics, window_sums=sums,
                                window_edges=edges, initial_positions=initial[:, :2], initial_angles=initial[:, 2],
                                final_positions=final[:, :2], final_angles=final[:, 2], final_memory=final[:number//2, 3],
                                final_colors=final[:, 4].astype(int), final_rng_state=rng_state,
                                final_metric_names=np.array(FINAL_METRICS), final_metrics=final_metrics,
                                final_time=params["stepLimit"] * settings["dt"], elapsed_time=total * settings["dt"],
                                settings_json=json.dumps(settings, sort_keys=True), source_hash=source_hash(),
                                parameters_text=(work / "parameters").read_text())
        temporary.replace(path)
    return sample
