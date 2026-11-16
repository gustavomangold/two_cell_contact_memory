"""Record the paper engine for visualization without changing sweep source hashes."""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
from paper_engine import ROOT, source_hash, prepare_source, parameters


MOVIE_SOURCE = Path(__file__).with_suffix(".c")


def build_movie_engine(gsl_prefix=None):
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = []
    if gsl_prefix:
        prefix = Path(gsl_prefix).expanduser().resolve()
        flags = [f"-I{prefix/'include'}", f"-L{prefix/'lib'}", f"-Wl,-rpath,{prefix/'lib'}"]
    identity = hashlib.sha256(json.dumps([source_hash(), MOVIE_SOURCE.read_text(),
        Path(__file__).read_text(), compiler, flags]).encode()).hexdigest()[:20]
    cache = ROOT / ".paper-build"
    executable = cache / f"movie_{identity}" / "paper_movie"
    if executable.exists():
        return executable
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="movie_build_", dir=cache) as directory:
        build = Path(directory)
        prepare_source(build / "source")
        driver = build / "source/driver.c"
        marker = "int main(int argc, char **argv) {"
        if driver.read_text().count(marker) != 1:
            raise ValueError("Unexpected paper driver layout")
        driver.write_text(driver.read_text().split(marker)[0] + MOVIE_SOURCE.read_text())
        code = build / "source/upstream/cell_simulation.c"
        marker = "void EnforcePeriodicBoundaries(){"
        if code.read_text().count(marker) != 1:
            raise ValueError("Unexpected upstream periodic wrap")
        # Observe the unwrapped displacement without altering positions or RNG.
        code.write_text("void MovieObserveBeforeWrap(void);\n" + code.read_text().replace(
            marker, marker + "\n    MovieObserveBeforeWrap();"))
        command = [*compiler, "-O3", "-std=gnu11", *flags, "source/driver.c",
                   "source/upstream/cell_simulation.c", "source/upstream/xoshiro_rng.c",
                   "-lgsl", "-lgslcblas", "-lm", "-o", "paper_movie"]
        try:
            result = subprocess.run(command, cwd=build, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError("A C compiler and GSL are required, as for paper_sweep.py") from error
        if result.returncode:
            raise RuntimeError(f"GIF engine compilation failed; check compiler/GSL.\n{result.stderr}")
        executable.parent.mkdir(exist_ok=True)
        (build / "paper_movie").replace(executable)
    return executable


def record_movie(number, diffusion, tau, seed, settings, start, save_every, max_frames, log, gsl_prefix=None):
    params = parameters(number, diffusion, tau, settings)
    total = params["stepLimit"] + 1
    first = round(start / settings["dt"])
    regular = (total-first)//save_every + 1
    count = regular + (first+(regular-1)*save_every != total)
    indices = np.linspace(0,count-1,min(count,max_frames),dtype=int)
    steps = np.minimum(first+indices*save_every,total)
    executable = build_movie_engine(gsl_prefix)
    with tempfile.TemporaryDirectory(prefix="paper_movie_") as directory:
        work = Path(directory)
        (work / "parameters").write_text("".join(f"{k} : {v:.17g}\n" for k,v in params.items()))
        (work / "frames.txt").write_text(f"{len(steps)}\n" + " ".join(map(str, steps)) + "\n")
        with Path(log).open("w") as stream:
            result = subprocess.run([str(executable), "parameters"], cwd=work,
                                    env={**os.environ, "PAPER_SEED": str(seed)}, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Paper simulation failed; see {log}")
        values = np.fromfile(work / "movie.bin").reshape(len(steps), 1 + 7 * number)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite recorded state; see {log}")
        records = values[:, 1:].reshape(len(steps), number, 7)
        return dict(times=values[:, 0], tracks=records[:, :, :2], angles=records[:, :, 2],
                    memory=records[:, :, 3], colors=records[:, :, 4].astype(int),
                    unwrapped=records[:, :, 5:7], rng_state=np.fromfile(work / "rng.bin", dtype=np.uint64),
                    L=float(np.sqrt(number * .785398 / settings["packing_fraction"])))
