"""Shared paper C engine: builds, observations, measurements and saved-run checks.

Particle dynamics are in native/paper/upstream/cell_simulation.c. This module
keeps their current behavior; the audit findings are deliberately not fixed here.
"""

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE = Path(__file__).with_name("native")
SOURCE = NATIVE / "paper"
REFERENCE = json.loads((NATIVE / "compatibility.json").read_text())
BOUNDARIES = {"arena": "circular-arena", "pbc": "periodic-square"}
DEFAULTS = dict(dt=1e-5, duration=5., packing_fraction=.4, Pe=1000., k=1000.,
                D_plus=0., seed=7, units="paper", boundary="periodic-square",
                implementation="upstream-c", revision="fa2989f7841ea4c8aea0d5691ed0710ee9432da4", schema=2)
METRICS = ("contact_fraction", "red_contact_fraction", "persistent_fraction")
FINAL_METRICS = ("demixing_index", "demixing_index_green", "demixing_index_red",
                 "neighbor_coverage", "neighbor_coverage_green", "neighbor_coverage_red")
LOG_WINDOWS = 64
DEFAULT_DIFFUSION = 360000.

# Measurements: the same definitions for sweeps, time series and plots.

def snapshot_demixing(positions, box_size=None):
    """Return raw M and non-isolated fractions for all, green and red disks."""
    positions = np.asarray(positions, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 2
            or len(positions) % 2 or not np.isfinite(positions).all()):
        raise ValueError("Expected finite (N, 2) positions with even N>=2")
    number = len(positions)
    green = np.arange(number) < number // 2
    if box_size is not None and (not np.isfinite(box_size) or box_size <= 0):
        raise ValueError("box_size must be finite and positive")
    count, same = np.zeros(number, dtype=int), np.zeros(number, dtype=int)
    # Bound memory use for the paper's 4,000 particles.
    for start in range(0, number, 256):
        stop = min(start + 256, number)
        displacement = positions[start:stop, None, :] - positions[None, :, :]
        if box_size is not None:
            displacement -= box_size * np.floor(displacement / box_size + .5)
        neighbors = np.sum(displacement ** 2, axis=-1) < 1.1 ** 2
        neighbors[np.arange(stop - start), np.arange(start, stop)] = False
        count[start:stop] = neighbors.sum(axis=1)
        same[start:stop] = (neighbors & (green[start:stop, None] == green[None, :])).sum(axis=1)
    index = np.divide(same, count, out=np.full(number, np.nan), where=count > 0)
    groups = (np.ones(number, dtype=bool), green, ~green)
    means, coverage = [], []
    for group in groups:
        eligible = group & (count > 0)
        means.append(float(index[eligible].mean()) if eligible.any() else np.nan)
        coverage.append(float((count[group] > 0).mean()))
    return np.array(means + coverage)


def window_edges(count):
    if count < 1:
        raise ValueError("Cannot average an empty run")
    return np.unique(np.r_[0, np.ceil(np.geomspace(1, count, LOG_WINDOWS)).astype(np.int64)])


def mean_sem(values):
    """Equal weight for independent seeds, not for correlated events within a run."""
    values = np.asarray(values)
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
    squares = np.where(valid, (values - mean) ** 2, 0).sum(axis=0)
    sem = np.sqrt(np.divide(squares, n * (n - 1), out=np.full(n.shape, np.nan), where=n > 1))
    return mean, sem, n


# Native builds: the author sources stay unchanged on disk.

def source_hash():
    digest = hashlib.sha256()
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(SOURCE).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def domain_size(number, packing_fraction, boundary="periodic-square"):
    return float(np.sqrt(number * (1. if boundary == "circular-arena" else .785398) / packing_fraction))


def neighbor_method(number, packing_fraction, boundary="periodic-square"):
    bins = int(domain_size(number, packing_fraction, boundary) / 2)
    return "unique-pairs-small-box" if bins < 3 else ("arena-linked-cells" if boundary == "circular-arena" else "upstream-linked-cells")


def engine_environment(seed, settings):
    env = {**os.environ, "PAPER_SEED": str(seed)}
    if settings["boundary"] == "circular-arena":
        env["PAPER_WALL_STIFFNESS"] = str(settings["wall_stiffness"])
    return env


def prepare_source(destination):
    """Prepare a build copy; tracked upstream files are never edited."""
    shutil.copytree(SOURCE, destination)
    config = destination / "upstream/agent_simulation_config.h"
    text = config.read_text()
    old = "#define SEED time(NULL);"
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream seed definition")
    config.write_text(text.replace(old, '#define SEED strtoull(getenv("PAPER_SEED"), NULL, 10);'))
    code = destination / "upstream/cell_simulation.c"
    text = code.read_text()
    old = "void BuildNeighbourList(){"
    if text.count(old) != 1:
        raise ValueError("Unexpected upstream neighbor-list definition")
    # Calls go through driver.c, which dispatches by box size. The original
    # function body is unchanged and still used whenever bins>=3.
    code.write_text("void BuildNeighbourList(void);\n" + text.replace(old, "void UpstreamBuildNeighbourList(){"))


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Unexpected upstream arena patch: {old}")
    return text.replace(old, new)


def prepare_arena_source(destination):
    prepare_source(destination)
    shutil.copy2(NATIVE / "arena.c", destination / "arena.c")
    header = destination / "upstream/base_types2D.h"
    text = header.read_text()
    end = text.rfind("#endif")
    header.write_text(text[:end] + "\n#undef VWrapAllTang\n#define VWrapAllTang(v) ((void)0)\n"
                      "#undef VWrapAll\n#define VWrapAll(v) ((void)0)\n" + text[end:])
    code = destination / "upstream/cell_simulation.c"
    text = code.read_text()
    for old, new in [
        ("void SetParameters(", "void ArenaBaseSetParameters("),
        ("void ComputeInteractions(){", "void ArenaParticleInteractions(){"),
        ("double fHarmonicPotential(", "double ArenaParticlePotential("),
        ("void dfHarmonicPotential(", "void ArenaParticleGradient("),
        ("sqrt(nParticles*0.785398/areaFraction)", "sqrt(nParticles/areaFraction)"),
        ("        particles[particleIdx].r.x = region.x*uniform1;\n        particles[particleIdx].r.y = region.y*uniform2;",
         "        particles[particleIdx].r.x = region.x/2 + (region.x/2-.5)*sqrt(uniform1)*cos(2*M_PI*uniform2);\n"
         "        particles[particleIdx].r.y = region.y/2 + (region.y/2-.5)*sqrt(uniform1)*sin(2*M_PI*uniform2);"),
        ("VCellWrapAll(cellIdx2);", "if (cellIdx2.x < 0 || cellIdx2.x >= cells.x || cellIdx2.y < 0 || cellIdx2.y >= cells.y) continue;"),
        ("VDiv(cellIdx,particles[particleIdx].r,cellWidth);",
         "VDiv(cellIdx,particles[particleIdx].r,cellWidth);\n"
         "        cellIdx.x = MAX(0, MIN(cells.x-1, cellIdx.x));\n"
         "        cellIdx.y = MAX(0, MIN(cells.y-1, cellIdx.y));"),
    ]:
        text = replace_once(text, old, new)
    # Wall forces act during the position update. There is no wrap or projection.
    text, count = re.subn(r"void EnforcePeriodicBoundaries\(\)\{.*?\n\}",
                         "void EnforcePeriodicBoundaries(){}", text, flags=re.DOTALL)
    if count != 1:
        raise ValueError("Unexpected upstream periodic-boundary function")
    code.write_text(text)


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


def tree_hash(folder):
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(folder).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare_native(destination, boundary, frames=False):
    """Prepare the same C programs as before the cleanup, including observations."""
    (prepare_arena_source if boundary == "circular-arena" else prepare_source)(destination)
    if frames:
        driver = destination / "driver.c"
        marker = "int main(int argc, char **argv) {"
        driver.write_text(driver.read_text().split(marker)[0] + (NATIVE / "movie.c").read_text())
        code = destination / "upstream/cell_simulation.c"
        marker = "void EnforcePeriodicBoundaries(){"
        code.write_text("void MovieObserveBeforeWrap(void);\n" + replace_once(
            code.read_text(), marker, marker + "\n    MovieObserveBeforeWrap();"))
    return tree_hash(destination)


def prepared_hash(boundary, frames=False):
    with tempfile.TemporaryDirectory(prefix="paper_sources_") as directory:
        return prepare_native(Path(directory) / "source", boundary, frames)


def measurement_hash():
    text = "".join(inspect.getsource(f) for f in [parameters, snapshot_demixing, window_edges])
    return hashlib.sha256(text.encode()).hexdigest()


def model_hash(boundary):
    return hashlib.sha256((prepared_hash(boundary) + measurement_hash()).encode()).hexdigest()


def make_settings(boundary, dt, duration, packing_fraction, seed, wall_stiffness=30000., D_plus=0., time_points=0):
    settings = {**DEFAULTS, "boundary": boundary, "dt": dt, "duration": duration,
                "packing_fraction": packing_fraction, "seed": seed, "D_plus": D_plus,
                "source_hash": source_hash(), "engine_hash": model_hash(boundary)}
    if boundary == "circular-arena":
        settings.update(wall_stiffness=wall_stiffness,
                        arena_source_hash=hashlib.sha256((NATIVE / "arena.c").read_bytes()).hexdigest())
    if time_points:
        settings.update(recorder="demixing-time-series-v1", time_points=time_points,
                        recorder_hash=hashlib.sha256((prepared_hash(boundary, True) + measurement_hash()).encode()).hexdigest())
    return settings


def check_settings(path, requested, write=True):
    """Resume old data only when parameters and generated C match the reference."""
    if path.exists():
        existing = json.loads(path.read_text())
        if existing == requested:
            return existing
        boundary = requested.get("boundary")
        if existing.get("schema") == 1 and requested.get("engine_hash") == REFERENCE["engine_hash"].get(boundary):
            historical = {k:v for k,v in requested.items() if k != "engine_hash"}
            historical["schema"] = 1
            if boundary == "circular-arena":
                historical["arena_source_hash"] = REFERENCE["legacy_arena_hash"]
            if "recorder" in requested:
                if boundary != "periodic-square" or prepared_hash(boundary, True) != REFERENCE["prepared"][boundary + "_frames"]:
                    raise ValueError(f"Changed recording code in {path.parent}; choose another --output")
                historical["recorder_hash"] = REFERENCE["legacy_recorder_hash"]
            if existing == historical:
                return existing
        raise ValueError(f"Changed or incompatible settings in {path.parent}; choose another --output")
    if not write:
        return requested
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(requested, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return requested


def data_directory(boundary, time_series=False):
    folder = ROOT / "data" / ("paper_arena/constant-density" if boundary == "circular-arena" else "exactly-like-paper")
    return folder / "time_series" if time_series else folder


def figure_directory(boundary, time_series=False):
    folder = ROOT / "figures" / ("paper_arena/constant-density" if boundary == "circular-arena" else "exactly-like-paper")
    return folder / "time_series" if time_series else folder


def build_engine(boundary, frames=False, gsl_prefix=None):
    """Build one cached executable; compilation never imports plotting packages."""
    compiler = shlex.split(os.environ.get("CC", "cc"))
    flags = []
    if gsl_prefix:
        prefix = Path(gsl_prefix).expanduser().resolve()
        flags = [f"-I{prefix/'include'}", f"-L{prefix/'lib'}", f"-Wl,-rpath,{prefix/'lib'}"]
    with tempfile.TemporaryDirectory(prefix="paper_compile_") as directory:
        work = Path(directory)
        fingerprint = prepare_native(work / "source", boundary, frames)
        key = hashlib.sha256(json.dumps([fingerprint, compiler, flags]).encode()).hexdigest()[:20]
        executable = ROOT / ".paper-build" / key / "engine"
        if executable.is_file():
            return executable
        command = [*compiler, "-O3", "-std=gnu11", *flags, "source/driver.c",
                   "source/upstream/cell_simulation.c", "source/upstream/xoshiro_rng.c",
                   *(["source/arena.c"] if boundary == "circular-arena" else []),
                   "-lgsl", "-lgslcblas", "-lm", "-o", "engine"]
        try:
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
        except FileNotFoundError as error:
            raise RuntimeError("A C compiler and GSL are required; see README.md") from error
        if result.returncode:
            raise RuntimeError(f"C build failed; check compiler/GSL.\n{result.stderr}")
        executable.parent.mkdir(parents=True, exist_ok=True)
        temporary = executable.parent / f".engine_{os.getpid()}"
        shutil.copy2(work / "engine", temporary)
        temporary.replace(executable)
    return executable


def frame_steps(settings, start, save_every, max_frames):
    total = round(settings["duration"] / settings["dt"]) + 1
    first = round(start / settings["dt"])
    regular = (total-first)//save_every + 1
    count = regular + (first+(regular-1)*save_every != total)
    indices = np.linspace(0, count-1, min(count, max_frames), dtype=int)
    return np.minimum(first + indices*save_every, total)


def record_motion(number, diffusion, tau, seed, settings, start, save_every, max_frames, log, gsl_prefix=None, executable=None):
    """Observe positions and true unwrapped motion without extra noise draws."""
    params = parameters(number, diffusion, tau, settings)
    steps = frame_steps(settings, start, save_every, max_frames)
    executable = executable or build_engine(settings["boundary"], True, gsl_prefix)
    with tempfile.TemporaryDirectory(prefix="paper_frames_") as directory:
        work = Path(directory)
        (work / "parameters").write_text("".join(f"{k} : {v:.17g}\n" for k,v in params.items()))
        (work / "frames.txt").write_text(f"{len(steps)}\n" + " ".join(map(str, steps)) + "\n")
        with Path(log).open("w") as stream:
            result = subprocess.run([str(executable), "parameters"], cwd=work,
                                    env=engine_environment(seed, settings), stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Simulation failed; see {log}")
        values = np.fromfile(work / "movie.bin").reshape(len(steps), 1 + 7*number)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Nonfinite recorded state; see {log}")
        records = values[:, 1:].reshape(len(steps), number, 7)
        return dict(times=values[:, 0], tracks=records[:, :, :2], angles=records[:, :, 2],
                    memory=records[:, :, 3], colors=records[:, :, 4].astype(int),
                    unwrapped=records[:, :, 5:7], rng_state=np.fromfile(work / "rng.bin", dtype=np.uint64),
                    L=domain_size(number, settings["packing_fraction"], settings["boundary"]))


def run_sample(task):
    """One independent seed; keep only compact measurements and endpoint states."""
    number, diffusion, tau, sample, seed, settings, executable, path, fingerprint = task
    path = Path(path)
    if "time_points" in settings:
        movie = record_motion(number, diffusion, tau, seed, settings, 0., 1,
                              settings["time_points"], path.with_suffix(".log"), executable=executable)
        box = movie["L"] if settings["boundary"] == "periodic-square" else None
        values = np.array([snapshot_demixing(p, box_size=box) for p in movie["tracks"]])
        payload = dict(N=number, L=movie["L"], D=diffusion, tau=tau, sample=sample, seed=seed,
                       times=movie["times"], metric_names=np.array(FINAL_METRICS), demixing=values,
                       final_rng_state=movie["rng_state"])
    else:
        params = parameters(number, diffusion, tau, settings)
        total = params["stepLimit"] + 1
        edges = window_edges(total)
        with tempfile.TemporaryDirectory(prefix="paper_sample_") as directory:
            work = Path(directory)
            (work / "parameters").write_text("".join(f"{k} : {v:.17g}\n" for k,v in params.items()))
            (work / "edges.txt").write_text(f"{len(edges)-1}\n" + " ".join(map(str, edges)) + "\n")
            with path.with_suffix(".log").open("w") as stream:
                result = subprocess.run([str(executable), "parameters"], cwd=work,
                                        env=engine_environment(seed, settings), stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Simulation failed; see {path.with_suffix('.log')}")
            with (work / "result.bin").open("rb") as file:
                initial = np.fromfile(file, dtype=np.float64, count=3*number).reshape(number, 3)
                final = np.fromfile(file, dtype=np.float64, count=5*number).reshape(number, 5)
                sums = np.fromfile(file, dtype=np.float64, count=3*(len(edges)-1)).reshape(-1, 3)
                rng = np.fromfile(file, dtype=np.uint64, count=4)
                if file.read(1) or len(rng) != 4:
                    raise RuntimeError("Unexpected engine output")
            length = domain_size(number, settings["packing_fraction"], settings["boundary"])
            box = length if settings["boundary"] == "periodic-square" else None
            payload = dict(N=number, L=length, D=diffusion, tau=tau, sample=sample, seed=seed,
                           metric_names=np.array(METRICS), metrics=sums.sum(axis=0)/(total*(number//2)),
                           window_sums=sums, window_edges=edges, initial_positions=initial[:, :2],
                           initial_angles=initial[:, 2], final_positions=final[:, :2], final_angles=final[:, 2],
                           final_memory=final[:number//2, 3], final_colors=final[:, 4].astype(int), final_rng_state=rng,
                           final_metric_names=np.array(FINAL_METRICS), final_metrics=snapshot_demixing(final[:, :2], box),
                           final_time=params["stepLimit"]*settings["dt"], elapsed_time=total*settings["dt"],
                           source_hash=settings["source_hash"],
                           neighbor_method=neighbor_method(number, settings["packing_fraction"], settings["boundary"]),
                           parameters_text=(work / "parameters").read_text())
    payload.update(settings_json=json.dumps(settings, sort_keys=True), engine_hash=fingerprint,
                   build_id=Path(executable).parent.name)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **payload)
    temporary.replace(path)
    return sample


