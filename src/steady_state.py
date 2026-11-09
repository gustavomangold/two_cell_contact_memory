"""Transient ensemble diagnostic; does not change two_disks.py defaults.

Run: .venv/bin/python src/steady_state.py --L 4 --tau 1
Default: 50 runs, 10 million steps, measurement every 1000 steps, no burn-in.
Each CSV contains distances after the recorded step, including the initial state.
Rerunning resumes completed samples with the same settings.
Plot separately with src/plot.py --steady-state OUTPUT_FOLDER.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import two_disks as simulation


def run_sample(task):
    sample, settings = task
    for name, value in settings.items():
        setattr(simulation, name, value)
    times, tracks, _, _ = simulation.simulate()
    distances = np.linalg.norm(tracks[:, 0] - tracks[:, 1], axis=1)
    return sample, times, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=4.)
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--measure-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--dt", type=float, default=simulation.DT)
    parser.add_argument("--seed", type=int, default=simulation.SEED)
    parser.add_argument("--reorient", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not np.isfinite(args.L) or args.L <= 1 or not np.isfinite(args.tau)
            or args.tau < 0 or args.samples < 2 or args.steps < 1
            or not np.isfinite(args.dt) or args.dt <= 0 or args.seed < 0
            or not 1 <= args.measure_every <= args.steps or args.workers < 1):
        parser.error("Require L>1, tau>=0, samples>=2, steps>=measure-every>=1, workers>=1")

    folder = args.output or (Path(__file__).resolve().parents[1] / "data" /
                            f"steady_state_L{args.L:g}_tau{args.tau:g}_n{args.samples}_steps{args.steps}")
    folder.mkdir(parents=True, exist_ok=True)
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    settings.update(ARENA_DIAMETER=args.L, PERSISTENCE_TIME=args.tau, DT=args.dt,
                    REORIENT_ON_CONTACT=args.reorient,
                    DURATION=args.steps * args.dt, STEADY_STATE_STEPS=0,
                    SAVE_EVERY=args.measure_every, STEP_SEPARATION=args.steps + 1)
    seeds = [args.seed + i for i in range(args.samples)]
    settings["SEED"] = args.seed
    metadata = dict(settings=settings, samples=args.samples, steps=args.steps, seeds=seeds)
    manifest = folder / "settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Existing results have different settings; choose another --output folder")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    steps = np.arange(0, args.steps + 1, args.measure_every, dtype=np.int64)
    if steps[-1] != args.steps:
        steps = np.append(steps, args.steps)
    times = steps * args.dt
    traces = np.empty((args.samples, len(steps)))
    pending = []
    for i, seed in enumerate(seeds):
        path = folder / f"sample_{i + 1:02d}.csv"
        if path.exists():
            saved = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            if (saved.shape != (len(steps), 3) or not np.isfinite(saved).all()
                    or not np.array_equal(saved[:, 0], steps)
                    or not np.array_equal(saved[:, 1], times)):
                raise ValueError(f"Incomplete or incompatible sample file: {path}")
            traces[i] = saved[:, 2]
        else:
            pending.append((i, {**settings, "SEED": seed}))

    print(f"L={args.L:g}, tau={args.tau:g}: {args.samples} runs × {args.steps:,} steps, "
          f"{len(steps):,} measurements/run; backend={'Numba' if simulation._simulate_kernel else 'NumPy'}", flush=True)
    completed = args.samples - len(pending)
    if pending:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(pending)),
                                 mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(run_sample, task) for task in pending]
            for future in as_completed(futures):
                i, measured_times, distances = future.result()
                np.testing.assert_array_equal(measured_times, times)
                traces[i] = distances
                path = folder / f"sample_{i + 1:02d}.csv"
                temporary = path.with_suffix(".tmp")
                np.savetxt(temporary, np.column_stack((steps, times, distances)),
                           delimiter=",", header="step,time,distance", comments="",
                           fmt=["%d", "%.17g", "%.17g"])
                temporary.replace(path)
                completed += 1
                print(f"Saved {completed}/{args.samples} samples (seed={seeds[i]})", flush=True)

    mean = traces.mean(axis=0)
    sem = traces.std(axis=0, ddof=1) / np.sqrt(args.samples)
    np.savetxt(folder / "mean_distance.csv", np.column_stack((steps, times, mean, sem)),
               delimiter=",", header="step,time,mean_distance,sem", comments="",
               fmt=["%d", "%.17g", "%.17g", "%.17g"])
    width = max(args.measure_every, args.steps // 20)
    blocks = []
    for start in range(0, args.steps, width):
        end = min(start + width, args.steps)
        mask = (steps >= start) & ((steps <= end) if end == args.steps else (steps < end))
        values = traces[:, mask].mean(axis=1)
        blocks.append([(start + end) / 2, values.mean(), values.std(ddof=1) / np.sqrt(args.samples)])
    blocks = np.array(blocks)
    np.savetxt(folder / "time_windows.csv", blocks, delimiter=",",
               header="center_step,mean_distance,sem", comments="")
    print(f"Saved time series in {folder}. Plot separately with src/plot.py --steady-state {folder}")


if __name__ == "__main__":
    main()
