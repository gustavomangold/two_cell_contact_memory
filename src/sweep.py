"""Run a resumable (L, contact-memory time) sweep; plotting is separate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from decimal import Decimal
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import two_disks as simulation
from sweep_storage import HEADER, compact_sample, completed_samples


ROOT = Path(__file__).resolve().parents[1]


def tau_grid(values, ranges):
    """Combine explicit values and inclusive ranges without accumulated roundoff."""
    points = set(values or [])
    for start, stop, step in ranges or []:
        start, stop, step = map(lambda value: Decimal(str(value)), (start, stop, step))
        if not all(value.is_finite() for value in (start, stop, step)):
            raise ValueError("Tau ranges must be finite")
        if start < 0 or stop < start or step <= 0:
            raise ValueError("Tau ranges require 0 <= start <= stop and step > 0")
        points.update(float(start + index * step)
                      for index in range(int((stop - start) // step) + 1))
    if not points:
        points.update(tau_grid(None, [(0, 5, .25)]))
    if any(not np.isfinite(value) or value < 0 for value in points):
        raise ValueError("Tau values must be finite and nonnegative")
    return sorted(points)


def run_sample(task):
    length, tau, sample, settings = task
    for name, value in settings.items():
        setattr(simulation, name, value)
    simulation.ARENA_DIAMETER = length
    simulation.PERSISTENCE_TIME = tau
    simulation.SEED = settings["SEED"] + sample - 1
    distances = simulation.simulate()[3]
    stride = simulation.STEP_SEPARATION
    first = ((max(1, simulation.STEADY_STATE_STEPS) + stride - 1) // stride) * stride
    rows = compact_sample(distances, length, first, stride, simulation.DT)
    return sample, simulation.SEED, rows


def run_sweep(lengths, taus, samples, workers, output, settings):
    output.mkdir(parents=True, exist_ok=True)
    # L, tau and number of samples may be extended on a later invocation.
    fixed = {key: value for key, value in settings.items()
             if key not in ("ARENA_DIAMETER", "PERSISTENCE_TIME")}
    manifest = output / "settings.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fixed:
        raise ValueError(f"Settings changed in {output}; choose another --output folder")
    stride, warmup = settings["STEP_SEPARATION"], settings["STEADY_STATE_STEPS"]
    first = ((max(1, warmup) + stride - 1) // stride) * stride
    last = round(settings["DURATION"] / settings["DT"]) + warmup
    if first > last:
        raise ValueError("No separation measurements; increase --duration or reduce --record-every")

    # Check existing CSVs before adopting a folder without metadata.
    pending = []
    for length in lengths:
        for tau in taus:
            path = output / f"L{length:g}_tau{tau:g}.csv"
            done = completed_samples(path, length, tau, settings["SEED"], first,
                                     stride, last, samples, settings["DT"])
            pending.append((length, tau, path, done))
    manifest.write_text(json.dumps(fixed, indent=2, sort_keys=True) + "\n")

    with ProcessPoolExecutor(max_workers=min(workers, samples),
                             mp_context=get_context("spawn")) as pool:
        for length, tau, path, done in pending:
            if done == samples:
                print(f"Skipping L={length:g}, tau={tau:g}: {done} samples saved", flush=True)
                continue
            print(f"L={length:g}, tau={tau:g}: {done}/{samples} saved", flush=True)
            tasks = ((length, tau, sample, settings) for sample in range(done + 1, samples + 1))
            new_file = not path.exists() or path.stat().st_size == 0
            # Only the parent writes. Flush each complete sample for resuming.
            with path.open("a", newline="") as file:
                writer = csv.writer(file)
                if new_file:
                    writer.writerow(HEADER)
                for sample, seed, rows in pool.map(run_sample, tasks):
                    writer.writerows([[length, tau, sample, seed, *row] for row in rows])
                    file.flush()
                    print(f"  Saved sample {sample}/{samples}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, nargs="+", default=[3.])
    parser.add_argument("--tau", type=float, nargs="+", help="Explicit tau values")
    parser.add_argument("--tau-range", type=float, nargs=3, action="append",
                        metavar=("START", "STOP", "STEP"),
                        help="Inclusive grid; repeat to combine ranges (default: 0 5 0.25)")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=simulation.SEED,
                        help="First seed; subsequent samples use seed+1, seed+2, ...")
    parser.add_argument("--reorient", action="store_true")
    parser.add_argument("--dt", type=float, default=simulation.DT)
    parser.add_argument("--duration", type=float, default=simulation.DURATION)
    parser.add_argument("--steady-state-steps", type=int, default=simulation.STEADY_STATE_STEPS)
    parser.add_argument("--record-every", type=int, default=simulation.STEP_SEPARATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (any(not np.isfinite(length) or length <= 1 for length in args.L)
            or args.samples < 1 or args.workers < 1 or args.seed < 0
            or not np.isfinite(args.dt) or args.dt <= 0
            or not np.isfinite(args.duration) or args.duration <= 0
            or args.steady_state_steps < 0 or args.record_every < 1):
        parser.error("Require L>1, positive dt/duration/samples/workers/record-every and nonnegative seed/warm-up")
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    settings.update(DT=args.dt, DURATION=args.duration, STEADY_STATE_STEPS=args.steady_state_steps,
                    STEP_SEPARATION=args.record_every, SEED=args.seed,
                    REORIENT_ON_CONTACT=args.reorient)
    output = args.output or ROOT / "data" / ("reorient" if args.reorient else "no_reorient")
    try:
        taus = tau_grid(args.tau, args.tau_range)
        print(f"Backend: {'Numba' if simulation._simulate_kernel else 'NumPy'}", flush=True)
        run_sweep(sorted(set(args.L)), taus, args.samples, args.workers, output, settings)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved CSVs in {output}. Plot separately with src/plot.py.")


if __name__ == "__main__":
    main()
