"""Size sweeps: paper C dynamics at constant density, or legacy diameter scaling."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import two_disks as simulation
from many_disks import HAS_NUMBA, METRICS, initial_state, make_model, normalize, simulate_kernel
from sweep import tau_grid
from sweep_storage import window_edges


ROOT = Path(__file__).resolve().parents[1]
TAUS = [0., .0005, .001, .01, .1, .25, .5, 1., 2., 5.]


def arena_size(number, base_length=3., scaling="diameter"):
    factor = number / 2
    return base_length * (np.sqrt(factor) if scaling == "constant-density" else factor)


def run_sample(task):
    number, arena, tau, sample, seed, settings, path = task
    model = make_model(settings, arena, tau)
    rng = np.random.default_rng(seed)
    positions, angles = initial_state(number, arena, rng)
    edges = window_edges(model.steps)
    sums, _, _, _, final_positions, final_angles, final_memory = simulate_kernel(rng, positions, angles, model, edges)
    metrics = normalize(sums.sum(axis=0), number, model.steps)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, N=number, L=arena, tau=tau, sample=sample, seed=seed,
                            metrics=metrics, metric_names=np.array(METRICS), window_sums=sums,
                            window_edges=edges, initial_positions=positions, final_positions=final_positions,
                            final_angles=final_angles, final_memory=final_memory)
    temporary.replace(path)
    return sample


def check_settings(path, metadata):
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError(f"Changed settings in {path.parent}; choose another --output folder")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--base-L", type=float, help="Legacy N=2 diameter; for the paper arena sets phi=2/base-L²")
    parser.add_argument("--scaling", choices=["diameter", "constant-density"], default="diameter",
                        help="constant-density uses the paper C model in a circular arena")
    parser.add_argument("--legacy-model", action="store_true", help="Keep the old rescaled model for constant-density runs")
    parser.add_argument("--packing-fraction", type=float, help="Paper arena density (default: 0.4)")
    parser.add_argument("--D", type=float, nargs="+", help="Paper rotational diffusion (default: 360000)")
    parser.add_argument("--wall-stiffness", type=float, help="Paper arena wall stiffness (default: 30000)")
    parser.add_argument("--gsl-prefix", type=Path)
    parser.add_argument("--tau", type=float, nargs="+")
    parser.add_argument("--tau-range", type=float, nargs=3, action="append")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=simulation.SEED)
    parser.add_argument("--angular-diffusion", type=float, help="Legacy model only; the paper model uses --D")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--steady-state-steps", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.scaling == "constant-density" and not args.legacy_model:
        if args.angular_diffusion is not None or args.steady_state_steps not in (None, 0):
            parser.error("The paper arena uses --D in paper units and no discarded warm-up; use --legacy-model for old settings")
        if args.base_L is not None and args.packing_fraction is not None:
            parser.error("Choose either --base-L or --packing-fraction")
        if args.base_L is not None and (not np.isfinite(args.base_L) or args.base_L <= 1):
            parser.error("Use finite base-L>1")
        from paper_sweep import main as paper_sweep
        command = ["--boundary", "circular-arena", "--sizes", *map(str, args.sizes),
                   "--samples", str(args.samples), "--workers", str(args.workers), "--seed", str(args.seed)]
        for name, value in [("--packing-fraction", 2/args.base_L**2 if args.base_L is not None else args.packing_fraction),
                            ("--dt", args.dt), ("--duration", args.duration),
                            ("--wall-stiffness", args.wall_stiffness), ("--gsl-prefix", args.gsl_prefix), ("--output", args.output)]:
            if value is not None:
                command.extend([name, str(value)])
        for flag, values in [("--D", args.D), ("--tau", args.tau)]:
            if values is not None:
                command.extend([flag, *map(str, values)])
        for interval in args.tau_range or []:
            command.extend(["--tau-range", *map(str, interval)])
        print("Constant-density runs now use the paper C dynamics and PAPER time units; data: data/paper_arena/constant-density by default.", flush=True)
        return paper_sweep(command)
    if any(value is not None for value in [args.packing_fraction, args.D, args.wall_stiffness, args.gsl_prefix]):
        parser.error("Paper parameters require --scaling constant-density without --legacy-model")
    args.base_L = 3. if args.base_L is None else args.base_L
    args.angular_diffusion = simulation.ANGULAR_DIFFUSION if args.angular_diffusion is None else args.angular_diffusion
    args.dt = simulation.DT if args.dt is None else args.dt
    args.duration = simulation.DURATION if args.duration is None else args.duration
    args.steady_state_steps = simulation.STEADY_STATE_STEPS if args.steady_state_steps is None else args.steady_state_steps
    if (any(number < 2 or number % 2 for number in args.sizes)
            or not all(np.isfinite(value) for value in [args.base_L, args.angular_diffusion, args.dt, args.duration])
            or args.base_L <= 1 or min(args.dt, args.duration) <= 0 or args.angular_diffusion < 0
            or min(args.samples, args.workers) < 1 or min(args.seed, args.steady_state_steps) < 0
            or round(args.duration / args.dt) < 1):
        parser.error("Require even N>=2, base-L>1, positive times/counts and nonnegative diffusion/seed/warm-up")
    try:
        taus = tau_grid(args.tau, args.tau_range) if args.tau or args.tau_range else TAUS
    except ValueError as error:
        parser.error(str(error))
    root = args.output or ROOT / "data" / "size_scaling" / args.scaling
    root.mkdir(parents=True, exist_ok=True)
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    settings.update(REORIENT_ON_CONTACT=False, DT=args.dt, DURATION=args.duration,
                    STEADY_STATE_STEPS=args.steady_state_steps, SEED=args.seed,
                    ANGULAR_DIFFUSION=args.angular_diffusion)
    fixed = {name: value for name, value in settings.items() if name not in ("ARENA_DIAMETER", "PERSISTENCE_TIME")}
    try:
        check_settings(root / "settings.json", dict(physics=fixed, scaling=args.scaling, base_L=args.base_L, schema=1))
        print(f"Backend: {'Numba' if HAS_NUMBA else 'NumPy'}; equal populations, no reorientation", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, args.samples), mp_context=get_context("spawn")) as pool:
            for number in sorted(set(args.sizes)):
                arena = arena_size(number, args.base_L, args.scaling)
                print(f"N={number}, L={arena:g}, nominal packing fraction={number / arena ** 2:.5g}", flush=True)
                for tau in taus:
                    folder = root / f"N{number}_L{arena:.12g}_tau{tau:.12g}"
                    folder.mkdir(parents=True, exist_ok=True)
                    current = {**settings, "ARENA_DIAMETER": arena, "PERSISTENCE_TIME": tau}
                    metadata = dict(N=number, green=number // 2, red=number // 2, L=arena,
                                    tau=tau, packing_fraction=number / arena ** 2, settings=current,
                                    initialization="original_pair" if number == 2 else "uniform_nonoverlapping_shuffled", schema=1)
                    check_settings(folder / "settings.json", metadata)
                    jobs = []
                    for sample in range(1, args.samples + 1):
                        path = folder / f"sample_{sample:05d}.npz"
                        seed = args.seed + sample - 1
                        if path.exists():
                            with np.load(path) as data:
                                if (int(data["seed"]) != seed or int(data["sample"]) != sample
                                        or int(data["N"]) != number or float(data["L"]) != arena or float(data["tau"]) != tau):
                                    raise ValueError(f"Incompatible sample file: {path}")
                        else:
                            jobs.append(pool.submit(run_sample, (number, arena, tau, sample, seed, current, path)))
                    print(f"N={number}, tau={tau:g}: {args.samples - len(jobs)}/{args.samples} saved", flush=True)
                    for job in as_completed(jobs):
                        print(f"  Saved sample {job.result()}", flush=True)
                    temporary = folder / "summary.tmp"
                    with temporary.open("w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(["sample", "seed", *METRICS])
                        for path in sorted(folder.glob("sample_*.npz")):
                            with np.load(path) as data:
                                writer.writerow([int(data["sample"]), int(data["seed"]), *data["metrics"]])
                    temporary.replace(folder / "summary.csv")
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved compact samples and CSVs in {root}. Plot separately with src/plot_size_scaling.py --data-dir {root}")


if __name__ == "__main__":
    main()
