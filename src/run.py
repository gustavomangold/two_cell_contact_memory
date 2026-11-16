"""Run parallel, resumable paper-model sweeps. All times are paper units."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from decimal import Decimal
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import engine


TAUS = [0., 1e-5, 1e-4, 2.5e-4, 5e-4, .001, .002, .005, .01]


def tau_values(values, ranges, time_series):
    points = set(values or [])
    for interval in ranges or []:
        start, stop, step = map(lambda v: Decimal(str(v)), interval)
        if not all(v.is_finite() for v in (start, stop, step)) or start < 0 or stop < start or step <= 0:
            raise ValueError("Tau ranges require finite 0<=start<=stop and step>0")
        points.update(float(start + i*step) for i in range(int((stop-start)//step) + 1))
    if not points:
        points.update([.001, .01, 1.] if time_series else TAUS)
    if any(not np.isfinite(v) or v < 0 for v in points):
        raise ValueError("Tau values must be finite and nonnegative")
    return sorted(points)


def validate_sample(path, number, diffusion, tau, sample, seed, settings, fingerprint):
    with np.load(path) as data:
        if (int(data["N"]) != number or float(data["D"]) != diffusion or float(data["tau"]) != tau
                or int(data["sample"]) != sample or int(data["seed"]) != seed
                or float(data["L"]) != engine.domain_size(number, settings["packing_fraction"], settings["boundary"])
                or str(data["settings_json"]) != json.dumps(settings, sort_keys=True)
                or ("engine_hash" in data and str(data["engine_hash"]) != fingerprint)):
            raise ValueError(f"Incompatible saved sample: {path}")
        if "time_points" in settings:
            times = engine.frame_steps(settings, 0., 1, settings["time_points"]) * settings["dt"]
            if (tuple(data["metric_names"]) != engine.FINAL_METRICS or not np.array_equal(data["times"], times)
                    or data["demixing"].shape != (len(times), len(engine.FINAL_METRICS))):
                raise ValueError(f"Incompatible time-series recording: {path}")
        elif (tuple(data["metric_names"]) != engine.METRICS
              or tuple(data["final_metric_names"]) != engine.FINAL_METRICS
              or str(data["source_hash"]) != settings["source_hash"]
              or str(data["neighbor_method"]) != engine.neighbor_method(number, settings["packing_fraction"], settings["boundary"])):
            raise ValueError(f"Incompatible sweep recording: {path}")


def save_summary(folder, time_series):
    temporary = folder / "summary.tmp"
    names = engine.FINAL_METRICS if time_series else (*engine.METRICS, *engine.FINAL_METRICS)
    with temporary.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample", "seed", *names])
        for path in sorted(folder.glob("sample_*.npz")):
            with np.load(path) as data:
                values = data["demixing"][-1] if time_series else np.r_[data["metrics"], data["final_metrics"]]
                writer.writerow([int(data["sample"]), int(data["seed"]), *values])
    temporary.replace(folder / "summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=engine.BOUNDARIES, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    parser.add_argument("--tau", type=float, nargs="+")
    parser.add_argument("--tau-range", type=float, nargs=3, action="append", metavar=("START", "STOP", "STEP"))
    parser.add_argument("--D", type=float, nargs="+", default=[engine.DEFAULT_DIFFUSION])
    parser.add_argument("--D-plus", type=float, default=engine.DEFAULTS["D_plus"])
    parser.add_argument("--packing-fraction", type=float, default=engine.DEFAULTS["packing_fraction"])
    parser.add_argument("--wall-stiffness", type=float, default=30000.)
    parser.add_argument("--dt", type=float, default=engine.DEFAULTS["dt"])
    parser.add_argument("--duration", type=float, default=engine.DEFAULTS["duration"])
    parser.add_argument("--time-points", type=int, default=0, help="Record this many demixing snapshots; 0 saves sweep statistics only")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=engine.DEFAULTS["seed"])
    parser.add_argument("--gsl-prefix", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check settings and list missing samples without writing or simulating")
    args = parser.parse_args()
    if (any(n < 2 or n % 2 for n in args.sizes)
            or not all(np.isfinite(v) for v in [args.dt,args.duration,args.packing_fraction,args.wall_stiffness,args.D_plus,*args.D])
            or min(args.dt,args.duration,args.wall_stiffness) <= 0 or min(args.D_plus,*args.D) < 0
            or not 0 < args.packing_fraction <= 1 or min(args.samples,args.workers) < 1
            or not 0 <= args.seed <= args.seed+args.samples-1 < 2**32
            or not 20 <= round(args.duration/args.dt) < 2**31-1
            or (args.time_points != 0 and not 2 <= args.time_points <= round(args.duration/args.dt)+2)):
        parser.error("Use even N>=2, finite valid physics, positive counts, uint32 seeds, >=20 steps and 0 or >=2 time points")
    try:
        boundary = engine.BOUNDARIES[args.boundary]
        taus = tau_values(args.tau, args.tau_range, bool(args.time_points))
        requested = engine.make_settings(boundary,args.dt,args.duration,args.packing_fraction,args.seed,
                                         args.wall_stiffness,args.D_plus,args.time_points)
        root = (args.output or engine.data_directory(boundary,bool(args.time_points))).resolve()
        settings = engine.check_settings(root / "settings.json",requested,write=not args.dry_run)
        print(f"Boundary: {args.boundary}; PAPER units; phi={args.packing_fraction:g}, dt={args.dt:g}, duration={args.duration:g}")
        print(f"Data: {root}")
        pending, folders = [], []
        for number in sorted(set(args.sizes)):
            length = engine.domain_size(number,args.packing_fraction,boundary)
            for diffusion in sorted(set(args.D)):
                for tau in taus:
                    folder = root / f"N{number}_phi{args.packing_fraction:.12g}_D{diffusion:.12g}_tau{tau:.12g}"
                    metadata = dict(N=number,D=diffusion,tau=tau,settings=settings)
                    if not args.time_points:
                        metadata.update(L=length,neighbor_method=engine.neighbor_method(number,args.packing_fraction,boundary))
                    engine.check_settings(folder/"settings.json",metadata,write=not args.dry_run)
                    missing = 0
                    for sample in range(1,args.samples+1):
                        path = folder / f"sample_{sample:05d}.npz"
                        seed = args.seed+sample-1
                        if path.exists():
                            validate_sample(path,number,diffusion,tau,sample,seed,settings,requested["engine_hash"])
                        else:
                            pending.append((number,diffusion,tau,sample,seed,settings,None,path,requested["engine_hash"]))
                            missing += 1
                    print(f"N={number}, L={length:g}, D={diffusion:g}, tau={tau:g}: {args.samples-missing}/{args.samples} saved",flush=True)
                    folders.append(folder)
        if args.dry_run:
            print(f"Dry run: {len(pending)} missing samples; no files written.")
            return
        if pending:
            executable = engine.build_engine(boundary,bool(args.time_points),args.gsl_prefix)
            with ProcessPoolExecutor(max_workers=min(args.workers,len(pending)),mp_context=get_context("spawn")) as pool:
                jobs = [pool.submit(engine.run_sample,task[:6]+(executable,)+task[7:]) for task in pending]
                for count,job in enumerate(as_completed(jobs),1):
                    job.result()
                    if count % 10 == 0 or count == len(jobs):
                        print(f"Saved {count}/{len(jobs)} missing samples",flush=True)
        for folder in folders:
            save_summary(folder,bool(args.time_points))
        print(f"Saved data. Plot separately: python3 src/plot.py --data-dir {root}")
    except (ValueError,RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
