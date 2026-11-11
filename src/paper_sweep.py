"""Parallel, resumable Fig. 4 persistence sweeps in the paper's original units."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from demixing import FINAL_METRICS
from paper_engine import ROOT, DEFAULTS, METRICS, build_engine, run_sample, source_hash
from sweep import tau_grid


TAUS = [0., 1e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]


def check_settings(path, settings):
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Changed paper settings in {path.parent}; use another --output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4000])
    parser.add_argument("--D", type=float, nargs="+", default=[360000.], help="Actual paper diffusion coefficient, not sqrt(D)")
    parser.add_argument("--tau", type=float, nargs="+")
    parser.add_argument("--tau-range", type=float, nargs=3, action="append")
    parser.add_argument("--packing-fraction", type=float, default=DEFAULTS["packing_fraction"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--duration", type=float, default=DEFAULTS["duration"])
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--gsl-prefix", type=Path, help="Optional private GSL installation prefix")
    parser.add_argument("--output", type=Path, default=ROOT / "data/exactly-like-paper")
    args = parser.parse_args()
    if (any(n < 2 or n % 2 for n in args.sizes)
            or not all(np.isfinite(v) for v in [args.dt, args.duration, args.packing_fraction, *args.D])
            or min(args.dt, args.duration, args.packing_fraction) <= 0 or args.packing_fraction > 1
            or min(args.D) < 0 or min(args.samples, args.workers) < 1 or not 0 <= args.seed < 2**32
            or args.seed + args.samples - 1 >= 2**32
            or not 20 <= round(args.duration / args.dt) < 2**31 - 1):
        parser.error("Use even N, positive times, 0<packing fraction<=1, nonnegative D, positive counts, uint32 seeds and 20..2^31-2 steps")
    if any(int(np.sqrt(n * .785398 / args.packing_fraction) / 2) < 3 for n in args.sizes):
        parser.error("The original linked-cell code needs >=3 cells per side: increase N or reduce packing fraction")
    try:
        taus = tau_grid(args.tau, args.tau_range) if args.tau or args.tau_range else TAUS
        settings = {**DEFAULTS, "dt": args.dt, "duration": args.duration,
                    "packing_fraction": args.packing_fraction, "seed": args.seed,
                    "source_hash": source_hash()}
        root = args.output.resolve()
        check_settings(root / "settings.json", settings)
        print("Building/checking the original C engine (GSL); all times use PAPER units.", flush=True)
        executable = build_engine(args.gsl_prefix)
        print(f"Data: {root}; periodic square, phi={args.packing_fraction:g}, dt={args.dt:g}, final time={args.duration:g}", flush=True)
        with ProcessPoolExecutor(max_workers=min(args.workers, args.samples), mp_context=get_context("spawn")) as pool:
            for number in sorted(set(args.sizes)):
                length = float(np.sqrt(number * .785398 / args.packing_fraction))
                for diffusion in sorted(set(args.D)):
                    for tau in taus:
                        folder = root / f"N{number}_phi{args.packing_fraction:.12g}_D{diffusion:.12g}_tau{tau:.12g}"
                        metadata = dict(N=number, L=length, D=diffusion, tau=tau, settings=settings)
                        check_settings(folder / "settings.json", metadata)
                        jobs = []
                        for sample in range(1, args.samples + 1):
                            seed = args.seed + sample - 1
                            path = folder / f"sample_{sample:05d}.npz"
                            if path.exists():
                                with np.load(path) as data:
                                    if (int(data["N"]) != number or float(data["L"]) != length
                                            or float(data["D"]) != diffusion or float(data["tau"]) != tau
                                            or int(data["sample"]) != sample or int(data["seed"]) != seed
                                            or str(data["settings_json"]) != json.dumps(settings, sort_keys=True)
                                            or str(data["source_hash"]) != settings["source_hash"]):
                                        raise ValueError(f"Incompatible paper sample: {path}")
                            else:
                                jobs.append(pool.submit(run_sample, (number, diffusion, tau, sample, seed,
                                                                     settings, executable, path)))
                        print(f"N={number}, D={diffusion:g}, tau={tau:g}: {args.samples-len(jobs)}/{args.samples} saved", flush=True)
                        for job in as_completed(jobs):
                            print(f"  Saved sample {job.result()}", flush=True)
                        temporary = folder / "summary.tmp"
                        with temporary.open("w", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow(["sample", "seed", *METRICS, *FINAL_METRICS])
                            for path in sorted(folder.glob("sample_*.npz")):
                                with np.load(path) as data:
                                    writer.writerow([int(data["sample"]), int(data["seed"]), *data["metrics"], *data["final_metrics"]])
                        temporary.replace(folder / "summary.csv")
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved in {root}. Plot separately: python3 src/paper_plot.py --data-dir {root}")


if __name__ == "__main__":
    main()
