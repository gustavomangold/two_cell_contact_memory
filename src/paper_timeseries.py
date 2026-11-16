"""Record compact demixing time series with the unchanged paper dynamics."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from demixing import FINAL_METRICS, snapshot_demixing
from paper_engine import ROOT, DEFAULTS, source_hash
from paper_movie import build_movie_engine, record_movie


RECORDER = "demixing-time-series-v1"


def recorder_hash():
    digest = hashlib.sha256()
    for name in ["paper_timeseries.py", "paper_movie.py", "paper_movie.c", "demixing.py"]:
        digest.update(Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def check_settings(path, settings):
    if path.exists():
        if json.loads(path.read_text()) != settings:
            raise ValueError(f"Changed time-series settings in {path.parent}; use another --output")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def record_sample(task):
    number, diffusion, tau, sample, seed, settings, path, gsl_prefix = task
    path = Path(path)
    movie = record_movie(number, diffusion, tau, seed, settings, 0., 1,
                         settings["time_points"], path.with_suffix(".log"), gsl_prefix)
    values = np.array([snapshot_demixing(p, box_size=movie["L"]) for p in movie["tracks"]])
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, N=number, L=movie["L"], D=diffusion, tau=tau,
                            sample=sample, seed=seed, times=movie["times"],
                            metric_names=np.array(FINAL_METRICS), demixing=values,
                            final_rng_state=movie["rng_state"],
                            settings_json=json.dumps(settings, sort_keys=True))
    temporary.replace(path)
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--tau", type=float, nargs="+", default=[.001, .01, 1.])
    parser.add_argument("--D", type=float, default=360000.)
    parser.add_argument("--packing-fraction", type=float, default=DEFAULTS["packing_fraction"])
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    parser.add_argument("--duration", type=float, default=DEFAULTS["duration"])
    parser.add_argument("--time-points", type=int, default=21, help="Evenly spaced snapshots, including initial and final states")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--gsl-prefix", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/exactly-like-paper/time_series")
    args = parser.parse_args()
    if (any(n < 2 or n % 2 for n in args.sizes)
            or not all(np.isfinite(v) for v in [args.D, args.dt, args.duration, args.packing_fraction, *args.tau])
            or min(args.D, *args.tau) < 0 or min(args.dt, args.duration) <= 0
            or not 0 < args.packing_fraction <= 1 or min(args.samples, args.workers) < 1
            or not 0 <= args.seed <= args.seed + args.samples - 1 < 2**32
            or not 20 <= round(args.duration / args.dt) < 2**31 - 1
            or not 2 <= args.time_points <= round(args.duration / args.dt) + 2):
        parser.error("Use even N>=2, nonnegative D/tau, positive times/counts, 0<phi<=1, uint32 seeds, >=20 steps and 2..steps+2 time points")
    settings = {**DEFAULTS, "dt": args.dt, "duration": args.duration,
                "packing_fraction": args.packing_fraction, "seed": args.seed,
                "source_hash": source_hash(), "recorder": RECORDER,
                "recorder_hash": recorder_hash(), "time_points": args.time_points}
    try:
        check_settings(args.output / "settings.json", settings)
        build_movie_engine(args.gsl_prefix)
        with ProcessPoolExecutor(max_workers=min(args.workers, args.samples), mp_context=get_context("spawn")) as pool:
            for number in sorted(set(args.sizes)):
                for tau in sorted(set(args.tau)):
                    folder = args.output / f"N{number}_phi{args.packing_fraction:.12g}_D{args.D:.12g}_tau{tau:.12g}"
                    check_settings(folder / "settings.json", dict(N=number, D=args.D, tau=tau, settings=settings))
                    jobs = []
                    for sample in range(1, args.samples + 1):
                        seed = args.seed + sample - 1
                        path = folder / f"sample_{sample:05d}.npz"
                        if path.exists():
                            with np.load(path) as data:
                                if (int(data["N"]) != number or float(data["D"]) != args.D
                                        or float(data["tau"]) != tau or int(data["sample"]) != sample
                                        or int(data["seed"]) != seed
                                        or str(data["settings_json"]) != json.dumps(settings, sort_keys=True)):
                                    raise ValueError(f"Incompatible time series: {path}")
                        else:
                            jobs.append(pool.submit(record_sample, (number, args.D, tau, sample, seed,
                                                                     settings, path, args.gsl_prefix)))
                    print(f"N={number}, tau={tau:g}: {args.samples-len(jobs)}/{args.samples} saved", flush=True)
                    for job in as_completed(jobs):
                        print(f"  Saved sample {job.result()}", flush=True)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(f"Saved {args.output}. Plot separately: python3 src/paper_plot.py --time-series --data-dir {args.output}")


if __name__ == "__main__":
    main()
