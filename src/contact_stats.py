"""Optional per-step measurements; run this file to collect parallel samples.

No forces, angles or random draws are changed. Events use hysteresis only for
measurement. Results are compressed per-seed files plus a readable summary CSV.
"""

from collections import namedtuple
import csv
from pathlib import Path

import numpy as np

try:
    from numba import njit, types
    from numba.typed import List
except ModuleNotFoundError:
    List = None

    def njit(**kwargs):
        return lambda function: function


Parameters = namedtuple("Parameters", "dt first_step steps epsilon near_wall center_limit wall_strength tau")
Recorder = namedtuple("Recorder", "parameters state counts lags history responses events trace_steps traces")
COUNT_NAMES = ["observations", "contact_steps", "hysteresis_steps", "persistent_steps",
               "activations", "renewals", "new_contacts", "wall_green", "wall_red",
               "clearance_green", "clearance_red"]
STATE_DTYPE = [("contact", "?"), ("start", "f8"), ("left_censored", "?"),
               ("persistent", "f8"), ("wall", "f8"), ("angle_green", "f8"),
               ("angle_red", "f8"), ("memory", "f8"), ("trace_index", "i8")]
EVENT_COLUMNS = ["kind", "start", "end", "duration", "left_censored", "right_censored",
                 "persistent_at_start", "persistent_at_end", "near_wall_at_start",
                 "angle_green_at_start", "angle_red_at_start", "memory_at_start"]
TRACE_COLUMNS = ["time", "x_green", "y_green", "x_red", "y_red", "angle_green",
                 "angle_red", "memory_left", "contact", "wall_green", "wall_red",
                 "wall_clearance_green", "wall_clearance_red"]
RESPONSE_NAMES = ["green_away", "red_away", "delta_distance", "persistent_fraction",
                  "memory_left", "initial_heading_cosine"]
ROOT = Path(__file__).resolve().parents[1]


def _rows(width):
    return List.empty_list(types.UniTuple(types.float64, width)) if List is not None else []


def make_recorder(settings, epsilon=.01, near_wall=.1, max_lag=2., lag_points=32, trace_points=64):
    dt = settings["DT"]
    steps = round(settings["DURATION"] / dt)
    horizon = min(round(max_lag / dt), steps - 1)
    lags = np.unique(np.r_[0, np.ceil(np.geomspace(1, max(1, horizon), lag_points)).astype(np.int64)])
    lags = lags[lags <= horizon]
    traces = np.unique(np.r_[0, steps - 1, np.ceil(np.geomspace(1, steps, trace_points)).astype(np.int64) - 1])
    params = Parameters(dt, settings["STEADY_STATE_STEPS"] + 1, steps, epsilon,
                        near_wall, settings["ARENA_DIAMETER"] / 2 - .5,
                        settings["WALL_REPULSION"], settings["PERSISTENCE_TIME"])
    return Recorder(params, np.zeros(1, dtype=STATE_DTYPE),
                    np.zeros(1, dtype=[(name, "f8") for name in COUNT_NAMES]),
                    lags, np.zeros((int(lags[-1]) + 1, 11)),
                    np.zeros((3, 2, len(lags), 7)), _rows(12), traces, _rows(13))


@njit(cache=True)
def _end_phase(recorder, time, memory, right_censored):
    state = recorder.state[0]
    recorder.events.append((0. if state["contact"] else 1., state["start"], time,
                            time - state["start"], (1. if state["left_censored"] else 0.),
                            (1. if right_censored else 0.), state["persistent"], (1. if memory > 0 else 0.),
                            state["wall"], state["angle_green"], state["angle_red"], state["memory"]))


def _observe_python(recorder, step, positions, angles, memory, activated):
    """Observe the pre-position-update state, after contact response and timer update."""
    params, state = recorder.parameters, recorder.state[0]
    dx, dy = positions[0, 0] - positions[1, 0], positions[0, 1] - positions[1, 1]
    distance = np.sqrt(dx * dx + dy * dy)
    was_contact = state["contact"]
    contact = was_contact
    if distance < 1.:
        contact = True
    elif distance > 1. + params.epsilon:
        contact = False
    if step < params.first_step:
        state["contact"] = contact
        return

    index = step - params.first_step
    time = index * params.dt
    clearance_green = params.center_limit - np.sqrt(positions[0, 0] ** 2 + positions[0, 1] ** 2)
    clearance_red = params.center_limit - np.sqrt(positions[1, 0] ** 2 + positions[1, 1] ** 2)
    wall_green = clearance_green < 0 and params.wall_strength > 0
    wall_red = clearance_red < 0 and params.wall_strength > 0
    near_wall = min(clearance_green, clearance_red) <= params.near_wall
    entered = contact and not was_contact

    if index == 0 or contact != was_contact:
        if index > 0:
            _end_phase(recorder, time, memory, False)
        state["start"] = time
        state["left_censored"] = index == 0
        state["persistent"] = (1. if memory > 0 else 0.)
        state["wall"] = (1. if near_wall else 0.)
        state["angle_green"], state["angle_red"] = angles[0], angles[1]
        state["memory"] = memory
        # An episode already present at the observation boundary has unknown age.
        if index == 0:
            entered = False
    state["contact"] = contact

    counts = recorder.counts[0]
    counts["observations"] += 1
    counts["contact_steps"] += (1. if distance < 1. else 0.)
    counts["hysteresis_steps"] += (1. if contact else 0.)
    counts["persistent_steps"] += (1. if memory > 0 else 0.)
    counts["activations"] += (1. if activated else 0.)
    counts["renewals"] += (1. if activated and was_contact else 0.)
    counts["new_contacts"] += (1. if entered else 0.)
    counts["wall_green"] += (1. if wall_green else 0.)
    counts["wall_red"] += (1. if wall_red else 0.)
    counts["clearance_green"] += clearance_green
    counts["clearance_red"] += clearance_red

    # A ring buffer retains only the response horizon, not the whole trajectory.
    slot = index % len(recorder.history)
    current = recorder.history[slot]
    current[:4] = positions.reshape(4)
    current[4] = distance
    current[9] = (1. if activated or (params.tau == 0 and entered) else 0.)
    if current[9]:
        nx, ny = (dx / distance, dy / distance) if distance > 1e-12 else (1., 0.)
        u = np.cos(angles[0]) * nx + np.sin(angles[0]) * ny
        current[5], current[6] = nx, ny
        current[7] = 0. if u < -.5 else (2. if u > .5 else 1.)
        current[8] = (1. if near_wall else 0.)
        current[10] = u
    # No references exist before the first contact/activation.
    for lag_index in range(len(recorder.lags)):
        lag = recorder.lags[lag_index]
        if index < lag:
            break
        reference = recorder.history[(index - lag) % len(recorder.history)]
        if not reference[9]:
            continue
        heading, wall = int(reference[7]), int(reference[8])
        values = recorder.responses[heading, wall, lag_index]
        nx, ny = reference[5], reference[6]
        values[0] += 1
        values[1] += (positions[0, 0] - reference[0]) * nx + (positions[0, 1] - reference[1]) * ny
        values[2] -= (positions[1, 0] - reference[2]) * nx + (positions[1, 1] - reference[3]) * ny
        values[3] += distance - reference[4]
        values[4] += (1. if memory > 0 else 0.)
        values[5] += memory
        values[6] += reference[10]

    if state["trace_index"] < len(recorder.trace_steps) and index == recorder.trace_steps[state["trace_index"]]:
        recorder.traces.append((time, positions[0, 0], positions[0, 1], positions[1, 0],
                                positions[1, 1], angles[0], angles[1], memory,
                                (1. if distance < 1. else 0.), (1. if wall_green else 0.),
                                (1. if wall_red else 0.), clearance_green, clearance_red))
        state["trace_index"] += 1


observe_kernel = njit(cache=True)(_observe_python)


def finish(recorder):
    """Right-censor the final episode; return compact, NumPy-only results."""
    params = recorder.parameters
    # Traces always include the last observed integration step.
    memory = recorder.traces[-1][7] if len(recorder.traces) else 0.
    _end_phase(recorder, params.steps * params.dt, memory, True)
    counts = {name: float(recorder.counts[name][0]) for name in COUNT_NAMES}
    n = counts["observations"]
    duration = n * params.dt
    activations = counts["activations"]
    summary = dict(contact_fraction=counts["contact_steps"] / n,
                   hysteresis_contact_fraction=counts["hysteresis_steps"] / n,
                   persistent_fraction=counts["persistent_steps"] / n,
                   activations=activations, activations_per_time=activations / duration,
                   new_contacts=counts["new_contacts"],
                   contacts_per_time=counts["new_contacts"] / duration,
                   renewal_fraction=counts["renewals"] / activations if activations else np.nan,
                   wall_fraction_green=counts["wall_green"] / n,
                   wall_fraction_red=counts["wall_red"] / n,
                   mean_wall_clearance_green=counts["clearance_green"] / n,
                   mean_wall_clearance_red=counts["clearance_red"] / n,
                   mean_distance_to_wall_green=counts["clearance_green"] / n + .5,
                   mean_distance_to_wall_red=counts["clearance_red"] / n + .5)
    return dict(summary=summary, events=np.array(list(recorder.events)).reshape(-1, 12),
                traces=np.array(list(recorder.traces)).reshape(-1, 13),
                lags=recorder.lags * params.dt, responses=recorder.responses.copy())


def _run_sample(task):
    import two_disks as simulation
    # Use the importable module's recorder type also when this file is the CLI.
    from contact_stats import make_recorder, finish

    sample, settings, options, path = task
    for name, value in settings.items():
        setattr(simulation, name, value)
    recorder = make_recorder(settings, **options)
    simulation.simulate(diagnostics=recorder)
    result = finish(recorder)
    temporary = path.with_suffix(".tmp")
    # One compressed file per independent seed makes resuming atomic and compact.
    with temporary.open("wb") as file:
        np.savez_compressed(file, events=result["events"], traces=result["traces"],
                            lags=result["lags"], responses=result["responses"],
                            metric_names=np.array(list(result["summary"])),
                            metrics=np.array(list(result["summary"].values())),
                            sample=sample, seed=settings["SEED"])
    temporary.replace(path)
    return sample


def main():
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import json
    from multiprocessing import get_context
    import two_disks as simulation
    from sweep import tau_grid

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--L", type=float, default=3.)
    parser.add_argument("--tau", type=float, nargs="+", default=None)
    parser.add_argument("--tau-range", type=float, nargs=3, action="append")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=simulation.SEED)
    parser.add_argument("--reorient", action="store_true")
    parser.add_argument("--dt", type=float, default=simulation.DT)
    parser.add_argument("--duration", type=float, default=simulation.DURATION)
    parser.add_argument("--steady-state-steps", type=int, default=simulation.STEADY_STATE_STEPS)
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--near-wall", type=float, default=.1)
    parser.add_argument("--max-lag", type=float, default=2.)
    parser.add_argument("--lag-points", type=int, default=32)
    parser.add_argument("--trace-points", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (not all(np.isfinite(value) for value in [args.L, args.dt, args.duration, args.epsilon, args.near_wall, args.max_lag])
            or args.L <= 1 or min(args.dt, args.duration, args.max_lag) <= 0
            or min(args.epsilon, args.near_wall, args.seed, args.steady_state_steps) < 0
            or min(args.samples, args.workers, args.lag_points, args.trace_points) < 1
            or round(args.duration / args.dt) < 1):
        parser.error("Require L>1, positive times/counts and nonnegative epsilon/near-wall/seed/warm-up")
    try:
        taus = tau_grid(args.tau, args.tau_range) if args.tau or args.tau_range else [0., .001, 1., 2.]
    except ValueError as error:
        parser.error(str(error))
    root = args.output or ROOT / "data" / "contact_stats" / ("reorient" if args.reorient else "no_reorient")
    options = dict(epsilon=args.epsilon, near_wall=args.near_wall, max_lag=args.max_lag,
                   lag_points=args.lag_points, trace_points=args.trace_points)
    settings = {name: value for name, value in vars(simulation).items()
                if name.isupper() and isinstance(value, (int, float))}
    settings.update(ARENA_DIAMETER=args.L, REORIENT_ON_CONTACT=args.reorient, DT=args.dt,
                    DURATION=args.duration, STEADY_STATE_STEPS=args.steady_state_steps, SEED=args.seed)
    print(f"Backend: {'Numba' if simulation._simulate_kernel else 'NumPy'}", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, args.samples), mp_context=get_context("spawn")) as pool:
        for tau in taus:
            folder = root / f"L{args.L:g}_tau{tau:g}"
            folder.mkdir(parents=True, exist_ok=True)
            current = {**settings, "PERSISTENCE_TIME": tau}
            metadata = dict(settings=current, diagnostics=options, schema=1)
            manifest = folder / "settings.json"
            if manifest.exists() and json.loads(manifest.read_text()) != metadata:
                parser.error(f"Changed settings in {folder}; choose another --output")
            temporary_manifest = manifest.with_suffix(".tmp")
            temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            temporary_manifest.replace(manifest)
            jobs = []
            for sample in range(1, args.samples + 1):
                path = folder / f"sample_{sample:05d}.npz"
                if not path.exists():
                    jobs.append(pool.submit(_run_sample, (sample, {**current, "SEED": args.seed + sample - 1}, options, path)))
            print(f"tau={tau:g}: {args.samples - len(jobs)}/{args.samples} already saved", flush=True)
            for job in as_completed(jobs):
                print(f"tau={tau:g}: saved sample {job.result()}", flush=True)
            with (folder / "summary.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                for sample in range(1, args.samples + 1):
                    with np.load(folder / f"sample_{sample:05d}.npz") as data:
                        if sample == 1:
                            writer.writerow(["sample", "seed", *data["metric_names"]])
                        writer.writerow([sample, int(data["seed"]), *data["metrics"]])
    print(f"Saved diagnostics in {root}. Plot with src/plot_contact_stats.py --data-dir {root}")


if __name__ == "__main__":
    main()
