"""Plot sweep means, compare contact rules, or inspect a steady-state time series."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from sweep_storage import HEADER, RAW_HEADER

ROOT = Path(__file__).resolve().parents[1]


def summarize_file(task):
    path, samples = task
    sums = np.zeros(samples + 1)
    counts = np.zeros(samples + 1, dtype=np.int64)
    with path.open("rb") as file:
        header = file.readline()
        fields = header.decode().strip().split(",")
        if fields not in (RAW_HEADER, HEADER):
            raise ValueError(f"Unexpected CSV header: {path}")
        compact = fields == HEADER
        length, tau = map(float, file.readline().split(b",")[:2])
        file.seek(len(header))
        while True:
            position = file.tell()
            if not file.readline():
                break
            file.seek(position)
            # Bound memory use even for CSVs containing millions of rows.
            columns = (2, 7, 9) if compact else (2, 7)
            data = np.loadtxt(file, delimiter=",", usecols=columns,
                              max_rows=250_000, ndmin=2)
            if (not np.isfinite(data).all() or (data[:, 0] < 1).any()
                    or (data[:, 0] > samples).any()):
                raise ValueError(f"Invalid data or sample IDs in {path}")
            ids = data[:, 0].astype(np.int64)
            if not np.array_equal(ids, data[:, 0]):
                raise ValueError(f"Noninteger sample IDs in {path}")
            if compact:
                weights = data[:, 1].astype(np.int64)
                if (weights < 1).any() or not np.array_equal(weights, data[:, 1]):
                    raise ValueError(f"Invalid window counts in {path}")
                sums += np.bincount(ids, weights=data[:, 2] * weights, minlength=samples + 1)
                counts += np.bincount(ids, weights=weights, minlength=samples + 1).astype(np.int64)
            else:
                sums += np.bincount(ids, weights=data[:, 1], minlength=samples + 1)
                counts += np.bincount(ids, minlength=samples + 1)
    if not counts[1] or not np.all(counts[1:] == counts[1]):
        raise ValueError(f"{path.name}: not all {samples} samples are complete")
    means = sums[1:] / counts[1:]
    return length, tau, samples, means.mean(), means.std(ddof=1) / np.sqrt(samples) if samples > 1 else 0.


def read_sweep(folder, samples, workers, length=None, tau_max=None):
    paths = []
    for path in sorted(folder.glob("L*_tau*.csv")):
        saved_l, saved_tau = map(float, path.stem[1:].split("_tau"))
        if (length is None or saved_l == length) and (tau_max is None or saved_tau <= tau_max):
            paths.append(path)
    if not paths:
        raise ValueError(f"No matching sweep CSVs in {folder}")
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(paths)),
                             mp_context=get_context("spawn")) as pool:
        jobs = {}
        for path in paths:
            before = path.stat()
            jobs[pool.submit(summarize_file, (path, samples))] = path, (before.st_size, before.st_mtime_ns)
        for job in as_completed(jobs):
            path, before = jobs[job]
            try:
                row = job.result()
                after = path.stat()
                if before != (after.st_size, after.st_mtime_ns):
                    raise ValueError("file is still being written")
                rows.append(row)
            except ValueError as error:
                print(f"Skipped {path.name}: {error}", flush=True)
    if not rows:
        raise ValueError(f"No complete points in {folder}")
    return np.array(sorted(rows))


def draw_sweeps(series, output, xscale, linear_threshold, tau_max, split_at=.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

    taus = np.concatenate([data[:, 1] for data in series.values()])
    # Logarithmic spacing for positive tau, with a short linear region at zero.
    if xscale == "log" and (taus == 0).any():
        xscale = "symlog"
    if linear_threshold is None:
        positive = taus[taus > 0]
        linear_threshold = positive.min() if positive.size else .0001
    upper = tau_max if tau_max is not None else taus.max()
    # A wide linear panel gives the larger memory times more room.
    split = xscale == "split" and upper > split_at
    if xscale == "split" and not split:
        xscale = "symlog"

    lengths = np.unique(np.concatenate([rows[:, 0] for rows in series.values()]))
    columns = min(2 if split else 3, len(lengths))
    rows = (len(lengths) + columns - 1) // columns
    panels = 2 if split else 1
    fig, axes = plt.subplots(rows, panels * columns,
                             figsize=((9.2 if split else 6) * columns, 4.6 * rows),
                             sharex=not split, sharey=True, squeeze=False, layout="constrained",
                             gridspec_kw={"width_ratios": [1, 2.6] * columns} if split else None)
    styles = [("#0072B2", "o", "-"), ("#D55E00", "D", "--")]
    for index, length in enumerate(lengths):
        for panel in range(panels):
            ax = axes.flat[index * panels + panel]
            for (label, data), (color, marker, line) in zip(series.items(), styles):
                points = data[data[:, 0] == length]
                if not len(points):
                    continue
                tau, mean, sem = points[:, 1], points[:, 3], points[:, 4]
                ax.fill_between(tau, mean - sem, mean + sem, color=color, alpha=.13, linewidth=0)
                ax.plot(tau, mean, color=color, marker=marker, linestyle=line,
                        markersize=4.5, linewidth=1.8, markeredgecolor="white",
                        markeredgewidth=.6, label=label)
            scale = ("symlog" if panel == 0 else "linear") if split else xscale
            if scale == "symlog":
                ax.set_xscale(scale, linthresh=linear_threshold, linscale=.8)
            else:
                ax.set_xscale(scale)
            if split:
                ax.set_xlim((0, split_at) if panel == 0 else (split_at, upper))
                if panel == 0:
                    # Keep the small panel's logarithmic tick labels readable.
                    decades = np.arange(np.ceil(np.log10(linear_threshold)),
                                        np.floor(np.log10(split_at)) + 1)
                    tick_stride = max(1, int(np.ceil(len(decades) / 3)))
                    ax.xaxis.set_major_locator(FixedLocator(np.r_[0, 10. ** decades[::tick_stride]]))
                else:
                    ticks = ax.get_xticks()
                    ax.set_xticks(np.r_[split_at, ticks[(ticks > split_at) & (ticks <= upper)]])
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
                    ax.spines["left"].set_visible(False)
                    ax.tick_params(axis="y", left=False, labelleft=False)
                # Diagonal marks indicate the change of horizontal scale.
                ax.plot([1 if panel == 0 else 0] * 2, [0, 1], transform=ax.transAxes,
                        marker=[(-1, -.5), (1, .5)], markersize=9, linestyle="none",
                        color="#94a3b8", clip_on=False)
            else:
                if xscale != "log":
                    ax.set_xlim(left=0)
                if tau_max is not None:
                    ax.set_xlim(right=tau_max)
                ax.set_xlabel("Contact-memory time, τ", labelpad=10)
            if panel == 0:
                ax.set_ylabel("Mean separation / L", labelpad=10)
            ax.yaxis.set_major_locator(MaxNLocator(6))
            ax.grid(axis="y", color="#dbe2ea", linewidth=.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color("#94a3b8")
            ax.tick_params(color="#94a3b8", pad=6)
            if panel == panels - 1:
                ax.text(.97, 1.02, f"L = {length:g}", ha="right", va="bottom", transform=ax.transAxes)
                if len(series) > 1:
                    ax.legend(loc="lower right", frameon=False)
    if split:
        fig.supxlabel("Contact-memory time, τ")
    for ax in list(axes.flat)[len(lengths) * panels:]:
        ax.set_visible(False)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def draw_steady_state(folder, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.loadtxt(folder / "mean_distance.csv", delimiter=",", skiprows=1, ndmin=2)
    blocks = np.loadtxt(folder / "time_windows.csv", delimiter=",", skiprows=1, ndmin=2)
    steps, mean, sem = data[:, 0], data[:, 2], data[:, 3]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, layout="constrained")
    axes[0].plot(steps, mean, color="#0072B2", linewidth=.9)
    axes[0].fill_between(steps, mean - 2 * sem, mean + 2 * sem, color="#0072B2", alpha=.18)
    axes[1].errorbar(blocks[:, 0], blocks[:, 1], yerr=2 * blocks[:, 2],
                     fmt="o-", color="#D55E00", capsize=3, markersize=4)
    tail = mean[steps >= .75 * steps[-1]].mean()
    for ax, label in zip(axes, ("Instantaneous mean", "Time-window means")):
        ax.axhline(tail, color="#64748b", linestyle="--", linewidth=1, label="Final-quarter mean")
        ax.text(.02, .95, label, va="top", transform=ax.transAxes)
        ax.set_ylabel("Distance / disk diameter")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlabel("Simulation step")
    axes[1].set_xlim(0, steps[-1])
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[1].legend(frameon=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path, help="Sweep folder")
    source.add_argument("--steady-state", type=Path, help="Steady-state diagnostic folder")
    parser.add_argument("--compare", type=Path, help="Re-orient sweep folder to overlay")
    parser.add_argument("--labels", nargs=2, default=["Do not re-orient", "Re-orient"])
    parser.add_argument("--L", type=float, help="Show only this arena diameter")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--xscale", choices=["split", "linear", "symlog", "log"],
                        help="Comparison default: split (small-tau symlog panel + wider linear panel)")
    parser.add_argument("--split-at", type=float, default=.25,
                        help="Tau where the split view changes from symlog to linear (default: 0.25)")
    parser.add_argument("--linear-threshold", type=float,
                        help="End of the linear region near zero (default: smallest positive tau)")
    parser.add_argument("--tau-max", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.samples < 1 or args.workers < 1 or not np.isfinite(args.split_at) or args.split_at <= 0
            or (args.linear_threshold is not None and
                (not np.isfinite(args.linear_threshold) or args.linear_threshold <= 0))
            or (args.L is not None and (not np.isfinite(args.L) or args.L <= 1))
            or (args.tau_max is not None and (not np.isfinite(args.tau_max) or args.tau_max <= 0))):
        parser.error("Require positive samples/workers/split-at/linear-threshold/tau-max and L>1")
    if args.steady_state and args.compare:
        parser.error("--compare requires --data-dir")
    folder = args.data_dir or args.steady_state
    name = "comparison" if args.compare else folder.name
    output = args.output or ROOT / "figures" / f"{name}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.steady_state:
            draw_steady_state(folder, output)
        else:
            if args.compare and args.labels[0] == args.labels[1]:
                raise ValueError("Comparison labels must be distinct")
            series = {args.labels[0]: read_sweep(folder, args.samples, args.workers, args.L, args.tau_max)}
            if args.compare:
                series[args.labels[1]] = read_sweep(args.compare, args.samples, args.workers, args.L, args.tau_max)
            scale = args.xscale or ("split" if args.compare else "linear")
            draw_sweeps(series, output, scale, args.linear_threshold, args.tau_max, args.split_at)
            with output.with_suffix(".csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["mode", "L", "tau", "samples", "mean_separation_over_L", "sem"])
                for label, rows in series.items():
                    writer.writerows([[label, *row] for row in rows])
            print(f"Saved summary: {output.with_suffix('.csv')}")
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()

