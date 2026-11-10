"""Compare green contact-time curves across system sizes; plotting runs no simulations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from many_disks import METRICS, normalize
from plot_contact_stats import mean_sem, style


ROOT = Path(__file__).resolve().parents[1]


def load_points(root):
    points = []
    for folder in root.glob("N*_L*_tau*"):
        manifest = folder / "settings.json"
        if not manifest.exists():
            continue
        metadata = json.loads(manifest.read_text())
        runs = []
        for path in sorted(folder.glob("sample_*.npz")):
            with np.load(path) as data:
                if tuple(data["metric_names"]) != METRICS:
                    raise ValueError(f"Unexpected metric columns in {path}")
                runs.append({key: data[key].copy() for key in data.files})
        if runs:
            mean, sem, n = mean_sem([run["metrics"] for run in runs])
            by_seed = {int(run["seed"]): run["metrics"][0] for run in runs}
            if len(by_seed) != len(runs):
                raise ValueError(f"Duplicate seeds in {folder}")
            points.append(dict(metadata=metadata, runs=runs, mean=mean, sem=sem, n=n, by_seed=by_seed))
    if not points:
        raise ValueError(f"No completed size-sweep samples in {root}")
    return sorted(points, key=lambda point: (point["metadata"]["N"], point["metadata"]["tau"]))


def baseline_change(point, baseline):
    seeds = sorted(point["by_seed"].keys() & baseline["by_seed"].keys())
    differences = [point["by_seed"][seed] - baseline["by_seed"][seed] for seed in seeds]
    if not differences:
        return np.nan, np.nan, 0
    mean, sem, n = mean_sem(np.array(differences))
    return float(mean), float(sem), int(n)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "size_scaling" / "diameter")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--xscale", choices=["linear", "symlog"], default="symlog")
    parser.add_argument("--transient-tau", type=float, help="Also plot contact means versus recording time for this tau")
    parser.add_argument("--demixing", action="store_true", help="Also plot local same-type preference and neighbor coverage")
    args = parser.parse_args()
    try:
        points = load_points(args.data_dir)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or ROOT / "figures" / "size_scaling" / args.data_dir.name
    output.mkdir(parents=True, exist_ok=True)
    sizes = sorted({point["metadata"]["N"] for point in points})
    colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(sizes)))
    positive = [point["metadata"]["tau"] for point in points if point["metadata"]["tau"] > 0]
    threshold = min(positive) if positive else .0001
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    rows, any_baseline = [], False

    def curve(ax, x, mean, sem, color, label):
        ax.plot(x, mean, "o-", color=color, markersize=4, linewidth=1.7, label=label)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=.14, linewidth=0)

    def axis(ax, ylabel):
        if args.xscale == "symlog":
            ax.set_xscale("symlog", linthresh=threshold, linscale=.8)
        ax.set_xlim(left=0)
        style(ax, "Contact-memory time, τ", ylabel)

    for number, color in zip(sizes, colors):
        group = [point for point in points if point["metadata"]["N"] == number]
        length = group[0]["metadata"]["L"]
        tau = np.array([point["metadata"]["tau"] for point in group])
        mean = np.array([point["mean"][0] for point in group])
        sem = np.array([point["sem"][0] for point in group])
        label = f"N={number}, L={length:g}"
        curve(axes[0], tau, mean, sem, color, label)
        baseline = next((point for point in group if point["metadata"]["tau"] == 0), None)
        changes = [baseline_change(point, baseline) for point in group] if baseline else [(np.nan, np.nan, 0)] * len(group)
        if baseline:
            any_baseline = True
            curve(axes[1], tau, 100 * np.array([value[0] for value in changes]),
                  100 * np.array([value[1] for value in changes]), color, label)
        for point, difference in zip(group, changes):
            meta = point["metadata"]
            contacts = sum(run["metrics"][0] > 0 for run in point["runs"])
            for index, name in enumerate(METRICS):
                rows.append([number, length, meta["packing_fraction"], meta["tau"], name,
                             point["mean"][index], point["sem"][index], point["n"][index], contacts])
            rows.append([number, length, meta["packing_fraction"], meta["tau"], "contact_fraction_change",
                         *difference, contacts])
            print(f"N={number}, tau={meta['tau']:g}: {contacts}/{len(point['runs'])} runs observed unlike contact", flush=True)
    axis(axes[0], "Green particles in unlike contact (% of time)")
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=9)
    axis(axes[1], "Added contact time versus τ=0\n(percentage points)")
    axes[1].axhline(0, color="#64748b", linewidth=.8)
    if not any_baseline:
        axes[1].text(.5, .5, "Collect τ=0 for the baseline", transform=axes[1].transAxes, ha="center")
    fig.savefig(output / "contact_fraction.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (output / "size_scaling_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "L", "packing_fraction", "tau", "metric", "mean", "sem", "runs", "runs_with_unlike_contact"])
        writer.writerows(rows)

    if args.transient_tau is not None:
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        found = False
        for number, color in zip(sizes, colors):
            selected = [point for point in points if point["metadata"]["N"] == number
                        and point["metadata"]["tau"] == args.transient_tau]
            if not selected:
                continue
            point = selected[0]
            values = [normalize(run["window_sums"], number, np.diff(run["window_edges"]))[:, 0] for run in point["runs"]]
            mean, sem, _ = mean_sem(values)
            time = point["runs"][0]["window_edges"][1:] * point["metadata"]["settings"]["DT"]
            ax.plot(time, mean, color=color, label=f"N={number}")
            ax.fill_between(time, mean - sem, mean + sem, color=color, alpha=.14)
            found = True
        if not found:
            parser.error(f"No recorded tau={args.transient_tau:g} for the transient plot")
        ax.set_xscale("log")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        style(ax, "Time after warm-up (simulation units)", "Green contact fraction within time window (%)")
        ax.legend(frameon=False)
        fig.savefig(output / f"contact_fraction_time_tau{args.transient_tau:g}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if args.demixing:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        for number, color in zip(sizes, colors):
            group = [point for point in points if point["metadata"]["N"] == number]
            tau = np.array([point["metadata"]["tau"] for point in group])
            if number > 2:
                random_mixing = (number / 2 - 1) / (number - 1)
                mean = (np.array([point["mean"][4] for point in group]) - random_mixing) / (1 - random_mixing)
                sem = np.array([point["sem"][4] for point in group]) / (1 - random_mixing)
                curve(axes[0], tau, mean, sem, color, f"N={number}")
            curve(axes[1], tau, np.array([point["mean"][5] for point in group]),
                  np.array([point["sem"][5] for point in group]), color, f"N={number}")
        axis(axes[0], "Local same-type preference\n(chance corrected, N≥4)")
        axes[0].axhline(0, color="#64748b", linewidth=.8)
        axis(axes[1], "Particles with a neighbor (%)")
        axes[1].set_ylim(0, 1)
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].legend(frameon=False)
        fig.savefig(output / "local_demixing.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved plots and summary CSV in {output}. Bands: ±1 SEM across independent runs; baseline differences use paired seeds.")


if __name__ == "__main__":
    main()
