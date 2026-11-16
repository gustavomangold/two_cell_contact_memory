"""Plot only the isolated exactly-like-paper data; never start simulations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from demixing import FINAL_METRICS
from paper_engine import ROOT, METRICS
from plot_contact_stats import mean_sem, style


def plot_time_series(data_dir, output, taus):
    import matplotlib.pyplot as plt

    manifest = data_dir / "settings.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("recorder") != "demixing-time-series-v1":
        raise ValueError("Select data recorded by paper_timeseries.py; ordinary sweep files contain only final demixing snapshots")
    points, rows = [], []
    for folder in sorted(data_dir.glob("N*_phi*_D*_tau*")):
        meta = json.loads((folder / "settings.json").read_text())
        if not np.isclose(meta["tau"], taus, rtol=1e-10, atol=0).any():
            continue
        times, runs = None, []
        for path in sorted(folder.glob("sample_*.npz")):
            with np.load(path) as data:
                if (tuple(data["metric_names"]) != FINAL_METRICS
                        or data["demixing"].shape != (len(data["times"]), len(FINAL_METRICS))):
                    raise ValueError(f"Unexpected time-series metrics in {path}")
                if times is None:
                    times = data["times"].copy()
                elif not np.array_equal(times, data["times"]):
                    raise ValueError(f"Mismatched recording times in {path}")
                runs.append(data["demixing"][:, 0])
        if not runs:
            continue
        mean, sem, count = mean_sem(runs)
        points.append((meta, times, mean, sem))
        rows.extend([meta["N"], meta["D"], meta["tau"], t, m, s, n]
                    for t, m, s, n in zip(times, mean, sem, count))
    if not points:
        raise ValueError(f"No completed time series for the requested tau values in {data_dir}")
    missing = [tau for tau in taus if not any(np.isclose(tau, p[0]["tau"], rtol=1e-10, atol=0) for p in points)]
    if missing:
        print("No recorded time series for tau=" + ", ".join(f"{tau:g}" for tau in missing))
    output.mkdir(parents=True, exist_ok=True)
    for number, diffusion in sorted({(p[0]["N"], p[0]["D"]) for p in points}):
        fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
        group = sorted([p for p in points if (p[0]["N"], p[0]["D"]) == (number, diffusion)], key=lambda p:p[0]["tau"])
        for meta, times, mean, sem in group:
            ax.errorbar(times, mean, yerr=sem, fmt="o-", capsize=3, markersize=4,
                        linewidth=1.5, label=f"τ={meta['tau']:g}")
        ax.axhline(.5, color="#64748b", linewidth=.8, linestyle="--")
        ax.set(xlim=(0, max(p[1][-1] for p in group)), ylim=(0, 1))
        style(ax, "Time (paper units)", "Demixing index, ⟨M⟩")
        ax.legend(title=f"N={number}, D={diffusion:g}", frameon=False, fontsize=9)
        path = output / f"paper_demixing_time_N{number}_D{diffusion:g}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}; error bars: ±1 SEM.")
    with (output / "paper_demixing_time.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "D", "tau", "time", "mean", "sem", "runs"])
        writer.writerows(rows)


def main(argv=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-series", action="store_true", help="Plot demixing versus time instead of final demixing versus tau")
    parser.add_argument("--tau", type=float, nargs="+", default=[.001, .01, 1.], help="Tau values for --time-series (default: .001 .01 1)")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.data_dir = args.data_dir or ROOT / "data/exactly-like-paper" / ("time_series" if args.time_series else "")
    if args.time_series:
        args.output = args.output or ROOT / "figures/exactly-like-paper/time_series"
        if any(not np.isfinite(tau) or tau < 0 for tau in args.tau):
            parser.error("Tau values must be finite and nonnegative")
        try:
            plot_time_series(args.data_dir, args.output, args.tau)
        except ValueError as error:
            parser.error(str(error))
        return
    manifest = args.data_dir / "settings.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("implementation") != "upstream-c":
        parser.error("Select an exactly-like-paper data folder")
    arena = json.loads(manifest.read_text()).get("boundary") == "circular-arena"
    args.output = args.output or ROOT / "figures" / ("paper_arena/constant-density" if arena else "exactly-like-paper")
    points, rows = [], []
    names = (*METRICS, *FINAL_METRICS)
    for folder in sorted(args.data_dir.glob("N*_phi*_D*_tau*")):
        metadata = json.loads((folder / "settings.json").read_text())
        runs = []
        for path in sorted(folder.glob("sample_*.npz")):
            with np.load(path) as data:
                if tuple(data["metric_names"]) != METRICS or tuple(data["final_metric_names"]) != FINAL_METRICS:
                    parser.error(f"Unexpected metrics in {path}")
                runs.append(np.r_[data["metrics"], data["final_metrics"]])
        if not runs:
            continue
        mean, sem, n = mean_sem(runs)
        points.append((metadata, mean, sem))
        for index, name in enumerate(names):
            rows.append([metadata["N"], metadata["L"], metadata["settings"]["packing_fraction"],
                         metadata["D"], metadata["tau"], name, mean[index], sem[index], n[index], len(runs)])
    if not points:
        parser.error(f"No completed paper samples in {args.data_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    groups = sorted({(meta["N"], meta["D"]) for meta, _, _ in points})
    positive = [meta["tau"] for meta, _, _ in points if meta["tau"] > 0]
    threshold = min(positive) if positive else 1e-5
    colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(groups)))
    fig, ax = plt.subplots(figsize=(7, 4.8), layout="constrained")
    for (number, diffusion), color in zip(groups, colors):
        group = sorted([p for p in points if (p[0]["N"], p[0]["D"]) == (number, diffusion)], key=lambda p:p[0]["tau"])
        tau = np.array([p[0]["tau"] for p in group])
        mean, sem = np.array([p[1][3] for p in group]), np.array([p[2][3] for p in group])
        ax.plot(tau, mean, "o-", color=color, label=f"N={number}", linewidth=1.7, markersize=4)
        ax.fill_between(tau, mean-sem, mean+sem, color=color, alpha=.14, linewidth=0)
    ax.set_xscale("symlog", linthresh=threshold, linscale=.8)
    taus = sorted({meta["tau"] for meta, _, _ in points})
    ax.set_xticks(taus, labels=[f"{tau:g}" for tau in taus], rotation=45, ha="right")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    style(ax, "Contact-memory time, τ (paper units)", "Final-state demixing index, ⟨M⟩")
    ax.axhline(.5, color="#64748b", linewidth=.8, linestyle="--")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(args.output / "paper_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (args.output / "paper_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "L", "packing_fraction", "D", "tau", "metric", "mean", "sem", "runs", "total_runs"])
        writer.writerows(rows)
    print(f"Saved {args.output / 'paper_comparison.png'} and paper_means.csv; bands: ±1 SEM.")


if __name__ == "__main__":
    main()
