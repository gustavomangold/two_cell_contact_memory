"""Plot only the isolated exactly-like-paper data; never start simulations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from demixing import FINAL_METRICS
from paper_engine import ROOT, METRICS
from plot_contact_stats import mean_sem, style


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/exactly-like-paper")
    parser.add_argument("--output", type=Path, default=ROOT / "figures/exactly-like-paper")
    args = parser.parse_args()
    manifest = args.data_dir / "settings.json"
    if not manifest.exists() or json.loads(manifest.read_text()).get("implementation") != "upstream-c":
        parser.error("Select an exactly-like-paper data folder")
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for (number, diffusion), color in zip(groups, colors):
        group = sorted([p for p in points if (p[0]["N"], p[0]["D"]) == (number, diffusion)], key=lambda p:p[0]["tau"])
        tau = np.array([p[0]["tau"] for p in group])
        for ax, index in zip(axes, [3, 0, 6]):
            mean, sem = np.array([p[1][index] for p in group]), np.array([p[2][index] for p in group])
            ax.plot(tau, mean, "o-", color=color, label=f"N={number}, D={diffusion:g}", linewidth=1.7, markersize=4)
            ax.fill_between(tau, mean-sem, mean+sem, color=color, alpha=.14, linewidth=0)
    for ax, label in zip(axes, ["Final-state demixing index, ⟨M⟩", "Green particles in unlike contact (% of time)",
                               "Particles with a neighbor at final time (%)"]):
        ax.set_xscale("symlog", linthresh=threshold, linscale=.8)
        if positive:
            ax.set_xticks([0., *10. ** np.arange(np.ceil(np.log10(threshold)), np.floor(np.log10(max(positive))) + 1)])
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        style(ax, "Contact-memory time, τ (paper units)", label)
    axes[0].axhline(.5, color="#64748b", linewidth=.8, linestyle="--")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].yaxis.set_major_formatter(PercentFormatter(1))
    axes[2].legend(frameon=False, fontsize=9)
    fig.savefig(args.output / "paper_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    with (args.output / "paper_means.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["N", "L", "packing_fraction", "D", "tau", "metric", "mean", "sem", "runs", "total_runs"])
        writer.writerows(rows)
    print(f"Saved {args.output / 'paper_comparison.png'} and paper_means.csv; bands: ±1 SEM.")


if __name__ == "__main__":
    main()
