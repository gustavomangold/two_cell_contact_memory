"""Plot post-separation displacement and an equal-time no-memory comparison."""

import argparse
import csv
from pathlib import Path

import numpy as np
from contact_stats import ROOT
from post_separation import POST_COLUMNS, FIXED_COLUMNS, post_summary
from plot_contact_stats import load_cases, mean_sem, style


def selected_measurements(run, wall="all", wall_free=False):
    post, fixed = run["post_separation"], run["fixed_after_separation"]
    if wall != "all":
        near = 1 if wall == "near" else 0
        post, fixed = post[post[:, 8] == near], fixed[fixed[:, 5] == near]
    # Availability concerns the separation instant, not later wall encounters.
    available = np.mean(fixed[:, 2] > 0) if len(fixed) else np.nan
    starts = len(fixed)
    if wall_free:
        post, fixed = post[post[:, 9] == 0], fixed[fixed[:, 8] == 0]
    metrics = post_summary(post, fixed)
    metrics["post_separation_active_fraction"] = available
    counts = [starts, int(np.count_nonzero(post[:, 7] != 2)), int(np.count_nonzero(post[:, 7] == 2)),
              int(np.count_nonzero(fixed[:, 10] == 1)), int(np.count_nonzero(fixed[:, 10] == 0))]
    return metrics, counts, post, fixed


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "contact_stats" / "post_separation" / "no_reorient")
    parser.add_argument("--L", type=float, default=3.)
    parser.add_argument("--wall", choices=["all", "near", "away"], default="all")
    parser.add_argument("--wall-free", action="store_true", help="Exclude displacement intervals that reach a wall-force region")
    parser.add_argument("--xscale", choices=["linear", "symlog"], default="symlog")
    parser.add_argument("--export-events", action="store_true")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()
    try:
        cases = load_cases(args.data_dir, args.L)
        windows = []
        for _, _, runs, folder in cases:
            for run in runs:
                if "post_separation" not in run:
                    raise ValueError(f"{folder} contains old recordings. Collect new runs with --post-separation-time 1.")
                windows.append(float(run["post_separation_window"]))
        if not np.allclose(windows, windows[0], rtol=1e-12, atol=1e-12):
            raise ValueError("The fixed-time windows differ; plot matching window lengths together.")
    except ValueError as error:
        parser.error(str(error))
    window = windows[0]
    output = args.output or ROOT / "figures" / "post_separation" / args.data_dir.name / f"L{args.L:g}"
    output.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.wall}" if args.wall != "all" else "") + ("_wall_free" if args.wall_free else "")
    summaries, rows = [], []
    for tau, _, runs, _ in cases:
        metrics, counts, exports = [], [], []
        for run in runs:
            values, count, post, fixed = selected_measurements(run, args.wall, args.wall_free)
            metrics.append(list(values.values()))
            names = list(values)
            counts.append(count)
            exports.append((run, post, fixed))
        mean, sem, n = mean_sem(metrics)
        summaries.append((tau, dict(zip(names, mean)), dict(zip(names, sem))))
        totals = np.sum(counts, axis=0)
        rows.extend([[tau, name, m, error, int(number), *totals]
                     for name, m, error, number in zip(names, mean, sem, n)])
        if args.export_events:
            for key, columns, index in [("memory", POST_COLUMNS, 1), ("fixed", FIXED_COLUMNS, 2)]:
                path = output / f"tau{tau:g}_post_separation_{key}{suffix}.csv"
                with path.open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sample", "seed", *columns])
                    for entry in exports:
                        run = entry[0]
                        for row in entry[index]:
                            values = list(row)
                            if key == "memory":
                                values[7] = ["memory_expired", "recontact", "observation_end"][int(row[7])]
                            writer.writerow([int(run["sample"]), int(run["seed"]), *values])
        print(f"tau={tau:g}: {totals[0]} separations; {totals[1]} finished / {totals[2]} censored memory excursions", flush=True)

    memory_fig, memory_axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    fixed_fig, fixed_axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    panels = [
        (memory_axes[0], "post_separation_active_fraction", "Separations with memory still active (%)", False),
        (memory_axes[1], "post_separation_net_displacement", "Green displacement before expiry/recontact\n(disk diameters)", False),
        (memory_axes[2], "post_separation_max_gap_increase", "Largest increase in pair distance\n(disk diameters)", False),
        (fixed_axes[0], "fixed_separation_net_displacement", f"Green displacement after {window:g} time units\n(disk diameters)", True),
        (fixed_axes[1], "fixed_separation_gap_change", f"Change in pair distance after {window:g} time units\n(disk diameters)", True),
    ]
    taus = np.array([entry[0] for entry in summaries])
    positive = taus[taus > 0]
    for ax, metric, ylabel, control in panels:
        mean = np.array([entry[1][metric] for entry in summaries])
        sem = np.array([entry[2][metric] for entry in summaries])
        valid = np.isfinite(mean)
        if valid.any():
            ax.plot(taus, mean, "o-", color="#0072B2", markersize=4, linewidth=1.6)
            ax.errorbar(taus[valid], mean[valid], yerr=np.nan_to_num(sem[valid], nan=0.),
                        fmt="none", color="#0072B2", capsize=3)
        else:
            message = "No observed separations" if metric == "post_separation_active_fraction" else "No completed intervals"
            ax.text(.5, .5, message, ha="center", transform=ax.transAxes, color="#64748b")
        if control:
            baseline = mean[taus == 0]
            if len(baseline) and np.isfinite(baseline[0]):
                ax.axhline(baseline[0], color="#475569", linestyle="--", label="τ=0, same elapsed time")
                ax.legend(frameon=False, fontsize=8)
        if args.xscale == "symlog":
            ax.set_xscale("symlog", linthresh=positive.min() if len(positive) else .0001)
        ax.set_xlim(left=0)
        style(ax, "Contact-memory time, τ", ylabel)
        if "gap_change" in metric:
            ax.axhline(0, color="#94a3b8", linewidth=.7)
        else:
            ax.set_ylim(bottom=0)
    memory_axes[0].set_ylim(0, 1)
    memory_axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    for fig, name in [(memory_fig, "post_separation_memory"), (fixed_fig, "post_separation_fixed_time")]:
        fig.savefig(output / f"{name}{suffix}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    with (output / f"post_separation_means{suffix}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tau", "metric", "mean", "sem", "runs", "separations", "finished_excursions",
                         "censored_excursions", "complete_fixed_windows", "censored_fixed_windows"])
        writer.writerows(rows)
    print(f"Saved post-separation plots and CSVs in {output}. Error bars: ±1 SEM across independent runs.")


if __name__ == "__main__":
    main()
