"""Plot contact/return survival and activation-conditioned motion; requires Matplotlib."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from contact_stats import EVENT_COLUMNS, TRACE_COLUMNS, RESPONSE_NAMES, ROOT


def survival(events, times):
    """Kaplan-Meier survival: exclude unknown-age episodes; retain right censoring."""
    events = events[(events[:, 4] == 0) & (events[:, 3] >= 0)]
    if not len(events):
        return np.full(len(times), np.nan)
    durations, inverse, removed = np.unique(events[:, 3], return_inverse=True, return_counts=True)
    deaths = np.bincount(inverse, weights=1 - events[:, 5], minlength=len(durations))
    at_risk = len(events) - np.r_[0, np.cumsum(removed[:-1])]
    values = np.cumprod(1 - deaths / at_risk)
    index = np.searchsorted(durations, times, side="right") - 1
    result = np.where(index < 0, 1., values[np.maximum(index, 0)])
    result[times > durations[-1]] = np.nan  # No extrapolation beyond observed follow-up.
    return result


def mean_sem(values):
    """Equal weight for independent seeds, not for correlated events within a run."""
    values = np.asarray(values)
    valid = np.isfinite(values)
    n = valid.sum(axis=0)
    mean = np.divide(np.nansum(values, axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)
    squares = np.where(valid, (values - mean) ** 2, 0).sum(axis=0)
    sem = np.sqrt(np.divide(squares, n * (n - 1), out=np.full(n.shape, np.nan), where=n > 1))
    return mean, sem, n


def response_curve(response, heading, wall):
    selected = response[heading]
    totals = selected.sum(axis=0) if wall == "all" else selected[0 if wall == "away" else 1]
    counts = totals[:, 0]
    means = np.divide(totals[:, 1:], counts[:, None],
                      out=np.full(totals[:, 1:].shape, np.nan), where=counts[:, None] > 0)
    return means, counts


def load_cases(root, length):
    cases = []
    for folder in root.glob(f"L{length:g}_tau*"):
        manifest = folder / "settings.json"
        if not manifest.exists():
            continue
        metadata = json.loads(manifest.read_text())
        runs = []
        for path in sorted(folder.glob("sample_*.npz")):
            with np.load(path) as data:
                runs.append({name: data[name].copy() for name in data.files})
        if runs:
            cases.append((metadata["settings"]["PERSISTENCE_TIME"], metadata, runs, folder))
    if not cases:
        raise ValueError(f"No completed diagnostic samples for L={length:g} in {root}")
    return sorted(cases, key=lambda case: case[0])


def style(ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "contact_stats" / "no_reorient")
    parser.add_argument("--L", type=float, default=3.)
    parser.add_argument("--wall", choices=["all", "near", "away"], default="all",
                        help="Location at activation/contact entry for motion curves")
    parser.add_argument("--observable", choices=RESPONSE_NAMES[:3], default="delta_distance")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--export-events", action="store_true", help="Also export per-seed events and sparse traces as CSV")
    args = parser.parse_args()
    try:
        cases = load_cases(args.data_dir, args.L)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or ROOT / "figures" / "contact_stats" / args.data_dir.name / f"L{args.L:g}"
    output.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("viridis")(np.linspace(.15, .85, len(cases)))
    headings = ["Inward", "Tangential", "Outward"]
    contact_fig, contact_ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
    return_fig, return_axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, layout="constrained")
    motion_fig, motion_axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True, layout="constrained")
    curve_rows, motion_rows, metric_rows = [], [], []
    for (tau, metadata, runs, folder), color in zip(cases, colors):
        dt = metadata["settings"]["DT"]
        events = [run["events"] for run in runs]
        longest = max(array[:, 3].max() for array in events)
        times = np.r_[0, np.geomspace(dt, max(dt, longest), 120)]
        label = f"τ={tau:g}" + (" (contact entries)" if tau == 0 else "")
        line = "--" if tau == 0 else "-"
        groups = [("contact", 0, None, contact_ax), ("return_all", 1, None, return_axes[0]),
                  ("return_persistent", 1, 1, return_axes[1]), ("return_nonpersistent", 1, 0, return_axes[2])]
        for name, kind, persistent, ax in groups:
            curves = []
            for array in events:
                selected = array[array[:, 0] == kind]
                if persistent is not None:
                    selected = selected[selected[:, 6] == persistent]
                curves.append(survival(selected, times))
            mean, sem, n = mean_sem(curves)
            if np.isfinite(mean).any():
                ax.plot(times, mean, color=color, linestyle=line, linewidth=1.5, label=label)
                ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=.14, linewidth=0)
            ax.set_xscale("symlog", linthresh=dt)
            curve_rows.extend([[tau, name, t, m, error, int(count)] for t, m, error, count in zip(times, mean, sem, n)])

        for heading, ax in enumerate(motion_axes):
            means, counts = zip(*(response_curve(run["responses"], heading, args.wall) for run in runs))
            mean, sem, n = mean_sem(means)
            lags = runs[0]["lags"]
            metric = RESPONSE_NAMES.index(args.observable)
            if np.isfinite(mean[:, metric]).any():
                ax.plot(lags, mean[:, metric], color=color, linestyle=line, linewidth=1.5, label=label)
                ax.fill_between(lags, mean[:, metric] - sem[:, metric], mean[:, metric] + sem[:, metric],
                                color=color, alpha=.14, linewidth=0)
            for index, lag in enumerate(lags):
                for column, name in enumerate(RESPONSE_NAMES):
                    motion_rows.append([tau, headings[heading].lower(), args.wall, lag, name,
                                        mean[index, column], sem[index, column], int(n[index, column]),
                                        int(np.sum(counts, axis=0)[index])])

        values = np.stack([run["metrics"] for run in runs])
        mean, sem, n = mean_sem(values)
        metric_rows.extend([[tau, name, m, error, int(count)]
                            for name, m, error, count in zip(runs[0]["metric_names"], mean, sem, n)])
        if args.export_events:
            for name, columns in [("events", EVENT_COLUMNS), ("traces", TRACE_COLUMNS)]:
                with (output / f"tau{tau:g}_{name}.csv").open("w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sample", "seed", *columns])
                    for run in runs:
                        writer.writerows([[int(run["sample"]), int(run["seed"]), *row] for row in run[name]])
        print(f"tau={tau:g}: {len(runs)} independent samples", flush=True)

    style(contact_ax, "Contact duration", "Contact survival, S(t)")
    contact_ax.set(ylim=(0, 1.03), xlim=(0, None))
    contact_ax.legend(frameon=False, fontsize=9)
    for ax, name in zip(return_axes, ["All separations", "Persistent at separation", "Not persistent at separation"]):
        style(ax, "Return time", "Return survival, S(t)")
        ax.set(ylim=(0, 1.03), xlim=(0, None))
        ax.text(.03, .04, name, transform=ax.transAxes, fontsize=9)
        if not ax.get_legend_handles_labels()[0]:
            ax.text(.5, .5, "No eligible episodes", ha="center", transform=ax.transAxes, color="#64748b")
    handles, labels = contact_ax.get_legend_handles_labels()
    if handles:
        return_fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1),
                          ncol=min(4, len(handles)), frameon=False)
        motion_fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, 1),
                          ncol=min(4, len(handles)), frameon=False)
    ylabel = {"green_away": "Green outward displacement", "red_away": "Red outward displacement",
              "delta_distance": "Change in pair distance"}[args.observable]
    for ax, heading in zip(motion_axes, headings):
        style(ax, "Time since activation / contact entry", ylabel)
        if not ax.get_legend_handles_labels()[0]:
            ax.text(.5, .5, "No reference events", ha="center", transform=ax.transAxes, color="#64748b")
        ax.axhline(0, color="#64748b", linewidth=.7)
        ax.text(.03, .95, heading, va="top", transform=ax.transAxes)
    for fig, name in [(contact_fig, "contact_survival"), (return_fig, "return_survival"),
                      (motion_fig, f"{args.observable}_{args.wall}")]:
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    for name, header, rows in [
        ("survival", ["tau", "episode", "time", "mean", "sem", "runs"], curve_rows),
        (f"motion_{args.wall}", ["tau", "heading", "wall", "lag", "metric", "mean", "sem", "runs", "events"], motion_rows),
        ("metrics", ["tau", "metric", "mean", "sem", "runs"], metric_rows),
    ]:
        with (output / f"{name}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    print(f"Saved plots and mean/SEM CSVs in {output}. Bands: ±1 SEM across independent seeds.")


if __name__ == "__main__":
    main()
