"""Plot sweep or time-series data. This script never runs simulations."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from engine import ROOT, METRICS, FINAL_METRICS, mean_sem, snapshot_demixing, figure_directory


LABELS = {
    "demixing_index": "Demixing index, ⟨M⟩",
    "demixing_index_green": "Green demixing index, ⟨M⟩",
    "demixing_index_red": "Red demixing index, ⟨M⟩",
    "contact_fraction": "Green particles in unlike contact (fraction)",
    "red_contact_fraction": "Red particles in unlike contact (fraction)",
    "persistent_fraction": "Green particles with active memory (fraction)",
    "neighbor_coverage": "Particles with a neighbor (fraction)",
    "neighbor_coverage_green": "Green particles with a neighbor (fraction)",
    "neighbor_coverage_red": "Red particles with a neighbor (fraction)",
}


def selected(meta, sizes, taus):
    return ((not sizes or int(meta["N"]) in sizes)
            and (not taus or np.isclose(float(meta["tau"]),taus,rtol=1e-10,atol=0).any()))


def point(meta, metric, mean, sem, count, total):
    return dict(N=int(meta["N"]), D=float(meta.get("D",-1)),
                packing_fraction=float(meta.get("packing_fraction",meta.get("settings",{}).get("packing_fraction",-1))),
                tau=float(meta["tau"]), metric=metric, mean=float(mean), sem=float(sem),
                runs=int(count), total_runs=total)


def load_sweep(folder, metric, sizes, taus):
    points = []
    for directory in sorted(folder.glob("N*")):
        manifest = directory / "settings.json"
        if not manifest.is_file():
            continue
        meta = json.loads(manifest.read_text())
        if not selected(meta,sizes,taus):
            continue
        values = []
        for path in sorted(directory.glob("sample_*.npz")):
            with np.load(path) as data:
                names = tuple(data["metric_names"])
                if metric in FINAL_METRICS and "final_metrics" in data:
                    values.append(float(data["final_metrics"][tuple(data["final_metric_names"]).index(metric)]))
                elif metric in FINAL_METRICS and "final_positions" in data:
                    # The archived arena files use ordinary Euclidean distances.
                    values.append(float(snapshot_demixing(data["final_positions"])[FINAL_METRICS.index(metric)]))
                elif metric in names:
                    values.append(float(data["metrics"][names.index(metric)]))
                else:
                    raise ValueError(f"{metric} is not available in {path}")
        if values:
            mean, sem, count = mean_sem(values)
            points.append(point(meta,metric,mean,sem,count,len(values)))
    if points:
        return points
    # Earlier raw files may be archived; the exported means still support plotting.
    for filename in ["demixing_means.csv","paper_means.csv","size_scaling_means.csv"]:
        path = folder / filename
        if path.is_file():
            with path.open() as file:
                for row in csv.DictReader(file):
                    if row["metric"] == metric and selected(row,sizes,taus):
                        points.append(point(row,metric,float(row["mean"]),float(row["sem"]),int(row["runs"]),int(row.get("total_runs",row["runs"]))))
            if points:
                return points
    raise ValueError(f"No sweep results for {metric} in {folder}")


def load_time_series(folder, metric, sizes, taus):
    if metric not in FINAL_METRICS:
        raise ValueError("Time series contain demixing and neighbor coverage, not contact-window statistics")
    curves, rows = [], []
    for directory in sorted(folder.glob("N*")):
        manifest = directory / "settings.json"
        if not manifest.is_file():
            continue
        meta = json.loads(manifest.read_text())
        if not selected(meta,sizes,taus):
            continue
        times, values = None, []
        for path in sorted(directory.glob("sample_*.npz")):
            with np.load(path) as data:
                if "times" not in data or "demixing" not in data:
                    raise ValueError(f"No intermediate demixing in {path}; record with run.py --time-points")
                if times is None:
                    times = data["times"].copy()
                elif not np.array_equal(times,data["times"]):
                    raise ValueError(f"Different recording times in {path}")
                values.append(data["demixing"][:,tuple(data["metric_names"]).index(metric)])
        if values:
            mean, sem, count = mean_sem(values)
            header = point(meta,metric,0.,0.,0,len(values))
            curves.append((header,times,mean,sem))
            rows.extend({**header,"time":float(t),"mean":float(m),"sem":float(s),"runs":int(n)}
                        for t,m,s,n in zip(times,mean,sem,count))
    if not curves:
        raise ValueError(f"No selected time series in {folder}")
    return curves, rows


def style(ax, xlabel, metric):
    ax.set(xlabel=xlabel,ylabel=LABELS[metric],ylim=(0,1))
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.2)
    if metric.startswith("demixing_index"):
        ax.axhline(.5,color="#64748b",lw=.8,ls="--")


def group_key(row):
    return row["N"], row["D"], row["packing_fraction"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--metric",choices=LABELS,default="demixing_index")
    parser.add_argument("--sizes",type=int,nargs="+")
    parser.add_argument("--tau",type=float,nargs="+")
    parser.add_argument("--time-series",action="store_true",help="Normally detected from the data manifest")
    parser.add_argument("--xscale",choices=["symlog","linear"],default="symlog")
    args = parser.parse_args()
    try:
        manifest = args.data_dir / "settings.json"
        settings = json.loads(manifest.read_text()) if manifest.is_file() else {}
        time_series = args.time_series or "recorder" in settings
        output = args.output or (figure_directory(settings["boundary"],time_series) if "boundary" in settings else ROOT/"figures/legacy")
        output.mkdir(parents=True,exist_ok=True)
        if time_series:
            curves, rows = load_time_series(args.data_dir,args.metric,args.sizes,args.tau)
            for key in sorted({group_key(p[0]) for p in curves}):
                fig, ax = plt.subplots(figsize=(7,4.8),layout="constrained")
                group = sorted([p for p in curves if group_key(p[0])==key],key=lambda p:p[0]["tau"])
                for meta,times,mean,sem in group:
                    ax.errorbar(times,mean,yerr=sem,fmt="o-",capsize=3,ms=4,lw=1.5,label=f"τ={meta['tau']:g}")
                style(ax,"Time (paper units)",args.metric)
                ax.set_xlim(0,max(p[1][-1] for p in group))
                ax.legend(title=f"N={key[0]}, D={key[1]:g}",frameon=False,fontsize=9)
                path = output / f"{args.metric}_time_N{key[0]}_D{key[1]:g}.png"
                fig.savefig(path,dpi=200,bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {path}")
        else:
            rows = load_sweep(args.data_dir,args.metric,args.sizes,args.tau)
            fig, ax = plt.subplots(figsize=(7,4.8),layout="constrained")
            groups = sorted({group_key(row) for row in rows})
            colors = plt.get_cmap("viridis")(np.linspace(.1,.85,len(groups)))
            for key,color in zip(groups,colors):
                group = sorted([r for r in rows if group_key(r)==key],key=lambda r:r["tau"])
                x,y,e = [np.array([r[name] for r in group]) for name in ["tau","mean","sem"]]
                label = f"N={key[0]}"
                if len({r['D'] for r in rows})>1: label += f", D={key[1]:g}"
                if len({r['packing_fraction'] for r in rows})>1: label += f", φ={key[2]:g}"
                ax.plot(x,y,"o-",color=color,label=label,lw=1.7,ms=4)
                ax.fill_between(x,y-e,y+e,color=color,alpha=.14,lw=0)
            taus = sorted({r["tau"] for r in rows})
            positive = [t for t in taus if t>0]
            if args.xscale=="symlog": ax.set_xscale("symlog",linthresh=min(positive) if positive else 1e-5,linscale=.8)
            ax.set_xticks(taus,labels=[f"{t:g}" for t in taus],rotation=45,ha="right")
            ax.set_xlim(left=0)
            units = "paper units" if settings.get("units")=="paper" or (args.data_dir/"paper_means.csv").is_file() else "legacy units"
            style(ax,f"Contact-memory time, τ ({units})",args.metric)
            ax.legend(frameon=False,fontsize=9)
            path = output / f"{args.metric}_vs_tau.png"
            fig.savefig(path,dpi=200,bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path}")
        with (output/"plot_means.csv").open("w",newline="") as file:
            writer = csv.DictWriter(file,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
        print("Uncertainty: ±1 SEM across seeds. Dashed M=0.5 is the large-N mixing reference.")
    except (ValueError,KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
