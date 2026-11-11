# Two-cell contact memory

On branch **`exactly-like-paper`**, use the paper commands below. They run the
published C engine in a periodic square, in the paper's units. Results are kept
separately in **`data/exactly-like-paper/`** and plots in
**`figures/exactly-like-paper/`**. The circular-model commands later in this README
remain available for the earlier experiments.

```bash
python3 src/paper_sweep.py --samples 50 --workers 10
python3 src/paper_plot.py
```

Defaults match Fig. 4's persistence-only model: N=4000, packing fraction 0.4,
Pe=k=1000, D=360000, D+=0, dt=1e-5, final time 5. To continue the size study:

```bash
python3 src/paper_sweep.py --sizes 32 64 128 256 512 --samples 50 --workers 10
```

This requires NumPy, a C compiler and GSL development files; plotting needs
Matplotlib. Numba is not used by this engine. Runs resume per completed seed.
**Tau now uses paper units: the earlier tau=1 is tau=0.001 here.** See the
[paper implementation guide](docs/paper.md) for the exact source, setup, data
format and a short validation run.

Two active disks in a circular arena, using soft disk and wall forces. Contact
activates memory in disk 0 (green; purple while active). Disk 1 is red.
`L` is the arena **diameter in disk diameters**; `tau` is contact-memory time.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run commands from the repository root, using `.venv/bin/python`. Code lives in
`src/`; `data/` and `figures/` stay at the repository root. Numba is used automatically when
installed; otherwise the same model runs with NumPy. On a cluster used only for
simulation, installing `numpy numba` is sufficient; plotting needs Matplotlib
and GIF export also needs Pillow.

## Scripts

| File | Purpose |
| --- | --- |
| `src/two_disks.py` | Readable model, parameter defaults, single-run histogram/animation |
| `src/_two_disks_numba.py` | Equivalent compiled time loop |
| `src/sweep.py` | Parallel samples, resumable sweeps, compact CSV output |
| `src/plot.py` | Sweep plots, re-orient comparisons, steady-state plots |
| `src/steady_state.py` | Ensemble distance time series starting at step zero |
| `src/make_gif.py` | Contact GIF with displacement-velocity arrows |
| `src/sweep_storage.py` | Log-window storage, resume checks, legacy CSV compaction |
| `src/velocity_arrows.py` | Shared arrow drawing helpers |
| `src/contact_stats.py` | Optional contact, return, activation and wall diagnostics |
| `src/plot_contact_stats.py` | Survival and activation-conditioned motion plots |
| `src/post_separation.py` | Optional post-separation measurement helpers |
| `src/plot_post_separation.py` | Persistent excursions and fixed-time control plots |
| `src/many_disks.py` | Same circular model with equal populations and variable N |
| `src/sweep_size_scaling.py` | N=2,4,8,16,32 contact-time sweep |
| `src/plot_size_scaling.py` | Compare contact-time curves across sizes |
| `src/demixing.py` | Paper's demixing index from saved final configurations |
| `src/make_many_gif.py` | Size-study GIF and snapshot with displacement arrows |

The simulation implementations and their defaults are unchanged. The separate
L/tau sweep wrappers and plotting scripts have been replaced by these commands.
`examples/regenerate_gifs.py` retains the earlier GIF recipes, including the
historical parameter set. Original helper scripts are kept as references in
`archived_runs/legacy_scripts/`. Existing data, figures and archives are preserved.

## Sweeps and plots

```bash
python src/sweep.py --L 2.24 3 4 --tau-range 0 5 0.25 --samples 500 --workers 10
python src/sweep.py --L 3 --tau-range 0 5 0.25 --samples 500 --workers 10 --reorient
python src/plot.py --data-dir data/no_reorient
python src/plot.py --data-dir data/no_reorient --compare data/reorient --L 3 --tau-max 5
```

Repeat `--tau-range` to combine grids, or use `--tau` for explicit values:

```bash
python src/sweep.py --L 3 --tau-range 0 0.5 0.075 --tau-range 0.5 5 0.25
python src/sweep.py --L 3 --tau 0 0.00001 0.0001 --output data/small_tau
```

Ranges include the stop only when it lies on the grid. Rerun the same command to
resume; complete samples are skipped and an interrupted final sample is redone.
Seeds are `--seed` (default 7), then 8, 9, etc.; the same seeds are reused across
parameter points. Increase `--samples` to add runs. Each folder records fixed
settings in `settings.json`; changed settings require another `--output` folder.

Default outputs are `data/no_reorient/` and `data/reorient/`, one CSV per `(L,tau)`.
CSV rows contain means over up to 64 logarithmically spaced recording windows,
with observation counts. Plots weight those counts within each run, then average
the independent run means; bands show ±1 SEM. Incomplete points are skipped.
Plots and summary CSVs go to `figures/`; use `--output figures/name.png` to rename.
Comparisons default to a split view: a narrow `symlog` panel includes tau=0 and
small tau; a wider linear panel expands tau=0.25 onward. Adjust the boundary with
`--split-at`. Both panels share the same y scale and retain every data point.
Use `--xscale symlog` for the previous single-axis view, or `--xscale linear`.
`--xscale log` keeps zero via symlog; `--linear-threshold` overrides the linear
region near zero (default: smallest positive tau).
Legacy raw and compact CSV folders can also be supplied with `--data-dir`.

## Steady state and GIFs

```bash
python src/steady_state.py --L 4 --tau 1 --samples 50 --steps 10000000 --workers 10 --output data/steady_state
python src/plot.py --steady-state data/steady_state
python src/make_gif.py --L 3 --tau 0.001 --contact-window --save-every 2000
python src/make_gif.py --L 3 --tau 2 --reorient --seed 8 --contact-window --save-every 50
```

Steady-state diagnostics save individual traces, ensemble means/SEM and time-window
means; plots show ±2 SEM. GIF export verifies that the selected clip contains
contact; decrease `--save-every` if brief contacts are missed. Arrows follow actual
displacements between recorded frames.

Sweep and GIF commands accept `--dt`, `--duration` and `--steady-state-steps`.
Warm-up time equals `DT * STEADY_STATE_STEPS`: current defaults give **20** time
units (`1e-5 * 2,000,000`); use 20,000,000 steps for a warm-up of 200.

Compact old raw sweep CSVs in place (without rerunning):
`python src/sweep_storage.py path/to/sweep`. Run checks with
`python -m unittest discover`. Each script supports `--help`.

## Contact diagnostics

```bash
python src/contact_stats.py --L 3 --tau 0 0.001 1 2 --samples 50 --workers 10
python src/plot_contact_stats.py --data-dir data/contact_stats/no_reorient --L 3
```

Measurements resolve every integration step and save compact per-seed files with
resume support. **Start with `contact_fraction.png`: it shows the percentage of
time the pair spends in contact versus tau**, with the no-memory baseline marked.
The other plots show how long contacts last, how long returns take, and distance
changes after activation. See [how to read each plot](docs/contact_statistics.md)
for concrete examples and commands.

To measure motion after separation, add `--post-separation-time 1` to the
diagnostic command. New data goes to `data/contact_stats/post_separation/no_reorient/`;
plot it with `python src/plot_post_separation.py`. See the
[post-separation guide](docs/post_separation.md) for examples and definitions.

To compare the contact peak across particle numbers:

```bash
python src/sweep_size_scaling.py --samples 50 --workers 10
python src/plot_size_scaling.py
```

This uses (N,L)=(2,3),(4,6),(8,12),(16,24),(32,48). Doubling diameter halves
density; `--scaling constant-density` provides a separate density-controlled study.
See [the size-scaling guide](docs/size_scaling.md) for plot definitions and options.

Calculate the paper's final-state demixing index from an existing size sweep:

```bash
python3 src/plot_size_scaling.py --data-dir data/size_scaling/constant-density --demixing
```

This also saves `demixing_index.png`, `demixing_means.csv` and
`demixing_samples.csv` in `figures/size_scaling/constant-density/`, including
green/red indices and neighbor coverage in the CSVs. No simulations are rerun.

Visualize one 32-disk run (GIF, final-frame PNG and parameter JSON):

```bash
.venv/bin/python src/make_many_gif.py --N 32 --L 48 --tau 0.001
```

Files go to `figures/size_scaling/`. Purple marks active green memory; arrows
follow displacement between displayed frames. See the guide for recording options.

## No-memory control

```bash
bash src/sweep_no_memory.sh --samples 50 --workers 10
```

This collects the contact diagnostics with tau=0 and reorientation off, for
L=2.24, 3, 4, 6 and baseline angular diffusion Dr=1, 10, 100, 360. Other model
parameters stay fixed. Samples run in parallel and completed runs are resumed.
Results go to `data/contact_stats/no_memory/Dr.../L..._tau0/`. Edit the two arrays
in the script to change the grid; options such as `--duration` and
`--steady-state-steps` pass through. The script uses `.venv/bin/python` when
available, or `SIM_PYTHON` if set. Plot a control point, for example, with
`python src/plot_contact_stats.py --data-dir data/contact_stats/no_memory/Dr360 --L 3`.
