# Two-cell contact memory

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

Measurements are optional and leave trajectories unchanged. They resolve every
integration step and save compact per-seed files, with resume support. The three
plots show contact survival, return survival, and distance changes after activation
split by initial heading; tau=0 uses contact entries as its reference events.
See [measurement definitions and options](docs/contact_statistics.md).
