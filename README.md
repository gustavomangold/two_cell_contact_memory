# Contact-memory particles

One paper C engine, two explicit boundaries: `arena` (soft circular wall) and
`pbc` (periodic square). **All active commands use paper units.** The old
NumPy/Numba model and previous scripts are archived, not mixed into this engine.

| File | What to review |
| --- | --- |
| [src/engine.py](src/engine.py) | Defaults, measurements, C build, saved states and compatibility checks |
| [src/run.py](src/run.py) | Sweep arguments, independent seeds, parallel execution and resume |
| [src/plot.py](src/plot.py) | Demixing/contact plots and time-series plots |
| [src/make_gif.py](src/make_gif.py) | Arena/PBC animation and displacement-velocity arrows |

The C code is grouped under `src/native/`. For the **particle dynamics**, start
with `SingleStep`, `ComputeInteractions` and `UpdateInternalStates` in
[cell_simulation.c](src/native/paper/upstream/cell_simulation.c).
[arena.c](src/native/arena.c) supplies the wall force;
[movie.c](src/native/movie.c) only records motion. The pinned author sources
and their provenance remain intact under `native/paper/`.

Defaults: Pe=k=1000, D=360000, D+=0, phi=0.4, dt=1e-5, duration=5.
Lengths are disk diameters; old rescaled tau=1 corresponds to paper tau=0.001.
Arena diameter is `sqrt(N/phi)`; square side is `sqrt(N*0.785398/phi)`.
The original inclusive loop advances `round(duration/dt)+1` steps.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Simulation needs **NumPy, a C compiler and GSL/CBLAS development files**.
Matplotlib is needed only for plots; Pillow is also needed for GIFs. Numba is no
longer required. On the cluster, load its compiler/GSL modules or supply
`--gsl-prefix /path/to/your/gsl`. `CC` selects the compiler.

## Sweep and plot

```bash
.venv/bin/python src/run.py --boundary arena \
  --sizes 2 4 8 16 32 64 128 256 512 1024 2048 --samples 50 --workers 10
.venv/bin/python src/plot.py --data-dir data/paper_arena/constant-density
```

This plots final demixing versus tau, one curve per N. The default tau grid is
`0, 1e-5, 1e-4, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01`.
Use `--tau ...` or `--tau-range START STOP STEP` to choose another grid.
Use `--boundary pbc` for the square; its default data folder is
`data/exactly-like-paper/`. Both modes keep nominal phi fixed as N changes.

Samples run in parallel and are saved atomically; completed seeds are skipped.
`--dry-run` checks settings and lists missing samples without writing or running.
Changed physics is rejected in an existing folder: choose another `--output`.
Old C sweep/time-series data can resume when its generated C program and settings
match the compatibility fingerprints. The cleanup does not convert existing data.

The plotter saves `demixing_index_vs_tau_<parameters>.png` under the corresponding
`figures/` folder, plus `plot_means.csv`. Select another observable with
`--metric contact_fraction`, `persistent_fraction`, or `neighbor_coverage`.
`--sizes`, `--tau`, `--xscale linear` and `--output` control selection/display.
Legacy exported means can also be read, for example:

```bash
.venv/bin/python src/plot.py --data-dir figures/size_scaling/constant-density \
  --output figures/legacy_review
```

### Matched arena and PBC sweeps at phi=2/9

Run on the remote machine using bash. This uses 11 sizes, the 9 default tau values,
50 seeds per point, and paper dt=1e-5/duration=5 for both boundaries:

```bash
for boundary in arena pbc; do
  .venv/bin/python src/run.py --boundary "$boundary" \
    --sizes 2 4 8 16 32 64 128 256 512 1024 2048 \
    --packing-fraction 0.2222222222222222 \
    --samples 50 --workers 10 --output "data/phi_2over9/$boundary"
done
```

Copy `data/phi_2over9/` locally, then plot (Matplotlib required):

```bash
for boundary in arena pbc; do
  .venv/bin/python src/plot.py --data-dir "data/phi_2over9/$boundary" \
    --output "figures/phi_2over9/$boundary"
done
```

Each figure is `demixing_index_vs_tau_<parameters>.png`, with one curve per N and ±1 SEM
bands. Explicit data/figure folders keep this comparison separate from phi=0.4.
The filename includes the boundary, phi, D, N range, D-plus, Pe, k, timestep,
duration and (for the arena) wall stiffness. Time-series filenames use the same
parameter suffix. Output folders and the plotted content are unchanged.
Append `--dry-run` to the runner command to check before launching.

## Timestep convergence

The arena timestep study keeps phi=2/9, D=360000, D+=0 and physical duration
0.12 fixed (the old size-scaling snapshot time, converted to paper units).
It tests dt=`1e-5, 2e-6, 5e-7, 1e-7, 2e-8, 1e-8` at
tau=`0, 5e-7, 1e-6, 1e-5, 1e-4, 1e-3, 2e-3`, for N=2 through 512,
doubling N. Each timestep has its own resumable folder; simulation dynamics
are unchanged. The inherited final time is duration+dt.

Start with a small pilot, then increase sizes and seeds in the same folders:

```bash
bash scripts/sweep_timestep.sh --sizes 32 --samples 10 --workers 10
bash scripts/sweep_timestep.sh --samples 50 --workers 10
```

The finest timestep takes 12 million steps per sample. To run only selected
timesteps, set `TIMESTEPS="1e-5 2e-6 5e-7"` before the command. `PYTHON` selects
another interpreter (default `.venv/bin/python`). `DURATION` changes the fixed
duration and its default output path; `OUTPUT_ROOT` selects a different data root.
Use `DURATION=5` for a separate convergence study at the current sweep's final time.
All times above are paper units; old tau=0.0005 corresponds to paper tau=5e-7.

Plot separately, locally if the cluster has no Matplotlib:

```bash
.venv/bin/python src/plot.py --timestep-sweep \
  --data-dir data/timestep_sweep/arena_phi2over9_T0.12 \
  --output figures/timestep_sweep/arena_phi2over9_T0.12
```

The figure has one panel per N and one curve per dt, with SEM bands; its CSV
includes dt for every point. Repeat with `--metric neighbor_coverage` or
`--metric contact_fraction` to check who contributes to M and time spent in contact.
Convergence means successive refinements agree within uncertainty across tau,
without a systematic drift. If the low-tau dip returns as dt decreases, the
timestep contributes to its disappearance. If it does not, investigate the
other changed contact rules and initialization. This is a finite-time test,
not a stationarity check; N=2 cannot diagnose demixing and always has M=0
when defined. Do not pool old and current-engine samples.

## Demixing versus time

```bash
.venv/bin/python src/run.py --boundary arena --sizes 32 --tau 0.001 0.01 1 \
  --time-points 21 --samples 50 --workers 10
.venv/bin/python src/plot.py --data-dir data/paper_arena/constant-density/time_series
```

The same runner records compact demixing/coverage snapshots, including initial
and final states. The plotter detects time-series data automatically. For PBC,
use `--boundary pbc` and `data/exactly-like-paper/time_series`.

## GIF

```bash
.venv/bin/python src/make_gif.py --boundary arena --N 32 --tau 0.001
```

The GIF uses the same engine and parameters as the runner. It displays the final
0.05 time units by default. Use `--start 0 --duration 0.02` for a short initial
transient, and `--save-every`, `--max-frames`, `--fps` to control recording.
Purple means active memory; arrows show mean unwrapped velocity between frames.

## Model status, archive and checks

This cleanup preserves current dynamics, including the inherited pair-counting,
shared-contact and tau=0 behavior described in the
[mechanism audit](docs/mechanism_audit_2026-11-16/review.md).
It does not fix those issues. For a genuine label-independent control, set
`--D-plus 360000` in a separate output folder. Plot bands/error bars are SEM;
the M=0.5 line is only the large-N mixing reference. Isolated particles are
excluded, so inspect neighbor coverage alongside M.

Previous source, tests and usage guides are preserved in
`archive/code_before_cleanup_2026-11-16.tar.gz`, with a SHA256 manifest. Research
reports, existing data, figures and data zip archives remain in place. Historical
analysis scripts are included in that code archive.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The single test suite compares endpoint states/RNG against the archived engine,
checks old/new resume, and exercises both boundaries, plots, time series and GIFs.
