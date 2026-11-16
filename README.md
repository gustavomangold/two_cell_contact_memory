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

The plotter defaults to `demixing_index_vs_tau.png` under the corresponding
`figures/` folder, plus `plot_means.csv`. Select another observable with
`--metric contact_fraction`, `persistent_fraction`, or `neighbor_coverage`.
`--sizes`, `--tau`, `--xscale linear` and `--output` control selection/display.
Legacy exported means can also be read, for example:

```bash
.venv/bin/python src/plot.py --data-dir figures/size_scaling/constant-density \
  --output figures/legacy_review
```

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
