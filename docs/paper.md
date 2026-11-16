# Exactly-like-paper branch

The new entry points are `src/paper_sweep.py` and `src/paper_plot.py`.
They use the original C implementation published with
[the paper](https://doi.org/10.1038/s42005-026-02619-2), pinned at
[`fa2989f`](https://github.com/mb176/cell_simulation/tree/fa2989f7841ea4c8aea0d5691ed0710ee9432da4).
The unchanged sources are in `src/paper/upstream/`, with hashes in
`src/paper/provenance.json`.

## What matches

- Periodic square, including minimum-image interactions and the original box-size calculation.
- Original harmonic force and shared-contact bookkeeping; original linked-cell
  pair order in boxes with at least three spatial bins per side (see below).
- Original position, angular-noise, boundary and memory update order.
- Green-only activation; active timers are not refreshed; no reorientation or adhesion.
- Original xoshiro256** noise and Box–Muller transform, random positions followed
  by the original GSL Fletcher–Reeves minimization, and original random headings.
- Paper units: Pe=k=1000, translational diffusion 1, D=360000, activated D+=0,
  packing fraction 0.4, dt=1e-5 and final time 5.

Python accepts the **diffusion coefficient** D; it supplies sqrt(D) to the C
engine, as required by the authors' noise-amplitude convention. The branch's tau
values are the paper's values: **old tau=1 corresponds to new tau=0.001**.
The default tau grid is `0, 1e-5, 1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2`.
This is an explicit grid around the expected optimum, not a claim that every
sampled coordinate in Fig. 4 has been reconstructed. `--D` can sweep additional
diffusion coefficients for the other axis of Fig. 4a.

Original discrete edge cases are retained, including the one-step effect at
tau=0. That point is **not a strict no-memory control**. The original loop is
inclusive (`0..stepLimit`): its final label is 5 while 500001 updates advance
5.00001 time units. Both `final_time` and `elapsed_time` are saved. The only
reproducibility change is supplying a seed instead of using the wall clock;
the original uint32 heading seed range is enforced.

## Setup and commands

Use a Python environment with NumPy; Matplotlib is needed only for plotting.
The first run builds the C engine with `cc`, GSL and CBLAS under `.paper-build/`.
Numba is not needed. On a cluster, load its compiler/GSL modules. For GSL
installed under your own prefix, pass `--gsl-prefix /path/to/gsl`; this adds the
include/library paths and a runtime library path. `CC` selects the compiler.
No sudo or automatic package installation is performed by these scripts.

Full paper particle count, 50 independent seeds per point:

```bash
python3 src/paper_sweep.py --samples 50 --workers 10
```

Requested finite-size bridge, keeping the paper's packing fraction fixed:

```bash
python3 src/paper_sweep.py --sizes 32 64 128 256 512 --samples 50 --workers 10
```

The latter intentionally differs from the paper in particle count. Box side
length is calculated from N and packing fraction, rather than supplied as L.
For N=4,8,16 at packing fraction 0.4, the box has fewer than three spatial bins
per side. Periodic bin offsets then alias in the original linked-cell search.
Those small boxes use an all-pairs neighbor list: each unordered pair appears
once, with the same periodic minimum-image distance, cutoff and rebuild rule.
Only neighbor discovery changes; force, noise, contact and timer functions stay
the original C functions. N>=32 at this density keeps the original linked-cell
algorithm and pair order, with bit-for-bit regression checks. The selected
`neighbor_method` is saved in each point's settings and per-seed NPZ file.

The small-box extension evaluates each pair once. The upstream list retained for
larger boxes includes duplicate entries for particles in the same spatial bin;
its original multiplicity is deliberately preserved for literal reproducibility.
Thus small and large systems use different pair-list conventions, even though
they call the same original force and state-update functions. This distinction
must be considered when interpreting a size comparison.

```bash
python3 src/paper_sweep.py --sizes 4 8 16 32 64 128 256 512 --samples 50 --workers 10
```

Short validation run, kept apart from production data:

```bash
python3 src/paper_sweep.py --sizes 32 --tau 0 0.001 --samples 2 --workers 2 \
  --duration 0.002 --output data/exactly-like-paper/validation-small-boxes
python3 src/paper_plot.py --data-dir data/exactly-like-paper/validation-small-boxes \
  --output figures/exactly-like-paper/validation-small-boxes
```

This short run checks execution, not steady state. Use `--tau`, repeated
`--tau-range START STOP STEP`, `--D`, `--packing-fraction`, `--dt`, `--duration`,
`--samples`, `--workers` or `--seed` to choose a study. Parameter changes that
alter fixed settings require a separate `--output` folder. New N/D/tau values
and extra seeds may be added to an existing sweep. No production sweep is
automatically launched when installing the code.

## Paper dynamics in a circular arena

```bash
python3 src/paper_sweep.py --boundary circular-arena \
  --sizes 2 4 8 16 32 64 128 256 512 --samples 50 --workers 10
python3 src/paper_plot.py --data-dir data/paper_arena/constant-density
```

`sweep_size_scaling.py --scaling constant-density` is a shortcut to this engine.
It now uses Pe=k=1000, D=360000, D+=0, translational diffusion 1, dt=1e-5,
duration 5 and phi=0.4, in **paper units**, with no discarded warm-up. Add
`--tau 0.001 0.01 1` to select those points; otherwise the standard paper grid
is used. N is varied at fixed nominal phi, using the outer circular diameter
`L=sqrt(N/phi)` instead of the square side `sqrt(N*0.785398/phi)`.

Only boundary-dependent code is adapted in a temporary build:

- All pair/contact distances are Euclidean, with no minimum-image wrapping or
  neighboring-bin wrap. Positions are never wrapped, reflected or projected.
- A radial soft wall adds `-k_wall * max(0, r-(L-1)/2) * r_hat`, centered at
  `(L/2,L/2)`. This is the same linear-overlap form as disk repulsion. Wall
  contact does not change memory or orientation directly. The default
  `--wall-stiffness 30000` is the old circular wall's 30 converted to paper
  units; it can be varied independently. Nominal phi uses the outer arena area,
  not the smaller center-accessible area; a soft wall permits penetration.
- Initial centers are sampled uniformly inside the circle of radius `(L-1)/2`,
  then relaxed with the same GSL minimizer, including the wall energy/gradient.

The inherited particle force formula, neighbor-pair convention, RNG draws,
contact bookkeeping, activation/expiry rules, angular update and inclusive
integration loop are retained. In particular the inherited same-bin duplicated
pairs in larger systems and small-box unique-pair fallback are not corrected by
this boundary adapter. The numerical caveats of literal reproduction still apply.
Controlled tests give bitwise-identical interior updates/RNG for the same
configuration/grid when the wall and PBC are inactive, including timer expiry.

The pinned sources and periodic engine hash are unchanged. The arena adapter
has a separate build hash and manifest field. Data is isolated under
`data/paper_arena/constant-density/`; demixing uses Euclidean distances, and the
plotter selects `figures/paper_arena/constant-density/` automatically. Both
`paper_plot.py --data-dir data/paper_arena/constant-density` and the default
`plot_size_scaling.py` command plot final demixing versus tau, one curve per N,
using all saved tau values. The figure is named `demixing_index_vs_tau.png`.
Old data is never converted or resumed as this new model.
`paper_timeseries.py` and `paper_gif.py` still record the periodic paper model;
`make_many_gif.py` still uses the legacy circular kernel.

To retain the old constant-density model and old output location, use
`sweep_size_scaling.py --scaling constant-density --legacy-model`. Its time
units, warm-up, placement, random generator and update order differ from the
paper engine. To use the old *density* with the new dynamics, pass
`--packing-fraction 0.222222222222` (or `--base-L 3` to the size-sweep shortcut).

## Data and plots

`data/exactly-like-paper/` is separate from all circular-model data. Each
N/packing-fraction/D/tau point has a settings manifest, per-seed compressed NPZ
files, small progress logs and `summary.csv`. Completed NPZ files are atomic
and skipped on resume; interrupted seeds restart. Old data is never converted.
The settings and engine source hash are checked before reuse.

NPZ files contain full-precision initial/final positions and headings, final
memory/colors/RNG state, paper parameters and seeds, and contact/persistence
sums over up to 64 log-spaced time windows. No full trajectories are saved.
Contacts mean at least one opposite-type disk within one diameter; each particle
counts once regardless of duplicated neighbor-list entries or multiple contacts.
Statistics use positions/states before each integration update, with no discarded
warm-up. They observe the dynamics without changing it or drawing extra noise.

Final demixing uses Eq. (4), neighbors within 1.1 diameters **with periodic
minimum-image distances**, and excludes isolated cells. All/green/red values and
neighbor coverage are saved. Demixing is a final snapshot, not a time average.

```bash
python3 src/paper_plot.py
```

This saves `figures/exactly-like-paper/paper_comparison.png` and
`paper_means.csv`. The figure shows final demixing with ±1 SEM bands across seeds.
The tau axis includes zero and labels every sampled value on a symmetric-log
scale. Plotting never runs simulations.

### Demixing versus time

Existing sweep files contain only initial/final configurations. Their contact
windows cannot reconstruct intermediate demixing, so record new runs first:

```bash
python3 src/paper_timeseries.py --sizes 32 --tau 0.001 0.01 1 --samples 50 --workers 10
python3 src/paper_plot.py --time-series --tau 0.001 0.01 1
```

The recorder uses the existing observational C backend without changing the
integrator, random draws or production sweep source hash. It saves only the
all/green/red demixing indices and neighbor coverage at 21 evenly spaced times,
including the initial and final states. Change `--time-points` for finer recording,
`--duration` for longer runs, and `--sizes` for other particle counts. Times are in
paper units, rounded to integration steps; the original inclusive loop ends at
`duration + dt` when duration is an exact multiple of dt. No warm-up is discarded.

Data defaults to `data/exactly-like-paper/time_series/`, with atomic per-seed NPZ
files and settings checks. Completed seeds are skipped on resume; sizes, tau
values and sample counts can be extended. Recording requires NumPy, a C compiler
and GSL, but no Matplotlib. It never launches plotting.

The plotting command saves one `paper_demixing_time_N<N>_D<D>.png` per N/D pair
and `paper_demixing_time.csv` in `figures/exactly-like-paper/time_series/`.
Each curve is one tau, with mean demixing and ±1 SEM error bars across independent
seeds at each time. Isolated particles are excluded from M; undefined sample
values are excluded pointwise, and the CSV reports the valid sample count.
The dashed 0.5 line is the large-population random-mixing reference, not an added
control simulation. Select custom folders using the recorder's `--output` and
the plotter's `--data-dir` / `--output`.

## GIFs of the periodic paper model

Use your existing Python environment with NumPy, Matplotlib and Pillow:

```bash
python3 src/paper_gif.py --N 4 --tau 0.001
```

This uses the paper C engine, including the small-box fallback, with the same
compiler/GSL requirement as the sweep. The earlier `make_gif.py` and
`make_many_gif.py` run the old circular model. A virtual environment is supported;
run its Python executable instead of `python3`. No Numba is needed for this GIF.

Default output is `figures/exactly-like-paper/disks_N4_tau0.001.gif`, plus a PNG
snapshot, parameter JSON and simulation progress log. No sweep data or settings
are changed. It reruns one seed because compact sweep files contain no trajectory.

Defaults simulate to paper time 5 and display the final 0.05 time units, recording
every 50 integration steps and limiting output to 300 frames. `--start` selects
the first displayed time; `--duration` selects the final simulation time. A quick
initial-transient preview is:

```bash
python3 src/paper_gif.py --N 4 --tau 0.001 --duration 0.02 --start 0
```

`--N`, `--D`, `--packing-fraction`, `--dt`, `--seed`, `--save-every`, `--fps`,
`--max-frames`, `--output` and `--gsl-prefix` are also available. Units are the
paper's units, including tau. Purple means active green memory. Periodic copies
show disks crossing the square's edges. Arrows follow mean displacement between
displayed frames, using unwrapped coordinates to handle boundary crossings;
their lengths are normalized by Pe and capped for display.

The GIF-only executable observes displacement before the original boundary wrap.
It makes no extra RNG draws and does not modify the sweep engine's tracked
sources or source hash. Regression tests verify identical final positions,
headings, timers, colors and RNG state with and without recording, and check
unwrapped displacement over multiple boundary crossings.

## Validation

`python3 -m unittest tests.test_paper_engine tests.test_demixing` verifies source
hashes, coefficient conversion and periodic demixing. A regression test compiles
the original executable and compares initial states, final positions, headings,
timers, colors and RNG state **bit for bit** against the compact driver for
N=32,64,128 and several tau values, including zero and timer expiry. Tests also
check unique pair discovery, minimum-image forces (including a pair across the
periodic boundary), and successful runs for N=2,4,8,16. Small boxes extend the
paper's model rather than literally executing its large-system neighbor search.
GSL/compiler are needed for these checks. Exact archived figure images or the authors'
unpublished seeds are not reproduced; the published engine and stated protocol are.
