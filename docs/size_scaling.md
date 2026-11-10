# Does the contact-time peak survive with more particles?

This study keeps the same circular arena, disk diameter, speed, translational
noise, spring forces, and baseline/activated angular diffusion as `two_disks.py`.
There are equal green/red populations. Reorientation is off. Any red contact can
activate a green's expired timer; an active timer is not refreshed. Each green
has one timer, even if it touches several red particles. Same-type pairs repel
with the same spring force but do not activate memory.

## The requested sweep

```bash
.venv/bin/python src/sweep_size_scaling.py --samples 50 --workers 10
.venv/bin/python src/plot_size_scaling.py
```

The default geometry is:

| Particles N | Green + red | Arena diameter L | Nominal packing fraction N/L² |
| ---: | ---: | ---: | ---: |
| 2 | 1 + 1 | 3 | 0.2222 |
| 4 | 2 + 2 | 6 | 0.1111 |
| 8 | 4 + 4 | 12 | 0.0556 |
| 16 | 8 + 8 | 24 | 0.0278 |
| 32 | 16 + 16 | 48 | 0.0139 |

Tau defaults to `0, 0.0005, 0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5`, matching
the recent contact-time study. That is 50 parameter points, each with 50 runs.
Use `--sizes`, `--tau` or repeated `--tau-range START STOP STEP` to select a
smaller pilot or refine the peak. Samples run in parallel on one node. Use
`--workers` equal to the CPU allocation; Numba is used automatically if installed.

The equations of the existing two-disk simulation are unchanged. For N=2 the new
kernel preserves the original positions, headings, noise sequence, and contact
fraction for identical parameters/seeds. Larger systems start uniformly inside
the permitted circle with no initial overlaps and random headings. Positions are
shuffled before assigning colors, avoiding placement-order bias. The same
seed reuses the same initial configuration across tau values for each size.

## Read the contact plot

Open `figures/size_scaling/diameter/contact_fraction.png`.

- **Left:** percentage of time a typical green particle touches at least one red
  particle, versus tau. Each curve is one system size. For N=2 this is exactly the
  contact fraction we have been plotting. A green touching multiple reds still
  counts once; "any contact anywhere in the system" would grow with N and would
  not be a comparable statistic.
- **Right:** added contact time relative to that size's own tau=0 baseline,
  in percentage points. Positive values mean memory increases unlike association.
  Differences are calculated using matching seeds before averaging.

Both plots include zero using symlog and show ±1 SEM across independent runs.
Use `--xscale linear` to expand the larger tau values. The exported
`size_scaling_means.csv` includes density, sample counts, and how many runs
observed unlike contact. A zero curve with few encounters is not evidence that a
peak has disappeared; it may indicate insufficient sampling.

## Separate system-size effects from density effects

**The requested doubling halves density each time.** It tests particle number,
confinement, and dilution together. To test size while keeping density fixed:

```bash
.venv/bin/python src/sweep_size_scaling.py --scaling constant-density --samples 50 --workers 10
.venv/bin/python src/plot_size_scaling.py --data-dir data/size_scaling/constant-density
```

This uses L=3*sqrt(N/2): `3, 4.2426, 6, 8.4853, 12`, keeping N/L²=2/9.
It has its own data folder. `--base-L` changes the N=2 arena in either protocol.
Use the same physical parameters and time settings when comparing protocols.

## Check whether recording has settled

Default warm-up is 2,000,000 steps (20 time units at DT=0.00001), followed by
100 time units of observation. This is not a guarantee of stationarity, especially
as arena size and encounter times increase. Inspect the saved window means:

```bash
.venv/bin/python src/plot_size_scaling.py --transient-tau 0.001
```

This adds `contact_fraction_time_tau0.001.png`: contact fractions in successive
log-spaced recording windows, after warm-up. Continued drift means longer warm-up
or observation is needed. Change these with `--steady-state-steps` and `--duration`
in a new output folder. Window means do not reconstruct the discarded warm-up.

## Optional connection to demixing

`--demixing` adds a local same-type preference plot and the fraction of particles
with neighbors. Neighbors are within 1.1 disk diameters. The raw index is the mean
same-type neighbor fraction over non-isolated particle-time observations; isolated
particles are undefined, rather than counted as perfectly sorted.

Small balanced populations have a random-label baseline of `(N/2-1)/(N-1)`,
not 0.5. The plot subtracts that baseline and divides by one minus the baseline:
zero means random-label preference, positive favors same-type neighbors.
N=2 is excluded from this preference plot because it has no same-type partners.
Coverage is essential when the systems become dilute. This local time average
is a diagnostic, not a reproduction of the paper's final-state many-cell heatmap.

## Files and resuming

Data is stored under `data/size_scaling/diameter/` or `constant-density/`.
Each `(N,L,tau)` folder has `settings.json`, `summary.csv` (one row per seed), and
atomic compressed `sample_00001.npz` files. Rerun the same command to resume;
increase `--samples` to add runs, or extend the size/tau grid. Changed fixed settings
require another `--output` directory. Plotting never starts simulations.

NPZ files store scalar metrics, raw sums over up to 64 log-spaced time windows,
their integration-step boundaries, and initial/final configurations. They do not
store full trajectories. Counts weight the window means correctly. Final positions
are retained for later spatial analyses. The neighbor list is rebuilt whenever
particles move enough to invalidate it; it preserves all contact forces.
