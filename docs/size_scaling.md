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

## Demixing index from the paper

```bash
python3 src/plot_size_scaling.py --data-dir data/size_scaling/constant-density --demixing
```

This reads saved final positions; **no simulation needs to be rerun**. Open
`figures/size_scaling/constant-density/demixing_index.png`. The left panel uses
the paper's Eq. (4): for each disk, divide its number of same-type neighbors by
its total neighbors, then average these ratios over non-isolated disks. Neighbors
have center-to-center distance **less than 1.1 disk diameters**, excluding self.
This is a final-configuration measurement, as in Fig. 4, without a chance
correction. The right panel shows the fraction of disks that contribute.

The [paper's Methods](https://doi.org/10.1038/s42005-026-02619-2) specify the
formula, cutoff and final-state measurement. The
[authors' analysis function](https://github.com/mb176/Cell-Tracking-Analysis/blob/main/analysis_library.py#L682)
excludes isolated disks. If an entire population has no neighbors, its index is
undefined (`nan`), not zero or one.

`demixing_samples.csv` saves M and neighbor coverage for all disks, green disks,
and red disks separately, per seed, along with final time. `demixing_means.csv`
contains their means, SEM and numbers of contributing runs. Each seed has equal
weight; runs with an undefined index do not contribute to that index's mean.
Bands show ±1 SEM. Around 0.5 is the large-population random-mixing reference;
1 means all observed neighbors have the same type. The pooled index weights each
eligible disk equally, rather than averaging the two population means.

The older `local_demixing.png` is also retained. It uses the time-averaged index
over non-isolated particle-time observations, with a chance correction.

Small balanced populations have a random-label baseline of `(N/2-1)/(N-1)`,
not 0.5. The plot subtracts that baseline and divides by one minus the baseline:
zero means random-label preference, positive favors same-type neighbors.
N=2 is excluded from this preference plot because it has no same-type partners.
Coverage is essential when the systems become dilute. With N=2 there are no
same-type partners, so raw M is zero whenever neighbors exist; it cannot measure
demixing in that case. Matching the statistic alone does not reproduce Fig. 4:
our confinement, density and observation time still differ from the paper, and
the final configuration is not automatically a stationary state.

## Visualize a run

```bash
.venv/bin/python src/make_many_gif.py --N 32 --L 48 --tau 0.001
```

This reruns one seed with the size-study model and saves a GIF, a final-frame
PNG, and parameter JSON in `figures/size_scaling/`. There are 16 green and 16 red
disks; green turns purple while memory is active. Arrows follow displacement
between displayed frames, enlarged for visibility and capped at 0.65 disk
diameters. They show frame-averaged motion rather than instantaneous headings.

Defaults record every 50,000 integration steps (0.5 time units), display at most
300 frames, and omit warm-up. Use `--save-every`, `--max-frames`, `--fps`,
`--include-warmup` or `--output` to adjust the export. `--seed`, `--duration`,
`--dt` and `--steady-state-steps` select the run. No contact is forced or required.
For the density-controlled geometry, omit `--L` and use `--scaling constant-density`
(N=32 gives L=12). Existing sweep files contain no full trajectories, so this
exporter simulates a new trajectory rather than loading a sweep sample.

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
