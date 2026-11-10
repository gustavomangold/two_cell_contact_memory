# Recording details and saved files

For explanations of the figures, start with [Reading the contact plots](contact_statistics.md).

## Running and resuming

```bash
python src/contact_stats.py --L 3 --tau 0 0.001 1 2 --samples 50 --workers 10
python src/plot_contact_stats.py --data-dir data/contact_stats/no_reorient --L 3
```

The runner needs NumPy and optionally Numba; plotting also needs Matplotlib.
`--reorient` enables turning and changes the default output folder to
`data/contact_stats/reorient`. Other options include `--angular-diffusion`,
`--tau-range START STOP STEP`, `--seed`, `--dt`, `--duration`,
`--steady-state-steps`, and `--output`. Rerunning skips complete samples;
increasing `--samples` adds independent seeds. Fixed settings must match on resume.

## Measurement rules

The raw contact fraction counts steps with center distance r<1. Contact episodes
start at r<1 and end at r>1+epsilon (default epsilon=0.01), preventing tiny noisy
crossings from splitting one episode repeatedly. This hysteresis changes only
measurement. `hysteresis_contact_fraction` includes these bridged gaps.
`--epsilon 0` removes the gap; use a different output folder to compare epsilons.

Activations count timer changes from zero to a positive value. Renewals are
activations within an existing hysteretic contact. Both disks' wall-force
fractions count penetration with nonzero wall stiffness. Signed surface clearance
is L/2-0.5-|position|; negative means wall penetration. Center-to-wall distance
is clearance+0.5. An event is near the wall if either disk's clearance is at most
`--near-wall` (default 0.1), at its start.

For motion, the reference normal points from red to green at activation. Green's
outward displacement is projected onto this normal; red's onto its negative.
The initial green heading cosine u selects inward (u<-0.5), tangential
(-0.5<=u<=0.5), or outward (u>0.5). With turning enabled this is the heading after
turning, as used by the position update. Wall groups refer to the event start,
not later wall encounters.

States are measured every integration step before the position update, after
the timer/contact response. The observation window starts after warm-up and has
round(duration/dt) steps. Sparse traces include first/last observed states and
up to 64 log-spaced recording times, with actual angles and remaining memory.
`--max-lag` (default 2) sets the motion horizon; `--lag-points` (default 32) and
`--trace-points` (default 64) control saved resolution.

## Censoring and uncertainty

An episode already underway at the observation start has unknown age. Its duration
is flagged as left-censored and excluded from survival estimation. An episode
still underway when recording stops is right-censored: its duration is a known
lower bound, and is retained by the Kaplan–Meier estimator.

Each survival line is one Kaplan–Meier estimate over the eligible recorded
episodes. Its band is one estimated standard error from 200 bootstrap resamples
of whole independent runs, preserving correlations between episodes within a run.
The line remains non-increasing. It is not forced to zero when recording ends.
The far tail has fewer observations; the survival CSV's `runs` column reports
how many runs have episode follow-up extending to that time.

Contact-fraction error bars and motion bands use ±1 SEM across independent run
means. Empty groups remain undefined; a single run has no SEM estimate. Motion
counts decrease at long lags when events occur too close to recording's end.

## Saved files

Each parameter folder contains `settings.json`, per-run scalar `summary.csv`,
and atomic compressed `sample_00001.npz` files. NPZ arrays are:

- `events`: columns listed in `EVENT_COLUMNS` in `src/contact_stats.py`.
  Kind 0 is contact, 1 is return. Both censoring flags are saved.
- `traces`: columns in `TRACE_COLUMNS`, including positions, angles, memory and wall clearance.
- `lags`: response lag times.
- `responses`: axes [heading, wall, lag, statistic]. Heading order is
  inward/tangential/outward; wall order is away/near. Statistic 0 is count;
  remaining values are sums in `RESPONSE_NAMES` order. Divide sums by counts.
- `metrics`, `metric_names`, `sample`, `seed`: per-run scalar results and identifiers.

Plotting exports `metrics.csv` (scalar means/SEM), `survival.csv`, and
`motion_all.csv` (or `motion_near.csv`/`motion_away.csv`). `--export-events` also
writes readable event/trace CSVs. No new simulation is run during plotting.

Opt-in post-separation recordings use schema 2 and a separate output folder.
Their extra arrays and flags are described in the [post-separation guide](post_separation.md).
Schema-1 recordings and the existing four-array simulation return value are preserved.
