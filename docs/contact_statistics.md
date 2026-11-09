# Contact diagnostics

Run from the repository root with the same NumPy/Numba environment as the sweeps.
Matplotlib is needed only for plotting. The default `sweep.py` workflow is unchanged.

```bash
python src/contact_stats.py --L 3 --tau 0 0.001 1 2 --samples 50 --workers 10
python src/plot_contact_stats.py --data-dir data/contact_stats/no_reorient --L 3
```

Use `--reorient` to measure the turning model; it saves to `data/contact_stats/reorient`.
The runner also accepts `--tau-range START STOP STEP`, `--seed`, `--dt`,
`--duration`, `--steady-state-steps` and `--output`. Increasing `--samples` adds
independent seeds; rerunning skips complete files. Fixed settings are recorded
in each parameter folder and must match when resuming. No simulation runs are
started by the plotting command.

## Definitions

- **Contact fraction:** fraction of integration steps with center distance `r < 1`.
- **Contact episodes:** start at `r < 1` and end at `r > 1 + epsilon` (default 0.01).
  This hysteresis affects measurement only. `hysteresis_contact_fraction` includes
  the short gaps bridged by hysteresis; `contact_fraction` uses the strict `r < 1` rule.
- **Return episodes:** time from the end of one hysteretic contact to the next entry.
  Events retain whether green was persistent at separation.
- **Activations:** timer changes from zero to a positive duration. For tau=0 the
  count is zero. Renewals are activations within an existing hysteretic contact,
  distinguished from new encounters.
- **Motion:** at each activation, keep the contact normal pointing from red to
  green. `green_away` projects green displacement onto that normal; `red_away`
  projects red displacement onto its negative. Positive means moving away from
  the original partner. `delta_distance` is the change in center separation.
- **Heading:** initial green propulsion heading dotted with that normal. Groups
  are inward (`u < -0.5`), tangential (`-0.5 <= u <= 0.5`), outward (`u > 0.5`).
  With reorientation enabled, the measured heading is after turning, as used
  in the actual position update.
- **Wall:** signed disk-surface clearance is `L/2 - 0.5 - |position|`; negative
  means soft-wall penetration. Center-to-boundary distance is clearance + 0.5.
  Wall-force fractions count steps with penetration and nonzero wall stiffness.
  An encounter is near the wall if either disk has clearance <= `--near-wall`
  (default 0.1), evaluated at activation or episode start.

Statistics use pre-position-update states, after the timer/contact response.
Times are relative to the end of warm-up. The observation window contains exactly
`round(duration/dt)` steps, starting after `steady_state_steps`. Contacts and
returns spanning the start/end of that window are flagged as censored.

## Files and plots

Each `data/contact_stats/no_reorient/L3_tau.../` folder contains:

- `settings.json`: simulation parameters and diagnostic options.
- `summary.csv`: one row per independent run, including contact/persistence/wall
  fractions, activation/encounter rates, renewal fraction, and mean wall distances.
- `sample_00001.npz`, etc.: compressed events, sparse traces, lag times, and response
  accumulators. Each sample is saved atomically so an interrupted run can resume.

NPZ keys `events` and `traces` have columns listed in `EVENT_COLUMNS` and
`TRACE_COLUMNS` in `src/contact_stats.py`. Event kind 0 is contact, 1 is return;
the two censoring flags distinguish incomplete observation from completed episodes.
Traces save actual angles and remaining memory along with positions at up to
64 log-spaced times, including the first and final observed states. Angles are
propulsion headings, not the displacement arrows used in GIFs.

Response accumulators have axes `[heading, wall, lag, statistic]`. Heading order
is inward/tangential/outward; wall order is away/near. Statistic 0 is event count;
the remaining entries are sums in `RESPONSE_NAMES` order. Divide each sum by its
count to obtain that run's mean. The ring buffer stores only the lag horizon,
not every position for the full simulation.

Plots are saved under `figures/contact_stats/no_reorient/L3/`:
`contact_survival.png`, `return_survival.png` (all, persistent and nonpersistent
separations), and `delta_distance_all.png` (one panel per heading group).
CSV files save the plotted curves and scalar means with SEM across independent
seeds. Missing groups remain undefined; counts are retained for every lag.

Survival uses Kaplan–Meier estimates, keeping right-censored episodes and excluding
left-censored episodes whose full duration is unknown. Curves are averaged equally
across seeds with data; they stop beyond each seed's observed follow-up. Shading
is ±1 SEM across independent seeds, rather than treating correlated encounters as
independent measurements. A single seed has no SEM estimate.

Additional views and optional readable event/trace exports:

```bash
python src/plot_contact_stats.py --wall near --observable green_away
python src/plot_contact_stats.py --wall away --observable red_away --export-events
```

`--wall` conditions on location at the reference event, not later wall encounters.
All six motion/state observables are also exported in `motion_all.csv` (or
`motion_near.csv`/`motion_away.csv`). To test sensitivity, repeat with a different
`--epsilon` and a new output folder. `--max-lag` (default 2) sets the motion horizon;
`--lag-points` (default 32) and `--trace-points` (default 64) control storage.

For tau=0, motion is conditioned on new contact entries; positive tau uses actual
activations, including renewals. These are different event ensembles. The control
is useful descriptively, but is not a paired restart of identical encounter states.
No replay or change to the contact-reactivation rule is introduced here.
