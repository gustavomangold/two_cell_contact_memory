# How far does green move after separation?

These measurements start when contact ends: center distance crosses above 1.01
disk diameters with the default hysteresis. Positions are checked every step.
Distances below use disk diameters, and times use simulation time units.

## Collect new runs

```bash
.venv/bin/python src/contact_stats.py --L 3 --angular-diffusion 360 \
    --tau 0 0.0005 0.001 0.01 0.1 1 2 5 --samples 50 --workers 10 \
    --post-separation-time 1
.venv/bin/python src/plot_post_separation.py
```

New data goes to `data/contact_stats/post_separation/no_reorient/`; plots go to
`figures/post_separation/no_reorient/L3/`. `--reorient` selects the sibling
`reorient/` data folder. Both commands support `--output`; the plotter also accepts
`--data-dir` and `--L`.

Without the new flag, existing diagnostics behave as before. Old recordings are
preserved; their sparse positions cannot recover these measurements accurately.
Rerun the same command to resume. Changing the fixed-time window requires another
output folder. The requested window is rounded to whole integration steps.

## Motion while memory remains active

Open `post_separation_memory.png`. Its three panels show:

1. **How often memory survives separation:** percentage of separations where green
   still has memory. For tau=0 this is zero, provided separations were observed.
2. **How far green moves during that memory:** start at separation and stop at
   memory expiry or the next contact, whichever happens first. Measure the
   straight-line distance between green's start and end positions.
3. **The largest increase in the green–red gap during that interval:** this detects
   an escape even if the disks later return to each other.

An event can end only 0.1 diameters from its starting position after having reached
0.4 away earlier. Both final and maximum green displacement are saved. Green's
movement and its escape from red differ: the disks can move together without
increasing their gap.

There is no conditional displacement point for tau=0 because no active-memory
interval exists. Missing points mean no completed qualifying events, not zero
displacement. A timer renewal at recontact cannot extend the old event.

## Compare with no memory over the same elapsed time

**Use `post_separation_fixed_time.png` for the tau=0 comparison.** After every
separation, follow the disks for exactly the requested window (one time unit in
the example). This clock continues through memory expiry and new contacts.
The panels show green's net displacement and the change in pair distance after
that same time. The dashed line is the tau=0 result.

This avoids comparing positive memory intervals with a zero-length control.
It is a comparison across encounters, not a replay of identical encounter states.

## Walls, incomplete events and uncertainty

```bash
.venv/bin/python src/plot_post_separation.py --wall away --wall-free --export-events
```

`--wall away` selects separations starting away from the wall using the existing
0.1-diameter clearance rule. `--wall-free` further excludes displacement intervals
where either disk reaches a wall-force region. The availability panel still uses
all separation starts in the selected start-location group. Filtered filenames
have a suffix so they do not overwrite the unfiltered view.

Error bars are ±1 SEM across independent run means. Events interrupted by the end
of recording are saved as censored and excluded from completed-displacement means.
Check the censored counts: many unfinished events make those means selective;
longer observation times are then needed. Fixed-time windows with insufficient
follow-up are also flagged and excluded, rather than filled with zero.

## Files

Each new compressed sample adds `post_separation`, `fixed_after_separation`, and
`post_separation_window`. Column names are in `POST_COLUMNS` and `FIXED_COLUMNS`
in `src/post_separation.py`. Memory-event end reasons are 0=memory expiry,
1=recontact, 2=observation ended. Fixed-event `complete` is 1 only if the full
window was observed. Censored endpoints use the last observed position.

`summary.csv` gains post-separation metrics. `post_separation_means.csv` exports
means, SEMs, contributing-run counts and completed/censored event counts.
`--export-events` also writes readable event CSVs with sample and seed IDs.
Duration, remaining memory, maximum green excursion, wall visits and recontact
outcomes are saved alongside the three main measurements.

These are net displacements and maximum excursions, not sums of tiny Brownian
steps; those path lengths would depend strongly on the integration time step.
