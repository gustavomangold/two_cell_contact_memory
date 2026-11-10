# Reading the contact plots

**Start with `contact_fraction.png`.** It answers the simplest question:
*For what percentage of the run are the two disks touching/overlapping?*

Generate the plots from your existing data; this does not run more simulations:

```bash
python src/plot_contact_stats.py --data-dir data/contact_stats/fig4/Dr360 --L 3
```

For that command, open `figures/contact_stats/Dr360/L3/`.
Every coloured line represents a different **contact-memory time tau**, not a
particle. The shaded bands or error bars show uncertainty between independent runs.
Time uses simulation units: with DT=0.00001, one time unit is 100,000 steps.

## 1. Time spent in contact: `contact_fraction.png`

- **Horizontal axis:** the memory parameter tau used for the whole run.
- **Vertical axis:** percentage of the observation time with center distance < 1 disk diameter.
- **Dashed horizontal line:** the no-memory result, tau=0.

A point at 40% means the disks were in contact for 40% of the measured run.
Below the dashed line means less time in contact than without memory; above it
means more. This plot measures how much time the pair spends together, rather
than their average distance.

This is the first plot to use when asking whether memory reduces unlike-particle
association. It is a pair statistic, not the many-particle demixing index.

## 2. How long one contact lasts: `contact_survival.png`

**Imagine starting a stopwatch whenever a new contact begins.**

- **Horizontal axis:** how much time has passed since that contact began.
- **Vertical axis:** estimated percentage of contacts still ongoing at that time.

For example, at x=1 and y=60%, about 60% of contacts last longer than one time
unit. A higher curve means contacts tend to last longer. A faster fall means
contacts end sooner. The curve should stay flat or fall; it should never rise.

This is NOT the percentage of total simulation time in contact. A few very long
contacts and many brief ones can give a different picture from plot 1.

## 3. How long they stay apart: `return_survival.png`

**Now start the stopwatch when a contact ends. Stop it when they meet again.**

- **Horizontal axis:** time since separation.
- **Vertical axis:** estimated percentage of returns that have not happened yet.

At x=1 and y=60%, about 60% of separation episodes last longer than one time
unit. A higher curve means the disks tend to stay apart longer.

Read the left panel first: it includes all separations. The other panels split
these by whether green's memory was still active when they separated. At tau=0,
there are no events in the persistent-at-separation group.

Higher curves in plot 2 mean **longer contact**. Higher curves in plot 3 mean
**longer separation**. The same vertical direction has different meanings.

## 4. What happens after activation: `delta_distance_all.png`

**Set a new clock to zero when green's memory activates.**

- **Horizontal axis:** time after that event, not time since the simulation began.
- **Vertical axis:** current pair distance minus its value at that event, measured
  in disk diameters. This is a change, not the absolute distance.

A value of +0.2 means the centers are 0.2 disk diameters farther apart than at
activation. A value of -0.2 means they are 0.2 disk diameters closer. It does not
mean the distance itself is negative. Every curve starts at zero by definition.

The panels distinguish green's heading when memory activates: pointing toward
red, sideways, or away from red. This reveals whether memory preserves a heading
that pushes the pair together or helps it separate. For example, a negative curve
in the toward-red panel is consistent with continued pressing into the partner.

The dashed tau=0 control uses **new contact entries** as its reference events,
since memory never activates. Positive tau also includes timer renewals during
existing contacts. Thus these curves describe different event groups; they are
not paired replays of identical encounters.

## Reading a possible result

If plot 1 goes up, plot 2 shifts upward, and plot 3 changes little, the simplest
interpretation is longer contact episodes, rather than a large change in return
times. If plot 1 goes down and plot 3 shifts upward, delayed re-encounters may
contribute. Use the three plots together; one curve alone does not establish sorting.

## Useful options

```bash
python src/plot_contact_stats.py --data-dir data/contact_stats/fig4/Dr360 --L 3 --wall near
python src/plot_contact_stats.py --data-dir data/contact_stats/fig4/Dr360 --L 3 --wall away --observable green_away
```

`--wall near` selects motion events beginning near the wall; `--wall away` selects
those beginning farther away. `--observable green_away` or `red_away` shows that
disk's displacement away from the partner's original position, rather than the
change in pair distance. All plots and CSVs are regenerated from saved runs.

[Recording details and saved-file format](contact_statistics_format.md).

For displacement specifically **after separation**, see the
[post-separation guide](post_separation.md), including the fixed-time tau=0
comparison. The activation curves above start from a different event.
