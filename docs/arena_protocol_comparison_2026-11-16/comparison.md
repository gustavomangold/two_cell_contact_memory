# Why the old and current arena demixing plots differ

16 November 2026. Read-only comparison of the two figures and their exported
CSVs, current C manifests/NPZs, and the legacy source defaults. No simulations
were launched and no existing source, run data or figure was changed.

## Main conclusion

Switching to the authors' C engine also switched the algorithm and several run
settings. This was not a controlled comparison of language or acceleration.
The demixing statistic in the two named plots is the same final-snapshot,
per-particle same-type-neighbor fraction with cutoff 1.1 diameters. These are
not a time-averaged index versus a final index, nor raw versus chance-corrected
indices. The differently named `local_demixing.png` is a separate diagnostic.

## 1. The horizontal axes have different units

The legacy arena uses speed 1, the paper model speed 1000. Its time variable is
rescaled by 1000 relative to paper time. Therefore

`tau_paper = tau_legacy / 1000`, and `t_paper = t_legacy / 1000`.

Legacy tau=1,2,5 correspond to paper tau=0.001,0.002,0.005. Comparing the same
printed number on the two original axes compares different nominal persistent
travel distances. The included overlay uses `v*tau/sigma` to align this length.
It corrects the horizontal units only; it does not make the protocols identical.

For N=512 the old maximum is at legacy tau=2, or paper tau=0.002. It is thus
already in the same order-of-magnitude range as the current largest-system
maximum near paper tau=0.001. The old figure should not be interpreted as a
peak at paper tau=2. The old export contains N<=512; the new figure extends
through N=2048, so direct old/new comparisons below use overlapping counts.

## 2. Density, snapshot time and timestep also changed

| Setting, expressed in paper units | Legacy arena | Current C arena |
| --- | ---: | ---: |
| Nominal packing fraction | 2/9 = 0.222222 | 0.4 |
| N=512 arena diameter | 48 | 35.7771 |
| Final snapshot time | 0.12 | 5.00001 |
| Integration timestep, using documented legacy defaults | 1e-8 | 1e-5 |
| Bare speed after unit conversion | 1000 | 1000 |
| Bare rotational diffusion after unit conversion | 360000 | 360000 |
| Bare pair stiffness after unit conversion | 1000 | 1000 |
| Wall stiffness after unit conversion | 30000 | 30000 |
| Translational diffusion after unit conversion | 1 | 1 |

The old exported sample CSV explicitly records final time 120 in legacy units,
including its warm-up. That is paper time 0.12. The new snapshot is approximately
41.7 times later; the new nominal density is 1.8 times higher. There is much
more time for wall encounters, radial redistribution and collective rearrangement.
Neither longer time nor higher density guarantees increased M at every tau.

Provenance limitation: the old per-run NPZs/settings manifests are no longer
present locally. Density, tau grid, sample counts and final time are confirmed
from the old exported CSVs. Its dt, diffusion and force settings above use the
unchanged documented defaults, also present in legacy commit e1be937. Overrides
to those parameters cannot be independently excluded without the old manifests.
The new settings are directly verified from its root/per-point manifests.

Even after converting tau, the values remain different. For N=512:

| Same nominal travel distance v*tau/sigma | Legacy M ± SEM | Current C M ± SEM |
| ---: | ---: | ---: |
| 1 | 0.7634 ± 0.0071 | 0.6542 ± 0.0044 |
| 2 | 0.8244 ± 0.0061 | 0.6559 ± 0.0052 |
| 5 | 0.8120 ± 0.0061 | 0.6658 ± 0.0040 |

Those gaps exceed sampling uncertainty. They cannot be attributed solely to an
axis-label conversion. Separating the effects of the changed settings and rules
requires controlled simulations; their contributions have not been quantified.

The contributing population also differs. At the equivalent N=512, travel-length
2 point, old neighbor coverage is 0.3086 and current coverage is 0.7562. The old
M=0.8244 describes only the non-isolated 31% of particles; current M=0.6559
describes about 76%. Both scripts intentionally exclude isolated particles.
Therefore raw M should be read together with coverage, especially across
different densities; a larger value does not alone imply more whole-population
separation. This is an interpretation issue, not a different formula in the
two named plots.

## 3. Tau=0 is a different model in the two implementations

The legacy kernel activates persistence only when its timer is positive, so
tau=0 genuinely leaves green and red with equal angular diffusion. Its exported
N=512 mean is M=0.5001 and persistent fraction=0.

The C implementation assigns the active color and D=0 even with a zero timer.
Repeated accepted contacts can maintain the active state. In the current N=512
data, tau=0 has mean M=0.5752 and time-averaged green active fraction=0.3354.
For N=2048 the corresponding values are M=0.6168 and active fraction=0.3328.

Thus the newer baseline already has species-dependent contact motility. The
large-N rise of the tau=0 curve is not a disagreement between two simulations
of identical no-memory ABPs. A true control would set D_plus=D with turning off.

## 4. The contact algorithms and numerical resolution differ

- Legacy pair forces count each unordered pair once. The C linked-cell routine
  doubles pairs sharing a spatial bin. This inherited bug changes the force,
  not only runtime. The small-box fallback avoids it, with different switch
  thresholds for square and circular geometry.
- Legacy contact activation checks each green independently. The C routine
  accepts one shared partner per particle, so an already-active green can block
  another green from activating through the same red.
- Legacy activation happens before the angular-noise update; C activation
  happens after it. With D_plus=0, legacy green receives no angular noise on
  its activating step. C first receives a baseline angular kick. At the new
  defaults that kick has RMS sqrt(2*360000*1e-5)=2.68 radians. At a matched
  paper timestep of 1e-8 it would instead be 0.0849 radians. This can change the
  direction inherited by the persistent state and hence contact escape dynamics.
- The coarser C position update also changes free-ABP transport. Analytically
  its default free-particle diffusivity is about 52% above the continuous-time
  value. The difference becomes small at the legacy timestep. This does not
  quantify the resulting error in M.
- Initialization differs: legacy non-overlapping placement with shuffled labels
  (special fixed pair at N=2), versus uniform circle sampling and GSL relaxation.
  The noise generators differ as well. These matter for exact trajectories;
  changing RNG alone should not explain reproducible ensemble differences under
  an otherwise identical, adequately sampled model.

These are algorithm/protocol differences. Compilation in C is not a physical
mechanism that should shift a correctly matched ensemble by itself.

## 5. The current high-tau curves show strong wall-associated segregation

I measured radial positions directly from 50 current NPZs at each point. Define
the outer one-diameter band as `r > L/2 - 1.5`, i.e. within one diameter inward
of the wall-force onset radius `(L-1)/2`, including any soft-wall penetration.
At paper tau=0.01:

| N | Fraction of greens in outer band | Fraction of reds in outer band |
| ---: | ---: | ---: |
| 32 | 0.9250 | 0.2188 |
| 128 | 0.7331 | 0.0853 |
| 512 | 0.5077 | 0.0338 |
| 2048 | 0.3212 | 0.0142 |

This is direct evidence of radial segregation associated with the boundary in
the current run. It is consistent with a wall contribution to the high-tau
increase of M in smaller boxes. The fraction of the population that can occupy
a narrow wall band changes with N. It is not proof that the wall explains the
whole old/new difference, nor that wall accumulation is absent from the old runs.
Old wall diagnostics are time averages, whereas these radial statistics are
final snapshots, so I did not compare them as the same measurement.

## A useful next comparison

Before launching another broad size study, compare one N and a few tau values
under the same physical protocol: phi=2/9, final paper time 0.12, paper dt=1e-8,
and paper tau equal to legacy tau/1000, using the documented old defaults. If
the old manifests are available, verify those settings first. The old discarded
warm-up does not change its final-state measurement time.

Then vary one difference at a time: pair counting, contact acceptance and
activation timing, numerical timestep, and finally density/time. Keep current
data as a separate reference. This is the way to identify causes rather than
assigning every change in M to the C implementation or to system size.

## Accompanying files

- `unit_aligned_comparison.png` / `.pdf`: left, equal nominal travel-length axes
  for matching N (legacy dashed, current C solid); right, current radial wall-band
  fractions at tau=0.01. Bands/error bars are SEM across seeds.
- `aligned_means.csv`: original and converted tau, nominal travel length, M and SEM.
- `wall_profiles.csv`: final wall-force-region fractions, outer-band fractions,
  normalized mean radii, counts and SEMs for the current data.
- `compare.py`: read-only reproducible analysis; `analysis.log`: its output.
