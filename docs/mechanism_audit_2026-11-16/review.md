# Mechanism and implementation audit

16 November 2026. Branch `exactly-like-paper`, local commit `34a4824`.
Core source hash: `2f688bd318ffbed2ebd6fe06648f0c7572a55776e3fa6c6bcf21246a49b1c7eb`.
Arena adapter hash: `e086e8cf64263ed92a40bc32ade49b9097bbda380b9ef23a3d7d9e445541eddf`.

This is a review of the current implementation and approach. No simulation
source or existing run data was changed. The included probes compile temporary
copies and observe controlled states. Their frozen geometries isolate rules;
they do not measure how often these configurations occur in real trajectories.

## Assessment

**The qualitative mechanism matches the persistence-only mechanism of Fig. 4:
heterotypic contact activates a lower rotational diffusion in green particles,
with red particles retaining baseline diffusion. Reorientation and adhesion are
disabled. However, “the exact same physical model, differing only in boundary
conditions” is too strong.** We reproduce the supplied authors' implementation,
including important discrete rules and numerical defects. The circular geometry
also changes initialization, wall interactions, finite-size effects and the point
at which our small-box neighbor fallback changes to linked cells.

The reference is [Bothe et al. (2026)](https://doi.org/10.1038/s42005-026-02619-2).
The [authors' repository](https://github.com/mb176/cell_simulation) explicitly
documents the shared-partner contact bookkeeping. Matching that executable is
valuable reproduction evidence, but does not by itself validate its numerical
approximation to the intended stochastic model.

## What is correct

- The current paper/arena sweeps use the inherited C update functions and RNG,
  rather than the earlier NumPy/Numba circular kernel.
- Baseline rotational diffusion is D=360000; Python correctly supplies its
  amplitude sqrt(D)=600 to the C integrator. The activated coefficient is zero.
- Pe=k=1000, translational diffusion 1, h=1e-5, phi=0.4, no reorientation and no
  adhesion match the chosen Fig. 4 setting. Particle count and circular confinement
  are intentional extensions; the paper typically uses 4000 particles.
- Accepted contacts use distance <1 diameter; demixing neighbors use distance
  <1.1 diameters. Those are intentionally different thresholds.
- The arena removes position and minimum-image wrapping, adds only an inward
  soft wall force, and uses Euclidean distances for its demixing index. The
  corresponding wall potential and gradient agree in the initialization test.
- The final-snapshot measurement near paper time 5 follows the paper's Fig. 4
  protocol. It is not a time average. Stationarity still needs independent checks
  for each new size/confinement regime.

## High-priority findings

### 1. Pair forces still depend on computational-bin membership

**Confirmed inherited bug.** In `src/paper/upstream/cell_simulation.c:432`, both
ordered versions of a pair are listed when particles share a spatial bin, and
the force loop updates both particles for each entry. Cross-bin pairs are listed
once. The earlier audit's force-counting issue remains present.

Both current builds were probed with a pair at separation 0.2, k=1000, well
inside the circle so that wall forces vanish. Translating both particles by 0.2
across an internal bin edge changed the force magnitude from 1600 to 800. The
pair formula for one unordered pair gives 800. This violates the intended
interior translation invariance.

Our small-box fallback uses unique pairs and therefore avoids this bug. However,
at phi=0.4 the **arena switches to linked cells at N=16**, whereas the periodic
model switches at N=32. At N=16, the two geometries therefore use different
pair-counting conventions, beyond the physical effect of the wall. Larger boxes
also have different bin widths, which can alter which physical pairs are doubled.

**Recommendation:** retain current results as literal-reproduction data, but use
one unique-pair convention at every N for an intended-equation study. Compare
against all-pairs forces and test interior translations. Treat corrected runs as
a separate model/output schema, never as resumable extensions of existing data.
The effect on the demixing maximum has not been measured.

### 2. Shared contacts can suppress activation of an otherwise touching green

**Confirmed inherited restriction, not a new arena-adapter error.**
`AddContact` at `src/paper/upstream/cell_simulation.c:543` only assigns a pair
when **both** particles have no stored partner. An already-persistent green can
still take that partner slot on a red particle.

The probe places two greens within 0.6 diameters of the same red, with greens
1.2 diameters apart. With contact-list order `(G0,R),(G1,R)`, G0 activates and G1
does not. Under a fixed geometry, this remains true for 200 updates. Reversing
the pair order reverses which green activates. All this occurs in both builds.

Thus the operational rule is **activation through an accepted pair matching**,
not independent activation of every green with any red neighbor. The restriction
is described in the authors' code documentation, but is an additional detail
relative to the simpler verbal description of heterotypic-contact activation.

**Recommendation:** quantify multiple-contact frequency and sensitivity to pair
order. For a simplified intended mechanism, activate each eligible green based
on whether it has any unlike contact, separately from any reorientation partner
bookkeeping. That would be an explicit model change from literal reproduction;
its effect should be tested rather than assumed negligible.

### 3. Tau=0 does not disable contact-induced persistence

**Confirmed inherited discrete behavior.** State activation occurs after the
angular update (`SingleStep`, lines 812–816). A contact sets color=2 and D=0 even
when tau=0 (lines 761–769). The next integration step therefore uses zero angular
noise. Continuing accepted contact resets this state again after expiry.

The probe remains active at every one of 20 consecutive tau=0 contact updates.
It returns to baseline at the next state update without a contact; the angular
update preceding that reset still uses the active diffusion. This is more than a
one-off transient if contact continues: tau=0 retains a contact-dependent motility
rule throughout the accepted-contact episode.

Read-only comparison of **450 pairs of saved runs** at tau=0 and tau=h=1e-5 found
identical final positions, headings, colors, RNG states and demixing metrics in
every pair. Stored timer values need not be identical and were not part of this
equivalence check. Full time-resolved trajectories are not stored in these files.

**Recommendation:** use D_plus=D with turning disabled for a true label-independent
ABP control. This keeps the baseline motility unchanged. A D_plus CLI option is
not currently exposed by the sweep. Keep the existing tau=0 point explicitly
identified as the upstream discrete limit, rather than calling it “no memory.”
Also distinguish contact-entry activation from the present rule, which can
reactivate an expired timer while the same physical contact continues.

### 4. Numerical convergence is not established by executable agreement

At the baseline parameters, sqrt(2 D h)=2.68 radians. The angular Gaussian
increment is a valid Brownian increment, but Euler position integration samples
the propulsion direction too coarsely to guarantee converged motion.

An analytical check for a free 2D ABP with the same update rule gives

`D_eff,discrete = D_T + (v²h/4)coth(Dh/2)`

instead of `D_T + v²/(2D)`. These follow from discrete and continuous orientation
correlations respectively. At v=1000, D=360000, D_T=1, h=1e-5, the values are
3.64046 and 2.38889: a **52.4% discrepancy in free-particle diffusivity**. This
calculation does not measure the error in sorting or prove the peak is an artifact.

**Recommendation:** repeat selected tau points with h, h/2, h/4, h/8 while fixing
physical duration, then extend if still drifting. Check force stiffness as well
as angular diffusion. Preserving the published h is appropriate for replication;
it is not a substitute for convergence in a new physical claim.

## Approach and interpretation

### Nominal constant density does not eliminate confinement effects

The code correctly holds nominal phi=N/L² fixed for the outer circle. Wall force
starts at center radius `(L-1)/2`, however, so the ratio of outer area to the
circle inside this threshold is `(L/(L-1))²`. At phi=0.4 this is 3.27 for N=2,
1.27 for N=32 and 1.06 for N=512. This is a geometric reference, not an exact
effective density: the wall is soft and centers can penetrate the force region.

Wall influence is therefore strongly size-dependent even at constant nominal
phi. The added wall stiffness 30000 is not a parameter of the periodic parent
model. Measure wall occupancy by type, compare wall strengths, and retain PBC
controls when attributing size effects to the microscopic mechanism. Holding
center-accessible area per particle fixed is another control, but would change
nominal phi and should not be described as the same density convention.

### Finite-N mixing reference and N=2

For a fixed configuration with randomly assigned balanced labels, the expected
same-type neighbor fraction is `b_N=(N/2-1)/(N-1)`, not exactly 0.5. Values are
0 at N=2, 1/3 at N=4, 0.4667 at N=16, and 0.4990 at N=512.
`paper_plot.py:131` currently draws the same 0.5 line for every N. This is only
the large-population limit. Report the finite-N reference and optionally
`(M-b_N)/(1-b_N)` alongside the raw paper index. An affine correction at a fixed
N does not change that curve's maximizing tau.

With one green and one red, M can only be zero when defined; it cannot diagnose
homotypic sorting. Pair contact/return/separation statistics are appropriate
there. The local index also does not establish macroscopic phase separation.
Track green/red values and neighbor coverage, since exclusion of isolated
particles changes the population contributing to the combined mean.

### Observation time and uncertainty

The code measures a final snapshot near t=5, which follows the parent Fig. 4
protocol. A plateau in an ensemble mean does not establish that domain sizes,
composition spectra or all size-dependent relaxation times have converged.
The actual elapsed time is 5.00001 because of the inherited inclusive loop;
this tiny offset is documented and is not a plausible explanation for a large
sorting effect. Current plot bands are SEM across seeds, not the standard
deviation shown in some paper figures. Identifying a common sampled maximum
does not locate an exact continuous optimum or establish universality.

## Secondary robustness issues

- `HarmonicForce` divides by distance at exact coincidence (line 468). A probe
  at r=0 returns a nonfinite force. This has not been observed in production
  runs here, but should have a defined treatment in a robust intended-model
  implementation.
- The inherited capacity assertion is `nNeighbourPairs <= MAX_NEIGHBOUR_PAIRS`
  before writing the next pair (line 436); at equality that write is already
  out of bounds. This is a static finding, not a reproduced production overflow.
  The small-box fallback correctly checks `>=` before writing.
- The printed sign in Eq. (3), visually checked in the local PDF, appears
  opposite to the repulsive force described in the text. The executable's force
  pushes overlapping particles apart. I interpret this as a likely typesetting
  sign inconsistency, not a reason to implement attraction. It should be noted
  when claiming literal agreement with every displayed equation.
- The initial headings use the C library `rand()` whereas subsequent noise uses
  xoshiro. Bitwise reproducibility across different C libraries/platforms is not
  guaranteed just by sharing a seed; local executable comparisons are narrower
  than a cross-platform guarantee.

## Recommended order of work

1. Keep the current executable-faithful implementation/data as a reference.
2. Establish a separately versioned intended-model implementation with unique
   pairs, explicitly chosen multi-contact/activation semantics, and a genuine
   no-memory control. Compare one correction at a time before combining them.
3. Test timestep convergence at fixed duration around the sorting maximum.
4. Compare PBC and arena at matched nominal phi; measure wall occupancy and
   finite-N chance baselines. Use time-dependent structural diagnostics before
   calling the state stationary or macroscopically demixed.

The scientifically useful result would be robustness of the optimum to those
choices, or a quantitative explanation of its sensitivity. The current audit
does not establish which of these details shifts the peak or by how much.

## Evidence included

- `probe.py`, `probe_results.txt`: same-bin force, shared-contact ordering,
  tau=0 activation and exact-overlap probes for both builds; also geometry and
  the free-ABP analytical calculation.
- `compare_tau_zero.py`, `tau_zero_comparison.csv`, `tau_zero_summary.txt`:
  comparisons of existing data without rerunning or modifying it.
- `test_results.txt`: all 16 current paper-related tests pass. These confirm
  implementation fidelity, observations and boundary handling; they intentionally
  preserve upstream rules and therefore do not contradict the findings above.

Run the probe with `PYTHONPATH=src python3 docs/mechanism_audit_2026-11-16/probe.py`.
No production sweep was launched and no simulation code was changed for this review.
