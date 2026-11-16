# Publication assessment: finite-size robustness of contact-memory sorting

**Review date:** 12 November 2026. **Assessment:** potentially publishable research direction; the present common peak is not sufficient for a standalone original research paper.

This is a targeted literature review, not an exhaustive systematic review or a guarantee of novelty or acceptance. Searches covered contact-induced persistence, environment-dependent cell motility, finite-size demixing, transient versus asymptotic sorting, ABP phase separation, and active microphases. Primary papers, author manuscripts and institutional repositories were preferred. Abstract-only findings are identified below. No production simulations or simulation sources were changed for this review.

## 1. What would be new?

The current result is that the **sampled maximum of a local neighbor-based demixing index is near tau=0.001 across several system sizes**, at one density, baseline diffusivity, speed, stiffness and observation time. It is a useful result to build on, but neither an optimal memory time nor size convergence of an intensive order parameter is new by itself.

The strongest candidate question is:

> Under what conditions does contact memory select size-independent local sorting, and when does that local ordering develop into global composition separation?

A second possible question is how small a system can reproduce a bulk sorting curve **within a declared accuracy**, and what length and time scales determine that minimum size. A quantitative prediction for that boundary would be more substantial than showing overlapping curves at one parameter choice.

The original study already reports an optimal memory duration and the approximate inverse-speed scaling of that optimum. Reproducing those findings in smaller boxes is an extension, not a new mechanism. [Bothe et al., 2026](https://doi.org/10.1038/s42005-026-02619-2)

## 2. Closest literature and its implications

| Work | Relevant established result | Consequence for this project |
|---|---|---|
| **Bothe et al. (2026), Contact-induced differential persistence as a mechanism for cell sorting**, Communications Physics 9, 204. [DOI](https://doi.org/10.1038/s42005-026-02619-2) | Direct parent model; optimal transient persistence and sorting already established. The preferred nominal travel length is of order a diameter. | A paper needs a finite-size, dynamical, predictive or phase-behavior result beyond the existing optimum. Full text/local PDF examined. |
| **Strandkvist et al. (2014), A kinetic mechanism for cell sorting based on local variations in cell motility**, Interface Focus 4, 20140013. [Author PDF](https://kalab.emma.cam.ac.uk/pdfs/Strandkvist2014.pdf), [DOI](https://doi.org/10.1098/rsfs.2014.0013) | Environment-dependent persistence/diffusion drives sorting. The paper explicitly distinguishes early size-insensitive behavior from size-dependent saturation and points to Supplementary Fig. S6. | This is a particularly close predecessor. Demonstrating early curve agreement alone is insufficient novelty. Their rules differ from a fixed-duration heterotypic-contact memory. Main text examined; the supplementary figure was not independently inspected. |
| **Beatrici, de Almeida & Brunnet (2017), Mean-cluster approach indicates cell sorting time scales are determined by collective dynamics**, PRE 95, 032402. [Publisher](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.032402) | Cluster transport and finite-size corrections explain sorting kinetics; collective dynamics can control the growth law. | Measure cluster motion/growth and establish what the present memory model adds. Publisher abstract examined, not subscription full text. |
| **Bain & Bartolo (2017), Critical mingling and universal correlations in model binary active liquids**, Nature Communications 8, 15969. [PDF, Fig. 2a](https://www.nature.com/articles/ncomms15969.pdf#page=3) | Order-parameter curves overlap for five particle counts from 493 to 7892 at fixed density. | Direct precedent for collective curve invariance. Their observable is an orientational indicator of lanes/mixing in a torque-driven counter-moving system, not this model's compositional neighbor index. Full text examined. |
| **Nie et al. (2020), Stability phase diagram of active Brownian particles**, PR Research 2, 023010. [Author-hosted PDF, Appendix C1](https://doylegroup.mit.edu/wp-content/uploads/sites/68/2022/11/Stability-phase-diagram-of-active-Brownian-particles.pdf#page=9) | Standard-ABP separation is assessed using density distributions and a check that doubling size gives consistent results. | Finite-size convergence is an established validation criterion. Density separation and two-species sorting require different observables. Full text examined. |
| **Yu & Jack (2024), Competition between lanes and transient jammed clusters in driven binary mixtures**, PRE 109, 024123. [Author manuscript](https://arxiv.org/abs/2305.05990), [DOI](https://doi.org/10.1103/PhysRevE.109.024123) | Apparent demixing in finite driven mixtures becomes a crossover under size analysis; large systems remain mixed in their model. | A useful counterexample to inferring a bulk demixed phase from finite-box patterns. It is a different driven model, not a refutation of this contact-memory model. Full text examined. |
| **Weber, Weber & Frey (2016), Binary Mixtures of Particles with Different Diffusivities Demix**, PRL 116, 058301. [Publisher](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.058301) | A diffusivity-contrast mixture separates above a critical system size. | A minimum population for segregation is already a known concept. A new result should quantify the memory-specific controlling scales. Publisher abstract examined. |
| **Franke et al. (2022), Is cell segregation like oil and water: Asymptotic versus transitory regime**, PLOS Computational Biology 18, e1010460. [Full text](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010460) | Transient sorting can mimic asymptotic scaling over limited observation windows; different mechanisms can yield similar apparent kinetics. | Observation-time convergence and explicit competing hypotheses are essential. This also limits novelty of a generic transient-versus-asymptotic discussion. Full text/abstract and relevant discussion examined. |
| **Pathania, Chakraborty & Höfling (2021), Continuous demixing transition of binary liquids: finite-size scaling from the analysis of sub-systems**, Advanced Theory and Simulations 4, 2000235. [Manuscript](https://arxiv.org/abs/2009.14807), [DOI](https://doi.org/10.1002/adts.202000235) | Composition structure factors, subvolume fluctuations and cumulants distinguish genuine demixing and finite-size effects. | A methodological reference if a critical transition is actually found. A maximum in M versus tau alone does not establish criticality. Abstract and available full-text sections examined. |
| **Neiman, Dong & Lindner (2025), Metastability in the mixing/demixing of two species with reciprocally concentration-dependent diffusivity**, Frontiers in Network Physiology 5, 1612495. [Full text](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2025.1612495/full) | Composition-dependent motility produces finite-population switching and size-dependent metastable lifetimes. | Inspect distributions and residence times, not only ensemble averages. Their cross-diffusive 1D model is different from an ABP contact timer. Full text/abstract examined. |
| **Antonioli, Gnan & Maggi (2026), Microphase separation in quorum-sensing active particles with competing sensing scales**, JCP 165, 124909. [Manuscript](https://arxiv.org/abs/2607.23259), [DOI](https://doi.org/10.1063/5.0353835) | Competing sensing scales select finite-length active microphases, with a particle-to-continuum prediction. | Finite active clusters are not themselves new. Memory would need to supply a distinct mechanism or quantitative prediction. Publisher metadata/abstract and author abstract examined. |
| **Cochet-Escartin et al. (2017), Physical Mechanisms Driving Cell Sorting in Hydra**, Biophysical Journal 113, 2827–2841. [Publisher](https://doi.org/10.1016/j.bpj.2017.10.045) | Aggregate geometry affects sorting kinetics; neighbor-based sorting indices have size-dependent saturation. | Useful biological context for interpreting a local index and confinement, without claiming a disk model captures all tissue mechanics. Accessible full-text sections examined. |
| **Stenhammar et al. (2014), Phase behaviour of active Brownian particles: The role of dimensionality**, Soft Matter 10, 1489–1499. [Manuscript](https://arxiv.org/abs/1310.6290), [DOI](https://doi.org/10.1039/C3SM52813H) | Bulk ABP separation and coarsening can require substantial control of finite-size effects. | Include at least one substantially larger reference system before describing N=512 as bulk. Author abstract examined. |

The reviewed sources do not establish full sorting-curve invariance down to a handful of particles for this exact contact-memory model. That is a candidate gap, not a verified absence from all literature. The relevant novelty would concern a quantitative scale, prediction, or regime boundary specific to transient contact memory.

## 3. What the existing results establish

I reanalyzed **4050 saved runs at 81 parameter points**, all at phi=0.4, Pe=k=1000, baseline rotational diffusion 360000, activated diffusion 0, and final label t=5. Each point has 50 seeds. The original inclusive loop advances 5.00001 time units. The source hash is `2f688bd318ffbed2ebd6fe06648f0c7572a55776e3fa6c6bcf21246a49b1c7eb`.

These are exploratory diagnostics of the current implementation, not validated new physical conclusions. Derived data and analysis scripts accompany this report.

### Peak location

At N=8 through 512, tau=0.001 is the largest sampled mean M. A paired bootstrap resampling entire seeds across all tau points, with 5000 replicates, selects that grid point with frequencies 0.57, 0.63, 0.72, 0.42, 0.86, 0.88 and 1.00 for N=8,16,32,64,128,256,512 respectively. These are **selection frequencies on the existing grid**, not probabilities or confidence intervals for an exact continuous optimum. At N=4 the winner is split between 0.00025 and 0.001. N=2 has no same-type partners and its M cannot diagnose sorting.

### Local versus large-scale order

At tau=0.001:

| N | Local M, mean ± SEM | Low-q composition power | Largest green cluster / green population |
|---:|---:|---:|---:|
| 32 | 0.6197 ± 0.0175 | 0.03847 | 0.329 |
| 64 | 0.6110 ± 0.0142 | 0.01926 | 0.222 |
| 128 | 0.6334 ± 0.0074 | 0.01102 | 0.146 |
| 256 | 0.6318 ± 0.0063 | 0.00485 | 0.095 |
| 512 | 0.6427 ± 0.0046 | 0.00250 | 0.058 |

For the composition power, let s_i=+1 for green and -1 for red. I averaged `|sum_i s_i exp(i q.r_i)|²/N²` over q=(2pi/L) times (1,0),(0,1),(1,1),(1,-1). The diagonals reduce orientation blindness. This is a limited low-wavevector diagnostic, not the complete structure factor. A balanced random-label expectation conditional on the actual positions is also saved; at N=512 it is 0.00196.

The power falls approximately inversely with N across this range while local M remains similar. The largest connected green cluster, defined by the same 1.1-diameter cutoff, grows from about 5 particles at N=32 to about 15 at N=512, not proportionally to population. These observations are compatible with limited-range sorting at this observation time. They do **not** establish stationary microphase separation, rule out slower coarsening, or rule out order outside the measured modes.

The key question is therefore not simply whether M has converged. It is whether the system has a finite stationary compositional correlation length, or is still coarsening toward a larger-scale state.

## 4. Numerical issues to resolve before interpreting a physical size effect

### A. Use one pair-counting convention for every N

The current small-box fallback counts each pair once. The inherited linked-cell code used in larger boxes can list a pair twice when both particles are in the same spatial bin. This affects the **total force**, not only performance.

I compiled a controlled probe using the inherited functions. A pair at separation 0.2, k=1000, experiences force magnitude 1600 when its particles share a bin, and 800 after both are translated across a bin edge without changing separation. The intended single-pair formula gives 800. The list had two and one entries respectively. The probe output is included in `pair_list_probe.txt`.

This behavior was deliberately preserved earlier for literal reproduction of the published executable. That reproduction target is different from a uniform, translation-invariant implementation of the stated pairwise model. For a physical finite-size study, build and validate a single unique-pair implementation for all sizes; retain the faithful reproduction results separately. Test force invariance under global translations and agreement between all-pairs and accelerated searches. This is a requirement before trusting the present cross-size comparison, not evidence that the parent paper's qualitative result is wrong.

### B. Establish timestep convergence

The angular increment standard deviation at the current baseline is sqrt(2 D_R h)=2.68 radians per step. Angular Gaussian increments can be correct while the Euler position update still poorly resolves motion between orientation samples.

As an independent calculation from the update rule, a **free** 2D ABP integrated with Euler positions has

`D_eff,EM = D_T + (v² h/4) coth(D_R h/2)`,

whereas its continuous-time value is `D_T + v²/(2 D_R)`. This follows by summing discrete orientation correlations exp(-D_R h |j-k|); it is not a result claimed by the cited papers.

At v=1000, D_R=360000 and D_T=1, the continuous value is 2.3889. The discrete values at h=1e-5,5e-6,2.5e-6,1.25e-6 are 3.6405,2.7451,2.4814,2.4122. The default free-particle diffusivity differs by about 52%. The effect of this discrepancy on the sorting optimum has not been determined. Test the physical result at fixed elapsed time as h decreases.

### C. Use a genuine no-memory control

The literal upstream tau=0 rule still activates the green state for a subsequent step during continuing contact. Set activated diffusivity equal to baseline diffusivity, D_plus=D, to obtain a label-independent ABP control with turning disabled. Keep the upstream tau=0 point separately labelled. Compare local M to its finite balanced-label baseline `(N/2-1)/(N-1)`; 0.5 is only the large-N limit. Save neighbor coverage because isolated disks have undefined M.

## 5. A practical sequence of studies

### First gate: validate before enlarging the existing sweep

Use the consistent unique-pair engine. A useful pilot is N=16 and 128, tau=0.0005,0.001,0.002, and h=1e-5,5e-6,2.5e-6,1.25e-6, initially 10–20 seeds. Hold physical final time fixed. Include D_plus=D and periodic translation tests. Expand sampling where peak shifts or differences are close to uncertainty. Archive current data as reproduction/exploration rather than overwriting it.

Proceed only if the effect survives numerical convergence. Otherwise first identify which conclusions depend on the numerical conventions.

### Second gate: distinguish a stationary length scale from finite-time sorting

At a converged timestep and the three tau values above, compare N=32,128,512 and a larger reference such as 2048 or 4000. Record selected configurations at physical times 5,10,20,50 in the same trajectories. Use mixed and initially segregated configurations as complementary relaxation checks; neither initial state alone establishes stationarity.

Measure M by type, coverage, the full composition structure factor, a compositional correlation length, same-type cluster-size distributions, largest cluster fractions, and their time dependence. In a short-range mixed/clustered state, the normalized lowest-mode composition power typically decreases with N. A macroscopic composition domain produces order-one normalized power in appropriate low-q modes. A finite-q peak whose location and width stabilize with time and box size would support a finite intrinsic length scale. Cluster connectivity alone can be sensitive to cutoff and percolation, so use both real-space and Fourier-space measures.

If correlation lengths continue growing, characterize coarsening and the resulting finite-time crossover instead of calling the state a stationary microphase. A common maximum at t=5 may optimize sorting speed rather than the eventual stationary degree of segregation.

### Third gate: test a distinct, quantitative claim

**Option A — finite-size accuracy boundary:** establish a predictive minimum N for matching the large-system M(tau) curve. Choose an equivalence tolerance before interpreting agreement, for example an absolute difference of 0.02 in M if scientifically appropriate. Bootstrap entire seed trajectories across tau to obtain simultaneous curve-difference bands. Failing to reject a difference is not evidence of equivalence. Relate the minimum box length to measured correlation length and observation time; in 2D a criterion L much greater than xi suggests an N scale proportional to density times xi², with a coefficient to be established.

**Option B — local sorting and global separation select different parameters:** show a reproducible separation between the tau optimizing local M and the tau optimizing long-wavelength composition order or a defined sorting time. Determine where it occurs in density and memory-state diffusivity, and supply a kinetic interpretation. This is more specific than generic finite-size convergence.

**Option C — memory-selected stationary microphases:** show that transient contact memory alone arrests compositional coarsening and predicts a finite domain scale, robust to larger boxes and longer times. This would be an attractive physical result if established; it is not demonstrated by current snapshots. Compare carefully with the competing-sensing and other active-microphase literature.

After choosing a route, refine tau around the peak rather than extending a broad coarse grid. Add at least a few densities, such as 0.2,0.4,0.6, and a finite activated diffusivity. Vary speed to locate where inverse-speed optimal-memory scaling succeeds and fails, recognizing that the basic inverse-speed observation is already in the parent paper. Avoid scaling every rate together and presenting a mere change of time units as a new collapse.

For a mechanistic extension, obtain attachment/escape rates from properly sampled few-particle encounters and predict many-particle observables at held-out parameters. Green–green–red triplets are a useful minimum for forming a same-type pair. Hand-picking a geometry that yields the desired optimum would not provide an independent prediction. A successful calculation should predict a curve, shift, or breakdown boundary without refitting its coefficients to each many-body result.

## 6. What a convincing manuscript could contain

1. A well-defined model and numerical convergence; faithful reproduction and intended-equation results clearly separated.
2. M curves with peak-position uncertainty and a declared criterion for size equivalence.
3. Composition spectra and cluster distributions distinguishing local order from macroscopic separation.
4. Time and size dependence showing a converged length scale or a reproducible coarsening crossover.
5. A memory/density/diffusivity regime map, with a mechanism or predictive reduced description tested beyond its calibration point.

A neutral working title is **“Finite-size and observation-time limits of contact-memory-driven sorting.”** A title asserting memory-controlled microphase separation should be used only after stationarity and a finite intrinsic length are established. Computational work can support a physics paper without new experiments, but biological claims require appropriate validation; this disk study alone does not establish a new property of real tissue.

## 7. Decision

| Outcome | Assessment |
|---|---|
| Only the already-known optimum near 0.001 and convergence of M at one parameter set | Useful replication/extension; insufficient original-paper evidence on its own. |
| Quantitative minimum-size law or a memory-specific finite-size/time crossover with out-of-sample prediction | Credible basis for an original computational/theoretical paper. |
| Stationary finite-length compositional order selected by transient memory, with controlled numerics and a mechanism | Stronger candidate contribution; check the closely related active-microphase literature. |
| Apparent sorting or invariance disappears after numerical corrections | Reassess the physical claim; a reproducibility study would require its own substantial, carefully demonstrated result. |

My recommendation is to pursue the project through the first two gates before investing in a large publication sweep. The current data justify further investigation, not a claim of established universality or guaranteed publishability.

## Accompanying files

- `finite_size_diagnostics.png` / `.pdf`: exploratory four-panel summary, all at the current model's t=5. Top left shows M curves, top right low-q power at tau=0.001, bottom left largest-cluster fractions, bottom right grid-peak bootstrap selection frequencies. Bands/error bars are SEM across seeds.
- `finite_size_diagnostics.csv`: point means/SEM, coverage, chance-corrected M, composition diagnostics and cluster measures.
- `per_seed_diagnostics.csv`: seed-level derived measurements.
- `peak_grid_bootstrap.csv`: discrete-grid selection frequencies, 5000 paired seed bootstraps.
- `analyze.py` / `plot_diagnostics.py`: read-only analysis of existing simulation files; these scripts do not run simulations. The analysis script contains the repository path used for this review.
- `pair_list_probe.txt`: controlled inherited-code force probe.
- `probe_pair_list.py`: reproduce the force probe using a temporary copy of the inherited sources; requires a C compiler and GSL.

The literature summaries describe established results. The diagnostic values and numerical calculations above are new analyses of the current files/update rules. Proposed mechanisms, regime maps and paper routes are hypotheses and recommendations, not completed findings.
