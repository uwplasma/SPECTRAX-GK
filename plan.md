# SPECTRAX-GK Quasilinear Transport and Optimization Plan

Last updated: 2026-04-29
Active repository: `uwplasma/SPECTRAX-GK`
Historical planning archive: private repo `rogeriojorge/spectraxgk_plan`
Current public baseline: `main` at v1.4.0, with the historical ship-readiness log archived before this file was reset.

This file is both the active plan and the running log. Keep entries concise, dated, and tied to artifacts, tests, and figures.

## Current Goal

Bake a research-grade quasilinear transport capability into SPECTRAX-GK and use it as the reduced-model layer for differentiable stellarator optimization. The work must remain honest about what quasilinear theory can and cannot claim: linear weights and sensitivities can be exact within the implemented model; absolute saturated flux prediction requires saturation-rule calibration and nonlinear validation.

The target paper should show:

1. A JAX-native implementation of eigenfunction-resolved quasilinear heat and particle flux diagnostics.
2. A controlled comparison of saturation rules against nonlinear gyrokinetic runs across axisymmetric and non-axisymmetric cases.
3. Differentiable quasilinear objectives with finite-difference, tangent, and covariance validation.
4. A full `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` pipeline for stellarator sensitivity analysis, uncertainty quantification, inverse design, and optimization.
5. Nonlinear audit runs that confirm where the reduced objective does and does not predict saturated transport trends.

## Literature Anchors From Final Pass

Use these references to define the paper claims and required figures:

- Parker et al. 2023, "Comparison of Saturation Rules Used for Gyrokinetic Quasilinear Transport Modeling": compares multiple saturation rules using linear gyrokinetic fluxes and discusses nonlinear/global comparisons. This is the primary saturation-rule benchmark template.
  https://www.mdpi.com/2571-6182/6/4/42
- Stephens et al. 2021, "Quasilinear gyrokinetic theory: A derivation of QuaLiKiz": self-contained derivation of a quasilinear gyrokinetic transport model; use for notation, flux weights, and transport channels.
  https://arxiv.org/abs/2103.10569
- Citrin et al. 2017, QuaLiKiz profile evolution: establishes the integrated-modelling use case and the importance of multi-channel heat, particle, and momentum transport.
  https://arxiv.org/abs/1708.01224
- Staebler, Bourdelle, Citrin, Waltz 2024 review: use for validation philosophy of reduced quasilinear models, cross-phases, dominant instabilities, and transport-property tests.
  https://www.ornl.gov/publication/quasilinear-theory-and-modelling-gyrokinetic-turbulent-transport-tokamaks
- Waltz, Casati, Staebler 2009: use to add quasilinear transport approximation tests with nonlinear spectral intensity, flux ratios, and tracer-style validation ideas.
  https://impact.ornl.gov/en/publications/gyrokinetic-simulation-tests-of-quasilinear-and-tracer-transport/
- Dudding et al. SAT3 paper and Sar et al. 2026 SAT3-NN preprint: use as evidence that modern saturation models are database-calibrated, spectrum-aware, and must be validated against held-out nonlinear data, especially kinetic-electron/TEM regimes.
  https://eprints.whiterose.ac.uk/188258/1/10_06_hgdudding_SAT3_paper_accepted.pdf
  https://arxiv.org/abs/2604.00462
- Jorge et al. 2023, "Direct Microstability Optimization of Stellarator Devices": primary stellarator quasilinear optimization anchor, especially `sum gamma / <k_perp^2>` and the need to compare against W7-X/HSX-like devices.
  https://arxiv.org/abs/2301.09356
- Kim et al. 2024, "Optimization of nonlinear turbulence in stellarators": primary nonlinear-optimization comparison; use for Birkhoff/window averaging, noisy objective handling, multi-surface and multi-field-line audits.
  https://arxiv.org/abs/2310.18842
- Gonzalez-Jerez et al. 2022 W7-X stella/GENE benchmark: canonical stellarator validation ladder with multiple flux tubes, ITG/TEM scans, zonal response, and nonlinear ITG heat flux.
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/electrostatic-gyrokinetic-simulations-in-wendelstein-7x-geometry-benchmark-between-the-codes-stella-and-gene/434DAA225FC5340D6E2C929C773644E3
- Maximum-J / density-gradient W7-X/HSX paper: use as a required kinetic-electron and nonlinear-stabilization stress test, because simple quasilinear or mixing-length rules can fail when stable-mode energy transfer or maximum-J physics matters.
  https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/turbulence-mitigation-in-maximumj-stellarators-with-electrondensity-gradient/CE3F398EB40134ADFF8D6D9E2758FCEF
- Tiwari et al. 2025 W7-X/QSTK zonal-flow paper: use to justify zonal-response and recurrence closure as paper-level stellarator physics, not only code validation.
  https://arxiv.org/abs/2501.12722
- Mandell et al. 2024 code paper and related reference-code examples: use only for independent-code validation and performance comparison where needed.
  https://arxiv.org/abs/2209.06731

## Honest Claim Ladder

Only make claims at the level supported by gates:

1. **Implemented diagnostic**: linear heat/particle flux weights are computed and phase/amplitude invariant.
2. **Validated linear physics**: growth rates, frequencies, eigenfunctions, and `kperp_eff` match reference cases within gates.
3. **Validated quasilinear weights**: species heat/particle weights, signs, and flux ratios agree with direct linear diagnostics and independent references.
4. **Trend-level saturation model**: uncalibrated saturation rules reproduce qualitative spectra, rankings, or gradient trends.
5. **Calibrated reduced model**: a fitted saturation rule predicts held-out nonlinear total fluxes and spectra with uncertainty intervals.
6. **Optimization proxy**: reducing quasilinear objective reduces held-out nonlinear flux in audited cases, with documented failures.
7. **Stellarator optimization claim**: optimization is multi-field-line and multi-surface, derivative-checked end-to-end, and audited by nonlinear runs.

Do not claim universal nonlinear flux prediction. Do not claim production stellarator optimization until multi-surface, multi-alpha, kinetic-electron and nonlinear audit gates pass.

## Quasilinear Model Design

### Tier 1: Linear Transport Weights

Add `src/spectraxgk/quasilinear.py` with functions for:

- normalized eigenstate extraction from time-path or Krylov linear runs;
- `phi_rms`, `phi_midplane`, and `field_energy` normalization;
- species-resolved amplitude-normalized heat flux weights;
- species-resolved amplitude-normalized particle flux weights;
- electrostatic channel first, with electromagnetic channel placeholders guarded by explicit unsupported errors until validated;
- `kperp_eff^2 = <kperp^2 |phi|^2 J dz> / <|phi|^2 J dz>`;
- flux-weight phase invariance and amplitude-scaling diagnostics;
- JSON-friendly result dataclasses.

Initial formulas should reuse the existing SPECTRAX-GK diagnostic conventions rather than duplicating flux kernels. The diagnostic must reject or clearly mark particle flux for single-kinetic-species adiabatic-electron cases where the model cannot support a meaningful ambipolar particle-flux claim.

### Tier 2: Saturation Rules

Implement rules as explicit named models with serialized metadata:

- `none`: linear weights only;
- `mixing_length`: `A2(ky) = Csat * max(gamma, 0) / kperp_eff^2` or the squared-amplitude convention selected in the code, documented precisely;
- `lapillonne_2011`: eigenfunction-weighted `gamma / kperp_eff^2` amplitude scaling;
- `bourdelle_2007`: diffusivity-style scaling with gradient-length dependence;
- `kumar_2021`: non-divergent broader-spectrum rule from Parker et al.;
- `tglf_like_zonal_mixing`: planned, only after zonal-response ingredients and calibration data exist;
- `calibrated_spectral`: planned, trained on nonlinear spectra and validated on holdout cases;
- `ml_saturation`: post-core feature, inspired by SAT3-NN, only after a transparent non-ML baseline is validated.

Every output must include:

- rule name and version;
- amplitude normalization;
- `Csat` and any fitted parameters;
- training cases if calibrated;
- held-out cases if validated;
- whether output is trend-level or calibrated absolute flux.

### Tier 3: Runtime and Artifact Integration

Add config section:

```toml
[quasilinear]
enabled = false
mode = "weights"                  # weights | saturated | calibrated
saturation_rule = "none"          # none | mixing_length | lapillonne_2011 | bourdelle_2007 | kumar_2021
amplitude_normalization = "phi_rms" # phi_rms | phi_midplane | field_energy
kperp_average = "phi_weighted"
csat = 1.0
gamma_floor = 0.0
include_stable_modes = false
delta_ky = "auto"
species = "all"
channels = ["es"]
write_spectrum = true
```

Add executable flags:

- `--quasilinear`
- `--ql-mode weights|saturated|calibrated`
- `--ql-saturation-rule ...`
- `--ql-csat VALUE`
- `--ql-output PATH`
- `--ql-normalization phi_rms|phi_midplane|field_energy`

Add artifacts:

- `*.quasilinear.summary.json`
- `*.quasilinear_spectrum.csv`
- `*.quasilinear_species.csv`
- `*.quasilinear.png`
- calibration tables under `docs/_static/quasilinear_*`

## Benchmark Matrix

### Axisymmetric Adiabatic-Electron ITG

Cases:

- Cyclone circular linear `ky` scan;
- Cyclone circular nonlinear saturation;
- Cyclone Miller linear and nonlinear long-window runs;
- selected Miller-shaping scan for geometry sensitivity.

Figures:

- growth/frequency/eigenfunction panel;
- `kperp_eff`, `gamma/kperp_eff^2`, and flux-weight spectra;
- nonlinear heat-flux time trace with windowed/Birkhoff mean;
- quasilinear rule comparison against nonlinear spectrum;
- model-error table and heatmap.

Gates:

- linear `gamma`, `omega`, eigenfunction overlap;
- phase invariance and amplitude scaling of weights;
- nonlinear window mean/RMS/std;
- saturation-rule trend ranking across `a/LTi` and `ky`.

### Axisymmetric Kinetic-Electron ITG/TEM/ETG

Cases:

- kinetic-electron Cyclone ITG/TEM branch scan;
- density-gradient TEM scan;
- ETG pilot only if benchmark operating point is literature anchored;
- optional isotope-scaling stress test if runtime is feasible.

Figures:

- growth/frequency branch map with mode classification;
- heat and particle flux spectra by species;
- flux-ratio and cross-phase panel;
- failure-mode panel showing where mixing length misses kinetic-electron trends.

Gates:

- species heat/particle sign and ratio checks;
- finite-difference vs AD derivatives for `gamma`, `omega`, and quasilinear flux;
- branch-continuity and near-marginality handling;
- clear demotion if a case remains outside validated scope.

### Electromagnetic / KBM

Cases:

- KBM linear beta scan;
- KBM branch-continuity scan;
- KBM nonlinear long-window audit.

Figures:

- beta-threshold and branch-continuity panel;
- electromagnetic channel contribution panel after validation;
- nonlinear heat-flux comparison with window statistics.

Gates:

- branch overlap and continuity;
- beta threshold monotonicity where expected;
- nonlinear window-statistics gate;
- electromagnetic quasilinear output remains disabled until field-channel normalization is validated.

### Stellarator Adiabatic-Electron ITG

Cases:

- W7-X multi-alpha and multi-surface ITG;
- HSX multi-alpha and multi-surface ITG;
- W7-X/HSX nonlinear long-window validation;
- W7-X zonal-response closure remains a dedicated paper-level lane.

Figures:

- W7-X and HSX linear spectrum atlas;
- field-line dependence heatmaps over `(rho, alpha)`;
- nonlinear heat-flux windows with uncertainty;
- quasilinear-predicted vs nonlinear-observed reduction scatter;
- zonal-response panel only after long-window recurrence is closed.

Gates:

- geometry parity and field-line reproducibility;
- multi-alpha spread and uncertainty;
- nonlinear window acceptance;
- no broad W7-X claim from a single flux tube.

### Stellarator Kinetic-Electron ITG/TEM

Cases:

- W7-X kinetic-electron ITG with and without density gradient;
- HSX kinetic-electron ITG with and without density gradient;
- W7-X/HSX TEM density-gradient scans.

Figures:

- heat and particle flux panels for adiabatic vs kinetic electrons;
- growth-rate and real-frequency panels at low and broad `ky`;
- maximum-J / stable-mode failure-mode panel;
- quasilinear-vs-nonlinear flux-ratio plot.

Gates:

- particle and heat flux signs;
- low-`ky` growth-rate sensitivity;
- model failure is acceptable if it is explained and bounded;
- saturation-rule calibration must use holdout geometry, not just holdout `ky`.

## Full Quasilinear Model Study

### Training and Holdout Policy

Use three tiers:

1. **No calibration**: compare spectrum shapes and trends only.
2. **One-constant calibration**: fit `Csat` on one axisymmetric adiabatic ITG case; validate on Miller, HSX, W7-X.
3. **Multi-parameter spectral calibration**: train on a small mixed set and hold out complete geometries and physics regimes.

Never calibrate and validate on slices of the same nonlinear run without marking it as interpolation. Hold out at least one geometry and one kinetic-electron regime.

### Error Metrics

Report:

- total heat flux relative error;
- total particle flux relative error where meaningful;
- spectrum-shape cosine similarity;
- peak `ky` location error;
- species flux-ratio error;
- electron/ion heat-flux ratio;
- particle-to-heat flux ratio;
- cross-phase agreement where available;
- ranking accuracy across gradient/geometric scans;
- uncertainty intervals from nonlinear windows and calibration covariance.

### Required Publication Figures

Main-paper figures:

1. **Theory and implementation schematic**: linear eigenstate -> flux weights -> saturation rule -> flux prediction -> nonlinear audit.
2. **Linear benchmark atlas**: `gamma`, `omega`, eigenfunction overlap, `kperp_eff` across representative axisymmetric and stellarator cases.
3. **Flux-weight validation**: heat and particle flux weights by species and `ky`, including phase/amplitude invariance tests.
4. **Saturation-rule comparison**: Parker-style rule comparison with total and spectral fluxes.
5. **Nonlinear validation atlas**: time traces, Birkhoff/window averages, spectra, and uncertainty bands.
6. **Model-error map**: cases vs saturation rules, split into uncalibrated and calibrated performance.
7. **Kinetic-electron failure-mode panel**: TEM/ITG-TEM cases showing where simple mixing length fails.
8. **Stellarator field-line/surface panel**: W7-X/HSX multi-alpha and multi-rho maps.
9. **AD validation panel**: AD vs central finite difference for `gamma`, `f_Q`, quasilinear heat/particle flux, and geometry observables.
10. **UQ panel**: covariance ellipses and propagated uncertainty in quasilinear flux.
11. **Optimization panel**: boundary before/after, Boozer `|B|`, QS residual, quasilinear objective history, nonlinear audit bars.
12. **Performance panel**: cold compile, warm throughput, memory, batched `ky` scan speedup, and nonlinear runtime context.

README figures:

- one concise quasilinear diagnostic example;
- one saturation-rule benchmark summary;
- one differentiable optimization summary;
- one runtime/performance summary.

Docs figures:

- all main-paper figures plus diagnostic details, artifact schema, and failure cases.

## Differentiable Geometry and Stellarator Optimization Plan

### Existing Starting Point

SPECTRAX-GK already has a tracer-safe geometry contract in `spectraxgk.geometry.differentiable`:

- backend discovery for `vmec_jax` and `booz_xform_jax`;
- `flux_tube_geometry_from_mapping(...)`;
- geometry observables;
- finite-difference Jacobian helper;
- inverse-design and covariance report utilities.

`vmec_jax` already has boundary-param optimization APIs, exact/discrete-adjoint machinery, fixed-boundary optimization examples, quasisymmetry objectives, aspect-ratio objectives, and Boozer input generation from VMEC state.

### Integration Stages

1. **In-memory geometry parity**
   - Build `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` without writing `wout` or `*.eik.nc`.
   - Compare arrays against the existing imported VMEC/eik path.
   - Gate: max relative errors per geometry coefficient and exact shape/normalization metadata.

2. **Geometry derivative validation**
   - Differentiate geometry observables with respect to boundary modes.
   - Compare AD/JVP/VJP against central finite differences.
   - Gate: absolute/relative derivative tolerances with step-size scan.

3. **Linear observable derivative validation**
   - Differentiate `gamma`, `omega`, `kperp_eff`, flux weights, and `f_Q` with respect to boundary modes.
   - Use smooth branch tracking and smooth positive-growth functions.
   - Gate: AD vs finite differences and branch-continuity diagnostics.

4. **Quasilinear inverse design**
   - Recover target `gamma`, `kperp_eff`, or flux-weight observables using a bounded two- to four-parameter problem.
   - Add covariance estimates and conditioning diagnostics.
   - Gate: inverse recovery, gradient check, covariance finite and interpretable.

5. **Stellarator optimization**
   - Optimize boundary parameters for reduced quasilinear heat flux plus quasisymmetry/aspect/iota regularization.
   - Run multi-alpha and multi-surface objective variants.
   - Audit initial and optimized geometries with nonlinear runs.
   - Gate: nonlinear audit confirms reduction within uncertainty or clearly documents failure.

### Optimization Objectives

Base objective:

```text
J = w_Q * f_Q
  + w_QS * f_QS
  + w_A * (A - A_target)^2
  + w_iota * iota_penalty
  + w_reg * ||boundary - boundary0||^2
```

Use smooth optimization forms:

- `gamma_plus = eps * softplus(gamma / eps)` instead of hard `max(gamma, 0)`;
- softmax-weighted dominant-mode objectives when branches cross;
- multi-`ky` objective as a differentiable weighted sum;
- robust loss or Birkhoff/window objective only for nonlinear audit, not first-pass AD optimization.

### What Goes Beyond Existing Literature

The target contribution should go beyond previous quasilinear or nonlinear stellarator optimization by combining:

- a JAX-native differentiable gyrokinetic quasilinear objective;
- direct AD through geometry-to-linear-transport observables;
- explicit uncertainty and conditioning diagnostics;
- calibrated and uncalibrated saturation-rule comparison;
- multi-surface and multi-field-line stellarator optimization;
- nonlinear audit of optimized geometries;
- documented failure modes for kinetic-electron and nonlinear-stabilization regimes.

## Functionality Checklist

### New Source Functionality

- `spectraxgk.quasilinear` module.
- `RuntimeQuasilinearConfig` in `runtime_config.py`.
- Runtime linear artifact integration.
- Batched `ky` quasilinear scan support.
- Plotting helper for quasilinear spectra and model comparison.
- Calibration utility with explicit train/holdout split.
- Differentiable objective wrappers for `f_Q` and saturated quasilinear flux.
- Geometry-to-linear observable bridge for `vmec_jax` output.
- Optimization example using `vmec_jax` boundary parameters.

### New Examples

- `examples/quasilinear/cyclone_ql_weights.py`
- `examples/quasilinear/cyclone_saturation_rules.py`
- `examples/quasilinear/kinetic_electron_flux_channels.py`
- `examples/quasilinear/w7x_hsx_ql_scan.py`
- `examples/quasilinear/calibrate_saturation_rule.py`
- `examples/optimization/vmec_jax_quasilinear_stellarator.py`
- `examples/optimization/nonlinear_audit_optimized_geometry.py`

### New Documentation

- `docs/quasilinear.rst`: equations, assumptions, rules, outputs, limitations.
- `docs/quasilinear_validation.rst`: benchmark matrix, gates, figures, failure cases.
- `docs/differentiable_stellarator_optimization.rst`: vmec_jax/booz_xform_jax pipeline and examples.
- `docs/artifacts.rst`: JSON/CSV schemas for quasilinear outputs.
- `docs/testing.rst`: quasilinear and AD gate policy.
- README section with install/run/plot plus one quasilinear example and one optimization summary.

## Test and CI Plan

### Fast Unit/Property Tests

- phase invariance of linear weights;
- amplitude quadratic scaling of fluxes;
- stable-mode zeroing for saturated rules when `include_stable_modes=false`;
- `delta_ky` integration and units metadata;
- species-channel sign and shape checks;
- adiabatic-electron particle-flux behavior is explicitly marked;
- serialization round trips for quasilinear result dataclasses;
- CLI/TOML parsing and artifact names.

### Numerical and Physics Tests

- growth/frequency parity for selected linear cases;
- eigenfunction overlap and branch continuity;
- `kperp_eff` analytic limits in simple geometry;
- flux weight consistency with existing nonlinear diagnostic kernels in a linearized state;
- observed convergence with `Nl`, `Nm`, `ntheta`, and `ky` grid;
- nonlinear window statistics for selected audit cases;
- W7-X/HSX multi-alpha sensitivity gates.

### Autodiff Tests

- AD vs finite difference for `kperp_eff`;
- AD vs finite difference for linear `gamma` and `omega` where branch is isolated;
- AD vs finite difference for `f_Q`;
- JVP/VJP consistency on small linear objectives;
- covariance positive-semidefinite checks;
- inverse-design convergence and conditioning diagnostics;
- vmec_jax boundary derivative check through geometry bridge.

### Coverage and CI

- Keep package-wide coverage target at 95%.
- Shard wide coverage so local and CI shards stay bounded.
- Add quasilinear tests to fast PR CI.
- Keep heavy nonlinear validation in manual/office workflows with tracked artifacts.
- Every paper-facing figure must have a script, input manifest, output JSON, and test that validates schema and key metrics.

## Remaining Open Lanes From Previous Plan

These remain active and must be integrated with the quasilinear/optimization work:

1. **Code refactoring**
   - Continue behavior-preserving decomposition of `runtime.py`, `linear.py`, `nonlinear.py`, `benchmarks.py`, plotting, and geometry adapters.
   - Every extraction needs tests and compatibility exports.

2. **Physics gates and validation artifacts**
   - Keep `tools/validation_coverage_manifest.toml` current.
   - Keep gate reports machine-readable and connected to docs figures.
   - Add quasilinear gates to the validation index.

3. **95% package-wide coverage**
   - Continue focused tranches on runtime, linear, nonlinear, benchmarks, geometry, diagnostics, plotting, and new quasilinear code.
   - Coverage must come from physics/numerics/AD tests, not shallow import tests.

4. **Performance and profiling**
   - Keep cold compile, warm runtime, memory, output, and plotting time separated.
   - Profile nonlinear RHS hot paths before making optimization claims.
   - Keep batched `ky` scans and UQ ensembles as the first production parallelization path.

5. **Parallelization**
   - Production path starts with independent `ky` scans, sensitivity sweeps, calibration grids, and UQ ensembles.
   - Nonlinear domain decomposition remains post-release unless a numerical-identity gate is implemented.

6. **Differentiable geometry**
   - Complete `vmec_jax -> booz_xform_jax -> FluxTubeGeometryData` in-memory parity.
   - Add derivative checks before optimizing growth rates or fluxes.

7. **W7-X zonal response**
   - State initializer and observable conventions are closed.
   - Long-window damping, recurrence, moment-resolution, and closure remain paper-level work.
   - Use this lane to support stellarator nonlinear-stabilization claims, not as a release blocker.

8. **W7-X fluctuation spectrum**
   - Add reproducible spectral estimator and publication panel.
   - Tie to nonlinear heat-flux and zonal-response interpretation.

9. **W7-X multi-flux-tube/TEM extension**
   - Required before broad W7-X stellarator-validation claims.
   - Must include kinetic-electron density-gradient/TEM cases.

10. **Nonlinear window statistics tightening**
    - Current release-level gates are useful but broad.
    - Paper-level tolerances must become case-specific where references support them.

11. **Autodiff validation**
    - Keep finite-difference and tangent checks.
    - Strengthen UQ covariance and sensitivity-map workflows for every differentiated observable.

12. **KBM branch continuity and nonlinear audit**
    - Keep selected-branch continuity gates.
    - Treat electromagnetic quasilinear channels as disabled until validated.

13. **Cyclone velocity-space convergence**
    - Maintain the real monotone velocity-space convergence sweep.
    - Extend to quasilinear weights and flux spectra.

14. **Documentation and README**
    - Keep docs user-first: install, run, plot, inspect artifacts.
    - Move long benchmark caveats into validation pages.
    - Main README should show only clean, stable, high-information figures.

15. **Release hygiene**
    - Keep untracked local artifacts such as `.cache_w7x_jpp_2022.pdf` out of release.
    - Keep CI/CD green before tags.
    - PyPI release workflow remains `release.yml`.

## Initial Milestones

### Milestone A: Quasilinear Core Diagnostic

Deliverables:

- source module and dataclasses;
- TOML and executable flags;
- linear runtime artifact writer;
- Cyclone linear weight example;
- unit/property/AD tests;
- docs page with equations and limitations.

Exit gate:

- fast tests pass;
- phase/amplitude invariance passes;
- `gamma`, `omega`, `kperp_eff`, and weights are serialized and plotted.

### Milestone B: Saturation Rule Study

Deliverables:

- named saturation rules;
- calibration utility;
- Cyclone and Miller nonlinear comparisons;
- W7-X/HSX exploratory scans;
- model-error panel.

Exit gate:

- uncalibrated rules clearly marked as trend-only;
- one-constant calibration has held-out validation;
- figure scripts write JSON/CSV/PNG/PDF.

### Milestone C: Kinetic-Electron and Failure-Mode Study

Deliverables:

- kinetic-electron heat/particle flux channels;
- TEM/ITG-TEM branch scans;
- W7-X/HSX density-gradient comparison;
- cross-phase/flux-ratio diagnostics.

Exit gate:

- failures of simple saturation models are documented;
- no unsupported absolute particle-flux claim remains.

### Milestone D: Differentiable Geometry Integration

Deliverables:

- in-memory `vmec_jax` bridge;
- optional `booz_xform_jax` bridge;
- geometry parity artifact;
- AD/FD geometry derivative artifact.

Exit gate:

- geometry arrays match existing path;
- derivative checks pass with step-size scan.

### Milestone E: Quasilinear Stellarator Optimization

Deliverables:

- differentiable quasilinear objective;
- optimization example with QS/aspect regularization;
- UQ/covariance panel;
- nonlinear audit of initial/final geometry.

Exit gate:

- optimization reduces quasilinear objective;
- derivative checks pass;
- nonlinear audit confirms or bounds the reduced-model prediction.

### Milestone F: Paper and Release Packaging

Deliverables:

- docs figure stack;
- README summary figures;
- validation matrix;
- performance panel;
- CI and coverage gates;
- release notes.

Exit gate:

- 95% package-wide coverage confirmed;
- docs build;
- package build;
- fast and selected artifact tests pass;
- all claims in README/docs map to artifacts.

## Running Log

### 2026-04-29

- Archived the historical 2,755-line root `plan.md` into private repo `rogeriojorge/spectraxgk_plan`.
- Replaced root `plan.md` with this active quasilinear transport and differentiable stellarator optimization plan.
- Final literature pass added missing requirements:
  - flux-ratio and cross-phase diagnostics for reviewer-proof quasilinear validation;
  - kinetic-electron W7-X/HSX cases where maximum-J physics and nonlinear stable-mode transfer can invalidate simple mixing-length expectations;
  - calibration/holdout split for any absolute flux claim;
  - nonlinear audit of optimized stellarators before broad optimization claims;
  - SAT3/SAT3-NN style spectrum-aware saturation as future calibrated work, not the first implementation step.
- Removed the stale untracked `.cache_w7x_jpp_2022.pdf` from the active repo checkout.
- Started Milestone A implementation:
  - added `src/spectraxgk/quasilinear.py` with electrostatic linear heat/particle flux weights, `kperp_eff2`, amplitude normalizations, and explicit saturation-rule metadata;
  - added `[quasilinear]` to the runtime TOML schema and executable flags (`--quasilinear`, `--ql-mode`, `--ql-saturation-rule`, `--ql-csat`, `--ql-normalization`, `--ql-output`);
  - wired Krylov/time linear states into quasilinear artifacts without forcing state artifacts unless requested;
  - added `*.quasilinear.summary.json` and `*.quasilinear_species.csv` writers;
  - added `examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml`, README usage, and docs page `docs/quasilinear.rst`.
- Fast validation run:
  - `pytest -q tests/test_quasilinear.py tests/test_runtime_config.py::test_runtime_config_to_dict_contains_sections tests/test_runtime_config.py::test_load_runtime_from_toml_roundtrip tests/test_runtime_artifacts.py::test_write_runtime_linear_artifacts_writes_bundle tests/test_cli.py::test_cmd_run_runtime_linear_applies_quasilinear_flags` passed.
  - Broader adjacent shard with runtime/CLI/artifact tests passed (`31` tests).
  - Example smoke passed: `spectraxgk run-runtime-linear --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml --out tools_out/ql_smoke --no-progress`.
- Next best steps:
  - add quasilinear scan aggregation so `scan-runtime-linear` can produce `*.quasilinear_spectrum.csv`;
  - add nonlinear calibration table scaffolding and holdout metadata before any absolute flux claims;
  - add finite-difference vs autodiff checks for `kperp_eff2`, linear weights, and mixing-length objective;
  - then start nonlinear Cyclone/Cyclone Miller calibration comparisons for the first publication panel.
- Completed the first scan-level quasilinear spectrum slice:
  - serial `run_runtime_scan` now collects per-ky quasilinear payloads;
  - `scan-runtime-linear --quasilinear --out ...` writes `*.scan.csv` and `*.quasilinear_spectrum.csv`;
  - batched quasilinear scans are rejected until per-ky state extraction has a numerical-identity gate.
- Validation for scan slice:
  - `pytest -q tests/test_quasilinear.py tests/test_runtime_artifacts.py::test_write_runtime_linear_artifacts_writes_bundle tests/test_runtime_artifacts.py::test_write_runtime_linear_scan_artifacts_with_quasilinear_spectrum tests/test_cli.py::test_cmd_run_runtime_linear_applies_quasilinear_flags tests/test_cli.py::test_cmd_scan_runtime_linear_writes_quasilinear_spectrum tests/test_cli.py::test_cmd_scan_runtime_linear_branches tests/test_runtime_config.py::test_load_runtime_from_toml_roundtrip` passed.
  - Example scan smoke passed: `spectraxgk scan-runtime-linear --config examples/linear/axisymmetric/runtime_cyclone_quasilinear.toml --ky-values 0.2,0.3 --quasilinear --out tools_out/ql_scan_smoke --no-progress`.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Updated next best steps:
  - add differentiable quasilinear objective helpers and AD-vs-finite-difference validation;
  - add calibration/holdout artifact schema and nonlinear comparison script skeleton;
  - produce the first publication-ready quasilinear spectrum figure from the Cyclone scan.
- Completed first differentiability gate slice:
  - added JAX-native `mixing_length_amplitude2_jax`, `saturated_flux_from_linear_weight`, and `quasilinear_feature_objective`;
  - added generic `central_finite_difference_jacobian` and `autodiff_finite_difference_report` validation helpers;
  - added derivative tests for a closed-form vector function and the quasilinear reduced objective, including directional tangent diagnostics.
- Validation for differentiability slice:
  - `pytest -q tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_autodiff_validation.py` passed.
- Next best steps:
  - add nonlinear calibration/holdout artifact schema and scripts;
  - add the first quasilinear spectrum figure generator for Cyclone;
  - then wire finite-difference derivative gates to full linear-run outputs, not only reduced features.
- Completed calibration/holdout artifact schema slice:
  - added `QuasilinearCalibrationPoint`, `quasilinear_calibration_report`, and `write_quasilinear_calibration_report`;
  - added `tools/build_quasilinear_calibration_report.py` for JSON point lists;
  - reports are promoted to `calibrated_absolute_flux` only when train and holdout points exist and the holdout mean-relative gate passes.
- Validation for calibration slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/quasilinear_calibration.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tools/build_quasilinear_calibration_report.py` passed.
  - Tool smoke with temporary train/holdout points produced a valid JSON report.
- Next best steps:
  - implement the first publication-ready quasilinear spectrum plotting script;
  - generate a Cyclone spectrum figure from `runtime_cyclone_quasilinear.toml`;
  - then add nonlinear comparison ingestion that maps existing nonlinear window summaries into calibration points.
- Completed first quasilinear spectrum figure slice:
  - added `tools/plot_quasilinear_spectrum.py`;
  - generated tracked artifacts under `docs/_static/quasilinear_cyclone_spectrum*`;
  - added the figure to README, docs, and the manuscript figure index with the nonlinear-calibration caveat.
- Validation for figure slice:
  - `pytest -q tests/test_plot_quasilinear_spectrum.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear.py src/spectraxgk/quasilinear_calibration.py src/spectraxgk/autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py tests/test_autodiff_validation.py tests/test_plot_quasilinear_spectrum.py tools/build_quasilinear_calibration_report.py tools/plot_quasilinear_spectrum.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - add nonlinear-window-summary ingestion into calibration points;
  - add a light calibration figure once actual nonlinear holdout data are mapped;
  - add full-output derivative gates that differentiate through a small linear solve, then connect to `vmec_jax` geometry parameters.
- Completed nonlinear-window ingestion slice:
  - added `calibration_point_from_nonlinear_window_summary`;
  - the helper reads tracked nonlinear window JSON, applies the summary `tmin`/`tmax`, and records mean/std heat flux from diagnostics CSVs;
  - NetCDF-only ingestion remains explicitly unsupported until it has the same observable/window contract.
- Validation for ingestion slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_spectrum.py tests/test_autodiff_validation.py tests/test_quasilinear.py` passed.
  - targeted `ruff check` passed for the new quasilinear/autodiff/tool files.
- Next best steps:
  - add full-output derivative gates through a tiny differentiable linear solve;
  - start `vmec_jax` bridge planning/implementation from in-memory geometry arrays;
  - add a light nonlinear-calibration figure once calibration points are produced from real nonlinear windows.
- Completed first nonlinear-calibration audit slice:
  - added quasilinear spectrum integration into `spectraxgk.quasilinear_calibration`;
  - extended `tools/build_quasilinear_calibration_report.py` so it can generate an audit point directly from a quasilinear spectrum CSV and nonlinear window-summary JSON;
  - added `tools/plot_quasilinear_calibration.py`;
  - generated the tracked Cyclone audit artifacts `docs/_static/quasilinear_cyclone_calibration_audit*` from the current quasilinear spectrum and long-window nonlinear heat-flux summary;
  - kept the claim level at `training_or_audit_only` because the current artifact has no train/holdout split and uses uncalibrated `C_sat = 1`.
- Validation for calibration-audit slice:
  - `pytest -q tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_calibration.py tests/test_plot_quasilinear_spectrum.py tests/test_quasilinear.py` passed.
  - `ruff check src/spectraxgk/quasilinear_calibration.py tests/test_quasilinear_calibration.py tests/test_plot_quasilinear_calibration.py tools/build_quasilinear_calibration_report.py tools/plot_quasilinear_calibration.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - add full-output derivative gates through a tiny differentiable linear/eigen solve;
  - build the first train/holdout quasilinear calibration dataset across Cyclone/Cyclone Miller/KBM/HSX/W7-X once matching linear spectra exist;
  - start the `vmec_jax` in-memory geometry bridge so quasilinear objectives can be differentiated with respect to boundary parameters.
