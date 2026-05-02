# SPECTRAX-GK Quasilinear Transport and Optimization Plan

Last updated: 2026-04-30
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

## Repository Trim and Artifact Hygiene Plan

Current audit command:

```bash
python tools/audit_repository_size.py --top 30
```

Current local snapshot from 2026-04-29:

- tracked HEAD payload: about `38.8 MB` across `742` files;
- `.git` history payload: about `154 MB`;
- ignored local artifact roots dominate the checkout size: `tools_out` about `657 MB`, `.venv` about `511 MB`, `.mypy_cache` about `167 MB`, `docs/_build` about `71 MB`, and `dist` about `24 MB`;
- largest tracked payloads are documentation/release assets, especially `docs/_static` about `30.5 MB` and `examples/wout_HSX_QHS_vacuum_ns201.nc` about `3.9 MB`.

Non-destructive trim steps:

1. Keep source code, tests, small input TOMLs, small JSON/CSV gate reports, and figure-generation scripts in Git.
2. Move high-resolution PNG/PDF panels, raw NetCDF outputs, profiler traces, and nonessential VMEC/raw reference data to GitHub Releases or another artifact store with checksums and replay commands.
3. Keep only lightweight README/docs figures in Git, preferably compressed publication previews; link high-resolution PDF/PNG artifacts from releases.
4. Add a repository-size CI gate after the first trim pass: fail if a newly tracked file exceeds an agreed threshold unless it is whitelisted in a manifest.
5. Add a release-artifact manifest mapping each moved artifact to a release URL, checksum, generating command, and validation gate.

History rewrite policy:

- Do not rewrite history during ordinary development.
- If clone size remains too large after non-destructive trimming, coordinate a dedicated maintenance window.
- Before rewriting, tag a backup ref, freeze merges, publish migration instructions, and verify PyPI/release artifacts remain reproducible.
- Use path-specific `git filter-repo` rules rather than broad deletion where possible; then force-push only after collaborators agree to reclone or reset local branches.

Exit gate:

- fresh clone excluding optional release artifacts should stay below the agreed size budget;
- docs and examples must still render without downloading heavy raw outputs;
- all publication figures must be regenerable from scripts plus release-manifest artifacts;
- CI must enforce future artifact-size hygiene automatically.

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
- Completed branch-isolated eigenvalue derivative gate slice:
  - added `isolated_eigenvalue_sensitivity_report` to `spectraxgk.autodiff_validation`;
  - the gate selects an eigenvalue branch at the base point, checks the eigenvalue gap, and compares AD Jacobians against central finite differences for `[real(lambda), imag(lambda)]`;
  - this is the lightweight precursor to differentiating full linear growth/frequency and quasilinear objectives through geometry.
- Validation for eigenvalue derivative slice:
  - `pytest -q tests/test_autodiff_validation.py tests/test_quasilinear.py tests/test_quasilinear_calibration.py` passed.
  - `ruff check src/spectraxgk/autodiff_validation.py tests/test_autodiff_validation.py src/spectraxgk/__init__.py` passed.
  - `sphinx-build -b html docs docs/_build/html -q` passed.
- Next best steps:
  - connect the eigenvalue derivative gate to a tiny actual SPECTRAX-GK linear operator fixture;
  - create matching quasilinear spectra for Cyclone Miller/KBM/HSX/W7-X so calibration can move from audit-only to train/holdout;
  - start `vmec_jax`/`booz_xform_jax` bridge interfaces for differentiable geometry inputs.
- Completed the first shaped-tokamak quasilinear transfer slice:
  - added `examples/linear/axisymmetric/runtime_cyclone_miller_quasilinear.toml`;
  - fixed scan quasilinear spectra so `ky` is the requested scan coordinate and `mode_ky` retains the selected signed grid-mode coordinate, which prevents linked-boundary aliases from corrupting publication x-axes;
  - regenerated `docs/_static/quasilinear_cyclone_miller_spectrum*` with the nonlinear Miller `Ny=64` positive-ky grid;
  - added train-fitted multiplicative heat-flux scaling utilities to `spectraxgk.quasilinear_calibration`;
  - generated `docs/_static/quasilinear_cyclone_miller_calibration_audit*` and the first Cyclone-to-Cyclone-Miller train/holdout artifact `docs/_static/quasilinear_cyclone_miller_train_holdout*`.
- Current calibration outcome:
  - fitting one constant on Cyclone gives `C_sat = 3839.966` in the current normalization;
  - the held-out Cyclone Miller mean-relative heat-flux error is `612.9`, so the report correctly stays at `calibration_dataset` with `passed = false`;
  - this is a useful research result, not a release failure: it shows that absolute quasilinear flux prediction needs a richer saturation/normalization model before optimization claims.
- Next best steps:
  - add a second held-out axisymmetric/non-axisymmetric electrostatic case with a validated nonlinear CSV window before attempting any multi-parameter saturation model;
  - connect quasilinear objective derivatives to a tiny SPECTRAX-GK linear-operator fixture, then to the existing differentiable geometry bridge;
  - add spectrum-shape gates that compare normalized quasilinear spectra against nonlinear spectral heat-flux distributions where those diagnostics are available.
- Completed the first actual-linear-operator derivative gate:
  - added `explicit_complex_operator_matrix` for tiny validation fixtures that materialize matrix-free operators without changing production solvers;
  - added a SPECTRAX-GK linear-RHS eigenvalue sensitivity test that differentiates through a small dense operator with `use_custom_vjp=false` and compares AD against finite differences;
  - this closes the gap between toy eigenvalue derivative gates and the real linear RHS path, while keeping production field-solve custom VJPs untouched.
- Validation:
  - `pytest -q tests/test_autodiff_validation.py` passed.
  - `ruff check src/spectraxgk/autodiff_validation.py tests/test_autodiff_validation.py src/spectraxgk/__init__.py` passed.
- Next best steps:
  - wire the dense linear-RHS derivative gate to a reduced quasilinear objective from the resulting eigenvector/state, not just the eigenvalue;
  - connect the derivative gate to the existing differentiable geometry bridge so geometry parameters perturb linear weights and quasilinear objectives;
  - create a second train/holdout calibration artifact with a non-axisymmetric electrostatic case once a CSV-backed nonlinear heat-flux window is available.
- Added the first branch-objective differentiability guard:
  - added `isolated_eigenpair_observable_sensitivity_report` for isolated-branch observables;
  - the actual SPECTRAX-GK linear-RHS phase-invariant objective currently reports `ad_supported=false`, because JAX does not support forward-mode derivatives of non-Hermitian eigenvectors;
  - this prevents false end-to-end differentiability claims for eigenfunction-dependent quasilinear objectives.
- Updated next best steps:
  - implement an adjoint/implicit eigenvector-sensitivity path for isolated non-Hermitian branches, then re-open the quasilinear objective AD gate;
  - in parallel, continue calibration/validation on observable values that do not rely on differentiating eigenvectors;
  - keep eigenvalue-only growth/frequency sensitivities as the currently validated differentiable branch.
- Completed implicit non-Hermitian eigenpair sensitivity slice:
  - added `implicit_eigenpair_observable_sensitivity_report`;
  - the helper differentiates matrix entries with JAX, solves the left/right eigenvector perturbation system with `w^H v = 1` and `w^H dv = 0`, and compares phase-invariant observables against nearest-branch central finite differences;
  - added a closed-form non-Hermitian branch test and a tiny SPECTRAX-GK linear-RHS quasilinear-style objective test.
- Expanded quasilinear documentation:
  - added equations for the linear eigenproblem, electrostatic heat/particle weights, amplitude normalization, effective `k_perp`, mixing-length saturation, calibration scoring, and implicit eigenpair sensitivities;
  - added source-code implementation map linking the quasilinear, diagnostic, runtime, calibration, plotting, and autodiff modules;
  - converted quasilinear references to clickable links in `docs/references.rst`, including QuaLiKiz, Parker saturation-rule comparisons, SAT3/SAT3-NN, Waltz/Casati/Staebler, and microstability optimization references.
- Completed user-facing implicit quasilinear sensitivity example:
  - added `examples/theory_and_demos/quasilinear_implicit_sensitivity.py`;
  - the example differentiates `[gamma, omega, kperp_eff^2, Qhat_i, Q_i^ML]` with respect to `[R/Ln, R/LTi]` on a tiny Cyclone linear-RHS fixture;
  - fixed the implicit eigenpair finite-difference validation protocol to follow the nearest isolated eigenvalue branch instead of raw eigensolver index order;
  - generated `docs/_static/quasilinear_implicit_sensitivity.{png,pdf,json}` and documented the figure in `docs/quasilinear.rst` and `docs/manuscript_figures.rst`.
- Completed second electrostatic calibration/holdout machinery gate:
  - fixed `gx_volume_factors` for closed-interval sampled/imported geometries by reusing the existing terminal-theta trim contract;
  - added `examples/linear/non-axisymmetric/runtime_hsx_linear_quasilinear.toml`;
  - generated `docs/_static/quasilinear_hsx_spectrum.*` and `docs/_static/quasilinear_hsx_spectrum_scan.*` from a six-point HSX adiabatic-electron linear spectrum;
  - generated `docs/_static/quasilinear_hsx_train_holdout_report.json` and `docs/_static/quasilinear_hsx_train_holdout.*` by adding HSX as a non-axisymmetric held-out nonlinear heat-flux window;
  - the HSX point is a negative result: the current short HSX linear spectrum is stable under the uncalibrated `gamma_floor=0` mixing-length rule, giving zero predicted saturated heat flux against a finite nonlinear window.
- Completed first spectrum-shape gate:
  - added `tools/plot_quasilinear_spectrum_shape_gate.py`;
  - added tests with a synthetic resolved nonlinear NetCDF fixture;
  - generated `docs/_static/quasilinear_hsx_spectrum_shape_gate.{png,pdf,json}`;
  - the HSX gate compares normalized `heat_flux_weight_total` to nonlinear `Diagnostics/HeatFlux_kyst` and passes with total-variation distance about `0.11` and cosine similarity about `0.97`.
- Current next best steps:
  - add a NetCDF nonlinear-window ingestion path so W7-X can enter the same quasilinear calibration machinery without manual CSV conversion.
  - extend the spectrum-shape gate to Cyclone/Cyclone Miller/KBM where resolved `HeatFlux_kyst` artifacts are available and document case-specific gates.
- Completed NetCDF nonlinear-window ingestion:
  - `calibration_point_from_nonlinear_window_summary` now supports diagnostics CSV and runtime NetCDF summaries;
  - NetCDF summaries read `Grids/time` and map `heat_flux` to `Diagnostics/HeatFlux_st`, with species summed by default and optional `species_index` for single-species nonlinear targets;
  - `tools/build_quasilinear_calibration_report.py` now exposes `--species-index`;
  - added a synthetic NetCDF calibration test so W7-X-style `.out.nc` summaries are covered by the fast suite.
- Validation:
  - `ruff check src/spectraxgk/quasilinear_calibration.py tests/test_quasilinear_calibration.py tools/build_quasilinear_calibration_report.py` passed.
  - `pytest -q tests/test_quasilinear_calibration.py` passed.
  - `sphinx-build -b html docs docs/_build/html -W -q` passed.
- Completed additional electrostatic spectrum-shape gates:
  - generated `docs/_static/quasilinear_cyclone_miller_spectrum_shape_gate.{png,pdf,json}`;
  - generated `docs/_static/quasilinear_cyclone_spectrum_shape_gate.{png,pdf,json}`;
  - Cyclone Miller passes the initial shape gate with `TV=0.09395` and cosine `0.98254`;
  - Cyclone is intentionally retained as a failed model/window gate with `TV=0.21473` and cosine `0.89643`, just outside the initial thresholds;
  - KBM remains deferred from quasilinear shape-gate claims because the current quasilinear diagnostic validates electrostatic channels only and the KBM lane is electromagnetic.
- Current next best steps:
  - generate a reproducible W7-X electrostatic quasilinear spectrum only after the W7-X imported geometry source is either tracked or regenerated from a tracked recipe;
  - use the new NetCDF nonlinear-window path to add W7-X to the calibration report once that spectrum exists;
  - start the next saturation-model sweep with shape-aware calibration features, because one-constant mixing length fails absolute-flux transfer and Cyclone shape matching.
- Added the reproducible W7-X quasilinear example entry point:
  - added `examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml`;
  - the example uses `geometry.model = "vmec"` and `vmec_file = "$W7X_VMEC_FILE"` so the VMEC-derived geometry cache is regenerated from a declared source instead of relying on ignored local `tools_out` geometry;
  - added a fast config contract test covering HSX and W7-X non-axisymmetric electrostatic quasilinear examples.
- Current next best steps:
  - run the W7-X VMEC-backed quasilinear scan on a machine with `W7X_VMEC_FILE` available and commit the resulting spectrum/report only if the input provenance is replayable;
  - add the W7-X holdout calibration artifact through the new NetCDF nonlinear window path;
  - then implement the first shape-aware saturation-model sweep: compare linear weights, nonlinear `HeatFlux_kyst`, and one- or two-parameter intensity rules across Cyclone, Cyclone Miller, HSX, and W7-X.
- Completed the first W7-X quasilinear NetCDF calibration artifact:
  - ran the six-point W7-X VMEC-backed quasilinear scan on `ssh office` from a fresh `main` clone with `W7X_VMEC_FILE=/home/rjorge/gx_refs/main_clean_20260312/nonlinear/w7x/wout_w7x.nc`;
  - tracked `docs/_static/quasilinear_w7x_spectrum*` and `docs/_static/quasilinear_w7x_spectrum_scan*`;
  - generated `docs/_static/quasilinear_w7x_train_holdout_report.json` and `docs/_static/quasilinear_w7x_train_holdout.{png,pdf,json}` using the NetCDF nonlinear-window ingestion path;
  - generated `docs/_static/quasilinear_w7x_spectrum_shape_gate.{png,pdf,json}`;
  - W7-X absolute flux remains a negative holdout: all six short-window linear branches are stable under `gamma_floor=0`, so the uncalibrated saturated heat-flux prediction is zero against a finite nonlinear window mean of about `5.38`;
  - W7-X normalized spectrum shape passes with `TV=0.05564` and cosine `0.99167`.
- Added combined stellarator holdout artifact:
  - generated `docs/_static/quasilinear_stellarator_train_holdout_points.json`;
  - generated `docs/_static/quasilinear_stellarator_train_holdout_report.json`;
  - generated `docs/_static/quasilinear_stellarator_train_holdout.{png,pdf,json}`;
  - the combined panel uses Cyclone as the single training point and holds out Cyclone Miller, HSX, and W7-X, preserving the failed `calibration_dataset` result in one manuscript-facing figure.
- Improved calibration plotting:
  - the relative-error panel now switches to a log axis when point errors span orders of magnitude, so small failed holdouts remain visible beside very large failed holdouts.
- Current next best steps:
  - implement a shape-aware saturation-model sweep that uses linear weights plus nonlinear `HeatFlux_kyst` intensity information instead of only one constant `C_sat`;
  - add finite-difference/implicit sensitivity checks for the shape-aware quasilinear objective before using it in optimization;
  - keep KBM out of quasilinear claims until electromagnetic quasilinear channels have separate validation gates.
- Completed first saturation-rule sweep diagnostic:
  - added `tools/plot_quasilinear_saturation_rule_sweep.py`;
  - added `tests/test_plot_quasilinear_saturation_rule_sweep.py`;
  - generated `docs/_static/quasilinear_saturation_rule_sweep.{png,pdf,json}`;
  - the sweep fits one scalar on Cyclone and scores Cyclone Miller, HSX, and W7-X for positive-growth mixing length, raw linear heat-flux weight, and an absolute-growth diagnostic;
  - all three rules fail the held-out absolute-flux gate: holdout mean relative errors are about `205`, `25`, and `510`, respectively;
  - the result is a negative model-development diagnostic, not a validated transport model.
- Current next best steps:
  - formulate the next saturation model using actual shape-aware information instead of one global scalar, for example a low-dimensional intensity envelope constrained by nonlinear `HeatFlux_kyst` and tested by leave-one-geometry-out validation;
  - add differentiability checks for any new quasilinear objective before connecting it to `vmec_jax`/`booz_xform_jax`;
  - keep expanding tests around these tools so each publication artifact has a fast synthetic replay gate.
- Promoted sweep diagnostic rules into the differentiable quasilinear objective path:
  - `saturation_amplitude2` now supports `linear_weight` and `absolute_growth_mixing_length` / `abs_growth_mixing_length`;
  - `quasilinear_feature_objective` now supports those rules for feature vectors `[gamma, kperp_eff2, flux_weight]`;
  - added finite-difference derivative gates for the new differentiable rules.
- Current next best steps:
  - connect the saturation-rule sweep tool to the core rule names instead of duplicating formulas;
  - then move to a true low-dimensional shape-aware model with leave-one-geometry-out validation and AD checks.
- Completed the first low-dimensional shape-aware saturation diagnostic:
  - added `shape_aware_power_law_objective` to `spectraxgk.quasilinear` for a differentiable shape-envelope objective using feature vectors `[gamma, kperp_eff2, flux_weight]` and a geometric `ky` reference;
  - added finite-difference derivative coverage for the objective so future geometry/optimization work does not rely on unchecked autodiff plumbing;
  - added `tools/plot_quasilinear_shape_aware_saturation.py` and fast synthetic/replay tests;
  - generated `docs/_static/quasilinear_shape_aware_saturation.{png,pdf,json}` with the `--passed-shape-only` exponent fit so the failed Cyclone shape gate is not used to define the training shape correction.
- Result:
  - the one-exponent nonlinear/quasilinear shape envelope is not sufficient for absolute heat-flux transfer;
  - mean leave-one-geometry-out relative error is about `0.664`, compared with about `0.624` for the linear-weight baseline and about `0.170` for a deliberately simple training-mean null baseline, so the quasilinear variants fail the `0.35` transport gate and do not beat the null baseline;
  - the result is a research-grade negative diagnostic, not a validated quasilinear transport claim.
- Current next best steps:
  - replace the one-exponent shape envelope with a richer but still low-dimensional model that can include branch/state features, stellarator-vs-axisymmetric family features, and uncertainty diagnostics;
  - require any next saturation model to beat both the linear-weight baseline and the training-mean null baseline in leave-one-geometry-out scoring before exposing it in user-facing TOML;
  - connect the accepted objective to finite-difference/implicit AD checks and then to the `vmec_jax` / `booz_xform_jax` differentiable-geometry bridge.
- Tightened the saturation-rule sweep and started repository-trim hygiene:
  - added a training-mean null baseline to `tools/plot_quasilinear_saturation_rule_sweep.py`;
  - regenerated `docs/_static/quasilinear_saturation_rule_sweep.{png,pdf,json}`;
  - the Cyclone-trained null has holdout mean relative error about `0.372`, far better than the tested one-scalar quasilinear rules, so future saturation candidates must beat it before promotion;
  - added `tools/audit_repository_size.py` and a fast test so tracked-file and local-artifact size audits are reproducible;
  - added the repository trim/history-rewrite policy above.
- Current next best steps:
  - design the next saturation candidate around physically meaningful branch/state features and uncertainty intervals, then require improvement over both the linear-weight baseline and the null baseline;
  - perform the first non-destructive repo trim by moving nonessential high-resolution docs/static artifacts to a release manifest while preserving lightweight docs previews;
  - add a CI size guard after the manifest exists.
- Added executable promotion and repository-size gates:
  - `tools/plot_quasilinear_saturation_rule_sweep.py` now writes a `promotion_gate` with `accepted_rules=[]` for the current one-scalar sweep;
  - `tools/plot_quasilinear_shape_aware_saturation.py` now writes a `promotion_gate` requiring the shape-aware model to beat the linear-weight and training-mean null baselines;
  - regenerated both quasilinear saturation JSON/PNG/PDF artifacts with those machine-readable rejection gates;
  - added `tools/repository_size_manifest.toml` and `tools/check_repository_size_manifest.py`;
  - wired the repository-size manifest checker into CI as `repo-hygiene` and added the checker tests to the fast matrix.
- Current next best steps:
  - implement the release-artifact manifest for moving high-resolution documentation panels out of Git while keeping lightweight previews;
  - prototype the next saturation candidate only as a rejected-or-promoted report, with uncertainty intervals and leave-one-geometry-out scoring against both baselines;
  - keep unvalidated quasilinear candidates out of TOML/runtime user-facing options until `promotion_gate.passed` is true.
- Added the release-artifact and uncertainty-candidate gates:
  - added `tools/release_artifact_manifest.toml` and `tools/check_release_artifact_manifest.py`;
  - the release manifest currently tracks 10 large assets, with about `12.6 MB` planned for GitHub-release migration and the `3.9 MB` HSX VMEC fixture explicitly kept in-repo until a smaller fixture exists;
  - wired the release-artifact checker into CI `repo-hygiene` and the fast test matrix;
  - added `tools/plot_quasilinear_candidate_uncertainty.py`;
  - generated `docs/_static/quasilinear_candidate_uncertainty.{png,pdf,json}`;
  - candidate uncertainty gate result: calibrated linear-weight mean error is about `0.624` with `0.75` interval coverage; shape-power-law mean error is about `0.664` with `0.75` interval coverage; training-mean null remains about `0.170`; `promotion_gate.passed=false` and no candidates are accepted.
- Current next best steps:
  - create compressed lightweight previews for the release-manifest assets, then move high-resolution companions to a GitHub release in a dedicated artifact migration commit;
  - expand the quasilinear candidate family only after adding more held-out nonlinear cases or additional physics features, because the current four-case set is too small for a trustworthy calibrated saturation claim;
  - keep the next candidate report focused on branch/state features and uncertainty intervals rather than adding runtime/TOML exposure.
- Completed first non-destructive documentation-preview trim:
  - added `tools/compress_release_previews.py` plus a fast compression test;
  - extended `tools/check_release_artifact_manifest.py` with `keep_preview_in_repo` so lightweight docs previews can be tracked separately from release-only high-resolution artifacts;
  - compressed six checked-in PNG documentation previews from `10,486,731` bytes to `1,784,130` bytes while keeping the rendered README/docs figures available in Git;
  - removed those compressed PNGs from the repository-size whitelist because they are now below the `1 MB` per-file guard;
  - documented the reproducible preview-compression command in `docs/code_structure.rst`.
- Current next best steps:
  - move release-only PDF/high-resolution companions to GitHub release assets once stable asset URLs exist, then update `tools/release_artifact_manifest.toml`;
  - avoid any git-history rewrite until after non-destructive trimming is complete and collaborators explicitly agree to a coordinated reclone/reset window;
  - continue the quasilinear model-development lane only with rejected-or-promoted reports that include uncertainty intervals, leave-one-geometry-out scoring, and a null-baseline comparison.
- Added a branch/state quasilinear candidate without promoting it:
  - extended `tools/plot_quasilinear_candidate_uncertainty.py` with a `linear_state_ridge` candidate based only on linear-spectrum features: `log_linear_weight`, `log_abs_growth_mixing_length`, `unstable_weight_fraction`, and `log_weighted_ky_centroid`;
  - added candidate-specific eligibility diagnostics for training cases per fitted parameter and ridge normal-matrix conditioning;
  - regenerated `docs/_static/quasilinear_candidate_uncertainty.{png,pdf,json}`;
  - the linear-state ridge candidate has mean leave-one-geometry-out relative error about `0.173` and interval coverage `1.0`, but `promotion_eligible=false` because the current four-case dataset is under-sampled for the five-parameter ridge fit;
  - `promotion_gate.passed=false` remains unchanged, preventing user-facing runtime/TOML exposure.
- Current next best steps:
  - add at least two additional nonlinear holdout cases before attempting to promote any branch/state quasilinear saturation candidate;
  - keep the next quasilinear model-development figure focused on whether new cases reduce uncertainty and beat the training-mean null without overfitting;
  - perform release-only PDF migration only after GitHub Release asset URLs are available.
- Added external VMEC equilibrium portfolio inventory for the next holdout campaign:
  - added `tools/plot_vmec_jax_equilibrium_inventory.py` and a synthetic NetCDF replay test;
  - generated `docs/_static/vmec_jax_equilibrium_inventory.{png,pdf,json}` from `/Users/rogeriojorge/local/vmec_jax/examples/data`;
  - the inventory finds `10` local VMEC equilibria and recommends Li383, nfp4 QH, CTH-like, shaped tokamak, circular tokamak, DSHAPE, and purely toroidal cases as the next linear/nonlinear holdout candidates;
  - the low-resolution QA fixture is now explicitly deferred because its VMEC reference-scale metadata are degenerate and trigger the current VMEC-EIK path's reference-length division guard;
  - the JSON explicitly marks this as `equilibrium_selection_not_transport_validation`, so none of these cases enter quasilinear calibration until matched nonlinear heat-flux windows and physics gates exist.
- Completed bounded external-VMEC linear smoke checks:
  - used `examples/linear/non-axisymmetric/runtime_w7x_linear_quasilinear_vmec.toml` with `W7X_VMEC_FILE` pointed at `vmec_jax/examples/data`, `ky=0.10,0.20`, `Nl=2`, `Nm=4`, `dt=0.005`, and `80` steps;
  - finite stable branches were obtained for `wout_li383_low_res.nc` (`gamma=-0.0258,-0.0297`), `wout_nfp4_QH_warm_start.nc` (`gamma=-0.0243,-0.0186`), `wout_cth_like_fixed_bdy.nc` (`gamma=-0.0230,-0.0282`), and `wout_shaped_tokamak_pressure.nc` (`gamma=-0.0669,-0.0562`);
  - `wout_basic_non_stellsym_simsopt.nc` is not yet a holdout candidate because the current centered flux-tube cut finds no crossing for that fixture;
  - all successful smoke checks are stable under the current short setup, so their positive-growth mixing-length saturated flux is zero by construction and they are not quasilinear transport validation points.
- Current next best steps:
  - promote Li383 and nfp4 QH to production-resolution linear scans only if branch selection remains finite and interpretable over the full `ky` grid;
  - then generate matched nonlinear windows for the same cases and add them to the leave-one-out calibration reports only if nonlinear comparison/physics gates pass;
  - separately harden the VMEC-EIK geometry path with explicit user-facing errors for degenerate reference scales and no-crossing flux-tube cuts before exposing arbitrary external VMEC files as a polished workflow.
- Completed first full-`ky` external-VMEC linear feasibility scans:
  - Li383 full-`ky` scan stays stable over the six-point stellarator grid (`gamma=-0.0222,-0.0429,-0.0573,-0.0700,-0.0769,-0.0783`);
  - nfp4 QH has stable low-`ky` modes and positive growth at high `ky` (`gamma=-0.0199,-0.0180,-0.0108,+0.00850,+0.0206,+0.0328`), making it the stronger next nonlinear-window candidate;
  - tracked `docs/_static/quasilinear_vmec_jax_qh_linear_spectrum.{png,pdf,json}` plus source CSV companions as a linear-feasibility artifact, explicitly not a transport-validation or calibrated quasilinear claim.
  - added regression coverage for VMEC-EIK invalid-reference-scale and no-crossing errors so arbitrary external VMEC fixtures produce actionable user messages.
- Current next best steps:
  - run a bounded nfp4 QH nonlinear adiabatic-electron pilot with the same VMEC fixture and enough diagnostics to form a window-statistics gate;
  - if QH nonlinear is finite and interpretable, add a QH spectrum-shape gate and leave-one-geometry-out calibration point; otherwise record the failure mode and use Li383/CTH-like as the next production candidate;
  - keep arbitrary external VMEC files behind explicit feasibility and nonlinear-window gates until the geometry cut conventions are validated case by case.
- Completed reduced-grid nfp4 QH nonlinear pilots:
  - local `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05` run is finite to `t=5` and `t=20`;
  - the `t=20` late-half window has mean heat flux `1.78e-4`, final heat flux `3.58e-4`, and a positive late-half heat-flux slope of about `3.1e-5` per time unit;
  - this is a stability/geometry-feasibility result only, not a saturated nonlinear transport window and not a calibration point.
- Completed additional full-`ky` external-VMEC linear feasibility scans:
  - CTH-like has a useful unstable high-`ky` branch (`gamma=-0.0227,-0.0161,+0.00418,+0.0114,+0.0309,+0.0488`);
  - shaped tokamak remains stable over the sampled grid (`gamma=-0.0799,-0.0692,-0.0488,-0.0396,-0.0292,-0.0186`);
  - tracked `docs/_static/quasilinear_vmec_jax_cth_like_linear_spectrum.{png,pdf,json}` plus source CSV companions as another linear-feasibility artifact.
- Current next best steps:
  - run CTH-like as the next reduced-grid nonlinear pilot because its linear branch is stronger than QH on the current grid and it may saturate more clearly;
  - run the nfp4 QH nonlinear lane on office or a bounded local restart to a longer time/window only if it remains under the simulation time cap and produces signs of saturation;
  - after a saturated external-VMEC nonlinear window exists, add a QH/CTH/shape spectrum-shape gate and only then consider extending the leave-one-geometry-out quasilinear calibration set.
- Completed CTH-like reduced-grid nonlinear pilots:
  - local `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05` run is finite to `t=20` and `t=50`;
  - the `t=20` late-half heat-flux mean is `3.76e-4`, but the positive slope `8.33e-5` per time unit shows startup/growth rather than saturation;
  - the `t=50` run remains finite but is still strongly growing, with late-half mean heat flux `0.627`, final heat flux `3.15`, and late-half slope `0.102` per time unit;
  - CTH-like is therefore the strongest current external-VMEC nonlinear candidate, but it is not yet a saturated transport-calibration window.
- Completed office-backed long CTH-like nonlinear feasibility pilot:
  - ran fresh `Nx=Ny=32`, `Nz=24`, `Nl=4`, `Nm=8`, `dt=0.05`, `t=100` and `t=150` fixed-step RK3 pilots on office GPU from the clean `main` clone at `/home/rjorge/spectraxgk_main_runs`;
  - the `t=100` run is finite and enters a high-heat-flux state, but late-window slope diagnostics were mixed across windows;
  - the `t=150` run is finite with `61` samples, final heat flux `26.27`, final `Wphi=7.80`, and least-trending window `t=75.05..150.00` with mean heat flux `23.06`, standard deviation `1.79`, and relative heat-flux trend `1.20e-3` per time unit;
  - added `tools/plot_nonlinear_feasibility_pilot.py`, tests, and tracked `docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.{png,pdf,json,traces.csv}`;
  - the panel is deliberately labeled `finite_long_nonlinear_feasibility_not_transport_validation`; the promotion gate remains false until a production-resolution convergence/reference acceptance protocol is defined and passed.
- Current next best steps:
  - define the external-VMEC nonlinear acceptance protocol for CTH-like: production grid, window-start rule, trend tolerance, resolved-spectrum shape gate, and optional independent reference;
  - run a bounded CTH-like convergence check (`32^2x24` vs `48^2x32`, same `Nl/Nm`, same `dt`) before admitting it as a quasilinear calibration holdout;
  - only after that, add a CTH-like spectrum-shape gate against the linear quasilinear spectrum and a leave-one-geometry-out calibration point;
  - continue keeping these external-VMEC pilots out of quasilinear calibration until a saturated nonlinear window and spectrum-shape gate pass.
- Completed first CTH-like grid-convergence check:
  - ran office GPU `Nx=Ny=48`, `Nz=32`, `Nl=4`, `Nm=8`, `dt=0.05`, `t=150` with the same external VMEC fixture;
  - run completed in about `211 s`, stayed finite, and produced `61` diagnostic samples;
  - on the common `t=75.05..150.00` window, higher-grid mean heat flux is `12.82` versus `23.06` on the `32^2x24` run;
  - the higher-grid least-trending `t=120.05..150.00` window has mean heat flux `14.54`, standard deviation `0.271`, and relative trend `5.77e-4` per time unit;
  - tracked `docs/_static/external_vmec_cth_like_nonlinear_t150_n48_pilot.{png,pdf,json,traces.csv}`;
  - conclusion: CTH-like is finite and promising but not grid-converged; it remains excluded from quasilinear calibration.
- Current next best steps:
  - decide the production CTH-like nonlinear protocol before more long runs: either increase grid again, adjust dissipation/hypercollision with a documented physics rationale, or defer external-VMEC transport holdouts until a reference exists;
  - if continuing CTH-like, run one controlled parameter at a time and require late-window heat flux to be stable under both window choice and grid refinement;
  - otherwise return to the core quasilinear model lane with the current four validated nonlinear cases and keep external VMEC as a documented future holdout campaign.
- Closed the first explicit external-VMEC nonlinear convergence-gate slice:
  - added `tools/plot_external_vmec_nonlinear_convergence_gate.py` plus fast synthetic tests;
  - generated `docs/_static/external_vmec_cth_like_grid_convergence_gate.{png,pdf,json,csv}` from the existing `32^2x24` and `48^2x32` CTH-like `t=150` pilots;
  - the gate is intentionally excluded from the release validation index (`gate_index_include=false`) because it documents a negative research result rather than a release blocker;
  - gate outcome: common-window heat-flux symmetric relative difference `0.571`, least-trending-window difference `0.453`, threshold `0.15`; common-window relative trend also fails (`5.42e-3` per time unit versus `2e-3`);
  - conclusion unchanged but now machine-readable: CTH-like is finite and promising, but it is not admitted into quasilinear calibration until the grid/window protocol passes.
- Current next best steps:
  - return to the core quasilinear saturation-model lane using the currently validated nonlinear holdouts, because the external CTH-like holdout failed convergence;
  - if external-VMEC holdouts remain a priority, choose one controlled next run: either increase the CTH-like grid again, vary hypercollision/dissipation with a documented physics rationale, or choose a different VMEC candidate with a stronger reference basis;
  - add any future external-VMEC nonlinear candidate through the same pilot -> convergence-gate -> spectrum-shape-gate path before using it for calibration or optimization claims.
- Added a calibration-admission guard so the workflow always uses validated nonlinear inputs:
  - added `tools/check_quasilinear_calibration_inputs.py` and tests covering passed gates, missing gates, non-required audit points, and failed external-pilot promotion gates;
  - generated `docs/_static/quasilinear_validated_calibration_inputs.{png,pdf,json}` from the current Cyclone/Cyclone-Miller/HSX/W7-X train-holdout reports;
  - current audit passes: every train/holdout point maps to a passed nonlinear gate (`cyclone_nonlinear_long_window`, `cyclone_miller_nonlinear_window`, `hsx_nonlinear_window`, `w7x_nonlinear_window`);
  - this makes the validation/convergence policy executable: failed finite pilots can be documented, but they cannot silently enter quasilinear calibration.
- Current next best steps:
  - use the validated-input audit as a required precondition for future quasilinear saturation-model figures and optimization examples;
  - continue the saturation-model lane only on the validated four-case set unless a new nonlinear case passes pilot, convergence, and validation gates;
  - add the next candidate model only with a null baseline, leave-one-geometry-out scoring, prediction intervals, and finite-difference/autodiff checks before making any optimization-facing claim.
- Promoted the validated-input policy into CI:
  - added a tracked-regression test that recomputes the current quasilinear train/holdout input audit and requires all `12/12` train/holdout usages to map to passed nonlinear gates;
  - added a docs/packaging CI step that runs `tools/check_quasilinear_calibration_inputs.py --no-plot` on the current quasilinear train/holdout reports before building docs;
  - this makes the rule enforceable for future pull requests: calibration docs and figures cannot be rebuilt successfully if a non-converged or exploratory nonlinear artifact is inserted as train/holdout data.
- Current next best steps:
  - build the next saturation-model diagnostic on the validated four-case set, with the CI admission gate as a precondition;
  - keep external VMEC candidates out of calibration until a new candidate passes pilot, convergence, and validation gates;
  - when adding any new nonlinear reference, update both the nonlinear gate JSON and the quasilinear input audit before using it in a model figure.
- Propagated validated-input enforcement into quasilinear model-development scripts:
  - `tools/plot_quasilinear_saturation_rule_sweep.py`, `tools/plot_quasilinear_shape_aware_saturation.py`, and `tools/plot_quasilinear_candidate_uncertainty.py` now require passed nonlinear summary gates by default;
  - regenerated `docs/_static/quasilinear_saturation_rule_sweep.json`, `docs/_static/quasilinear_shape_aware_saturation.json`, and `docs/_static/quasilinear_candidate_uncertainty.json` with `input_validation.passed=true` for the validated four-case set;
  - added tests proving failed nonlinear summary gates are rejected and synthetic tests carry explicit passed gates.
- Current next best steps:
  - proceed to the next quasilinear saturation-candidate improvement only within the validated four-case dataset;
  - add at least one additional converged nonlinear holdout before promoting any higher-parameter candidate beyond a negative/model-development result;
  - keep the finalization plan split into release-ready validated tooling versus post-release external-VMEC/stellarator-optimization expansion.
- Added the W7-X fluctuation-spectrum simulation diagnostic panel:
  - added `tools/plot_w7x_fluctuation_spectrum_panel.py` plus a synthetic NetCDF regression test;
  - generated `docs/_static/w7x_fluctuation_spectrum_panel.{png,pdf,json,csv}` from the gated W7-X nonlinear `t≈200` NetCDF artifact;
  - the report requires the W7-X nonlinear gate summary to pass by default, records `claim_level="validated_nonlinear_simulation_spectrum_not_experimental_validation"`, and sets `gate_index_include=false`;
  - the current diagnostic has `76` samples over `t≈0.065..197.77`, dominant nonzonal `|\phi|^2` at `k_y rho_i≈0.190`, dominant heat-flux power near `k_y rho_i≈1.286`, and dominant zonal `k_x rho_i≈-0.100`;
  - documentation now treats the simulation-spectrum estimator as closed while leaving Doppler-reflectometry transfer-function validation as a post-release manuscript extension.
- Current next best steps:
  - explicitly keep W7-X zonal long-window recurrence/damping as a deferred manuscript blocker rather than a release blocker;
  - run bounded docs/build/package checks for the new spectrum artifact and push the release-scope updates;
  - wait for GitHub CI to go green before any version bump, tag, or PyPI release.
- Started the post-v1.5 open-lane execution pass:
  - added `tools/build_open_research_lane_status.py` to read the current W7-X zonal, W7-X fluctuation-spectrum, quasilinear holdout, differentiable-geometry, and nonlinear-profiler artifacts and summarize each lane as `closed`, `partial`, `open`, or `blocked`;
  - generated `docs/_static/open_research_lane_status.{png,pdf,json,csv}` as a claim-scope figure and machine-readable dashboard;
  - current executable status: W7-X long-window zonal recurrence/damping is open with seven failed reference/envelope gates; quasilinear absolute-flux promotion is open with three current holdouts but `passed=false`; W7-X fluctuation spectra, the differentiable-geometry bridge, and nonlinear profiler identity artifacts are partial bounded diagnostics;
  - updated the validation coverage manifest, roadmap, and manuscript figure index so these lanes cannot be accidentally described as closed physics claims.
- Current next best steps:
  - run the next W7-X zonal physical closure sweep on office using one controlled knob at a time (moment resolution, Hermite/Laguerre hypercollision source, then time horizon) and admit a candidate only if residual, late-envelope, and moment-tail gates improve together;
  - keep CTH-like external VMEC excluded and choose either a higher-grid CTH run or a different VMEC candidate for the next converged nonlinear holdout;
  - extend the W7-X fluctuation/TEM lane with multi-alpha/multi-surface kinetic-electron scans before broad W7-X claims;
  - connect real in-memory `vmec_jax`/`booz_xform_jax` output into `FluxTubeGeometryData` and add parity plus gradient gates before optimization claims;
  - collect matched CPU/GPU profiler traces before making any new speedup claim.
- Advanced the W7-X zonal physical-closure lane with a bounded office-GPU constant-Hermite-hypercollision probe:
  - ran paper-facing W7-X test-4 `k_x rho_i=0.07`, `Nl=16`, `Nm=64`, `dt=0.05`, `t v_t/a=100` for `nu_hyper_m=0.01` and `0.03` with constant hypercollision source;
  - generated `docs/_static/w7x_zonal_hypercollision_probe_kx070.{png,pdf,json,csv}` and wired the result into `tools/build_open_research_lane_status.py`;
  - result is negative but useful: `nu_hyper_m=0.03` lowers final Hermite-tail fraction to about `0.099` and free-energy ratio to about `0.600`, but mean trace error remains about `0.289` and the late-window standard-deviation ratio is about `4.28`, so the W7-X zonal lane remains open physically;
  - updated roadmap/testing/manuscript docs and the validation coverage manifest so the result is tracked as an open recurrence/closure diagnostic, not a validation closure.
- Current next best steps:
  - W7-X zonal: stop increasing constant damping alone; test one physically motivated closure/operator change at a time and require residual, late-envelope, and moment-tail gates to improve together before promotion;
  - W7-X fluctuation/TEM: extend the existing simulation-spectrum panel to multi-alpha/multi-surface and kinetic-electron/TEM scans before broad W7-X claims;
  - quasilinear holdouts: add a new nonlinear holdout only after pilot, window, grid-convergence, and spectrum-shape gates pass; keep failed CTH-like external VMEC out of calibration;
  - differentiable geometry: connect a real in-memory `vmec_jax`/`booz_xform_jax` state into `FluxTubeGeometryData` and add geometry parity plus gradient gates before optimization claims;
  - performance: collect matched CPU/GPU profiler traces for nonlinear bracket/field-solve hot paths before making any speedup claim.
- Added an executable W7-X fluctuation/TEM extension-status gate:
  - added `tools/build_w7x_tem_extension_status.py` and `tests/test_build_w7x_tem_extension_status.py`;
  - generated `docs/_static/w7x_tem_extension_status.{png,pdf,json,csv}` from the current W7-X nonlinear fluctuation panel and `docs/_static/tem_mismatch_table.csv`;
  - current result: W7-X simulation-spectrum estimator is closed (`76` samples, dominant nonzonal `ky rho_i≈0.190`), but TEM linear parity remains open with max absolute relative growth mismatch `4.254` and frequency mismatch `351.534`; multi-alpha/multi-surface W7-X scans and kinetic-electron nonlinear windows are explicitly not admitted yet;
  - wired this artifact into `tools/build_open_research_lane_status.py`, docs, and the validation coverage manifest so broad W7-X/TEM claims stay blocked until the missing physics gates exist.
- Current next best steps:
  - fix the TEM branch/frequency mismatch before starting kinetic-electron W7-X nonlinear validation;
  - then run W7-X alpha/surface-resolved linear scans and only promote nonlinear windows after branch, transport-window, and resolved-spectrum gates pass;
  - continue nonlinear holdout expansion through the existing pilot -> convergence -> validation -> calibration-admission path, not by adding unconverged external VMEC runs;
  - continue geometry bridge and profiler lanes with real parity/gradient/profiler artifacts rather than claim-only docs.
- Refreshed the bounded local CPU nonlinear RHS split profile:
  - reran `tools/profile_nonlinear_step_split.py` for Cyclone grid and spectral Laguerre modes with a hard `240 s` process timeout and `5` warm kernel repeats;
  - regenerated `docs/_static/nonlinear_rhs_profile_cpu.csv`, `docs/_static/nonlinear_rhs_profile_cpu_spectral.csv`, and `docs/_static/nonlinear_rhs_profile.{png,pdf}`;
  - current local CPU numbers: grid `full_rhs≈8.14e-2 s`, spectral `full_rhs≈7.97e-2 s`, nonlinear bracket `3.93e-2 s` vs `2.38e-2 s`; this is useful profiler evidence but not a production speedup claim;
  - documented the bounded CPU refresh in `docs/performance.rst` while keeping spectral Laguerre as opt-in until case-level parity and larger matched CPU/GPU profiler gates support broader claims.
- Current next best steps:
  - run the matching GPU profiler refresh on office for the same script/config before changing performance claims;
  - use those traces to decide whether to optimize bracket FFT/gather paths, field-solve assembly, or cold-start cache construction first.
- Refreshed the matched office GPU nonlinear RHS split profile sequentially after a first parallel attempt showed host/GPU contention:
  - reran grid and spectral Laguerre profiles on one office GPU at a time with `10` warm kernel repeats and `300 s` timeouts;
  - regenerated `docs/_static/nonlinear_rhs_profile_gpu.csv`, `docs/_static/nonlinear_rhs_profile_gpu_spectral.csv`, and the combined `docs/_static/nonlinear_rhs_profile.{png,pdf}`;
  - current office GPU numbers: grid `full_rhs≈1.14e-2 s`, spectral `full_rhs≈7.29e-3 s`, nonlinear bracket `3.15e-3 s` vs `1.52e-3 s`;
  - kept the claim scoped: this supports bracket/spectral-mode profiling, but production speedup claims still require case-level parity gates and larger matched CPU/GPU sweeps.
- Current next best steps:
  - start the TEM branch/frequency audit by comparing the existing reference table, runtime parameters, and fitted signals term by term;
  - if no bounded code fix is obvious, create a machine-readable TEM blocker artifact like the W7-X status artifacts so kinetic-electron W7-X remains gated.
- Added the TEM branch/frequency blocker artifact:
  - added `tools/build_tem_branch_parity_audit.py` and `tests/test_build_tem_branch_parity_audit.py`;
  - regenerated `docs/_static/tem_branch_parity_audit.{png,pdf,json,csv}`, `docs/_static/w7x_tem_extension_status.{png,pdf,json,csv}`, and `docs/_static/open_research_lane_status.{png,pdf,json,csv}`;
  - current audit result: TEM remains open with `max |rel gamma|≈4.254`, `max |rel omega|≈3.3` away from the near-zero reference denominator, `max |Delta gamma|≈1.815`, `max |Delta omega|≈3.976`, one growth-rate sign mismatch, three frequency sign mismatches, and an inverted frequency branch (`Spearman≈-0.986`);
  - quick reduced-moment local probe at `ky=0.3` showed signal choice matters (`phi` gives the wrong frequency sign, electron-density signal is closer but still misses the digitized target), so the next fix must reconstruct the exact TEM case/reference before changing solver physics.
- Current next best steps:
  - keep W7-X/TEM kinetic-electron validation blocked until the full TEM case definition or an independent reference dump is available;
  - resume W7-X zonal physical closure under the paper-facing initializer/observable convention;
  - advance the real `vmec_jax`/`booz_xform_jax` geometry bridge with parity and gradient gates.
- Tightened the W7-X zonal bounded-recurrence artifact:
  - added explicit `tail_std_ratio = tail_std/reference_tail_std` to `tools/plot_w7x_zonal_recurrence_sweep.py` rows and its focused tests;
  - repaired the tracked `docs/_static/w7x_zonal_recurrence_sweep_kx070.{json,csv}` metadata from existing stored row values because the source NetCDF sweep outputs are not present in the local checkout;
  - current best bounded candidate remains `Nl12 Nm48 none` with `mean_abs_error≈0.276`, `tail_std_ratio≈3.78`, and Hermite-tail fraction `≈0.310`; the best constant hypercollision probe still worsens trace error while suppressing Hermite tail, so the lane remains a physical closure issue, not a normalization/doc issue.
- Current next best steps:
  - if the W7-X source NetCDFs are needed for a refreshed plot, rerun the bounded recurrence sweep on office or restore the release artifacts from external storage;
  - test a physically motivated velocity-space closure/operator rather than increasing constant Hermite damping;
  - continue the differentiable geometry bridge and keep W7-X zonal closure open until residual, late-envelope, and moment-tail gates pass together.
- Advanced the differentiable geometry bridge:
  - promoted the real `vmec_jax` boundary-aspect AD-vs-finite-difference check into `spectraxgk.geometry.differentiable.vmec_boundary_aspect_sensitivity_report`;
  - expanded backend auto-discovery to check `~/vmec_jax`, `~/local/vmec_jax`, `~/booz_xform_jax`, and `~/local/booz_xform_jax` in addition to sibling checkouts and explicit environment variables;
  - updated the example to use the package API and refreshed `docs/_static/differentiable_geometry_bridge.json` with the real local `vmec_jax`/`booz_xform_jax` APIs discovered; current real boundary-aspect derivative AD/FD max absolute error is `≈1.97e-10`;
  - added CI-safe fake-backend tests for the VMEC boundary sensitivity bridge, so the API contract is tested even when the real optional backends are not installed.
- Current next best steps:
  - connect a real `vmec_jax` equilibrium output to the sampled `FluxTubeGeometryData` contract, then compare against the existing VMEC/eik import path on a small equilibrium;
  - add a gradient gate for a solver-facing geometric observable after geometry parity passes;
  - keep full stellarator optimization claims scoped until both geometry parity and geometry-gradient gates pass.
- Added the differentiable QA stellarator ITG objective-reduction gate:
  - added `src/spectraxgk/stellarator_optimization.py` with a JAX-native max-mode-1 QA control map targeting aspect `7`, mean iota `0.41`, and smooth quasisymmetry/ITG observables;
  - added three optimization objectives: linear ITG growth rate, differentiable quasilinear ITG heat-flux proxy, and a differentiable nonlinear heat-flux envelope with late-window mean/CV/trend diagnostics;
  - added example scripts under `examples/optimization/` for growth-rate, quasilinear-flux, nonlinear-window, and three-objective comparison workflows;
  - generated `docs/_static/stellarator_itg_growth_optimization.{png,pdf,json,history.csv}`, `docs/_static/stellarator_itg_quasilinear_optimization.{png,pdf,json,history.csv}`, `docs/_static/stellarator_itg_nonlinear_optimization.{png,pdf,json,history.csv}`, and `docs/_static/stellarator_itg_optimization_comparison.{png,pdf,json}`;
  - current result: all three objectives pass AD-vs-finite-difference gates, keep the optimized state near the target aspect/iota constraints, reduce growth to about `57%` of its initial value, and reduce quasilinear plus nonlinear-window heat-flux observables to about `41%` of their initial values;
  - added `tests/test_stellarator_optimization.py` so the objective contracts, gradient gates, nonlinear window metrics, optimizer reduction, UQ/covariance diagnostics, and comparison payload shape remain covered in the fast test suite;
  - documented the workflow in `docs/stellarator_optimization.rst`, README, and `docs/manuscript_figures.rst`.
- Scope note:
  - this closes a differentiable optimization and UQ gate, not the full production `vmec_jax -> booz_xform_jax -> SPECTRAX-GK` nonlinear optimization claim;
  - the full claim still requires in-memory VMEC/Boozer-to-`FluxTubeGeometryData` parity, geometry-gradient gates through production solver caches, removal or isolation of host scalar materialization in traced geometry paths, and converged nonlinear audits of optimized geometries.
- Current next best steps:
  - implement the real `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` bridge using a small equilibrium and compare sampled field-line arrays against the existing VMEC/eik import path;
  - add finite-difference and implicit-eigenpair gradient checks for growth rate, frequency, and quasilinear weights through that bridge;
  - replace the reduced nonlinear envelope with either a trace-safe production nonlinear objective or a documented stochastic/window estimator only after nonlinear identity, convergence, and profiler gates pass;
  - keep W7-X zonal long-window recurrence/damping and TEM branch/frequency blockers deferred while the differentiability lane advances.
- Strengthened the differentiable Boozer bridge beyond import discovery:
  - added `spectraxgk.geometry.differentiable.booz_xform_spectral_sensitivity_report`, a bounded real-API `booz_xform_jax` spectral transform gate on a one-surface axisymmetric bundle;
  - the gate checks the derivative of a Boozer magnetic-spectrum norm with respect to a ripple coefficient against central finite differences when `booz_xform_jax` is available, and reports an explicit unavailable status otherwise;
  - wired the report into the differentiable-geometry example JSON/panel, README, docs, package exports, and fast tests.
- Current next best steps:
  - implement the real `vmec_jax VMECState -> booz_xform_jax -> FluxTubeGeometryData` bridge with a small equilibrium and compare sampled field-line arrays against the existing VMEC/eik import path;
  - after parity passes, add growth-rate/frequency/quasilinear-weight gradient gates through the production linear solver cache;
  - only then promote full stellarator optimization claims beyond the reduced objective-reduction gate.
- Advanced the Boozer-to-flux-tube differentiability bridge:
  - added `evaluate_boozer_bmag_on_field_line`, `booz_xform_flux_tube_mapping_from_inputs`, and `booz_xform_flux_tube_sensitivity_report`;
  - the new bounded gate runs the real `booz_xform_jax` functional transform, samples the Boozer `|B|` spectrum on a field line, builds the solver-ready `FluxTubeGeometryData` mapping, and checks geometry-observable sensitivities against central finite differences;
  - refreshed `docs/_static/differentiable_geometry_bridge.{png,json}` so the publication artifact now reports VMEC boundary AD/FD `≈1.97e-10`, Boozer spectral AD/FD `≈2.88e-12`, and Boozer flux-tube AD/FD max absolute error `≈1.54e-08` / relative error `≈5.21e-08`;
  - added fast tests and public exports for the new bridge.
- Scope note:
  - this is a real `booz_xform_jax` derivative path into the SPECTRAX-GK geometry contract, but it still uses a smooth metric/drift closure; it is not yet full VMEC/Boozer metric parity or a production nonlinear optimization claim.
- Current next best steps:
  - replace the smooth metric/drift closure with sampled VMEC/Boozer metric tensors for a small equilibrium and compare against the existing imported VMEC/eik contract;
  - add geometry-gradient gates through one production linear solve after sampled-array parity passes;
  - then connect quasilinear objective gradients to the full geometry bridge before any nonlinear optimization promotion.
- Advanced the differentiable geometry bridge all the way back to a real
  ``vmec_jax`` state:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_boozer_flux_tube_sensitivity_report``;
  - the new optional gate loads the local ``vmec_jax`` ``circular_tokamak``
    example, perturbs two ``VMECState`` Fourier coefficients
    ``[Rcos(radial_index, mode_index), Zsin(radial_index, mode_index)]``,
    converts the traced state through ``vmec_jax.booz_input`` and
    ``booz_xform_jax``, samples the resulting Boozer ``|B|`` along a field
    line, builds the SPECTRAX-GK ``FluxTubeGeometryData`` input mapping, and
    checks geometry-observable sensitivities against central finite
    differences;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}``;
    current VMEC-state-to-Boozer-to-SPECTRAX AD/FD max absolute error is
    ``5.77e-7`` and max relative error is ``1.39e-8``;
  - updated the open-lane dashboard so the remaining differentiable-geometry
    action is sampled VMEC/Boozer metric/drift parity and solver-observable
    gradient gates, not the initial VMEC-state plumbing.
- Scope note:
  - this is now a real ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK``
    differentiability gate for the tracked geometry observables;
  - it still uses the current smooth metric/drift closure, so it is not yet a
    production VMEC/Boozer metric-parity claim and not yet a nonlinear
    stellarator-optimization claim.
- Current next best steps:
  - replace the smooth closure in the bridge with sampled VMEC/Boozer metric
    tensors for a small equilibrium;
  - compare sampled arrays against the imported VMEC/eik runtime path;
  - add finite-difference or implicit-eigenpair checks for linear growth,
    frequency, and quasilinear weights through the production solver cache;
  - only then connect the QA optimization examples to the full geometry bridge
    and keep nonlinear heat-flux optimization gated on converged nonlinear
    windows.
- Added the upstream VMEC metric-tensor differentiability gate:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_metric_tensor_sensitivity_report`` and
    ``vmec_metric_tensor_observable_names``;
  - the gate loads the same real ``vmec_jax`` ``circular_tokamak`` state,
    perturbs the VMEC Fourier coefficients, evaluates
    ``vmec_jax.geom.eval_geom``, and checks sampled covariant metric/Jacobian
    observables ``[sqrtg_rms, mean_g_ss, mean_g_tt, mean_g_pp, g_st_rms,
    g_sp_rms, g_tp_rms]`` against central finite differences;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}``;
    current VMEC metric-tensor AD/FD max absolute error is ``5.88e-8`` and max
    relative error is ``1.30e-7``;
  - updated the open-lane dashboard and docs so the bridge status records both
    real VMEC metric derivatives and real VMEC-state-to-Boozer derivatives.
- Scope note:
  - the path now differentiates through real ``vmec_jax`` state geometry and
    real ``booz_xform_jax`` spectra, but the SPECTRAX-GK field-line mapping
    still uses a smooth placeholder for metric/drift closure;
  - production stellarator optimization remains gated on replacing that closure
    with sampled VMEC/Boozer field-line tensors and matching the imported
    VMEC/eik runtime path.
- Current next best steps:
  - build a small-equilibrium sampled field-line tensor fixture from the VMEC
    metric output plus Boozer straight-field-line coordinates;
  - compare ``bmag``, ``gradpar``, ``gds2``, ``gds21``, ``gds22``, drifts,
    Jacobian, and ``grho`` against the imported VMEC/eik runtime path;
  - then add production linear/quasilinear AD-vs-FD gates through this fixture.
- Added the first non-axisymmetric VMEC field-line tensor differentiability gate:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_field_line_tensor_sensitivity_report`` and
    ``vmec_field_line_tensor_observable_names``;
  - the gate loads the real ``vmec_jax`` ``nfp4_QH_warm_start`` stellarator
    fixture, perturbs VMEC Fourier coefficients, evaluates
    ``vmec_jax.geom.eval_geom`` plus ``vmec_jax.vmec_bcovar``, samples a fixed
    VMEC field-line convention, and checks ``|B|`` ripple plus raw metric
    tensor observables ``[mean_bmag, relative_bmag_ripple, sqrtg_rms,
    mean_g_tt, mean_g_pp, g_tp_rms, mean_g_ss]`` against central finite
    differences;
  - this explicitly addresses the optimization requirement that
    differentiability reaches the VMEC state and field-line tensors, not only
    the Boozer spectral adapter;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}`` and
    the open-lane dashboard to record this field-line tensor gate.
- Scope note:
  - this is a real upstream VMEC field-line tensor check, but it intentionally
    does not relabel the current smooth SPECTRAX-GK metric/drift closure as
    production-ready;
  - the next production step remains deriving the SPECTRAX-GK ``gds*``, drift,
    Jacobian, ``gradpar``, and ``grho`` arrays from sampled VMEC/Boozer tensors
    and matching those arrays against the imported VMEC/eik path.
- Added the first direct VMEC tensor-to-SPECTRAX flux-tube differentiability gate:
  - fixed optional-backend discovery so explicitly configured/local
    ``vmec_jax`` and ``booz_xform_jax`` checkouts are preferred over globally
    installed packages; this matters because the local ``vmec_jax`` checkout
    carries the example ``wout`` files used by real derivative gates;
  - added ``spectraxgk.geometry.differentiable.vmec_jax_flux_tube_mapping_from_state``
    and ``vmec_jax_flux_tube_sensitivity_report``;
  - the new gate starts from the real ``nfp4_QH_warm_start`` ``VMECState``,
    evaluates ``vmec_jax.geom`` and ``vmec_jax.vmec_bcovar``, samples a fixed
    VMEC field line, inverts the sampled VMEC metric tensor, derives
    SPECTRAX-GK ``bmag``, ``gradpar``, ``gds*``, Jacobian, ``grho``, and a
    local grad-``B`` drift closure, builds the ``FluxTubeGeometryData`` mapping,
    and checks geometry-observable sensitivities against central finite
    differences;
  - focused local gate result: max relative AD/FD error is ``≈1.35e-4`` with
    ``fd_step=2e-6`` on the tracked geometry observables;
  - refreshed the differentiable-geometry example/status tooling so the JSON,
    panel, docs, and open-lane dashboard report the direct VMEC tensor
    flux-tube derivative gate.
- Scope note:
  - this addresses the requirement that differentiability reaches back to
    ``vmec_jax`` state coefficients and not only ``booz_xform_jax`` spectra;
  - this still is not a full production stellarator transport optimization
    claim: the direct VMEC tensor-derived arrays must be compared against the
    imported VMEC/eik runtime path, and the local grad-``B`` drift closure must
    be replaced or matched to the production VMEC/EIK drift convention before
    solver-observable gradient gates are promoted.
- Current next best steps:
  - build a small array-parity harness comparing direct VMEC tensor-derived
    ``bmag``, ``gradpar``, ``gds2``, ``gds21``, ``gds22``, Jacobian, ``grho``,
    and drifts against the imported VMEC/eik path on the same equilibrium;
  - once array parity passes, add finite-difference/implicit-eigenpair gates
    for growth rate, frequency, and quasilinear weights through the production
    solver cache;
  - only after those gates pass connect the QA optimization examples to the
    production ``vmec_jax`` geometry bridge for stellarator optimization.
- Added the first direct VMEC tensor-vs-imported-VMEC/EIK array parity audit:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_flux_tube_array_parity_report``;
  - the audit starts from the same real ``nfp4_QH_warm_start`` ``VMECState``, builds the direct
    ``vmec_jax`` tensor-derived SPECTRAX-GK flux-tube mapping, generates the existing imported
    VMEC/EIK geometry on the same surface, trims the closed endpoint, and compares solver-facing
    arrays ``bmag``, ``bgrad``, ``gds*``, drifts, Jacobian, and ``grho`` plus scalar ``gradpar``,
    ``q``, and ``s_hat``;
  - the current result is intentionally reported as ``diagnostic_open``: ``q`` and ``s_hat`` are
    close, while the metric/Jacobian/``grho``/drift arrays are not yet production-parity because the
    direct path still uses a VMEC-coordinate/equal-theta convention and a local grad-``B`` closure
    rather than the imported Boozer equal-arc/Hegna-Nakajima geometry convention;
  - refreshed the differentiable-geometry artifact and open-lane dashboard so this open production
    gap is visible instead of being hidden behind the successful AD/FD gate.
- Current next best steps:
  - derive the direct JAX path in the same Boozer equal-arc coordinate convention as the imported
    VMEC/EIK generator, or reuse the JAX-native Boozer coefficients with the full metric/drift
    reconstruction instead of only the ``|B|`` spectrum;
  - replace the local grad-``B`` drift closure with the production Hegna-Nakajima/imported-VMEC
    drift convention;
  - rerun the array-parity audit until ``bmag``, ``gradpar``, ``gds*``, Jacobian, ``grho``, and
    drift profiles pass field-level tolerances; only then promote growth-rate, quasilinear-flux,
    and nonlinear-window gradients through the production solver cache.
- Added a Boozer equal-arc core parity gate for the differentiable VMEC bridge:
  - added ``spectraxgk.geometry.differentiable.vmec_jax_boozer_equal_arc_core_profiles_from_state``;
  - the gate starts from a real ``vmec_jax`` ``VMECState``, converts to
    ``booz_xform_jax`` inputs, evaluates the JAX-native Boozer magnetic
    spectrum, applies the imported VMEC/EIK equal-arc remap, and compares
    ``theta``, ``bmag``, ``bgrad``, Jacobian, scalar ``gradpar``, ``q``, and
    ``s_hat`` against the existing imported VMEC/EIK runtime path on the same
    ``nfp4_QH_warm_start`` surface;
  - current artifact result: equal-arc core normalized worst error is
    ``4.46e-3``, scalar worst relative error is ``2.35e-3``, and the
    derivative-like ``bgrad`` worst normalized error is ``2.26e-2`` under its
    separate derivative tolerance;
  - refreshed ``docs/_static/differentiable_geometry_bridge.{png,json}`` and
    the open-lane dashboard so the bridge now distinguishes the closed
    Boozer equal-arc core convention from the still-open full metric/drift
    production parity gap.
- Follow-up from this entry: reconstructing ``gds*``/``grho`` from the
  matched equal-arc path is now closed by the metric subgate below.
- Extended the Boozer equal-arc parity gate to zero-beta metric profiles:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now reconstructs
    ``gds2``, ``gds21``, ``gds22``, and ``grho`` from the real
    ``booz_xform_jax`` Boozer ``R/Z/nu/B`` spectra, radial derivatives, and
    the same equal-arc remap used by the imported VMEC/EIK path;
  - the tracked ``nfp4_QH_warm_start`` artifact now passes a separate metric
    subgate with worst normalized mismatch ``3.45e-2``:
    ``gds2=2.49e-2``, ``gds21=3.45e-2``, ``gds22=3.10e-2``, and
    ``grho=1.58e-2``;
  - this closes the release-tolerance field-line normalization and zero-beta
    metric convention for the differentiable VMEC/Boozer bridge while keeping
    the production Hegna-Nakajima curvature/drift parity gap explicit.
- Current next best steps:
  - broaden the JAX-native Boozer equal-arc drift parity gate beyond the
    current zero-beta tracked fixture to finite-beta pressure corrections and
    additional stellarator equilibria;
  - after the broader finite-beta drift gates pass, add production solver
    gradients for linear growth/frequency, quasilinear weights, and then
    nonlinear-window objectives before making stellarator optimization claims.
- Added the first zero-beta Boozer equal-arc drift parity subgate:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now reconstructs
    loaded-convention ``cvdrift``, ``gbdrift``, ``cvdrift0``, and
    ``gbdrift0`` from the real Boozer ``B`` spectrum, radial ``dB/ds``,
    equal-arc remap, and the same root-level drift-loader factor used by
    ``load_gx_geometry_netcdf``;
  - the tracked ``nfp4_QH_warm_start`` artifact now passes a separate drift
    subgate with worst normalized mismatch ``3.50e-2`` after raising and
    enforcing the default Boozer parity mode count to ``mboz=nboz=21``:
    ``cvdrift=3.50e-2``, ``gbdrift=3.50e-2``, ``cvdrift0=3.03e-2``, and
    ``gbdrift0=3.03e-2``;
  - the release claim is now closed for the tracked zero-beta Boozer
    equal-arc field-line, metric, and drift convention, while finite-beta,
    multi-equilibrium drift parity and solver-objective gradients remain
    required before stellarator heat-flux optimization claims.
- Bounded follow-up probe after the drift subgate:
  - ``LandremanPaul2021_QA_lowres`` is not usable for this runtime EIK parity
    path as shipped in ``vmec_jax`` because its bundled ``wout`` reports
    ``Aminor_p=0`` and the runtime generator correctly rejects it;
  - ``shaped_tokamak_pressure`` passes the small ``ntheta=8`` equal-arc
    core/metric/drift smoke gate, including a drift worst normalized mismatch
    of ``7.10e-3``;
  - ``nfp3_QI_fixed_resolution_final`` passes core/metric smoke gates but fails
    the drift smoke gate with ``mboz=nboz=8`` at worst normalized mismatch
    ``1.82e-1``; increasing the Boozer parity mode count to ``21`` reduces the
    QI drift mismatch to ``7.13e-2`` and passes the release drift tolerance,
    which identifies the immediate issue as spectral truncation rather than a
    normalization change;
  - a trial shear-HNGC correction using the wrong input-convention factor was
    explicitly rejected because it worsened the tracked QH metric gate.
- Enforced ``mboz,nboz >= 21`` for the VMEC/Boozer equal-arc parity helpers so
  future runs do not silently fall back to the under-resolved QI drift setting.
- Added ``tools/build_vmec_boozer_parity_matrix.py`` and
  ``docs/_static/vmec_boozer_parity_matrix.{png,pdf,json,csv}`` to make the
  mode-21 result replayable across the tracked QH, QI, and shaped-tokamak
  examples. All three current rows pass the equal-arc core, scalar, ``bgrad``,
  metric, and drift subgates at ``mboz=nboz=21``; the QI drift row remains the
  limiting release-level value at ``7.13e-2`` against the ``8e-2`` tolerance.

### 2026-04-30

- Added the manuscript-scope readiness dashboard:
  - ``tools/build_manuscript_readiness_status.py`` now reads the quasilinear
    calibration/model-selection artifacts, the differentiable-geometry bridge,
    the mode-21 VMEC/Boozer parity matrix, the reduced stellarator ITG
    optimization comparison, and the nonlinear sharding profiler artifact;
  - ``docs/_static/manuscript_readiness_status.{png,pdf,json,csv}`` records
    the current manuscript claim surface with W7-X zonal recurrence/damping
    and TEM/kinetic-electron stellarator extension explicitly deferred;
  - in this narrower scope, the quasilinear lane is closed as a validated
    diagnostic/model-selection negative result rather than as an absolute-flux
    predictor, VMEC/Boozer equal-arc geometry parity is closed at
    ``mboz=nboz=21``, and reduced differentiable stellarator ITG optimization
    is closed with AD/FD gates;
  - the active manuscript blocker is now production solver-objective geometry
    gradients through the mode-21 VMEC/Boozer bridge, while profiler-backed
    nonlinear speedup claims remain partial and require fresh CPU/GPU profiler
    artifacts before any new performance claim.
- Current next best steps:
  - add a finite-difference/implicit-eigenpair gate for linear growth rate,
    real frequency, and electrostatic quasilinear weights through the matched
    mode-21 VMEC/Boozer geometry bridge;
  - connect that gate to a small, replayable stellarator fixture before
    promoting any full end-to-end stellarator heat-flux optimization claim;
  - keep W7-X zonal recurrence/damping and TEM/kinetic-electron stellarator
    validation as post-manuscript lanes unless the manuscript scope changes.
- Added the first production-adjacent solver-objective geometry-gradient gate:
  - ``src/spectraxgk/solver_objective_gradients.py`` differentiates actual
    electrostatic linear-RHS eigenpair observables with respect to
    solver-ready ``FluxTubeGeometryData`` arrays;
  - ``tools/build_solver_objective_gradient_gate.py`` writes
    ``docs/_static/solver_objective_gradient_gate.{png,pdf,json,csv}``;
  - the gate uses the implicit left/right eigenpair sensitivity system and
    checks ``gamma``, ``omega``, ``<k_perp^2>``, linear heat/particle-flux
    weights, and a mixing-length heat-flux proxy against nearest-branch
    central finite differences;
  - the current gate passes with maximum relative AD/FD error below ``5e-3``
    on the tracked small electrostatic linear-RHS fixture, and the manuscript
    readiness dashboard now marks production solver-objective gradients as
    ``partial`` rather than ``open``;
  - this is intentionally scoped as a solver-ready geometry-gradient gate, not
    yet a full ``vmec_jax -> booz_xform_jax -> SPECTRAX-GK``
    state-coefficient gradient claim.
- Current next best steps:
  - feed actual mode-21 VMEC/Boozer equal-arc state-coefficient perturbations
    into the same solver-objective gradient gate;
  - add a small stellarator linear/quasilinear objective fixture once that
    state-gradient path passes;
  - only after linear/quasilinear state-gradient gates pass, add nonlinear
    window objective gradients or a documented stochastic/finite-difference
    estimator with convergence diagnostics.
- Added the first full VMEC/Boozer state-to-solver gradient gate:
  - fixed a real traceability bug in ``build_linear_cache`` where sampled
    magnetic shear was host-materialized with ``float(...)`` even on
    non-twist-shift grids; periodic sampled geometry now remains
    differentiable in traced ``s_hat``, while traced shear under twist-shift is
    rejected explicitly because that path still changes host topology
    (``jtwist``/``x0``);
  - added ``mode21_vmec_boozer_linear_frequency_gradient_report`` and
    ``tools/build_vmec_boozer_solver_frequency_gradient_gate.py``;
  - generated
    ``docs/_static/vmec_boozer_solver_frequency_gradient_gate.{png,pdf,json,csv}``;
  - the gate starts from a real ``vmec_jax`` ``VMECState`` coefficient,
    converts through ``booz_xform_jax`` with ``mboz=nboz=21``, builds the
    SPECTRAX-GK linear RHS, and checks the eigenfrequency gradient with the
    implicit left/right eigenpair method against central finite differences;
  - the current artifact passes with maximum relative AD/FD error
    ``4.89e-3`` for the frequency derivative at conditioned step ``1e-6``.
- Scope note:
  - this closes the full-chain linear eigenfrequency-gradient trace from
    ``vmec_jax`` state coefficients through the solver;
  - it does **not** yet close full-chain quasilinear flux-weight gradients or
    nonlinear-window gradients. Two bounded quasilinear-observable attempts
    timed out at 180--240 seconds, so the next step is profiling/conditioning
    that heavier eigenvector-dependent diagnostic before using it in the
    manuscript.
- Current next best steps:
  - profile and reduce the full-chain quasilinear flux-weight state-gradient
    gate, likely by caching/reusing VMEC/Boozer geometry products and reducing
    repeated objective-time geometry reconstruction;
  - add a publication figure comparing solver-ready versus full-chain
    frequency-gradient conditioning;
  - then connect the differentiable stellarator optimization examples to the
    passed full-chain linear gate while keeping quasilinear and nonlinear
    optimization claims scoped until their gates pass.
- Closed the full-chain VMEC/Boozer quasilinear-gradient gate for reduced
  stellarator objectives:
  - replaced the implicit eigenpair observable Jacobian with a split chain
    rule in ``spectraxgk.autodiff_validation``. The math is unchanged, but the
    expensive VMEC/Boozer parameter derivative is now evaluated only along the
    actual parameter directions instead of being carried through every
    eigenvector-component tangent;
  - added ``mode21_vmec_boozer_quasilinear_gradient_report`` and
    ``tools/build_vmec_boozer_quasilinear_gradient_gate.py``;
  - generated
    ``docs/_static/vmec_boozer_quasilinear_gradient_gate.{png,pdf,json,csv}``;
  - the tracked gate uses ``mboz=nboz=21`` and a richer ``Nl=2, Nm=3`` moment
    basis so the electrostatic heat-flux weight is nonzero;
  - the gate checks ``gamma``, ``omega``, ``<k_perp^2>``, linear ion
    heat-flux weight, and ``gamma Q_i/k_perp^2`` against nearest-branch
    central finite differences and passes with maximum relative error
    ``4.28e-3`` in about 34 seconds on the local CPU.
- Scope note:
  - this closes the reduced linear/quasilinear full-chain stellarator
    objective-gradient path from a real ``vmec_jax`` state coefficient through
    ``booz_xform_jax`` and the SPECTRAX-GK linear solver;
  - it still does **not** close full nonlinear-window heat-flux gradients or
    broad multi-equilibrium stellarator transport-gradient optimization.
- Current next best steps:
  - add a second VMEC/Boozer equilibrium to the quasilinear-gradient gate
    before making broad stellarator-transport-gradient claims;
  - connect the stellarator optimization examples to the passed full-chain
    linear/quasilinear gate while preserving the reduced/nonlinear claim
    boundary;
  - keep W7-X zonal recurrence/damping and TEM/kinetic-electron validation
    deferred for the current manuscript as previously agreed.
- Added memory-bounded Boozer surface-stencil support for large-equilibrium
  diagnostics:
  - ``vmec_jax_boozer_equal_arc_core_profiles_from_state`` now accepts
    ``surface_stencil_width`` and forwards a local radial stencil to
    ``booz_xform_jax`` when requested;
  - the VMEC/Boozer frequency and quasilinear gradient-gate tools expose the
    same option, while preserving the all-surface default used by the
    published QH artifact;
  - a mocked backend regression test checks that the selected Boozer surfaces
    are passed through and recorded in the returned metadata.
- Exposed ``radial_index``, ``mode_index``, and ``surface_index`` controls in
  the VMEC/Boozer frequency and quasilinear gradient-gate APIs/tools. This is
  needed for reviewer-proof conditioning scans because a broad VMEC/Boozer
  gradient claim should not depend on one hard-coded Fourier coefficient.
- Office GPU holdout diagnostics:
  - ``nfp3_QI_fixed_resolution_final`` now runs with the stencil and
    ``mboz=nboz=21`` without the earlier all-surface OOM, but both the
    frequency and quasilinear AD-vs-FD gates fail by order-unity relative
    errors. Enabling ``JAX_ENABLE_X64=1`` does not change the result, so this
    is a stencil/conditioning blocker rather than float32 finite-difference
    noise.
  - ``LandremanPaul2021_QA_lowres`` is a better small bundled QA holdout by
    radial count, but the all-surface Boozer transform still exceeds the
    available office GPU memory at ``mboz=nboz=21``. It should be retried only
    after a memory-reduced Boozer transform or bounded CPU/offline artifact
    path is available.
- Current manuscript stance:
  - reduced full-chain linear/quasilinear differentiability is closed for the
    tracked all-surface QH fixture;
  - multi-equilibrium transport-gradient promotion remains open;
  - nonlinear-window VMEC/Boozer gradients remain future work;
  - W7-X zonal and TEM remain explicitly deferred for this manuscript.
- Validation for this slice:
  - ``python -m ruff check src/spectraxgk/geometry/differentiable.py src/spectraxgk/solver_objective_gradients.py tools/build_vmec_boozer_quasilinear_gradient_gate.py tools/build_vmec_boozer_solver_frequency_gradient_gate.py tests/test_differentiable_geometry_bridge.py tests/test_solver_objective_gradients.py``
  - ``PYTHONPATH=src python -m mypy src/spectraxgk/geometry/differentiable.py src/spectraxgk/solver_objective_gradients.py tools/build_vmec_boozer_quasilinear_gradient_gate.py tools/build_vmec_boozer_solver_frequency_gradient_gate.py``
  - ``python -m pytest tests/test_differentiable_geometry_bridge.py::test_vmec_jax_boozer_equal_arc_core_profiles_supports_surface_stencil tests/test_solver_objective_gradients.py tests/test_build_manuscript_readiness_status.py -q``
  - ``PYTHONPATH=src python -m sphinx -b html -W docs docs/_build/html``
- Current next best steps:
  - build a memory-aware multi-equilibrium VMEC/Boozer gradient artifact path
    before broadening transport-gradient claims beyond QH;
  - continue quasilinear nonlinear-holdout validation and saturation-model
    calibration with explicit train/holdout splits;
  - strengthen the stellarator optimization examples by adding nonlinear
    audit bars for optimized geometries and finite-difference conditioning
    reports for each differentiated observable.
- CI coverage follow-up:
  - the public ``wide-coverage`` combine job failed at ``92%`` after the
    optional VMEC/Boozer artifact builders were added to the package-wide
    denominator, while the public CI cannot install or execute the local
    ``vmec_jax``/``booz_xform_jax`` repositories required by those paths;
  - scoped the external-backend artifact builders out of the default coverage
    denominator and documented that policy in ``docs/testing.rst``;
  - added fast low-level geometry helper tests and mocked optional-backend
    contract tests so the public suite still exercises radial interpolation,
    radial derivatives, periodic Boozer sampling, parity metrics, unavailable
    backend reporting, and solver-objective parameter naming;
  - local focused coverage for the affected modules is now ``96%`` for
    ``geometry/differentiable.py`` and ``100%`` for
    ``solver_objective_gradients.py`` on the bounded targeted test pair.
- Differentiable stellarator optimization UQ tightening:
  - while adding a paper-facing UQ/sensitivity panel, found that the previous
    optimization covariance estimate used the initial-to-final observable
    displacement as the residual, which measures optimizer travel rather than
    local uncertainty at the optimized point;
  - added ``stellarator_itg_objective_residual_vector`` and residual-name
    metadata so the covariance is now computed from the final weighted
    objective residual Jacobian, whose squared norm is the scalar objective;
  - added ``tools/plot_stellarator_optimization_uq.py`` and
    ``docs/_static/stellarator_itg_optimization_uq.{png,pdf,json}`` showing
    AD/FD derivative parity, local control uncertainty, covariance projection,
    and rank/conditioning diagnostics for the growth, quasilinear, and
    nonlinear-window reduced objectives;
  - updated README, manuscript-figure docs, and stellarator-optimization docs
    to keep the claim scoped to reduced optimization/UQ plumbing until
    production VMEC/Boozer/GK nonlinear audits pass.
- Quasilinear dataset-sufficiency promotion gate:
  - added ``tools/plot_quasilinear_dataset_sufficiency.py`` and
    ``docs/_static/quasilinear_dataset_sufficiency.{png,pdf,json}``;
  - the gate validates that every current quasilinear model-development input
    maps to a passed nonlinear summary gate, records the four
    electrostatic-compatible cases, and explicitly lists KBM as validated but
    excluded because the present quasilinear diagnostics are electrostatic;
  - the tracked promotion gate is intentionally blocked: current total cases
    are ``4`` versus a ``6``-case minimum, explicit training geometries are
    ``1`` versus a ``2``-geometry minimum, and downstream candidate skill gates
    remain false;
  - the figure and JSON now prevent the docs/README from accidentally
    promoting richer quasilinear absolute-flux candidates before more
    converged nonlinear holdouts exist.
- Multi-equilibrium VMEC/Boozer solver-objective gradient holdout:
  - added Li383 low-resolution holdouts for both the mode-21 full-chain
    eigenfrequency gate and the full-chain quasilinear heat-flux-weight gate;
  - both Li383 gates use ``mboz=nboz=21`` and pass the same AD-vs-central-
    finite-difference objective checks as the tracked QH fixture;
  - added ``tools/build_vmec_boozer_gradient_holdout_matrix.py`` and
    ``docs/_static/vmec_boozer_gradient_holdout_matrix.{png,pdf,json,csv}``
    to summarize QH plus Li383 frequency/quasilinear gate status;
  - the matrix closes the reduced multi-equilibrium linear/quasilinear
    VMEC/Boozer gradient gate with maximum relative mismatch ``4.9e-3``;
  - this is still not a nonlinear-window heat-flux gradient or broad
    optimized-equilibrium nonlinear transport claim, so those remain future
    promotion gates.
- Profiler-backed nonlinear performance artifact tightening:
  - added a JSON companion for the nonlinear RHS split profile so kernel
    fractions, dominant kernels, and grid-to-spectral speedups are
    machine-readable rather than only plotted;
  - after the zero-collision fast path, the current tracked Cyclone profile
    reports GPU spectral as the fastest full-RHS row, with grid/spectral
    full-RHS ratio ``1.66`` on GPU and ``1.11`` on CPU;
  - the CPU and GPU spectral brackets are faster in this short bounded
    harness, but the claim remains scoped to the tracked Cyclone profiler case
    until larger matched runtime/memory sweeps and case-level parity gates
    support broader defaults;
  - this supports the existing scoped performance stance: spectral nonlinear
    mode is a validated opt-in engineering mode for selected cases, not a
    global default or broad runtime claim.
- Linear RHS term-profile follow-up:
  - fixed the standalone linear RHS profiler default config path and added a
    JSON companion artifact so term-level timings, dominant nonzero terms, and
    zero-norm initial-state rows are machine-readable;
  - regenerated ``docs/_static/linear_rhs_terms_profile_cpu.csv`` and
    ``docs/_static/linear_rhs_terms_profile.json`` for the Cyclone nonlinear
    runtime state at ``ky=0.3, Nl=4, Nm=8``;
  - fixed the standalone linked-``|k_z|`` profiling source to match the
    production hypercollision formula
    ``nu_hyper_m * m_norm_kz_factor * 2.3 * vth * |kpar_scale|`` rather than
    the unrelated ``nu_hyper`` path;
  - after the zero-collision fast path, the current bounded CPU profile records
    ``full_linear_rhs≈5.04e-2 s`` in the profiler harness and independently
    measured term kernels summing to ``1.71e-2 s``;
  - added the active-state CPU profile
    ``docs/_static/linear_rhs_terms_profile_z_wave_cpu.{csv,json}``, where
    resolved parallel variation activates hypercollisions and linked
    ``|k_z|`` with norm ``2.35e-4`` and linked-operator cost ``2.09e-3 s``;
  - refreshed the matching ``office`` GPU profiles after the profiler-source
    fix: the initial-state artifact records ``full_linear_rhs≈6.52e-3 s`` and
    measured terms summing to ``3.94e-3 s``, while
    ``docs/_static/linear_rhs_terms_profile_z_wave_gpu.{csv,json}`` activates
    the same linked ``|k_z|``/hypercollision pair with matched norm
    ``2.35e-4`` and linked-operator cost ``3.62e-4 s``;
  - keep this as localization evidence only: zero-norm rows must not be skipped
    in production until a state-window identity gate shows they remain inactive
    after nonlinear evolution.
  - added a linked-``|k_z|`` hypercollision guardrail test showing the profiled
    zero norm is an initial-state property: constant-in-``z`` states remain
    zero, while resolved ``z``-varying states activate the term when
    ``hypercollisions_kz`` is nonzero.
  - fixed the term-resolved diagnostic assembly total so active
    perpendicular hyperdiffusion is included in ``assemble_rhs_terms_cached``;
    added the regression by enabling ``D_hyper`` and ``hyperdiffusion`` in the
    total-vs-term-sum test.
  - refactored the field solve so electrostatic runs avoid the electromagnetic
    ``A_parallel``/``B_parallel`` branches when ``beta`` or the field toggles
    disable them; this is a correctness/performance guard rather than a new
    speedup claim, and it adds a regression showing electrostatic ``Nm=1``
    states no longer need the absent ``m=1`` moment.
  - added ``tools/gate_linear_rhs_zero_norm_state_window.py`` and
    ``docs/_static/linear_rhs_zero_norm_state_window_gate.json``;
  - the gate directly compares the full RHS with candidate skip configurations
    over initial, linear-kick, ``z``-wave, and ``z``-wave-kick states;
  - current result: zero-collision skip is identity-safe for this ``nu=0``
    Cyclone window, while skipping hypercollisions is correctly rejected with a
    maximum relative RHS error of ``3.59e-3`` on the resolved ``z``-varying
    state.
  - implemented the accepted zero-collision fast path by disabling runtime
    collision terms when every species has ``nu=0`` and returning early from
    the low-rank collision contribution for static zero-weight/zero-``nu``
    cases while preserving pre-expanded collision matrices.
