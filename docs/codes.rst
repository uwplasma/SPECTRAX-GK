Related Codes
=============

SPECTRAX-GK is a clean JAX implementation with explicit benchmark contracts.
Other codes are used as independent references for equations, diagnostics,
geometry conventions, validation cases, and performance expectations.

Comparison-code roles
---------------------

GX [GX]_
  The closest algorithmic and parity reference. GX uses a
  Fourier-Hermite-Laguerre formulation and GPU-native kernels. For
  SPECTRAX-GK, GX is most useful for Cyclone, KBM, W7-X, HSX, nonlinear
  transport, velocity-space convergence, and performance/scaling comparisons.
  Source-level audits should inform tests and diagnostics, not copy
  implementation.

stella and GENE [GENE]_
  The W7-X benchmark between stella and GENE defines the strongest public
  stellarator validation ladder: multiple flux tubes, linear ITG/TEM scans,
  zonal-flow response, and nonlinear ITG heat fluxes. SPECTRAX-GK should use
  this ladder for paper-level W7-X claims.

GENE, ORB5, and XGC
  Verification literature around these codes motivates tying equation
  verification and numerical verification together. Relevant SPECTRAX-GK tests
  should include reduced-equation limits, observed order, conservation or
  free-energy behavior, and electromagnetic/KBM benchmark observables.

VMEC and Boozer-coordinate tools
  File-based VMEC geometry remains an important compatibility path. The planned
  differentiable path should use ``vmec_jax`` in memory, and add
  ``booz_xform_jax`` when Boozer-coordinate quantities are required.

DESC, SIMSOPT, TORAX, Equinox, and Lineax
  These projects provide useful patterns for differentiable plasma workflows:
  typed PyTrees, exact or validated derivatives, objective APIs, solver
  adapters, persistent compilation-cache workflows, and optimization examples.

How references are used
-----------------------

Each comparison must be traceable:

- reference code, paper, or dataset;
- exact input file and geometry source;
- generated SPECTRAX-GK artifact;
- comparison script;
- fit/window policy;
- numeric gate;
- figure path used in docs or publication material.

For linear comparisons, preferred observables are growth rate, real frequency,
branch identity, eigenfunction overlap, and convergence with resolution.

For nonlinear comparisons, preferred observables are windowed heat-flux
statistics, mode-resolved spectra, conserved or nearly conserved quantities in
reduced limits, and restart/diagnostic reproducibility.

For response-function comparisons, preferred observables are residual level,
damping rate, oscillation frequency, recurrence behavior, and sensitivity to
velocity-space closure or hypercollision.

Current benchmarking policy
---------------------------

- Cyclone, ETG, and KBM are tracked against curated benchmark reference sets
  and exact diagnostic audits.
- Imported-geometry and exact-window audits against GX are the authoritative
  parity checks for W7-X, HSX, Miller, and KAW.
- Reviewer-facing benchmark panels should only include tracked assets generated
  by repository scripts.
- Open lanes must remain labeled as open until their gates are frozen.

Implementation lessons carried forward
--------------------------------------

Source-code audits of comparison projects have led to concrete engineering
targets for SPECTRAX-GK:

- keep geometry contracts explicit and testable;
- precompute and reuse cacheable linear pieces;
- avoid expensive cold-start work in warm-runtime measurements;
- keep diagnostics order-stable across restarts;
- separate benchmark-policy code from solver kernels;
- profile before introducing custom kernels or deeper JAX abstractions;
- make every differentiable objective validate its gradients before it is used
  for optimization.
