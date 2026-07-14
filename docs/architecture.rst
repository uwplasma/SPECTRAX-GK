Architecture
============

Core modules
------------

- ``spectraxgk.core.velocity``: Hermite/Laguerre basis functions, gyroaverage
  coefficients, and polarization helpers.
- ``spectraxgk.geometry``: analytic s-alpha flux-tube geometry.
- ``spectraxgk.terms``: term-wise RHS kernels (streaming, mirror, drifts, drive, collisions, fields).
- ``spectraxgk.linear``: explicit public linear API for documented operators,
  cache construction, parallel kernels, and integrators. Private kernel helpers
  stay in their owning modules and are not compatibility exports.
- ``spectraxgk.operators.linear``: cache construction, linked-boundary maps,
  Hermite/Laguerre moment operators, linear parameter pytrees, and cached RHS
  assembly entry points.
- ``spectraxgk.solvers.linear``: matrix-free eigensolver policy, linear
  fixed-step/diagnostic integration policy, implicit GMRES/preconditioner
  policy, and gated velocity-parallel linear RHS dispatch.
- ``spectraxgk.nonlinear``: public nonlinear runtime facade for explicit,
  adaptive, diagnostic, and cached IMEX workflows.
- ``spectraxgk.solvers.nonlinear``: explicit RK/SSP/K10 and IMEX fixed-point,
  GMRES, and stage-composition policy.
- ``spectraxgk.operators.nonlinear.diagnostics``: sampling, resolved-diagnostic
  packing, and ``SimulationDiagnostics`` construction shared by nonlinear
  diagnostic scans.
- ``spectraxgk.operators.nonlinear.projection``: Hermitian and fixed-mode state
  projections used by compressed-real-FFT nonlinear scans and fixed-mode
  diagnostics.
- ``spectraxgk.operators.nonlinear.collisions``: diagonal collision and
  hypercollision split policies shared by explicit and IMEX nonlinear scans.
- ``spectraxgk.operators.nonlinear.policies``: diagnostic cache/weight/projection
  setup, adaptive time-step policy, fixed-mode omega masks used by comparison
  parity audits, reusable nonlinear IMEX operator construction, and public
  facades for the focused projection/collision owners.
- ``spectraxgk.runtime`` / ``spectraxgk.workflows.runtime.config``: user-facing runtime entrypoints and configuration schema.
- ``spectraxgk.workflows.runtime.policies``: pure runtime selection policies for solver names, scan modes, nonlinear monitored modes, external fields, and step-count inference.
- ``spectraxgk.workflows.runtime.orchestration_scan``, ``spectraxgk.workflows.runtime.chunks``, and ``spectraxgk.workflows.runtime.orchestration_artifacts``: runtime scan batching, progress/ETA formatting, and nonlinear restart/checkpoint artifact handoff behind injectable seams.
- ``spectraxgk.benchmarking.shared``: reviewed reference tables,
  normalization constants, and comparison-only branch policies.
- ``spectraxgk.diagnostics.growth_rates``: reusable growth/frequency fitting.
- ``spectraxgk.benchmarks``: compact reference-policy facade; it does not own
  simulation execution.
- ``spectraxgk.artifacts.plotting``: reusable, publication-ready plotting utilities.

Term-level source mapping
-------------------------

- streaming, mirror, curvature, grad-B, diamagnetic, collisions,
  hypercollisions, hyperdiffusion, end damping:
  ``src/spectraxgk/terms/linear_terms.py``
- field solves:
  ``src/spectraxgk/terms/fields.py``
- nonlinear E×B, flutter, and Bessel-grid transforms:
  ``src/spectraxgk/terms/nonlinear.py``
- assembled RHS:
  ``src/spectraxgk/terms/assembly.py``

For the full operator equations, see :doc:`operators`.

Data flow
---------

The linear solve is structured as:

1. build the spectral grid and geometry
2. compute gyroaverage coefficients
3. convert ``LinearTerms`` into one canonical ``TermConfig``
4. solve the field equations for :math:`(\\phi, B_\\parallel, A_\\parallel)`
5. build the gyrokinetic variable ``H``
6. assemble RHS by summing per-term kernels from ``spectraxgk.terms``
7. advance in time using ``integrate_linear``/diffrax/Krylov with the same RHS

This structure is intentionally modular so that nonlinear terms, collisions,
geometry adapters, and electromagnetic extensions can be inserted with minimal
refactoring.
