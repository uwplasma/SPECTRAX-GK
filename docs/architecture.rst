Architecture
============

Core modules
------------

- ``spectraxgk.basis``: Hermite and Laguerre basis functions.
- ``spectraxgk.gyroaverage``: gyroaverage coefficients and polarization helpers.
- ``spectraxgk.geometry``: analytic s-alpha flux-tube geometry.
- ``spectraxgk.terms``: term-wise RHS kernels (streaming, mirror, drifts, drive, collisions, fields).
- ``spectraxgk.linear``: public linear API facade for documented linear imports
  and fixed-step integration entry points.
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
- ``spectraxgk.nonlinear_diagnostics``: sampling, resolved-diagnostic
  packing, and ``SimulationDiagnostics`` construction shared by nonlinear
  diagnostic scans.
- ``spectraxgk.nonlinear_helpers``: Hermitian/fixed-mode projectors,
  diagnostic cache/weight/projection setup, fixed-mode omega masks used by
  comparison parity audits, collision-split policies, and reusable nonlinear
  IMEX operator construction.
- ``spectraxgk.runtime`` / ``spectraxgk.runtime_config``: user-facing runtime entrypoints and configuration schema.
- ``spectraxgk.workflows.runtime.policies``: pure runtime selection policies for solver names, scan modes, nonlinear monitored modes, external fields, and step-count inference.
- ``spectraxgk.workflows.runtime.orchestration``: runtime progress/ETA formatting, combined-ky scan batching, and nonlinear restart/checkpoint artifact handoff behind injectable compatibility seams.
- ``spectraxgk.benchmark_defaults``: normalization constants and Krylov policies for shipped benchmark lanes.
- ``spectraxgk.benchmark_reference`` / ``spectraxgk.benchmark_fit_signals`` / ``spectraxgk.benchmark_species``: reference data loaders, result containers, fitting policies, and benchmark species policies.
- ``spectraxgk.benchmark_scan``: shared scan-window, batching, and fit-signal policies used by benchmark runners.
- ``spectraxgk.benchmarks``: public benchmark runners and compatibility import surface.
- ``spectraxgk.plotting``: reusable, publication-ready plotting utilities.

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
