Code Structure
==============

Purpose
-------

This page documents where the main physics, numerics, runtime, and artifact
surfaces live in the source tree. It is meant to make refactoring safer by
keeping the boundary between public APIs and internal implementation modules
explicit.

Public API vs Internal Modules
------------------------------

Public surfaces that examples, scripts, and external users are expected to rely
on:

- ``spectraxgk.cli``
- ``spectraxgk.runtime``
- ``spectraxgk.runtime_config``
- ``spectraxgk.runtime_artifacts``
- ``spectraxgk.plotting``
- documented benchmark/example scripts under ``examples/`` and ``tools/``

Internal modules that are free to move as long as the public behavior and tests
remain unchanged:

- ``spectraxgk.terms.*``
- ``spectraxgk.runtime_startup``
- ``spectraxgk.runtime_diagnostics``
- ``spectraxgk.runtime_chunks``
- ``spectraxgk.runtime_results``
- ``spectraxgk.from_gx.*``
- low-level geometry adapters and import bridges

Runtime Flow
------------

The executable-facing runtime path is split conceptually into four layers:

1. **configuration and startup**
   - ``runtime_config.py``
   - ``runtime_startup.py``
2. **solver execution**
   - ``runtime.py``
   - ``linear.py``
   - ``nonlinear.py``
   - ``diffrax_integrators.py``
3. **diagnostics and artifacts**
   - ``diagnostics.py``
   - ``runtime_diagnostics.py``
   - ``runtime_results.py``
   - ``runtime_artifacts.py``
   - ``plotting.py``
4. **benchmark and validation tooling**
   - ``benchmarking.py``
   - ``benchmarks.py``
   - ``tools/*.py``

Physics / Numerics / IO Map
---------------------------

.. list-table::
   :header-rows: 1

   * - Responsibility
     - Primary files
     - Typical tests
   * - Basis and spectral grids
     - ``basis.py``, ``grids.py``
     - orthonormality, indexing, symmetry
   * - Geometry and imported equilibria
     - ``geometry.py``, ``miller_eik.py``, ``vmec_eik.py``, ``from_gx/vmec.py``
     - parser, remap, normalization, geometry-response tests
   * - Linear operators and fields
     - ``linear.py``, ``terms/linear_terms.py``, ``terms/fields.py``, ``terms/assembly.py``
     - manufactured solutions, observed-order, eigenfunction and branch tests
   * - Nonlinear operators
     - ``nonlinear.py``, ``terms/nonlinear.py``
     - fixed-mode, diagnostics, collision-split, transport-window tests
   * - Runtime/executable behavior
     - ``runtime.py``, ``cli.py``
     - runtime contract, startup/restart, output-path, executable smoke tests
   * - Artifacts and plots
     - ``runtime_artifacts.py``, ``plotting.py``
     - serialization, reload, plotting contract tests
   * - Benchmark harness
     - ``benchmarking.py``, ``benchmarks.py``
     - late-time/windowed gate tests, reference loading, fallback policy tests

Refactor Mapping
----------------

The current modularization branch is preserving the public runtime surface while
extracting internal responsibilities out of ``runtime.py`` and other large
modules.

Completed extractions:

- startup/loading/initial-condition helpers:
  ``runtime_startup.py``
- GX-style runtime diagnostic chunk helpers:
  ``runtime_diagnostics.py``
- adaptive GX-style chunk execution:
  ``runtime_chunks.py``
- runtime result containers and nonlinear result assembly:
  ``runtime_results.py``

Next planned extractions:

- benchmark-family runners and fit-policy helpers
- remaining linear result-assembly helpers
- runtime output/artifact handoff helpers

Testing Taxonomy
----------------

The source tree should be validated through five distinct test classes:

1. **unit tests**
   - cheap, deterministic, local behavior
2. **numerical verification tests**
   - observed-order, manufactured solutions, invariants
3. **benchmark/validation tests**
   - growth rates, frequencies, eigenfunctions, transport windows
4. **autodiff tests**
   - finite-difference and complex-step gradient checks, tangent/adjoint consistency
5. **regression tests**
   - runtime contracts, saved artifacts, reference-lane preservation

Every future source extraction should update this page if module ownership
changes materially.
