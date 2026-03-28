Benchmarks
==========

SPECTRAX-GK’s benchmark figures are organized as a compact atlas instead of a
case-by-case gallery. The layout follows the standard gyrokinetic comparison
pattern used in the GX, stella, and stellarator benchmark literature [GX]_
[STELLA]_ [W7XBENCH]_:

- linear growth-rate and real-frequency overlays versus ``k_y`` (or ``beta``),
- nonlinear time traces of heat flux, free energy, electrostatic field energy,
  and magnetic field energy when present,
- compact cross-code panels that separate headline validation cases from
  broader stress lanes.

The figures in this page are generated directly from tracked CSV assets and
curated comparison traces in ``docs/_static``.

Figure generation
-----------------

Regenerate the atlas figures with:

.. code-block:: bash

   python tools/make_benchmark_atlas.py

The atlas builder now reads its inputs from
``tools/benchmark_atlas_manifest.toml`` and writes a machine-readable summary to
``tools_out/benchmark_atlas_summary.json`` so the panel provenance stays
explicit.

This produces:

- ``docs/_static/benchmark_imported_linear_panel.png``
- ``docs/_static/benchmark_extended_linear_panel.png``
- ``docs/_static/benchmark_core_linear_atlas.png``
- ``docs/_static/benchmark_core_nonlinear_atlas.png``
- ``docs/_static/benchmark_readme_panel.png``

PDF copies are emitted alongside each PNG for manuscript workflows.

Tracked benchmark metrics
-------------------------

The benchmark atlas uses the same small set of physically interpretable metrics
throughout:

- growth rate ``gamma``
- real frequency ``omega``
- ion heat flux
- free energy
- electrostatic field energy (legacy variable name ``Wphi``)
- magnetic field energy when ``A_parallel`` or ``B_parallel`` are active

Primary publication set
-----------------------

The headline validation set is limited to the full-gyrokinetic lanes that are
already used in the current parity and regression workflow:

- Cyclone ITG linear and nonlinear
- KBM linear and nonlinear
- W7-X VMEC linear and nonlinear
- HSX VMEC linear and nonlinear
- Cyclone Miller geometry linear and nonlinear

.. figure:: _static/benchmark_core_linear_atlas.png
   :width: 100%
   :align: center
   :alt: Core linear benchmark atlas

   Core linear benchmark atlas. The first three tiles are the main tokamak
   cross-code scans, while the fourth tile collects imported-geometry and
   exact-diagnostic checks for W7-X, HSX, Cyclone Miller geometry, and KAW.

.. figure:: _static/benchmark_core_nonlinear_atlas.png
   :width: 100%
   :align: center
   :alt: Core nonlinear benchmark atlas

   Core nonlinear benchmark atlas. These panels track the time histories used
   for the main nonlinear validation claim: Cyclone, KBM, W7-X, and HSX.

Cross-code linear comparisons
-----------------------------

The core linear atlas intentionally combines two types of benchmark inputs:

- published or curated cross-code scans for Cyclone ITG, ETG, and KBM,
- direct imported-geometry or exact-diagnostic comparisons for W7-X, HSX,
  Cyclone Miller geometry, and KAW.

The current tracked cross-code coverage is:

- Cyclone ITG against external tokamak references
- ETG against GS2 and stella
- KBM against GX, GS2, and stella where matching inputs are available
- W7-X and HSX short-window imported-geometry audits against GX
- Cyclone Miller imported-geometry audit against GX
- KAW exact late diagnostic reconstruction from same-run field dumps

The README uses the compact summary panel below, but the separate core linear
atlas is the more legible publication asset:

.. figure:: _static/benchmark_readme_panel.png
   :width: 100%
   :align: center
   :alt: README benchmark atlas

   Compact README benchmark atlas. This is a summary figure; the individual
   linear and nonlinear atlas figures are preferred for papers and talks.

Extended stress matrix
----------------------

Not every benchmark lane belongs in the headline validation claim. The
extended linear stress matrix keeps exploratory or still-evolving lanes
visible without mixing them into the primary publication set.

The current extended panel covers:

- Cyclone kinetic electrons
- TEM
- KBM Miller exact late growth window

.. figure:: _static/benchmark_extended_linear_panel.png
   :width: 90%
   :align: center
   :alt: Extended linear stress matrix

   Extended linear stress matrix. These lanes are useful for stress testing
   solver contracts, but they are intentionally separated from the main
   publication-facing benchmark atlas.

Case groups
-----------

Tokamak core cases
^^^^^^^^^^^^^^^^^^

- Cyclone ITG: linear ``gamma``/``omega`` scans and nonlinear transport traces
- KBM: linear branch-following scans and nonlinear transport traces
- Cyclone Miller geometry: imported linear scan and nonlinear time-trace audit

Stellarator core cases
^^^^^^^^^^^^^^^^^^^^^^

- W7-X VMEC: short-window linear scans and nonlinear transport traces
- HSX VMEC: short-window linear scans and nonlinear transport traces

Reduced and stress cases
^^^^^^^^^^^^^^^^^^^^^^^^

- ETG: linear cross-code overlays
- KAW: exact late-time diagnostic reconstruction and exact-window energy audit
- KBM Miller: exact late-time growth replay on the tracked low-``k_y`` branch
- Cyclone kinetic electrons: extended linear stress lane
- TEM: extended linear stress lane

Notes on interpretation
-----------------------

The atlas is split deliberately:

- the README and top-level docs show the compact atlas and the core linear and
  nonlinear panels,
- the extended stress matrix remains visible in the benchmark docs,
- open exploratory lanes are not mixed into the headline validation figure.

This keeps the public benchmark story reviewer-proof while still preserving the
wider stress-test record inside the docs and ``plan.md``.
