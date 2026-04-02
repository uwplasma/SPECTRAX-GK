Geometry
========

S-alpha flux-tube model
-----------------------

We start with a simple analytic geometry based on the s-alpha model, which is
widely used for Cyclone base case benchmarks. The field-aligned perpendicular
wave number is

.. math::

   k_x(\theta) = k_{x0} - \left(\hat{s}\,\theta - \alpha \sin\theta \right) k_y,

and the metric coefficients are

.. math::

   g_{ds2} = 1 + \left(\hat{s}\,\theta - \alpha \sin\theta\right)^2,\quad
   g_{ds21} = -\hat{s}\left(\hat{s}\,\theta - \alpha \sin\theta\right),\quad
   g_{ds22} = \hat{s}^2.

The perpendicular wave number is then

.. math::

   k_\perp^2(\theta) =
   k_y \left(k_y g_{ds2} + 2 k_x g_{ds21} \right) + k_x^2 g_{ds22},

with an additional :math:`B^{-2}` factor from the s-alpha magnetic field
strength,

.. math::

   B(\theta) = \frac{1}{1 + \epsilon \cos\theta}.

Parameters
----------

The geometry is specified by:

- ``q``: safety factor
- ``s_hat``: magnetic shear
- ``epsilon``: inverse aspect ratio
- ``R0``: reference major radius
- ``B0``: reference magnetic field
- ``drift_scale``: drift normalization (``1.0`` is the tracked default; ``2.0`` selects the alternate doubled-drift convention)

Field-aligned grid parameters
-----------------------------

For direct comparison with published Cyclone base case benchmarks,
``GridConfig`` exposes field-aligned grid inputs:

- ``y0`` sets the minimum binormal wave number via :math:`k_y \rho = 1/y_0`.
  Internally this maps to ``Ly = 2\pi y0`` so that the FFT grid spacing matches.
- ``ntheta`` and ``nperiod`` (or ``zp``) control the parallel grid. We set
  :math:`Z_p = 2\,nperiod-1` and choose ``Nz = ntheta * Zp``, which spans
  :math:`[-\pi Z_p, \pi Z_p)`.

Curvature and grad-B drift
--------------------------

The magnetic drift frequency used in the linear operator follows the standard
s-alpha form

.. math::

   \omega_d(\theta) = k_y \left(\mathcal{C}_v + \mathcal{C}_g\right)
   + k_x \left(\mathcal{C}_v^0 + \mathcal{C}_g^0\right),

with

.. math::

   \mathcal{C}_v = \mathcal{C}_g =
   \frac{\cos\theta + (\hat{s}\,\theta - \alpha \sin\theta)\sin\theta}{R_0},
   \qquad
   \mathcal{C}_v^0 = \mathcal{C}_g^0 =
   -\frac{\hat{s}\sin\theta}{R_0}.

These parameters will be extended to VMEC/DESC geometry once the linear solver
is validated against Cyclone benchmarks.

Slab Model
----------

SPECTRAX-GK now also exposes GX's slab geometry contract directly with
``geometry.model = "slab"``. This is the correct backend for GX's
``secondary`` and ``cETG`` benchmarks; it is not an ``s-alpha`` approximation.

The slab overrides follow the audited GX implementation:

- ``bmag = 1`` and ``bgrad = 0``
- ``cvdrift = gbdrift = cvdrift0 = gbdrift0 = 0``
- ``gradpar = 1`` by default, or ``1/z0`` when ``geometry.z0 > 0``
- the metric still uses the supplied ``s_hat`` unless ``geometry.zero_shat = true``
- with ``zero_shat = true``, the slab metric becomes ``gds2 = 1``,
  ``gds21 = 0``, ``gds22 = 1`` and the effective solver shear is zero

That contract is now locked by unit tests so future secondary/cETG work is
built on the same geometry semantics GX uses.
It does not mean both benchmarks are solved already: ``secondary`` is a
geometry-plus-runtime parity problem, while GX's ``cETG`` benchmark is a
dedicated collisional reduced model with its own solver/RHS path. The slab
backend is the correct prerequisite for both, but only ``secondary`` sits on
the generic-runtime parity path today.

Geometry Data Contract
----------------------

The linear cache now accepts either:

- the analytic ``SAlphaGeometry`` model, or
- a sampled ``FluxTubeGeometryData`` contract.

``FluxTubeGeometryData`` stores the solver-ready profiles on a specific
``theta`` grid:

- ``bmag`` and ``bgrad``,
- ``gradpar``,
- metric coefficients ``(gds2, gds21, gds22)``,
- curvature / grad-B drift coefficients ``(cv, gb, cv0, gb0)``,
- geometry metadata such as ``q``, ``s_hat``, ``R0``, and the
  ``kperp2_bmag`` / ``bessel_bmag_power`` switches.

This is the insertion point for future VMEC/DESC or GX-imported field-line
geometry. The helper ``sample_flux_tube_geometry`` converts the analytic
s-alpha model into the same contract, and ``ensure_flux_tube_geometry_data``
normalizes analytic and sampled inputs onto one solver-facing representation.

The sampled geometry contract is now a JAX pytree and is accepted by the
linear cache, runtime initial-condition builder, RHS assembly entry points,
nonlinear config runner, and reference-compatible volume-weight diagnostics.
That means upcoming VMEC or imported field-line geometry can be threaded into
more of the codebase without rebuilding solver-specific side paths.

The contract also preserves explicit ``jacobian`` and ``grho`` profiles when
they are available from imported geometry. The helper
``load_gx_geometry_netcdf`` reads both GX full-output NetCDF files with
``Geometry``/``Grids`` groups and root-level GX ``eik.nc`` geometry files from
the VMEC workflow. That is the intended short path to the GX W7-X examples:
import the sampled field-line geometry first, prove solver/diagnostic parity on
that contract, and only then add a native VMEC path that generates the same
contract inside SPECTRAX-GK.

Runtime and CLI paths can now construct that bridge directly from config with
``geometry.model = "gx-netcdf"`` and
``geometry.geometry_file = "/path/to/geometry.nc"``. Analytic s-alpha remains
the default with ``geometry.model = "s-alpha"``. For slab cases use
``geometry.model = "slab"`` with optional ``geometry.z0`` and
``geometry.zero_shat`` controls. In practice an imported geometry file
can be either a GX ``*.out.nc`` file or a GX/VMEC-generated ``*.eik.nc`` file
such as the W7-X examples in the GX benchmark tree. Root-level ``*.eik.nc``
files are no longer assumed to be closed-interval grids: the importer now
infers whether the terminal theta point is present from the periodic endpoint
content of the geometry profiles, so both VMEC-style closed grids and GX
Miller's already-open ``*.eiknc.nc`` grids are mapped onto the correct solver
contract. For imported geometry, the
runtime now also adopts the file's ``theta`` extent, twist-shift
``jtwist/x0`` defaults for both ``linked`` and ``fix aspect`` boundaries, and
``kxfac`` metadata so the flux-tube grid is built from the same field-line
domain GX used to generate the file.
The same importer is also exposed under the aliases
``geometry.model = "gx-eik"``, ``"vmec-eik"``, and ``"desc-eik"`` so configs
can reflect the provenance of a root-level ``*.eik.nc`` file without changing
the solver-facing geometry contract.
The linear KBM benchmark entry point now uses the same geometry builder, so
the benchmark audit harness can exercise imported sampled geometry through
``run_kbm_linear`` instead of only through the runtime wrappers.
Regression coverage now runs that benchmark path explicitly for both
``"vmec-eik"`` and ``"desc-eik"`` aliases, so imported W7-X-style geometry is
checked through both runtime and benchmark entry points. The test suite now
locks both root-level contracts explicitly:

- imported VMEC/DESC closed-interval ``*.eik.nc`` files must preserve
  ``theta_scale``/``nfp`` metadata and trim the terminal theta point
  consistently when mapped onto the solver's open field-line grid, and
- imported GX Miller ``*.eiknc.nc`` files must stay on their already-open theta
  grid without a spurious terminal-point trim.

With the corrected GX-time damping contract, that imported-geometry bridge now
also reproduces the corrected GX W7-X linear ITG ``t=2`` reference on the same
sampled field line. The tracked short-window scan in
``docs/_static/w7x_linear_t2_scan.csv`` shows mean absolute ``omega`` errors of
about ``7e-6`` to ``9e-6`` and mean relative ``gamma`` errors of about
``0.4%`` to ``4.2%`` across ``ky = 0.1, 0.2, 0.3, 0.4``.

That same imported contract now has a first-class nonlinear runtime workflow:
``examples/nonlinear/non-axisymmetric/w7x_nonlinear_imported_geometry.py`` and
``examples/nonlinear/non-axisymmetric/runtime_w7x_nonlinear_imported_geometry.toml`` mirror the
GX nonlinear W7-X adiabatic-electron setup while keeping the geometry source
explicitly tied to a VMEC/DESC ``*.eik.nc`` field-line file.

SPECTRAX-GK now also supports a direct VMEC runtime bridge with
``geometry.model = "vmec"``. This path uses the existing compatibility helper
to produce an imported ``*.eik.nc`` file and then re-enters the same
imported-geometry contract described above. The bridge is cached by input
content and VMEC file timestamp when SPECTRAX chooses the output path itself.
If the user supplies an explicit ``geometry_file`` target, the runtime now
regenerates that file instead of silently reusing whatever stale ``*.eik.nc``
may already be present there.
That keeps the native JAX geometry contract centered on ``FluxTubeGeometryData``
while preserving reproducible imported-geometry workflows.
For VMEC ``fix aspect`` cases, the bridge now leaves ``x0`` unset when calling
GX so the helper chooses the same cut that GX would choose from ``y0`` and the
geometry itself. SPECTRAX no longer back-solves ``x0 = Lx/(2 pi)`` into the
helper input, which was generating the wrong HSX/W7-X ``*.eik.nc`` files.
When ``booz_xform_jax`` is not installed into the active environment, point
SPECTRAX at it through ``BOOZ_XFORM_JAX_PATH`` or
``SPECTRAX_BOOZ_XFORM_JAX_PATH``. The internal backend is preferred. A legacy
``booz_xform`` install is only needed as fallback compatibility for older
helper environments.
The VMEC bridge now also expands environment variables in ``geometry.vmec_file``
and resolves relative VMEC paths against ``gx_repo`` before falling back to the
current working directory. That lets the tracked W7-X runtime TOML point at the
benchmark ``wout`` file through the GX repo itself, while the HSX runtime TOML
can stay portable via ``$HSX_VMEC_FILE``.
The nonlinear W7-X and HSX startup audits now confirm that this VMEC runtime
path reproduces GX startup ``g_state`` and ``phi`` to roundoff when the
generated ``*.eik.nc`` is rebuilt from the same VMEC input.
The late-time W7-X diagnostic-state audit now also matches GX to roundoff on
the exact dumped nonlinear state once the comparison tool reconstructs the
compressed real-FFT positive-``ky`` dump grid directly from
``diag_state_ky_t*.bin``. That closes the remaining imported-geometry
diagnostic-contract gap for nonlinear VMEC cases: startup, ``phi``, ``kperp2``,
``fluxfac``, ``Wg``, ``Wphi``, and heat flux all agree on the same GX state.
The follow-on exact-state linear audit on that same W7-X dump now also matches
GX to roundoff. The remaining operator-level fixes were:

- treat ``boundary = "fix aspect"`` and ``"continuous drifts"`` as GX linked
  twist-and-shift boundaries in the linear cache, and
- include the GX collision-conservation correction on top of the
  Lenard-Bernstein damping term.

With those in place, the imported VMEC/eik bridge, the late-time linear RHS,
and the late-time nonlinear E x B diagnostics all agree with GX on the same
dumped stellarator state. The final nonlinear W7-X free-run mismatch then
collapsed once the runtime de-alias mask matched GX exactly: the two-thirds
cutoff must be strict (``< 1/3``), not inclusive. With that correction, the
tracked stock-GX W7-X ``t = 200`` VMEC runtime rerun also passes the native
late-window comparison, so the shipped nonlinear W7-X example is now closed at
startup, exact-state, and long-horizon levels.

Tokamak Miller geometry now follows the same imported-geometry bridge pattern.
With ``geometry.model = "miller"``, SPECTRAX-GK shells out to the existing
Miller helper, generates the matching root-level ``*.eiknc.nc`` file, and then
re-enters the same imported-geometry
contract described above. This keeps the Miller lane geometry-honest without
introducing a second hand-maintained Miller implementation in the runtime path.
On the tracked Cyclone Miller parameters, the generated ``*.eiknc.nc`` file
matches the clean GX grouped ``Geometry`` arrays to roundoff in the main
metric and drift profiles. With the root-level open/closed theta inference
corrected, the clean-mainline Cyclone Miller late-state audit also now closes
on the exact dumped GX state: ``kperp2``, ``fluxfac``, ``phi``, ``Wg``,
``Wphi``, and heat flux all match to roundoff on the same nonlinear state.

Two user-facing entry points now exercise that bridge:

- ``tools/generate_gx_vmec_eik.py --config ...`` generates a compatible
  ``*.eik.nc`` file from a SPECTRAX runtime TOML.
- ``tools/generate_gx_miller_eik.py --config ...`` generates a compatible
  Miller ``*.eiknc.nc`` file from a SPECTRAX runtime TOML.
- ``examples/nonlinear/non-axisymmetric/hsx_nonlinear_vmec_geometry.py`` and
  ``examples/nonlinear/non-axisymmetric/runtime_hsx_nonlinear_vmec_geometry.toml`` run a nonlinear
  adiabatic-electron ITG case on the supplied HSX VMEC equilibrium file while
  letting SPECTRAX generate and reuse the field-line geometry automatically.

Imported geometry currently bypasses analytic twist-shift reconstruction and
uses the provided grid as-is. That keeps the GX-import bridge honest while the
native VMEC path is still being generalized.
