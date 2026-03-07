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
- ``drift_scale``: drift normalization (``1.0`` matches GX; ``2.0`` matches GS2)

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
nonlinear config runner, and GX-style volume-weight diagnostics.
That means upcoming VMEC or imported field-line geometry can be threaded into
more of the codebase without rebuilding solver-specific side paths.

The contract also preserves explicit ``jacobian`` and ``grho`` profiles when
they are available from imported geometry. The helper
``load_gx_geometry_netcdf`` reads a GX-style ``Geometry``/``Grids`` NetCDF
layout directly into ``FluxTubeGeometryData``. That is the intended short path
to the GX W7-X examples: import the sampled field-line geometry first, prove
solver/diagnostic parity on that contract, and only then add a native VMEC path
that generates the same contract inside SPECTRAX-GK.
