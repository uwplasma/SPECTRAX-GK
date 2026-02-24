Normalization
=============

This section documents the normalization conventions used in SPECTRAX-GK and
the calibration parameters that scale drift/drive terms for benchmark
comparisons.

Canonical normalization contract
--------------------------------

SPECTRAX-GK now centralizes benchmark-family normalization values in
``spectraxgk.normalization`` via :class:`spectraxgk.normalization.NormalizationContract`.
This is the single source of truth for case defaults:

.. list-table:: Canonical per-case normalization contracts
   :header-rows: 1

   * - Case key
     - ``rho_star``
     - ``omega_d_scale``
     - ``omega_star_scale``
     - ``diagnostic_norm_default``
   * - ``cyclone``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``etg``
     - ``1.0``
     - ``0.4``
     - ``0.8``
     - ``none``
   * - ``kinetic`` (``kinetic_itg`` alias)
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``tem``
     - ``1.0``
     - ``1.0``
     - ``1.0``
     - ``none``
   * - ``kbm``
     - ``1.0``
     - ``1.0``
     - ``0.8``
     - ``none``

These contracts are consumed by benchmark constants for backward compatibility
(``CYCLONE_OMEGA_D_SCALE``, etc.), so existing scripts keep working while all
new calibration updates flow through one module.

Dimensionless units
-------------------

We evolve a dimensionless gyrokinetic system normalized to ion thermal
quantities. The parallel streaming term uses the normalized thermal velocity
:math:`v_{th}` and the flux-tube coordinate :math:`z` (theta-like). The
perpendicular wave numbers are normalized by the ion gyro-radius
:math:`\rho_i`, while the distribution function and potential use the standard
gyrokinetic scaling:

.. math::

   \tilde{\phi} = \frac{e \phi}{T_i}, \qquad
   \tilde{\omega} = \frac{\omega}{v_{th}/R_0}.

For the Cyclone base case we take the reference length :math:`L_{ref}=a` so
that the input gradients are expressed as
:math:`a/L_T` and :math:`a/L_n`. With :math:`R_0 = R/a`, this means
:math:`R/L_T = (R_0)\,(a/L_T)` and similarly for :math:`R/L_n`.

Kinetic species conventions
---------------------------

SPECTRAX-GK supports multi-species kinetic systems. Species-dependent arrays
carry the charge, mass, density, temperature, and gradient inputs:

- ``charge_sign``: :math:`Z_s` (e.g., :math:`+1` for ions, :math:`-1` for electrons).
- ``temp`` / ``mass`` / ``density``: normalized to the reference species.
- ``tz``: :math:`Z_s / T_s` coupling used in the field terms.
- ``R_over_LTi`` / ``R_over_Ln``: normalized gradients for each species.

For adiabatic closures, ``tau_e`` provides the ratio between the kinetic
species temperature and the Boltzmann species temperature.

Field-aligned grid parameters
-----------------------------

For the Cyclone base case we use a field-aligned grid with:

.. math::

   y_0 = 20,\qquad n_\theta = 32,\qquad n_{period} = 2.

In SPECTRAX-GK these map to ``GridConfig(y0=20, ntheta=32, nperiod=2)``, which
sets:

.. math::

   L_y = 2 \pi y_0,\qquad
   z \in [-\pi Z_p, \pi Z_p),\qquad
   Z_p = 2 n_{period} - 1.

The reduced scan tables and regression tests use ``Nx=1, Ny=24, Nz=96`` on this
grid to match the discrete ky set used in the reference CSV.

GX-aligned spectral grids
-------------------------

SPECTRAX-GK’s GX-aligned integrator uses the same Fourier conventions as GX.
The perpendicular wave numbers are defined as

.. math::

   k_x = \frac{n_x}{x_0}, \qquad k_y = \frac{n_y}{y_0},

with ``x0 = Lx / (2π)`` and ``y0 = Ly / (2π)``. The parallel wave number is

.. math::

   k_z = \frac{n_z}{Z_p},

where :math:`Z_p` sets the field-line length
(:math:`z \in [-\pi Z_p, \pi Z_p)`), and :math:`k_z` is defined *without* the
``gradpar`` factor. These definitions are implemented by
``spectraxgk.grids.build_spectral_grid`` and are consistent with GX’s
``kInit`` kernel.

The midplane index used by the GX growth-rate diagnostic corresponds to
``z_index = Nz//2 + 1``, matching the GX kernel logic when ``Nz > 1``.

Sign conventions
----------------

The growth-rate fitting in :func:`spectraxgk.analysis.fit_growth_rate` assumes

.. math::

   s(t) \sim \exp((\gamma - i \omega)\, t),

so:

- ``gamma > 0`` indicates instability.
- ``omega`` is obtained from the negative phase slope.

This is consistent across time-integration and Krylov post-processing paths.
For scan tables and figures, values are reported in the same sign convention as
the solver output unless an explicit diagnostic normalization is requested.

Normalization parameters
------------------------

The linear operator exposes three normalization parameters that influence the
drift/drive terms:

- ``rho_star``: scales :math:`k_x` and :math:`k_y` in the drift and drive
  terms.
- ``omega_d_scale``: scales curvature/grad-:math:`B`/mirror couplings.
- ``omega_star_scale``: scales the diamagnetic drive.

In code, ``rho_star`` multiplies the Fourier grids inside
:func:`spectraxgk.linear.build_linear_cache`, while ``omega_d_scale`` and
``omega_star_scale`` enter directly in :func:`spectraxgk.linear.linear_rhs_cached`.

Diagnostic normalization mode
-----------------------------

Benchmark runners expose ``diagnostic_norm`` and route it through
``spectraxgk.normalization.apply_diagnostic_normalization``:

- ``none``: return raw solver ``(gamma, omega)``.
- ``gx`` / ``rho_star``: multiply reported ``(gamma, omega)`` by ``rho_star``.

This affects reporting only; it does not alter the RHS/operator.

GX end-damping strength (``damp_ends_amp``) is scaled by the timestep inside the
integrator to match the GX implementation: the damping kernel receives
``damp_ends_amp / dt`` so that the damping is defined per unit time.

Defaults (model parameters):

- ``rho_star = 1.0`` (model default)
- ``omega_d_scale = 1.0`` (model default)
- ``omega_star_scale = 1.0`` (model default)

These parameters are surfaced in the regression tables so that future
normalization refinements can be tracked in a reproducible way.

Programmatic usage
------------------

.. code-block:: python

   from spectraxgk.normalization import get_normalization_contract

   contract = get_normalization_contract("etg")
   # contract.omega_d_scale == 0.4
   # contract.omega_star_scale == 0.8
