Normalization
=============

This section documents the normalization conventions used in SPECTRAX-GK and
the calibration parameters used to compare against published Cyclone base case
results.

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

The reduced scan tables and regression tests use ``Nx=1, Ny=24, Nz=16`` on this
grid to match the discrete ky set used in the reference CSV.

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

Defaults:

- ``rho_star = 1.0`` (model default)
- ``omega_d_scale = 1.0`` (model default)
- ``omega_star_scale = 1.0`` (model default)

Cyclone base case calibration:

For the current Cyclone reference matching sweep we use:

.. math::

   \rho_* = 0.9,\qquad \omega_d\_scale = 0.1,\qquad \omega_* \, scale = 0.6.

These parameters are surfaced in the regression tables so that future
normalization refinements can be tracked in a reproducible way.
