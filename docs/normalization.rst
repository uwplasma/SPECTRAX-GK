Normalization
=============

This section documents the normalization conventions used in SPECTRAX-GK and
the specific calibration steps used to compare against GX Cyclone base case
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

GX grid matching
----------------

GX reports Cyclone base case results on a field-aligned grid with:

.. math::

   y_0 = 20,\qquad n_\theta = 32,\qquad n_{period} = 2.

In SPECTRAX-GK these map to ``GridConfig(y0=20, ntheta=32, nperiod=2)``, which
sets:

.. math::

   L_y = 2 \pi y_0,\qquad
   z \in [-\pi Z_p, \pi Z_p),\qquad
   Z_p = 2 n_{period} - 1.

The GX-style scan tables and regression tests use ``Nx=1, Ny=24, Nz=16`` on
this grid to match the discrete ky set used in the reference CSV.

Rho-star calibration
--------------------

GX and SPECTRAX-GK use the same gyro-radius normalization, but small
discrepancies in the analytic geometry and drift terms can be absorbed by a
single scale factor:

.. math::

   k_y \rightarrow \rho_* k_y,\qquad
   k_x \rightarrow \rho_* k_x.

This is exposed as ``LinearParams.rho_star`` and applied inside the linear
cache. For the current GX comparison sweep, we use:

.. math::

   \rho_* = 0.9,\qquad \omega_d\_scale = 0.2,\qquad \omega_\* \, scale = 0.55.

These parameters are intentionally surfaced in the regression tables so that
future normalization refinements can be tracked in a reproducible way.
