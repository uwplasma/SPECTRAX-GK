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

GX grid matching
----------------

For direct comparison with GX benchmarks, ``GridConfig`` exposes GX-style
inputs:

- ``y0`` sets the minimum binormal wave number via :math:`k_y \rho = 1/y_0`.
  Internally this maps to ``Ly = 2\pi y0`` so that the FFT grid spacing matches.
- ``ntheta`` and ``nperiod`` (or ``zp``) control the parallel grid. We set
  :math:`Z_p = 2\,nperiod-1` and choose ``Nz = ntheta * Zp``, which reproduces
  the GX ``z`` grid spanning :math:`[-\pi Z_p, \pi Z_p)`.

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
