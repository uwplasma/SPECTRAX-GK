Geometry
========

S-alpha flux-tube model
-----------------------

We start with a simple analytic geometry based on the s-alpha model, which is
widely used for Cyclone base case benchmarks. The field-aligned perpendicular
wave number is

.. math::

   k_x(\theta) = k_{x0} + \hat{s} \, \theta \, k_y,

so that

.. math::

   k_\perp^2(\theta) = k_x(\theta)^2 + k_y^2.

Parameters
----------

The geometry is specified by:

- ``q``: safety factor
- ``s_hat``: magnetic shear
- ``epsilon``: inverse aspect ratio
- ``R0``: reference major radius
- ``B0``: reference magnetic field

These parameters will be extended to VMEC/DESC geometry once the linear solver
is validated against Cyclone benchmarks.
