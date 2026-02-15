Theory
======

Gyrokinetic ordering
--------------------

SPECTRAX-GK targets the low-frequency, strongly magnetized regime where the
characteristic fluctuation frequency is small compared to the ion cyclotron
frequency. In this limit, the phase-space dynamics can be reduced to a
five-dimensional gyrokinetic system for the non-adiabatic part of the
distribution function. Classic derivations of the gyrokinetic equation can be
found in Frieman & Chen (1982) and Antonsen & Lane (1980). [FC82]_ [AL80]_

Flux-tube model
---------------

We employ a field-aligned, local flux-tube model in which the perpendicular
spatial dependence is represented spectrally and the parallel coordinate is
resolved along a field line. This approximation underlies the Cyclone base case
benchmark commonly used in gyrokinetic validation studies. [GX]_

Hermite-Laguerre velocity space
-------------------------------

The perturbed distribution is expanded in a Hermite (parallel velocity) and
Laguerre (magnetic moment) basis. For a single species, the expansion is

.. math::

   g(\mathbf{k}, \theta, v_\parallel, \mu) =
   \sum_{\ell=0}^{N_\ell-1} \sum_{m=0}^{N_m-1}
   G_{\ell m}(\mathbf{k}, \theta)
   L_\ell(b) H_m(v_\parallel),

with the gyroaverage factor

.. math::

   J_\ell(b) = e^{-b/2} L_\ell(b),

where :math:`b = k_\perp^2 \rho^2`. This Laguerre-Hermite formulation is detailed
by Mandell, Dorland & Landreman (2017). [MDL17]_

Electrostatic quasineutrality (adiabatic electrons)
---------------------------------------------------

For the current linear operator, we assume adiabatic electrons and solve a
Fourier-space quasineutrality equation of the form

.. math::

   \left(\tau_e + 1 - \sum_{\ell} J_\ell^2 \right) \phi
   = \sum_{\ell} J_\ell G_{\ell, m=0},

where :math:`\tau_e = T_i / T_e`. The electrostatic potential is then used to
construct the standard gyrokinetic variable

.. math::

   H_{\ell m} = G_{\ell m} + J_\ell \phi \, \delta_{m0}.

These relations match the Laguerre-Hermite pseudo-spectral form used in GX for
Cyclone benchmarks. [GX]_

Linear gyrokinetic operator
---------------------------

In the current linear electrostatic model, the Hermite-Laguerre moments evolve
according to

.. math::

   \frac{\partial G_{\ell m}}{\partial t}
   + v_{\mathrm{th}}\,\mathcal{L}_m[H]
   = i \, \omega_d \, \mathcal{E}[H]
   + i \, \omega_*^T \, \mathcal{W}[\phi],

where :math:`\mathcal{L}_m` is the Hermite streaming ladder, :math:`\omega_d`
is the curvature/grad-:math:`B` drift frequency, and :math:`\omega_*^T` is the
diamagnetic drive frequency. The operators :math:`\mathcal{E}` and
:math:`\mathcal{W}` control how velocity-space weighting enters the drift and
drive terms. We use the full Hermite-Laguerre energy operator
:math:`\mathcal{E}[H] = \frac{1}{2} v_\parallel^2 H + \mu H` in the Cyclone
benchmark harness. [GX]_
