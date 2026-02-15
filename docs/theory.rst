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

   H_{\ell m} = G_{\ell m} + \frac{Z}{T}\,J_\ell \phi \, \delta_{m0}.

These relations match the Laguerre-Hermite pseudo-spectral form used in GX for
Cyclone benchmarks. [GX]_

Linear gyrokinetic operator
---------------------------

In the current linear electrostatic model, the Hermite-Laguerre moments evolve
according to a GX-style drift/mirror operator,

.. math::

   \frac{\partial G_{\ell m}}{\partial t}
   + v_{\mathrm{th}}\,\mathcal{L}_m[H]
   + v_{\mathrm{th}}\,b^\prime(\theta)\,\mathcal{M}_{\ell m}[H]
   = -i Z/T\,\bigl(c_v \mathcal{C}_m[H] + g_b \mathcal{G}_\ell[H]\bigr)
   + i k_y \phi \,\mathcal{D}_{\ell m},

where :math:`\mathcal{L}_m` is the Hermite streaming ladder and
:math:`b^\prime(\theta)` is the parallel magnetic field gradient used in the
mirror force. The curvature (``cv``) and grad-:math:`B` (``gb``) drift couplings
are encoded in :math:`\mathcal{C}_m` and :math:`\mathcal{G}_\ell`. Explicitly,

.. math::

   \mathcal{C}_m[H] =
   \sqrt{(m+1)(m+2)} H_{\ell, m+2}
   + (2m+1) H_{\ell m}
   + \sqrt{m(m-1)} H_{\ell, m-2},

.. math::

   \mathcal{G}_\ell[H] =
   (\ell+1) H_{\ell+1, m}
   + (2\ell+1) H_{\ell m}
   + \ell H_{\ell-1, m},

.. math::

   \mathcal{M}_{\ell m}[H] =
   -\sqrt{m+1}\,(\ell+1) H_{\ell, m+1}
   -\sqrt{m+1}\,\ell H_{\ell-1, m+1}
   +\sqrt{m}\,\ell H_{\ell, m-1}
   +\sqrt{m}\,(\ell+1) H_{\ell+1, m-1}.

The diamagnetic drive term :math:`\mathcal{D}_{\ell m}` matches the GX
Laguerre formulation with explicit :math:`R/L_n` and :math:`R/L_T` dependence,
including a separate coupling in :math:`m=2` for temperature-gradient drive.
