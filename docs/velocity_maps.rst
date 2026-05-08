Differentiable velocity-basis maps
==================================

``spectraxgk.velocity_maps`` contains standalone primitives for fixed-shape
differentiable maps of the Hermite-Laguerre velocity basis.

SPECTRAX-GK does not use a movable cell grid in velocity space. Its velocity
representation is modal, with Laguerre index ``l`` and Hermite index ``m``.
The natural analogue of a velocity-coordinate map is therefore a shifted and
scaled basis:

.. code-block:: text

   v_parallel = u + a * vhat
   mu B / T   = b * muhat

where ``a > 0`` and ``b > 0``. The identity map is ``u = 0``, ``a = 1``, and
``b = 1``.

Current scope
-------------

The primitive layer provides:

- Hermite multiplication and derivative matrices using the local
  ``basis.hermite_normed`` convention;
- Laguerre coordinate multiplication using the local ``basis.laguerre``
  convention;
- mapped parallel operators for ``v_parallel = u + a vhat``;
- a scaled perpendicular-energy matrix;
- smooth modal gates for fixed-shape p-adaptation experiments;
- regularization diagnostics for optimization objectives.

This module does not yet modify the gyrokinetic solver. Linear and nonlinear
operator integration should happen only after the standalone algebra and
AD/finite-difference tests are trusted.

Why this is not cell AMR
------------------------

Changing ``Nl`` or ``Nm`` dynamically would change array shapes and disrupt JAX
compilation. The JAX-compatible relaxation is to keep a fixed maximum shape and
differentiate smooth map or gate parameters. This makes the branch suitable for
gradient-based studies while preserving XLA-friendly shapes.

Near-term acceptance tests
--------------------------

Before solver integration, this primitive layer must pass:

- identity-map regression;
- Hermite and Laguerre recurrence tests against the local basis functions;
- AD/finite-difference checks for shift, scale, perpendicular scale, and modal
  gate parameters;
- bounded-map and modal-gate regularization tests.

Later physics tests should compare mapped Hermite bases against same-DOF default
Hermite-Laguerre runs, higher-DOF references, and moment-matched non-AD
shift/scale baselines.
