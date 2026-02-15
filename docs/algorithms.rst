Algorithms
==========

Linear operator workflow
------------------------

For each time step, the linear operator proceeds as follows:

1. Compute :math:`b = k_\perp^2 \rho^2` from the s-alpha geometry.
2. Evaluate the Laguerre gyroaverage coefficients :math:`J_\ell(b)`.
3. Solve the adiabatic-electron quasineutrality equation for :math:`\phi`.
4. Construct :math:`H_{\ell m} = G_{\ell m} + J_\ell \phi \delta_{m0}`.
5. Apply the Hermite ladder streaming operator to :math:`H`.

This sequence is designed to be JIT-compilable, differentiable, and efficient
under JAX.

Linear operator decomposition
-----------------------------

The linear electrostatic operator is decomposed into physically motivated
pieces that act on :math:`H_{\ell m}`:

.. math::

   \frac{\partial G_{\ell m}}{\partial t}
   = - v_{\mathrm{th}}\,\mathcal{L}_m[H]
   - v_{\mathrm{th}}\,b^\prime(\theta)\,\mathcal{M}_{\ell m}[H]
   - i\,c_v\,\mathcal{C}_m[H]
   - i\,g_b\,\mathcal{G}_\ell[H]
   + i k_y \phi \,\mathcal{D}_{\ell m}
   + \mathcal{D}_{\mathrm{coll}}.

Here:

- :math:`\mathcal{L}_m` is the Hermite ladder streaming operator.
- :math:`\mathcal{M}_{\ell m}` is the mirror coupling involving
  :math:`b^\prime(\theta)`.
- :math:`\mathcal{C}_m` and :math:`\mathcal{G}_\ell` encode curvature and
  grad-:math:`B` couplings, respectively.
- :math:`\mathcal{D}_{\ell m}` is the diamagnetic drive (nonzero for
  :math:`m=0,2`).
- :math:`\mathcal{D}_{\mathrm{coll}}` represents optional Lenard-Bernstein and
  hyper-diffusion damping.

Operator splitting summary
--------------------------

Time integration is handled with explicit Runge-Kutta schemes or IMEX/implicit
updates. The operator splitting used in ``imex`` mode treats
``\mathcal{D}_{\mathrm{coll}}`` implicitly while keeping the streaming and
drift/drive terms explicit. The fully implicit option performs a backward-Euler
solve using GMRES with a diagonal preconditioner that includes damping and
drift/mirror diagonals.

Data layout and memory
----------------------

We store Laguerre and Hermite indices in the leading axes and use FFT-friendly
ordering for the perpendicular Fourier grid. The layout is optimized for vector
operations and will be extended with sharding strategies once nonlinear terms
are added.
