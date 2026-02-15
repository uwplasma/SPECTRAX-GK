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

Data layout and memory
----------------------

We store Laguerre and Hermite indices in the leading axes and use FFT-friendly
ordering for the perpendicular Fourier grid. The layout is optimized for vector
operations and will be extended with sharding strategies once nonlinear terms
are added.
