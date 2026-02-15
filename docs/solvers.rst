Solvers
=======

Time integration
----------------

The current linear solver uses a forward Euler update inside a JAX ``scan``
loop, which enables JIT compilation and differentiability of the entire time
history. The time integrator lives in ``spectraxgk.linear.integrate_linear``.

We will introduce higher-order explicit integrators and semi-implicit schemes
once curvature and gradient-drive terms are in place.

Growth rate extraction
----------------------

Given a complex mode time series

.. math::

   \phi(t) \approx \exp[(\gamma + i \omega) t],

we estimate :math:`\gamma` and :math:`\omega` by least-squares fits of
:math:`\log|\phi|` and the unwrapped phase versus time. This method is used in
the Cyclone linear benchmark harness.
