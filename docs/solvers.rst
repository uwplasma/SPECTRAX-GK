Solvers
=======

Time integration
----------------

The linear solver supports explicit Euler, RK2, and RK4 updates inside a JAX
``scan`` loop, enabling JIT compilation and differentiability of the entire time
history. The time integrator lives in ``spectraxgk.linear.integrate_linear`` and
is configured via the ``method`` argument. RK4 is used in the Cyclone harness.

Growth rate extraction
----------------------

Given a complex mode time series

.. math::

   \phi(t) \approx \exp[(\gamma + i \omega) t],

we estimate :math:`\gamma` and :math:`\omega` by least-squares fits of
:math:`\log|\phi|` and the unwrapped phase versus time. This method is used in
the Cyclone linear benchmark harness.
