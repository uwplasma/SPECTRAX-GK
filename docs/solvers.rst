Solvers
=======

Time integration
----------------

The linear solver supports explicit Euler, RK2, and RK4 updates inside a JAX
``scan`` loop, enabling JIT compilation and differentiability of the entire time
history. The time integrator lives in ``spectraxgk.linear.integrate_linear`` and
is configured via the ``method`` argument. RK4 is used in the Cyclone harness.

To stabilize higher-order Hermite-Laguerre scans, two additional options are
available:

- ``method="imex"``: a semi-implicit update that treats Lenard-Bernstein and
  hyper-diffusion damping implicitly while keeping the drift/streaming terms
  explicit.
- ``method="implicit"``: a backward-Euler solve using a matrix-free GMRES
  iteration. This is slower but robust for stiff linear runs. A few stabilized
  fixed-point sub-iterations provide an initial guess and a diagonal
  preconditioner based on the damping terms accelerates convergence for
  higher-order scans.

Optional damping
----------------

To stabilize high-order Hermite-Laguerre moments, the linear operator supports
two optional damping models:

- A Lenard-Bernstein diagonal rate ``nu`` with coefficients
  :math:`\nu(\alpha m + \beta l)`.
- A hyper-damping term ``nu_hyper`` with exponent ``p_hyper`` that suppresses
  the highest Hermite/Laguerre indices.

Both are disabled by default and can be enabled via ``LinearParams`` for
resolution studies.

Performance caching
-------------------

To reduce repeated geometry work inside the time loop, we cache the gyroaverage
coefficients, drift frequency, and zero-mode mask in a ``LinearCache`` object.
The helper ``build_linear_cache`` constructs this cache, and
``integrate_linear`` will build and reuse it automatically.

Growth rate extraction
----------------------

Given a complex mode time series

.. math::

   \phi(t) \approx \exp[(\gamma + i \omega) t],

we estimate :math:`\gamma` and :math:`\omega` by least-squares fits of
:math:`\log|\phi|` and the unwrapped phase versus time. The helper
``fit_growth_rate_auto`` can automatically select a fitting window by scanning
for the most exponential-like segment of the time history, reducing sensitivity
to early transients in ky scans. This method is used in the Cyclone linear
benchmark harness when ``auto_window=True``.
