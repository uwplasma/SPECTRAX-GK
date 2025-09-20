# SPECTRAX-GK

> A differentiable, JAX-based solver for the multispecies **Vlasov–Poisson** system in 1D1V, supporting **Fourier–Hermite** and **Discontinuous Galerkin (DG)** discretizations.

[![PyPI](https://img.shields.io/pypi/v/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![Python](https://img.shields.io/pypi/pyversions/spectraxgk.svg)](https://pypi.org/project/spectraxgk/)
[![CI](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/SPECTRAX-GK/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-material--mkdocs-blue)](https://uwplasma.github.io/SPECTRAX-GK/)

## Why SPECTRAX-GK?

- **Two discretizations**: Fourier–Hermite (landau problems) & DG-in-x + Hermite-in-v (robust nonlinearity)
- **Linear & nonlinear** physics with multi-species coupling
- **Units-aware inputs**: time in \(1/\omega_p\), length in \(\lambda_D\), \(T\) in eV, drift as \(u/c\)
- **Differentiable & JIT-able** via JAX
- **Publication-quality diagnostics** and example configs
- **Modern workflow**: Ruff (lint/format), MyPy (types), pytest (tests), pre-commit, CI/CD, auto docs

👉 New here? Start with the [Quickstart](quickstart.md).

---

## At a Glance

We solve the Vlasov–Poisson system

\[
\frac{\partial f_s}{\partial t}
+ v \frac{\partial f_s}{\partial x}
+ \frac{q_s}{m_s} E(x,t) \frac{\partial f_s}{\partial v} = 0
\]

\[
\frac{\partial E}{\partial x} = \frac{1}{\epsilon_0}
\sum_s q_s \int f_s\, dv
\]

See the [Physics](physics.md) page for the discretizations and normalization.
