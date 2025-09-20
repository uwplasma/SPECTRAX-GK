
# Physics & Discretizations

## Equations

**Vlasov–Poisson** in 1D1V for species \(s\):
\[
\frac{\partial f_s}{\partial t}
+ v \frac{\partial f_s}{\partial x}
+ \frac{q_s}{m_s} E(x,t) \frac{\partial f_s}{\partial v} = 0,
\]
\[
\frac{\partial E}{\partial x} = \frac{1}{\epsilon_0} \sum_s q_s \int f_s \, dv.
\]

Here \(f_s(x,v,t)\) is the distribution, \(q_s, m_s\) charge & mass.

---

## Fourier–Hermite

Expand in Fourier (x) and orthonormal Hermite (v):
\[
f_s(x,v,t) \approx f_{0,s}(v) + \sum_{k,n} c_{k,n}^{(s)}(t)\, e^{ikx}\, \phi_n(u),
\quad u = \frac{v-u_{0,s}}{v_{\text{th},s}}.
\]

Streaming and field operators become banded in \(n\), with coupling across species via \(E_k\):
\[
E_k(t) = \frac{i}{k\epsilon_0} \sum_s q_s c^{(s)}_{k,0}(t), \quad (k\neq 0).
\]

Nonlinearity uses a pseudo-spectral product \(E(x)\,\partial_v f\) with de-aliasing (e.g. 2/3-rule).

---

## DG–Hermite

- **DG in \(x\)**: upwind flux for advection \(v \partial_x f\), periodic boundaries by default.
- **Hermite in \(v\)**: the same orthonormal basis as above.

The Poisson operator \(P\) satisfies \(E = P\,\rho\), built per boundary condition.

---

## Energetics (diagnostics)

**Kinetic energy** for species \(s\) (Hermite proxy in orthonormal basis):
\[
\mathcal{E}^{(s)}_{\text{kin}}(t)
= \frac{n_{0,s} m_s v_{\text{th},s}^2}{4\sqrt{2}}
\int_0^L \left[C_{0}^{(s)}(x,t) + \sqrt{2}\, C_{2}^{(s)}(x,t)\right]\, dx.
\]

**Field energy**:
\[
\mathcal{E}_{\text{field}}(t) = \int_0^L \frac{E(x,t)^2}{2\epsilon_0}\, dx.
\]

The code computes these with the proper normalization constants (see `spectraxgk/diagnostics.py`).
