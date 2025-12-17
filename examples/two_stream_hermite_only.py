# examples/two_stream_hermite_only.py
"""
Hermite-only 1D two-stream toy model (Vlasov-Poisson-ish) for testing the Hermite ladder
and producing a clean "two-stream" figure without Laguerre/GK complexity.

This is NOT full GK. It's a sanity/benchmark harness.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import diffeqsolve, ODETerm, SaveAt, Tsit5, ConstantStepSize

def hermite_ladder_streaming(Hk, kz, vth=1.0):
    # Hk: (Nh, Nk) complex, kz: (Nk,)
    Nh = Hk.shape[0]
    m = jnp.arange(Nh, dtype=jnp.float64)
    sp = jnp.sqrt(m + 1.0)[:, None]
    sm = jnp.sqrt(m)[:, None]

    def shift(dm):
        P = jnp.pad(Hk, ((1, 1), (0, 0)))
        return P[1 + dm : 1 + dm + Nh, :]

    Hp = shift(+1)
    Hm = shift(-1)
    return -1j * kz[None, :] * vth * (sp * Hp + sm * Hm)

def rhs(t, y, args):
    # y: (2*Nh*Nk,) real packed
    Nh, Nk, kz, gamma = args
    N = Nh * Nk
    re = y[:N].reshape(Nh, Nk)
    im = y[N:].reshape(Nh, Nk)
    Hk = re + 1j * im

    # Toy "two-stream drive" on m=1 at k=1 (just to make a clean instability)
    drive = jnp.zeros_like(Hk).at[1, Nk//2 + 1].set(gamma + 0.0j)

    dH = hermite_ladder_streaming(Hk, kz) + drive
    d = jnp.concatenate([jnp.real(dH).reshape(-1), jnp.imag(dH).reshape(-1)])
    return d

def main():
    Nh = 32
    Nk = 33
    Lz = 2 * jnp.pi
    kz = jnp.fft.fftshift(jnp.fft.fftfreq(Nk, d=float(Lz) / Nk)) * (2 * jnp.pi)

    gamma = 1e-3

    H0 = jnp.zeros((Nh, Nk), dtype=jnp.complex128)
    H0 = H0.at[0, Nk//2 + 1].set(1e-3 + 0j)
    y0 = jnp.concatenate([jnp.real(H0).reshape(-1), jnp.imag(H0).reshape(-1)])

    term = ODETerm(rhs)
    ts = jnp.linspace(0.0, 200.0, 400)

    sol = diffeqsolve(
        term,
        solver=Tsit5(),
        t0=0.0, t1=float(ts[-1]),
        dt0=0.05,
        y0=y0,
        args=(Nh, Nk, kz, gamma),
        saveat=SaveAt(ts=ts),
        stepsize_controller=ConstantStepSize(),
        max_steps=2_000_000,
    )

    ys = sol.ys
    N = Nh * Nk
    re = ys[:, :N].reshape(len(ts), Nh, Nk)
    im = ys[:, N:].reshape(len(ts), Nh, Nk)
    Ht = re + 1j * im

    # plot growth of one mode
    amp = jnp.abs(Ht[:, 0, Nk//2 + 1])
    plt.figure(figsize=(7, 4))
    plt.plot(ts, amp)
    plt.yscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$|H_{m=0}(k=1)|$")
    plt.title("Hermite-only toy two-stream growth (sanity harness)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
