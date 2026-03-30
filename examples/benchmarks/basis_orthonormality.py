import jax.numpy as jnp

from spectraxgk.basis import hermite_normed, laguerre


def hermite_check(n_max: int = 6, x_max: float = 6.0, nx: int = 8001) -> None:
    x = jnp.linspace(-x_max, x_max, nx)
    dx = x[1] - x[0]
    h = hermite_normed(x, n_max)
    w = jnp.exp(-x * x)
    gram = jnp.einsum("ix,jx,x->ij", h, h, w) * dx
    off_diag = gram - jnp.eye(n_max + 1)
    print("Hermite max off-diagonal:", jnp.max(jnp.abs(off_diag)))


def laguerre_check(l_max: int = 6, x_max: float = 40.0, nx: int = 20001) -> None:
    x = jnp.linspace(0.0, x_max, nx)
    dx = x[1] - x[0]
    l = laguerre(x, l_max)
    w = jnp.exp(-x)
    gram = jnp.einsum("ix,jx,x->ij", l, l, w) * dx
    off_diag = gram - jnp.eye(l_max + 1)
    print("Laguerre max off-diagonal:", jnp.max(jnp.abs(off_diag)))


if __name__ == "__main__":
    hermite_check()
    laguerre_check()
