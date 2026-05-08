"""Differentiable velocity-basis map primitives for Hermite-Laguerre models."""

from __future__ import annotations

from dataclasses import dataclass

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VelocityMapConfig:
    """Parameters for a fixed-shape shifted/scaled velocity-basis map.

    The identity map is ``parallel_shift=0``, ``parallel_log_scale=0``, and
    ``perpendicular_log_scale=0``.
    """

    parallel_shift: Array | float = 0.0
    parallel_log_scale: Array | float = 0.0
    perpendicular_log_scale: Array | float = 0.0

    def tree_flatten(self):
        children = (
            jnp.asarray(self.parallel_shift),
            jnp.asarray(self.parallel_log_scale),
            jnp.asarray(self.perpendicular_log_scale),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @property
    def parallel_scale(self) -> Array:
        return jnp.exp(jnp.asarray(self.parallel_log_scale))

    @property
    def perpendicular_scale(self) -> Array:
        return jnp.exp(jnp.asarray(self.perpendicular_log_scale))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MappedParallelOperators:
    """Hermite-space matrices for a shifted/scaled parallel velocity map."""

    multiply: Array
    derivative: Array
    energy: Array
    identity: Array
    regularization: dict[str, Array]

    def tree_flatten(self):
        children = (
            self.multiply,
            self.derivative,
            self.energy,
            self.identity,
            tuple(sorted(self.regularization.items())),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        multiply, derivative, energy, identity, reg_items = children
        return cls(
            multiply=multiply,
            derivative=derivative,
            energy=energy,
            identity=identity,
            regularization=dict(reg_items),
        )


def hermite_multiply_matrix(n_modes: int, *, dtype=jnp.float64) -> Array:
    """Return matrix for multiplication by ``vhat`` in the local Hermite basis.

    The convention matches ``basis.hermite_normed``:

    ``vhat psi_m = sqrt((m+1)/2) psi_{m+1} + sqrt(m/2) psi_{m-1}``.
    """

    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be >= 1")
    mat = jnp.zeros((n, n), dtype=dtype)
    m = jnp.arange(n, dtype=dtype)
    if n > 1:
        upper = jnp.sqrt((m[:-1] + 1.0) / 2.0)
        lower = jnp.sqrt(m[1:] / 2.0)
        mat = mat.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(upper)
        mat = mat.at[jnp.arange(0, n - 1), jnp.arange(1, n)].set(lower)
    return mat


def hermite_derivative_matrix(n_modes: int, *, dtype=jnp.float64) -> Array:
    """Return coefficient matrix for ``d/dvhat`` in ``basis.hermite_normed``."""

    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be >= 1")
    mat = jnp.zeros((n, n), dtype=dtype)
    if n > 1:
        m = jnp.arange(1, n, dtype=dtype)
        mat = mat.at[jnp.arange(0, n - 1), jnp.arange(1, n)].set(jnp.sqrt(2.0 * m))
    return mat


def laguerre_multiply_matrix(n_modes: int, *, dtype=jnp.float64) -> Array:
    """Return matrix for multiplication by the Laguerre coordinate.

    The convention follows ``basis.laguerre``:

    ``x L_l = (2l+1)L_l - (l+1)L_{l+1} - l L_{l-1}``.
    """

    n = int(n_modes)
    if n < 1:
        raise ValueError("n_modes must be >= 1")
    mat = jnp.zeros((n, n), dtype=dtype)
    ell = jnp.arange(n, dtype=dtype)
    mat = mat.at[jnp.arange(n), jnp.arange(n)].set(2.0 * ell + 1.0)
    if n > 1:
        mat = mat.at[jnp.arange(1, n), jnp.arange(0, n - 1)].set(-(ell[:-1] + 1.0))
        mat = mat.at[jnp.arange(0, n - 1), jnp.arange(1, n)].set(-ell[1:])
    return mat


def map_regularization(config: VelocityMapConfig) -> dict[str, Array]:
    """Return bounded-map diagnostics for optimization objectives."""

    shift = jnp.asarray(config.parallel_shift, dtype=jnp.float64)
    log_a = jnp.asarray(config.parallel_log_scale, dtype=jnp.float64)
    log_b = jnp.asarray(config.perpendicular_log_scale, dtype=jnp.float64)
    return {
        "parallel_shift_sq": shift * shift,
        "parallel_log_scale_sq": log_a * log_a,
        "perpendicular_log_scale_sq": log_b * log_b,
        "parallel_scale": jnp.exp(log_a),
        "perpendicular_scale": jnp.exp(log_b),
    }


def mapped_parallel_operators(
    n_hermite: int,
    config: VelocityMapConfig | None = None,
    *,
    dtype=jnp.float64,
) -> MappedParallelOperators:
    """Return Hermite-space operators for ``v_parallel = u + a vhat``."""

    cfg = config if config is not None else VelocityMapConfig()
    identity = jnp.eye(int(n_hermite), dtype=dtype)
    vhat = hermite_multiply_matrix(int(n_hermite), dtype=dtype)
    dvhat = hermite_derivative_matrix(int(n_hermite), dtype=dtype)
    shift = jnp.asarray(cfg.parallel_shift, dtype=dtype)
    scale = jnp.exp(jnp.asarray(cfg.parallel_log_scale, dtype=dtype))
    multiply = shift * identity + scale * vhat
    derivative = dvhat / scale
    energy = multiply @ multiply
    return MappedParallelOperators(
        multiply=multiply,
        derivative=derivative,
        energy=energy,
        identity=identity,
        regularization=map_regularization(cfg),
    )


def mapped_perpendicular_energy_matrix(
    n_laguerre: int,
    config: VelocityMapConfig | None = None,
    *,
    dtype=jnp.float64,
) -> Array:
    """Return scaled Laguerre multiplication matrix for perpendicular energy."""

    cfg = config if config is not None else VelocityMapConfig()
    scale = jnp.exp(jnp.asarray(cfg.perpendicular_log_scale, dtype=dtype))
    return scale * laguerre_multiply_matrix(int(n_laguerre), dtype=dtype)


@dataclass(frozen=True)
class ModalGate:
    """Smooth fixed-shape modal cutoff for differentiable p-adaptation studies."""

    cutoff: Array | float
    width: Array | float = 1.0

    def values(self, n_modes: int, *, dtype=jnp.float64) -> Array:
        n = int(n_modes)
        if n < 1:
            raise ValueError("n_modes must be >= 1")
        idx = jnp.arange(n, dtype=dtype)
        width = jnp.maximum(jnp.asarray(self.width, dtype=dtype), jnp.asarray(1.0e-12, dtype=dtype))
        cutoff = jnp.asarray(self.cutoff, dtype=dtype)
        return jax.nn.sigmoid((cutoff - idx) / width)

    def apply(self, arr: Array, *, axis: int) -> Array:
        arr = jnp.asarray(arr)
        n_modes = int(arr.shape[axis])
        vals = self.values(n_modes, dtype=arr.real.dtype if jnp.iscomplexobj(arr) else arr.dtype)
        shape = [1] * arr.ndim
        shape[axis] = n_modes
        return arr * vals.reshape(shape)


def gate_regularization(gate: ModalGate, n_modes: int, *, dtype=jnp.float64) -> dict[str, Array]:
    """Return diagnostics for a smooth modal gate."""

    vals = gate.values(n_modes, dtype=dtype)
    return {
        "gate_min": jnp.min(vals),
        "gate_max": jnp.max(vals),
        "gate_total_removed": jnp.sum(1.0 - vals),
        "gate_roughness": jnp.sum(jnp.diff(vals) ** 2) if int(n_modes) > 1 else jnp.asarray(0.0, dtype=dtype),
    }
