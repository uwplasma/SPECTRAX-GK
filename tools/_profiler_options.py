"""Shared JAX profiler options for trace tools."""

from __future__ import annotations


def make_profile_options(*, python_tracer_level: int = 0, host_tracer_level: int = 0):
    """Return JAX profile options with explicit tracer levels.

    The default levels avoid the optional TensorFlow Python-trace hook, which is
    not present on the lightweight `office` profiling environment.
    """

    import jax.profiler as jprof

    opts = jprof.ProfileOptions()
    opts.python_tracer_level = int(python_tracer_level)
    opts.host_tracer_level = int(host_tracer_level)
    return opts
