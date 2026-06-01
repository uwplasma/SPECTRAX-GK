#!/usr/bin/env python
"""VMEC-JAX QA optimization with an optional SPECTRAX-GK heat-flux objective.

This is a discoverable alias for ``vmec_jax_qa_low_turbulence_optimization.py``.
It mirrors the VMEC-JAX QA optimization example and adds the SPECTRAX-GK
transport residual. Use ``--constraints-only`` for the baseline QA/aspect/iota
run, or keep the default to include the reduced nonlinear-window heat-flux
residual.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def main() -> int:
    """Dispatch to the canonical VMEC-JAX/SPECTRAX-GK optimization example."""

    target = Path(__file__).with_name("vmec_jax_qa_low_turbulence_optimization.py")
    spec = importlib.util.spec_from_file_location("_spectraxgk_vmec_jax_qa_optimization", target)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"failed to load {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
