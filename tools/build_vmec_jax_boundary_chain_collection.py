#!/usr/bin/env python3
"""Build a VMEC-JAX boundary-chain collection summary from probe JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectraxgk.vmec_jax_boundary_chain import build_boundary_chain_collection_summary  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more JSON files from tools/probe_vmec_jax_boundary_chain.py.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--exact-relative-tolerance", type=float, default=1.0e-1)
    parser.add_argument("--internal-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    return parser.parse_args(argv)


def build_collection_payload(
    probe_paths: list[Path],
    *,
    exact_relative_tolerance: float = 1.0e-1,
    internal_relative_tolerance: float = 1.0e-8,
    absolute_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Return a JSON-safe collection summary from boundary-chain probe paths."""

    probes = [json.loads(path.read_text(encoding="utf-8")) for path in probe_paths]
    payload = build_boundary_chain_collection_summary(
        probes,
        exact_relative_tolerance=float(exact_relative_tolerance),
        internal_relative_tolerance=float(internal_relative_tolerance),
        absolute_tolerance=float(absolute_tolerance),
    )
    payload["probe_jsons"] = [str(path) for path in probe_paths]
    payload["claim_scope"] = (
        "sparse VMEC-JAX boundary-chain and optional SPECTRAX growth-branch "
        "locality collection gate; not a nonlinear transport optimization claim"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_collection_payload(
        list(args.probe_json),
        exact_relative_tolerance=float(args.exact_relative_tolerance),
        internal_relative_tolerance=float(args.internal_relative_tolerance),
        absolute_tolerance=float(args.absolute_tolerance),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if bool(payload.get("finite", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
