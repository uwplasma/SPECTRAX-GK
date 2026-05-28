#!/usr/bin/env python3
"""Guard real VMEC/Boozer reduced portfolio artifacts before promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectraxgk.stellarator_objective_portfolio import (  # noqa: E402
    ReducedPortfolioArtifactGuardConfig,
    reduced_portfolio_artifact_guard_report,
)


DEFAULT_ROW_ARTIFACT = ROOT / "docs" / "_static" / "vmec_boozer_multi_point_objective_gate.json"
DEFAULT_GRADIENT_ARTIFACT = ROOT / "docs" / "_static" / "vmec_boozer_quasilinear_gradient_gate.json"
DEFAULT_OUT = ROOT / "docs" / "_static" / "vmec_boozer_reduced_portfolio_guard.json"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def build_vmec_boozer_reduced_portfolio_guard_payload(
    *,
    row_artifact: str | Path = DEFAULT_ROW_ARTIFACT,
    gradient_artifacts: list[str | Path] | tuple[str | Path, ...] = (DEFAULT_GRADIENT_ARTIFACT,),
    min_alphas: int = 2,
    min_ky: int = 2,
    min_objectives: int = 1,
    min_boozer_mode: int = 21,
    value_rtol: float = 1.0e-8,
    value_atol: float = 1.0e-8,
) -> dict[str, object]:
    """Return the VMEC/Boozer reduced-portfolio promotion guard payload."""

    row_path = Path(row_artifact)
    gradient_paths = [Path(path) for path in gradient_artifacts]
    row_payload = _read_json(row_path)
    gradient_payloads = [_read_json(path) for path in gradient_paths]
    config = ReducedPortfolioArtifactGuardConfig(
        min_alphas=int(min_alphas),
        min_ky=int(min_ky),
        min_objectives=int(min_objectives),
        min_boozer_mode=int(min_boozer_mode),
        value_rtol=float(value_rtol),
        value_atol=float(value_atol),
    )
    report = reduced_portfolio_artifact_guard_report(
        row_payload,
        gradient_artifacts=gradient_payloads,
        config=config,
    )
    report["row_artifact"] = str(row_path)
    report["gradient_artifacts"] = [str(path) for path in gradient_paths]
    return report


def write_vmec_boozer_reduced_portfolio_guard_artifact(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> str:
    """Write the guard JSON artifact."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-artifact", type=Path, default=DEFAULT_ROW_ARTIFACT)
    parser.add_argument(
        "--gradient-artifact",
        type=Path,
        action="append",
        default=None,
        help="VMEC/Boozer gradient artifact with finite implicit/FD objective gates.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-alphas", type=int, default=2)
    parser.add_argument("--min-ky", type=int, default=2)
    parser.add_argument("--min-objectives", type=int, default=1)
    parser.add_argument("--min-boozer-mode", type=int, default=21)
    parser.add_argument("--value-rtol", type=float, default=1.0e-8)
    parser.add_argument("--value-atol", type=float, default=1.0e-8)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gradient_artifacts = (
        tuple(args.gradient_artifact)
        if args.gradient_artifact is not None
        else (DEFAULT_GRADIENT_ARTIFACT,)
    )
    payload = build_vmec_boozer_reduced_portfolio_guard_payload(
        row_artifact=args.row_artifact,
        gradient_artifacts=gradient_artifacts,
        min_alphas=args.min_alphas,
        min_ky=args.min_ky,
        min_objectives=args.min_objectives,
        min_boozer_mode=args.min_boozer_mode,
        value_rtol=args.value_rtol,
        value_atol=args.value_atol,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
    else:
        print(write_vmec_boozer_reduced_portfolio_guard_artifact(payload, out=args.out))
    return 0 if bool(payload.get("passed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
