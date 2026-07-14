#!/usr/bin/env python3
"""Build a VMEC state-to-input mapping artifact from solved perturbation WOUTs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.write_vmec_boundary_campaigns import _json_clean  # noqa: E402
from spectraxgk.geometry.vmec_boundary_chain import (  # noqa: E402
    build_boundary_chain_collection_summary,
)


DEFAULT_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_state_to_input_mapping_response"
)
DEFAULT_BRACKET_SWEEP_OUT_PREFIX = (
    ROOT / "docs" / "_static" / "nonlinear_gradient_state_control_bracket_sweep_status"
)
STATE_RE = re.compile(
    r"^(?P<family>Rcos|Rsin|Zcos|Zsin)_(?:(?:mid_surface)|(?:r(?P<radial>[0-9]+)))_m(?P<mode>[0-9]+)$"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _finite_json(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    val = float(value)
    if not math.isfinite(val):
        return None
    return float(val)


def _parse_state_parameter(parameter: str, ns: int) -> tuple[str, int, int]:
    match = STATE_RE.fullmatch(str(parameter))
    if match is None:
        raise ValueError(f"unsupported state parameter name: {parameter}")
    family = match.group("family")
    radial = match.group("radial")
    radial_index = int(ns // 2) if radial is None else int(radial)
    mode_index = int(match.group("mode"))
    return family, radial_index, mode_index


def _extract_state_vector_from_wout(
    wout_path: Path,
    state_parameters: Sequence[str],
) -> dict[str, float]:
    from vmec_jax import wout

    state = wout.state_from_wout(wout.read_wout(wout_path))
    first_array = np.asarray(getattr(state, "Rcos"))
    ns = int(first_array.shape[0])
    values: dict[str, float] = {}
    for parameter in state_parameters:
        family, radial_index, mode_index = _parse_state_parameter(parameter, ns)
        array = np.asarray(getattr(state, family), dtype=float)
        if not (0 <= radial_index < array.shape[0]) or not (
            0 <= mode_index < array.shape[1]
        ):
            raise ValueError(
                f"{parameter} index is outside state array shape {array.shape}"
            )
        values[parameter] = float(array[radial_index, mode_index])
    return values


def mapping_report_from_samples(
    *,
    case: str,
    admitted_state_controls: Sequence[Mapping[str, Any]],
    input_directions: Sequence[Mapping[str, Any]],
    samples: Mapping[str, Mapping[str, Mapping[str, float]]],
    max_condition_number: float = 1.0e6,
    max_relative_residual: float = 0.10,
    response_floor: float = 1.0e-14,
) -> dict[str, Any]:
    """Return a fail-closed state-to-input mapping report from state samples."""

    if max_condition_number <= 0.0:
        raise ValueError("max_condition_number must be positive")
    if max_relative_residual < 0.0:
        raise ValueError("max_relative_residual must be non-negative")
    if response_floor < 0.0:
        raise ValueError("response_floor must be non-negative")

    state_parameters = [str(row["state_parameter"]) for row in admitted_state_controls]
    coefficients = [str(row["coefficient"]) for row in input_directions]
    if not state_parameters:
        raise ValueError("at least one admitted state control is required")
    if not coefficients:
        raise ValueError("at least one input direction is required")

    jacobian = np.zeros((len(state_parameters), len(coefficients)), dtype=float)
    direction_rows: list[dict[str, Any]] = []
    for col, direction in enumerate(input_directions):
        coefficient = str(direction["coefficient"])
        delta = float(direction["delta_parameter"])
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(
                f"{coefficient} delta_parameter must be finite and positive"
            )
        direction_samples = samples.get(coefficient)
        if not isinstance(direction_samples, Mapping):
            raise ValueError(f"missing samples for {coefficient}")
        try:
            plus = direction_samples["plus_delta"]
            minus = direction_samples["minus_delta"]
            baseline = direction_samples["baseline"]
        except KeyError as exc:
            raise ValueError(
                f"{coefficient} is missing baseline/plus_delta/minus_delta samples"
            ) from exc
        for row, state_parameter in enumerate(state_parameters):
            jacobian[row, col] = (
                float(plus[state_parameter]) - float(minus[state_parameter])
            ) / (2.0 * delta)
        direction_rows.append(
            {
                "coefficient": coefficient,
                "coefficient_slug": direction.get("coefficient_slug"),
                "delta_parameter": float(delta),
                "baseline_state": {
                    key: float(baseline[key]) for key in state_parameters
                },
                "plus_state": {key: float(plus[key]) for key in state_parameters},
                "minus_state": {key: float(minus[key]) for key in state_parameters},
                "central_response": {
                    key: float(jacobian[row, col])
                    for row, key in enumerate(state_parameters)
                },
            }
        )

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    numerical_rank = int(np.linalg.matrix_rank(jacobian, tol=float(response_floor)))
    if singular_values.size == 0 or float(np.min(singular_values)) <= float(
        response_floor
    ):
        condition_number = math.inf
    else:
        condition_number = float(np.max(singular_values) / np.min(singular_values))
    matrix_has_response = bool(np.max(np.abs(jacobian)) > float(response_floor))
    matrix_condition_ok = math.isfinite(condition_number) and condition_number <= float(
        max_condition_number
    )

    controls: list[dict[str, Any]] = []
    for row, state_parameter in enumerate(state_parameters):
        target = np.zeros((len(state_parameters),), dtype=float)
        target[row] = 1.0
        weights, *_ = np.linalg.lstsq(jacobian, target, rcond=None)
        predicted = jacobian @ weights
        residual = float(
            np.linalg.norm(predicted - target)
            / max(np.linalg.norm(target), response_floor)
        )
        row_has_response = bool(
            np.max(np.abs(jacobian[row, :])) > float(response_floor)
        )
        residual_ok = residual <= float(max_relative_residual)
        passed = bool(
            matrix_has_response
            and row_has_response
            and matrix_condition_ok
            and residual_ok
        )
        blockers: list[str] = []
        if not matrix_has_response:
            blockers.append("zero_state_response")
        if numerical_rank < min(len(state_parameters), len(coefficients)):
            blockers.append("rank_deficient_response_matrix")
        if not row_has_response:
            blockers.append("state_control_not_observed")
        if not matrix_condition_ok:
            blockers.append("mapping_condition_number_too_large")
        if not residual_ok:
            blockers.append("mapping_relative_residual_too_large")
        terms = [
            {"coefficient": coefficient, "weight": float(weight)}
            for coefficient, weight in zip(coefficients, weights)
            if abs(float(weight)) > float(response_floor)
        ]
        controls.append(
            {
                "state_parameter": state_parameter,
                "passed": passed,
                "input_control_argument": None
                if not passed
                else " ".join(f"{t['coefficient']}:{t['weight']:.12g}" for t in terms),
                "input_parameter": None,
                "input_direction": {
                    "type": "least_squares_boundary_coefficient_direction",
                    "terms": terms,
                },
                "condition_number": _finite_json(condition_number),
                "condition_number_label": "infinite"
                if not math.isfinite(condition_number)
                else f"{condition_number:.6g}",
                "relative_residual": float(residual),
                "dominant_response_sign": float(
                    np.sign(jacobian[row, int(np.argmax(np.abs(jacobian[row, :])))])
                    if row_has_response
                    else 0.0
                ),
                "blockers": blockers,
            }
        )

    passed = bool(controls and all(bool(row["passed"]) for row in controls))
    matrix_blockers: list[str] = []
    if not matrix_has_response:
        matrix_blockers.append("zero_state_response")
    if numerical_rank < min(len(state_parameters), len(coefficients)):
        matrix_blockers.append("rank_deficient_response_matrix")
    if not matrix_condition_ok:
        matrix_blockers.append("mapping_condition_number_too_large")

    return {
        "kind": "vmec_state_to_input_control_mapping",
        "schema_version": 1,
        "case": str(case),
        "passed": passed,
        "claim_level": "measured_state_to_input_mapping_not_nonlinear_gradient_evidence",
        "source_scope": (
            "Re-equilibrated VMEC input perturbations mapped to vmec_jax state_from_wout "
            "internal controls. A failed artifact is a valid negative guardrail and must not "
            "launch nonlinear-gradient campaigns."
        ),
        "controls": controls,
        "input_directions": direction_rows,
        "jacobian": {
            "row_order": state_parameters,
            "column_order": coefficients,
            "matrix": jacobian.tolist(),
            "singular_values": [float(value) for value in singular_values],
            "rank": int(numerical_rank),
            "condition_number": _finite_json(condition_number),
            "condition_number_label": "infinite"
            if not math.isfinite(condition_number)
            else f"{condition_number:.6g}",
            "response_floor": float(response_floor),
        },
        "acceptance": {
            "max_condition_number": float(max_condition_number),
            "max_relative_residual": float(max_relative_residual),
            "response_floor": float(response_floor),
        },
        "blockers": matrix_blockers,
        "next_action": (
            "the current stellarator-symmetric RBC/ZBS directions do not move the admitted "
            "Rsin/Zcos VMEC-state controls; use an explicit LASYM=true RBS/ZBC branch or "
            "choose QL-admitted controls in the stellarator-symmetric subspace before "
            "launching nonlinear-gradient campaigns"
        )
        if not passed
        else "feed this mapping artifact to the nonlinear-gradient state-control runbook",
    }


def _short_state_parameter(raw: str) -> str:
    return raw.removeprefix("state_control_")


def _row_from_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{path} missing metrics object")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        config = {}
    parameter = str(payload.get("parameter_name", path.stem))
    return {
        "artifact": str(path),
        "alpha_delta": float(payload.get("delta_parameter", float("nan"))),
        "parameter_name": parameter,
        "state_parameter": _short_state_parameter(parameter),
        "passed": bool(payload.get("passed")),
        "blockers": list(payload.get("blockers", [])),
        "response_fraction": float(metrics.get("response_fraction", float("nan"))),
        "min_response_fraction": float(config.get("min_fd_response_fraction", 0.03)),
        "fd_asymmetry_rel": float(metrics.get("fd_asymmetry_rel", float("nan"))),
        "max_fd_asymmetry_rel": float(config.get("max_fd_asymmetry_rel", 0.5)),
        "gradient_uncertainty_rel": float(
            metrics.get("gradient_uncertainty_rel", float("nan"))
        ),
        "max_gradient_uncertainty_rel": float(
            config.get("max_gradient_uncertainty_rel", 0.5)
        ),
        "baseline_window_mean": float(
            metrics.get("baseline_window_mean", float("nan"))
        ),
        "plus_window_mean": float(metrics.get("plus_window_mean", float("nan"))),
        "minus_window_mean": float(metrics.get("minus_window_mean", float("nan"))),
    }


def build_bracket_sweep_status(
    gate_paths: Sequence[Path],
    *,
    run_summary: Path | None = None,
    out_prefix: Path = DEFAULT_BRACKET_SWEEP_OUT_PREFIX,
) -> dict[str, Any]:
    rows = [_row_from_gate(path) for path in gate_paths]
    if not rows:
        raise ValueError("at least one central-FD gate is required")
    rows = sorted(rows, key=lambda row: (row["alpha_delta"], row["state_parameter"]))
    run_payload: dict[str, Any] | None = None
    if run_summary is not None:
        run_payload = _load_json(run_summary)
    summary = {
        "central_fd_gates_passed": sum(1 for row in rows if row["passed"]),
        "central_fd_gates_total": len(rows),
        "max_response_fraction": max(row["response_fraction"] for row in rows),
        "min_gradient_uncertainty_rel": min(
            row["gradient_uncertainty_rel"] for row in rows
        ),
        "nonlinear_runs_completed": None
        if run_payload is None
        else int(run_payload.get("successes", 0)),
        "nonlinear_run_failures": None
        if run_payload is None
        else len(run_payload.get("failures", [])),
        "nonlinear_run_seconds": None
        if run_payload is None
        else float(
            run_payload.get("finished_at", 0.0) - run_payload.get("started_at", 0.0)
        ),
    }
    passed = bool(
        summary["central_fd_gates_passed"] == summary["central_fd_gates_total"]
    )
    report = {
        "kind": "vmec_state_control_bracket_sweep_status",
        "claim_level": "bracket_amplitude_sweep_negative_nonlinear_gradient_evidence",
        "passed": passed,
        "summary": summary,
        "rows": rows,
        "next_action": (
            "do not increase single-control bracket amplitude further; the 3e-3 and 1e-2 sweeps are runtime-stable "
            "but response fractions remain well below the resolved-gradient gate, so the next scientific step is "
            "variance reduction through longer replicated windows or a better-conditioned multi-control observable"
        ),
    }
    _write_bracket_sweep_outputs(out_prefix, report)
    return report


def _write_bracket_sweep_outputs(out_prefix: Path, report: Mapping[str, Any]) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "alpha_delta",
                "state_parameter",
                "passed",
                "response_fraction",
                "min_response_fraction",
                "fd_asymmetry_rel",
                "max_fd_asymmetry_rel",
                "gradient_uncertainty_rel",
                "max_gradient_uncertainty_rel",
                "blockers",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in report["rows"]:
            out = dict(row)
            out["blockers"] = ";".join(row["blockers"])
            writer.writerow({field: out[field] for field in writer.fieldnames})
    _plot_bracket_sweep_status(png_path, report)


def _plot_bracket_sweep_status(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from spectraxgk.artifacts.plotting import set_plot_style

    rows = list(report["rows"])
    labels = [
        f"{row['state_parameter']}\n$\\alpha$={row['alpha_delta']:.3g}" for row in rows
    ]
    x = np.arange(len(rows))
    response = np.asarray([row["response_fraction"] for row in rows], dtype=float)
    asymmetry = np.asarray([row["fd_asymmetry_rel"] for row in rows], dtype=float)
    uncertainty = np.asarray(
        [row["gradient_uncertainty_rel"] for row in rows], dtype=float
    )
    min_response = float(rows[0]["min_response_fraction"])
    max_asymmetry = float(rows[0]["max_fd_asymmetry_rel"])
    max_uncertainty = float(rows[0]["max_gradient_uncertainty_rel"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    axes[0].bar(x, response, color="#377eb8", edgecolor="0.2")
    axes[0].axhline(
        min_response, color="#e41a1c", lw=1.6, ls="--", label=f"gate {min_response:.2f}"
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=18, ha="right")
    axes[0].set_ylabel("FD response fraction")
    axes[0].set_title("Resolved-response gate")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    width = 0.38
    axes[1].bar(
        x - width / 2,
        asymmetry,
        width=width,
        color="#ff7f00",
        edgecolor="0.2",
        label="FD asymmetry",
    )
    axes[1].bar(
        x + width / 2,
        uncertainty,
        width=width,
        color="#4daf4a",
        edgecolor="0.2",
        label="gradient uncertainty",
    )
    axes[1].axhline(max_asymmetry, color="0.25", lw=1.2, ls="--", label="gate 0.5")
    axes[1].axhline(max_uncertainty, color="0.25", lw=1.2, ls=":")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("relative diagnostic")
    axes[1].set_title("Asymmetry and uncertainty gates")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    summary = report["summary"]
    fig.suptitle(
        f"VMEC-state nonlinear bracket sweep: {summary['central_fd_gates_passed']}/"
        f"{summary['central_fd_gates_total']} central-FD gates passed",
        fontsize=14,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _samples_from_campaign(
    campaign: Mapping[str, Any],
    extractor: Callable[[Path, Sequence[str]], dict[str, float]],
) -> dict[str, dict[str, dict[str, float]]]:
    controls = [
        str(row["state_parameter"]) for row in campaign["admitted_state_controls"]
    ]
    samples: dict[str, dict[str, dict[str, float]]] = {}
    for direction in campaign["input_directions"]:
        coefficient = str(direction["coefficient"])
        state_paths = direction.get("expected_wout_files")
        if not isinstance(state_paths, Mapping):
            raise ValueError(f"{coefficient} missing expected_wout_files")
        samples[coefficient] = {}
        for state in ("baseline", "plus_delta", "minus_delta"):
            raw_path = state_paths.get(state)
            if not isinstance(raw_path, str):
                raise ValueError(f"{coefficient} missing {state} wout path")
            path = (ROOT / raw_path).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            samples[coefficient][state] = extractor(path, controls)
    return samples


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["row_order/column_order", *report["jacobian"]["column_order"]])
        for label, values in zip(
            report["jacobian"]["row_order"], report["jacobian"]["matrix"]
        ):
            writer.writerow([label, *values])


def _plot(path: Path, report: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    matrix = np.asarray(report["jacobian"]["matrix"], dtype=float)
    rows = list(report["jacobian"]["row_order"])
    cols = list(report["jacobian"]["column_order"])
    residuals = np.asarray(
        [float(row["relative_residual"]) for row in report["controls"]]
    )
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), constrained_layout=True)
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    im = axes[0].imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[0].set_xticks(np.arange(len(cols)), cols, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(rows)), rows)
    axes[0].set_title("Measured state response")
    axes[0].set_xlabel("VMEC input perturbation")
    axes[0].set_ylabel("VMEC-state control")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axes[0].text(
                x, y, f"{matrix[y, x]:.1e}", ha="center", va="center", fontsize=8.0
            )
    fig.colorbar(im, ax=axes[0], shrink=0.85, label=r"$d s_i / d c_j$")

    x = np.arange(len(rows))
    axes[1].bar(x, residuals, color="#ff7f00", edgecolor="0.25")
    axes[1].axhline(
        float(report["acceptance"]["max_relative_residual"]),
        color="0.25",
        ls="--",
        lw=1.1,
    )
    axes[1].set_xticks(x, rows, rotation=24, ha="right")
    axes[1].set_ylim(
        0.0, max(1.05, float(np.max(residuals)) * 1.1 if residuals.size else 1.0)
    )
    axes[1].set_title("Least-squares target residual")
    axes[1].set_ylabel("relative residual")
    axes[1].grid(True, axis="y", alpha=0.25)
    title = "VMEC state-to-input mapping response"
    if not bool(report["passed"]):
        title += " fails closed"
    fig.suptitle(title, fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--case", default="qa_lowres_state_to_input_mapping_response")
    parser.add_argument("--max-condition-number", type=float, default=1.0e6)
    parser.add_argument("--max-relative-residual", type=float, default=0.10)
    parser.add_argument("--response-floor", type=float, default=1.0e-14)
    return parser


def build_bracket_sweep_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize VMEC-state nonlinear bracket-amplitude sweep gates."
    )
    parser.add_argument("gates", nargs="+", type=Path)
    parser.add_argument("--run-summary", type=Path)
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_BRACKET_SWEEP_OUT_PREFIX
    )
    return parser


def build_boundary_chain_collection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize VMEC-JAX boundary-chain probe JSON files."
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more historical or current boundary-chain probe JSON files.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--exact-relative-tolerance", type=float, default=1.0e-1)
    parser.add_argument("--internal-relative-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-10)
    return parser


def build_boundary_chain_collection_payload(
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


def main_boundary_chain_collection(argv: list[str] | None = None) -> int:
    args = build_boundary_chain_collection_parser().parse_args(argv)
    payload = build_boundary_chain_collection_payload(
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


def main_bracket_sweep_status(argv: list[str] | None = None) -> int:
    args = build_bracket_sweep_status_parser().parse_args(argv)
    report = build_bracket_sweep_status(
        args.gates, run_summary=args.run_summary, out_prefix=args.out_prefix
    )
    print(
        json.dumps(
            {
                "json": str(args.out_prefix.with_suffix(".json")),
                "passed": report["passed"],
                "central_fd_gates_passed": report["summary"]["central_fd_gates_passed"],
                "central_fd_gates_total": report["summary"]["central_fd_gates_total"],
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "boundary-chain-collection":
        return main_boundary_chain_collection(tokens[1:])
    if tokens and tokens[0] == "bracket-sweep-status":
        return main_bracket_sweep_status(tokens[1:])

    args = build_parser().parse_args(tokens)
    campaign = _load_json(args.campaign)
    samples = _samples_from_campaign(campaign, _extract_state_vector_from_wout)
    report = mapping_report_from_samples(
        case=str(args.case),
        admitted_state_controls=campaign["admitted_state_controls"],
        input_directions=campaign["input_directions"],
        samples=samples,
        max_condition_number=float(args.max_condition_number),
        max_relative_residual=float(args.max_relative_residual),
        response_floor=float(args.response_floor),
    )
    report["source_campaign"] = args.campaign.resolve()
    out_prefix = Path(args.out_prefix)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    clean = _json_clean(report)
    json_path.write_text(
        json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, clean)
    _plot(png_path, clean)
    print(
        json.dumps(
            {
                "json": _json_clean(json_path),
                "passed": report["passed"],
                "rank": report["jacobian"]["rank"],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
