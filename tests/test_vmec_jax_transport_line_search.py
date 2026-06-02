from __future__ import annotations

import json

import numpy as np
import pytest

import spectraxgk
from spectraxgk.vmec_jax_transport_line_search import (
    ProjectedLineSearchPolicy,
    projected_line_search_input_manifest,
    select_projected_line_search_candidate,
    sparse_descent_direction_from_gradient_report,
)


def _gradient_report() -> dict[str, object]:
    return {
        "parameter_count": 4,
        "top_gradient_components": [
            {"parameter_index": 1, "gradient": -3.0, "name": "zs10"},
            {"parameter_index": 3, "gradient": 4.0, "name": "rc11"},
            {"parameter_index": 0, "gradient": 12.0, "name": "rc01"},
        ],
    }


def test_sparse_descent_direction_uses_ranked_gradient_components() -> None:
    direction = sparse_descent_direction_from_gradient_report(_gradient_report(), top_n=2)

    assert direction.shape == (4,)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert direction[1] == pytest.approx(3.0 / 5.0)
    assert direction[3] == pytest.approx(-4.0 / 5.0)
    assert direction[0] == 0.0


def test_projected_line_search_input_manifest_is_json_safe() -> None:
    manifest = projected_line_search_input_manifest(_gradient_report(), steps=(0.1, 0.2), top_n=2)

    assert manifest["kind"] == "vmec_jax_projected_transport_line_search_input_manifest"
    assert manifest["parameter_count"] == 4
    assert manifest["direction_l2_norm"] == pytest.approx(1.0)
    assert manifest["steps"][0]["parameter_l2_norm"] == pytest.approx(0.1)
    assert manifest["steps"][0]["parameter_linf_norm"] == pytest.approx(0.08)
    json.dumps(manifest, allow_nan=False)


def test_sparse_descent_direction_rejects_bad_gradient_report() -> None:
    with pytest.raises(ValueError, match="zero descent direction"):
        sparse_descent_direction_from_gradient_report(
            {
                "parameter_count": 2,
                "top_gradient_components": [{"parameter_index": 0, "gradient": 0.0}],
            }
        )


def test_projected_line_search_admission_selects_best_gate_passing_candidate() -> None:
    report = select_projected_line_search_candidate(
        {"transport_metric_final": 10.0, "gate_passed": True},
        [
            {"label": "small", "step": 0.1, "transport_metric_final": 9.5, "gate_passed": True},
            {"label": "failed", "step": 0.2, "transport_metric_final": 8.0, "gate_passed": False},
            {"label": "best", "step": 0.15, "transport_metric_final": 9.0, "gate_passed": True},
        ],
        policy=ProjectedLineSearchPolicy(minimum_relative_improvement=0.02),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "best"
    failed = next(row for row in report["candidates"] if row["label"] == "failed")
    assert "gate_failed" in failed["admission_blockers"]
    assert failed["admitted"] is False
    json.dumps(report, allow_nan=False)


def test_projected_line_search_admission_fails_closed_without_improvement() -> None:
    report = select_projected_line_search_candidate(
        {"transport_metric_final": 10.0, "gate_passed": True},
        [{"label": "worse", "step": 0.1, "transport_metric_final": 10.1, "gate_passed": True}],
    )

    assert report["passed"] is False
    assert report["selected_candidate"] is None
    assert "insufficient_transport_improvement" in report["candidates"][0]["admission_blockers"]


def test_projected_line_search_public_api_exports() -> None:
    assert spectraxgk.ProjectedLineSearchPolicy is ProjectedLineSearchPolicy
    assert spectraxgk.sparse_descent_direction_from_gradient_report is sparse_descent_direction_from_gradient_report
    assert spectraxgk.projected_line_search_input_manifest is projected_line_search_input_manifest
    assert spectraxgk.select_projected_line_search_candidate is select_projected_line_search_candidate
