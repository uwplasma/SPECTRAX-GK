from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

import spectraxgk
from spectraxgk.objectives.vmec_transport_gradient import (
    boundary_spec_record,
    build_boundary_transport_gradient_report,
    write_boundary_transport_gradient_report,
)


@dataclass(frozen=True)
class FakeSpec:
    name: str
    kind: str
    index: int
    m: int
    n: int


class FakeOptimizer:
    _specs = (
        FakeSpec("rc01", "rc", 0, 0, 1),
        FakeSpec("zs10", "zs", 1, 1, 0),
        FakeSpec("rc11", "rc", 2, 1, 1),
    )

    def residual_fun(self, params):
        params = np.asarray(params, dtype=float)
        return np.asarray([0.4 + params[0] - 2.0 * params[1]])

    def objective_and_gradient_fun(self, params):
        residual = self.residual_fun(params)[0]
        jac = np.asarray([1.0, -2.0, 0.0])
        return 0.5 * residual**2, residual * jac

    def jacobian_fun(self, params):
        return np.asarray([[1.0, -2.0, 0.0]])


def test_boundary_spec_record_uses_vmec_jax_fields() -> None:
    row = boundary_spec_record(FakeSpec("zs10", "zs", 4, 1, 0), fallback_index=9)

    assert row == {
        "name": "zs10",
        "kind": "zs",
        "mode_index": 4,
        "m": 1,
        "n": 0,
    }


def test_transport_gradient_report_ranks_boundary_directions() -> None:
    report = build_boundary_transport_gradient_report(
        FakeOptimizer(),
        top_n=2,
        include_jacobian=True,
    )

    assert report["finite"] is True
    assert report["transport_sensitivity_detected"] is True
    assert report["classification"] == "sensitive_boundary_transport_objective"
    assert report["parameter_count"] == 3
    assert report["residual_count"] == 1
    assert report["objective_value"] == pytest.approx(0.08)
    assert report["gradient_norm_l2"] == pytest.approx(np.sqrt(0.16 + 0.64))
    assert [row["name"] for row in report["top_gradient_components"]] == ["zs10", "rc01"]
    assert report["jacobian"]["available"] is True
    assert report["jacobian"]["shape"] == [1, 3]
    assert report["jacobian"]["frobenius_norm"] == pytest.approx(np.sqrt(5.0))
    json.dumps(report, allow_nan=False)


def test_transport_gradient_report_classifies_flat_response() -> None:
    class FlatOptimizer(FakeOptimizer):
        def residual_fun(self, params):
            return np.asarray([0.0])

        def objective_and_gradient_fun(self, params):
            return 0.0, np.zeros(3)

    report = build_boundary_transport_gradient_report(
        FlatOptimizer(),
        sensitivity_atol=1.0e-10,
    )

    assert report["finite"] is True
    assert report["transport_sensitivity_detected"] is False
    assert report["classification"] == "locally_flat_or_underconditioned_transport_objective"
    assert "do not launch another blind scalar-weight ladder" in report["next_action"]


def test_transport_gradient_report_requires_params_without_specs() -> None:
    class NoSpecOptimizer:
        def residual_fun(self, params):
            return np.asarray([1.0])

        def objective_and_gradient_fun(self, params):
            return 0.5, np.ones(1)

    with pytest.raises(ValueError, match="params must be provided"):
        build_boundary_transport_gradient_report(NoSpecOptimizer())


def test_transport_gradient_report_writer_and_public_api(tmp_path) -> None:
    assert spectraxgk.build_boundary_transport_gradient_report is build_boundary_transport_gradient_report
    report = build_boundary_transport_gradient_report(FakeOptimizer(), top_n=1)
    out = write_boundary_transport_gradient_report(report, tmp_path / "gradient.json")

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["kind"] == "vmec_jax_transport_gradient_diagnostic"
    assert payload["top_gradient_components"][0]["name"] == "zs10"
