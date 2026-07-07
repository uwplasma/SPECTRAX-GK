from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "artifacts" / "build_vmec_state_to_input_mapping_response.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_vmec_state_to_input_mapping_response", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _controls() -> list[dict[str, object]]:
    return [
        {"state_parameter": "Rsin_mid_surface_m1"},
        {"state_parameter": "Zcos_mid_surface_m1"},
    ]


def _directions() -> list[dict[str, object]]:
    return [
        {
            "coefficient": "RBC(1,1)",
            "coefficient_slug": "rbc_1_1",
            "delta_parameter": 0.1,
        },
        {
            "coefficient": "ZBS(1,1)",
            "coefficient_slug": "zbs_1_1",
            "delta_parameter": 0.2,
        },
    ]


def _sample(value: tuple[float, float]) -> dict[str, float]:
    return {
        "Rsin_mid_surface_m1": value[0],
        "Zcos_mid_surface_m1": value[1],
    }


def test_mapping_report_fails_closed_for_zero_symmetric_response() -> None:
    mod = _load_tool_module()
    samples = {
        "RBC(1,1)": {
            "baseline": _sample((0.0, 0.0)),
            "plus_delta": _sample((0.0, 0.0)),
            "minus_delta": _sample((0.0, 0.0)),
        },
        "ZBS(1,1)": {
            "baseline": _sample((0.0, 0.0)),
            "plus_delta": _sample((0.0, 0.0)),
            "minus_delta": _sample((0.0, 0.0)),
        },
    }

    report = mod.mapping_report_from_samples(
        case="zero",
        admitted_state_controls=_controls(),
        input_directions=_directions(),
        samples=samples,
    )

    assert report["passed"] is False
    assert report["jacobian"]["rank"] == 0
    assert report["jacobian"]["condition_number"] is None
    assert "zero_state_response" in report["blockers"]
    assert all(
        "state_control_not_observed" in row["blockers"] for row in report["controls"]
    )
    json.dumps(report, allow_nan=False)


def test_mapping_report_passes_conditioned_square_response() -> None:
    mod = _load_tool_module()
    samples = {
        "RBC(1,1)": {
            "baseline": _sample((0.0, 0.0)),
            "plus_delta": _sample((0.1, 0.0)),
            "minus_delta": _sample((-0.1, 0.0)),
        },
        "ZBS(1,1)": {
            "baseline": _sample((0.0, 0.0)),
            "plus_delta": _sample((0.0, 0.2)),
            "minus_delta": _sample((0.0, -0.2)),
        },
    }

    report = mod.mapping_report_from_samples(
        case="identity",
        admitted_state_controls=_controls(),
        input_directions=_directions(),
        samples=samples,
    )

    assert report["passed"] is True
    assert report["jacobian"]["rank"] == 2
    assert report["jacobian"]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert [row["passed"] for row in report["controls"]] == [True, True]


def test_mapping_report_rejects_missing_states_and_bad_deltas() -> None:
    mod = _load_tool_module()
    directions = _directions()
    directions[0]["delta_parameter"] = float("nan")
    with pytest.raises(ValueError, match="delta_parameter"):
        mod.mapping_report_from_samples(
            case="bad",
            admitted_state_controls=_controls(),
            input_directions=directions,
            samples={},
        )

    with pytest.raises(ValueError, match="missing baseline"):
        mod.mapping_report_from_samples(
            case="missing",
            admitted_state_controls=_controls(),
            input_directions=_directions(),
            samples={"RBC(1,1)": {"baseline": _sample((0.0, 0.0))}},
        )
