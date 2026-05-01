from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "profile_linear_rhs_terms.py"
spec = importlib.util.spec_from_file_location("profile_linear_rhs_terms", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_summary_reports_dominant_and_zero_norm_terms() -> None:
    payload = mod._build_summary(
        [
            {"term": "field_solve", "seconds": 0.1, "norm": 1.0},
            {"term": "streaming", "seconds": 0.2, "norm": 2.0},
            {"term": "collisions", "seconds": 0.4, "norm": 0.0},
            {"term": "linked_abs_kz", "seconds": 0.3, "norm": 1.0e-16},
            {"term": "full_linear_rhs", "seconds": 1.2, "norm": 3.0},
        ],
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        repeats=5,
        backend="cpu",
    )

    assert payload["kind"] == "linear_rhs_terms_profile_summary"
    assert payload["case"] == "runtime_cyclone_nonlinear"
    assert payload["dominant_measured_term"] == "collisions"
    assert payload["dominant_nonzero_norm_term"] == "streaming"
    assert payload["zero_norm_terms_by_time"][0]["term"] == "collisions"
    assert payload["full_over_sum_independently_measured_components"] == 1.2
    assert "initial state only" in payload["claim_scope"]


def test_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    mod._write_summary_json({"kind": "linear", "value": 2.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "linear", "value": 2.0}
