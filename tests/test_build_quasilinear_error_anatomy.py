from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _load_tool_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "build_quasilinear_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("build_quasilinear_error_anatomy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_error_anatomy_locks_current_fail_closed_residual_story() -> None:
    mod = _load_tool_module()

    report = mod.build_error_anatomy_report()

    assert report["kind"] == "quasilinear_error_anatomy"
    assert report["claim_level"] == "model_development_residual_anatomy_not_absolute_flux_promotion"
    assert report["case_count"] == 10
    assert report["holdout_count"] == 8
    assert report["promotion_gate"]["passed"] is False
    assert "case_residuals_exceed_transport_gate" in report["promotion_gate"]["blockers"]
    assert 0.422 < report["candidate_mean_abs_relative_error"] < 0.425
    assert report["rows"][0]["case"] == "shaped_tokamak_pressure_external_vmec_t650_high_grid_window"
    assert report["rows"][0]["above_transport_gate"] is True
    assert report["rows"][0]["overpredicts"] is True
    groups = {row["geometry_group"]: row for row in report["geometry_group_summary"]}
    assert groups["external axisymmetric VMEC"]["error_budget_fraction"] > 0.55
    assert groups["stellarator benchmark"]["mean_abs_relative_error"] < 0.2


def test_error_anatomy_cli_writes_sidecars_and_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "ql_error_anatomy.png"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "build_quasilinear_error_anatomy.py"),
            "--out",
            str(out),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "promotion_passed=False" in completed.stdout
    assert out.exists()
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["promotion_gate"]["passed"] is False
    csv_text = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert csv_text.startswith("case,label,split,geometry")
