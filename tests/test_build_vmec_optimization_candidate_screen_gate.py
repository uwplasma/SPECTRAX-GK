from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_vmec_optimization_candidate_screen_gate.py"
    spec = importlib.util.spec_from_file_location("build_vmec_optimization_candidate_screen_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_spectrum(path: Path, rows: list[tuple[float, float, float, float, float]]) -> None:
    lines = ["ky,gamma,omega,kperp_eff2,heat_flux_weight_total"]
    lines.extend(f"{ky},{gamma},{omega},{kperp},{heat}" for ky, gamma, omega, kperp, heat in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_candidate_screen_rejects_nonpositive_kperp_even_with_large_growth(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "bad.csv"
    _write_spectrum(
        spectrum,
        [
            (0.1, 1.2, -0.5, -0.7, 0.1),
            (0.2, 0.8, -0.4, -0.1, 0.2),
            (0.3, 0.4, -0.2, -0.2, 0.3),
        ],
    )

    row = mod.summarize_spectrum(label="bad_metric", spectrum_path=spectrum)

    assert row["passed"] is False
    assert row["status"] == "invalid_metric_nonpositive_kperp2"
    assert "nonpositive_effective_kperp2" in row["blockers"]
    assert row["max_gamma"] == 1.2


def test_candidate_screen_accepts_positive_metric_launch_candidate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "good.csv"
    _write_spectrum(
        spectrum,
        [
            (0.1, 0.01, -0.5, 0.7, 0.1),
            (0.2, 0.04, -0.4, 0.8, 0.2),
            (0.3, 0.03, -0.2, 0.9, 0.3),
        ],
    )

    report = mod.build_report([("good", spectrum)])

    assert report["passed"] is True
    assert report["n_launch_candidates"] == 1
    assert report["rows"][0]["status"] == "nonlinear_launch_candidate"


def test_candidate_screen_tool_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "marginal.csv"
    _write_spectrum(
        spectrum,
        [
            (0.1, -0.01, -0.5, 0.7, 0.1),
            (0.2, 0.01, -0.4, 0.8, 0.2),
            (0.3, 0.015, -0.2, 0.9, 0.3),
        ],
    )
    out = tmp_path / "screen.json"

    assert mod.main(["--spectrum", f"marginal:{spectrum}", "--out", str(out)]) == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["rows"][0]["status"] == "marginal_or_incomplete_screen"
    assert out.with_suffix(".csv").exists()
