from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "campaigns" / "finalize_nonlinear_transport_matrix_release.py"
spec = importlib.util.spec_from_file_location(
    "finalize_nonlinear_transport_matrix_release", SCRIPT
)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_png_stub(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def test_finalize_imports_passing_portfolio_without_dashboard_regeneration(
    tmp_path: Path,
) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": True,
            "selected_family": "projected_0p001",
        },
    )
    portfolio_png = _write_png_stub(tmp_path / "portfolio.png")
    matrix_json = _write_json(
        tmp_path / "matrix.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "matrix.png")
    docs_static = tmp_path / "docs" / "_static"

    manifest = mod.finalize_release_artifacts(
        portfolio_json=portfolio,
        portfolio_figure=portfolio_png,
        matrix_report_jsons={"projected_0p001": matrix_json},
        matrix_report_figures={"projected_0p001": matrix_png},
        docs_static=docs_static,
        regenerate_dashboards=False,
    )

    assert manifest["passed"] is True
    assert manifest["selected_family"] == "projected_0p001"
    assert manifest["dashboards_regenerated"] is False
    assert (docs_static / "nonlinear_transport_matrix_portfolio.json").exists()
    assert (docs_static / "projected_0p001_matrix_report.json").exists()
    saved = json.loads(
        (
            docs_static / "nonlinear_transport_matrix_release_finalization.json"
        ).read_text(encoding="utf-8")
    )
    assert saved["kind"] == "nonlinear_transport_matrix_release_finalization"
    assert saved["selected_family"] == "projected_0p001"


def test_finalize_refuses_blocked_portfolio(tmp_path: Path) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": False,
            "selected_family": None,
        },
    )

    try:
        mod.finalize_release_artifacts(
            portfolio_json=portfolio,
            portfolio_figure=None,
            matrix_report_jsons={},
            matrix_report_figures={},
            docs_static=tmp_path / "docs" / "_static",
            regenerate_dashboards=False,
        )
    except ValueError as exc:
        assert "blocked" in str(exc)
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("blocked portfolio was finalized")


def test_finalize_cli_smoke(tmp_path: Path) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": True,
            "selected_family": "accepted_qa_ess",
        },
    )
    portfolio_png = _write_png_stub(tmp_path / "portfolio.png")
    matrix_json = _write_json(
        tmp_path / "matrix.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "matrix.png")
    docs_static = tmp_path / "docs" / "_static"

    rc = mod.main(
        [
            "--portfolio-json",
            str(portfolio),
            "--portfolio-figure",
            str(portfolio_png),
            "--matrix-report-json",
            f"accepted_qa_ess={matrix_json}",
            "--matrix-report-figure",
            f"accepted_qa_ess={matrix_png}",
            "--docs-static",
            str(docs_static),
            "--skip-dashboard-regeneration",
        ]
    )

    assert rc == 0
    payload = json.loads(
        (
            docs_static / "nonlinear_transport_matrix_release_finalization.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["selected_family"] == "accepted_qa_ess"
