from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "campaigns" / "import_nonlinear_transport_matrix_portfolio.py"
mod = load_campaign_tool("import_nonlinear_transport_matrix_portfolio")


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_png_stub(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def test_import_copies_only_selected_passing_family(tmp_path: Path) -> None:
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
        tmp_path / "projected.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "projected.png")
    docs_static = tmp_path / "docs_static"

    manifest = mod.import_artifacts(
        portfolio_json=portfolio,
        portfolio_figure=portfolio_png,
        matrix_report_jsons={"projected_0p001": matrix_json},
        matrix_report_figures={"projected_0p001": matrix_png},
        docs_static=docs_static,
    )

    assert manifest["passed"] is True
    assert manifest["selected_family"] == "projected_0p001"
    assert (docs_static / "nonlinear_transport_matrix_portfolio.json").exists()
    assert (docs_static / "nonlinear_transport_matrix_portfolio.png").exists()
    assert (docs_static / "projected_0p001_matrix_report.json").exists()
    assert (docs_static / "projected_0p001_matrix_report.png").exists()
    import_manifest = json.loads(
        (docs_static / "nonlinear_transport_matrix_portfolio_import.json").read_text(
            encoding="utf-8"
        )
    )
    assert import_manifest["kind"] == "nonlinear_transport_matrix_portfolio_import"


def test_import_refuses_blocked_portfolio(tmp_path: Path) -> None:
    portfolio = _write_json(
        tmp_path / "portfolio.json",
        {
            "kind": "nonlinear_transport_matrix_portfolio_gate",
            "passed": False,
            "selected_family": None,
        },
    )

    try:
        mod.import_artifacts(
            portfolio_json=portfolio,
            portfolio_figure=None,
            matrix_report_jsons={},
            matrix_report_figures={},
            docs_static=tmp_path / "docs_static",
        )
    except ValueError as exc:
        assert "blocked" in str(exc)
    else:  # pragma: no cover - defensive failure path
        raise AssertionError("blocked portfolio was imported")


def test_cli_writes_import_manifest(tmp_path: Path) -> None:
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
        tmp_path / "accepted.json",
        {"kind": "matched_nonlinear_transport_matrix_report", "passed": True},
    )
    matrix_png = _write_png_stub(tmp_path / "accepted.png")
    docs_static = tmp_path / "docs_static"

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
        ]
    )

    assert rc == 0
    payload = json.loads(
        (docs_static / "nonlinear_transport_matrix_portfolio_import.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["selected_family"] == "accepted_qa_ess"
