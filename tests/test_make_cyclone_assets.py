import sys
from pathlib import Path

import numpy as np


class _DummyFigure:
    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_make_tables_refresh_minimal_uses_reference_mismatch_scan(monkeypatch, tmp_path: Path) -> None:
    import tools.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.3, 0.4]),
        omega=np.array([0.5, 0.6]),
    )
    called: dict[str, object] = {}

    def fake_reference_scan(scan_ref, cfg, *, verbose: bool, progress: bool):
        called["helper"] = "reference"
        called["Ny"] = cfg.grid.Ny
        assert scan_ref is ref
        assert verbose is False
        assert progress is False
        return ref

    def fail_gx_scan(*_args, **_kwargs):
        raise AssertionError("GX-balanced Cyclone scan should not be used for refresh_minimal")

    monkeypatch.setattr(make_tables, "ROOT", tmp_path)
    monkeypatch.setattr(make_tables, "load_cyclone_reference", lambda: ref)
    monkeypatch.setattr(make_tables, "_cyclone_reference_mismatch_scan", fake_reference_scan)
    monkeypatch.setattr(make_tables, "_cyclone_gx_scan", fail_gx_scan)
    monkeypatch.setattr(
        make_tables,
        "_build_rows",
        lambda scan, ref_scan: ["ky,gamma,omega", f"{scan.ky[0]},{scan.gamma[0]},{scan.omega[0]}"],
    )
    monkeypatch.setattr(sys, "argv", ["make_tables.py", "--case", "cyclone", "--refresh-minimal", "--no-progress", "--quiet"])

    assert make_tables.main() == 0
    assert called == {"helper": "reference", "Ny": 18}
    assert (tmp_path / "docs" / "_static" / "cyclone_mismatch_table.csv").exists()


def test_make_figures_cyclone_fallback_uses_reference_mismatch_scan(monkeypatch, tmp_path: Path) -> None:
    import tools.make_figures as make_figures

    ref = make_figures.LinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.3, 0.4]),
        omega=np.array([0.5, 0.6]),
    )
    called: dict[str, object] = {}

    def fake_reference_scan(scan_ref, cfg, *, verbose: bool, progress: bool):
        called["helper"] = "reference"
        called["Ny"] = cfg.grid.Ny
        assert scan_ref is ref
        assert verbose is False
        assert progress is False
        return ref

    def fail_gx_scan(*_args, **_kwargs):
        raise AssertionError("GX-balanced Cyclone figure scan should not be used in fallback path")

    monkeypatch.setattr(make_figures, "ROOT", tmp_path)
    monkeypatch.setattr(make_figures, "load_cyclone_reference", lambda: ref)
    monkeypatch.setattr(make_figures, "_cyclone_reference_mismatch_scan", fake_reference_scan)
    monkeypatch.setattr(make_figures, "_cyclone_gx_scan", fail_gx_scan)
    monkeypatch.setattr(make_figures, "cyclone_reference_figure", lambda _ref: (_DummyFigure(), None))
    monkeypatch.setattr(make_figures, "cyclone_comparison_figure", lambda _ref, _scan: (_DummyFigure(), None))
    monkeypatch.setattr(sys, "argv", ["make_figures.py", "--case", "cyclone", "--no-progress", "--quiet"])

    assert make_figures.main() == 0
    assert called == {"helper": "reference", "Ny": 18}
