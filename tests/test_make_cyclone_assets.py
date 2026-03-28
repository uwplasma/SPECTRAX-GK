import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp


class _DummyFigure:
    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_make_tables_refresh_minimal_uses_reference_mismatch_scan(monkeypatch, tmp_path: Path) -> None:
    import tools.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([0.1, 0.2, 0.55]),
        gamma=np.array([0.3, 0.4, 0.5]),
        omega=np.array([0.5, 0.6, 0.7]),
    )
    called: dict[str, object] = {}

    def fake_reference_scan(scan_ref, cfg, *, verbose: bool, progress: bool):
        called["helper"] = "reference"
        called["Ny"] = cfg.grid.Ny
        assert np.allclose(scan_ref.ky, [0.1, 0.2])
        assert verbose is False
        assert progress is False
        return make_tables.LinearScanResult(ky=np.array([0.1, 0.2]), gamma=np.array([0.3, 0.4]), omega=np.array([0.5, 0.6]))

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
    assert called == {"helper": "reference", "Ny": 7}
    assert (tmp_path / "docs" / "_static" / "cyclone_mismatch_table.csv").exists()


def test_make_figures_cyclone_fallback_uses_reference_mismatch_scan(monkeypatch, tmp_path: Path) -> None:
    import tools.make_figures as make_figures

    ref = make_figures.LinearScanResult(
        ky=np.array([0.1, 0.2, 0.55]),
        gamma=np.array([0.3, 0.4, 0.5]),
        omega=np.array([0.5, 0.6, 0.7]),
    )
    called: dict[str, object] = {}

    def fake_reference_scan(scan_ref, cfg, *, verbose: bool, progress: bool):
        called["helper"] = "reference"
        called["Ny"] = cfg.grid.Ny
        assert np.allclose(scan_ref.ky, [0.1, 0.2])
        assert verbose is False
        assert progress is False
        return make_figures.LinearScanResult(ky=np.array([0.1, 0.2]), gamma=np.array([0.3, 0.4]), omega=np.array([0.5, 0.6]))

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
    assert called == {"helper": "reference", "Ny": 7}


def test_make_tables_reference_mismatch_scan_uses_dedicated_gx_scan(monkeypatch) -> None:
    import tools.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.3, 0.4]),
        omega=np.array([0.5, 0.6]),
    )
    called: dict[str, object] = {}

    def fake_cyclone_gx_scan(ky_values, cfg, window_kw, *, verbose: bool, progress: bool):
        called["ky"] = np.asarray(ky_values).copy()
        called["Ny"] = cfg.grid.Ny
        called["window"] = dict(window_kw)
        called["verbose"] = verbose
        called["progress"] = progress
        return make_tables.CycloneScanResult(
            ky=np.asarray(ky_values), gamma=np.array([1.0, 2.0]), omega=np.array([3.0, 4.0])
        )

    monkeypatch.setattr(make_tables, "_cyclone_gx_scan", fake_cyclone_gx_scan)

    out = make_tables._cyclone_reference_mismatch_scan(
        ref,
        make_tables.CycloneBaseCase(grid=make_tables._cyclone_refresh_grid(ref)),
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["Ny"] == 4
    assert called["window"] == make_tables.GX_CYCLONE_WINDOW
    assert called["verbose"] is False
    assert called["progress"] is False
    assert np.allclose(out.gamma, [1.0, 2.0])


def test_make_figures_reference_mismatch_scan_uses_dedicated_gx_scan(monkeypatch) -> None:
    import tools.make_figures as make_figures

    ref = make_figures.LinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.3, 0.4]),
        omega=np.array([0.5, 0.6]),
    )
    called: dict[str, object] = {}

    def fake_cyclone_gx_scan(ky_values, cfg, window_kw, *, verbose: bool, progress: bool):
        called["ky"] = np.asarray(ky_values).copy()
        called["Ny"] = cfg.grid.Ny
        called["window"] = dict(window_kw)
        called["verbose"] = verbose
        called["progress"] = progress
        return (
            make_figures.LinearScanResult(
                ky=np.asarray(ky_values), gamma=np.array([1.0, 2.0]), omega=np.array([3.0, 4.0])
            ),
            0.2,
        )

    monkeypatch.setattr(make_figures, "_cyclone_gx_scan", fake_cyclone_gx_scan)

    out = make_figures._cyclone_reference_mismatch_scan(
        ref,
        make_figures.CycloneBaseCase(grid=make_figures._cyclone_refresh_grid(ref)),
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["Ny"] == 4
    assert called["window"] == make_figures.GX_CYCLONE_WINDOW
    assert called["verbose"] is False
    assert called["progress"] is False
    assert np.allclose(out.gamma, [1.0, 2.0])


def test_make_tables_etg_reference_mismatch_scan_uses_tracked_scan(monkeypatch) -> None:
    import tools.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([10.0, 20.0]),
        gamma=np.array([1.0, 2.0]),
        omega=np.array([3.0, 4.0]),
    )
    called: dict[str, object] = {}

    def fake_run_etg_scan(ky_values, **kwargs):
        called["ky"] = np.asarray(ky_values).copy()
        called["solver"] = kwargs["solver"]
        called["mode_method"] = kwargs["mode_method"]
        called["fit_signal"] = kwargs["fit_signal"]
        called["diagnostic_norm"] = kwargs["diagnostic_norm"]
        return make_tables.LinearScanResult(
            ky=np.asarray(ky_values), gamma=np.array([5.0, 6.0]), omega=np.array([7.0, 8.0])
        )

    monkeypatch.setattr(make_tables, "run_etg_scan", fake_run_etg_scan)

    out = make_tables._etg_reference_mismatch_scan(
        ref,
        make_tables.ETGBaseCase(),
        dt=0.01,
        steps=600,
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["solver"] == "krylov"
    assert called["mode_method"] == "z_index"
    assert called["fit_signal"] == "phi"
    assert called["diagnostic_norm"] == make_tables.DIAGNOSTIC_NORM
    assert np.allclose(out.gamma, [5.0, 6.0])


def test_make_tables_cyclone_gx_scan_falls_back_from_project_to_max(monkeypatch) -> None:
    import tools.make_tables as make_tables

    cfg = make_tables.CycloneBaseCase()

    monkeypatch.setattr(
        make_tables,
        "SAlphaGeometry",
        type(
            "Geom",
            (),
            {"from_config": staticmethod(lambda _cfg: type("G", (), {"gradpar": lambda self: 1.0})())},
        ),
    )
    monkeypatch.setattr(
        make_tables,
        "build_spectral_grid",
        lambda _grid: type("Grid", (), {"ky": np.array([0.4]), "z": np.array([0.0, 1.0, 2.0])})(),
    )
    monkeypatch.setattr(make_tables, "select_ky_index", lambda _ky, _val: 0)
    monkeypatch.setattr(make_tables, "select_ky_grid", lambda grid, _idx: grid)
    monkeypatch.setattr(make_tables, "_build_initial_condition", lambda *args, **kwargs: jnp.zeros((2, 2), dtype=jnp.complex64))
    monkeypatch.setattr(make_tables, "build_linear_cache", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        make_tables,
        "integrate_linear_gx",
        lambda *args, **kwargs: (np.array([0.0, 1.0]), np.ones((2, 1, 1, 3), dtype=np.complex128), None, None),
    )
    methods: list[str] = []

    def fake_extract(_phi_t, _sel, *, method: str):
        methods.append(method)
        if method == "project":
            return np.array([1.0e-12 + 0.0j, 1.0e-12 + 0.0j])
        return np.array([1.0 + 0.0j, np.exp((0.1 - 0.4j))])

    monkeypatch.setattr(make_tables, "extract_mode_time_series", fake_extract)

    def fake_fit(t, signal, **_kwargs):
        if np.max(np.abs(signal)) < 1.0e-6:
            return 1.0e-13, 0.0, float(t[0]), float(t[-1])
        return 0.1, 0.4, float(t[0]), float(t[-1])

    monkeypatch.setattr(make_tables, "fit_growth_rate_auto", fake_fit)

    scan = make_tables._cyclone_gx_scan(np.array([0.4]), cfg, make_tables.WINDOWS["cyclone"], verbose=False, progress=False)

    assert methods == ["project", "max"]
    assert np.allclose(scan.gamma, [0.1])
    assert np.allclose(scan.omega, [0.4])
