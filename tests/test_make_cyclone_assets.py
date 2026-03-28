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


def test_make_tables_etg_reference_mismatch_scan_uses_gx_growth_helper(monkeypatch) -> None:
    import tools.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([10.0, 20.0]),
        gamma=np.array([1.0, 2.0]),
        omega=np.array([3.0, 4.0]),
    )
    called: dict[str, object] = {}

    def fake_run_etg_gx_growth(**kwargs):
        ky = float(kwargs["ky"])
        called.setdefault("ky", []).append(ky)
        called["dt"] = kwargs["dt"]
        called["steps"] = kwargs["steps"]
        return (5.0 + ky, 7.0 + ky)

    monkeypatch.setattr(make_tables, "_run_etg_gx_growth", fake_run_etg_gx_growth)

    out = make_tables._etg_reference_mismatch_scan(
        ref,
        make_tables.ETGBaseCase(),
        dt=0.01,
        steps=make_tables.ETG_GX_MISMATCH_STEPS,
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["dt"] == 0.01
    assert called["steps"] == make_tables.ETG_GX_MISMATCH_STEPS
    assert np.allclose(out.gamma, [15.0, 25.0])
    assert np.allclose(out.omega, [17.0, 27.0])


def test_run_etg_tables_uses_tracked_mismatch_helper(monkeypatch, tmp_path) -> None:
    import tools.make_tables as make_tables

    called: dict[str, object] = {}

    def fake_run_etg_linear(**kwargs):
        return type("Res", (), {"gamma": 1.0, "omega": -2.0})()

    def fake_load_etg_reference():
        return make_tables.LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([1.0, 2.0]),
            omega=np.array([3.0, 4.0]),
        )

    def fake_etg_reference_mismatch_scan(ref, cfg, *, dt, steps, verbose, progress):
        called["ref"] = ref
        called["cfg"] = cfg
        called["dt"] = dt
        called["steps"] = steps
        called["verbose"] = verbose
        called["progress"] = progress
        return make_tables.LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([5.0, 6.0]),
            omega=np.array([7.0, 8.0]),
        )

    monkeypatch.setattr(make_tables, "run_etg_linear", fake_run_etg_linear)
    monkeypatch.setattr(make_tables, "load_etg_reference", fake_load_etg_reference)
    monkeypatch.setattr(
        make_tables,
        "_etg_reference_mismatch_scan",
        fake_etg_reference_mismatch_scan,
    )

    make_tables._run_etg_tables(outdir=tmp_path, verbose=False, progress=False)

    assert called["dt"] == 0.01
    assert called["steps"] == make_tables.ETG_GX_MISMATCH_STEPS
    assert called["verbose"] is False
    assert called["progress"] is False
    cfg = called["cfg"]
    assert cfg.model.adiabatic_ions is False
    assert cfg.model.R_over_LTi == 0.0
    assert cfg.model.R_over_Lni == 0.0
    assert cfg.model.R_over_Lne == 0.8
    assert (tmp_path / "etg_mismatch_table.csv").exists()


def test_run_etg_figures_uses_tracked_case(monkeypatch, tmp_path: Path) -> None:
    import tools.make_figures as make_figures

    called: dict[str, object] = {}

    def fake_scan_and_mode(
        _scan_fn,
        _linear_fn,
        ky_values,
        cfg,
        *,
        label,
        **kwargs,
    ):
        called["ky_values"] = np.asarray(ky_values).copy()
        called["cfg"] = cfg
        called["label"] = label
        return (
            make_figures.LinearScanResult(
                ky=np.asarray(ky_values),
                gamma=np.array([1.0, 2.0, 3.0]),
                omega=np.array([-4.0, -5.0, -6.0]),
            ),
            None,
            None,
            float(np.asarray(ky_values)[0]),
        )

    monkeypatch.setattr(
        make_figures,
        "load_etg_reference",
        lambda: make_figures.LinearScanResult(
            ky=np.array([10.0, 20.0, 30.0]),
            gamma=np.array([1.0, 2.0, 3.0]),
            omega=np.array([-4.0, -5.0, -6.0]),
        ),
    )
    monkeypatch.setattr(make_figures, "_scan_and_mode", fake_scan_and_mode)
    monkeypatch.setattr(
        make_figures,
        "scan_comparison_figure",
        lambda *args, **kwargs: (_DummyFigure(), None),
    )

    make_figures._run_etg_figures(outdir=tmp_path, verbose=False, progress=False)

    cfg = called["cfg"]
    assert np.allclose(called["ky_values"], [10.0, 20.0, 30.0])
    assert called["label"] == "ETG panel"
    assert cfg.model.adiabatic_ions is False
    assert cfg.model.R_over_LTi == 0.0
    assert cfg.model.R_over_Lni == 0.0
    assert cfg.model.R_over_Lne == 0.8


def test_run_etg_figures_prefers_existing_mismatch_csv(monkeypatch, tmp_path: Path) -> None:
    import tools.make_figures as make_figures

    mismatch = tmp_path / "etg_mismatch_table.csv"
    mismatch.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "10,1,2,3,4,0,0\n"
        "20,5,6,7,8,0,0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        make_figures,
        "load_etg_reference",
        lambda: make_figures.LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([1.0, 5.0]),
            omega=np.array([2.0, 6.0]),
        ),
    )
    monkeypatch.setattr(
        make_figures,
        "_scan_and_mode",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse mismatch csv")),
    )
    monkeypatch.setattr(
        make_figures,
        "scan_comparison_figure",
        lambda *args, **kwargs: (_DummyFigure(), None),
    )

    make_figures._run_etg_figures(outdir=tmp_path, verbose=False, progress=False)


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
