"""Tests for W7-X and zonal-response artifact panels."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest

from spectraxgk.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from spectraxgk.benchmarks import (
    EigenfunctionComparisonMetrics,
    save_eigenfunction_reference_bundle,
)
from spectraxgk.workflows.runtime.config import (
    RuntimeConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)
from support.paths import REPO_ROOT


def load_artifact_tool(script_name: str):
    """Load a repository artifact tool without relying on package installation."""

    tools_dir = REPO_ROOT / "tools" / "artifacts"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_loaded_{script_name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# W7-X TEM extension status assertions
def test_w7x_tem_extension_status_tracks_open_tem_and_multiflux(tmp_path: Path) -> None:
    spectrum = tmp_path / "w7x_spectrum.json"
    spectrum.write_text(
        json.dumps(
            {
                "source_gate_passed": True,
                "time_samples": 12,
                "dominant_phi_ky": 0.19,
                "dominant_heat_flux_ky": 1.28,
            }
        ),
        encoding="utf-8",
    )
    tem = tmp_path / "tem.csv"
    tem.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,2.0,1.0,3.0,0.5,0.5,-0.5\n"
        "0.3,2.0,1.0,2.1,1.1,0.05,0.1\n",
        encoding="utf-8",
    )

    payload = load_artifact_tool("build_w7x_tem_extension_status").build_status_payload(
        w7x_spectrum=spectrum, tem_table=tem, tem_audit=tmp_path / "missing.json"
    )
    rows = {row["lane"]: row for row in payload["rows"]}

    assert payload["summary"] == {
        "n_rows": 4,
        "n_closed": 1,
        "n_partial": 0,
        "n_open": 3,
    }
    assert rows["W7-X nonlinear fluctuation spectrum"]["status"] == "closed"
    assert rows["TEM / kinetic-electron linear parity"]["status"] == "open"
    assert (
        rows["TEM / kinetic-electron linear parity"]["key_metrics"]["max_abs_rel_gamma"]
        == 0.5
    )
    assert rows["W7-X multi-flux-tube and multi-surface scan"]["status"] == "open"
    assert rows["W7-X kinetic-electron/TEM nonlinear window"]["status"] == "open"


def test_w7x_tem_extension_status_prefers_tem_audit_when_available(
    tmp_path: Path,
) -> None:
    spectrum = tmp_path / "w7x_spectrum.json"
    spectrum.write_text(json.dumps({"source_gate_passed": True}), encoding="utf-8")
    tem = tmp_path / "tem.csv"
    tem.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,2.0,1.0,2.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    audit = tmp_path / "tem_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "open",
                "claim_level": "provisional",
                "metrics": {
                    "n_ky": 1,
                    "max_abs_rel_gamma": 4.0,
                    "max_abs_rel_omega_ref_ge_0p2": 2.0,
                    "omega_branch_inversion": True,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = load_artifact_tool("build_w7x_tem_extension_status").build_status_payload(
        w7x_spectrum=spectrum, tem_table=tem, tem_audit=audit
    )
    row = {row["lane"]: row for row in payload["rows"]}[
        "TEM / kinetic-electron linear parity"
    ]

    assert row["status"] == "open"
    assert row["primary_artifact"] == "docs/_static/tem_branch_parity_audit.json"
    assert row["key_metrics"]["audit_available"] is True
    assert row["key_metrics"]["max_abs_rel_gamma"] == 4.0


def test_w7x_tem_extension_status_writes_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "w7x_tem_extension_status",
        "rows": [
            {
                "lane": "W7-X nonlinear fluctuation spectrum",
                "status": "closed",
                "claim_level": "validated",
                "primary_artifact": "w7x.json",
                "key_metrics": {"time_samples": 4, "dominant_phi_ky": 0.2},
                "next_action": "Keep scoped.",
            },
            {
                "lane": "TEM / kinetic-electron linear parity",
                "status": "open",
                "claim_level": "open",
                "primary_artifact": "tem.csv",
                "key_metrics": {"max_abs_rel_gamma": 0.75},
                "next_action": "Fix mismatch.",
            },
        ],
        "summary": {"n_rows": 2, "n_closed": 1, "n_partial": 0, "n_open": 1},
    }

    paths = load_artifact_tool("build_w7x_tem_extension_status").write_artifacts(
        payload, out_png=tmp_path / "status.png"
    )

    for path in paths.values():
        assert Path(path).exists()
    assert (
        json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["summary"][
            "n_open"
        ]
        == 1
    )


# W7-X reference overlay assertions
def test_w7x_reference_loader_rejects_nonfinite_bundle(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_w7x_reference_overlay")
    bundle = tmp_path / "bad_w7x_ref.npz"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=np.array([-1.0, 0.0, 1.0]),
        mode=np.array([1.0 + 0.0j, np.nan + 0.0j, 0.5 + 0.0j]),
        source="GX",
        case="w7x_linear",
    )

    with pytest.raises(ValueError, match="non-finite reference mode"):
        mod._load_finite_reference(bundle)


def test_w7x_eigenfunction_gate_report_uses_strict_publication_thresholds() -> None:
    mod = load_artifact_tool("generate_w7x_reference_overlay")

    report = mod._w7x_eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.50, relative_l2=0.80, phase_shift=0.0)
    )

    assert report.case == "w7x_linear_eigenfunction_ky0p3000"
    assert report.source == "GX raw eigenfunction bundle"
    assert report.passed is False
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"] == 0.95
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"] == 0.25


def test_w7x_overlay_main_writes_gate_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("generate_w7x_reference_overlay")
    theta = np.linspace(-np.pi, np.pi, 32)
    reference = np.cos(theta) + 0.25j * np.sin(theta)
    bundle = tmp_path / "w7x_ref.npz"
    spectrax_csv = tmp_path / "w7x_spectrax.csv"
    out_png = tmp_path / "w7x_overlay.png"
    out_json = tmp_path / "w7x_overlay.json"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=theta,
        mode=reference,
        source="GX",
        case="w7x_linear",
        metadata={"ky": 0.3},
    )

    def fake_gx_reference(_path):
        time = np.array([0.0, 1.0])
        ky = np.array([0.0, 0.3])
        kx = np.array([0.0])
        zero = np.zeros((time.size, ky.size, kx.size), dtype=float)
        return time, ky, kx, zero, zero, zero, zero, zero

    def fake_run(_args, *, reference_times, output_steps):
        assert np.array_equal(reference_times, np.array([0.0, 1.0]))
        assert np.array_equal(output_steps, np.array([0, 1]))
        return {
            "theta": theta,
            "mode": reference * np.exp(0.37j),
            "gamma_last": 0.0093,
            "omega_last": -0.2319,
            "Wg_last": 1.0,
            "Wphi_last": 2.0,
            "Wapar_last": 0.0,
            "Phi2_last": 3.0,
            "t_final": 1.0,
            "nl": 8,
            "nm": 16,
            "ny": 82,
            "kx_local": 0.0,
            "kx_ref": 0.0,
        }

    monkeypatch.setattr(mod, "_load_gx_reference", fake_gx_reference)
    monkeypatch.setattr(mod, "_run_w7x_spectrax_mode", fake_run)

    mod.main(
        [
            "--gx",
            str(tmp_path / "dummy.out.nc"),
            "--gx-input",
            str(tmp_path / "dummy.in"),
            "--geometry-file",
            str(tmp_path / "dummy.eik.nc"),
            "--bundle-out",
            str(bundle),
            "--out-csv",
            str(spectrax_csv),
            "--out-png",
            str(out_png),
            "--out-json",
            str(out_json),
        ]
    )

    assert spectrax_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["eigenfunction_gate_passed"] is True
    assert data["validation_status"] == "closed"
    assert data["gate_report"]["case"] == "w7x_linear_eigenfunction_ky0p3000"


# W7-X zonal response panel assertions
def test_generate_w7x_zonal_response_panel_main(tmp_path, monkeypatch) -> None:
    mod = load_artifact_tool("generate_w7x_zonal_response_panel")

    config = tmp_path / "w7x_test4.toml"
    config.write_text(
        """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 0.0
fprim = 0.0
kinetic = true

[grid]
Nx = 6
Ny = 4
Nz = 32
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"
nperiod = 4

[time]
t_max = 10.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "vmec"
vmec_file = "$W7X_VMEC_FILE"
torflux = 0.64
alpha = 0.0
R0 = 5.485

[init]
init_field = "phi"
init_amp = 1.0e-6
gaussian_init = true
gaussian_width = 0.5
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false
collisions = false
hypercollisions = false

[run]
ky = 0.0
kx = 0.05
Nl = 4
Nm = 8
dt = 0.1
steps = 100
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"
    run_calls = []

    def _fake_run(cfg, *, out, kx_target, **kwargs):
        run_calls.append(
            (
                float(kx_target),
                cfg.grid,
                cfg.time.nstep_restart,
                cfg.output,
                cfg.init,
                cfg.physics,
                cfg.terms,
                cfg.collisions,
                dict(kwargs),
            )
        )
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 10.0, 41)
        signal = np.exp(-0.18 * t) * np.cos(1.35 * t) + 0.12
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array(
                [-float(kx_target), 0.0, float(kx_target)]
            )
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = signal
            diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = (
                raw
            )
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--out-png",
            str(out_png),
            "--dt",
            "0.2",
            "--steps",
            "80",
            "--sample-stride",
            "2",
            "--time-scale",
            "3",
            "--checkpoint-steps",
            "20",
            "--Nl",
            "6",
            "--Nm",
            "10",
            "--gaussian-width",
            "1.25",
            "--nu-hyper-m",
            "0.01",
            "--p-hyper-m",
            "4",
            "--show-progress",
        ],
    )

    assert mod.main() == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    assert out_png.with_suffix(".traces.csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["summary_csv"].endswith("w7x_panel.csv")
    assert meta["traces_csv"].endswith("w7x_panel.traces.csv")
    assert meta["initial_policy"] == "first_abs"
    assert meta["initial_normalization"] == "line_first"
    assert meta["initial_level_override"] is None
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert meta["validation_status"] == "open"
    assert len(meta["cases"]) == 4
    assert meta["literature_reference"]["test"] == 4
    assert meta["literature_reference"]["flux_tube"] == "bean"
    assert (
        meta["literature_reference"]["observable"]
        == "unweighted line-averaged electrostatic potential"
    )
    assert "t=0 line-average" in meta["literature_reference"]["normalization"]
    assert "slower stellarator-specific oscillation" in meta["notes"]
    assert "default --initial-normalization=line_first" in meta["notes"]
    assert "clipped initial portion of Fig. 11" in meta["notes"]
    assert "manuscript-policy inference" in meta["notes"]
    assert "digitized-reference gate" in meta["notes"]
    assert meta["audit_overrides"]["enable_hypercollisions"] is True
    assert meta["audit_overrides"]["gaussian_width"] == 1.25
    assert meta["audit_overrides"]["nu_hyper"] is None
    assert meta["audit_overrides"]["nu_hyper_m"] == 0.01
    assert meta["audit_overrides"]["p_hyper_m"] == 4.0
    assert meta["audit_overrides"]["hypercollisions_const"] == 1.0
    assert meta["audit_overrides"]["hypercollisions_kz"] is None
    assert meta["runtime"] == {
        "dt": 0.2,
        "steps": 80,
        "sample_stride": 2,
        "checkpoint_steps": 20,
        "resume_output": False,
        "time_scale": 3.0,
        "diagnostics": True,
        "show_progress": True,
        "expected_tmax": 16.0,
        "Nl": 6,
        "Nm": 10,
    }
    assert len(run_calls) == 4
    trace = np.loadtxt(out_dir / "w7x_test4_kx050.csv", delimiter=",", skiprows=1)
    assert np.isclose(trace[-1, 0], 30.0)
    combined = np.genfromtxt(
        out_png.with_suffix(".traces.csv"), delimiter=",", names=True
    )
    assert combined.size == 4 * 41
    assert np.isclose(np.max(combined["t_reference"]), 30.0)
    assert "response_normalized" in combined.dtype.names
    for (
        kx_target,
        grid,
        nstep_restart,
        output,
        init,
        physics,
        terms,
        collisions,
        kwargs,
    ) in run_calls:
        assert grid.boundary == "periodic"
        assert grid.non_twist is True
        assert grid.jtwist is None
        assert np.isclose(grid.Lx, 2.0 * np.pi / kx_target)
        assert nstep_restart == 20
        assert output.restart_if_exists is False
        assert output.append_on_restart is True
        assert output.save_for_restart is True
        assert init.gaussian_width == 1.25
        assert physics.hypercollisions is True
        assert terms.hypercollisions == 1.0
        assert collisions.nu_hyper_m == 0.01
        assert collisions.p_hyper_m == 4.0
        assert collisions.hypercollisions_const == 1.0
        assert collisions.hypercollisions_kz == 1.0
        assert kwargs["dt"] == 0.2
        assert kwargs["steps"] == 80
        assert kwargs["sample_stride"] == 2
        assert kwargs["show_progress"] is True
        assert kwargs["Nl"] == 6
        assert kwargs["Nm"] == 10


def test_generate_w7x_zonal_response_panel_resume_output(tmp_path, monkeypatch) -> None:
    mod = load_artifact_tool("generate_w7x_zonal_response_panel")

    config = tmp_path / "w7x_test4.toml"
    config.write_text(
        """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0

[grid]
Nx = 6
Ny = 4
Nz = 16
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"

[time]
t_max = 1.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "s-alpha"
R0 = 5.485

[init]
init_field = "phi"
init_amp = 1.0e-6
gaussian_init = true
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false

[run]
ky = 0.0
Nl = 2
Nm = 4
dt = 0.1
steps = 4
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"
    seen = []

    def _fake_run(cfg, *, out, kx_target, **_kwargs):
        seen.append(
            (
                cfg.output.restart_if_exists,
                cfg.output.append_on_restart,
                Path(cfg.output.path),
                Path(out),
            )
        )
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 1.0, 8)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array(
                [-float(kx_target), 0.0, float(kx_target)]
            )
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = 1.0 + 0.01 * t
            diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = (
                raw
            )
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--out-png",
            str(out_png),
            "--kx-values",
            "0.07",
            "--resume-output",
        ],
    )

    assert mod.main() == 0
    assert seen == [
        (
            True,
            True,
            out_dir / "w7x_test4_kx070.out.nc",
            out_dir / "w7x_test4_kx070.out.nc",
        )
    ]
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["runtime"]["resume_output"] is True
    assert meta["runtime"]["time_scale"] == 1.0


def test_generate_w7x_zonal_response_formats_unresolved_damping() -> None:
    mod = load_artifact_tool("generate_w7x_zonal_response_panel")

    assert mod._finite_or_none(float("nan")) is None
    assert mod._format_metric(None) == "not fitted"
    assert mod._format_metric(float("nan")) == "not fitted"
    assert mod._format_metric(1.23456) == "1.235"


# W7-X exact-state audit assertions
def _plot_w7x_exact_state_audit_audit_dir(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "startup.log").write_text(
        "\n".join(
            [
                "g_state      max|ref|=6.751e-04 max|test|=6.751e-04 max|diff|=6.000e-11 max|rel|=1.332e-07 rms_rel=7.303e-08 idx=(0,)",
                "phi          max|ref|=5.252e-04 max|test|=5.252e-04 max|diff|=1.567e-10 max|rel|=7.362e-07 rms_rel=1.218e-07 idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "diag_state.log").write_text(
        "\n".join(
            [
                "kperp2       max|ref|=1.474e+01 max|test|=1.474e+01 max|diff|=3.338e-06 max|rel|=7.082e-07 rms_rel=9.822e-08 idx=(0,)",
                "fluxfac      max|ref|=1.950e-02 max|test|=1.950e-02 max|diff|=5.588e-09 max|rel|=3.055e-07 rms_rel=1.379e-07 idx=(0,)",
                "apar         max|ref|=0.000e+00 max|test|=0.000e+00 max|diff|=0.000e+00 max|rel|=nan rms_rel=nan idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wg",
                "gx_out": 11.0,
                "spectrax_dump": 11.0 * (1 + 1.0e-7),
                "rel_dump": 1.0e-7,
                "spectrax_solve": 11.0 * (1 + 2.0e-7),
                "rel_solve": 2.0e-7,
            },
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wapar",
                "gx_out": 0.0,
                "spectrax_dump": 0.0,
                "rel_dump": 0.0,
                "spectrax_solve": 0.0,
                "rel_solve": 0.0,
            },
        ]
    ).to_csv(path / "diag_state.csv", index=False)


def test_w7x_exact_state_audit_parses_and_writes_outputs(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_exact_state_audit")
    audit_dir = tmp_path / "audit" / "w7x_vmec"
    _plot_w7x_exact_state_audit_audit_dir(audit_dir)

    rows = mod.build_rows(audit_dir)
    assert {row["phase"] for row in rows} == {
        "startup",
        "late arrays",
        "late diagnostics",
    }
    assert (
        max(float(row["value"]) for row in rows if row["value"] == row["value"])
        < 1.0e-6
    )

    out_png = tmp_path / "w7x_exact_state_audit.png"
    rc = mod.main(["--audit-dir", str(audit_dir), "--out-png", str(out_png)])

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "closed"
    assert payload["gate_index_include"] is False
    assert payload["max_finite_relative_error"] < 1.0e-6


# W7-X fluctuation spectrum assertions
def _plot_w7x_fluctuation_spectrum_panel_synthetic_output(path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    time = np.linspace(0.0, 30.0, 10)
    ky = np.array([0.0, 0.1, 0.2])
    kx = np.array([-0.2, 0.0, 0.2, 0.4])
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", time.size)
        root.createDimension("ky", ky.size)
        root.createDimension("kx", kx.size)
        root.createDimension("s", 1)
        root.createDimension("ri", 2)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = time
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        diag = root.createGroup("Diagnostics")
        envelope = 1.0 + 0.2 * np.sin(0.5 * time)
        phi2 = np.outer(envelope, np.array([0.02, 1.0, 0.4]))
        diag.createVariable("Phi2_kyt", "f8", ("time", "ky"))[:] = phi2
        phi2_map = np.zeros((time.size, ky.size, kx.size))
        for t_idx, scale in enumerate(envelope):
            phi2_map[t_idx] = scale * np.outer(
                np.array([0.02, 1.0, 0.4]), np.array([0.1, 0.2, 1.0, 0.3])
            )
        diag.createVariable("Phi2_kxkyt", "f8", ("time", "ky", "kx"))[:] = phi2_map
        wphi = np.zeros((time.size, 1, ky.size))
        wphi[:, 0, :] = np.outer(
            1.0 + 0.1 * np.cos(0.4 * time), np.array([0.01, 0.5, 1.5])
        )
        diag.createVariable("Wphi_kyst", "f8", ("time", "s", "ky"))[:] = wphi
        heat = np.zeros((time.size, 1, ky.size))
        heat[:, 0, :] = np.outer(
            1.0 + 0.05 * np.sin(0.3 * time), np.array([0.0, 2.0, 0.7])
        )
        diag.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = heat
        zonal = np.zeros((time.size, kx.size, 2))
        zonal[:, 2, 0] = np.cos(0.6 * time)
        zonal[:, 2, 1] = np.sin(0.6 * time)
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = zonal


def test_w7x_fluctuation_spectrum_report_and_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_fluctuation_spectrum_panel")
    out_nc = tmp_path / "w7x.out.nc"
    _plot_w7x_fluctuation_spectrum_panel_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {"gate_passed": True, "gate_report": {"passed": True, "case": "W7-X"}}
        )
    )

    report = mod.build_w7x_fluctuation_spectrum_report(
        nonlinear=out_nc,
        gate_summary=gate,
        time_min=3.0,
        time_max=28.0,
    )
    paths = mod.write_w7x_fluctuation_spectrum_artifacts(
        report, out=tmp_path / "panel.png"
    )

    assert (
        report["claim_level"]
        == "validated_nonlinear_simulation_spectrum_not_experimental_validation"
    )
    assert report["gate_index_include"] is False
    assert report["source_gate_passed"] is True
    assert report["dominant_phi_ky"] == pytest.approx(0.1)
    assert report["dominant_heat_flux_ky"] == pytest.approx(0.1)
    assert report["dominant_zonal_kx"] == pytest.approx(0.2)
    assert np.sum(report["phi2_ky_distribution"]) == pytest.approx(1.0)
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()


def test_w7x_fluctuation_spectrum_rejects_failed_gate(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_fluctuation_spectrum_panel")
    out_nc = tmp_path / "w7x.out.nc"
    _plot_w7x_fluctuation_spectrum_panel_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {"gate_passed": False, "gate_report": {"passed": False, "case": "W7-X"}}
        )
    )

    with pytest.raises(ValueError, match="did not pass"):
        mod.build_w7x_fluctuation_spectrum_report(nonlinear=out_nc, gate_summary=gate)


# W7-X zonal closure ladder assertions
def _plot_w7x_zonal_closure_ladder_reference(path: Path, *, kx: float = 0.07) -> None:
    t = np.linspace(0.0, 20.0, 21)
    rows = []
    for code, offset in (("stella", -0.01), ("GENE", 0.01)):
        for time_value in t:
            rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "t_vti_over_a": time_value,
                    "response": 0.2 + np.exp(-0.2 * time_value) + offset,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_w7x_zonal_closure_ladder_output(
    path: Path, *, kx: float = 0.07, offset: float = 0.02
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 21)
    kx_grid = np.array([-kx, 0.0, kx])
    response = 0.2 + np.exp(-0.2 * t) + offset
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx_grid.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", 8)
        ds.createDimension("l", 4)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx_grid
        phi = np.zeros((t.size, kx_grid.size, 2), dtype=float)
        phi[:, 2, 0] = response / response[0]
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi
        wg = np.ones((t.size, 1, 8, 4), dtype=float)
        wg[:, 0, -2:, :] *= 3.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg


def test_w7x_zonal_closure_ladder_builds_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_closure_ladder")
    reference = tmp_path / "reference.csv"
    out_nc = tmp_path / "run" / "w7x_test4_kx070.out.nc"
    _plot_w7x_zonal_closure_ladder_reference(reference)
    _plot_w7x_zonal_closure_ladder_output(out_nc)

    reference_t, reference_y = mod.load_reference_trace(reference, 0.07)
    rows, traces = mod.build_rows(
        [("synthetic", "paper", out_nc)],
        reference_t=reference_t,
        reference_y=reference_y,
        kx=0.07,
        t_compare=20.0,
        tail_fraction=0.3,
    )

    assert len(rows) == 1
    assert rows[0]["label"] == "synthetic"
    assert rows[0]["mean_abs_error"] < 1.0
    assert rows[0]["reference_tail_std"] > 0.0
    assert rows[0]["tail_std_ratio"] is not None
    assert rows[0]["hermite_tail_last"] > 0.0
    assert "synthetic" in traces

    out_png = tmp_path / "closure.png"
    rc = mod.main(
        [
            "--reference-traces",
            str(reference),
            "--run",
            "synthetic",
            "paper",
            str(out_nc),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["rows"][0]["family"] == "paper"


# W7-X zonal contract audit assertions
def _plot_w7x_zonal_contract_audit_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    reference_trace_rows = []
    spectrax_trace_rows = []
    compare_rows = []
    residual_rows = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        t = np.linspace(0.0, 20.0, 31)
        response = 0.1 + kx + 0.04 * np.exp(-0.12 * t) * np.cos(0.7 * t)
        for code, offset in (("stella", -0.002), ("GENE", 0.002)):
            for time_value, value in zip(t, response + offset, strict=True):
                reference_trace_rows.append(
                    {
                        "kx_rhoi": kx,
                        "code": code,
                        "t_vti_over_a": time_value,
                        "response": value,
                    }
                )
            residual_rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "residual_median": 0.1 + kx + offset,
                }
            )
        spectrax_response = response + (0.004 if kx < 0.1 else 0.02)
        for time_value, value in zip(t, spectrax_response, strict=True):
            spectrax_trace_rows.append(
                {
                    "kx_target": kx,
                    "kx_selected": kx,
                    "t_reference": time_value,
                    "phi_zonal_real": value,
                    "response_normalized": value,
                    "initial_level": 1.0,
                    "initial_normalization": "line_first",
                    "source_path": "synthetic.nc",
                }
            )
        compare_rows.append(
            {
                "kx": kx,
                "spectrax_residual": 0.1 + kx + (0.004 if kx < 0.1 else 0.02),
                "spectrax_residual_std": 0.01,
                "spectrax_tmax": 20.0,
                "reference_residual": 0.1 + kx,
                "reference_min": 0.1 + kx - 0.002,
                "reference_max": 0.1 + kx + 0.002,
                "reference_tmax": 20.0,
                "coverage_ratio": 1.0,
                "residual_abs_error": 0.004 if kx < 0.1 else 0.02,
                "residual_atol_effective": 0.02,
                "trace_available": 1,
                "tail_std": 0.01 + 0.1 * kx,
                "reference_tail_std": 0.01,
                "tail_mean_abs_error": 0.003,
                "tail_max_abs_error": 0.005,
            }
        )
    ref_traces = tmp_path / "reference_traces.csv"
    ref_residuals = tmp_path / "reference_residuals.csv"
    summary = tmp_path / "spectrax_summary.csv"
    traces = tmp_path / "spectrax_traces.csv"
    compare = tmp_path / "compare.csv"
    pd.DataFrame(reference_trace_rows).to_csv(ref_traces, index=False)
    pd.DataFrame(residual_rows).to_csv(ref_residuals, index=False)
    pd.DataFrame(
        {
            "kx_target": [0.05],
            "residual_level": [0.1],
            "residual_std": [0.01],
            "tmax": [20.0],
        }
    ).to_csv(
        summary,
        index=False,
    )
    pd.DataFrame(spectrax_trace_rows).to_csv(traces, index=False)
    pd.DataFrame(compare_rows).to_csv(compare, index=False)
    return ref_traces, ref_residuals, summary, traces, compare


def test_w7x_zonal_contract_audit_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_contract_audit")
    ref_traces, ref_residuals, summary, traces, compare = (
        _plot_w7x_zonal_contract_audit_inputs(tmp_path)
    )
    rows = mod.load_audit_rows(compare)

    assert len(rows) == 4
    assert rows[0]["residual_gate_passed"] is True
    assert rows[-1]["tail_std_ratio"] > 1.0

    out_png = tmp_path / "audit.png"
    out_csv = tmp_path / "audit.csv"
    out_json = tmp_path / "audit.json"
    rc = mod.main(
        [
            "--reference-traces",
            str(ref_traces),
            "--reference-residuals",
            str(ref_residuals),
            "--spectrax-summary",
            str(summary),
            "--spectrax-traces",
            str(traces),
            "--compare-csv",
            str(compare),
            "--out-png",
            str(out_png),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_csv.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["reference_contract"]["normalization"].startswith(
        "line-averaged potential"
    )


# W7-X zonal moment-tail audit assertions
def _plot_w7x_zonal_moment_tail_audit_output(
    path: Path, *, kx_target: float = 0.07, nm: int = 8, nl: int = 4
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 11)
    kx = np.array([-kx_target, 0.0, kx_target])
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", nm)
        ds.createDimension("l", nl)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        wg = np.ones((t.size, 1, nm, nl), dtype=float)
        wg[:, 0, -2:, :] *= np.linspace(1.0, 5.0, t.size)[:, None, None]
        wg[:, 0, :, -1:] *= 2.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg
        phi = np.zeros((t.size, kx.size, 2), dtype=float)
        phi[:, 2, 0] = 1.0 + 0.1 * np.sin(t)
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi


def test_w7x_zonal_moment_tail_loads_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_moment_tail_audit")
    run_dir = tmp_path / "run"
    _plot_w7x_zonal_moment_tail_audit_output(run_dir / "w7x_test4_kx070.out.nc")

    rows, heatmap = mod.load_audit_rows(
        [("synthetic", run_dir)],
        kx_values=(0.07,),
        tail_fraction=0.3,
        hermite_tail_fraction=0.25,
        laguerre_tail_fraction=0.25,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "synthetic"
    assert row["kx_index"] == 2
    assert row["hermite_tail_last"] > 0.0
    assert row["laguerre_tail_last"] > 0.0
    assert heatmap is None

    out_png = tmp_path / "audit.png"
    rc = mod.main(
        [
            "--run",
            "synthetic",
            str(run_dir),
            "--kx-values",
            "0.07",
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["rows"][0]["Nm"] == 8


# W7-X zonal recurrence sweep assertions
def _plot_w7x_zonal_recurrence_sweep_reference(path: Path, *, kx: float = 0.07) -> None:
    t = np.linspace(0.0, 20.0, 21)
    rows = []
    for code, offset in (("stella", -0.01), ("GENE", 0.01)):
        for time_value in t:
            rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "t_vti_over_a": time_value,
                    "response": 0.25 + np.exp(-0.15 * time_value) + offset,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_w7x_zonal_recurrence_sweep_output(
    path: Path, *, kx: float = 0.07, nm: int = 8, nl: int = 4, offset: float = 0.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 21)
    kx_grid = np.array([-kx, 0.0, kx])
    response = 0.25 + np.exp(-0.15 * t) + offset
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx_grid.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", nm)
        ds.createDimension("l", nl)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx_grid
        phi = np.zeros((t.size, kx_grid.size, 2), dtype=float)
        phi[:, 2, 0] = response
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi
        wg = np.ones((t.size, 1, nm, nl), dtype=float)
        wg[:, 0, -2:, :] *= 2.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg


def test_w7x_zonal_recurrence_sweep_builds_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_recurrence_sweep")
    reference = tmp_path / "reference.csv"
    out_a = tmp_path / "moment" / "a.out.nc"
    out_b = tmp_path / "closure" / "b.out.nc"
    _plot_w7x_zonal_recurrence_sweep_reference(reference)
    _plot_w7x_zonal_recurrence_sweep_output(out_a, nm=8, nl=4, offset=0.02)
    _plot_w7x_zonal_recurrence_sweep_output(out_b, nm=12, nl=6, offset=0.01)

    reference_t, reference_y = mod.load_reference_trace(reference, 0.07)
    rows, traces = mod.build_sweep(
        [
            ("moment", "moment_resolution", "none", out_a),
            ("closure", "closure_source", "kz", out_b),
        ],
        reference_t=reference_t,
        reference_y=reference_y,
        kx=0.07,
        analysis_tmax=20.0,
        tail_fraction=0.3,
    )

    assert len(rows) == 2
    assert rows[0]["sweep"] == "moment_resolution"
    assert rows[1]["closure_source"] == "kz"
    assert rows[0]["tail_std_ratio"] is not None
    assert rows[0]["tail_std_ratio"] > 0.0
    assert "moment" in traces

    out_png = tmp_path / "recurrence.png"
    rc = mod.main(
        [
            "--reference-traces",
            str(reference),
            "--run",
            "moment",
            "moment_resolution",
            "none",
            str(out_a),
            "--run",
            "closure",
            "closure_source",
            "kz",
            str(out_b),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False


def test_w7x_zonal_recurrence_sweep_allows_one_sweep_family(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_recurrence_sweep")
    reference = tmp_path / "reference.csv"
    out_nc = tmp_path / "closure" / "b.out.nc"
    _plot_w7x_zonal_recurrence_sweep_reference(reference)
    _plot_w7x_zonal_recurrence_sweep_output(out_nc, nm=12, nl=6, offset=0.01)

    out_png = tmp_path / "closure_only.png"
    rc = mod.main(
        [
            "--reference-traces",
            str(reference),
            "--run",
            "closure",
            "closure_source",
            "const",
            str(out_nc),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["sweep"] == "closure_source"


# W7-X zonal state-convention assertions
def _plot_w7x_zonal_state_convention_audit_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=6, Ny=4, Nz=32, Lx=2.0 * np.pi / 0.07, Ly=62.8, boundary="periodic"
        ),
        time=TimeConfig(t_max=0.1, dt=0.01, method="rk4", use_diffrax=False),
        geometry=GeometryConfig(
            model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778
        ),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, density=1.0, temperature=1.0, kinetic=True
            ),
        ),
        init=InitializationConfig(
            init_field="phi",
            init_amp=0.25,
            gaussian_init=True,
            gaussian_width=1.0,
            init_single=True,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )


def test_w7x_zonal_state_convention_audit_closes_synthetic_phi_state(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_w7x_zonal_state_convention_audit")
    audit = mod.build_state_audit(
        _plot_w7x_zonal_state_convention_audit_cfg(),
        kx_target=0.07,
        ky_target=0.0,
        Nl=2,
        Nm=2,
    )

    row = audit["row"]
    assert audit["passed"] is True
    assert row["profile_relative_l2"] < 1.0e-4
    assert row["line_helper_vs_manual_rel"] < 1.0e-6
    assert row["mode_helper_vs_manual_rel"] < 1.0e-6
    assert row["line_first_initial_over_init_amp"] < 1.0

    out_png = tmp_path / "state.png"
    mod.write_outputs(
        audit,
        out_png=out_png,
        out_csv=out_png.with_suffix(".csv"),
        out_json=out_png.with_suffix(".json"),
        config=Path("synthetic.toml"),
    )

    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "state_convention_closed"
    assert payload["gate_index_include"] is False
