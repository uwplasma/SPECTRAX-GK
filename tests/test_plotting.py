"""Plotting utilities should generate figures without errors."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from spectraxgk.benchmarks import CycloneReference, CycloneScanResult
import matplotlib.pyplot as plt
import pytest
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    etg_trend_figure,
    growth_rate_heatmap,
    growth_fit_figure,
    linear_validation_figure,
    linear_validation_multi_reference_figure,
    LinearValidationPanel,
    MultiReferenceValidationPanel,
    ReferenceSeries,
    linear_runtime_panel_figure,
    nonlinear_runtime_panel_figure,
    plot_saved_output,
    scan_comparison_figure,
    scan_multi_reference_figure,
)


def test_cyclone_reference_figure(tmp_path):
    """The Cyclone reference plot should save successfully."""
    ref = CycloneReference(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        gamma=np.array([0.05, 0.06]),
    )
    fig, _axes = cyclone_reference_figure(ref)
    out = tmp_path / "ref.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_cyclone_comparison_figure(tmp_path):
    """Comparison plot should render with both curves."""
    ref = CycloneReference(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        gamma=np.array([0.05, 0.06]),
    )
    scan = CycloneScanResult(
        ky=np.array([0.1, 0.2]),
        omega=np.array([0.25, 0.35]),
        gamma=np.array([0.04, 0.05]),
    )
    fig, _axes = cyclone_comparison_figure(ref, scan)
    out = tmp_path / "comparison.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_etg_trend_figure(tmp_path):
    """ETG trend plot should render and save."""
    R = np.array([4.0, 6.0, 8.0])
    gamma = np.array([0.1, 0.2, 0.3])
    omega = np.array([-0.4, -0.5, -0.6])
    fig, _axes = etg_trend_figure(R, gamma, omega, ky_target=3.0)
    out = tmp_path / "etg_trend.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_growth_rate_heatmap(tmp_path):
    """Heatmap plot should render and save."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])
    gamma = np.random.random((y.size, x.size))
    fig, _ax = growth_rate_heatmap(x, y, gamma, "Test", r"$R/L_n$", r"$R/L_T$")
    out = tmp_path / "heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_validation_figure(tmp_path):
    """Summary panel should render and save."""
    z = np.linspace(-1.0, 1.0, 8)
    panel = LinearValidationPanel(
        name="Cyclone",
        z=z,
        eigenfunction=np.exp(1j * z),
        x=np.array([0.2, 0.3]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        x_label=r"$k_y$",
        x_ref=np.array([0.2, 0.3]),
        gamma_ref=np.array([0.11, 0.21]),
        omega_ref=np.array([0.31, 0.41]),
    )
    fig, _axes = linear_validation_figure([panel])
    out = tmp_path / "summary.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_validation_empty():
    """Empty panel list should raise."""
    try:
        linear_validation_figure([])
    except ValueError:
        pass
    else:
        raise AssertionError("empty panels should raise ValueError")


def test_linear_validation_multiple_panels(tmp_path):
    """Multiple panels should render without errors."""
    z = np.linspace(-1.0, 1.0, 8)
    panels = [
        LinearValidationPanel(
            name="Cyclone",
            z=z,
            eigenfunction=np.exp(1j * z),
            x=np.array([0.2, 0.3]),
            gamma=np.array([0.1, 0.2]),
            omega=np.array([0.3, 0.4]),
            x_label=r"$k_y$",
        ),
        LinearValidationPanel(
            name="ITG",
            z=z,
            eigenfunction=np.exp(1j * 0.5 * z),
            x=np.array([0.2, 0.3]),
            gamma=np.array([0.15, 0.25]),
            omega=np.array([0.35, 0.45]),
            x_label=r"$k_y$",
        ),
    ]
    fig, _axes = linear_validation_figure(panels)
    out = tmp_path / "summary_multi.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_linear_runtime_panel_figure(tmp_path):
    t = np.linspace(0.1, 1.0, 8)
    signal = np.exp((0.2 - 0.3j) * t)
    z = np.linspace(-np.pi, np.pi, 16)
    eigen = np.cos(z) + 1j * np.sin(z)
    fig, _axes = linear_runtime_panel_figure(
        t=t,
        signal=signal,
        z=z,
        eigenfunction=eigen,
        gamma=0.2,
        omega=-0.3,
    )
    out = tmp_path / "linear_runtime_panel.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_nonlinear_runtime_panel_figure(tmp_path):
    t = np.linspace(0.1, 1.0, 8)
    fig, _axes = nonlinear_runtime_panel_figure(
        t=t,
        phi2=np.exp(t),
        wphi=np.linspace(1.0, 2.0, 8),
        heat_flux=np.linspace(0.1, 0.8, 8),
        gamma=np.linspace(0.01, 0.08, 8),
        omega=np.linspace(-0.1, -0.8, 8),
    )
    out = tmp_path / "nonlinear_runtime_panel.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_plot_saved_output_linear_bundle(tmp_path):
    base = tmp_path / "linear_case"
    (tmp_path / "linear_case.summary.json").write_text(
        '{"kind":"linear","gamma":0.2,"omega":-0.3}',
        encoding="utf-8",
    )
    (tmp_path / "linear_case.timeseries.csv").write_text(
        "t,signal_real,signal_imag,signal_abs\n0.1,1.0,0.0,1.0\n0.2,1.2,0.1,1.204159\n",
        encoding="utf-8",
    )
    (tmp_path / "linear_case.eigenfunction.csv").write_text(
        "z,eigen_real,eigen_imag,eigen_abs\n-1.0,0.5,-0.2,0.538516\n0.0,1.0,0.0,1.0\n1.0,0.5,0.2,0.538516\n",
        encoding="utf-8",
    )
    out = plot_saved_output(base.with_suffix(".summary.json"))
    assert out.exists()


def test_plot_saved_output_nonlinear_csv_bundle(tmp_path):
    base = tmp_path / "nonlinear_case"
    (tmp_path / "nonlinear_case.summary.json").write_text(
        '{"kind":"nonlinear"}',
        encoding="utf-8",
    )
    (tmp_path / "nonlinear_case.diagnostics.csv").write_text(
        "t,dt,gamma,omega,Wg,Wphi,Wapar,energy,heat_flux,particle_flux\n"
        "0.1,0.1,0.01,-0.02,1.0,2.0,0.0,3.0,0.4,0.0\n"
        "0.2,0.1,0.02,-0.03,1.1,2.1,0.0,3.2,0.5,0.0\n",
        encoding="utf-8",
    )
    out = plot_saved_output(base.with_suffix(".summary.json"))
    assert out.exists()


def test_scan_comparison_figure_with_reference_and_log_scale(tmp_path):
    x = np.array([0.1, 0.2, 0.4])
    fig, axes = scan_comparison_figure(
        x,
        np.array([0.2, 0.3, 0.4]),
        np.array([-0.1, -0.2, -0.3]),
        r"$k_y$",
        "Scan",
        x_ref=x,
        gamma_ref=np.array([0.21, 0.31, 0.41]),
        omega_ref=np.array([-0.11, -0.21, -0.31]),
        log_x=True,
    )
    out = tmp_path / "scan_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0].get_xscale() == "log"


def test_linear_validation_multi_reference_figure(tmp_path):
    z = np.linspace(-1.0, 1.0, 8)
    panel = MultiReferenceValidationPanel(
        name="Cyclone",
        z=z,
        eigenfunction=np.exp(1j * z),
        x=np.array([0.2, 0.3]),
        gamma=np.array([0.1, 0.2]),
        omega=np.array([0.3, 0.4]),
        x_label=r"$k_y$",
        references=[
            ReferenceSeries(
                label="RefA",
                x=np.array([0.2, 0.3]),
                gamma=np.array([0.11, 0.21]),
                omega=np.array([0.31, 0.41]),
                color="#1f77b4",
            )
        ],
        log_x=True,
    )
    fig, axes = linear_validation_multi_reference_figure([panel])
    out = tmp_path / "linear_validation_multi.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0, 1].get_xscale() == "log"


def test_linear_validation_multi_reference_figure_empty():
    with np.testing.assert_raises(ValueError):
        linear_validation_multi_reference_figure([])


def test_scan_multi_reference_figure(tmp_path):
    x = np.array([0.1, 0.2, 0.4])
    refs = [
        ReferenceSeries(
            label="GX",
            x=x,
            gamma=np.array([0.21, 0.31, 0.41]),
            omega=np.array([-0.11, -0.21, -0.31]),
            color="#1f77b4",
        )
    ]
    fig, axes = scan_multi_reference_figure(
        x,
        np.array([0.2, 0.3, 0.4]),
        np.array([-0.1, -0.2, -0.3]),
        r"$k_y$",
        "Multi-ref",
        refs,
        log_x=True,
    )
    out = tmp_path / "scan_multi_reference.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
    assert axes[0].get_xscale() == "log"


def test_growth_fit_figure_with_window(tmp_path):
    t = np.linspace(0.0, 4.0, 32)
    signal = np.exp((0.2 - 0.1j) * t)
    fig, _axes = growth_fit_figure(t, signal, tmin=1.0, tmax=3.0)
    out = tmp_path / "growth_fit.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()


def test_plot_saved_output_missing_summary_and_bad_kind(tmp_path):
    with np.testing.assert_raises(FileNotFoundError):
        plot_saved_output(tmp_path / "missing.summary.json")

    (tmp_path / "unknown.summary.json").write_text('{"kind":"mystery"}', encoding="utf-8")
    with np.testing.assert_raises(ValueError):
        plot_saved_output(tmp_path / "unknown.summary.json")


def test_plot_saved_output_nonlinear_netcdf_bundle(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    dataset = netcdf4.Dataset
    path = tmp_path / "nonlinear_case.out.nc"
    with dataset(path, "w") as root:
        diag = root.createGroup("Diagnostics")
        diag.createDimension("time", 2)
        diag.createDimension("species", 1)
        t_var = diag.createVariable("t", "f8", ("time",))
        t_var[:] = np.array([0.1, 0.2])
        phi2 = diag.createVariable("Phi2_t", "f8", ("time",))
        phi2[:] = np.array([1.0, 2.0])
        wphi = diag.createVariable("Wphi_st", "f8", ("time", "species"))
        wphi[:] = np.array([[2.0], [3.0]])
        heat = diag.createVariable("HeatFlux_st", "f8", ("time", "species"))
        heat[:] = np.array([[0.4], [0.5]])

    out = plot_saved_output(path)
    assert out.exists()
