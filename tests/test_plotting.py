"""Plotting utilities should generate figures without errors."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from spectraxgk.benchmarks import CycloneReference, CycloneScanResult
import matplotlib.pyplot as plt
from spectraxgk.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    etg_trend_figure,
    growth_rate_heatmap,
    linear_validation_figure,
    LinearValidationPanel,
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
