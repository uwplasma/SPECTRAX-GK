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
    mtm_trend_figure,
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


def test_mtm_trend_figure(tmp_path):
    """MTM trend plot should render and save."""
    nu = np.array([0.0, 0.1, 0.2])
    gamma = np.array([0.05, 0.08, 0.1])
    omega = np.array([-0.2, -0.25, -0.3])
    fig, _axes = mtm_trend_figure(nu, gamma, omega, ky_target=3.0)
    out = tmp_path / "mtm_trend.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists()
