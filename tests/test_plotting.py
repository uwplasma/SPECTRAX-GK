"""Plotting utilities should generate figures without errors."""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from spectraxgk.benchmarks import CycloneReference
import matplotlib.pyplot as plt
from spectraxgk.plotting import cyclone_reference_figure


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
