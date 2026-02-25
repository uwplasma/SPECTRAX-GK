import numpy as np

from spectraxgk.analysis import ModeSelection
from spectraxgk.benchmarks import _select_fit_signal_auto


def test_auto_fit_signal_prefers_higher_gamma() -> None:
    t = np.linspace(0.0, 1.0, 200)
    gamma_phi = 0.1
    gamma_den = 0.3
    omega = 1.2
    phi_signal = np.exp((gamma_phi - 1j * omega) * t)
    dens_signal = np.exp((gamma_den - 1j * omega) * t)

    phi_t = phi_signal[:, None, None, None]
    density_t = dens_signal[:, None, None, None]
    sel = ModeSelection(ky_index=0, kx_index=0, z_index=0)

    _signal, name, gamma, omega_fit = _select_fit_signal_auto(
        t,
        phi_t,
        density_t,
        sel,
        mode_method="z_index",
        tmin=None,
        tmax=None,
        window_fraction=0.4,
        min_points=40,
        start_fraction=0.0,
        growth_weight=1.0,
        require_positive=True,
        min_amp_fraction=0.0,
        max_amp_fraction=0.9,
        window_method="loglinear",
        max_fraction=0.8,
        end_fraction=0.9,
        num_windows=6,
        phase_weight=0.2,
        length_weight=0.05,
        min_r2=0.0,
        late_penalty=0.1,
        min_slope=None,
        min_slope_frac=0.0,
        slope_var_weight=0.0,
    )

    assert name == "density"
    assert gamma > gamma_phi
    assert np.isfinite(omega_fit)
