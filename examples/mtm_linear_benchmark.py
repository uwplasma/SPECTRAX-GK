import numpy as np

from spectraxgk.benchmarks import run_mtm_linear
from spectraxgk.config import GridConfig, MTMBaseCase, MTMModelConfig
from spectraxgk.plotting import mtm_trend_figure


def main() -> None:
    grid = GridConfig(Nx=1, Ny=16, Nz=64, Lx=6.28, Ly=6.28)
    nu_vals = np.array([0.0, 0.1, 0.2, 0.3])
    gamma = []
    omega = []
    for nu in nu_vals:
        cfg = MTMBaseCase(grid=grid, model=MTMModelConfig(R_over_LTe=6.0, nu=float(nu)))
        result = run_mtm_linear(cfg=cfg, ky_target=3.0, Nl=4, Nm=8, steps=400, dt=0.01, method="rk4")
        gamma.append(result.gamma)
        omega.append(result.omega)
        print(f"nu={nu:.2f} gamma={result.gamma:.6f} omega={result.omega:.6f}")
    fig, _axes = mtm_trend_figure(nu_vals, np.array(gamma), np.array(omega), ky_target=3.0)
    fig.savefig("mtm_trend.png", dpi=200)


if __name__ == "__main__":
    main()
