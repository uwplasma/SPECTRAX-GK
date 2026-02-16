import numpy as np

from spectraxgk.benchmarks import run_etg_linear
from spectraxgk.config import ETGBaseCase, ETGModelConfig, GridConfig
from spectraxgk.plotting import etg_trend_figure


def main() -> None:
    grid = GridConfig(Nx=1, Ny=16, Nz=64, Lx=6.28, Ly=6.28)
    R_vals = np.array([4.0, 6.0, 8.0, 10.0])
    gamma = []
    omega = []
    for R in R_vals:
        cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=float(R)))
        result = run_etg_linear(cfg=cfg, ky_target=3.0, Nl=4, Nm=8, steps=400, dt=0.01, method="rk4")
        gamma.append(result.gamma)
        omega.append(result.omega)
        print(f"R/LTe={R:.1f} gamma={result.gamma:.6f} omega={result.omega:.6f}")
    fig, _axes = etg_trend_figure(R_vals, np.array(gamma), np.array(omega), ky_target=3.0)
    fig.savefig("etg_trend.png", dpi=200)


if __name__ == "__main__":
    main()
