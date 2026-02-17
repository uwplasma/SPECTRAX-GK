import argparse
import numpy as np

from spectraxgk.benchmarks import run_etg_linear
from spectraxgk.config import ETGBaseCase, ETGModelConfig, GridConfig, TimeConfig
from spectraxgk.plotting import etg_trend_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="ETG linear trend example.")
    parser.add_argument("--diffrax", action="store_true", help="Use diffrax integrator.")
    parser.add_argument("--solver", default="Tsit5", help="Diffrax solver name.")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive step sizes.")
    args = parser.parse_args()

    grid = GridConfig(Nx=1, Ny=12, Nz=32, Lx=6.28, Ly=6.28)
    dt = 0.01
    steps = 200
    time_cfg = TimeConfig(
        t_max=dt * steps,
        dt=dt,
        method="rk4",
        use_diffrax=args.diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=args.adaptive,
    )
    R_vals = np.array([4.0, 6.0, 8.0, 10.0])
    gamma = []
    omega = []
    for R in R_vals:
        cfg = ETGBaseCase(grid=grid, model=ETGModelConfig(R_over_LTe=float(R)), time=time_cfg)
        result = run_etg_linear(
            cfg=cfg,
            ky_target=3.0,
            Nl=4,
            Nm=8,
            steps=steps,
            dt=dt,
            method="rk4",
            time_cfg=time_cfg if args.diffrax else None,
            auto_window=True,
            mode_method="z_index",
        )
        gamma.append(result.gamma)
        omega.append(result.omega)
        print(f"R/LTe={R:.1f} gamma={result.gamma:.6f} omega={result.omega:.6f}")
    fig, _axes = etg_trend_figure(R_vals, np.array(gamma), np.array(omega), ky_target=3.0)
    fig.savefig("etg_trend.png", dpi=200)


if __name__ == "__main__":
    main()
