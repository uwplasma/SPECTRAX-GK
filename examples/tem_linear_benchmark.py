import argparse
import numpy as np

from spectraxgk.benchmarks import load_tem_reference, run_tem_scan
from spectraxgk.config import TimeConfig
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="TEM linear scan example.")
    parser.add_argument("--diffrax", action="store_true", help="Use diffrax integrator.")
    parser.add_argument("--solver", default="Tsit5", help="Diffrax solver name.")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive step sizes.")
    args = parser.parse_args()

    ref = load_tem_reference()
    ky_vals = ref.ky
    dt = 0.01
    steps = 800
    time_cfg = TimeConfig(
        t_max=dt * steps,
        dt=dt,
        method="rk4",
        use_diffrax=args.diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=args.adaptive,
    )
    scan = run_tem_scan(
        ky_vals,
        Nl=12,
        Nm=32,
        steps=steps,
        dt=dt,
        method="rk4",
        time_cfg=time_cfg if args.diffrax else None,
    )
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        x_label=r"$k_y \rho_s$",
        title="TEM (s-alpha)",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig("tem_scan.png", dpi=200)


if __name__ == "__main__":
    main()
