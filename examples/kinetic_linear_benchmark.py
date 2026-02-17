import argparse
import numpy as np

from spectraxgk.benchmarks import load_cyclone_reference_kinetic, run_kinetic_scan
from spectraxgk.config import TimeConfig
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Kinetic-electron ITG benchmark example.")
    parser.add_argument("--no-diffrax", action="store_true", help="Disable diffrax integrator.")
    parser.add_argument("--solver", default="Tsit5", help="Diffrax solver name.")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive step sizes.")
    args = parser.parse_args()

    ref = load_cyclone_reference_kinetic()
    ky_vals = ref.ky
    dt = 0.01
    steps = 800
    time_cfg = TimeConfig(
        t_max=dt * steps,
        dt=dt,
        method="rk2",
        use_diffrax=not args.no_diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=not args.no_adaptive,
    )
    scan = run_kinetic_scan(ky_vals, Nl=12, Nm=32, time_cfg=time_cfg)
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        x_label=r"$k_y \rho_i$",
        title="Kinetic-electron ITG",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig("kinetic_scan.png", dpi=200)


if __name__ == "__main__":
    main()
