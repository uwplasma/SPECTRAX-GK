import argparse
import numpy as np

from spectraxgk.benchmarks import load_kbm_reference, run_kbm_beta_scan
from spectraxgk.config import TimeConfig
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="KBM beta scan example.")
    parser.add_argument("--diffrax", action="store_true", help="Use diffrax integrator.")
    parser.add_argument("--solver", default="Tsit5", help="Diffrax solver name.")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive step sizes.")
    args = parser.parse_args()

    ref = load_kbm_reference()
    beta_vals = ref.ky
    dt = 0.01
    steps = 600
    time_cfg = TimeConfig(
        t_max=dt * steps,
        dt=dt,
        method="rk4",
        use_diffrax=args.diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=args.adaptive,
    )
    scan = run_kbm_beta_scan(
        beta_vals,
        ky_target=0.3,
        Nl=12,
        Nm=32,
        steps=steps,
        dt=dt,
        time_cfg=time_cfg if args.diffrax else None,
    )
    fig, _axes = scan_comparison_figure(
        scan.ky,
        scan.gamma,
        scan.omega,
        x_label=r"$\beta_{ref}$",
        title="KBM beta scan",
        x_ref=ref.ky,
        gamma_ref=ref.gamma,
        omega_ref=ref.omega,
        ref_label="Reference",
    )
    fig.savefig("kbm_beta_scan.png", dpi=200)


if __name__ == "__main__":
    main()
