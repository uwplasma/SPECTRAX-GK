import argparse
import numpy as np

from spectraxgk.benchmarks import (
    compare_cyclone_to_reference,
    load_cyclone_reference,
    run_cyclone_linear,
    run_cyclone_scan,
)
from spectraxgk.config import TimeConfig
from spectraxgk.plotting import cyclone_comparison_figure, cyclone_reference_figure


def main():
    parser = argparse.ArgumentParser(description="Cyclone base case example.")
    parser.add_argument("--no-diffrax", action="store_true", help="Disable diffrax integrator.")
    parser.add_argument("--solver", default="Heun", help="Diffrax solver name.")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive step sizes.")
    args = parser.parse_args()

    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig("cyclone_reference.png", dpi=200)

    ky_target = 0.3
    dt = 0.01
    steps = 800
    time_cfg = TimeConfig(
        t_max=dt * steps,
        dt=dt,
        method="rk2",
        use_diffrax=not args.no_diffrax,
        diffrax_solver=args.solver,
        diffrax_adaptive=args.adaptive,
    )
    result = run_cyclone_linear(
        ky_target=ky_target,
        Nl=6,
        Nm=12,
        time_cfg=time_cfg,
    )
    comparison = compare_cyclone_to_reference(result, ref)
    print(
        "Cyclone reference ky="
        f"{comparison.ky:.3f} gamma_ref={comparison.gamma_ref:.6f} omega_ref={comparison.omega_ref:.6f}"
    )
    print(f"SPECTRAX-GK gamma={comparison.gamma:.6f} omega={comparison.omega:.6f}")
    print(f"Relative error gamma={comparison.rel_gamma:.2%} omega={comparison.rel_omega:.2%}")

    ky_sample = ref.ky[::2]
    scan = run_cyclone_scan(ky_sample, Nl=6, Nm=12, time_cfg=time_cfg)
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig("cyclone_comparison.png", dpi=200)


if __name__ == "__main__":
    main()
