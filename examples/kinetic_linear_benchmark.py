import numpy as np

from spectraxgk.benchmarks import load_cyclone_reference_kinetic, run_kinetic_scan
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    ref = load_cyclone_reference_kinetic()
    ky_vals = ref.ky
    scan = run_kinetic_scan(ky_vals, Nl=12, Nm=32, steps=800, dt=0.01, method="rk4")
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
