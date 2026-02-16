import numpy as np

from spectraxgk.benchmarks import load_tem_reference, run_tem_scan
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    ref = load_tem_reference()
    ky_vals = ref.ky
    scan = run_tem_scan(ky_vals, Nl=12, Nm=32, steps=800, dt=0.01, method="rk4")
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
