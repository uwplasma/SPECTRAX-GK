import numpy as np

from spectraxgk.benchmarks import load_kbm_reference, run_kbm_beta_scan
from spectraxgk.plotting import scan_comparison_figure


def main() -> None:
    ref = load_kbm_reference()
    beta_vals = ref.ky
    scan = run_kbm_beta_scan(beta_vals, ky_target=0.3, Nl=12, Nm=32, steps=600, dt=0.01)
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
