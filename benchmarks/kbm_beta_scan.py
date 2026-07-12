import argparse
from dataclasses import replace
from pathlib import Path

from spectraxgk.benchmarks import load_kbm_reference
from spectraxgk.artifacts.plotting import scan_comparison_figure
from spectraxgk.runtime import run_runtime_parameter_scan
from spectraxgk.solvers.linear.krylov import KrylovConfig
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "examples" / "linear" / "axisymmetric" / "runtime_kbm.toml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KBM beta scan example.")
    parser.add_argument("--transition", type=float, default=0.3)
    args = parser.parse_args()

    ref = load_kbm_reference()
    beta_vals = ref.ky
    config, _raw = load_runtime_from_toml(CONFIG)
    base_solver = KrylovConfig(
        method="shift_invert", krylov_dim=16, restarts=1,
        omega_sign=-1, omega_cap_factor=2.0,
        shift_source="target", shift_maxiter=40, shift_restart=12,
        shift_tol=5.0e-4, shift_preconditioner="hermite-line",
        shift_selection="targeted", mode_family="kbm",
    )
    scan = run_runtime_parameter_scan(
        config, beta_vals, parameter_name="beta",
        update_config=lambda cfg, beta, _index: replace(
            cfg, physics=replace(cfg.physics, beta=beta)
        ),
        ky_target=0.3,
        linear_options={"Nl": 12, "Nm": 32, "solver": "krylov"},
        candidate_options=lambda _beta, _index, _previous: tuple(
            {"krylov_cfg": replace(base_solver, omega_target_factor=target)}
            for target in (0.7, 1.5)
        ),
        select_candidate=lambda beta, _index, _candidates, _previous: (
            1 if beta >= args.transition else 0
        ),
        continuation=True,
    )
    fig, _axes = scan_comparison_figure(
        scan.values,
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
