"""Generate CSV tables for documentation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    ky_subset = np.array([0.3, 0.4])
    cfg = CycloneBaseCase(grid=GridConfig(Nx=8, Ny=12, Nz=24, Lx=62.8, Ly=62.8))
    scan = run_cyclone_scan(ky_subset, cfg=cfg, steps=300, dt=0.02, tmin=3.0, method="rk4")

    rows = [
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega"
    ]
    for ky, gamma, omega in zip(scan.ky, scan.gamma, scan.omega):
        idx = int(np.argmin(np.abs(ref.ky - ky)))
        gamma_ref = float(ref.gamma[idx])
        omega_ref = float(ref.omega[idx])
        rel_gamma = (gamma - gamma_ref) / gamma_ref if gamma_ref != 0.0 else np.nan
        rel_omega = (omega - omega_ref) / omega_ref if omega_ref != 0.0 else np.nan
        rows.append(
            f"{ky:.3f},{gamma_ref:.6f},{omega_ref:.6f},{gamma:.6f},{omega:.6f},{rel_gamma:.3f},{rel_omega:.3f}"
        )

    table_path = outdir / "cyclone_scan_table.csv"
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
