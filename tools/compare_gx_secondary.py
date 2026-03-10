#!/usr/bin/env python3
"""Compare the GX secondary-instability slab benchmark against SPECTRAX-GK."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.io import load_runtime_from_toml
from spectraxgk.secondary import build_secondary_stage2_config, run_secondary_modes, run_secondary_seed


DEFAULT_MODES = (
    (0.0, -0.05),
    (0.0, 0.0),
    (0.0, 0.05),
    (0.1, -0.05),
    (0.1, 0.0),
    (0.1, 0.05),
)

README_TARGET_PATH = (
    Path(__file__).resolve().parents[2] / "gx" / "benchmarks" / "nonlinear" / "secondary" / "README.md"
)
README_TARGET_ZERO_MODES = {
    (0.0, 0.0): {"gamma_gx": 0.0, "omega_gx": 0.0},
    (0.1, 0.0): {"gamma_gx": 0.0, "omega_gx": 0.0},
}


def _select_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def _load_gx_modes(path: Path, modes: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    root = Dataset(path, "r")
    grids = root.groups["Grids"]
    diag = root.groups["Diagnostics"]
    ky = np.asarray(grids.variables["ky"][:], dtype=float)
    kx = np.asarray(grids.variables["kx"][:], dtype=float)
    omega_kxkyt = np.asarray(diag.variables["omega_kxkyt"][:], dtype=float)
    root.close()
    if omega_kxkyt.ndim != 4 or omega_kxkyt.shape[-1] != 2:
        raise ValueError(f"unexpected GX omega_kxkyt shape {omega_kxkyt.shape}")
    last = omega_kxkyt[-1]
    rows: list[dict[str, float]] = []
    for ky_target, kx_target in modes:
        ky_i = _select_index(ky, ky_target)
        kx_i = _select_index(kx, kx_target)
        rows.append(
            {
                "ky": float(ky_target),
                "kx": float(kx_target),
                "gamma_gx": float(last[ky_i, kx_i, 1]),
                "omega_gx": float(last[ky_i, kx_i, 0]),
            }
        )
    return pd.DataFrame(rows)


def _load_gx_readme_targets(path: Path, modes: tuple[tuple[float, float], ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"secondary README target file not found: {path}")
    text = path.read_text(encoding="utf-8")
    targets: dict[tuple[float, float], dict[str, float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) not in (2, 4):
            continue
        try:
            floats = [float(part) for part in parts]
        except ValueError:
            continue
        if len(parts) == 4:
            ky, kx, omega, gamma = floats
            targets[(ky, kx)] = {"gamma_gx": gamma, "omega_gx": omega}
        elif len(parts) == 2:
            ky, kx = floats
            if (ky, kx) in README_TARGET_ZERO_MODES:
                targets[(ky, kx)] = dict(README_TARGET_ZERO_MODES[(ky, kx)])
    rows: list[dict[str, float]] = []
    for ky_target, kx_target in modes:
        key = (float(ky_target), float(kx_target))
        if key not in targets:
            raise ValueError(f"secondary README target missing mode {key}")
        rows.append(
            {
                "ky": float(ky_target),
                "kx": float(kx_target),
                "gamma_gx": float(targets[key]["gamma_gx"]),
                "omega_gx": float(targets[key]["omega_gx"]),
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/configs/runtime_secondary_slab.toml"),
        help="Stage-1 secondary runtime config.",
    )
    parser.add_argument("--gx-out", type=Path, default=None, help="GX kh01a out.nc file.")
    parser.add_argument(
        "--gx-readme",
        type=Path,
        default=README_TARGET_PATH,
        help="GX secondary README containing the published target table.",
    )
    parser.add_argument(
        "--gx-source",
        choices=("out-nc", "readme"),
        default="readme",
        help="Use a real GX out.nc file or the published README target table.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--Nl", type=int, default=3)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument("--seed-ky", type=float, default=0.1)
    parser.add_argument("--stage1-dt", type=float, default=1.0)
    parser.add_argument("--stage1-steps", type=int, default=2)
    parser.add_argument("--stage2-dt", type=float, default=0.01)
    parser.add_argument("--stage2-tmax", type=float, default=2.0)
    parser.add_argument("--restart-scale", type=float, default=500.0)
    parser.add_argument("--init-amp", type=float, default=1.0e-5)
    parser.add_argument("--sample-stride", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg, _data = load_runtime_from_toml(args.config)
    modes = tuple(DEFAULT_MODES)

    with TemporaryDirectory(prefix="spectrax_secondary_compare_") as tmpdir:
        restart_path = Path(tmpdir) / "secondary_seed.bin"
        run_secondary_seed(
            cfg,
            restart_path=restart_path,
            ky_target=float(args.seed_ky),
            Nl=int(args.Nl),
            Nm=int(args.Nm),
            dt=float(args.stage1_dt),
            steps=int(args.stage1_steps),
        )
        stage2_cfg = build_secondary_stage2_config(
            cfg,
            restart_file=restart_path,
            restart_scale=float(args.restart_scale),
            init_amp=float(args.init_amp),
            dt=float(args.stage2_dt),
            t_max=float(args.stage2_tmax),
        )
        sp_rows = run_secondary_modes(
            stage2_cfg,
            modes=modes,
            Nl=int(args.Nl),
            Nm=int(args.Nm),
            sample_stride=int(args.sample_stride),
        )

    if args.gx_source == "out-nc":
        if args.gx_out is None:
            raise ValueError("--gx-out is required when --gx-source=out-nc")
        gx_df = _load_gx_modes(args.gx_out, modes)
    else:
        gx_df = _load_gx_readme_targets(args.gx_readme, modes)
    sp_df = pd.DataFrame([row.__dict__ for row in sp_rows]).rename(
        columns={"gamma": "gamma_sp", "omega": "omega_sp"}
    )
    table = gx_df.merge(sp_df, on=["ky", "kx"], how="inner")
    table["gx_source"] = args.gx_source
    table["rel_gamma"] = np.abs(table["gamma_sp"] - table["gamma_gx"]) / np.maximum(
        np.abs(table["gamma_gx"]), 1.0e-12
    )
    table["rel_omega"] = np.abs(table["omega_sp"] - table["omega_gx"]) / np.maximum(
        np.abs(table["omega_gx"]), 1.0e-12
    )

    print("ky      kx      gamma_gx    gamma_sp    rel_gamma    omega_gx    omega_sp    rel_omega")
    for row in table.itertuples(index=False):
        print(
            f"{row.ky:0.4f} {row.kx:0.4f} "
            f"{row.gamma_gx: .6e} {row.gamma_sp: .6e} {row.rel_gamma: .3e} "
            f"{row.omega_gx: .6e} {row.omega_sp: .6e} {row.rel_omega: .3e}"
        )

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
