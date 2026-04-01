#!/usr/bin/env python3
"""Run a small set of GX-aligned linear stress cases against local GX outputs.

This is a *developer* gate runner. It is intentionally not part of the default
CI suite because it requires a local GX checkout with benchmark NetCDF files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import pandas as pd


def _default_gx_repo() -> Path | None:
    for candidate in (
        os.environ.get("GX_REPO"),
        str(Path(__file__).resolve().parents[2].parent / "GX"),
        str(Path(__file__).resolve().parents[2].parent / "gx"),
    ):
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if path.exists():
            return path
    return None


def _case_defs(gx_repo: Path) -> dict[str, dict[str, Path]]:
    bench = gx_repo / "benchmarks" / "linear"
    return {
        "kaw": {
            "gx_nc": bench / "KAW" / "kaw_betahat10.0_kp0.01_correct.out.nc",
            "gx_in": bench / "KAW" / "kaw_betahat10.0_kp0.01.in",
        },
        "cyclone_ke": {
            "gx_nc": bench / "ITG_cyclone" / "itg_miller_kinetic_electrons_correct.out.nc",
            "gx_in": bench / "ITG_cyclone" / "itg_miller_kinetic_electrons.in",
        },
        "kbm_miller": {
            "gx_nc": bench / "KBM" / "kbm_miller_correct.out.nc",
            "gx_in": bench / "KBM" / "kbm_miller.in",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx-repo", type=Path, default=None, help="Path to a GX checkout (defaults to $GX_REPO or common locations).")
    p.add_argument("--outdir", type=Path, default=Path("tools_out") / "stress_matrix_linear", help="Output directory for CSV artifacts.")
    p.add_argument(
        "--cases",
        nargs="*",
        default=["kaw", "cyclone_ke", "kbm_miller"],
        help="Subset of cases to run (kaw, cyclone_ke, kbm_miller).",
    )
    p.add_argument("--Nl", type=int, default=8)
    p.add_argument("--Nm", type=int, default=16)
    return p


def _run_case(*, name: str, gx_nc: Path, gx_in: Path, out_csv: Path, Nl: int, Nm: int) -> pd.DataFrame:
    if not gx_nc.exists():
        raise FileNotFoundError(f"Missing GX benchmark output: {gx_nc}")
    if not gx_in.exists():
        raise FileNotFoundError(f"Missing GX benchmark input: {gx_in}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python",
        str(Path(__file__).resolve().parent / "compare_gx_imported_linear.py"),
        "--gx",
        str(gx_nc),
        "--geometry-file",
        str(gx_nc),
        "--gx-input",
        str(gx_in),
        "--Nl",
        str(int(Nl)),
        "--Nm",
        str(int(Nm)),
        "--out",
        str(out_csv),
    ]
    subprocess.run(cmd, check=True)
    df = pd.read_csv(out_csv)
    df.insert(0, "case", name)
    return df


def main() -> None:
    args = build_parser().parse_args()
    gx_repo = args.gx_repo
    if gx_repo is None:
        gx_repo = _default_gx_repo()
    if gx_repo is None:
        raise SystemExit("Could not find a GX repo. Pass --gx-repo or set GX_REPO.")
    gx_repo = gx_repo.expanduser().resolve()

    defs = _case_defs(gx_repo)
    cases = [str(c) for c in args.cases]
    unknown = [c for c in cases if c not in defs]
    if unknown:
        raise SystemExit(f"Unknown cases: {unknown}. Valid: {sorted(defs)}")

    outdir = Path(args.outdir).expanduser().resolve()
    rows: list[pd.DataFrame] = []
    for name in cases:
        out_csv = outdir / f"{name}.csv"
        df = _run_case(
            name=name,
            gx_nc=defs[name]["gx_nc"],
            gx_in=defs[name]["gx_in"],
            out_csv=out_csv,
            Nl=int(args.Nl),
            Nm=int(args.Nm),
        )
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined_path = outdir / "combined.csv"
    combined.to_csv(combined_path, index=False)
    print(combined.to_string(index=False))
    print(f"saved {combined_path}")


if __name__ == "__main__":
    main()
