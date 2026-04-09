#!/usr/bin/env python3
"""Convert imported-linear scan diagnostics into the legacy last-value mismatch table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SCAN_COLUMNS = {
    "ky",
    "gamma_last",
    "omega_last",
    "gamma_ref_last",
    "omega_ref_last",
}


def _load_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_SCAN_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    return df.copy()


def _build_lastvalue_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ky": df["ky"].astype(float),
            "gamma": df["gamma_last"].astype(float),
            "omega": df["omega_last"].astype(float),
            "gamma_gx": df["gamma_ref_last"].astype(float),
            "omega_gx": df["omega_ref_last"].astype(float),
        }
    )
    out["rel_gamma"] = (out["gamma"] - out["gamma_gx"]) / out["gamma_gx"].where(out["gamma_gx"] != 0.0)
    out["rel_omega"] = (out["omega"] - out["omega_gx"]) / out["omega_gx"].where(out["omega_gx"] != 0.0)
    return out.sort_values("ky").reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path, required=True, help="Imported-linear scan CSV.")
    parser.add_argument("--out", type=Path, required=True, help="Legacy last-value mismatch CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = _build_lastvalue_table(_load_scan(args.scan.expanduser().resolve()))
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
