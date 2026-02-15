"""Extract Cyclone base case reference data from an external NetCDF output."""

from __future__ import annotations

import argparse
from pathlib import Path

from netCDF4 import Dataset


def extract(input_nc: Path, output_csv: Path) -> None:
    with Dataset(input_nc) as ds:
        ky = ds.groups["Grids"].variables["ky"][:]
        omega = ds.groups["Diagnostics"].variables["omega_kxkyt"][:]
        omega_last = omega[-1, :, 0, :]

    lines = ["ky,omega,gamma\n"]
    for k, (w, g) in zip(ky, omega_last):
        lines.append(f"{float(k):.8f},{float(w):.8f},{float(g):.8f}\n")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    extract(args.input_nc, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
