"""Generate a publication-ready eigenfunction-overlap summary figure.

The current tracked use case is the KBM GX comparison candidate table, which
already stores normalized overlap and relative-L2 metrics per ``k_y``. This
script turns that tracked CSV into a reusable manuscript-quality summary panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.plotting import eigenfunction_overlap_summary_figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot eigenfunction-overlap summary from a tracked CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "docs" / "_static" / "kbm_gx_candidates.csv",
        help="Input CSV with ky/eig_overlap_gx/eig_rel_l2 columns.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "_static" / "kbm_eigenfunction_overlap_summary.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        default="KBM eigenfunction overlap against GX",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = pd.read_csv(args.csv)
    if "selected" in df.columns:
        df = df[df["selected"].astype(bool)]
    needed = {"ky", "eig_overlap_gx", "eig_rel_l2"}
    missing = needed.difference(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")
    df = df[np.isfinite(df["eig_overlap_gx"]) & np.isfinite(df["eig_rel_l2"])]
    df = df.sort_values("ky")
    fig, _axes = eigenfunction_overlap_summary_figure(
        np.asarray(df["ky"], dtype=float),
        np.asarray(df["eig_overlap_gx"], dtype=float),
        np.asarray(df["eig_rel_l2"], dtype=float),
        title=str(args.title),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
