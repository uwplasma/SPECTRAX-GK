"""Generate publication-ready figures for docs and README."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarks import load_cyclone_reference, run_cyclone_scan
from spectraxgk.plotting import cyclone_comparison_figure, cyclone_reference_figure


def main() -> int:
    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)

    ref = load_cyclone_reference()
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    ky_sample = ref.ky[::2]
    scan = run_cyclone_scan(ky_sample, Nl=2, Nm=4, steps=400, dt=0.02, method="rk4")
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
