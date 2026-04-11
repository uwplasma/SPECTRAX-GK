from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from examples.theory_and_demos.autodiff_inverse_growth import run_demo  # noqa: E402


def test_autodiff_demo_summary(tmp_path: Path):
    summary = run_demo(
        outdir=tmp_path,
        steps=24,
        dt=0.05,
        ky_index=1,
        kx_index=0,
        z_index=0,
        tprim_true=2.2,
        tprim_init=1.8,
        gd_steps=4,
        gd_lr=0.5,
        plot=False,
        write_files=False,
    )
    assert summary["grad_rel_error"] < 0.5
    assert summary["loss_final"] >= 0.0
