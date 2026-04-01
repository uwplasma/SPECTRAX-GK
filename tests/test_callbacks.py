import io
from contextlib import redirect_stdout

from spectraxgk.utils.callbacks import _emit_progress


def test_emit_progress_reports_one_based_step_once() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_progress(4, 5, 1.0, -2.0, 3.0, 4.0)
    out = buf.getvalue().strip()
    assert "step=5/5" in out
    assert "progress=100.0%" in out
    assert "step=6/5" not in out
