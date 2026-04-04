import io
from contextlib import redirect_stdout

from spectraxgk.utils.callbacks import _emit_progress, progress_update_stride, should_emit_progress


def test_emit_progress_reports_one_based_step_once() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_progress(4, 5, 1.0, -2.0, 3.0, 4.0)
    out = buf.getvalue().strip()
    assert "step=5/5" in out
    assert "progress=100.0%" in out
    assert "step=6/5" not in out


def test_progress_update_stride_caps_long_runs() -> None:
    assert progress_update_stride(5) == 1
    assert progress_update_stride(50) == 1
    assert progress_update_stride(51) == 2
    assert progress_update_stride(500) == 10


def test_should_emit_progress_reports_first_interval_and_last() -> None:
    assert bool(should_emit_progress(0, 200)) is True
    assert bool(should_emit_progress(3, 200)) is True
    assert bool(should_emit_progress(4, 200)) is False
    assert bool(should_emit_progress(199, 200)) is True
