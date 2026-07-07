import io
from contextlib import redirect_stdout

from spectraxgk.utils.callbacks import (
    _PROGRESS_START,
    _emit_progress,
    _format_duration,
    print_callback,
    progress_update_stride,
    should_emit_progress,
)


def test_emit_progress_reports_one_based_step_once() -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit_progress(4, 5, 1.0, -2.0, 3.0, 4.0, sim_time=2.0, sim_total=2.5)
    out = buf.getvalue().strip()
    assert "step=5/5" in out
    assert "progress=100.0%" in out
    assert "t=2/2.5" in out
    assert "elapsed=" in out
    assert "eta=00:00" in out
    assert "step=6/5" not in out


def test_format_duration_clamps_and_rolls_over() -> None:
    assert _format_duration(-2.0) == "00:00"
    assert _format_duration(65.4) == "01:05"
    assert _format_duration(3661.0) == "1:01:01"


def test_progress_update_stride_caps_long_runs() -> None:
    assert progress_update_stride(5) == 1
    assert progress_update_stride(50) == 1
    assert progress_update_stride(51) == 2
    assert progress_update_stride(500) == 10


def test_progress_update_stride_sanitizes_inputs() -> None:
    assert progress_update_stride(0) == 1
    assert progress_update_stride(-10, target_updates=0) == 1
    assert progress_update_stride(9, target_updates=4) == 3


def test_should_emit_progress_reports_first_interval_and_last() -> None:
    assert bool(should_emit_progress(0, 200)) is True
    assert bool(should_emit_progress(3, 200)) is True
    assert bool(should_emit_progress(4, 200)) is False
    assert bool(should_emit_progress(199, 200)) is True


def test_should_emit_progress_sanitizes_steps_and_targets() -> None:
    assert bool(should_emit_progress(0, 0, target_updates=0)) is True
    assert bool(should_emit_progress(1, 9, target_updates=4)) is False
    assert bool(should_emit_progress(2, 9, target_updates=4)) is True


def test_emit_progress_handles_time_variants_and_metric_labels(monkeypatch) -> None:
    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(
        "spectraxgk.utils.callbacks.time.perf_counter", lambda: next(ticks)
    )
    _PROGRESS_START.clear()

    first = io.StringIO()
    with redirect_stdout(first):
        _emit_progress(
            0, 3, 0.1, 0.2, 0.3, 0.4, sim_time=1.25, metric_labels=("A", "B")
        )
    first_out = first.getvalue()
    assert "step=1/3" in first_out
    assert "t=1.25" in first_out
    assert "eta=--:--" in first_out
    assert "A=0.3 B=0.4" in first_out

    second = io.StringIO()
    with redirect_stdout(second):
        _emit_progress(1, 3, 0.1, 0.2, 0.3, 0.4, sim_time=2.0, sim_total=0.0)
    second_out = second.getvalue()
    assert "step=2/3" in second_out
    assert "t=2" in second_out
    assert "/0" not in second_out
    assert "eta=00:01" in second_out


def test_print_callback_returns_state_and_forwards_values(monkeypatch) -> None:
    calls = []

    def fake_callback(fn, *args):
        calls.append(args)
        fn(*args)

    monkeypatch.setattr("spectraxgk.utils.callbacks.jax.debug.callback", fake_callback)

    state = {"unchanged": True}
    buf = io.StringIO()
    with redirect_stdout(buf):
        returned = print_callback(
            state,
            0,
            1,
            1.5,
            -0.5,
            2.0,
            3.0,
            sim_time=None,
            sim_total=None,
            metric_labels=("heat", "free"),
        )

    assert returned is state
    assert calls == [(0, 1, 1.5, -0.5, 2.0, 3.0, None, None)]
    out = buf.getvalue()
    assert "heat=2" in out
    assert "free=3" in out
