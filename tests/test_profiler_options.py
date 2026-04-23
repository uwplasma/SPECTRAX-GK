from tools._profiler_options import make_profile_options


def test_make_profile_options_defaults_disable_python_and_host_tracers() -> None:
    opts = make_profile_options()
    assert opts.python_tracer_level == 0
    assert opts.host_tracer_level == 0


def test_make_profile_options_accepts_explicit_levels() -> None:
    opts = make_profile_options(python_tracer_level=1, host_tracer_level=2)
    assert opts.python_tracer_level == 1
    assert opts.host_tracer_level == 2
