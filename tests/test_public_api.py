import spectraxgk


def test_standalone_public_aliases_match_current_exports() -> None:
    assert spectraxgk.LinearExplicitTimeConfig is spectraxgk.ExplicitTimeConfig
    assert (
        spectraxgk.integrate_nonlinear_diagnostics
        is spectraxgk.integrate_nonlinear_explicit_diagnostics
    )
