import spectraxgk


def test_standalone_public_aliases_match_compatibility_exports() -> None:
    assert spectraxgk.LinearExplicitTimeConfig is spectraxgk.ExplicitTimeConfig
    assert spectraxgk.GXTimeConfig is spectraxgk.ExplicitTimeConfig
    assert spectraxgk.GXDiagnostics is spectraxgk.SimulationDiagnostics
    assert spectraxgk.integrate_linear_explicit is spectraxgk.integrate_linear_gx
    assert (
        spectraxgk.integrate_linear_explicit_diagnostics
        is spectraxgk.integrate_linear_gx_diagnostics
    )
    assert spectraxgk.integrate_nonlinear_diagnostics is spectraxgk.integrate_nonlinear_gx_diagnostics
    assert spectraxgk.growth_rate_from_phi is spectraxgk.gx_growth_rate_from_phi
