"""I/O helper tests for TOML loaders."""

from __future__ import annotations

from pathlib import Path

from spectraxgk.io import load_case_from_toml, load_linear_terms_from_toml


def test_load_case_from_toml_gx_reference_flag(tmp_path: Path) -> None:
    toml = """
case = "cyclone"
gx_reference = true
"""
    path = tmp_path / "case.toml"
    path.write_text(toml, encoding="utf-8")
    case_name, cfg, _data = load_case_from_toml(path)
    assert case_name == "cyclone"
    assert getattr(cfg, "gx_reference", False) is True


def test_load_case_from_toml_gx_reference_table(tmp_path: Path) -> None:
    toml = """
case = "cyclone"

[gx_reference]
enabled = false
"""
    path = tmp_path / "case.toml"
    path.write_text(toml, encoding="utf-8")
    case_name, cfg, _data = load_case_from_toml(path)
    assert case_name == "cyclone"
    assert getattr(cfg, "gx_reference", True) is False


def test_load_linear_terms_ignores_nonlinear_only_keys() -> None:
    terms = load_linear_terms_from_toml(
        {
            "terms": {
                "streaming": 0.0,
                "apar": 0.0,
                "bpar": 0.0,
                "nonlinear": 0.0,
            }
        }
    )

    assert terms is not None
    assert terms.streaming == 0.0
    assert terms.apar == 0.0
    assert terms.bpar == 0.0
