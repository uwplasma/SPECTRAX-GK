"""I/O helper tests for TOML loaders."""

from __future__ import annotations

from pathlib import Path

from spectraxgk.io import load_case_from_toml


def test_load_case_from_toml_gx_parity_flag(tmp_path: Path) -> None:
    toml = """
case = "cyclone"
gx_parity = true
"""
    path = tmp_path / "case.toml"
    path.write_text(toml, encoding="utf-8")
    case_name, cfg, _data = load_case_from_toml(path)
    assert case_name == "cyclone"
    assert getattr(cfg, "gx_parity", False) is True


def test_load_case_from_toml_gx_parity_table(tmp_path: Path) -> None:
    toml = """
case = "cyclone"

[gx_parity]
enabled = false
"""
    path = tmp_path / "case.toml"
    path.write_text(toml, encoding="utf-8")
    case_name, cfg, _data = load_case_from_toml(path)
    assert case_name == "cyclone"
    assert getattr(cfg, "gx_parity", True) is False
