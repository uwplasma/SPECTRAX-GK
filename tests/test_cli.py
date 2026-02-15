"""CLI tests for basic command execution."""

import sys

from spectraxgk import __version__
from spectraxgk.cli import main


def test_version_exposed():
    """Version string should be exported from the package."""
    assert __version__ == "0.0.0"


def test_cli_cyclone_info(capsys, monkeypatch):
    """The cyclone-info command should print default parameters."""
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-info"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Cyclone base case" in out


def test_cli_cyclone_kperp(capsys, monkeypatch):
    """The cyclone-kperp command should print k_perp^2 ranges."""
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-kperp", "--kx0", "0.0", "--ky", "0.3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "k_perp^2" in out
