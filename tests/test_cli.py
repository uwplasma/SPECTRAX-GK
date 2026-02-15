import sys

from spectraxgk import __version__
from spectraxgk.cli import main


def test_version_exposed():
    assert __version__ == "0.0.0"


def test_cli_cyclone_info(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-info"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "Cyclone base case" in out


def test_cli_cyclone_kperp(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spectrax-gk", "cyclone-kperp", "--kx0", "0.0", "--ky", "0.3"])
    code = main()
    out = capsys.readouterr().out
    assert code == 0
    assert "k_perp^2" in out
