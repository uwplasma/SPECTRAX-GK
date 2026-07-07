from pathlib import Path
import json

import pytest

from tools.prepare_external_vmec_holdout_from_screen import (
    _default_case_slug,
    _read_screen,
    resolve_vmec_file,
    select_candidate,
    write_selection_summary,
)


def _write_screen(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "case,vmec_file,returncode,best_ky,best_gamma,best_omega,log",
                "DSHAPE_nc,/remote/wout_DSHAPE.nc,0,0.4762,0.09619,0.0625,/tmp/dshape.log",
                "circular_tokamak_nc,/remote/wout_circular_tokamak.nc,0,0.4762,0.089094,0.4078,/tmp/circular.log",
                "ITERModel_reference_nc,/remote/wout_ITERModel_reference.nc,0,0.4762,0.088737,0.4086,/tmp/iter.log",
                "stable_case,/remote/wout_stable.nc,0,0.3,-0.01,0.1,/tmp/stable.log",
                "failed_case,/remote/wout_failed.nc,1,,,,/tmp/failed.log",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_select_candidate_skips_excluded_cases(tmp_path: Path) -> None:
    screen = tmp_path / "screen.csv"
    _write_screen(screen)
    rows = _read_screen(screen)
    selected = select_candidate(rows, excluded_cases={"DSHAPE_nc", "circular_tokamak_nc"})
    assert selected.case == "ITERModel_reference_nc"
    assert selected.best_gamma == pytest.approx(0.088737)
    assert _default_case_slug(selected.case) == "ITERModel_reference"


def test_select_candidate_raises_when_no_unstable_cases_remain(tmp_path: Path) -> None:
    screen = tmp_path / "screen.csv"
    _write_screen(screen)
    rows = _read_screen(screen)
    with pytest.raises(ValueError, match="no finite unstable candidate"):
        select_candidate(rows, excluded_cases={"DSHAPE_nc", "circular_tokamak_nc", "ITERModel_reference_nc"})


def test_resolve_vmec_file_uses_search_roots(tmp_path: Path) -> None:
    root = tmp_path / "vmec"
    root.mkdir()
    target = root / "wout_ITERModel_reference.nc"
    target.write_text("fixture", encoding="utf-8")
    resolved = resolve_vmec_file("/remote/wout_ITERModel_reference.nc", search_roots=[root])
    assert resolved == target


def test_write_selection_summary_serializes_choice(tmp_path: Path) -> None:
    screen = tmp_path / "screen.csv"
    _write_screen(screen)
    rows = _read_screen(screen)
    selected = select_candidate(rows, excluded_cases={"DSHAPE_nc", "circular_tokamak_nc"})
    generated = [tmp_path / "cfg_a.toml", tmp_path / "cfg_b.toml"]
    summary = write_selection_summary(
        tmp_path,
        selected=selected,
        resolved_vmec_file=tmp_path / "wout_ITERModel_reference.nc",
        excluded_cases={"DSHAPE_nc", "circular_tokamak_nc"},
        generated_paths=generated,
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["selected_case"] == "ITERModel_reference_nc"
    assert payload["excluded_cases"] == ["DSHAPE_nc", "circular_tokamak_nc"]
    assert payload["generated_configs"] == [path.as_posix() for path in generated]
