#!/usr/bin/env python3
"""Select the next unstable external-VMEC holdout candidate from a screen CSV."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.write_external_vmec_holdout_configs import (  # noqa: E402
    DEFAULT_HORIZONS,
    DEFAULT_GRIDS,
    _parse_grid,
    _parse_horizons,
    write_configs,
    write_manifest,
)

DEFAULT_VMEC_SEARCH_ROOTS = (
    Path("/Users/rogeriojorge/local/vmec_jax/examples/data"),
    Path("/Users/rogeriojorge/local/vmec_jax/examples_single_grid/data"),
    Path("/Users/rogeriojorge/vmec_jax/examples/data"),
    Path("/Users/rogeriojorge/vmec_jax/examples_single_grid/data"),
    Path("/Users/rogeriojorge/src/vmec_jax/examples/data"),
    Path("/Users/rogeriojorge/src/vmec_jax/examples_single_grid/data"),
)


@dataclass(frozen=True)
class CandidateRow:
    case: str
    vmec_file: str
    returncode: int
    best_ky: float | None
    best_gamma: float | None
    best_omega: float | None
    log: str


def _maybe_float(raw: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    return float(text)


def _read_screen(path: Path) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                CandidateRow(
                    case=str(row["case"]).strip(),
                    vmec_file=str(row["vmec_file"]).strip(),
                    returncode=int(str(row["returncode"]).strip()),
                    best_ky=_maybe_float(str(row.get("best_ky", ""))),
                    best_gamma=_maybe_float(str(row.get("best_gamma", ""))),
                    best_omega=_maybe_float(str(row.get("best_omega", ""))),
                    log=str(row.get("log", "")).strip(),
                )
            )
    return rows


def _candidate_sort_key(row: CandidateRow) -> tuple[float, float]:
    gamma = float("-inf") if row.best_gamma is None else float(row.best_gamma)
    ky = float("-inf") if row.best_ky is None else float(row.best_ky)
    return (gamma, ky)


def select_candidate(rows: Iterable[CandidateRow], *, excluded_cases: set[str]) -> CandidateRow:
    eligible = [
        row
        for row in rows
        if row.case not in excluded_cases
        and row.returncode == 0
        and row.best_gamma is not None
        and row.best_gamma > 0.0
        and row.best_ky is not None
    ]
    if not eligible:
        raise ValueError("no finite unstable candidate remains after exclusions")
    return max(eligible, key=_candidate_sort_key)


def resolve_vmec_file(raw: str, *, search_roots: Iterable[Path] = DEFAULT_VMEC_SEARCH_ROOTS) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    basename = path.name
    for root in search_roots:
        candidate = root / basename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve local VMEC file for {raw!r}")


def _default_case_slug(case: str) -> str:
    slug = case
    if slug.endswith("_nc"):
        slug = slug[: -len("_nc")]
    return slug


def write_selection_summary(
    out_dir: Path,
    *,
    selected: CandidateRow,
    resolved_vmec_file: Path,
    excluded_cases: set[str],
    generated_paths: list[Path],
) -> Path:
    payload = {
        "kind": "external_vmec_holdout_candidate_selection",
        "selected_case": selected.case,
        "selected_vmec_file_source": selected.vmec_file,
        "selected_vmec_file_resolved": resolved_vmec_file.as_posix(),
        "best_ky": selected.best_ky,
        "best_gamma": selected.best_gamma,
        "best_omega": selected.best_omega,
        "log": selected.log,
        "excluded_cases": sorted(excluded_cases),
        "generated_configs": [path.as_posix() for path in generated_paths],
    }
    path = out_dir / "selection_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        type=Path,
        default=ROOT / "docs" / "_static" / "external_vmec_candidate_linear_screen.csv",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--exclude-case", action="append", default=[])
    parser.add_argument("--case-slug", default=None, help="Override output slug for the selected case")
    parser.add_argument("--ky", type=float, default=None, help="Override ky instead of using the screen winner")
    parser.add_argument("--horizons", default=",".join(str(v).rstrip("0").rstrip(".") for v in DEFAULT_HORIZONS))
    parser.add_argument("--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = _read_screen(Path(args.screen))
    excluded = {str(item) for item in args.exclude_case}
    selected = select_candidate(rows, excluded_cases=excluded)
    vmec_file = resolve_vmec_file(selected.vmec_file)
    case_slug = str(args.case_slug) if args.case_slug else _default_case_slug(selected.case)
    horizons = _parse_horizons(str(args.horizons))
    grids = tuple(_parse_grid(raw) for raw in (args.grid or DEFAULT_GRIDS))
    ky = float(args.ky) if args.ky is not None else float(selected.best_ky)
    written = write_configs(
        case=case_slug,
        vmec_file=vmec_file,
        out_dir=Path(args.out_dir),
        grids=grids,
        horizons=horizons,
        ky=ky,
    )
    manifest = write_manifest(Path(args.out_dir), written)
    summary = write_selection_summary(
        Path(args.out_dir),
        selected=selected,
        resolved_vmec_file=vmec_file,
        excluded_cases=excluded,
        generated_paths=[item.path for item in written],
    )
    print(f"selected {selected.case} ky={ky:.6g} gamma={float(selected.best_gamma):.6g}")
    print(f"resolved {vmec_file}")
    print(f"wrote {manifest}")
    print(f"wrote {summary}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
