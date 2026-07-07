#!/usr/bin/env python3
"""Write a reproducible W7-X zonal-closure sweep manifest.

The W7-X test-4 recurrence lane is currently blocked by long-window residual and
tail-envelope mismatches. The next productive step is a controlled operator
sweep on one wavelength, using one closure family at a time. This tool writes
that sweep as a JSON manifest and prints launch commands that can be executed on
the appropriate remote machine.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "benchmarks" / "runtime_w7x_zonal_response_vmec.toml"
DEFAULT_OUT_DIR = ROOT / "tools_out" / "zonal_response" / "closure_sweep_manifest"


@dataclass(frozen=True)
class SweepCase:
    slug: str
    label: str
    family: str
    nu_hyper: float | None = None
    nu_hyper_l: float | None = None
    nu_hyper_m: float | None = None
    nu_hyper_lm: float | None = None
    p_hyper_l: float | None = None
    p_hyper_m: float | None = None
    p_hyper_lm: float | None = None
    hypercollisions_const: float | None = None
    hypercollisions_kz: float | None = None


DEFAULT_CASES = (
    SweepCase(
        slug="baseline_none",
        label="paper baseline",
        family="baseline",
        hypercollisions_const=0.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuhm_0p01",
        label="constant Hermite hypercollision nu_hyper_m=0.01",
        family="constant_hermite",
        nu_hyper_m=0.01,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuhm_0p03",
        label="constant Hermite hypercollision nu_hyper_m=0.03",
        family="constant_hermite",
        nu_hyper_m=0.03,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="kz_nuhm_0p01",
        label="kz Hermite hypercollision nu_hyper_m=0.01",
        family="kz_hermite",
        nu_hyper_m=0.01,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    ),
    SweepCase(
        slug="kz_nuhm_0p03",
        label="kz Hermite hypercollision nu_hyper_m=0.03",
        family="kz_hermite",
        nu_hyper_m=0.03,
        hypercollisions_const=0.0,
        hypercollisions_kz=1.0,
    ),
    SweepCase(
        slug="const_nuhlm_0p01",
        label="constant mixed LM hypercollision nu_hyper_lm=0.01",
        family="constant_mixed_lm",
        nu_hyper_lm=0.01,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuhlm_0p03",
        label="constant mixed LM hypercollision nu_hyper_lm=0.03",
        family="constant_mixed_lm",
        nu_hyper_lm=0.03,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuhl_0p01",
        label="constant Laguerre hypercollision nu_hyper_l=0.01",
        family="constant_laguerre",
        nu_hyper_l=0.01,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuhl_0p03",
        label="constant Laguerre hypercollision nu_hyper_l=0.03",
        family="constant_laguerre",
        nu_hyper_l=0.03,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuh_0p01",
        label="constant isotropic hypercollision nu_hyper=0.01",
        family="constant_isotropic",
        nu_hyper=0.01,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
    SweepCase(
        slug="const_nuh_0p03",
        label="constant isotropic hypercollision nu_hyper=0.03",
        family="constant_isotropic",
        nu_hyper=0.03,
        hypercollisions_const=1.0,
        hypercollisions_kz=0.0,
    ),
)


def _repo_relative(path: Path | str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _slug_float(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _format_optional(flag: str, value: float | int | None) -> list[str]:
    if value is None:
        return []
    return [flag, f"{value:.12g}" if isinstance(value, float) else str(value)]


def _command_for_case(
    case: SweepCase,
    *,
    config: Path,
    out_dir_root: Path,
    kx: float,
    dt: float,
    steps: int,
    nl: int,
    nm: int,
    sample_stride: int,
    checkpoint_steps: int,
) -> list[str]:
    out_dir = out_dir_root / case.slug
    panel_png = out_dir / "panel.png"
    cmd = [
        "python3",
        "tools/generate_w7x_zonal_response_panel.py",
        "--config",
        _repo_relative(config),
        "--out-dir",
        _repo_relative(out_dir),
        "--out-png",
        _repo_relative(panel_png),
        "--kx-values",
        f"{kx:.12g}",
        "--dt",
        f"{dt:.12g}",
        "--steps",
        str(steps),
        "--Nl",
        str(nl),
        "--Nm",
        str(nm),
        "--sample-stride",
        str(sample_stride),
        "--checkpoint-steps",
        str(checkpoint_steps),
        "--enable-hypercollisions",
        "--show-progress",
    ]
    if case.hypercollisions_const == 0.0 and case.hypercollisions_kz == 0.0:
        cmd.remove("--enable-hypercollisions")
        cmd.extend(["--hypercollisions-const", "0.0", "--hypercollisions-kz", "0.0"])
    else:
        cmd.extend(
            _format_optional("--hypercollisions-const", case.hypercollisions_const)
        )
        cmd.extend(_format_optional("--hypercollisions-kz", case.hypercollisions_kz))
    cmd.extend(_format_optional("--nu-hyper", case.nu_hyper))
    cmd.extend(_format_optional("--nu-hyper-l", case.nu_hyper_l))
    cmd.extend(_format_optional("--nu-hyper-m", case.nu_hyper_m))
    cmd.extend(_format_optional("--nu-hyper-lm", case.nu_hyper_lm))
    cmd.extend(_format_optional("--p-hyper-l", case.p_hyper_l))
    cmd.extend(_format_optional("--p-hyper-m", case.p_hyper_m))
    cmd.extend(_format_optional("--p-hyper-lm", case.p_hyper_lm))
    return cmd


def build_manifest(
    *,
    config: Path = DEFAULT_CONFIG,
    out_dir: Path = DEFAULT_OUT_DIR,
    cases: Iterable[SweepCase] = DEFAULT_CASES,
    kx: float = 0.07,
    dt: float = 0.05,
    steps: int = 2000,
    nl: int = 16,
    nm: int = 64,
    sample_stride: int = 4,
    checkpoint_steps: int = 500,
) -> dict[str, object]:
    rows = []
    launch_commands = []
    plot_args = []
    for case in cases:
        cmd = _command_for_case(
            case,
            config=config,
            out_dir_root=out_dir,
            kx=kx,
            dt=dt,
            steps=steps,
            nl=nl,
            nm=nm,
            sample_stride=sample_stride,
            checkpoint_steps=checkpoint_steps,
        )
        case_out = (
            out_dir / case.slug / f"w7x_test4_kx{int(round(1000.0 * kx)):03d}.out.nc"
        )
        panel_png = out_dir / case.slug / "panel.png"
        rows.append(
            {
                "slug": case.slug,
                "label": case.label,
                "family": case.family,
                "out_dir": _repo_relative(out_dir / case.slug),
                "panel_png": _repo_relative(panel_png),
                "out_nc": _repo_relative(case_out),
                "nu_hyper": case.nu_hyper,
                "nu_hyper_l": case.nu_hyper_l,
                "nu_hyper_m": case.nu_hyper_m,
                "nu_hyper_lm": case.nu_hyper_lm,
                "p_hyper_l": case.p_hyper_l,
                "p_hyper_m": case.p_hyper_m,
                "p_hyper_lm": case.p_hyper_lm,
                "hypercollisions_const": case.hypercollisions_const,
                "hypercollisions_kz": case.hypercollisions_kz,
            }
        )
        launch_commands.append(" ".join(cmd))
        plot_args.append(
            f'--run "{case.label}" "{case.family}" "{_repo_relative(case_out)}"'
        )
    plot_out = out_dir / "w7x_zonal_closure_ladder_full.png"
    plot_json = out_dir / "w7x_zonal_closure_ladder_full.json"
    plot_csv = out_dir / "w7x_zonal_closure_ladder_full.csv"
    plot_command = (
        "python3 tools/artifacts/plot_w7x_zonal_closure_ladder.py "
        f"--out-png {_repo_relative(plot_out)} "
        f"--out-json {_repo_relative(plot_json)} "
        f"--out-csv {_repo_relative(plot_csv)} " + " ".join(plot_args)
    )
    return {
        "kind": "w7x_zonal_closure_sweep_manifest",
        "reference_case": "W7-X test 4 kx rho_i = 0.07",
        "config": _repo_relative(config),
        "out_dir": _repo_relative(out_dir),
        "kx": kx,
        "dt": dt,
        "steps": steps,
        "Nl": nl,
        "Nm": nm,
        "sample_stride": sample_stride,
        "checkpoint_steps": checkpoint_steps,
        "notes": (
            "This sweep isolates closure families one at a time on the paper-facing W7-X "
            "test-4 zonal-response contract. Constant Hermite, kz Hermite, mixed LM, "
            "Laguerre-only, and isotropic hypercollision variants are separated so late-time "
            "trace changes can be attributed to one operator family at a time."
        ),
        "cases": rows,
        "launch_commands": launch_commands,
        "plot_command": plot_command,
        "plot_outputs": {
            "png": _repo_relative(plot_out),
            "json": _repo_relative(plot_json),
            "csv": _repo_relative(plot_csv),
        },
    }


def write_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUT_DIR / "w7x_zonal_closure_sweep_manifest.json",
    )
    parser.add_argument("--kx", type=float, default=0.07)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--Nl", type=int, default=16)
    parser.add_argument("--Nm", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = build_manifest(
        config=args.config,
        out_dir=args.out_dir,
        kx=float(args.kx),
        dt=float(args.dt),
        steps=int(args.steps),
        nl=int(args.Nl),
        nm=int(args.Nm),
        sample_stride=int(args.sample_stride),
        checkpoint_steps=int(args.checkpoint_steps),
    )
    path = write_manifest(args.manifest, payload)
    print(f"Wrote {path}")
    launch_commands = payload["launch_commands"]
    if not isinstance(launch_commands, list):
        raise TypeError("manifest launch_commands must be a list")
    for cmd in launch_commands:
        print(cmd)
    plot_command = payload["plot_command"]
    if not isinstance(plot_command, str):
        raise TypeError("manifest plot_command must be a string")
    print(plot_command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
