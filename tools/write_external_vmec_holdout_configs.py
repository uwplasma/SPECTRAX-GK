#!/usr/bin/env python3
"""Write reproducible nonlinear external-VMEC holdout run configs.

The generated TOMLs encode the nonlinear transport-promotion protocol used by
the quasilinear calibration campaign: fixed-step ITG/adiabatic-electron runs at
two grid resolutions, with optional restart continuations for the longer
transport window.  The tool intentionally writes configs only; large runs should
still be launched on the appropriate CPU/GPU machine and promoted only through
``plot_external_vmec_nonlinear_convergence_gate.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRIDS = (
    "n48:48:48:32:32",
    "n64:64:64:40:40",
)
DEFAULT_HORIZONS = (150.0, 250.0)


@dataclass(frozen=True)
class GridSpec:
    """Resolution tuple used in one external-VMEC nonlinear holdout run."""

    label: str
    nx: int
    ny: int
    nz: int
    ntheta: int


@dataclass(frozen=True)
class VariantSpec:
    """Seed or timestep variant used in a replicated nonlinear campaign."""

    label: str
    random_seed: int
    dt: float
    axis: str


@dataclass(frozen=True)
class WrittenConfig:
    """One generated TOML plus launch metadata."""

    path: Path
    output_path: Path
    case: str
    grid: GridSpec
    horizon: float
    dt: float
    steps: int
    restart_if_exists: bool
    variant: VariantSpec | None = None


def _parse_grid(raw: str) -> GridSpec:
    parts = raw.split(":")
    if len(parts) != 5:
        raise ValueError("grid specs must be label:Nx:Ny:Nz:ntheta")
    label, nx, ny, nz, ntheta = parts
    try:
        values = tuple(int(item) for item in (nx, ny, nz, ntheta))
    except ValueError as exc:
        raise ValueError(f"grid spec {raw!r} contains a non-integer resolution") from exc
    if any(value <= 0 for value in values):
        raise ValueError(f"grid spec {raw!r} must contain positive resolutions")
    return GridSpec(label=label, nx=values[0], ny=values[1], nz=values[2], ntheta=values[3])


def _parse_horizons(raw: str) -> tuple[float, ...]:
    out = tuple(float(item) for item in raw.split(",") if item.strip())
    if not out:
        raise ValueError("at least one horizon is required")
    if any(value <= 0.0 for value in out):
        raise ValueError("horizons must be positive")
    if tuple(sorted(out)) != out:
        raise ValueError("horizons must be sorted increasingly")
    return out


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


def _horizon_label(value: float) -> str:
    rounded = int(round(value))
    if abs(value - rounded) < 1.0e-12:
        return str(rounded)
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _float_label(value: float) -> str:
    return f"{float(value):.12g}".replace(".", "p").replace("-", "m")


def _seed_dt_label(seed: int, dt: float) -> str:
    return f"seed{int(seed)}_dt{_float_label(float(dt))}"


def _parse_seed_dt_variant(raw: str) -> tuple[int, float]:
    """Parse a joint seed/timestep variant encoded as ``SEED:DT``."""

    parts = raw.split(":")
    if len(parts) != 2:
        raise ValueError("joint seed/timestep variants must have format SEED:DT")
    try:
        seed = int(parts[0])
        dt_value = float(parts[1])
    except ValueError as exc:
        raise ValueError(f"invalid seed/timestep variant {raw!r}") from exc
    if seed < 0:
        raise ValueError("joint seed/timestep variant seed must be non-negative")
    if dt_value <= 0.0:
        raise ValueError("joint seed/timestep variant dt must be positive")
    return seed, dt_value


def _resolved_vmec_file(path: Path) -> Path:
    """Return an absolute VMEC source path before rendering generated TOMLs.

    The rendered TOML stores a path relative to each generated config when
    possible.  Keeping the validated source path absolute here avoids ambiguous
    input handling while still producing portable campaign artifacts that can
    be copied from a laptop checkout to an office/GPU checkout.
    """

    return Path(path).expanduser().resolve()


def _vmec_file_for_config(vmec_file: Path, config_path: Path) -> Path:
    """Return a VMEC path portable with respect to ``config_path``."""

    try:
        return Path(os.path.relpath(vmec_file, start=config_path.parent))
    except ValueError:
        # Different Windows drives, or another path relation edge case.
        return vmec_file


def _build_variants(
    *,
    dt: float,
    baseline_seed: int,
    seed_variants: Iterable[int] | None,
    dt_variants: Iterable[float] | None,
    seed_dt_variants: Iterable[tuple[int, float]] | None = None,
) -> tuple[VariantSpec | None, ...]:
    seed_values = tuple(int(value) for value in (seed_variants or ()))
    dt_values = tuple(float(value) for value in (dt_variants or ()))
    seed_dt_values = tuple((int(seed), float(dt_value)) for seed, dt_value in (seed_dt_variants or ()))
    if not seed_values and not dt_values and not seed_dt_values:
        return (None,)
    variants: list[VariantSpec] = []
    for seed in seed_values:
        variants.append(
            VariantSpec(
                label=f"seed{seed}",
                random_seed=seed,
                dt=float(dt),
                axis="seed",
            )
        )
    for dt_value in dt_values:
        if dt_value <= 0.0:
            raise ValueError("dt variants must be positive")
        variants.append(
            VariantSpec(
                label=f"dt{_float_label(dt_value)}",
                random_seed=int(baseline_seed),
                dt=dt_value,
                axis="timestep",
            )
        )
    for seed, dt_value in seed_dt_values:
        if seed < 0:
            raise ValueError("joint seed/timestep variant seed must be non-negative")
        if dt_value <= 0.0:
            raise ValueError("joint seed/timestep variant dt must be positive")
        variants.append(
            VariantSpec(
                label=_seed_dt_label(seed, dt_value),
                random_seed=seed,
                dt=dt_value,
                axis="seed_timestep",
            )
        )
    labels = [variant.label for variant in variants]
    if len(labels) != len(set(labels)):
        raise ValueError("seed/timestep variants produced duplicate labels")
    return tuple(variants)


def _render_config(
    *,
    case: str,
    vmec_file: Path,
    grid: GridSpec,
    horizon: float,
    dt: float,
    steps: int,
    output_path: str,
    restart_if_exists: bool,
    ky: float,
    nl: int,
    nm: int,
    torflux: float,
    alpha: float,
    npol: float,
    tprim: float,
    fprim: float,
    nu: float,
    init_amp: float,
    y0: float,
    lx: float,
    ly: float,
    sample_stride: int,
    diagnostics_stride: int,
    progress_bar: bool,
    random_seed: int,
    variant: VariantSpec | None,
) -> str:
    metadata = ""
    if variant is not None:
        metadata = f"""
[metadata]
case = "{case}"
variant_axis = "{variant.axis}"
variant_label = "{variant.label}"
seed = {variant.random_seed}
timestep = {variant.dt:.16g}
"""
    return f"""# Generated by tools/write_external_vmec_holdout_configs.py.
# Case: {case}

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = {tprim:.16g}
fprim = {fprim:.16g}
nu = {nu:.16g}
kinetic = true

[grid]
Nx = {grid.nx}
Ny = {grid.ny}
Nz = {grid.nz}
Lx = {lx:.16g}
Ly = {ly:.16g}
boundary = "fix aspect"
y0 = {y0:.16g}
ntheta = {grid.ntheta}
nperiod = 1

[time]
t_max = {horizon:.16g}
dt = {dt:.16g}
method = "rk3"
use_diffrax = false
sample_stride = {sample_stride}
diagnostics_stride = {diagnostics_stride}
fixed_dt = true
progress_bar = {_toml_bool(progress_bar)}

[geometry]
model = "vmec"
vmec_file = "{vmec_file.as_posix()}"
torflux = {torflux:.16g}
alpha = {alpha:.16g}
npol = {npol:.16g}

[init]
init_field = "density"
init_amp = {init_amp:.16g}
random_seed = {random_seed}
gaussian_init = false
init_single = false

[physics]
linear = false
nonlinear = true
electrostatic = true
electromagnetic = false
adiabatic_electrons = true
adiabatic_ions = false
tau_e = 1.0
beta = 0.0
collisions = true
hypercollisions = true

[collisions]
nu_hermite = 1.0
nu_laguerre = 2.0
nu_hyper = 0.0
p_hyper = 4.0
hypercollisions_const = 0.0
hypercollisions_kz = 1.0
D_hyper = 0.05
damp_ends_amp = 0.1
damp_ends_widthfrac = 0.125

[normalization]
contract = "kinetic"
diagnostic_norm = "gx"

[terms]
streaming = 1.0
mirror = 1.0
curvature = 1.0
gradb = 1.0
diamagnetic = 1.0
collisions = 1.0
hypercollisions = 1.0
hyperdiffusion = 1.0
end_damping = 1.0
apar = 0.0
bpar = 0.0
nonlinear = 1.0

[run]
ky = {ky:.16g}
Nl = {nl}
Nm = {nm}
steps = {steps}
sample_stride = {sample_stride}
diagnostics = true

[output]
path = "{output_path}"
restart_if_exists = {_toml_bool(restart_if_exists)}
append_on_restart = true
save_for_restart = true
nsave = {steps}
{metadata}"""


def write_configs(
    *,
    case: str,
    vmec_file: Path,
    out_dir: Path,
    grids: Iterable[GridSpec],
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    dt: float = 0.05,
    ky: float = 0.47619047619047616,
    nl: int = 4,
    nm: int = 8,
    torflux: float = 0.64,
    alpha: float = 0.0,
    npol: float = 1.0,
    tprim: float = 3.0,
    fprim: float = 1.0,
    nu: float = 0.01,
    init_amp: float = 1.0e-3,
    y0: float = 21.0,
    lx: float = 62.8,
    ly: float = 62.8,
    sample_stride: int = 50,
    diagnostics_stride: int = 50,
    progress_bar: bool = False,
    baseline_seed: int = 22,
    seed_variants: Iterable[int] | None = None,
    dt_variants: Iterable[float] | None = None,
    seed_dt_variants: Iterable[tuple[int, float]] | None = None,
) -> list[WrittenConfig]:
    """Write all configs and return their metadata."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if nl <= 0 or nm <= 0:
        raise ValueError("Nl and Nm must be positive")
    vmec_file = _resolved_vmec_file(vmec_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[WrittenConfig] = []
    variants = _build_variants(
        dt=float(dt),
        baseline_seed=int(baseline_seed),
        seed_variants=seed_variants,
        dt_variants=dt_variants,
        seed_dt_variants=seed_dt_variants,
    )
    for variant in variants:
        variant_dt = float(dt if variant is None else variant.dt)
        previous_horizon = 0.0
        for horizon in horizons:
            delta = horizon - previous_horizon
            steps = int(round(delta / variant_dt))
            if steps <= 0:
                raise ValueError("horizons and dt imply a non-positive step count")
            restart_if_exists = previous_horizon > 0.0
            for grid in grids:
                variant_suffix = "" if variant is None else f"_{variant.label}"
                stem = f"{case}_nonlinear_t{_horizon_label(horizon)}_{grid.label}{variant_suffix}"
                config_path = out_dir / f"{stem}.toml"
                output_filename = f"{stem}.out.nc"
                rendered_vmec_file = _vmec_file_for_config(vmec_file, config_path)
                config_path.write_text(
                    _render_config(
                        case=case,
                        vmec_file=rendered_vmec_file,
                        grid=grid,
                        horizon=horizon,
                        dt=variant_dt,
                        steps=steps,
                        output_path=output_filename,
                        restart_if_exists=restart_if_exists,
                        ky=ky,
                        nl=nl,
                        nm=nm,
                        torflux=torflux,
                        alpha=alpha,
                        npol=npol,
                        tprim=tprim,
                        fprim=fprim,
                        nu=nu,
                        init_amp=init_amp,
                        y0=y0,
                        lx=lx,
                        ly=ly,
                        sample_stride=sample_stride,
                        diagnostics_stride=diagnostics_stride,
                        progress_bar=progress_bar,
                        random_seed=int(baseline_seed if variant is None else variant.random_seed),
                        variant=variant,
                    ),
                    encoding="utf-8",
                )
                written.append(
                    WrittenConfig(
                        path=config_path,
                        output_path=out_dir / output_filename,
                        case=case,
                        grid=grid,
                        horizon=horizon,
                        dt=variant_dt,
                        steps=steps,
                        restart_if_exists=restart_if_exists,
                        variant=variant,
                    )
                )
            previous_horizon = horizon
    return written


def _bundle_base(path: Path) -> Path:
    name = path.name
    for suffix in (".out.nc", ".big.nc", ".restart.nc"):
        if name.endswith(suffix):
            return path.with_name(name[: -len(suffix)])
    return path.with_suffix("") if path.suffix == ".nc" else path


def write_manifest(out_dir: Path, written: list[WrittenConfig]) -> Path:
    """Write launch and restart-copy commands next to generated configs."""

    previous_by_grid: dict[tuple[str, str], WrittenConfig] = {}
    configs: list[dict[str, Any]] = []
    launch_commands: list[str] = []
    restart_seed_commands: list[str] = []
    staged_ladder_commands: list[str] = []
    direct_full_horizon_launch_commands: list[str] = []
    segment_step_counts: dict[str, int] = {}
    direct_full_horizon_step_counts: dict[str, int] = {}
    manifest: dict[str, Any] = {
        "kind": "external_vmec_holdout_config_manifest",
        "configs": configs,
        "launch_commands": launch_commands,
        "restart_seed_commands": restart_seed_commands,
        "staged_ladder_commands": staged_ladder_commands,
        "direct_full_horizon_launch_commands": direct_full_horizon_launch_commands,
        "segment_step_counts": segment_step_counts,
        "direct_full_horizon_step_counts": direct_full_horizon_step_counts,
        "restart_ladder_note": (
            "launch_commands are restart-ladder segments. For continuation "
            "horizons, run the corresponding restart_seed_command first, or run "
            "the direct_full_horizon_launch_command when starting from t=0. "
            "Running a final segment command from t=0 intentionally reaches only "
            "the segment duration, not the horizon encoded in the file name."
        ),
    }
    for item in written:
        direct_steps = int(round(float(item.horizon) / float(item.dt)))
        config_key = item.path.stem
        variant_payload = None
        if item.variant is not None:
            variant_payload = {
                "label": item.variant.label,
                "axis": item.variant.axis,
                "seed": item.variant.random_seed,
                "timestep": item.variant.dt,
            }
        configs.append(
            {
                "path": item.path.as_posix(),
                "output_path": item.output_path.as_posix(),
                "case": item.case,
                "grid": item.grid.label,
                "horizon": item.horizon,
                "dt": item.dt,
                "steps": item.steps,
                "direct_full_horizon_steps": direct_steps,
                "restart_if_exists": item.restart_if_exists,
                "variant": variant_payload,
            }
        )
        segment_command = (
            "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {item.path.as_posix()} --steps {int(item.steps)} --no-progress"
        )
        direct_command = (
            "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear "
            f"--config {item.path.as_posix()} --steps {int(direct_steps)} --no-progress"
        )
        launch_commands.append(segment_command)
        direct_full_horizon_launch_commands.append(direct_command)
        segment_step_counts[config_key] = int(item.steps)
        direct_full_horizon_step_counts[config_key] = int(direct_steps)
        variant_key = "" if item.variant is None else item.variant.label
        previous_key = (item.grid.label, variant_key)
        if item.restart_if_exists:
            previous = previous_by_grid[previous_key]
            src_base = _bundle_base(previous.output_path)
            dst_base = _bundle_base(item.output_path)
            restart_command = (
                "for ext in out.nc restart.nc big.nc; do "
                f"cp {src_base.as_posix()}.$ext {dst_base.as_posix()}.$ext; "
                "done"
            )
            restart_seed_commands.append(restart_command)
            staged_ladder_commands.append(restart_command)
        staged_ladder_commands.append(segment_command)
        previous_by_grid[previous_key] = item
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, help="Short slug used in file names, e.g. circular or itermodel")
    parser.add_argument("--vmec-file", required=True, type=Path, help="External VMEC wout file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for generated TOMLs and manifest")
    parser.add_argument("--ky", type=float, default=0.47619047619047616)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--Nl", type=int, default=4)
    parser.add_argument("--Nm", type=int, default=8)
    parser.add_argument(
        "--baseline-seed",
        type=int,
        default=22,
        help="Seed used for timestep variants and non-replicate baseline configs.",
    )
    parser.add_argument(
        "--seed-variant",
        action="append",
        type=int,
        default=None,
        help="Random-seed replicate to generate. Repeat for multiple seed variants.",
    )
    parser.add_argument(
        "--dt-variant",
        action="append",
        type=float,
        default=None,
        help="Timestep replicate to generate. Repeat for multiple dt variants.",
    )
    parser.add_argument(
        "--seed-dt-variant",
        action="append",
        default=None,
        help="Joint seed/timestep replicate encoded as SEED:DT. Repeat for multiple cross variants.",
    )
    parser.add_argument("--horizons", default=",".join(str(v).rstrip("0").rstrip(".") for v in DEFAULT_HORIZONS))
    parser.add_argument("--grid", action="append", default=None, help="Grid spec label:Nx:Ny:Nz:ntheta")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    grids = tuple(_parse_grid(raw) for raw in (args.grid or DEFAULT_GRIDS))
    horizons = _parse_horizons(str(args.horizons))
    written = write_configs(
        case=str(args.case),
        vmec_file=args.vmec_file,
        out_dir=args.out_dir,
        grids=grids,
        horizons=horizons,
        dt=float(args.dt),
        ky=float(args.ky),
        nl=int(args.Nl),
        nm=int(args.Nm),
        baseline_seed=int(args.baseline_seed),
        seed_variants=args.seed_variant,
        dt_variants=args.dt_variant,
        seed_dt_variants=tuple(_parse_seed_dt_variant(raw) for raw in (args.seed_dt_variant or ())),
    )
    manifest = write_manifest(args.out_dir, written)
    print(f"wrote {len(written)} configs")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
