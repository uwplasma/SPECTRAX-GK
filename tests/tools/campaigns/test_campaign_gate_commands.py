from __future__ import annotations

from pathlib import Path
import tomllib

from tools.campaigns import run_kbm_lowky_extractor_audit as kbm_lowky
from tools.campaigns.run_benchmark_refresh import _load_manifest, _select_jobs
from tools.campaigns.run_device_parity_gate import (
    _resolve_manifest_path as resolve_device_manifest_path,
)
from tools.campaigns.run_device_parity_gate import build_parser as device_parser
from tools.campaigns.run_imported_linear_targeted_audit import (
    build_parser as imported_linear_parser,
)
from tools.campaigns.run_kbm_lowky_extractor_audit import (
    build_parser as kbm_lowky_parser,
)
from tools.campaigns.run_restart_parity_gate import (
    _resolve_manifest_path as resolve_restart_manifest_path,
)
from tools.campaigns.run_restart_parity_gate import build_parser as restart_parser
from tools.campaigns.run_vmec_roundtrip_gate import (
    _resolve_manifest_path as resolve_roundtrip_manifest_path,
)
from tools.campaigns.run_vmec_roundtrip_gate import build_parser as roundtrip_parser


ROOT = Path(__file__).resolve().parents[3]
W7X_RUNTIME_EXAMPLE = (
    ROOT
    / "examples"
    / "nonlinear"
    / "non-axisymmetric"
    / "runtime_w7x_nonlinear_vmec_geometry.toml"
)
KBM_RUNTIME_EXAMPLE = (
    ROOT
    / "examples"
    / "nonlinear"
    / "axisymmetric"
    / "runtime_kbm_nonlinear_t100.toml"
)


def _manifest_lane_config(manifest_name: str, lane: str) -> tuple[Path, str]:
    manifest = ROOT / "tools" / manifest_name
    data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    return manifest, data["lane"][lane]["config"]


def test_device_parity_gate_parser_and_manifest_paths(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    ns = device_parser().parse_args(
        ["--manifest", "lanes.toml", "--outdir", str(outdir)]
    )
    assert ns.manifest == Path("lanes.toml")
    assert ns.outdir == outdir

    manifest, config = _manifest_lane_config("device_parity_lanes.office.toml", "w7x_vmec")
    resolved = resolve_device_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == W7X_RUNTIME_EXAMPLE
    assert resolved.is_file()


def test_vmec_roundtrip_gate_parser_and_manifest_paths(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    ns = roundtrip_parser().parse_args(
        ["--manifest", "lanes.toml", "--outdir", str(outdir)]
    )
    assert ns.manifest == Path("lanes.toml")
    assert ns.outdir == outdir

    manifest, config = _manifest_lane_config("vmec_roundtrip_lanes.office.toml", "w7x_vmec")
    resolved = resolve_roundtrip_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == W7X_RUNTIME_EXAMPLE
    assert resolved.is_file()


def test_restart_parity_gate_parser_and_manifest_paths() -> None:
    args = restart_parser().parse_args(
        ["--manifest", "tools/restart_gate_lanes.office.toml", "--lane", "kbm_salpha"]
    )
    assert args.manifest == Path("tools/restart_gate_lanes.office.toml")
    assert args.lane == "kbm_salpha"

    manifest, config = _manifest_lane_config("restart_gate_lanes.office.toml", "w7x_vmec")
    resolved = resolve_restart_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == W7X_RUNTIME_EXAMPLE
    assert resolved.is_file()

    manifest, config = _manifest_lane_config("restart_gate_lanes.office.toml", "kbm_salpha")
    resolved = resolve_restart_manifest_path(config, manifest_dir=manifest.parent)
    assert resolved == KBM_RUNTIME_EXAMPLE
    assert resolved.is_file()


def test_benchmark_refresh_manifest_loading_and_selection(tmp_path: Path) -> None:
    manifest = ROOT / "tools" / "benchmark_refresh_manifest.toml"
    jobs = _load_manifest(manifest)

    names = [job.name for job in jobs]
    assert "cyclone-core-assets" in names
    assert "benchmark-atlas" in names
    assert any(job.requires_env for job in jobs)
    commands = {job.name: job.command for job in jobs}
    assert commands["imported-linear-w7x"].startswith("JAX_PLATFORMS=cpu ")
    assert commands["imported-linear-hsx"].startswith("JAX_PLATFORMS=cpu ")
    outputs = {job.name: job.outputs for job in jobs}
    assert "docs/_static/w7x_linear_t2_lastvalue.csv" in outputs["imported-linear-w7x"]
    assert "docs/_static/hsx_linear_t2_lastvalue.csv" in outputs["imported-linear-hsx"]

    mini_manifest = tmp_path / "mini.toml"
    mini_manifest.write_text(
        """
[[job]]
name = "one"
description = "first"
command = "echo one"

[[job]]
name = "two"
description = "second"
command = "echo two"
enabled = false
""",
        encoding="utf-8",
    )
    selected = _select_jobs(_load_manifest(mini_manifest), {"one", "two"})
    assert [job.name for job in selected] == ["one"]


def test_imported_linear_targeted_audit_parser_controls(tmp_path: Path) -> None:
    out = tmp_path / "combined.csv"
    args = imported_linear_parser().parse_args(
        ["--gx", "gx.out.nc", "--geometry-file", "geom.nc", "--out", str(out)]
    )
    assert args.gx == Path("gx.out.nc")
    assert args.geometry_file == Path("geom.nc")
    assert args.out == out
    assert args.max_kys is None
    assert args.Nl is None
    assert args.Nm is None
    assert args.sample_step_stride == 1
    assert args.max_samples is None
    assert args.sample_window == "head"

    args = imported_linear_parser().parse_args(
        [
            "--gx",
            "gx.out.nc",
            "--geometry-file",
            "geom.nc",
            "--out",
            str(out),
            "--sample-step-stride",
            "4",
            "--max-samples",
            "16",
            "--sample-window",
            "tail",
            "--reuse-cache",
        ]
    )
    assert args.sample_step_stride == 4
    assert args.max_samples == 16
    assert args.sample_window == "tail"
    assert args.reuse_cache is True

    args = imported_linear_parser().parse_args(
        [
            "--gx",
            "gx.out.nc",
            "--geometry-file",
            "geom.nc",
            "--out",
            str(out),
            "--mode-method",
            "project",
        ]
    )
    assert args.mode_method == "project"


def test_kbm_lowky_extractor_audit_parser_and_dispatch(tmp_path: Path) -> None:
    out = tmp_path / "branch.csv"
    args = kbm_lowky_parser().parse_args(["--gx", "kbm.out.nc", "--out", str(out)])
    assert args.gx == Path("kbm.out.nc")
    assert args.out == out
    assert args.ky == "0.3,0.4"
    assert args.gx_input is None

    gx = tmp_path / "kbm_salpha_correct.out.nc"
    gx.write_text("", encoding="utf-8")
    gx_input = tmp_path / "kbm_salpha.in"
    gx_input.write_text("[Geometry]\ngeo_option = 's-alpha'\n", encoding="utf-8")
    args = kbm_lowky_parser().parse_args(
        ["--gx", str(gx), "--gx-input", str(gx_input), "--out", str(tmp_path / "out.csv")]
    )
    cmd = kbm_lowky._build_command(
        args,
        here=Path("/toolroot"),
        gx=gx,
        gx_input=gx_input,
        gx_big=tmp_path / "gx.big.nc",
    )
    assert cmd[1] == "/toolroot/compare_gx_kbm.py"
    assert "--candidate-out" in cmd

    gx = tmp_path / "kbm_miller_correct.out.nc"
    gx.write_text("", encoding="utf-8")
    gx_input = tmp_path / "kbm_miller.in"
    gx_input.write_text("[Geometry]\ngeo_option = 'miller'\n", encoding="utf-8")
    args = kbm_lowky_parser().parse_args(
        [
            "--gx",
            str(gx),
            "--gx-input",
            str(gx_input),
            "--out",
            str(tmp_path / "out.csv"),
            "--sample-step-stride",
            "2",
            "--max-samples",
            "16",
        ]
    )
    cmd = kbm_lowky._build_command(
        args,
        here=Path("/toolroot"),
        gx=gx,
        gx_input=gx_input,
        gx_big=tmp_path / "gx.big.nc",
    )
    assert cmd[1] == "/toolroot/run_imported_linear_targeted_audit.py"
    assert "--geometry-file" in cmd
    assert "--sample-step-stride" in cmd
    assert "--max-samples" in cmd
