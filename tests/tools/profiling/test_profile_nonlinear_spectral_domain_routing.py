from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "profiling" / "profile_nonlinear_spectral_domain_routing.py"
spec = importlib.util.spec_from_file_location(
    "profile_nonlinear_spectral_domain_routing", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_profile_nonlinear_spectral_domain_routing_parser_defaults() -> None:
    args = mod.build_parser().parse_args([])

    assert args.out_prefix == mod.DEFAULT_OUT_PREFIX
    assert (args.nl, args.nm, args.ny, args.nx, args.nz) == (2, 4, 32, 32, 4)
    assert args.y_chunks == (16, 16)
    assert args.x_chunks == (16, 16)
    assert args.min_speedup == 1.5


def test_profile_nonlinear_spectral_domain_routing_builds_identity_payload() -> None:
    payload = mod.build_profile(
        shape=(2, 2, 4, 4, 1),
        y_chunks=(2, 2),
        x_chunks=(2, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        min_speedup=1.5,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    assert payload["kind"] == "nonlinear_spectral_domain_routing_profile"
    assert payload["identity_passed"] is True
    assert payload["decomposed_path_enabled"] is True
    assert payload["timing_identity_max_abs_error"] <= payload["atol"]
    assert payload["timing_identity_max_rel_error"] <= payload["rtol"]
    assert payload["production_speedup_claim_allowed"] is False
    assert payload["work_model"]["num_tiles"] == 4
    assert payload["work_model"]["production_speedup_feasible"] is False
    assert payload["communication_to_owned_work_ratio"] > 1.0
    assert payload["parallel_efficiency_ceiling"] < 0.5
    assert payload["serial_stats_s"]["median"] > 0.0
    assert payload["logical_domain_stats_s"]["median"] > 0.0
    assert payload["strong_speedup_vs_serial"] is not None


def test_profile_nonlinear_spectral_domain_routing_writes_artifacts(
    tmp_path: Path,
) -> None:
    payload = mod.build_profile(
        shape=(2, 2, 4, 4, 1),
        y_chunks=(2, 2),
        x_chunks=(2, 2),
        steps=1,
        dt=0.001,
        warmups=0,
        repeats=1,
        min_speedup=1.5,
        atol=5.0e-6,
        rtol=5.0e-6,
    )

    paths = mod.write_artifacts(payload, tmp_path / "domain_profile")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "domain_profile.json").read_text(encoding="utf-8"))
    assert saved["identity_passed"] is True
    assert saved["work_model_speedup_feasible"] is False
