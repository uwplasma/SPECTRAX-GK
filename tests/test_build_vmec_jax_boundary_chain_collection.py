from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_jax_boundary_chain_collection.py"
spec = importlib.util.spec_from_file_location(
    "build_vmec_jax_boundary_chain_collection",
    SCRIPT,
)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _probe(path: Path, *, index: int, exact_ok: bool, growth_ok: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_boundary_chain_probe",
                "index": index,
                "name": f"coeff{index}",
                "summary": {
                    "kind": "vmec_jax_boundary_chain_summary",
                    "finite": True,
                    "classification": (
                        "exact_fd_and_frozen_axis_replay_consistent"
                        if exact_ok
                        else "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
                    ),
                    "passes": {
                        "final_state_matches_exact_fd": True,
                        "frozen_axis_matches_exact_fd": exact_ok,
                        "frozen_axis_jvp_vjp_consistent": True,
                    },
                    "errors": {
                        "frozen_axis_vs_exact_fd_rel": 0.02 if exact_ok else 0.4,
                    },
                    "metrics": {
                        "exact_fd_cost_gradient": 0.1,
                        "frozen_axis_replay_cost_gradient": 0.1 if exact_ok else 0.2,
                    },
                },
                "growth_branch_locality": {
                    "enabled": True,
                    "passed": growth_ok,
                    "classification": (
                        "all_samples_dominant_growth_branch_locally_consistent"
                        if growth_ok
                        else "growth_branch_locality_failed_or_incomplete"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_collection_payload_counts_growth_branch_status(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _probe(first, index=1, exact_ok=True, growth_ok=True)
    _probe(second, index=2, exact_ok=False, growth_ok=False)

    payload = mod.build_collection_payload([first, second])

    assert payload["finite"] is True
    assert payload["classification"] == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    assert payload["counts"]["n_exact_fd_consistent"] == 1
    assert payload["counts"]["n_growth_branch_locality_checked"] == 2
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1
    assert payload["rows"][0]["growth_branch_locality_passed"] is True
    assert payload["rows"][1]["growth_branch_locality_passed"] is False
    assert payload["probe_jsons"] == [str(first), str(second)]
    assert "not a nonlinear transport optimization claim" in payload["claim_scope"]


def test_build_collection_main_writes_json(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    out = tmp_path / "collection.json"
    _probe(first, index=1, exact_ok=True, growth_ok=True)

    rc = mod.main(["--probe-json", str(first), "--out-json", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["counts"]["n_total"] == 1
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1
