from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "tools"
    / "campaigns"
    / "write_vmec_jax_projected_transport_line_search_inputs.py"
)
spec = importlib.util.spec_from_file_location(
    "write_vmec_jax_projected_transport_line_search_inputs", SCRIPT
)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _gradient_report(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_transport_gradient_diagnostic",
                "parameter_count": 4,
                "top_gradient_components": [
                    {"parameter_index": 1, "gradient": -3.0, "name": "zs10"},
                    {"parameter_index": 3, "gradient": 4.0, "name": "rc11"},
                    {"parameter_index": 0, "gradient": 2.0, "name": "rc01"},
                    {"parameter_index": 2, "gradient": -1.0, "name": "zs11"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _boundary_chain_collection(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_boundary_chain_collection_summary",
                "classification": "mixed_exact_fd_consistency_with_branch_sensitive_modes",
                "rows": [
                    {
                        "index": 1,
                        "name": "zs10",
                        "frozen_axis_jvp_vjp_consistent": True,
                        "frozen_axis_matches_exact_fd": True,
                        "frozen_axis_convention_verified": False,
                        "growth_branch_locality_checked": True,
                        "growth_branch_locality_passed": True,
                    },
                    {
                        "index": 3,
                        "name": "rc11",
                        "frozen_axis_jvp_vjp_consistent": True,
                        "frozen_axis_matches_exact_fd": False,
                        "frozen_axis_convention_verified": True,
                        "growth_branch_locality_checked": True,
                        "growth_branch_locality_passed": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_projected_writer_defaults_to_multisample_transport_contract(
    tmp_path: Path,
) -> None:
    args = mod._parse_args(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(tmp_path / "gradient.json"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    sample_set = mod._sample_set_from_args(args)
    summary = mod.transport_objective_sample_summary(sample_set)

    assert args.surfaces == mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == mod.DEFAULT_TRANSPORT_KY_VALUES
    assert summary["passed"] is True
    assert summary["sample_count"] == 18


def test_projected_writer_manifest_records_sample_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)
    saved: list[tuple[Path, object]] = []

    class FakeOptimizer:
        def save_input(self, path, delta):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("! projected input\n", encoding="utf-8")
            saved.append((Path(path), delta))

    fake_stage = SimpleNamespace(
        specs=[object(), object(), object(), object()], optimizer=FakeOptimizer()
    )
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--boundary-chain-collection-json",
            str(collection),
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3,2e-3",
            "--top-n",
            "4",
        ]
    )

    manifest_path = tmp_path / "out" / "projected_line_search_inputs.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert len(saved) == 2
    assert payload["objective_sample_summary"]["passed"] is True
    assert payload["objective_sample_summary"]["sample_count"] == 18
    assert payload["transport_objective_sample_set"]["surfaces"] == [0.45, 0.64, 0.78]
    assert payload["transport_objective_sample_set"]["alphas"] == [
        0.0,
        0.7853981633974483,
    ]
    assert payload["transport_objective_sample_set"]["ky_values"] == [0.1, 0.3, 0.5]
    assert payload["boundary_chain_filter"]["accepted_parameter_indices"] == [1]
    command = payload["rows"][0]["replay_command"]
    assert "--surfaces" in command
    assert "0.45,0.64,0.78" in command
    assert "--alphas" in command
    assert "0.0,0.7853981633974483" in command
    assert "--ky-values" in command
    assert "0.1,0.3,0.5" in command
    assert "--target-aspect" in command
    assert "6.0" in command
    assert "--iota-objective" in command
    assert "floor" in command
    assert "--iota-profile-floor" in command
    assert "--solved-wout-gate-min-abs-iota" in command
    assert "--surface-chunk-size" in command
    assert "0" in command


def test_projected_writer_requires_boundary_chain_collection_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    _gradient_report(gradient)

    def unexpected_stage(_args):
        raise AssertionError(
            "ungated projected update should fail before VMEC-JAX stage construction"
        )

    monkeypatch.setattr(mod, "_build_stage", unexpected_stage)
    with pytest.raises(ValueError, match="require --boundary-chain-collection-json"):
        mod.main(
            [
                "--input",
                str(tmp_path / "input.final"),
                "--gradient-json",
                str(gradient),
                "--outdir",
                str(tmp_path / "out"),
            ]
        )


def test_projected_writer_filters_direction_by_boundary_chain_collection(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)
    saved: list[object] = []

    class FakeOptimizer:
        def save_input(self, path, delta):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("! projected input\n", encoding="utf-8")
            saved.append(delta)

    fake_stage = SimpleNamespace(
        specs=[object(), object(), object(), object()], optimizer=FakeOptimizer()
    )
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    rc = mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--boundary-chain-collection-json",
            str(collection),
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3",
            "--top-n",
            "4",
        ]
    )

    payload = json.loads(
        (tmp_path / "out" / "projected_line_search_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    assert rc == 0
    assert len(saved) == 1
    assert list(saved[0]) == pytest.approx([0.0, 1.0e-3, 0.0, 0.0])
    assert payload["boundary_chain_filter"]["accepted_parameter_indices"] == [1]
    assert payload["boundary_chain_collection_json"] == str(collection)


def test_projected_writer_can_mark_branch_sensitive_filter_as_diagnostic(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)
    saved: list[object] = []

    class FakeOptimizer:
        def save_input(self, path, delta):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("! projected input\n", encoding="utf-8")
            saved.append(delta)

    fake_stage = SimpleNamespace(
        specs=[object(), object(), object(), object()], optimizer=FakeOptimizer()
    )
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--boundary-chain-collection-json",
            str(collection),
            "--allow-boundary-chain-branch-sensitive",
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3",
            "--top-n",
            "4",
        ]
    )

    payload = json.loads(
        (tmp_path / "out" / "projected_line_search_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(saved[0]) == pytest.approx([0.0, 6.0e-4, 0.0, -8.0e-4])
    assert payload["boundary_chain_filter"]["require_exact_fd"] is False
    assert payload["boundary_chain_filter"]["accepted_parameter_indices"] == [1, 3]


def test_projected_writer_can_require_growth_branch_locality(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)
    saved: list[object] = []

    class FakeOptimizer:
        def save_input(self, path, delta):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("! projected input\n", encoding="utf-8")
            saved.append(delta)

    fake_stage = SimpleNamespace(
        specs=[object(), object(), object(), object()], optimizer=FakeOptimizer()
    )
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--boundary-chain-collection-json",
            str(collection),
            "--allow-boundary-chain-branch-sensitive",
            "--require-growth-branch-locality",
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3",
            "--top-n",
            "4",
        ]
    )

    payload = json.loads(
        (tmp_path / "out" / "projected_line_search_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    assert list(saved[0]) == pytest.approx([0.0, 1.0e-3, 0.0, 0.0])
    assert payload["boundary_chain_filter"]["require_exact_fd"] is False
    assert payload["boundary_chain_filter"]["require_growth_branch_locality"] is True
    assert payload["boundary_chain_filter"]["accepted_parameter_indices"] == [1]


def test_projected_writer_replay_command_honors_strict_qa_gate_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)

    class FakeOptimizer:
        def save_input(self, path, delta):
            del delta
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("! projected input\n", encoding="utf-8")

    fake_stage = SimpleNamespace(
        specs=[object(), object(), object(), object()], optimizer=FakeOptimizer()
    )
    monkeypatch.setattr(mod, "_build_stage", lambda _args: fake_stage)

    mod.main(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(gradient),
            "--boundary-chain-collection-json",
            str(collection),
            "--allow-boundary-chain-branch-sensitive",
            "--require-growth-branch-locality",
            "--outdir",
            str(tmp_path / "out"),
            "--steps",
            "1e-3",
            "--top-n",
            "4",
            "--target-aspect",
            "5.0",
            "--min-iota",
            "0.4102",
            "--iota-objective",
            "target",
            "--disable-iota-profile-floor",
            "--solved-wout-gate-min-abs-iota",
            "0.41",
            "--solved-wout-gate-aspect-atol",
            "0.02",
            "--solved-wout-gate-qs-max",
            "0.01",
            "--surface-chunk-size",
            "1",
            "--solver-device",
            "gpu",
            "--python-executable",
            "python3",
            "--save-rerun-wouts",
            "--require-rerun-wout-gate",
            "--admit-authoritative-rerun-wout",
        ]
    )

    payload = json.loads(
        (tmp_path / "out" / "projected_line_search_inputs.json").read_text(
            encoding="utf-8"
        )
    )
    command = payload["rows"][0]["replay_command"]
    assert command[0] == "python3"
    assert command[command.index("--target-aspect") + 1] == "5.0"
    assert command[command.index("--min-iota") + 1] == "0.4102"
    assert command[command.index("--iota-objective") + 1] == "target"
    assert "--disable-iota-profile-floor" in command
    assert "--iota-profile-floor" not in command
    assert command[command.index("--solved-wout-gate-min-abs-iota") + 1] == "0.41"
    assert command[command.index("--solved-wout-gate-aspect-atol") + 1] == "0.02"
    assert command[command.index("--solved-wout-gate-qs-max") + 1] == "0.01"
    assert command[command.index("--surface-chunk-size") + 1] == "1"
    assert command[command.index("--solver-device") + 1] == "gpu"
    assert "--save-rerun-wouts" in command
    assert "--require-rerun-wout-gate" in command
    assert "--admit-authoritative-rerun-wout" in command
    assert payload["boundary_chain_filter"]["accepted_parameter_indices"] == [1]


def test_projected_writer_fails_closed_for_underresolved_sample_set(
    tmp_path: Path, monkeypatch
) -> None:
    gradient = tmp_path / "gradient.json"
    collection = tmp_path / "boundary_chain.json"
    _gradient_report(gradient)
    _boundary_chain_collection(collection)

    def unexpected_stage(_args):
        raise AssertionError(
            "under-resolved sample set should fail before VMEC-JAX stage construction"
        )

    monkeypatch.setattr(mod, "_build_stage", unexpected_stage)
    with pytest.raises(
        ValueError, match="under-resolved transport objective sample set"
    ):
        mod.main(
            [
                "--input",
                str(tmp_path / "input.final"),
                "--gradient-json",
                str(gradient),
                "--boundary-chain-collection-json",
                str(collection),
                "--outdir",
                str(tmp_path / "out"),
                "--surfaces",
                "0.64",
                "--alphas",
                "0.0",
                "--ky-values",
                "0.3",
            ]
        )
