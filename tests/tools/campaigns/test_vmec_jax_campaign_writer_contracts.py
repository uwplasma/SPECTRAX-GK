from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

from support.paths import REPO_ROOT, load_campaign_tool


ROOT = REPO_ROOT
OPTIMIZER_SCRIPT = (
    ROOT / "tools" / "campaigns" / "write_vmec_jax_optimizer_comparison_manifest.py"
)
SPSA_SCRIPT = ROOT / "tools" / "campaigns" / "write_vmec_jax_spsa_candidate_campaign.py"
optimizer_mod = load_campaign_tool("write_vmec_jax_optimizer_comparison_manifest")
projected_mod = load_campaign_tool("write_vmec_jax_projected_transport_line_search_inputs")
spsa_mod = load_campaign_tool("write_vmec_jax_spsa_candidate_campaign")


def _entry(payload: dict[str, object], case_id: str) -> dict[str, object]:
    entries = payload["entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry.get("case_id") == case_id:
            return entry
    raise AssertionError(f"missing entry {case_id}")


def test_optimizer_comparison_manifest_builds_matched_runnable_commands(
    tmp_path: Path,
) -> None:
    args = optimizer_mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(tmp_path / "manifest.json"),
            "--transport-kinds",
            "growth,quasilinear_flux",
            "--runnable-methods",
            "scipy,scalar_trust",
            "--outer-loop-methods",
            "spsa,bo",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = optimizer_mod.build_manifest(args)

    assert payload["kind"] == "vmec_jax_qa_optimizer_comparison_manifest"
    assert payload["comparison_fingerprint"]
    assert payload["comparison_policy"]["sample_policy"]["mboz"] == 21
    assert payload["comparison_policy"]["sample_policy"]["nboz"] == 21
    assert payload["comparison_policy"]["sample_policy"]["surfaces"] == [0.64]
    assert (
        payload["comparison_policy"]["landscape_policy"][
            "rbc11_points_admit_optimized_candidates"
        ]
        is False
    )
    assert (
        "simple seed"
        in payload["comparison_policy"]["landscape_policy"]["optimization_seed_policy"]
    )
    ladder = payload["comparison_policy"]["optimizer_ladder_policy"]
    assert ladder[1]["stage"] == "linear_quasilinear_transport"
    assert ladder[3]["stage"] == "quasilinear_screening_refit"
    assert len(payload["runnable_commands"]) == 5

    baseline = _entry(payload, "qa_baseline_scipy")
    assert baseline["status"] == "runnable"
    baseline_parts = shlex.split(str(baseline["command"]))
    assert "--strict-upstream-qa-baseline" in baseline_parts
    assert "--admit-authoritative-rerun-wout" in baseline_parts
    assert "--allow-failed-solved-wout-gate" not in baseline_parts
    assert baseline["expected_authoritative_wout"].endswith(
        "runs/qa_baseline_scipy/wout_final_rerun.nc"
    )

    growth_scipy = _entry(payload, "growth_scipy_from_strict_baseline")
    growth_scalar = _entry(payload, "growth_scalar_trust_from_strict_baseline")
    for entry in (growth_scipy, growth_scalar):
        command = str(entry["command"])
        parts = shlex.split(command)
        assert parts[0:2] == [
            "python3",
            "tools/campaigns/vmec_jax_qa_low_turbulence_optimization.py",
        ]
        assert parts[parts.index("--input") + 1].endswith(
            "runs/qa_baseline_scipy/input.final"
        )
        assert parts[parts.index("--transport-kind") + 1] == "growth"
        assert parts[parts.index("--surfaces") + 1] == "0.64"
        assert parts[parts.index("--alphas") + 1] == "0"
        assert parts[parts.index("--ky-values") + 1] == "0.3"
        assert parts[parts.index("--mboz") + 1] == "21"
        assert parts[parts.index("--nboz") + 1] == "21"
        assert "--save-rerun-wouts" in parts
        assert "--admit-authoritative-rerun-wout" in parts
        assert "--allow-failed-solved-wout-gate" in parts
        assert "--make-plots" in parts
        audit = str(entry["recommended_nonlinear_audit_command"])
        assert audit.startswith(
            "python3 tools/campaigns/write_optimized_equilibrium_transport_configs.py"
        )
        assert "--horizons 700,1100,1500" in audit
        assert "--window-tmin 1100 --window-tmax 1500" in audit
        assert "--seed-variant 41 --seed-variant 42" in audit
        assert "--dt-variant 0.04" in audit
        strategy = entry["optimizer_strategy"]
        assert strategy["stage"] == "linear_quasilinear_continuation_multistart"
        assert strategy["uses_transport_weight_continuation"] is True
        assert strategy["uses_multistart_from_simple_seed_qa_solves"] is True
        assert (
            strategy["rbc_landscape_role"] == "conditioning_and_noise_diagnostic_only"
        )

    scipy_parts = shlex.split(str(growth_scipy["command"]))
    assert scipy_parts[scipy_parts.index("--method") + 1] == "scipy"
    assert scipy_parts[scipy_parts.index("--scipy-tr-solver") + 1] == "lsmr"
    assert scipy_parts[scipy_parts.index("--scipy-lsmr-maxiter") + 1] == "200"

    scalar_parts = shlex.split(str(growth_scalar["command"]))
    assert scalar_parts[scalar_parts.index("--method") + 1] == "scalar_trust"


def test_optimizer_comparison_manifest_marks_spsa_cma_bo_as_outer_loop_contracts(
    tmp_path: Path,
) -> None:
    args = optimizer_mod.parse_args(
        [
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(tmp_path / "manifest.json"),
            "--transport-kinds",
            "nonlinear_window_heat_flux",
            "--runnable-methods",
            "lbfgs_adjoint",
            "--outer-loop-methods",
            "spsa,cma_es,bo",
        ]
    )

    payload = optimizer_mod.build_manifest(args)

    methods = {"spsa", "cma_es", "bo"}
    for method in methods:
        entry = _entry(payload, f"nonlinear_window_heat_flux_{method}_outer_loop")
        assert entry["status"] == "planned_outer_loop"
        assert entry["candidate_generator_required"] is True
        contract = entry["candidate_contract"]
        strategy = entry["optimizer_strategy"]
        assert isinstance(contract, dict)
        assert strategy["requires_common_random_numbers"] is True
        assert strategy["requires_matched_t1500_replicated_audit"] is True
        assert (
            strategy["rbc_landscape_role"] == "conditioning_and_noise_diagnostic_only"
        )
        metric_command = str(contract["metric_eval_command_template"])
        audit_command = str(contract["nonlinear_audit_command_template"])
        spsa_command = contract["spsa_candidate_campaign_command"]
        if method == "spsa":
            assert isinstance(spsa_command, str)
            assert spsa_command.startswith(
                "python3 tools/campaigns/write_vmec_jax_spsa_candidate_campaign.py"
            )
            assert "--transport-kind nonlinear_window_heat_flux" in spsa_command
            assert "--audit-seed-variant 32 --audit-seed-variant 33" in spsa_command
        else:
            assert spsa_command is None
        assert metric_command.startswith(
            "python3 tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py"
        )
        assert audit_command.startswith(
            "python3 tools/campaigns/write_optimized_equilibrium_transport_configs.py"
        )
        assert (
            "tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py"
            in metric_command
        )
        assert "{candidate_input_final}" in metric_command
        assert "{candidate_id}" in metric_command
        assert "--transport-kind nonlinear_window_heat_flux" in metric_command
        assert (
            "tools/campaigns/write_optimized_equilibrium_transport_configs.py"
            in audit_command
        )
        assert "{candidate_id}" in audit_command
        assert "--window-tmin 1100 --window-tmax 1500" in audit_command


def test_optimizer_comparison_manifest_cli_writes_json(tmp_path: Path) -> None:
    out_json = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(OPTIMIZER_SCRIPT),
            "--campaign-root",
            str(tmp_path / "campaign"),
            "--out-json",
            str(out_json),
            "--transport-kinds",
            "growth",
            "--runnable-methods",
            "lbfgs_adjoint",
            "--outer-loop-methods",
            "bo",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert status["out_json"] == str(out_json)
    assert status["comparison_fingerprint"] == payload["comparison_fingerprint"]
    assert (
        _entry(payload, "growth_lbfgs_adjoint_from_strict_baseline")["status"]
        == "runnable"
    )
    assert _entry(payload, "growth_bo_outer_loop")["status"] == "planned_outer_loop"



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
    args = projected_mod._parse_args(
        [
            "--input",
            str(tmp_path / "input.final"),
            "--gradient-json",
            str(tmp_path / "gradient.json"),
            "--outdir",
            str(tmp_path / "out"),
        ]
    )

    sample_set = projected_mod._sample_set_from_args(args)
    summary = projected_mod.transport_objective_sample_summary(sample_set)

    assert args.surfaces == projected_mod.DEFAULT_TRANSPORT_SURFACES
    assert args.alphas == projected_mod.DEFAULT_TRANSPORT_ALPHAS
    assert args.ky_values == projected_mod.DEFAULT_TRANSPORT_KY_VALUES
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
    monkeypatch.setattr(projected_mod, "_build_stage", lambda _args: fake_stage)

    rc = projected_mod.main(
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

    monkeypatch.setattr(projected_mod, "_build_stage", unexpected_stage)
    with pytest.raises(ValueError, match="require --boundary-chain-collection-json"):
        projected_mod.main(
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
    monkeypatch.setattr(projected_mod, "_build_stage", lambda _args: fake_stage)

    rc = projected_mod.main(
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
    monkeypatch.setattr(projected_mod, "_build_stage", lambda _args: fake_stage)

    projected_mod.main(
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
    monkeypatch.setattr(projected_mod, "_build_stage", lambda _args: fake_stage)

    projected_mod.main(
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
    monkeypatch.setattr(projected_mod, "_build_stage", lambda _args: fake_stage)

    projected_mod.main(
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

    monkeypatch.setattr(projected_mod, "_build_stage", unexpected_stage)
    with pytest.raises(
        ValueError, match="under-resolved transport objective sample set"
    ):
        projected_mod.main(
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



def _write_input(path: Path) -> None:
    path.write_text(
        """
&INDATA
  RBC(0,0) = 1.0000000000000000E+00
  ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
  RBC(1,1) = 1.0000000000000000E-01
/
""".lstrip(),
        encoding="utf-8",
    )


def test_spsa_candidate_campaign_writes_plus_minus_common_random_number_commands(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    args = spsa_mod.parse_args(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);ZBS(1,1);RBC(1,1)",
            "--iterations",
            "2",
            "--seed",
            "123",
            "--relative-delta",
            "0.1",
            "--audit-seed-variant",
            "41",
            "--audit-seed-variant",
            "42",
        ]
    )

    payload = spsa_mod.build_campaign(args)

    assert payload["kind"] == "vmec_jax_spsa_transport_candidate_campaign"
    assert payload["controls"] == ["ZBS(1,0)", "ZBS(1,1)", "RBC(1,1)"]
    assert payload["common_random_number_policy"]["audit_seed_variants"] == [41, 42]
    assert len(payload["pairs"]) == 2
    first = payload["pairs"][0]
    assert set(first["states"]) == {"plus", "minus"}
    assert first["states"]["plus"]["input"].endswith("iter_000/plus/input.final")
    plus_text = (out_dir / "iter_000" / "plus" / "input.final").read_text(
        encoding="utf-8"
    )
    minus_text = (out_dir / "iter_000" / "minus" / "input.final").read_text(
        encoding="utf-8"
    )
    assert plus_text != minus_text
    assert "RBC(1,1)" in plus_text
    metric_command = first["states"]["plus"]["metric_eval_command"]
    metric_parts = shlex.split(metric_command)
    assert metric_parts[:2] == [
        "python3",
        "tools/campaigns/evaluate_vmec_jax_spectrax_transport_metric.py",
    ]
    assert (
        metric_parts[metric_parts.index("--transport-kind") + 1]
        == "nonlinear_window_heat_flux"
    )
    assert metric_parts[metric_parts.index("--mboz") + 1] == "21"
    audit_command = first["states"]["plus"]["nonlinear_audit_command"]
    assert "--seed-variant 41 --seed-variant 42" in audit_command
    assert "--window-tmin 1100 --window-tmax 1500" in audit_command
    assert "dJ/dx_i" in first["gradient_estimator"]["RBC(1,1)"]
    manifest = json.loads(
        (out_dir / "vmec_jax_spsa_candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["claim_scope"].startswith(
        "SPSA common-random-number candidate generation"
    )


def test_spsa_candidate_campaign_cli_writes_manifest(tmp_path: Path) -> None:
    baseline = tmp_path / "input.final"
    out_dir = tmp_path / "spsa"
    _write_input(baseline)

    completed = subprocess.run(
        [
            sys.executable,
            str(SPSA_SCRIPT),
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--controls",
            "ZBS(1,0);RBC(1,1)",
            "--iterations",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    assert status["pairs"] == 1
    assert status["controls"] == ["ZBS(1,0)", "RBC(1,1)"]
    assert (out_dir / "vmec_jax_spsa_candidate_manifest.json").exists()

