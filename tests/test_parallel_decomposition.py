from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectraxgk.parallel.decomposition import (
    DecompositionContract,
    ReconstructionIdentityReport,
    ShardAssignment,
    build_diagnostic_nonlinear_domain_decomposition,
    build_independent_portfolio_decomposition,
    reconstruct_serial,
    serial_reconstruction_identity_report,
    shard_sequence,
)
from tools.artifacts.build_parallel_decomposition_status import (
    build_status,
    write_csv_artifact,
    write_json_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_ky_decomposition_is_deterministic_balanced_and_ordered() -> None:
    first = build_independent_portfolio_decomposition(
        7,
        requested_shards=3,
        workload="independent_ky_scan",
    )
    second = build_independent_portfolio_decomposition(
        7,
        requested_shards=3,
        workload="independent_ky_scan",
    )

    assert first == second
    assert first.production_independent_batching is True
    assert first.diagnostic_nonlinear_partition is False
    assert first.independent_work is True
    assert first.changes_solver_layout is False
    assert first.actual_shards == 3
    assert [shard.indices for shard in first.shards] == [
        (0, 1, 2),
        (3, 4),
        (5, 6),
    ]
    assert [shard.size for shard in first.shards] == [3, 2, 2]
    assert [shard.start for shard in first.shards] == [0, 3, 5]
    assert [shard.stop for shard in first.shards] == [3, 5, 7]
    assert "production independent batching" in first.claim_label
    assert "not a nonlinear state-domain decomposition speedup claim" in first.claim_label
    assert first.to_dict()["workload"] == "independent_ky_scan"
    assert first.shards[0].to_dict()["label"].startswith(
        "independent_ky_scan:shard_000"
    )


def test_uq_decomposition_reconstructs_serial_identity() -> None:
    values = tuple(f"member-{idx}" for idx in range(8))
    contract = build_independent_portfolio_decomposition(
        len(values),
        requested_shards=4,
        workload="uq_ensemble",
    )

    shards = shard_sequence(values, contract)
    reconstructed = reconstruct_serial(contract, shards)
    report = serial_reconstruction_identity_report(values, contract)

    assert shards == (
        ("member-0", "member-1"),
        ("member-2", "member-3"),
        ("member-4", "member-5"),
        ("member-6", "member-7"),
    )
    assert reconstructed == values
    assert report == ReconstructionIdentityReport(
        workload="uq_ensemble",
        claim_level="production_independent_batching",
        claim_label=contract.claim_label,
        n_items=8,
        requested_shards=4,
        actual_shards=4,
        identity_passed=True,
        expected_indices=tuple(range(8)),
        reconstructed_indices=tuple(range(8)),
        missing_indices=(),
        duplicate_indices=(),
        out_of_range_indices=(),
        out_of_order=False,
    )
    assert report.to_dict()["identity_passed"] is True


def test_optimization_ensemble_decomposition_uses_production_independent_contract() -> None:
    values = tuple({"candidate": idx, "objective": idx * idx} for idx in range(5))
    contract = build_independent_portfolio_decomposition(
        len(values),
        requested_shards=8,
        workload="optimization_ensemble",
    )
    report = serial_reconstruction_identity_report(values, contract)

    assert contract.workload == "optimization_ensemble"
    assert contract.claim_level == "production_independent_batching"
    assert contract.actual_shards == 5
    assert contract.independent_work is True
    assert contract.changes_solver_layout is False
    assert "independent optimization ensemble" in contract.claim_label
    assert "not a nonlinear state-domain decomposition" in contract.claim_label
    assert report.identity_passed is True
    assert reconstruct_serial(contract, shard_sequence(values, contract)) == values


def test_decomposition_handles_empty_and_oversharded_portfolios_without_empty_shards() -> None:
    empty = build_independent_portfolio_decomposition(
        0,
        requested_shards=4,
        workload="uq_ensemble",
    )
    oversharded = build_independent_portfolio_decomposition(
        3,
        requested_shards=8,
        workload="independent_ky_scan",
    )

    assert empty.actual_shards == 0
    assert empty.shards == ()
    assert serial_reconstruction_identity_report((), empty).identity_passed is True
    assert oversharded.actual_shards == 3
    assert [shard.indices for shard in oversharded.shards] == [(0,), (1,), (2,)]
    assert all(shard.size == 1 for shard in oversharded.shards)
    assert reconstruct_serial(oversharded, shard_sequence(("a", "b", "c"), oversharded)) == (
        "a",
        "b",
        "c",
    )


def test_decomposition_rejects_invalid_counts_workloads_and_mismatched_values() -> None:
    with pytest.raises(ValueError, match="requested_shards"):
        build_independent_portfolio_decomposition(
            3,
            requested_shards=0,
            workload="independent_ky_scan",
        )
    with pytest.raises(ValueError, match="n_items"):
        build_independent_portfolio_decomposition(
            -1,
            requested_shards=1,
            workload="uq_ensemble",
        )
    with pytest.raises(ValueError, match="workload"):
        build_independent_portfolio_decomposition(
            3,
            requested_shards=1,
            workload="diagnostic_nonlinear_domain",  # type: ignore[arg-type]
        )

    contract = build_independent_portfolio_decomposition(
        3,
        requested_shards=2,
        workload="uq_ensemble",
    )
    with pytest.raises(ValueError, match="values length"):
        shard_sequence(("only-one",), contract)
    with pytest.raises(ValueError, match="actual_shards"):
        reconstruct_serial(contract, (("a", "b"),))
    with pytest.raises(ValueError, match="assignment size"):
        reconstruct_serial(contract, (("a",), ("b",)))


def test_diagnostic_nonlinear_domain_decomposition_is_split_reassemble_only() -> None:
    contract = build_diagnostic_nonlinear_domain_decomposition(
        (4, 6, 2),
        axis=-2,
        requested_shards=4,
    )
    values = tuple(range(contract.n_items))
    report = serial_reconstruction_identity_report(values, contract)

    assert contract.workload == "diagnostic_nonlinear_domain"
    assert contract.claim_level == "diagnostic_nonlinear_domain_partition"
    assert contract.production_independent_batching is False
    assert contract.diagnostic_nonlinear_partition is True
    assert contract.independent_work is False
    assert contract.changes_solver_layout is True
    assert contract.state_shape == (4, 6, 2)
    assert contract.axis == 1
    assert contract.actual_shards == 4
    assert [shard.indices for shard in contract.shards] == [
        (0, 1),
        (2, 3),
        (4,),
        (5,),
    ]
    assert all("axis_1" in shard.label for shard in contract.shards)
    assert report.identity_passed is True
    assert "diagnostic nonlinear state-domain partition" in report.claim_label
    assert "no production routing or speedup claim" in report.claim_label


def test_diagnostic_nonlinear_domain_decomposition_validates_shape_and_shards() -> None:
    oversharded = build_diagnostic_nonlinear_domain_decomposition(
        (2, 3),
        axis=1,
        requested_shards=10,
    )

    assert oversharded.actual_shards == 3
    assert [shard.indices for shard in oversharded.shards] == [(0,), (1,), (2,)]
    with pytest.raises(ValueError, match="at least one axis"):
        build_diagnostic_nonlinear_domain_decomposition(
            (),
            axis=0,
            requested_shards=1,
        )
    with pytest.raises(ValueError, match="positive"):
        build_diagnostic_nonlinear_domain_decomposition(
            (2, 0),
            axis=0,
            requested_shards=1,
        )
    with pytest.raises(ValueError, match="requested_shards"):
        build_diagnostic_nonlinear_domain_decomposition(
            (2, 3),
            axis=0,
            requested_shards=0,
        )


def test_claim_levels_separate_production_batches_from_diagnostic_domain_partitions() -> None:
    ky = build_independent_portfolio_decomposition(
        5,
        requested_shards=2,
        workload="independent_ky_scan",
    )
    uq = build_independent_portfolio_decomposition(
        5,
        requested_shards=2,
        workload="uq_ensemble",
    )
    nonlinear = build_diagnostic_nonlinear_domain_decomposition(
        (5, 4),
        axis=0,
        requested_shards=2,
    )

    assert {ky.claim_level, uq.claim_level} == {"production_independent_batching"}
    assert nonlinear.claim_level == "diagnostic_nonlinear_domain_partition"
    assert ky.independent_work and uq.independent_work
    assert not nonlinear.independent_work
    assert not ky.changes_solver_layout
    assert nonlinear.changes_solver_layout
    assert "not a nonlinear state-domain decomposition" in ky.claim_label
    assert "not a nonlinear state-domain decomposition" in uq.claim_label
    assert "no production routing or speedup claim" in nonlinear.claim_label


def test_manual_bad_assignment_report_can_expose_claim_scoped_identity_failure() -> None:
    bad_contract = DecompositionContract(
        workload="diagnostic_nonlinear_domain",
        claim_level="diagnostic_nonlinear_domain_partition",
        claim_label="diagnostic nonlinear state-domain partition contract",
        n_items=3,
        requested_shards=2,
        actual_shards=2,
        shards=(
            ShardAssignment(
                shard_id=0,
                start=0,
                stop=2,
                indices=(0, 2),
                label="bad:0",
            ),
            ShardAssignment(
                shard_id=1,
                start=2,
                stop=3,
                indices=(1,),
                label="bad:1",
            ),
        ),
        independent_work=False,
        changes_solver_layout=True,
    )
    report = serial_reconstruction_identity_report(("a", "b", "c"), bad_contract)

    assert report.identity_passed is False
    assert report.missing_indices == ()
    assert report.duplicate_indices == ()
    assert report.out_of_range_indices == ()
    assert report.out_of_order is True
    assert report.reconstructed_indices == (0, 2, 1)


def test_parallel_decomposition_status_summarizes_existing_artifacts(
    tmp_path: Path,
) -> None:
    status = build_status(ROOT)
    lanes = {lane["lane"]: lane for lane in status["lanes"]}

    assert status["kind"] == "parallel_decomposition_status"
    assert status["passed"] is True
    assert status["production_independent_lanes"] == 2
    assert status["diagnostic_nonlinear_lanes"] == 1
    assert "Deterministic decomposition-contract status only" in status["claim_scope"]
    assert lanes["independent_ky_scan"]["claim_level"] == "production_independent_batching"
    assert lanes["uq_ensemble"]["claim_level"] == "production_independent_batching"
    assert (
        lanes["diagnostic_nonlinear_domain"]["claim_level"]
        == "diagnostic_nonlinear_domain_partition"
    )
    assert all(lane["reconstruction_identity_passed"] for lane in lanes.values())
    assert all(lane["claim_separation_passed"] for lane in lanes.values())

    prefix = tmp_path / "parallel_decomposition_status"
    paths = {
        **write_json_artifact(status, prefix),
        **write_csv_artifact(status, prefix),
    }

    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["passed"] is True
    assert "claim_level" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_parallel_decomposition_contracts_are_exported_at_package_top_level() -> None:
    import spectraxgk as sgk

    contract = sgk.build_independent_portfolio_decomposition(
        2,
        requested_shards=2,
        workload="independent_ky_scan",
    )

    assert isinstance(contract, sgk.DecompositionContract)
    assert sgk.shard_sequence(("a", "b"), contract) == (("a",), ("b",))
