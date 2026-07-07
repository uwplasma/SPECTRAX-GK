from __future__ import annotations

import pytest

from tools.campaigns.nonlinear_replicate_followup import (
    NonlinearReplicateFollowupConfig,
    nonlinear_replicate_followup_plan,
)


def test_followup_plan_selects_cross_variants_for_mixed_spread() -> None:
    spread_report = {
        "summary": {"failed_states": ["plus_delta"]},
        "state_rows": [
            {
                "state": "plus_delta",
                "classification": "mixed_seed_timestep_spread",
                "high_variant_label": "seed32",
                "low_variant_label": "dt0p04",
            }
        ],
    }
    metadata = [
        {
            "state": "plus_delta",
            "variant_label": "seed31",
            "variant_axis": "seed",
            "seed": 31,
            "timestep": 0.05,
        },
        {
            "state": "plus_delta",
            "variant_label": "seed32",
            "variant_axis": "seed",
            "seed": 32,
            "timestep": 0.05,
        },
        {
            "state": "plus_delta",
            "variant_label": "dt0p04",
            "variant_axis": "timestep",
            "seed": 22,
            "timestep": 0.04,
        },
    ]

    plan = nonlinear_replicate_followup_plan(spread_report, variant_metadata=metadata)

    assert plan["passed"] is False
    assert plan["summary"]["planned_run_count"] == 3
    assert [row["variant_label"] for row in plan["planned_runs"]] == [
        "seed22_dt0p05",
        "seed32_dt0p04",
        "seed33_dt0p05",
    ]
    assert plan["planned_runs"][0]["reason"].startswith("test whether the low window")
    assert plan["missing_metadata"] == []


def test_followup_plan_records_missing_metadata_fail_closed() -> None:
    spread_report = {
        "summary": {"failed_states": ["plus_delta"]},
        "state_rows": [
            {
                "state": "plus_delta",
                "classification": "mixed_seed_timestep_spread",
                "high_variant_label": "seed32",
                "low_variant_label": "dt0p04",
            }
        ],
    }

    plan = nonlinear_replicate_followup_plan(spread_report, variant_metadata=[])

    assert plan["summary"]["planned_run_count"] == 0
    assert plan["summary"]["missing_metadata_count"] == 1
    assert (
        plan["missing_metadata"][0]["reason"]
        == "missing seed/timestep metadata for high or low variant"
    )


def test_followup_plan_covers_seed_and_timestep_limited_branches() -> None:
    spread_report = {
        "summary": {"failed_states": ["baseline", "plus_delta", "missing_state"]},
        "state_rows": [
            {"state": "baseline", "classification": "seed_spread_limited"},
            {"state": "plus_delta", "classification": "timestep_spread_limited"},
        ],
    }
    metadata = [
        {
            "state": "baseline",
            "label": "seed10",
            "axis": "seed",
            "seed": 10,
            "dt": 0.05,
            "source_config": "baseline_seed10.toml",
        },
        {
            "state": "baseline",
            "variant_label": "seed11",
            "variant_axis": "seed",
            "seed": 11,
            "timestep": 0.05,
        },
        {
            "state": "plus_delta",
            "variant_label": "dt0p04",
            "variant_axis": "timestep",
            "seed": 21,
            "timestep": 0.04,
        },
        {
            "state": "plus_delta",
            "variant_label": "bad_seed",
            "seed": -1,
            "timestep": 0.04,
        },
    ]

    plan = nonlinear_replicate_followup_plan(
        spread_report,
        variant_metadata=metadata,
        case="targeted_followup",
        config=NonlinearReplicateFollowupConfig(extra_seed_increment=3),
    )

    assert plan["case"] == "targeted_followup"
    assert plan["summary"]["planned_run_count"] == 2
    assert plan["summary"]["missing_metadata_count"] == 1
    assert [row["variant_label"] for row in plan["planned_runs"]] == [
        "seed14_dt0p05",
        "seed21_dt0p04",
    ]
    assert plan["missing_metadata"] == [
        {"state": "missing_state", "reason": "missing state row in spread report"}
    ]
    assert plan["config"]["extra_seed_increment"] == 3


def test_followup_plan_handles_empty_or_unrunnable_reports() -> None:
    assert (
        nonlinear_replicate_followup_plan({"summary": {}}, variant_metadata=[])[
            "passed"
        ]
        is True
    )

    unrunnable = nonlinear_replicate_followup_plan(
        {
            "summary": {"failed_states": ["baseline"]},
            "state_rows": [
                {"state": "baseline", "classification": "seed_spread_limited"}
            ],
        },
        variant_metadata=[],
    )

    assert unrunnable["passed"] is True
    assert unrunnable["state_plans"][0]["recommendation"].startswith(
        "No runnable follow-up"
    )


def test_followup_plan_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="max_runs_per_state"):
        nonlinear_replicate_followup_plan(
            {"summary": {}},
            variant_metadata=[],
            config=NonlinearReplicateFollowupConfig(max_runs_per_state=0),
        )

    with pytest.raises(ValueError, match="extra_seed_increment"):
        nonlinear_replicate_followup_plan(
            {"summary": {}},
            variant_metadata=[],
            config=NonlinearReplicateFollowupConfig(extra_seed_increment=0),
        )
