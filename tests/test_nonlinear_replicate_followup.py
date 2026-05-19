from __future__ import annotations

from spectraxgk.nonlinear_replicate_followup import nonlinear_replicate_followup_plan


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
    assert plan["missing_metadata"][0]["reason"] == "missing seed/timestep metadata for high or low variant"
