"""Metadata-only task summaries: no predictors, learners or numerical imports."""

from copy import deepcopy

import pytest

from src.app.reserved_probe_summary import (
    PHASES,
    TASKS,
    add_reserved_probe_summary,
    summarize_reserved_probe,
)
from src.core.probe_task_outcomes import task_outcome_changes


def recorded(phases=None):
    values = phases or (
        (True, True, False, False, False, True),
        (False, True, True, False, True, True),
        (True, True, True, True, True, False),
    )
    episodes = {}
    for phase, successes in zip(PHASES, values):
        episodes[phase] = [
            dict(
                phase=phase,
                policy_version=v,
                concept=c,
                role="controller_train",
                complete=True,
                episode=dict(
                    complete=True, success=success, scenario="ordinary", training_updates=0
                ),
            )
            for (v, c), success in zip(TASKS, successes)
        ]
    return dict(
        complete=False,
        error="original experiment failure",
        final_examples=0,
        reserved_probe_requested=True,
        training_work=dict(attempted_updates=12, attempted_rows=72, returned_updates=12),
        reserved_probe=dict(final_examples=0, episodes=episodes),
    )


@pytest.mark.parametrize(
    "before,after,returned,expected",
    [
        (True, False, True, (-1, False, False, True, 1, True, None)),
        (True, False, False, (-1, False, False, True, 0, False, None)),
        (True, True, True, (0, False, True, False, 0, None, True)),
        (False, True, True, (1, True, None, False, 0, None, True)),
        (False, False, True, (0, False, None, False, 1, None, None)),
        (True, None, True, (None, None, None, None, None, None, None)),
        (None, False, True, (None, None, None, None, 1, None, None)),
        (True, False, None, (-1, False, False, True, None, None, None)),
        (None, None, None, (None, None, None, None, None, None, None)),
    ],
)
def test_changes_require_observed_endpoints_and_loss(before, after, returned, expected):
    assert tuple(task_outcome_changes(before, after, returned).values()) == expected


@pytest.mark.parametrize("value", [0, 1, "true", [], {}])
def test_task_success_is_exact_bool_or_unknown(value):
    with pytest.raises(ValueError, match="exact boolean"):
        task_outcome_changes(value, True, True)


def test_fixed_cohorts_distinguish_acquisition_retention_and_conditional_recovery():
    summary = summarize_reserved_probe(recorded())
    assert summary["status"] == "complete"
    old, fresh = (summary["cohorts"][name] for name in ("source_trained", "fresh"))
    assert old["successes_by_phase"] == dict(zip(PHASES, (2, 2, 4)))
    assert fresh["successes_by_phase"] == dict(zip(PHASES, (1, 2, 1)))
    assert old["task_changes"]["v1:invoice"]["recovered_on_return"] is True
    assert old["task_changes"]["v2:parcel"]["recovered_on_return"] is None
    assert old["task_changes"]["v2:invoice"]["acquired_during_probe"] is True
    assert fresh["task_changes"]["v1:parcel"]["acquired_during_probe"] is True
    assert fresh["task_changes"]["v2:password"]["acquired_during_probe"] is False
    assert fresh["task_changes"]["v2:password"]["return_success_change"] == -1


@pytest.mark.parametrize("kind", ["missing", "empty", "prefix", "incomplete"])
def test_incomplete_phase_keeps_prefix_but_never_aggregates_successes(kind):
    report = recorded()
    episodes = report["reserved_probe"]["episodes"]
    if kind == "missing":
        del episodes["after_probe"]
    elif kind == "empty":
        episodes["after_probe"] = []
    elif kind == "prefix":
        episodes["after_probe"] = episodes["after_probe"][:1]
    else:
        episodes["after_probe"][-1]["complete"] = False
    result = summarize_reserved_probe(report)
    assert result["status"] == "partial"
    assert result["cohorts"]["source_trained"]["successes_by_phase"]["after_probe"] is None
    change = result["cohorts"]["source_trained"]["task_changes"]["v1:invoice"]
    assert change["probe_success_change"] is None
    assert change["recovered_on_return"] is None
    assert result["phases"]["before_probe"]["status"] == "complete"
    if kind in ("prefix", "incomplete"):
        assert result["phases"]["after_probe"]["observed_success"]["v1:invoice"] is False


def test_missing_return_does_not_erase_completed_probe_comparison():
    report = recorded()
    del report["reserved_probe"]["episodes"]["after_return"]
    summary = summarize_reserved_probe(report)
    changes = summary["cohorts"]["source_trained"]["task_changes"]["v1:invoice"]
    assert changes["lost_during_probe"] is True
    assert changes["probe_success_change"] == -1
    assert changes["recovered_on_return"] is None


@pytest.mark.parametrize("role", ["final", "probe_test", "validation", None])
def test_all_phase_role_headers_are_admitted_before_any_outcome_access(role):
    report = recorded()
    phases = report["reserved_probe"]["episodes"]
    phases["before_probe"][0]["episode"] = object()
    phases["after_return"][-1]["role"] = role
    with pytest.raises(ValueError, match="header/role/order"):
        summarize_reserved_probe(report)


@pytest.mark.parametrize(
    "kind",
    ["duplicate", "order", "mixed", "extra", "unknown_phase", "missing_header", "outer_complete"],
)
def test_refuses_changed_cohorts_order_bounds_and_headers(kind):
    report = recorded()
    phases = report["reserved_probe"]["episodes"]
    rows = phases["before_probe"]
    if kind == "duplicate":
        rows[1] = deepcopy(rows[0])
    elif kind == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif kind == "mixed":
        rows[0]["concept"] = "parcel"
    elif kind == "extra":
        rows.append(deepcopy(rows[0]))
    elif kind == "unknown_phase":
        phases["test"] = []
    elif kind == "missing_header":
        del rows[0]["phase"]
    else:
        rows[0]["complete"] = 1
    with pytest.raises(ValueError):
        summarize_reserved_probe(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", 1),
        ("success", "false"),
        ("complete", False),
        ("scenario", "training"),
        ("training_updates", 1),
        ("training_updates", False),
    ],
)
def test_complete_episode_requires_exact_observed_ordinary_zero_training(field, value):
    report = recorded()
    report["reserved_probe"]["episodes"]["before_probe"][0]["episode"][field] = value
    with pytest.raises(ValueError, match="zero-training"):
        summarize_reserved_probe(report)


@pytest.mark.parametrize("where,value", [("source", 1), ("private", 1), ("source", False)])
def test_refuses_final_evaluation_before_outcomes(where, value):
    report = recorded()
    record = report if where == "source" else report["reserved_probe"]
    record["final_examples"] = value
    report["reserved_probe"]["episodes"] = object()
    with pytest.raises(ValueError, match="final evaluation"):
        summarize_reserved_probe(report)


def test_preserves_paid_source_auxiliary_private_phase_and_readout_counts_separately():
    report = recorded()
    report["probe_initialization"] = dict(
        complete=False, preserved_fit_work=dict(closure_attempts=34, backward_rows=408)
    )
    private = report["reserved_probe"]
    private["preserved_training_work"] = dict(
        attempted_updates=32, attempted_rows=144, returned_updates=31
    )
    private["probe"] = dict(
        attempted_updates=16,
        completed_updates=15,
        attempted_training_examples=48,
        return_phase=dict(attempted_updates=16, attempted_training_examples=96),
    )
    private["task_readout_work"] = dict(
        before_probe=dict(decision_attempts=36, native_forward_attempts=36)
    )
    costs = summarize_reserved_probe(report)["reported_costs"]
    assert costs["source_training"]["attempted_rows"] == 72
    assert costs["auxiliary_fit"]["backward_rows"] == 408
    assert costs["private_training"]["returned_updates"] == 31
    assert costs["fresh_training"]["attempted_training_examples"] == 48
    assert costs["return_training"]["attempted_training_examples"] == 96
    assert costs["task_readouts"]["before_probe"]["decision_attempts"] == 36
    assert costs["task_readouts"]["after_probe"]["native_forward_attempts"] is None
    assert costs["auxiliary_fit"]["optimizer_constructions"] is None


@pytest.mark.parametrize("value", [True, -1, 2**53, "12"])
def test_work_counts_refuse_malformed_values(value):
    report = recorded()
    report["training_work"]["attempted_updates"] = value
    with pytest.raises(ValueError, match="bounded nonnegative"):
        summarize_reserved_probe(report)


def test_all_failure_baseline_has_no_invented_acquisition_or_recovery():
    summary = summarize_reserved_probe(recorded(((False,) * 6,) * 3))
    assert summary["status"] == "complete"
    for group in summary["cohorts"].values():
        assert list(group["successes_by_phase"].values()) == [0, 0, 0]
        for change in group["task_changes"].values():
            assert change["acquired_during_probe"] is False
            assert change["retained_through_probe"] is None
            assert change["lost_during_probe"] is False
            assert change["recovered_on_return"] is None


def test_failed_before_probe_stays_unknown_with_source_costs_preserved():
    report = recorded()
    report["reserved_probe"] = {}
    summary = summarize_reserved_probe(report)
    assert summary["status"] == "not_run"
    assert summary["cohorts"]["source_trained"]["successes_by_phase"] == dict.fromkeys(PHASES)
    assert summary["reported_costs"]["source_training"]["attempted_updates"] == 12
    assert summary["reported_costs"]["fresh_training"]["attempted_updates"] is None


@pytest.mark.parametrize("invalid", [False, True])
def test_appending_summary_preserves_original_raw_failure_and_work(invalid):
    report = recorded()
    if invalid:
        report["reserved_probe"]["episodes"]["after_return"][0]["role"] = "final"
    original = deepcopy(report)
    add_reserved_probe_summary(report)
    result = report.pop("reserved_probe_summary")
    assert report == original
    assert result["status"] == ("invalid" if invalid else "complete")
    if not invalid:
        result["phases"]["before_probe"]["observed_success"]["v1:invoice"] = False
        result["reported_costs"]["source_training"]["attempted_updates"] = 999
        assert report == original


def test_refuses_summary_without_original_explicit_request():
    report = recorded()
    report["reserved_probe_requested"] = False
    with pytest.raises(ValueError, match="explicitly requested"):
        summarize_reserved_probe(report)
