"""Saved-cohort replay comparison admission using nonnumeric callback stubs.

One accepted saved JSON cohort is parsed once. No models, arrays, fits, native
updates, predictions, gradients, RNG or simulations are created in these tests.
Synthetic replay reports exercise validation and signed summaries, not science.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import continuous_replay_example as study
from src.app.toy_execution_budget import ToyExecutionBudget


FAMILIES = ("bp", "pc", "neutral-cpc", "active-cpc")


@pytest.fixture(scope="module")
def saved():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/continuous-probe-continuation-20261020/remaining-arms.log"
    )
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf8"))
    assert value["complete"] is True and set(value["reports"]) == set(FAMILIES)
    return value, payload


def assert_unchanged_saved_bytes(payload):
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/continuous-probe-continuation-20261020/remaining-arms.log"
    )
    assert path.read_bytes() == payload


def replay_result(baseline, family):
    """Shape a consistent metadata-only zero-delta outcome from saved scalars."""
    report = deepcopy(baseline["reports"][family])
    final_rows = (0, 4, 9, 13)
    for index, point in enumerate(report["checkpoints"]):
        point.update(
            updates=2 * index,
            native_updates=2 * index,
            wake_updates=index,
            training_examples=final_rows[index],
        )
    for index, event in enumerate(report["events"]):
        event.update(candidate_updates=2 * index + 1, candidate_training_examples=(2, 7, 11)[index])
    for index, point in enumerate(report["probes"]):
        point.update(benchmark_updates=2 * index, benchmark_wake_updates=index)
        pair = point["report"]
        pair.update(source_updates=2 * index, source_training_examples=final_rows[index])
        for name in ("inherited", "zero_velocity"):
            nested = pair[name]
            nested.update(source_updates=2 * index, source_training_examples=final_rows[index])
            for offset, checkpoint in enumerate(nested["checkpoints"]):
                checkpoint["updates"] = 2 * index + offset
    events = []
    for index, applied in enumerate(report["applied_updates"], 1):
        receipt = dict(
            event_id=f"continuous-replay-{index}",
            actor_version="actor-0",
            learner_version="candidate-0",
            attempt_number=index,
            diagnostic=deepcopy(applied["diagnostic"]),
        )
        events.append(
            dict(
                event_id=receipt["event_id"],
                wake_updates=index,
                status="complete",
                attempted_updates=1,
                returned_updates=1,
                committed_updates=1,
                charged_exposures=2,
                diagnostics=[deepcopy(applied["diagnostic"])],
                receipt=receipt,
                error=None,
                native_updates_after=2 * index,
                native_training_examples_after=final_rows[index],
            )
        )
    report["replay"] = dict(
        plan=dict(
            every_updates=1,
            max_replay_examples=6,
            max_retained_rows=128,
            max_retained_bytes=1048576,
        ),
        planned_events=3,
        wake_updates_observed=3,
        total_updates=3,
        bank_episode_ids=[report["source_episodes"]["warm"]["episode_id"]],
        bank_contexts=[report["source_episodes"]["warm"]["context"]],
        bank_rows=2,
        retained_numeric_bytes=112,
        retained_numeric_scope="obs+next_obs+actions+reward float64 payload; containers/copies/native arrays/RSS additional",
        charged_exposures=6,
        charge_count_unknown=False,
        stopped=False,
        reason=None,
        events=events,
        original_limits=dict(
            wake=asdict(ToyExecutionBudget(max_training_updates=3)),
            replay=asdict(ToyExecutionBudget(max_replay_examples=6)),
        ),
        attempted_updates=3,
        returned_updates=3,
        committed_updates=3,
        observed_budget_exposures=6,
    )
    report["applied_consolidations"] = [deepcopy(event["receipt"]) for event in events]
    report["progress"].update(
        native_updates_completed=6, replay_updates_completed=3, replay_rows_charged=6
    )
    work = report["work"]
    if family == "bp":
        for key in (
            "optimizer_updates",
            "normalization_calls",
            "forward_passes",
            "backward_passes",
        ):
            work[key] += 3
        for key in (
            "training_rows",
            "repeated_training_rows",
            "normalization_rows",
            "forward_rows",
            "backward_rows",
        ):
            work[key] += 6
        work["normalization_values"] += 12
        work["gradient_values"] += 966
        work["gradient_numeric_bytes"] += 7728
        work["dense_multiply_accumulates"] += 3840
    else:
        report["native_updates"] = [
            deepcopy(record) for record in report["native_updates"] for _ in range(2)
        ]
        for record, rows in zip(report["native_updates"], (2, 2, 3, 2, 2, 2), strict=True):
            record["training_rows"] = rows
        work.update(optimizer_updates=6, training_rows=13, repeated_training_rows=8)
        work["native_update_counts"] = {
            key: sum(record["work"][key] for record in report["native_updates"])
            for key in work["native_update_counts"]
        }
        work["native_event_row_weights"] = {
            key: sum(
                record["work"][key] * record["training_rows"] for record in report["native_updates"]
            )
            for key in work["native_event_row_weights"]
        }
    work.update(
        replay_budget_rows=6,
        replay_optimizer_updates=3,
        replay_retained_numeric_bytes=112,
        replay_input_feature_target_numeric_bytes=192,
        replay_operation_forks=3,
        transient_replay_model_owners=1,
        peak_named_model_owners=5,
    )
    final = deepcopy(baseline["final_states"][family])
    final.update(completed_updates=6, training_examples=13)
    if final["wake_state"] is not None:
        final["wake_state"]["wake_updates"] = 6
    return SimpleNamespace(report=report, candidate=SimpleNamespace(final=final, family=family))


@pytest.fixture
def harness(saved, monkeypatch):
    baseline = deepcopy(saved[0])
    results = {family: replay_result(baseline, family) for family in FAMILIES}
    calls, comparisons = [], []

    def run_arm(family):
        calls.append(family)
        return results[family]

    def neutral(pc, control):
        comparisons.append((pc.candidate.family, control.candidate.family))
        return True

    def check(candidate, expected):
        if candidate.final != expected:
            raise RuntimeError("stub preceding native owner changed")

    monkeypatch.setattr(
        study, "_capture_owner", lambda candidate, _report: deepcopy(candidate.final)
    )
    monkeypatch.setattr(study, "_check_owner", check)
    monkeypatch.setattr(study, "_final_state", lambda candidate: deepcopy(candidate.final))
    monkeypatch.setattr(study, "_neutral_exact", neutral)
    return baseline, results, calls, comparisons, run_arm


@pytest.mark.parametrize(
    "damage",
    ["complete", "family", "old_failure", "preservation", "seed", "fit", "old_counter", "role"],
)
def test_should_refuse_invalid_saved_cohort_before_any_new_arm(saved, damage):
    baseline = deepcopy(saved[0])
    if damage == "complete":
        baseline["complete"] = False
    elif damage == "family":
        del baseline["reports"]["active-cpc"]
    elif damage == "old_failure":
        baseline["original_invocation"]["disposition"] = "accepted fresh baseline"
    elif damage == "preservation":
        baseline["saved_original_reports_exact"]["bp"] = False
    elif damage == "seed":
        baseline["reports"]["pc"]["policy"]["base"]["seed"] = 18
    elif damage == "fit":
        baseline["reports"]["bp"]["work"]["fit_calls"] = 2
    elif damage == "old_counter":
        baseline["final_states"]["pc"]["completed_updates"] = 4
    else:
        baseline["reports"]["bp"]["source_episodes"]["warm"]["role"] = "probe_train"
    calls = []

    def run(family):
        calls.append(family)
        pytest.fail("invalid prior cohort admitted live callback")

    payload = study.run_study(baseline, run_arm=run)
    assert payload["complete"] is False and calls == []
    assert payload["attempted_new_families"] == payload["returned_new_families"] == []
    assert "error" in payload


def test_should_complete_fixed_order_with_owned_paid_reports_and_all_signed_comparisons(
    harness, saved
):
    baseline, results, calls, comparisons, run_arm = harness
    original = deepcopy(baseline)
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is True and payload["baseline_reused"] is True
    assert (
        calls
        == payload["attempted_new_families"]
        == payload["returned_new_families"]
        == list(FAMILIES)
    )
    assert comparisons == [("pc", "neutral-cpc")] and payload["neutral_common_exact"] is True
    directions = dict(
        warm_acquisition_gain=True,
        new_acquisition_gain=True,
        warm_retention_loss=False,
        warm_return_recovery=True,
        warm_return_deficit=False,
        new_retention_loss=False,
    )
    for family in FAMILIES:
        assert payload["reports"][family] == results[family].report
        assert payload["final_states"][family] == results[family].candidate.final
        comparison = payload["comparisons"][family]
        assert set(comparison["benchmark"]) == set(directions)
        for name, higher in directions.items():
            row = comparison["benchmark"][name]
            assert row == dict(
                baseline=baseline["reports"][family]["metrics"][name],
                replay=results[family].report["metrics"][name],
                replay_minus_baseline=0.0,
                higher_is_better=higher,
            )
        assert [point["wake_updates"] for point in comparison["probes"]] == [0, 1, 2, 3]
        assert [point["baseline_native_updates"] for point in comparison["probes"]] == [0, 1, 2, 3]
        assert [point["replay_native_updates"] for point in comparison["probes"]] == [0, 2, 4, 6]
        for point in comparison["probes"]:
            assert set(point["views"]) == {"inherited", "zero_velocity"}
            for view in point["views"].values():
                assert view["fresh_acquisition_gain"]["replay_minus_baseline"] == 0.0
                assert view["fresh_training_gain"]["higher_is_better"] is True
                assert set(view["old_retention_loss"]) == {"old_0", "old_1"}
                assert all(
                    row["higher_is_better"] is False and row["replay_minus_baseline"] == 0.0
                    for row in view["old_retention_loss"].values()
                )
    assert payload["observed_new_work"]["retained_applied_receipts"] == 56
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"] == dict.fromkeys(
        FAMILIES, 14
    )
    assert payload["observed_new_work"]["unobserved_attempts"] == []
    assert baseline == original
    assert_unchanged_saved_bytes(saved[1])


@pytest.mark.parametrize("family", FAMILIES)
def test_should_retain_partial_paid_arm_before_refusal_and_stop_future_admission(harness, family):
    baseline, results, calls, _comparisons, run_arm = harness
    results[family].report.update(complete=False, error="paid replay result incomplete")
    payload = study.run_study(baseline, run_arm=run_arm)
    expected = list(FAMILIES[: FAMILIES.index(family) + 1])
    assert payload["complete"] is False
    assert (
        calls == payload["attempted_new_families"] == payload["returned_new_families"] == expected
    )
    assert payload["reports"][family] == results[family].report
    assert payload["reports"][family]["replay"]["charged_exposures"] == 6
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"][family] == 14


@pytest.mark.parametrize("family", FAMILIES)
def test_should_mark_unreturned_attempt_unknown_instead_of_inventing_zero_work(harness, family):
    baseline, _results, calls, _comparisons, run_arm = harness

    def run(selected):
        if selected == family:
            calls.append(selected)
            raise RuntimeError("unreturned paid attempt")
        return run_arm(selected)

    payload = study.run_study(baseline, run_arm=run)
    index = FAMILIES.index(family)
    assert payload["complete"] is False and "unreturned paid attempt" in payload["error"]
    assert calls == payload["attempted_new_families"] == list(FAMILIES[: index + 1])
    assert payload["returned_new_families"] == list(FAMILIES[:index])
    assert family not in payload["reports"]
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"][family] is None
    assert payload["observed_new_work"]["retained_applied_receipts"] is None
    assert payload["observed_new_work"]["unobserved_attempts"] == [family]


def test_should_refuse_neutral_mismatch_before_admitting_active_arm(harness, monkeypatch):
    baseline, _results, calls, comparisons, run_arm = harness

    def refuse(first, second):
        comparisons.append((first.candidate.family, second.candidate.family))
        return False

    monkeypatch.setattr(study, "_neutral_exact", refuse)
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is False and payload["neutral_common_exact"] is False
    assert calls == ["bp", "pc", "neutral-cpc"] and comparisons == [("pc", "neutral-cpc")]
    assert "active-cpc" not in payload["reports"]


@pytest.mark.parametrize(
    "baseline,replay,higher,expected",
    [(1.0, 2.0, True, 1.0), (2.0, 1.0, True, -1.0), (-1.0, -2.0, False, -1.0)],
)
def test_should_preserve_signed_differences_and_direction_without_favorable_threshold(
    baseline, replay, higher, expected
):
    assert study._difference(baseline, replay, higher) == dict(
        baseline=baseline, replay=replay, replay_minus_baseline=expected, higher_is_better=higher
    )


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf")])
def test_should_reject_nonfinite_or_equal_looking_nonnumeric_comparison_metrics(value):
    with pytest.raises(ValueError):
        study._difference(value, 1.0, True)


@pytest.mark.parametrize(
    "damage",
    [
        "plan",
        "charge",
        "wake_label",
        "metrics",
        "initial",
        "first_wake",
        "replay_receipt",
        "probe_role",
        "probe_budget",
        "probe_initial",
        "native_counts",
        "candidate",
    ],
)
def test_should_reject_corrupt_returned_metadata_after_preserving_paid_report(harness, damage):
    baseline, results, calls, _comparisons, run_arm = harness
    result, report = results["bp"], results["bp"].report
    if damage == "plan":
        report["replay"]["plan"]["max_replay_examples"] = 7
    elif damage == "charge":
        report["replay"]["charged_exposures"] = 5
    elif damage == "wake_label":
        report["checkpoints"][2]["wake_updates"] = 3
    elif damage == "metrics":
        report["metrics"]["warm_acquisition_gain"] += 0.01
    elif damage == "initial":
        report["checkpoints"][0]["contexts"]["warm"]["actor_predictions"][0][0] += 0.01
    elif damage == "first_wake":
        report["applied_updates"][0]["diagnostic"]["value"] += 0.01
        report["events"][0]["applied"] = [deepcopy(report["applied_updates"][0])]
    elif damage == "replay_receipt":
        report["applied_consolidations"][0]["event_id"] = "wrong replay source"
    elif damage == "probe_role":
        report["probes"][1]["report"]["zero_velocity"]["checkpoints"][1]["scores"][
            "fresh_training"
        ]["role"] = "train"
    elif damage == "probe_budget":
        report["probes"][1]["report"]["zero_velocity"]["progress"]["updates_completed"] = 2
    elif damage == "probe_initial":
        pair = report["probes"][2]["report"]
        zero = pair["zero_velocity"]
        zero["checkpoints"][0]["scores"]["fresh"]["residual"]["overall"]["normalized_half_mse"] += (
            0.01
        )
        zero["metrics"] = study._metrics(zero["checkpoints"])
        pair["contrast"] = study._contrast(pair["inherited"], zero)
    elif damage == "native_counts":
        report["progress"]["native_updates_completed"] = 5
    else:
        result.candidate = None
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["bp"]
    assert payload["reports"]["bp"] == report
    assert payload["returned_new_families"] == ["bp"] and "error" in payload


@pytest.mark.parametrize("damage", ["scalar", "report_identity", "owner_identity", "owner_state"])
def test_should_recheck_previous_paid_report_and_owner_after_later_arm_getters(harness, damage):
    baseline, results, calls, _comparisons, run_arm = harness
    expected = deepcopy(results["bp"].report)

    def run(family):
        returned = run_arm(family)
        if family == "pc":
            if damage == "scalar":
                results["bp"].report["work"]["training_rows"] += 1
            elif damage == "report_identity":
                results["bp"].report = deepcopy(results["bp"].report)
            elif damage == "owner_identity":
                results["bp"].candidate = SimpleNamespace(
                    final=deepcopy(results["bp"].candidate.final), family="bp"
                )
            else:
                results["bp"].candidate.final["training_examples"] += 1
        return returned

    payload = study.run_study(baseline, run_arm=run)
    assert payload["complete"] is False and calls == ["bp", "pc"]
    assert payload["reports"]["bp"] == expected
    assert payload["reports"]["pc"] == results["pc"].report
    assert "preceding" in payload["error"]


def test_should_recheck_all_owners_after_neutral_observer_before_active(harness, monkeypatch):
    baseline, results, calls, _comparisons, run_arm = harness

    def neutral(_pc, _control):
        results["bp"].candidate.final["completed_updates"] = 999
        return True

    monkeypatch.setattr(study, "_neutral_exact", neutral)
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["bp", "pc", "neutral-cpc"]
    assert "active-cpc" not in payload["reports"]
    assert "preceding native owner changed" in payload["error"]


def test_should_retain_paid_output_if_final_observer_throws(harness, monkeypatch):
    baseline, results, calls, _comparisons, run_arm = harness

    def final(_candidate):
        raise RuntimeError("final metadata observation failed")

    monkeypatch.setattr(study, "_final_state", final)
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["bp"]
    assert payload["reports"]["bp"] == results["bp"].report
    assert "final metadata observation failed" in payload["error"]
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"] == {"bp": 14}


def test_should_keep_malformed_paid_observations_unknown_without_masking_report_failure(harness):
    baseline, results, calls, _comparisons, run_arm = harness
    results["bp"].report.update(
        complete=False, applied_consolidations=None, error="paid report with unknown publication"
    )
    payload = study.run_study(baseline, run_arm=run_arm)
    assert payload["complete"] is False and calls == ["bp"]
    assert payload["reports"]["bp"] == results["bp"].report
    assert payload["observed_new_work"]["retained_applied_receipts_by_family"] == {"bp": None}
    assert payload["observed_new_work"]["retained_applied_receipts"] is None
    assert "incomplete" in payload["error"]


def test_should_finish_private_owner_checks_after_all_public_candidate_getters(harness):
    baseline, results, calls, _comparisons, run_arm = harness
    original, observations = results["pc"], []

    class LaterResult:
        @property
        def report(self):
            return original.report

        @property
        def candidate(self):
            observations.append(True)
            if len(observations) == 2:
                # The current check already read BP's public owner. Returning a
                # clean later owner must not mask this other native-state change.
                results["bp"].candidate.final["training_examples"] += 1
            return original.candidate

    def run(family):
        run_arm(family)
        return LaterResult() if family == "pc" else results[family]

    payload = study.run_study(baseline, run_arm=run)
    assert payload["complete"] is False and calls == ["bp", "pc"]
    assert len(observations) == 2
    assert payload["reports"]["pc"] == original.report
    assert "preceding native owner changed" in payload["error"]
