"""B22/B50 supervision and exact memory replay using tiny causal TRAIN traces.

Targets in the corrective packet retain the original student's history; they do
not advance that history. The replay trace instead follows its own float32
executed actions. All work is pure consistency/retrieval, without a learner.
"""

from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import struct
from typing import Any

import pytest

from scripts import diagnose_pick19_memory as diagnostic
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.pick_cube_action_history import ActionHistoryState, advance_action_history


def float32(value):
    return struct.unpack("!f", struct.pack("!f", value))[0]


def observation(position, history):
    return (
        0.0,
        position,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        *history.previous_action,
        history.closing_count / 6,
    )


def episode(name, observations, actions, final):
    return ContinuousEpisode(
        name,
        "pick-cube-train-19",
        "train",
        cube_taskspace_spec(),
        tuple(
            ContinuousTransition(
                index,
                before,
                action,
                0.0,
                observations[index + 1] if index + 1 < len(observations) else final,
                False,
                index == len(observations) - 1,
            )
            for index, (before, action) in enumerate(zip(observations, actions, strict=True))
        ),
    )


def inputs(*, changed_command=False):
    zero = ActionHistoryState()
    opened = (float32(0.123456789), 0.0, 0.0, 1.0)
    b22_actions = (opened, (-0.5, 0.0, 0.0, 1.0))
    old_history = advance_action_history(zero, opened)
    b_observations = (observation(0.0, zero), observation(10.0, old_history))
    b22 = episode(
        "b22",
        b_observations,
        b22_actions,
        observation(10.0, advance_action_history(old_history, b22_actions[1])),
    )
    targets = ((0.123456789, 0.0, 0.0, -1.0), (0.5, 0.0, 0.0, 1.0))
    # Complete target packet preserves the real open-gripper student history.
    b50 = episode("b50", b_observations, targets, observation(10.0, old_history))
    a = episode("a", (observation(30.0, zero),), ((0.0, 0.0, 0.0, -1.0),), observation(30.0, zero))
    executed = ((0.25 if changed_command else float32(targets[0][0]), 0.0, 0.0, -1.0), targets[1])
    history = advance_action_history(zero, executed[0])
    trace = episode(
        "trace",
        (observation(0.0, zero), observation(10.0, history)),
        executed,
        observation(10.0, advance_action_history(history, executed[1])),
    )
    return b22, b50, trace, (("a", a), ("b50", b50)), (1, 2)


def forbidden(*args, **options):
    pytest.fail("inadmissible diagnostic inputs reached numerical work")


@pytest.fixture(scope="module", autouse=True)
def ledger():
    counts = dict(
        consistency_attempts=0,
        consistency_returns=0,
        consistency_rows=0,
        pair_attempts=0,
        pair_returns=0,
        native_conversion_attempts=0,
        native_conversion_returns=0,
        nearest_attempts=0,
        nearest_returns=0,
        nearest_query_rows=0,
        candidates_completed=0,
        features_completed=0,
        additional_conversion_attempts=0,
        additional_conversion_returns=0,
    )
    consistency, nearest = (
        diagnostic.continuous_input_target_consistency,
        diagnostic.nearest_continuous_actions,
    )
    convert = diagnostic._native_consistency_targets

    def consistent(features, targets, spec, *, _work):
        counts["consistency_attempts"] += 1
        counts["consistency_rows"] += len(features)
        before = asdict(_work)
        try:
            result = consistency(features, targets, spec, _work=_work)
            counts["consistency_returns"] += 1
            return result
        finally:
            for name in (
                "pair_attempts",
                "pair_returns",
                "native_conversion_attempts",
                "native_conversion_returns",
            ):
                counts[name] += getattr(_work, name) - before[name]

    def retrieved(query, features, targets, spec):
        counts["nearest_attempts"] += 1
        counts["nearest_query_rows"] += len(query)
        result = nearest(query, features, targets, spec)
        counts["nearest_returns"] += 1
        counts["candidates_completed"] += result.distance_candidates
        counts["features_completed"] += result.distance_feature_values
        return result

    def converted(actions, work):
        before = asdict(work)
        try:
            return convert(actions, work)
        finally:
            counts["additional_conversion_attempts"] += (
                work.native_conversion_attempts - before["native_conversion_attempts"]
            )
            counts["additional_conversion_returns"] += (
                work.native_conversion_returns - before["native_conversion_returns"]
            )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(diagnostic, "continuous_input_target_consistency", consistent)
        patch.setattr(diagnostic, "nearest_continuous_actions", retrieved)
        patch.setattr(diagnostic, "_native_consistency_targets", converted)
        yield counts
    if os.environ.get("PICK19_MEMORY_DIAG_FIXTURE_LEDGER"):
        with Path(os.environ["PICK19_MEMORY_DIAG_FIXTURE_LEDGER"]).open(
            "x", encoding="utf8"
        ) as output:
            json.dump(counts, output, sort_keys=True)
            output.write("\n")
    assert counts == dict(
        consistency_attempts=6,
        consistency_returns=6,
        consistency_rows=24,
        pair_attempts=36,
        pair_returns=36,
        native_conversion_attempts=96,
        native_conversion_returns=96,
        nearest_attempts=7,
        nearest_returns=7,
        nearest_query_rows=14,
        candidates_completed=38,
        features_completed=456,
        additional_conversion_attempts=40,
        additional_conversion_returns=40,
    )


@pytest.fixture(scope="module")
def measured(ledger):
    data = inputs()
    before = (
        tuple(asdict(item) for item in data[:3]),
        tuple((name, asdict(item)) for name, item in data[3]),
    )
    result = diagnostic.analyze_pick19(*data)
    assert (
        tuple(asdict(item) for item in data[:3]),
        tuple((name, asdict(item)) for name, item in data[3]),
    ) == before
    return result, data


def test_should_bind_all_pairs_original_bank_rows_and_signed_b22_alternatives_without_control_claim(
    measured,
):
    result, data = measured
    assert result["complete"] is True and result["error"] is None
    assert result["consistency"]["rows"] == 4
    assert len(result["consistency"]["pairs"]) == 6
    assert result["consistency"]["duplicate_input_pairs"] == 2
    assert (
        result["consistency"]["raw_conflict_pairs"]
        == result["consistency"]["native_conflict_pairs"]
        == 2
    )
    pairs = {(row["first_row"], row["second_row"]): row for row in result["consistency"]["pairs"]}
    assert tuple(pairs[0, 2]["target_delta"]) == (
        0.123456789 - float32(0.123456789),
        0.0,
        0.0,
        -2.0,
    )
    assert tuple(pairs[0, 2]["native_target_delta"]) == (0.0, 0.0, 0.0, -2.0)
    assert (
        tuple(pairs[1, 3]["target_delta"])
        == tuple(pairs[1, 3]["native_target_delta"])
        == (1.0, 0.0, 0.0, 0.0)
    )
    assert tuple(result["replay"]["selected_indices"]) == (1, 2)
    assert tuple(result["b22_readout"]["selected_indices"]) == (0, 1)
    rows = json.loads(json.dumps(result["trace_rows"]))
    assert rows == [
        dict(
            step=0,
            selected_global_row=1,
            selected_group="b50",
            selected_group_row=0,
            executed_action=list(data[2].transitions[0].action),
            b22_row=0,
            b22_native_action=list(data[0].transitions[0].action),
            b22_minus_executed=[0.0, 0.0, 0.0, 2.0],
        ),
        dict(
            step=1,
            selected_global_row=2,
            selected_group="b50",
            selected_group_row=1,
            executed_action=list(data[2].transitions[1].action),
            b22_row=1,
            b22_native_action=list(data[0].transitions[1].action),
            b22_minus_executed=[-1.0, 0.0, 0.0, 0.0],
        ),
    ]
    work = result["work"]
    assert work["kernel_calls_attempted"] == work["kernel_calls_returned"] == 1
    assert work["nearest_calls_attempted"] == work["nearest_calls_returned"] == 2
    assert work["nearest_query_rows"] == 4
    assert work["nearest_candidates_completed"] == work["nearest_candidate_upper_bound"] == 10
    assert work["nearest_features_completed"] == work["nearest_feature_upper_bound"] == 120
    assert (
        work["additional_native_conversion_attempts"]
        == work["additional_native_conversion_returns"]
        == 16
    )
    assert (
        result["unknown_unreturned_kernel_work"]
        is result["unknown_unreturned_nearest_work"]
        is False
    )
    assert work["cross_group_pairs"] == 4 and work["cross_group_nearest_comparisons"] == 8
    assert result["cross_group_nearest"] == [
        dict(global_row=row, other_global_row=other, input_l2=0.0)
        for row, other in ((0, 2), (1, 3), (2, 0), (3, 1))
    ]


def test_should_own_diagnostic_rows_without_rewriting_original_student_or_target_history(measured):
    result, data = measured
    b22, b50, trace, _, _ = data
    assert b22.transitions[1].observation[10] == b50.transitions[1].observation[10] == 1.0
    assert trace.transitions[1].observation[10] == -1.0
    assert b50.transitions[0].action[0] != trace.transitions[0].action[0]
    original = result["trace_rows"][0]["executed_action"]
    try:
        result["trace_rows"][0]["executed_action"] = [999.0] * 4
        assert trace.transitions[0].action == (float32(0.123456789), 0.0, 0.0, -1.0)
        assert b50.transitions[0].action == (0.123456789, 0.0, 0.0, -1.0)
    finally:
        result["trace_rows"][0]["executed_action"] = original


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "bank_role",
        "spec",
        "missing_b50",
        "duplicate_b50",
        "bank_contents",
        "same_b_ids",
        "selector_list",
        "selector_length",
        "selector_bool",
        "selector_range",
        "history",
        "float32",
    ],
)
def test_should_refuse_bad_roles_scope_rows_selectors_and_actual_trace_history_before_numerics(
    monkeypatch, fault
):
    b22, b50, trace, bank, selected = inputs()
    if fault == "role":
        trace = replace(trace, role="probe_test")
    elif fault == "bank_role":
        bank = ((bank[0][0], replace(bank[0][1], role="final_test")), bank[1])
    elif fault == "spec":
        b22 = replace(
            b22, spec=replace(b22.spec, action_units=("other",) + b22.spec.action_units[1:])
        )
    elif fault == "missing_b50":
        bank = (("a", bank[0][1]), ("other", replace(b50, episode_id="not-b50")))
    elif fault == "duplicate_b50":
        bank = bank + (("again", b50),)
    elif fault == "bank_contents":
        changed = replace(
            b50,
            transitions=(
                replace(b50.transitions[0], action=(0.25, 0.0, 0.0, -1.0)),
                b50.transitions[1],
            ),
        )
        bank = (bank[0], ("b50", changed))
    elif fault == "same_b_ids":
        b22 = replace(b22, episode_id=b50.episode_id)
    elif fault == "selector_list":
        selected = list(selected)
    elif fault == "selector_length":
        selected = selected[:1]
    elif fault == "selector_bool":
        selected = (True, 2)
    elif fault == "selector_range":
        selected = (1, 3)
    elif fault == "history":
        object.__setattr__(
            trace.transitions[1], "observation", observation(10.0, ActionHistoryState())
        )
        object.__setattr__(
            trace.transitions[0], "next_observation", trace.transitions[1].observation
        )
    else:
        object.__setattr__(trace.transitions[0], "action", (0.123456789, 0.0, 0.0, -1.0))
    monkeypatch.setattr(diagnostic, "continuous_input_target_consistency", forbidden)
    monkeypatch.setattr(diagnostic, "nearest_continuous_actions", forbidden)
    with pytest.raises(ValueError):
        diagnostic.analyze_pick19(b22, b50, trace, bank, selected)


@pytest.mark.parametrize("fault", ["selected", "executed"])
def test_should_keep_consistency_and_paid_replay_when_observed_selector_or_command_cannot_be_reproduced(
    fault,
):
    data = inputs(changed_command=fault == "executed")
    if fault == "selected":
        data = (*data[:4], (1, 1))
    result = diagnostic.analyze_pick19(*data)
    assert result["complete"] is False and result["error"]
    assert result["consistency"] is not None and result["replay"] is not None
    assert result["b22_readout"] is None
    assert (
        result["work"]["nearest_calls_attempted"] == result["work"]["nearest_calls_returned"] == 1
    )
    assert result["unknown_unreturned_nearest_work"] is False


def test_should_keep_completed_consistency_work_if_its_return_is_unavailable(monkeypatch):
    kernel = diagnostic.continuous_input_target_consistency

    def unavailable(*args, **options):
        kernel(*args, **options)
        raise RuntimeError("paid consistency return unavailable")

    monkeypatch.setattr(diagnostic, "continuous_input_target_consistency", unavailable)
    result = diagnostic.analyze_pick19(*inputs())
    assert result["complete"] is False and result["consistency"] is None
    assert result["work"]["pair_returns"] == 6 and result["work"]["native_conversion_returns"] == 16
    assert result["unknown_unreturned_kernel_work"] is True
    assert result["work"]["nearest_calls_attempted"] == 0


@pytest.mark.parametrize("lost", [1, 2])
def test_should_keep_all_observed_prior_results_and_mark_only_unreturned_nearest_unknown(
    monkeypatch, lost
):
    kernel = diagnostic.nearest_continuous_actions
    calls = 0

    def unavailable(*args, **options):
        nonlocal calls
        calls += 1
        result = kernel(*args, **options)
        if calls == lost:
            raise RuntimeError("paid nearest return unavailable")
        return result

    monkeypatch.setattr(diagnostic, "nearest_continuous_actions", unavailable)
    result = diagnostic.analyze_pick19(*inputs())
    assert calls == lost and result["complete"] is False
    assert result["consistency"] is not None and result["unknown_unreturned_kernel_work"] is False
    assert (result["replay"] is None) is (lost == 1)
    assert result["b22_readout"] is None
    assert result["unknown_unreturned_nearest_work"] is True
    assert result["work"]["nearest_calls_attempted"] == lost
    assert result["work"]["nearest_calls_returned"] == lost - 1


@pytest.mark.parametrize("fault", [None, "partial", "source"])
def test_should_publish_paid_analysis_exclusively_and_refuse_replaced_bound_sources(
    tmp_path, monkeypatch, fault
):
    data = inputs()
    receipt = dict(
        sources=[
            dict(name="memory", path="synthetic-memory", sha256="memory", bytes=2),
            dict(name="teacher", path="synthetic-teacher", sha256="teacher", bytes=3),
        ]
    )
    analysis = dict(
        complete=fault != "partial",
        error="known partial" if fault == "partial" else None,
        work=dict(kernel_calls_attempted=1, pair_returns=1),
        trace_rows=[],
    )
    calls: list[Any] = []

    def read(*, work):
        work.update(attempts=2, returns=2, bytes=5)
        return (*data, receipt)

    def analyze(*args):
        calls.append(args)
        assert args == data
        return analysis

    def bound(original, work):
        assert original is receipt
        work.update(
            attempts=1 if fault == "source" else 2,
            returns=1 if fault == "source" else 2,
            bytes=2 if fault == "source" else 5,
        )
        if fault == "source":
            raise ValueError("original bound memory bytes replaced")

    monkeypatch.setattr(diagnostic, "read_pick19_inputs", read)
    monkeypatch.setattr(diagnostic, "analyze_pick19", analyze)
    monkeypatch.setattr(diagnostic, "_require_sources", bound)
    output = tmp_path / "diagnosis.json"
    result = diagnostic.run_diagnosis(output)
    assert len(calls) == 1 and result["complete"] is (fault is None)
    assert json.loads(json.dumps(result["analysis"])) == analysis
    assert result["source_bytes_unchanged"] is (fault != "source")
    assert result["work"]["input_reads"] == dict(attempts=2, returns=2, bytes=5)
    assert result["work"]["final_reads"] == dict(
        attempts=1 if fault == "source" else 2,
        returns=1 if fault == "source" else 2,
        bytes=2 if fault == "source" else 5,
    )
    assert all(
        result["work"][name] == 0
        for name in (
            "model_initializations",
            "model_random_draws",
            "optimizer_updates",
            "empirical_fits",
            "teacher_queries",
            "simulator_controls",
            "probe_updates",
        )
    )
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))


def test_should_preserve_existing_output_before_source_read_or_numerical_work(
    tmp_path, monkeypatch
):
    output = tmp_path / "diagnosis.json"
    output.write_text("preserve")
    monkeypatch.setattr(diagnostic, "read_pick19_inputs", forbidden)
    monkeypatch.setattr(diagnostic, "analyze_pick19", forbidden)
    with pytest.raises(ValueError):
        diagnostic.run_diagnosis(output)
    assert output.read_text() == "preserve"
