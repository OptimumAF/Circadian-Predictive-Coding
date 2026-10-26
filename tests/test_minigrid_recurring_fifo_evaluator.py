"""Distinct FIFO36 evaluator admission using scripted ports, not learning.

Pure literal snapshots include genuine FIFO retirement/order fields. The reused
fixture forbids native count/neural prediction and training, Torch comparison
and imports, and uses scripted games. Success here is harness behavior only.
"""

from dataclasses import replace
import pickle
from typing import Any

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_action_memory_learner import (
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
)
from src.core.action_memory import ActionMemoryEntry
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_key_door import predefined_layouts
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_minigrid_recurring_memory_evaluator import (
    ScriptedCountPort,
    case as _refusal_case_fixture,
    snapshot as refusal_snapshot,
)

POINTS = (0, 32, 64, 96, 128)
refusal_case = _refusal_case_fixture


def snapshot(updates=0, *, cap=36, payload=87264, rows=None, full=False, retired=0, evictions=0):
    policy = FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=cap, max_payload_bytes=payload
    )
    rows = 9 * updates if rows is None else rows
    entries = []
    if rows:
        if full:
            for index in range(36):
                features = [-1.0] * 296
                features[0] = (0, 1, 3, 4, 5, 6)[index // 6] / 5 - 1
                features[1] = (index % 6) / 2.5 - 1
                count = rows - retired - 35 if index == 0 else 1
                entries.append(ActionMemoryEntry(tuple(features), (count, 0, 0, 0, 0, 0, 0)))
        else:
            entries.append(ActionMemoryEntry((-1.0,) * 296, (rows - retired, 0, 0, 0, 0, 0, 0)))
    return InvariantMemorySnapshot(
        InvariantMemoryPolicy(policy),
        FifoActionMemorySnapshot(policy, tuple(entries), updates, rows, evictions, retired),
    )


@pytest.fixture
def fifo_case(refusal_case):
    current = refusal_case
    current.source = ScriptedCountPort(snapshot(), current.work)
    current.state = snapshot(128)
    return current


def evaluate(case, *, updates=128, state=None, **changes):
    options = dict(
        updates=updates,
        layouts=case.layouts,
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
    )
    options.update(changes)
    return evaluation.evaluate_invariant_fifo_checkpoint(
        case.source, case.state if state is None else state, **options
    )


@pytest.mark.parametrize("point", POINTS)
def test_should_score_all_declared_fifo_points_without_widening_input_authority(fifo_case, point):
    case = fifo_case
    state = snapshot(point)
    before = pickle.dumps((case.source.value, state, case.sources))
    result = evaluate(case, updates=point, state=state)
    assert result["status"] == "complete" and result["state_unchanged"] is True
    assert result["method"] == "public_invariant_observation_memory_fifo"
    assert result["role"] == "train" and result["native_training_examples"] == 9 * point
    assert result["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
    assert result["native_evictions"] == result["native_retired_examples"] == 0
    assert result["native_retained_entries"] == (0 if point == 0 else 1)
    assert result["native_retained_logical_payload_bytes"] == (0 if point == 0 else 2424)
    assert result["native_retirement_metadata_bytes"] == 16
    assert [(s["context"], s["cue"]) for s in result["scores"]] == [
        (context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    ]
    assert len(result["episodes"]) == 6 and all(
        s["status"] == "complete" for s in result["episodes"]
    )
    work = result["work"]
    assert work["raw_score_forwards_completed"] == 6 and work["raw_score_rows_completed"] == 54
    assert work["live_prediction_calls_returned"] == work["environment_steps_returned"] == 12
    assert work["internal_projection_rows_returned"] == 66
    assert work["internal_projection_array_bytes_returned"] == 66 * 2368
    assert all(value.shape == (9, 296) for value in case.work["returns"][:6])
    assert all(value.shape == (1, 296) for value in case.work["returns"][6:])
    assert all(slot["source_role"] == "train" for slot in result["episodes"])
    assert pickle.dumps((case.source.value, state, case.sources)) == before
    assert case.budget.updates_completed == 128 and case.budget.last_clock == 10.0


def test_should_report_retained_table_and_retirement_metadata_separately(fifo_case):
    state = snapshot(128, full=True, evictions=2, retired=3)
    result = evaluate(fifo_case, state=state)
    assert result["native_retained_entries"] == 36
    assert result["native_retained_logical_payload_bytes"] == 87264
    assert result["native_retirement_metadata_bytes"] == 16
    assert result["native_evictions"] == 2 and result["native_retired_examples"] == 3


@pytest.mark.parametrize(
    "fault",
    [
        "bool",
        "point",
        "clock",
        "rows",
        "raw",
        "refusal",
        "FIFO64",
        "cap35",
        "payload",
        "encoding",
        "extra",
        "evictions",
        "retired",
    ],
)
def test_should_refuse_wrong_backend_caps_clocks_or_retirement_before_external_ports(
    fifo_case, fault
):
    case = fifo_case
    updates, state = 128, case.state
    if fault == "bool":
        updates = False
    elif fault == "point":
        updates = 144
    elif fault == "clock":
        state = snapshot(96)
    elif fault == "rows":
        state = snapshot(128, rows=1151)
    elif fault == "raw":
        state = state.native_state
    elif fault == "refusal":
        state = refusal_snapshot(128)
    elif fault == "FIFO64":
        state = snapshot(128, cap=64, payload=155136)
    elif fault == "cap35":
        state = snapshot(128, cap=35)
    elif fault == "payload":
        state = snapshot(128, payload=87263)
    elif fault == "encoding":
        object.__setattr__(state.policy, "encoding_id", "other")
    elif fault == "extra":
        object.__setattr__(state.native_state, "extra", 1)
    elif fault == "evictions":
        object.__setattr__(state.native_state, "evictions", True)
    else:
        object.__setattr__(state.native_state, "retired_examples", 1)
    with pytest.raises(ValueError):
        evaluate(case, updates=updates, state=state)
    assert case.work == {}


@pytest.mark.parametrize(
    "fault",
    ["final", "probe", "A_only", "changed_layout", "extra_layout", "source_order", "source_role"],
)
def test_should_keep_original_train_triple_and_six_source_admission(fifo_case, fault):
    case = fifo_case
    options: dict[str, Any] = {}
    if fault == "final":
        options.update(role="final_test", layouts=(predefined_layouts()[-1],), sources=())
    elif fault == "probe":
        options["role"] = "probe_test"
    elif fault == "A_only":
        options.update(layouts=case.layouts[:1], sources=case.sources[:2])
    elif fault == "changed_layout":
        options["layouts"] = (replace(case.layouts[0], wall_color="red"), *case.layouts[1:])
    elif fault == "extra_layout":
        value = replace(case.layouts[0])
        object.__setattr__(value, "extra", 1)
        options["layouts"] = (value, *case.layouts[1:])
    elif fault == "source_order":
        options["sources"] = tuple(reversed(case.sources))
    else:
        options["sources"] = (replace(case.sources[0], role="final_test"), *case.sources[1:])
    with pytest.raises(ValueError):
        evaluate(case, **options)
    assert case.work == {}


@pytest.mark.parametrize("target", ["entries", "retirement", "order", "source", "budget"])
def test_should_refuse_full_fifo_changes_after_returned_scores_and_keep_paid_projection(
    fifo_case, target
):
    case = fifo_case
    case.state = snapshot(128, full=True, evictions=1, retired=1)

    def mutate(scorer, features):
        if target == "source":
            case.source.value = snapshot(32)
        elif target == "budget":
            case.budget.updates_completed = 127
        else:
            state = scorer.value.native_state
            if target == "entries":
                entry = state.entries[0]
                object.__setattr__(entry, "counts", (entry.counts[0] - 1, 1, 0, 0, 0, 0, 0))
            elif target == "order":
                object.__setattr__(state, "entries", tuple(reversed(state.entries)))
            else:
                # Still a valid same-clock FIFO snapshot: do not ignore these
                # fields merely because ordinary counters/geometry are equal.
                entry = state.entries[0]
                object.__setattr__(entry, "counts", (entry.counts[0] - 1, 0, 0, 0, 0, 0, 0))
                object.__setattr__(state, "retired_examples", 2)
                object.__setattr__(state, "evictions", 2)

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    report = caught.value.report
    assert report["status"] == "failed" and report["scores"] == []
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["internal_projection_rows_returned"] == 9
    assert report["work"]["internal_projection_array_bytes_returned"] == 9 * 2368
    assert not case.work.get("environments")


def test_should_refuse_changes_during_the_last_snapshot_parent_boundary(fifo_case):
    case = fifo_case
    changed = [False]

    def snapshot_read(port):
        if port.scorer and case.work.get("returns"):
            changed[0] = True

    def parent(owner):
        if changed[0]:
            raise ValueError("preceding complete stream evidence changed")

    case.work.update(on_snapshot=snapshot_read, on_guard=parent)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    assert caught.value.report["work"]["raw_score_forwards_completed"] == 1
    assert not case.work.get("environments")


@pytest.mark.parametrize("fault", ["prediction_error", "bad_probability"])
def test_should_keep_failed_or_invalid_returned_prediction_cost(fifo_case, fault):
    case = fifo_case
    case.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case)
    work = caught.value.report["work"]
    assert work["raw_score_forwards_attempted"] == 1
    completed = 0 if fault == "prediction_error" else 1
    assert work["raw_score_forwards_completed"] == completed
    assert work["internal_projection_rows_returned"] == completed * 9
    assert work["failure_work_may_be_unobserved"] is True


@pytest.mark.parametrize("route", ["raw", "refusal", "FIFO64"])
def test_should_not_admit_fifo36_through_old_raw_refusal_or_fifo64_routes(fifo_case, route):
    case = fifo_case
    with pytest.raises(ValueError):
        if route == "raw":
            evaluation.evaluate_checkpoint(
                case.source,
                case.state,
                method="observation_memory",
                updates=128,
                layouts=case.layouts,
                role="train",
                sources=case.sources,
                budget=case.budget,
                check_boundary=case.guard,
                public_history=True,
            )
        elif route == "refusal":
            evaluation.evaluate_invariant_memory_checkpoint(
                case.source,
                case.state,
                updates=128,
                layouts=case.layouts,
                role="train",
                sources=case.sources,
                budget=case.budget,
                check_boundary=case.guard,
            )
        else:
            evaluation.evaluate_representation_checkpoint(
                case.source, case.state, updates=128, budget=case.budget, check_boundary=case.guard
            )
    assert case.work == {}
