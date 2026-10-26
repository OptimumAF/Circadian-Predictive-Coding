"""Saved FIFO36 game admission through existing scripted, target-free ports.

These literal snapshots exercise endpoint binding, retirement accounting and
paid-prefix failures. Scripted successes are engineering evidence only: no
native count prediction, environment, training or scientific run occurs here.
"""

from copy import deepcopy
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.invariant_action_memory_learner import (
    InvariantMemoryPolicy,
    InvariantMemorySnapshot,
)
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.action_memory import ActionMemoryEntry
from src.core.fifo_action_memory import FifoActionMemoryPolicy, FifoActionMemorySnapshot
from src.core.minigrid_invariant_encoding import INVARIANT_HISTORY_ENCODING_ID
from test_minigrid_probe_live_evaluator import (
    ScriptedPort,
    TextSubclass,
    case as case,
    evaluate,
    snapshot,
)


def fifo_state(native, *, retired=False):
    if not retired:
        return snapshot(native, cap=36, payload=87264, fifo=True)
    # A full ordered table permits nonzero retirement under the native law.
    # Distinct discrete object codes are canonical public keys, not raw labels.
    examples = native * 9
    retired_examples = 864 if native == 160 else 18
    entries = []
    for index in range(36):
        features = [-1.0] * 296
        features[index * 3] = 2.0 * 6 / 10 - 1.0
        count = examples - retired_examples - 35 if index == 0 else 1
        entries.append(ActionMemoryEntry(tuple(features), (count, 0, 0, 0, 0, 0, 0)))
    policy = FifoActionMemoryPolicy(
        feature_dim=296, num_classes=7, max_entries=36, max_payload_bytes=87264
    )
    native_state = FifoActionMemorySnapshot(
        policy, tuple(entries), native, examples, 54 if native == 160 else 2, retired_examples
    )
    return InvariantMemorySnapshot(InvariantMemoryPolicy(policy), native_state)


def fifo_case(case, origin=0, *, retired=False):
    current = case(origin, "after_return")
    current.state = fifo_state(origin + 32, retired=retired)
    current.source = ScriptedPort(current.state, current.work)
    return current


@pytest.mark.parametrize("origin", [0, 128])
@pytest.mark.parametrize("retired", [False, True])
def test_should_admit_only_explicit_saved_fifo_and_report_all_four_target_free_games(
    case, origin, retired
):
    current = fifo_case(case, origin, retired=retired)
    budget = current.budget
    before = pickle.dumps(
        (
            current.state,
            current.source.value,
            budget.budget,
            budget.started_at,
            budget.updates_completed,
            budget.replay_examples_completed,
        )
    )
    clock = budget.clock

    report = evaluate(current, saved_fifo=True)

    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert (report["source_updates"], report["relative_updates"], report["phase"]) == (
        origin,
        32,
        "after_return",
    )
    assert report["updates"] == origin + 32
    assert report["native_training_examples"] == 9 * (origin + 32)
    assert report["method"] == "public_invariant_observation_memory_fifo"
    assert report["model_encoding"] == INVARIANT_HISTORY_ENCODING_ID
    assert report["memory_policy"] == dict(
        method="observation_memory_fifo",
        feature_dim=296,
        num_classes=7,
        max_entries=36,
        max_payload_bytes=87264,
    )
    native = current.state.native_state
    assert report["native_evictions"] == native.evictions
    assert report["native_retired_examples"] == native.retired_examples
    assert report["native_retained_entries"] == len(native.entries)
    assert report["native_retained_logical_payload_bytes"] == 2424 * len(native.entries)
    assert report["native_retirement_metadata_bytes"] == 16
    assert report["scores"] == []
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in report["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
        ("F-blue", "blue", "probe_test"),
        ("F-red", "red", "probe_test"),
    ]
    assert all(slot["status"] == "complete" and slot["success"] for slot in report["episodes"])
    assert all(slot["return"] == 0.5 for slot in report["episodes"])
    assert report["work"]["raw_score_forwards_attempted"] == 0
    assert report["work"]["raw_score_rows_attempted"] == 0
    assert report["work"]["live_prediction_calls_attempted"] == 8
    assert report["work"]["live_prediction_calls_returned"] == 8
    assert report["work"]["environment_resets_attempted"] == 4
    assert report["work"]["environment_resets_completed"] == 4
    assert report["work"]["environment_steps_attempted"] == 8
    assert report["work"]["environment_steps_returned"] == 8
    assert report["work"]["internal_projection_array_bytes_returned"] == 8 * 2368
    assert len(current.work["forks"]) == len(current.work["restores"]) == 1
    assert len(current.work["environments"]) == 4
    assert all(env.closed for env in current.work["environments"])
    assert budget.clock is clock
    assert (
        pickle.dumps(
            (
                current.state,
                current.source.value,
                budget.budget,
                budget.started_at,
                budget.updates_completed,
                budget.replay_examples_completed,
            )
        )
        == before
    )
    for slot in report["episodes"]:
        first = slot["predictions"][0]["observation"]
        for decision in slot["predictions"]:
            assert decision["features"][:148] == first
            assert decision["features"][148:] == decision["observation"]
            assert decision["action"] == 0  # Frozen first raw argmax, no target repair.


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_native_boolean_opt_in_before_any_port_or_environment(case, flag):
    current = fifo_case(case)
    with pytest.raises(ValueError):
        evaluate(current, saved_fifo=flag)
    assert current.work == {}


@pytest.mark.parametrize("origin", [0, 128])
@pytest.mark.parametrize("flag", [None, False])
def test_should_keep_fifo36_refused_without_explicit_opt_in(case, origin, flag):
    current = fifo_case(case, origin)
    options = {} if flag is None else {"saved_fifo": flag}
    with pytest.raises(ValueError):
        evaluate(current, **options)
    assert current.work == {}


@pytest.mark.parametrize(
    "phase", ["after_probe", "after_return_extra", TextSubclass("after_return")]
)
def test_should_never_invent_a_saved_fifo_probe_endpoint_for_an_unavailable_phase(case, phase):
    current = fifo_case(case)
    current.phase = phase
    with pytest.raises(ValueError):
        evaluate(current, saved_fifo=True)
    assert current.work == {}


@pytest.mark.parametrize("origin", [True, np.int64(0), 32, 160])
def test_should_bind_saved_fifo_to_exact_source_zero_or_128(case, origin):
    current = fifo_case(case)
    with pytest.raises(ValueError):
        evaluate(current, source_updates=origin, saved_fifo=True)
    assert current.work == {}


@pytest.mark.parametrize(
    "fault",
    [
        "refusal",
        "fifo64",
        "cap35",
        "payload",
        "raw",
        "clock",
        "rows",
        "extra",
        "native_extra",
        "encoding",
        "noncanonical",
    ],
)
def test_should_validate_the_complete_fifo36_endpoint_before_resources(case, fault):
    current = fifo_case(case)
    if fault == "refusal":
        current.state = snapshot(32)
    elif fault == "fifo64":
        current.state = snapshot(32, fifo=True)
    elif fault == "cap35":
        current.state = snapshot(32, fifo=True, cap=35, payload=87264)
    elif fault == "payload":
        current.state = snapshot(32, fifo=True, cap=36, payload=87265)
    elif fault == "raw":
        current.state = current.state.native_state
    elif fault == "clock":
        current.state = fifo_state(16)
    elif fault == "rows":
        current.state = snapshot(32, fifo=True, cap=36, payload=87264, rows=289)
    elif fault == "extra":
        object.__setattr__(current.state, "extra", 1)
    elif fault == "native_extra":
        object.__setattr__(current.state.native_state, "extra", 1)
    elif fault == "encoding":
        object.__setattr__(current.state.policy, "encoding_id", "unbound")
    else:
        entry = current.state.native_state.entries[0]
        object.__setattr__(entry, "features", (*entry.features[:-1], 1.0))
    with pytest.raises(ValueError):
        evaluate(current, saved_fifo=True)
    assert current.work == {}


@pytest.mark.parametrize("training,replay", [(1, 0), (0, None), (0, 1), (0, True)])
def test_should_preserve_zero_training_and_replay_authority_before_work(case, training, replay):
    current = fifo_case(case)
    object.__setattr__(current.budget.budget, "max_training_updates", training)
    object.__setattr__(current.budget.budget, "max_replay_examples", replay)
    with pytest.raises(ValueError):
        evaluate(current, saved_fifo=True)
    assert current.work == {}


@pytest.mark.parametrize("target", ["checkpoint", "source", "scorer"])
@pytest.mark.parametrize("fault", ["counts", "order", "evictions", "retired"])
def test_should_refuse_full_fifo_state_changes_after_prediction_before_physics(case, target, fault):
    current = fifo_case(case, 128, retired=True)

    def mutate(scorer):
        state = (
            current.state
            if target == "checkpoint"
            else (current.source.value if target == "source" else scorer.value)
        )
        native = state.native_state
        if fault == "order":
            object.__setattr__(native, "entries", tuple(reversed(native.entries)))
        elif fault == "evictions":
            object.__setattr__(native, "evictions", native.evictions + 1)
        else:
            entry = native.entries[0]
            counts = entry.counts
            if fault == "counts":
                object.__setattr__(entry, "counts", (counts[0] - 1, 1, *counts[2:]))
            else:
                # Remains a valid complete FIFO state with unchanged lifetime
                # clocks, but retirement is not the admitted checkpoint's.
                object.__setattr__(entry, "counts", (counts[0] - 1, *counts[1:]))
                object.__setattr__(native, "retired_examples", native.retired_examples + 1)

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)
    report = caught.value.report
    assert report["work"]["live_prediction_calls_attempted"] == 1
    assert report["work"]["live_prediction_calls_returned"] == 1
    assert report["work"]["environment_steps_returned"] == 0
    assert report["episodes"][0]["prediction_calls_returned"] == 1
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])


@pytest.mark.parametrize("fault", ["updates", "replay", "clock", "started", "config", "progress"])
def test_should_refuse_budget_replacement_or_new_charges_after_paid_prediction(case, fault):
    current = fifo_case(case)

    def mutate(_scorer):
        if fault == "updates":
            current.budget.updates_completed = 1
        elif fault == "replay":
            current.budget.replay_examples_completed = 1
        elif fault == "clock":
            current.budget.clock = lambda: 10.0
        elif fault == "started":
            current.budget.started_at += 1.0
        elif fault == "config":
            current.budget.budget = deepcopy(current.budget.budget)
        else:
            current.budget.progress = object()

    current.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)
    assert caught.value.report["work"]["live_prediction_calls_returned"] == 1
    assert caught.value.report["work"]["environment_steps_returned"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "identity_fork",
        "prediction_error",
        "bad_probabilities",
        "step_error",
        "close_error",
        "readback_error",
    ],
)
def test_should_preserve_partial_paid_prefix_and_unreached_games_on_failure(case, fault):
    current = fifo_case(case)
    current.work[fault] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)
    report = caught.value.report
    assert report["status"] == "failed" and report["phase"] == "after_return"
    assert report["native_retirement_metadata_bytes"] == 16
    assert len(report["episodes"]) == 4
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert all(slot["status"] == "not_run" for slot in report["episodes"][1:])
    if fault == "identity_fork":
        assert not current.work.get("environments")
        assert report["work"]["live_prediction_calls_attempted"] == 0
    else:
        assert len(current.work["environments"]) == 1 and current.work["environments"][0].closed
        assert report["episodes"][0]["status"] == "failed"
        assert report["work"]["live_prediction_calls_returned"] == (
            0
            if fault == "prediction_error"
            else 1
            if fault in ("bad_probabilities", "step_error")
            else 2
        )


def test_should_recheck_the_parent_after_the_last_fifo_snapshot_callback(case):
    current = fifo_case(case)
    changed = [False]

    def late_snapshot(port):
        if port.child and current.work.get("returns"):
            changed[0] = True

    def parent_boundary(_owner):
        if changed[0]:
            raise ValueError("preceding paid FIFO return changed")

    current.work.update(on_snapshot=late_snapshot, on_boundary=parent_boundary)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)
    assert caught.value.report["work"]["live_prediction_calls_returned"] == 1
    assert caught.value.report["work"]["environment_steps_returned"] == 0


def test_should_refuse_incompatible_fifo_source_before_fork_or_games(case):
    current = fifo_case(case)
    current.source.value = snapshot(32, fifo=True)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluate(current, saved_fifo=True)
    assert current.work.get("boundaries", 0) > 0
    assert not current.work.get("forks") and not current.work.get("environments")


@pytest.mark.parametrize("fault", ["counts", "order", "evictions", "retired"])
def test_should_bind_initial_source_full_state_even_when_policy_and_clocks_match(case, fault):
    current = fifo_case(case, 128, retired=True)
    native = current.source.value.native_state
    if fault == "order":
        object.__setattr__(native, "entries", tuple(reversed(native.entries)))
    elif fault == "evictions":
        object.__setattr__(native, "evictions", native.evictions + 1)
    else:
        entry = native.entries[0]
        counts = entry.counts
        if fault == "counts":
            object.__setattr__(entry, "counts", (counts[0] - 1, 1, *counts[2:]))
        else:
            object.__setattr__(entry, "counts", (counts[0] - 1, *counts[1:]))
            object.__setattr__(native, "retired_examples", native.retired_examples + 1)

    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)

    assert caught.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert not current.work.get("forks") and not current.work.get("environments")


def test_should_stop_an_expired_endpoint_without_renewing_its_clock_or_allowance(case):
    current = fifo_case(case)
    now = [10.0]
    current.budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0, max_wall_seconds=1.0),
        lambda: now[0],
    )
    now[0] = 12.0
    config, clock, started = current.budget.budget, current.budget.clock, current.budget.started_at
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(current, saved_fifo=True)
    assert caught.value.report["work"]["live_prediction_calls_attempted"] == 0
    assert not current.work.get("forks") and not current.work.get("environments")
    assert current.budget.budget is config and current.budget.clock is clock
    assert current.budget.started_at == started
    assert current.budget.updates_completed == current.budget.replay_examples_completed == 0
