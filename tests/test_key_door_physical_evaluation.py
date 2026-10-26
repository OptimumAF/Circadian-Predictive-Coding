"""Physical checkpoint identity and partial paid evidence with helper doubles."""

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

import src.adapters.key_door_physical_evaluation as evaluation
from src.app.coupled_attribution import AttributionBatch
from src.core.key_door_navigation import KeyDoorLayout


@dataclass
class State:
    completed_updates: int = 0
    training_examples: int = 0
    marker: float = 0.0


def inputs(native_origin=(0, 0)):
    physicals = (0, 24, 48, 72, 96)
    layouts = tuple(KeyDoorLayout(context) for context in ("A", "B", "C", "D"))
    contexts = tuple(
        f"{layout.layout_id}-{color}" for layout in layouts for color in ("red", "blue")
    )
    final = tuple(
        AttributionBatch(
            "final_test",
            context,
            tuple(f"final-{context}-{i}" for i in range(8)),
            np.zeros((8, 8), dtype=np.float64),
            np.zeros((8, 1), dtype=np.float64),
            num_classes=5,
        )
        for context in contexts
    )
    row = dict(
        method="recurrent_rl",
        replay_enabled=False,
        live_sessions_started=0,
        quota_checkpoints=[
            dict(physical_steps=p, native_updates=native_origin[0]) for p in physicals
        ],
        episodes=[
            dict(
                checkpoint_updates=native_origin[0],
                checkpoint_physical_steps=p,
                context=layout.layout_id,
                cue=cue,
                layout_partition="heldout_layout" if layout.layout_id == "D" else "known_layout",
                status="not_run",
                reason="not reached",
                state_unchanged=None,
            )
            for p in physicals
            for layout in layouts
            for cue in (1, -1)
        ],
    )
    if native_origin[0]:
        row["training"] = [
            dict(
                index=index,
                phase="demonstration",
                context="A",
                cue=1 if index % 2 == 0 else -1,
                status="complete",
            )
            for index in range(native_origin[0])
        ]
    return (
        row,
        object(),
        [(native_origin[0], State(*native_origin)) for _ in physicals],
        final,
        object(),
        layouts,
    )


def helpers(
    monkeypatch, *, fail_score=None, fail_live=None, mutate_past=False, native_origin=(0, 0)
):
    calls: dict[str, Any] = dict(score=[], live=[])

    def score(source, states, training, final, budget, **options):
        stage = len(calls["score"])
        calls["score"].append(deepcopy(states))
        assert len(states) == 1 and states[0][0] == native_origin[0] and training == ()
        assert options["verify_state"] is True
        if native_origin[0]:
            assert options["training_contexts"] == tuple(
                "A-red" if index % 2 == 0 else "A-blue" for index in range(native_origin[0])
            )
            context = options["training_contexts"][-1]
        else:
            assert "training_contexts" not in options
            context = None
        progress = options["score_progress"]
        scores: dict[str, Any] = {}
        point = dict(
            updates=native_origin[0], training_context=context, scores=scores, status="incomplete"
        )
        progress.update(
            status="running",
            reason=None,
            checkpoints=[point],
            prediction_calls_attempted=0,
            prediction_calls_completed=0,
            prediction_rows_attempted=0,
            prediction_rows_completed=0,
        )
        try:
            if mutate_past and stage == 1:
                row["stream_result"]["checkpoints"][0]["scores"]["A-red"]["ce"] = 99.0
            options["check_boundary"]()
            for index, batch in enumerate(final):
                progress["prediction_calls_attempted"] += 1
                progress["prediction_rows_attempted"] += 8
                if fail_score == stage and index == 2:
                    raise RuntimeError("third score failed")
                progress["prediction_calls_completed"] += 1
                progress["prediction_rows_completed"] += 8
                scores[batch.context] = dict(ce=1.0 + stage * 0.1 + index * 0.01, accuracy=0.25)
                options["check_boundary"]()
            point["status"] = progress["status"] = "complete"
            return [dict(updates=native_origin[0], training_context=context, scores=scores)]
        except Exception as error:
            progress.update(status="failed", reason=str(error))
            raise

    def live(view, source, states, budget, **options):
        stage = len(calls["live"])
        calls["live"].append(deepcopy(states))
        assert len(states) == 1 and states[0][0] == native_origin[0]
        assert len(view["episodes"]) == 8
        assert options["condition_on_legal_actions"] is True
        assert tuple(layout.layout_id for layout in options["declared_layouts"]) == (
            "A",
            "B",
            "C",
            "D",
        )
        for index, slot in enumerate(view["episodes"]):
            options["check_boundary"]()
            view["live_sessions_started"] += 1
            if fail_live == stage and index == 2:
                slot.update(
                    status="failed", reason="live third session failed", state_unchanged=False
                )
                raise RuntimeError("live third session failed")
            slot.update(status="complete", reason=None, state_unchanged=True, success=False)
            options["check_boundary"]()

    row, source, states, final, budget, layouts = inputs(native_origin)
    monkeypatch.setattr(evaluation, "_score_checkpoints", score)
    monkeypatch.setattr(evaluation.shared, "_live", live)
    return calls, (row, source, states, final, budget, layouts)


def run(arguments, boundary=lambda: None, *, native_origin=None):
    row, source, states, final, budget, layouts = arguments
    evaluation.evaluate_frozen_physical_checkpoints(
        row,
        source,
        states,
        final,
        budget,
        declared_layouts=layouts,
        check_boundary=boundary,
        **({"native_origin": native_origin} if native_origin is not None else {}),
    )
    return row


def test_should_keep_five_native_zero_score_groups_and_forty_distinct_physical_slots(monkeypatch):
    calls, arguments = helpers(monkeypatch)
    row = run(arguments)
    result = row["stream_result"]
    assert result["completed_updates"] == result["training_examples"] == 0
    assert result["applied_updates"] == [] and result["score_forward_calls"] == 40
    assert [point["updates"] for point in result["checkpoints"]] == [0] * 5
    assert [point["physical_steps"] for point in result["checkpoints"]] == [0, 24, 48, 72, 96]
    assert len(calls["score"]) == len(calls["live"]) == 5
    assert row["live_sessions_started"] == 40
    assert all(slot["status"] == "complete" for slot in row["episodes"])
    assert [slot["checkpoint_physical_steps"] for slot in row["episodes"]] == [
        p for p in (0, 24, 48, 72, 96) for _ in range(8)
    ]
    assert all(point["status"] == "complete" for point in row["physical_evaluation_progress"])
    assert row["outcomes"] == pytest.approx(
        dict(new_B_gain=-0.2, new_C_gain=-0.3, retention_loss=0.2, return_gain=-0.1)
    )


def test_should_keep_paid_partial_scores_and_refuse_later_stages(monkeypatch):
    calls, arguments = helpers(monkeypatch, fail_score=1)
    with pytest.raises(RuntimeError, match="third score"):
        run(arguments)
    row = arguments[0]
    assert row["stream_result"]["score_forward_calls"] == 10
    assert len(calls["score"]) == 2 and len(calls["live"]) == 1
    assert row["live_sessions_started"] == 8 and row["outcomes"] is None
    progress = row["physical_evaluation_progress"][1]["score_progress"]
    assert progress["prediction_calls_attempted"] == 3
    assert progress["prediction_calls_completed"] == 2
    assert set(progress["checkpoints"][0]["scores"]) == {"A-red", "A-blue"}
    assert row["stream_result"]["checkpoints"][1]["physical_steps"] == 24
    assert [entry["status"] for entry in row["physical_evaluation_progress"]] == [
        "complete",
        "failed",
        "not_run",
        "not_run",
        "not_run",
    ]


def test_should_copy_back_started_live_sessions_on_failure_and_keep_native_zero(monkeypatch):
    calls, arguments = helpers(monkeypatch, fail_live=1)
    with pytest.raises(RuntimeError, match="third session"):
        run(arguments)
    row = arguments[0]
    assert row["live_sessions_started"] == 11
    assert row["stream_result"]["score_forward_calls"] == 16
    assert len(calls["score"]) == len(calls["live"]) == 2
    assert all(slot["checkpoint_updates"] == 0 for slot in row["episodes"])
    assert row["episodes"][10]["checkpoint_physical_steps"] == 24
    assert all(slot["status"] == "not_run" for slot in row["episodes"][11:])


def test_should_refuse_mutated_preceding_score_evidence_before_more_work(monkeypatch):
    calls, arguments = helpers(monkeypatch, mutate_past=True)
    with pytest.raises(ValueError, match="preceding physical score evidence"):
        run(arguments)
    row = arguments[0]
    assert len(calls["score"]) == 2 and len(calls["live"]) == 1
    assert row["stream_result"]["score_forward_calls"] == 8
    assert row["physical_evaluation_progress"][2]["status"] == "not_run"


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "native_index",
        "native_bool",
        "native_updates",
        "native_rows",
        "changed_state",
        "physical_bool",
        "slot_count",
        "slot_native",
        "slot_physical",
        "final_role",
        "layouts",
    ],
)
def test_should_refuse_invalid_native_or_physical_schema_before_external_helpers(
    monkeypatch, fault
):
    calls, arguments = helpers(monkeypatch)
    row, source, states, final, budget, layouts = arguments
    if fault == "count":
        states.pop()
    elif fault == "native_index":
        states[2] = (24, State())
    elif fault == "native_bool":
        states[0] = (False, State())
    elif fault == "native_updates":
        states[2][1].completed_updates = 1
    elif fault == "native_rows":
        states[2][1].training_examples = 8
    elif fault == "changed_state":
        states[2][1].marker = 1.0
    elif fault == "physical_bool":
        row["quota_checkpoints"][0]["physical_steps"] = False
    elif fault == "slot_count":
        row["episodes"].pop()
    elif fault == "slot_native":
        row["episodes"][8]["checkpoint_updates"] = 24
    elif fault == "slot_physical":
        row["episodes"][8]["checkpoint_physical_steps"] = 0
    elif fault == "final_role":
        final = (replace(final[0], role="train"), *final[1:])
    elif fault == "layouts":
        layouts = (*layouts[:3], layouts[0])
    with pytest.raises(ValueError):
        run((row, source, states, final, budget, layouts))
    assert calls == dict(score=[], live=[])


def test_should_keep_active_paid_score_ledger_private_and_refuse_published_entry_replacement(
    monkeypatch,
):
    calls, arguments = helpers(monkeypatch)
    row = arguments[0]
    boundary_calls = 0

    def callback():
        nonlocal boundary_calls
        boundary_calls += 1
        if boundary_calls == 3:
            entry = row["physical_evaluation_progress"][0]
            assert entry["score_progress"] is None
            entry["score_progress"] = dict(prediction_calls_completed=999)

    with pytest.raises(ValueError, match="physical score progress changed"):
        run(arguments, callback)
    assert len(calls["score"]) == 1 and calls["live"] == []
    assert row["stream_result"]["score_forward_calls"] == 1
    entry = row["physical_evaluation_progress"][0]
    assert entry["status"] == "failed"
    assert entry["score_progress"]["prediction_calls_completed"] == 1
    assert row["physical_evaluation_progress_before_publish"][0]["score_progress"] == dict(
        prediction_calls_completed=999
    )


@pytest.mark.parametrize("mutation", ["completed_count", "returned_score", "future_entry"])
def test_should_refuse_changed_active_or_unreached_published_progress_during_live(
    monkeypatch, mutation
):
    calls, arguments = helpers(monkeypatch)
    row = arguments[0]
    original_live = evaluation.shared._live

    def live(view, *args, **options):
        original_live(view, *args, **options)
        progress = row["physical_evaluation_progress"]
        if mutation == "completed_count":
            progress[0]["score_progress"]["prediction_calls_completed"] = 0
        elif mutation == "returned_score":
            progress[0]["score_progress"]["checkpoints"][0]["scores"]["A-red"]["ce"] = 99.0
        else:
            progress[1]["status"] = "complete"
        options["check_boundary"]()

    monkeypatch.setattr(evaluation.shared, "_live", live)
    with pytest.raises(ValueError, match="physical score progress changed"):
        run(arguments)
    assert len(calls["score"]) == len(calls["live"]) == 1
    assert row["stream_result"]["score_forward_calls"] == 8
    assert row["live_sessions_started"] == 8
    assert row["physical_evaluation_progress"][0]["status"] == "failed"
    assert row["outcomes"] is None


@pytest.mark.parametrize("changed_counter", [999, False])
def test_should_refuse_parent_counter_change_after_last_live_callback_and_retain_observed_copyback(
    monkeypatch, changed_counter
):
    calls, arguments = helpers(monkeypatch)
    row = arguments[0]
    original_live = evaluation.shared._live

    def live(view, *args, **options):
        original_live(view, *args, **options)
        # Simulate a source callback after the last helper boundary. This must
        # fail even if Python equality considers False equal to prior count0.
        row["live_sessions_started"] = changed_counter

    monkeypatch.setattr(evaluation.shared, "_live", live)
    with pytest.raises(ValueError, match="parent live session counter changed"):
        run(arguments)
    assert row["live_sessions_started"] == 8
    assert row["physical_evaluation_live_counter_before_copyback"] == changed_counter
    assert type(row["physical_evaluation_live_counter_before_copyback"]) is type(changed_counter)
    assert row["physical_evaluation_progress"][0]["status"] == "failed"
    assert len(calls["score"]) == len(calls["live"]) == 1
    assert row["outcomes"] is None


@pytest.mark.parametrize(
    "field, replacement",
    [("checkpoint_updates", False), ("checkpoint_physical_steps", False), ("cue", True)],
)
def test_should_refuse_equal_valued_boolean_episode_identity_changes(
    monkeypatch, field, replacement
):
    calls, arguments = helpers(monkeypatch)
    row = arguments[0]

    def live(view, *args, **options):
        row["episodes"][0][field] = replacement
        options["check_boundary"]()

    monkeypatch.setattr(evaluation.shared, "_live", live)
    with pytest.raises(ValueError, match="physical episode identity changed"):
        run(arguments)
    assert len(calls["score"]) == 1 and calls["live"] == []
    assert row["stream_result"]["score_forward_calls"] == 8
    assert row["live_sessions_started"] == 0


def test_should_refuse_equal_valued_boolean_live_counter_during_callback(monkeypatch):
    calls, arguments = helpers(monkeypatch)
    row = arguments[0]

    def live(view, *args, **options):
        row["live_sessions_started"] = False
        options["check_boundary"]()

    monkeypatch.setattr(evaluation.shared, "_live", live)
    with pytest.raises(ValueError, match="parent live session counter changed"):
        run(arguments)
    assert len(calls["score"]) == 1 and calls["live"] == []
    assert row["physical_evaluation_live_counter_before_copyback"] is False
    assert row["live_sessions_started"] == 0


@pytest.mark.parametrize("native_origin", [(2, 16), (192, 1536)])
def test_should_preserve_acquired_native_origin_at_all_five_physical_stages(
    monkeypatch, native_origin
):
    calls, arguments = helpers(monkeypatch, native_origin=native_origin)
    original_training = deepcopy(arguments[0]["training"])
    original_states = deepcopy(arguments[2])
    row = run(arguments, native_origin=native_origin)
    result = row["stream_result"]
    assert (result["completed_updates"], result["training_examples"]) == native_origin
    assert result["applied_updates"] == []
    assert result["score_forward_calls"] == row["live_sessions_started"] == 40
    assert [point["updates"] for point in result["checkpoints"]] == [native_origin[0]] * 5
    assert [point["physical_steps"] for point in result["checkpoints"]] == [0, 24, 48, 72, 96]
    assert all(point["training_context"] == "A-blue" for point in result["checkpoints"])
    assert all(slot["checkpoint_updates"] == native_origin[0] for slot in row["episodes"])
    assert len(calls["score"]) == len(calls["live"]) == 5
    assert row["training"] == original_training and arguments[2] == original_states


def test_should_retain_acquired_origin_and_paid_partial_score_calls_on_failure(monkeypatch):
    calls, arguments = helpers(monkeypatch, native_origin=(2, 16), fail_score=1)
    with pytest.raises(RuntimeError, match="third score"):
        run(arguments, native_origin=(2, 16))
    row = arguments[0]
    result = row["stream_result"]
    assert (result["completed_updates"], result["training_examples"]) == (2, 16)
    assert result["score_forward_calls"] == 10 and result["applied_updates"] == []
    assert [point["updates"] for point in result["checkpoints"]] == [2, 2]
    assert [point["physical_steps"] for point in result["checkpoints"]] == [0, 24]
    assert row["live_sessions_started"] == 8
    assert len(calls["score"]) == 2 and len(calls["live"]) == 1
    assert row["physical_evaluation_progress"][2]["status"] == "not_run"


def test_should_keep_explicit_zero_origin_equal_to_omitted_default(monkeypatch):
    omitted_calls, omitted_arguments = helpers(monkeypatch)
    omitted = run(omitted_arguments)
    explicit_calls, explicit_arguments = helpers(monkeypatch)
    explicit = run(explicit_arguments, native_origin=(0, 0))
    assert explicit == omitted
    assert explicit_calls == omitted_calls


@pytest.mark.parametrize(
    "native_origin",
    [
        None,
        [0, 0],
        (),
        (0,),
        (0, 0, 0),
        (False, 0),
        (0, True),
        (0.0, 0),
        (0, 0.0),
        (-1, 0),
        (0, -1),
        (513, 4104),
        (0, 8),
        (2, 15),
        (192, 1535),
        (np.int64(2), 16),
    ],
)
def test_should_refuse_invalid_native_origin_before_external_work(monkeypatch, native_origin):
    calls, arguments = helpers(monkeypatch)
    row, source, states, final, budget, layouts = arguments
    with pytest.raises(ValueError, match="native.origin"):
        evaluation.evaluate_frozen_physical_checkpoints(
            row,
            source,
            states,
            final,
            budget,
            declared_layouts=layouts,
            check_boundary=lambda: None,
            native_origin=native_origin,
        )
    assert calls == dict(score=[], live=[])
    assert "stream_result" not in row


@pytest.mark.parametrize(
    "fault",
    ["native_index", "native_updates", "native_rows", "changed_state", "quota", "slot"],
)
def test_should_refuse_disagreement_with_acquired_native_origin_before_work(monkeypatch, fault):
    calls, arguments = helpers(monkeypatch, native_origin=(2, 16))
    row, _, states, *_ = arguments
    if fault == "native_index":
        states[4] = (0, State(2, 16))
    elif fault == "native_updates":
        states[4][1].completed_updates = 0
    elif fault == "native_rows":
        states[4][1].training_examples = 0
    elif fault == "changed_state":
        states[4][1].marker = 1.0
    elif fault == "quota":
        row["quota_checkpoints"][4]["native_updates"] = 0
    else:
        row["episodes"][39]["checkpoint_updates"] = 0
    with pytest.raises(ValueError):
        run(arguments, native_origin=(2, 16))
    assert calls == dict(score=[], live=[])
    assert "stream_result" not in row


@pytest.mark.parametrize(
    "fault",
    ["absent", "short", "phase", "status", "context", "cue", "cue_bool", "index_bool"],
)
def test_should_require_actual_completed_ordered_prefix_contexts_before_work(monkeypatch, fault):
    calls, arguments = helpers(monkeypatch, native_origin=(2, 16))
    row = arguments[0]
    if fault == "absent":
        row.pop("training")
    elif fault == "short":
        row["training"].pop()
    else:
        key, value = {
            "phase": ("phase", "reward"),
            "status": ("status", "failed"),
            "context": ("context", "B"),
            "cue": ("cue", -1),
            "cue_bool": ("cue", True),
            "index_bool": ("index", False),
        }[fault]
        row["training"][0][key] = value
    with pytest.raises(ValueError, match="prefix"):
        run(arguments, native_origin=(2, 16))
    assert calls == dict(score=[], live=[])


@pytest.mark.parametrize("mutation", ["state", "result", "quota", "prefix_context", "prefix_list"])
def test_should_refuse_changed_native_origin_or_prefix_binding_during_callbacks(
    monkeypatch, mutation
):
    calls, arguments = helpers(monkeypatch, native_origin=(2, 16))
    row, _, states, *_ = arguments
    original_live = evaluation.shared._live

    def live(view, *args, **options):
        original_live(view, *args, **options)
        if mutation == "state":
            states[0][1].training_examples = 0
        elif mutation == "result":
            row["stream_result"]["completed_updates"] = 0
        elif mutation == "quota":
            row["quota_checkpoints"][0]["native_updates"] = 0
        elif mutation == "prefix_context":
            row["training"][0]["context"] = "B"
        else:
            row["training"] = deepcopy(row["training"])
        options["check_boundary"]()

    monkeypatch.setattr(evaluation.shared, "_live", live)
    with pytest.raises(ValueError):
        run(arguments, native_origin=(2, 16))
    assert len(calls["score"]) == len(calls["live"]) == 1
    assert row["stream_result"]["score_forward_calls"] == 8
    assert row["live_sessions_started"] == 8
    assert row["physical_evaluation_progress"][0]["status"] == "failed"
    assert row["physical_evaluation_progress"][1]["status"] == "not_run"
    assert row["outcomes"] is None
