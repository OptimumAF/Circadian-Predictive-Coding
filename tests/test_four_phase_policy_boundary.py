"""Explicit declared layouts and fixed four-phase evaluation composition with fakes."""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import key_door_pilot as shared
from src.adapters import return_policy_evaluation as evaluator
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.key_door_navigation import KeyDoorLayout
from test_return_evaluation_boundaries import (
    Learner,
    live_fakes as live_fakes,
    row as live_row,
    state,
)


def descriptors():
    original = shared.layouts()
    return (*original[:2], shared.third_training_layout(), original[3])


def test_should_declare_distinct_third_train_geometry_without_mutating_historical_layouts():
    original = tuple(asdict(layout) for layout in shared.layouts())
    first, second = shared.third_training_layout(), shared.third_training_layout()
    assert first == second and first is not second
    legacy_layout = dict(
        layout_id="C_train", rotation=2, branch_row=3, swap_keys=True, appearance=-0.75
    )
    assert asdict(first) == {**legacy_layout, "movement_action_map": (0, 1, 2, 3)}
    assert shared.key_door_layout_record(first) == legacy_layout
    assert tuple(asdict(layout) for layout in shared.layouts()) == original
    assert shared.layouts()[2].layout_id == "C"


def test_should_allocate_all40_slots_with_current_training_partition_and_keep_old_defaults():
    assert evaluator.new_evaluation() == evaluator.new_evaluation(four_phase=False)
    assert len(evaluator.new_evaluation()["episodes"]) == 32
    row = evaluator.new_evaluation(four_phase=True)
    assert len(row["episodes"]) == 40
    assert (
        tuple(dict.fromkeys(slot["checkpoint_updates"] for slot in row["episodes"]))
        == evaluator.FOUR_PHASE_EVALUATION_POINTS
    )
    assert tuple(dict.fromkeys(slot["context"] for slot in row["episodes"])) == (
        "A",
        "B",
        "C_train",
        "D",
    )
    assert all(
        slot["layout_partition"] == ("heldout_layout" if slot["context"] == "D" else "known_layout")
        for slot in row["episodes"]
    )
    assert all(
        slot["status"] == "not_run" and slot["state_unchanged"] is None for slot in row["episodes"]
    )


@pytest.mark.parametrize("mode", [0, 1, None, np.bool_(True), "true"])
@pytest.mark.parametrize("endpoint", ["new", "evaluate"])
def test_should_refuse_nonbool_route_before_source_budget_layout_or_row_work(
    monkeypatch, mode, endpoint
):
    def forbidden(*args, **kwargs):
        raise AssertionError("resources were reached")

    monkeypatch.setattr(shared, "layouts", forbidden)
    if endpoint == "new":
        with pytest.raises(ValueError, match="exact bool"):
            evaluator.new_evaluation(four_phase=mode)
    else:
        untouched: dict[str, Any] = {}
        with pytest.raises(ValueError, match="exact bool"):
            evaluator.evaluate_policy(
                untouched,
                object(),
                [],
                object(),
                training=(),
                check_boundary=forbidden,
                four_phase=mode,
            )
        assert untouched == {}


class TupleSubclass(tuple):
    pass


class LayoutSubclass(KeyDoorLayout):
    pass


@pytest.mark.parametrize(
    "kind",
    [
        "list",
        "empty",
        "too_many",
        "tuple_subclass",
        "layout_subclass",
        "duplicate",
        "corrupted",
        "missing_context",
    ],
)
def test_should_validate_explicit_layouts_before_live_source_work(live_fakes, kind):
    report, source = live_row(), Learner()
    layouts: Any = descriptors()
    if kind == "list":
        layouts = list(layouts)
    elif kind == "empty":
        layouts = ()
    elif kind == "too_many":
        layouts = layouts * 9
    elif kind == "tuple_subclass":
        layouts = TupleSubclass(layouts)
    elif kind == "layout_subclass":
        layouts = (LayoutSubclass("A"), *layouts[1:])
    elif kind == "duplicate":
        layouts = (*layouts, layouts[0])
    elif kind == "corrupted":
        object.__setattr__(layouts[0], "rotation", 5)
    else:
        layouts = layouts[:2]
    with pytest.raises(ValueError):
        shared._live(report, source, [(192, state())], session(), declared_layouts=layouts)
    assert source.trace == []


def test_should_use_explicit_third_train_layout_without_replacing_global_defaults(live_fakes):
    report, source = live_row(), Learner()
    report["episodes"][2]["context"] = "C_train"
    report["episodes"][2]["layout_partition"] = "known_layout"
    before = tuple(asdict(layout) for layout in shared.layouts())
    shared._live(
        report,
        source,
        [(192, state())],
        session(),
        task_sequence=("A", "B", "C_train", "A"),
        declared_layouts=descriptors(),
    )
    assert all(slot["status"] == "complete" for slot in report["episodes"])
    assert all(
        slot["identity"]["task_sequence"] == ("A", "B", "C_train", "A")
        for slot in report["episodes"]
    )
    assert report["live_sessions_started"] == 4
    assert tuple(asdict(layout) for layout in shared.layouts()) == before


def session(updates=256):
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=updates), lambda: 0.0)
    budget.updates_completed = updates
    return budget


@dataclass
class Packet:
    episode_id: str
    context: str
    role: str


class CachedLearner:
    def __init__(self, calls=None, fail=False):
        self.state = 0
        self.calls = [] if calls is None else calls
        self.fail = fail

    def fork(self):
        return CachedLearner(self.calls, self.fail)

    def restore_state(self, state):
        self.state = state

    def snapshot_state(self):
        return self.state

    def predict(self, features):
        self.calls.append(self.state)
        if self.fail:
            raise RuntimeError("cached prediction failed")
        return np.full((len(features), 5), 0.2)


@pytest.fixture
def composition(monkeypatch):
    collected, live_options = [], []
    gate = {"ready": True, "corrupt_after_collection": False}

    def guard():
        if not gate["ready"]:
            raise ValueError("completed stream guard refused")

    def collect(identity, layout, cue, role):
        guard()
        collected.append((identity, layout, cue, role))
        if gate["corrupt_after_collection"]:
            gate["ready"] = False
        context = f"{layout.layout_id}-{'red' if cue == 1 else 'blue'}"
        return SimpleNamespace(
            episode=Packet(identity, context, role),
            context=context,
            row_ids=(identity,),
            features=np.zeros((8, 8)),
            targets=np.zeros((8, 1)),
            num_classes=5,
        )

    def live(row, source, states, budget, **kwargs):
        live_options.append(kwargs)
        kwargs["check_boundary"]()
        for slot in row["episodes"]:
            slot.update(
                status="complete",
                reason=None,
                state_unchanged=True,
                success=False,
                **{"return": 0},
                prediction_calls_attempted=1,
                prediction_calls_returned=1,
                steps_executed=0,
            )
        row["live_sessions_started"] = len(row["episodes"])

    # These component fakes isolate routing/counters. Full episode validation and
    # caller receipt authority are exercised by the separate integration suite.
    monkeypatch.setattr(shared, "_collect", collect)
    monkeypatch.setattr(shared, "_live", live)
    monkeypatch.setattr(evaluator, "_validate_batch", lambda *args: None)
    monkeypatch.setattr(evaluator, "_validated_batches", lambda training, final: (training, final))
    return SimpleNamespace(collected=collected, live_options=live_options, gate=gate, guard=guard)


def training(updates):
    sequence = ("A", "B", "C_train", "A") if updates == 256 else ("A", "B", "A")
    return tuple(
        SimpleNamespace(
            context=f"{sequence[index // 64]}-{'red' if index % 2 == 0 else 'blue'}",
            row_ids=(f"train-{index}",),
        )
        for index in range(updates)
    )


def evaluate(composition, *, four_phase=True, source=None, row=None, budget=None, states=None):
    updates = 256 if four_phase else 192
    points = evaluator.FOUR_PHASE_EVALUATION_POINTS if four_phase else evaluator.EVALUATION_POINTS
    row = evaluator.new_evaluation(four_phase=four_phase) if row is None else row
    source = CachedLearner() if source is None else source
    budget = session(updates) if budget is None else budget
    evaluator.evaluate_policy(
        row,
        source,
        [(point, point) for point in points] if states is None else states,
        budget,
        training=training(updates),
        check_boundary=composition.guard,
        four_phase=four_phase,
    )
    return row, source, budget


def test_should_route_allfive_checkpoints_eight_finals_and40_live_slots_after_guard(composition):
    row, source, budget = evaluate(composition)
    assert row["status"] == "complete" and budget.updates_completed == 256
    assert len(composition.collected) == 8
    assert {entry[1].layout_id for entry in composition.collected} == {"A", "B", "C_train", "D"}
    assert all(
        entry[0].startswith("four-phase-final-") and entry[3] == "final_test"
        for entry in composition.collected
    )
    assert source.calls == [
        point for point in evaluator.FOUR_PHASE_EVALUATION_POINTS for _ in range(8)
    ]
    assert row["work"]["cached_forwards_completed"] == 40
    assert row["work"]["cached_rows_completed"] == 320
    assert row["live_sessions_started"] == row["work"]["live_callback_returns"] == 40
    assert [point["updates"] for point in row["outcomes"]] == list(
        evaluator.FOUR_PHASE_EVALUATION_POINTS
    )
    assert set(row["outcomes"][-1]["by_layout"]) == {"A", "B", "C_train", "D"}
    options = composition.live_options[0]
    assert options["task_sequence"] == ("A", "B", "C_train", "A")
    assert options["declared_layouts"] == descriptors()
    assert source.state == 0


def test_should_preserve_old_evaluation_layouts_points_live_kwargs_and_packet_ids(composition):
    row, source, budget = evaluate(composition, four_phase=False)
    assert row["status"] == "complete" and budget.updates_completed == 192
    assert source.calls == [point for point in evaluator.EVALUATION_POINTS for _ in range(8)]
    assert {entry[1].layout_id for entry in composition.collected} == {"A", "B", "C", "D"}
    assert all(entry[0].startswith("return-final-") for entry in composition.collected)
    assert set(composition.live_options[0]) == {"task_sequence", "check_boundary"}
    assert len(row["episodes"]) == 32


@pytest.mark.parametrize("problem", ["guard", "budget", "points", "slots"])
def test_should_refuse_incomplete_original_authority_before_final_collection(composition, problem):
    row = evaluator.new_evaluation(four_phase=True)
    budget = session()
    states = [(point, point) for point in evaluator.FOUR_PHASE_EVALUATION_POINTS]
    if problem == "guard":
        composition.gate["ready"] = False
    elif problem == "budget":
        budget.updates_completed = 255
    elif problem == "points":
        states.pop()
    else:
        row["episodes"].pop()
    with pytest.raises(ValueError):
        evaluate(composition, row=row, budget=budget, states=states)
    assert row["status"] == "failed" and composition.collected == []
    assert row["work"]["final_collections_attempted"] == 0
    assert budget.updates_completed == (255 if problem == "budget" else 256)


def test_should_preserve_one_completed_collection_on_post_collection_guard_failure(composition):
    composition.gate["corrupt_after_collection"] = True
    row = evaluator.new_evaluation(four_phase=True)
    budget = session()
    with pytest.raises(ValueError, match="guard refused"):
        evaluate(composition, row=row, budget=budget)
    assert (
        row["work"]["final_collections_attempted"]
        == row["work"]["final_collections_completed"]
        == 1
    )
    assert len(row["final_demonstrations"]) == 1
    assert row["checkpoints"] == [] and composition.live_options == []
    assert budget.updates_completed == 256


def test_should_preserve_cached_failure_attempts_without_publishing_partial_points_or_live_work(
    composition,
):
    row = evaluator.new_evaluation(four_phase=True)
    source, budget = CachedLearner(fail=True), session()
    with pytest.raises(RuntimeError, match="cached prediction failed"):
        evaluate(composition, row=row, source=source, budget=budget)
    assert row["status"] == "failed" and row["checkpoints"] == []
    assert row["work"]["cached_forwards_attempted"] == 1
    assert row["work"]["cached_rows_attempted"] == 8
    assert row["work"]["cached_forwards_completed"] == 0
    assert composition.live_options == [] and budget.updates_completed == 256
    assert all(slot["status"] == "not_run" for slot in row["episodes"])
