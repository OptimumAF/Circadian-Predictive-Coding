"""Literal whole-composition checks; no solver, native forward or physics."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_controller_learning as app
from src.adapters import minigrid_context_linear_learner as native
from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.controller_objective import ControllerObjective, controller_utility
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from src.shared.process_memory import ProcessRssSampler
from test_a4_adaptation import sources

WORK = dict(owners=0, snapshots=0, updates=0, rows=0, predicts=0, predict_rows=0, games=0)


class LiteralLearner:
    """Scripted categorical heads retain actual typed native fit/row geometry."""

    hook = staticmethod(lambda kind, owner: None)

    def __init__(self):
        WORK["owners"] += 1
        self.state = native.ContextLinearTrainingSnapshot(ContextLinearPolicy())

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return native.copy_context_linear_training_snapshot(self.state)

    def restore_state(self, state):
        self.state = native.copy_context_linear_training_snapshot(state)

    def fork(self):
        child = LiteralLearner()
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, x, y):
        from src.core.learner_ports import TrainingDiagnostic

        WORK["updates"] += 1
        WORK["rows"] += len(x)
        self.hook("train", self)
        _, projected = native._project(x)
        assert len(x) == 9 and y.shape == (9, 1) and np.all(y == y[0])
        cue = int(projected[0, 297])
        assert np.all(projected[:, 296 + cue] == 1)
        marker = tuple(float(i == int(y[0, 0])) for i in range(7))
        fit = CategoricalLinearFit(296, 7, 9, ((0.0,) * 7,) * 296 + (marker,), 1, (0.0,) * 9, 0.0)
        heads = list(self.state.heads)
        heads[cue] = fit
        self.state = replace(
            self.state,
            heads=(heads[0], heads[1]),
            completed_updates=self.state.completed_updates + 1,
            training_examples=self.state.training_examples + len(x),
        )
        return TrainingDiagnostic(native.DIAGNOSTIC, 0.0)

    def predict(self, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        self.hook("predict", self)
        _, projected = native._project(x)
        p = np.full((len(x), 7), 1 / 7, dtype=np.float64)
        for i, head in enumerate(self.state.heads):
            if head is not None:
                p[projected[:, 296 + i] == 1] = head.coefficients[-1]
        return p


class Harness:
    def __init__(self, patch):
        self.time = 0.0
        self.games: list[dict[str, Any]] = []
        self.states = []
        self.train_calls = self.predict_calls = 0
        self.on_train = lambda owner: None
        self.on_predict = lambda owner: None
        self.on_game = lambda result: None
        self.original_context = app._Study.context
        self.patch = patch
        patch.setattr(app, "ContextLinearLearner", LiteralLearner)
        patch.setattr(app, "monotonic", lambda: self.time)
        patch.setattr(LiteralLearner, "hook", staticmethod(self.work))
        patch.setattr(app, "evaluate_context_linear_stream_checkpoint", self.game)

    def work(self, kind, owner):
        self.time += 0.1 if kind == "train" else 0.01
        if kind == "train":
            self.train_calls += 1
            self.on_train(owner)
        else:
            self.predict_calls += 1
            self.on_predict(owner)

    def game(self, cold, state, *, budget, check_boundary, reference_window):
        WORK["games"] += 1
        assert reference_window is True and cold.snapshot_state().completed_updates == 0
        assert (
            2 <= state.completed_updates <= 6
            and state.training_examples == state.completed_updates * 9
        )
        assert all(h.fit_rows == 9 for h in state.heads)
        assert budget.updates_completed == budget.replay_examples_completed == 0
        check_boundary()
        self.time += 0.1
        self.states.append(deepcopy(state))
        rows = []
        for i, (branch, cue) in enumerate(((3, "blue"), (3, "red"), (4, "blue"), (4, "red"))):
            action = int(np.argmax(state.heads[i % 2].coefficients[-1]))
            success = action == (2 if branch == 3 else 5)
            rows.append(
                dict(
                    status="complete",
                    source_role="train",
                    cue=cue,
                    success=success,
                    return_=float(success),
                    predictions=[dict(action=action)],
                    environment=dict(
                        episode_id=f"literal-{len(self.states)}-{i}",
                        role="train",
                        cue=cue,
                        layout=asdict(MiniGridKeyDoorLayout("A", branch_row=branch)),
                        seed=17,
                        max_steps=16,
                        closed=True,
                        terminated=True,
                        truncated=False,
                        failure_work_may_be_unobserved=False,
                        environment_steps=1,
                        transitions=[dict(action=action, applied=True)],
                    ),
                )
            )
            rows[-1]["return"] = rows[-1].pop("return_")
        result = dict(status="complete", state_unchanged=True, episodes=rows)
        self.games.append(result)
        self.on_game(result)
        check_boundary()
        return result

    def run(self, episodes=None, boundary=None):
        with ProcessRssSampler(interval_seconds=0.01) as memory:
            self.budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=150,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                lambda: self.time,
            )
            self.budget.attach_memory(memory)
            return app.run_controller_learning(
                sources() if episodes is None else episodes,
                readout_budget=self.budget,
                check_boundary=boundary,
            )


@pytest.fixture
def harness(monkeypatch):
    return Harness(monkeypatch)


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        result = h.run()
        assert result["status"] == "complete", result
        return result, h


def test_should_preserve_two_warm_fits_and_charge_each_disposable_trial(integrated):
    r, h = integrated
    assert r["base"]["state"]["completed_updates"] == 2
    assert r["base"]["state"]["training_examples"] == 18
    assert r["warmup"]["budget"]["updates"] == 2
    assert [(t["request"]["context"], t["request"]["action_id"]) for t in r["trials"]] == [
        ("clean", "wait"),
        ("clean", "wake1"),
        ("mistakes", "wait"),
        ("mistakes", "wake1"),
    ]
    for t, updates in zip(r["trials"], (0, 4, 0, 4), strict=True):
        assert t["source_unchanged"] and t["actor_unchanged"]
        assert t["budget"]["updates"] == updates and t["budget"]["replay"] == 0
        assert t["window"]["training_examples"] == 9 * updates
        assert t["checkpoints"] == [
            dict(relative_updates=n, native_updates=2 + n, native_rows=18 + 9 * n)
            for n in range(updates + 1)
        ]
        assert t["window"]["window_complete"] is True
        for n, receipt in enumerate(t["window"]["applied_updates"]):
            assert receipt["observed_at"] == n and receipt["arrived_at"] == n + 1
            assert receipt["applied_at"] == n + 1 and receipt["update_number"] == n + 1
    assert r["controller"]["attempts"] == r["controller"]["completed"] == 4
    assert r["controller"]["discarded_trials"] == 4
    assert r["controller"]["clock"] == 4 and r["source_unchanged"]
    assert h.train_calls == sum(b["updates"] for b in r["budgets"])


def test_should_derive_contexts_from_real_train_predictions_in_separate_epoch(integrated):
    r, _ = integrated
    assert [
        (c["context"], c["mistakes"], c["examples"], c["scored_at"]) for c in r["contexts"]
    ] == [("clean", 0, 9, 0), ("mistakes", 9, 9, 2)]
    assert [c["source_episode_id"] for c in r["contexts"]] == [
        sources()[0].episode_id,
        sources()[2].episode_id,
    ]
    assert r["context_work"] == dict(
        fork_attempts=2,
        fork_completed=2,
        predict_attempts=2,
        predict_returned=2,
        rows_attempted=18,
        rows_returned=18,
    )
    assert all(
        c["feedback"]["observed_at"] == c["feedback"]["arrived_at"] == 0 for c in r["contexts"]
    )


def test_should_fit_actual_after_task_game_utility_and_keep_return_separate(integrated):
    r, _ = integrated
    assert r["fitted"]["objective"] == asdict(ControllerObjective())
    for t, estimate in zip(r["trials"], r["fitted"]["estimates"], strict=True):
        m = t["measures"]
        assert (m["old_before"], m["new_before"]) == (2, 0)
        assert (m["old_after"], m["new_after"]) == (
            t["outcomes"]["after_task"]["old_A3"],
            t["outcomes"]["after_task"]["new_A4"],
        )
        assert m["execution_seconds"] > 0 and m["peak_rss_bytes"] > 0
        assert m["violations"] == 0 and m["work_complete"] is True
        assert (
            estimate["mean_utility"]
            == controller_utility(ControllerObjective(), app.ControllerMeasures(**m)).value
        )
    mistake_wake = r["trials"][3]
    assert mistake_wake["outcomes"] == dict(
        before=dict(old_A3=2, new_A4=0),
        after_task=dict(old_A3=0, new_A4=2),
        after_return=dict(old_A3=2, new_A4=0),
        recovery=[True, True],
    )


def test_should_reuse_only_same_full_native_state_and_keep_fixed_bounded_phases(integrated):
    r, h = integrated
    assert r["game_work"] == dict(
        readout_attempts=len(h.games), readout_returns=len(h.games), unreturned_work_unknown=False
    )
    for t in r["trials"][::2]:
        assert [p["reused_from"] for p in t["phases"]] == ["shared_base", "shared_base"]
        assert [p["native_updates"] for p in t["phases"]] == [2, 2]
    for t in r["trials"][1::2]:
        assert [p["native_updates"] for p in t["phases"]] == [4, 6]
        assert [p["reused_from"] for p in t["phases"]] == [None, None]
    # Equal scripted predictions/weights at base and clean native4/6 do not erase history.
    assert r["trials"][1]["outcomes"]["before"] == r["trials"][1]["outcomes"]["after_task"]
    assert len(h.games) >= 5 and r["unadmitted_game"] is None


def test_should_apply_same_frozen_owner_without_fifth_fit_or_budget_renewal(integrated):
    r, _ = integrated
    app_row = r["application"]
    assert app_row["purpose"] == "known_TRAIN_application"
    assert app_row["window"]["controller_application"]["fit"]["discarded_trials"] == 4
    assert app_row["window"]["window_complete"] is True
    assert len(app_row["phases"]) == 2 and len(r["budgets"]) == 6
    assert r["readout_budget"]["updates"] == r["readout_budget"]["replay"] == 0
    assert r["readout_budget"]["started_at"] == 0.0
    assert all(b["limits"]["max_wall_seconds"] == 90 for b in r["budgets"])


@pytest.mark.parametrize("invalid", ["role", "id", "context", "rows"])
def test_should_refuse_invalid_sources_before_constructing_learner(harness, invalid):
    packets = list(sources())
    if invalid == "role":
        packets[2] = replace(packets[2], role="final_test")
    elif invalid == "id":
        packets[2] = replace(packets[2], episode_id="replacement")
    elif invalid == "context":
        packets[2] = replace(packets[2], context="B")
    else:
        object.__setattr__(packets[2], "transitions", packets[2].transitions[:-1])
    before = WORK["owners"]
    r = harness.run(tuple(packets))
    assert r["status"] == "failed" and WORK["owners"] == before
    assert harness.train_calls == 0 and not harness.games


def test_should_refuse_unavailable_clean_context_without_search(harness, monkeypatch):
    original = LiteralLearner.predict

    def wrong(owner, x):
        result = original(owner, x)
        result[:] = np.eye(7)[0]
        return result

    monkeypatch.setattr(LiteralLearner, "predict", wrong)
    r = harness.run()
    assert r["status"] == "failed" and "unavailable" in r["error"]
    assert harness.train_calls == 2 and len(harness.games) == 1 and r["trials"] == []
    assert r["context_work"]["predict_returned"] == 1
    assert r["controller"]["attempts"] == 0 and r["application"] is None


def test_should_keep_unknown_context_prediction_cost_and_stop(harness, monkeypatch):
    def context(study, name, index, clock, row):
        harness.on_predict = lambda owner: (_ for _ in ()).throw(
            RuntimeError("lost context return")
        )
        return harness.original_context(study, name, index, clock, row)

    monkeypatch.setattr(app._Study, "context", context)
    r = harness.run()
    assert r["status"] == "failed" and "lost context" in r["error"]
    assert r["context_work"]["predict_attempts"] == 1 and r["context_work"]["predict_returned"] == 0
    assert r["context_work"]["rows_attempted"] == 9 and r["context_work"]["rows_returned"] == 0
    assert not r["trials"] and r["application"] is None


def test_should_keep_preceding_trial_when_next_update_has_no_return(harness):
    def lose(owner):
        if harness.train_calls == 3:
            raise RuntimeError("lost update return")

    harness.on_train = lose
    r = harness.run()
    assert r["status"] == "failed" and "lost update return" in r["error"]
    assert r["trials"][0]["status"] == "complete" and r["trials"][1]["status"] == "failed"
    assert r["controller"]["attempts"] == 2 and r["controller"]["completed"] == 1
    assert r["controller"]["stopped"] is True
    assert sum(b["updates"] for b in r["budgets"]) == 2 and harness.train_calls == 3
    assert r["fitted"] is r["application"] is None


@pytest.mark.parametrize("damage", ["value", "equal_replacement"])
def test_should_refuse_corrupted_or_replaced_previous_success_and_preserve_as_read(harness, damage):
    original = []

    def corrupt(result):
        if len(harness.games) == 1:
            original.append(deepcopy(result))
        elif len(harness.games) == 2:
            first = harness.games[0]
            if damage == "value":
                first["episodes"][0]["success"] = False
            else:
                first["episodes"][0]["environment"] = deepcopy(first["episodes"][0]["environment"])

    harness.on_game = corrupt
    r = harness.run()
    assert r["status"] == "failed" and "capture_error" in r
    assert r["baseline"] == original[0]
    assert len(harness.games) == 2 and r["controller"]["completed"] == 1
    assert r["application"] is None and r["game_work"]["readout_attempts"] == 2


@pytest.mark.parametrize("damage", ["source", "budget", "expired"])
def test_should_refuse_observed_source_budget_or_clock_changes_without_renewal(harness, damage):
    packets = sources()

    def mutate(owner):
        if harness.train_calls == 3:
            if damage == "source":
                object.__setattr__(packets[0], "episode_id", "mutated")
            elif damage == "budget":
                harness.budget.budget = replace(harness.budget.budget, max_wall_seconds=151)
            else:
                harness.time = 151.0

    harness.on_train = mutate
    r = harness.run(packets)
    assert r["status"] == "failed" and "capture_error" in r
    assert harness.train_calls == 3 and r["application"] is None
    assert r["budgets"][0]["updates"] == 2 and r["readout_budget"]["started_at"] == 0


@pytest.mark.parametrize("invalid", ["action", "incomplete", "return", "role"])
def test_should_refuse_invalid_game_utility_and_keep_returned_evidence(harness, invalid):
    def invalidate(result):
        if invalid == "action":
            result["episodes"][0]["environment"]["transitions"][0]["action"] = 7
        elif invalid == "incomplete":
            result["episodes"][0]["environment"]["closed"] = False
        elif invalid == "return":
            result["episodes"][0]["return"] = float("nan")
        else:
            result["episodes"][0]["source_role"] = "final"

    harness.on_game = invalidate
    r = harness.run()
    assert r["status"] == "failed" and r["baseline"] is None
    assert r["unadmitted_game"] is harness.games[0]
    assert r["game_work"] == dict(
        readout_attempts=1, readout_returns=1, unreturned_work_unknown=False
    )
    assert r["trials"] == [] and r["application"] is None


def test_should_keep_unknown_game_attempt_without_inventing_zero_work(harness):
    harness.on_game = lambda result: (_ for _ in ()).throw(RuntimeError("lost game return"))
    r = harness.run()
    assert r["status"] == "failed" and "lost game" in r["error"]
    assert r["game_work"] == dict(
        readout_attempts=1, readout_returns=0, unreturned_work_unknown=True
    )
    assert r["trials"] == [] and harness.train_calls == 2


def test_should_apply_learned_wake_to_actual_native_prefix(harness):
    def retained_old(result):
        # A deliberately different scripted world tests the positive action
        # path. It is not an experimental result or a change to the objective.
        if result["episodes"][2]["success"]:
            for row in result["episodes"][:2]:
                row["success"] = True
                row["return"] = 1.0

    harness.on_game = retained_old
    r = harness.run()
    assert r["status"] == "complete", r
    assert r["fitted"]["choices"] == dict(clean="wait", mistakes="wake1")
    app_row = r["application"]
    # Oldest-ready A3 packets are mistakes after A4; all four selected fits run.
    assert app_row["window"]["completed_updates"] == 4
    assert app_row["window"]["training_complete"] is True
    assert app_row["budget"]["updates"] == 4 and app_row["window"]["training_examples"] == 36
    assert [p["relative_updates"] for p in app_row["phases"]] == [2, 4]
    assert [p["native_updates"] for p in app_row["phases"]] == [4, 6]
    assert [p["reused_from"] for p in app_row["phases"]] == [None, None]
    assert app_row["window"]["controller_application"]["fit"]["attempts"] == 4


def test_should_retain_native_game_failure_report_without_scoring(harness, monkeypatch):
    failed = dict(status="failed", reason="lost physics", attempted_steps=3, complete_steps=2)

    def fail(*args, **kwargs):
        raise app.MiniGridEvaluationFailed(failed)

    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", fail)
    r = harness.run()
    assert r["status"] == "failed" and r["failed_game"] == r["unadmitted_game"] == failed
    assert r["game_work"]["readout_attempts"] == 1 and r["game_work"]["readout_returns"] == 0
    assert not r["trials"] and harness.train_calls == 2


@pytest.mark.parametrize("source", ["cold", "warm"])
def test_should_refuse_mutated_complete_source_before_later_update(harness, monkeypatch, source):
    owned = []

    def capture(study, name, index, clock, row):
        owned.append(study)
        return harness.original_context(study, name, index, clock, row)

    monkeypatch.setattr(app._Study, "context", capture)

    def corrupt(owner):
        if harness.train_calls == 3:
            model = getattr(owned[0], source)
            model.state = replace(model.state, failed_work=(1, 9, 0, 0, True))

    harness.on_train = corrupt
    r = harness.run()
    assert r["status"] == "failed" and "source changed" in r["capture_error"]
    assert harness.train_calls == 3 and r["application"] is None


@pytest.mark.parametrize("change", ["clock", "limits", "sampler"])
def test_should_refuse_replaced_original_readout_controls_before_next_update(harness, change):
    def corrupt(owner):
        if harness.train_calls == 3:
            if change == "clock":
                harness.budget.clock = lambda: harness.time
            elif change == "limits":
                harness.budget.budget = replace(harness.budget.budget)
            else:
                harness.budget.process_rss_sampler = None

    harness.on_train = corrupt
    r = harness.run()
    assert r["status"] == "failed" and "capture_error" in r
    assert harness.train_calls == 3 and r["readout_budget"]["started_at"] == 0.0
    assert r["application"] is None
