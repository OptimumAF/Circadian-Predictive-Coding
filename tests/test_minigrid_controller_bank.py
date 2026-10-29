"""Existing bank/stream/chooser composition; literal fits and games only."""

from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from scripts import minigrid_controller_learning as app
from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingDiagnostic
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from tests import test_minigrid_controller_learning as old


class BankLiteral(old.LiteralLearner):
    def fork(self):
        child = BankLiteral()
        child.restore_state(self.snapshot_state())
        return child

    def train_batch(self, x, y):
        old.WORK["updates"] += 1
        old.WORK["rows"] += len(x)
        self.hook("train", self)
        _, projected = native._project(x)
        assert len(x) in (9, 18) and y.shape == (len(x), 1)
        cue = int(projected[0, 297])
        assert np.all(projected[:, 296 + cue] == 1)
        # Literal set membership, no parameter-dependent linear forward/solve.
        marker = tuple(float(i in y[:, 0]) for i in range(7))
        fit = CategoricalLinearFit(
            296, 7, len(x), ((0.0,) * 7,) * 296 + (marker,), 0, (0.0,) * len(x), 0.0
        )
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
        old.WORK["predicts"] += 1
        old.WORK["predict_rows"] += len(x)
        self.hook("predict", self)
        _, projected = native._project(x)
        p = np.full((len(x), 7), 1 / 7, dtype=np.float64)
        for i, row in enumerate(x):
            head = self.state.heads[int(projected[i, 297])]
            if head is not None:
                wanted = 2 if row[3] < -0.6 else 5
                selected = (
                    wanted
                    if head.coefficients[-1][wanted]
                    else int(np.argmax(head.coefficients[-1]))
                )
                p[i] = np.eye(7)[selected]
        return p


class BankHarness(old.Harness):
    def __init__(self, patch):
        super().__init__(patch)
        self.rows = []
        patch.setattr(app, "ContextLinearLearner", BankLiteral)

    def game(
        self, cold, state, *, budget, check_boundary, reference_window, reference_catalogue=False
    ):
        old.WORK["games"] += 1
        assert reference_window is reference_catalogue is True
        assert cold.snapshot_state().completed_updates == 0
        n = state.completed_updates
        assert 2 <= n <= 6 and state.training_examples == 18 + (n - 2) * 18
        assert [h.fit_rows for h in state.heads] == (
            [9, 9] if n == 2 else [18, 9] if n == 3 else [18, 18]
        )
        assert budget.updates_completed == budget.replay_examples_completed == 0
        check_boundary()
        self.time += 0.1
        self.states.append(deepcopy(state))
        rows = []
        for i, (branch, cue) in enumerate(((3, "blue"), (3, "red"), (4, "blue"), (4, "red"))):
            action = 2 if branch == 3 else 5
            success = bool(state.heads[i % 2].coefficients[-1][action])
            rows.append(
                dict(
                    status="complete",
                    source_role="train",
                    cue=cue,
                    success=success,
                    predictions=[dict(action=action)],
                    environment=dict(
                        episode_id=f"bank-literal-{len(self.states)}-{i}",
                        role="train",
                        cue=cue,
                        layout=asdict(MiniGridKeyDoorLayout("A", branch_row=branch)),
                        seed=17,
                        max_steps=16,
                        closed=True,
                        environment_steps=1,
                        failure_work_may_be_unobserved=False,
                        transitions=[dict(action=action, applied=True)],
                    ),
                )
            )
            rows[-1]["return"] = float(success)
        result = dict(status="complete", state_unchanged=True, episodes=rows)
        self.games.append(result)
        self.on_game(result)
        check_boundary()
        return result

    def run(self, episodes=None, boundary=None, *, retain_arrived=True):
        from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
        from src.shared.process_memory import ProcessRssSampler

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
                old.sources() if episodes is None else episodes,
                readout_budget=self.budget,
                check_boundary=boundary,
                retain_arrived=retain_arrived,
            )


@pytest.fixture
def harness(monkeypatch):
    return BankHarness(monkeypatch)


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = BankHarness(patch)
        r = h.run()
        assert r["status"] == "complete", r
        return r, h


def test_should_keep_warmup_and_full_native_history_without_refitting_initial_templates(integrated):
    r, h = integrated
    assert r["warmup"]["budget"]["updates"] == 2 and "catalogue" not in r["warmup"]
    assert (
        r["base"]["state"]["completed_updates"] == 2
        and r["base"]["state"]["training_examples"] == 18
    )
    for t in r["trials"]:
        updates = t["budget"]["updates"]
        assert t["checkpoints"] == [
            dict(relative_updates=n, native_updates=2 + n, native_rows=18 + 18 * n)
            for n in range(updates + 1)
        ]
        assert t["source_unchanged"] and t["actor_unchanged"]
    assert h.train_calls == 12 and sum(b["updates"] for b in r["budgets"]) == 12


def test_should_separate_preparation_and_actual_current_replay_native_exposures(integrated):
    r, _ = integrated
    for row, n in zip([*r["trials"], r["application"]], (0, 4, 0, 4, 2), strict=True):
        planned, committed = row["catalogue"]["planned"], row["catalogue"]["committed"]
        assert planned["input_array_bytes"] == 90 * 297 * 8
        assert (
            row["catalogue"]["preparation_attempts"],
            row["catalogue"]["preparation_returns"],
            row["catalogue"]["unreturned_preparation_unknown"],
        ) == (1, 1, False)
        assert planned["prepared_plan_rows"] == 90 and planned["unused_warm_template_rows"] == 18
        assert (
            planned["future_current_rows"],
            planned["future_replay_rows"],
            planned["future_native_rows"],
        ) == (36, 36, 72)
        assert (
            planned["peak_bank_rows"] == 18
            and planned["peak_bank_observation_target_float64_equivalent_bytes"] == 18 * 593 * 8
        )
        assert committed == dict(
            verified_updates=n,
            current_rows=9 * n,
            replay_rows=9 * n,
            native_rows=18 * n,
            unverified_committed_updates=0,
        )
        assert row["window"]["training_examples"] == 18 * n
        assert row["budget"]["replay"] == 0 and row["budget"]["limits"]["max_training_updates"] == 4
    assert sum(t["catalogue"]["committed"]["native_rows"] for t in r["trials"]) == 144
    assert r["readout_budget"]["started_at"] == 0.0 and r["readout_budget"]["updates"] == 0


def test_should_bind_warm_and_local_arrival_epochs_without_future_replay(integrated):
    r, _ = integrated
    for row in [*r["trials"], r["application"]]:
        plan = row["catalogue"]["planned"]
        wakes = plan["wakes"]
        assert len(wakes) == 6
        assert [w["current_episode_id"] for w in wakes[:2]] == ["warmup-0", "warmup-1"]
        assert [w["retained_arrived_at"] for w in wakes[2:]] == [1, 2, 3, 4]
        assert [w["replay_rows"] for w in wakes] == [0, 0, 9, 9, 9, 9]
        for i, w in enumerate(wakes[2:]):
            assert w["retained_arrived_at"] <= w["observed_at"]
            spans = w["source_spans"]
            assert [(s["start"], s["stop"]) for s in spans] == [(0, 9), (9, 18)]
            assert spans[0]["episode_id"] == w["retained_episode_id"]
            assert spans[1]["episode_id"] == w["current_episode_id"]
            local = plan["window_wakes"][i]
            assert (local["observed_at"], local["arrived_at"]) == (i, i + 1)
            retained = local["retained"][0]
            assert retained["epoch"] == ("warmup" if i < 2 else "window")
            assert retained["arrived_at"] == i % 2 + 1


def test_should_apply_same_frozen_owner_and_leave_correct_oldest_pending(integrated):
    r, _ = integrated
    assert r["fitted"]["choices"] == dict(clean="wait", mistakes="wake1")
    assert r["controller"]["attempts"] == r["controller"]["completed"] == 4
    app_row = r["application"]
    assert app_row["window"]["completed_updates"] == 2
    assert app_row["window"]["window_complete"] and not app_row["window"]["training_complete"]
    assert [
        (p["native_updates"], p["native_rows"], p["reused_from"]) for p in app_row["phases"]
    ] == [(4, 54, None), (4, 54, 2)]
    assert (
        app_row["outcomes"]["after_task"]
        == app_row["outcomes"]["after_return"]
        == dict(old_A3=2, new_A4=2)
    )
    assert app_row["outcomes"]["recovery"] == [None, None]
    for row in [*r["trials"], app_row]:
        feedbacks = [
            d["observation"]["feedback"]
            for d in row["window"]["decisions"]
            if d["observation"]["feedback"] is not None
        ]
        assert all(f["examples"] == 18 for f in feedbacks)
        assert row["window"]["ready_feedback"]["prediction_rows_attempted"] == 72
    assert [c["examples"] for c in r["contexts"]] == [9, 9]


@pytest.mark.parametrize("invalid", [None, 0, 1, "yes"])
def test_should_refuse_non_boolean_bank_flag_before_model_work(harness, invalid):
    before = old.WORK["owners"]
    with pytest.raises(ValueError, match="exact bool"):
        harness.run(retain_arrived=invalid)
    assert old.WORK["owners"] == before and harness.train_calls == 0


@pytest.mark.parametrize("damage", ["value", "replacement", "owner"])
def test_should_refuse_damaged_preparation_and_keep_as_read_evidence(harness, monkeypatch, damage):
    captured = []
    original = app._Study.window

    def window(study, source, indices, name, row, **kwargs):
        if name == "trial-clean-wake1":
            captured.append(row)
        return original(study, source, indices, name, row, **kwargs)

    monkeypatch.setattr(app._Study, "window", window)

    def corrupt(owner):
        if harness.train_calls == 3:
            row = captured[0]
            if damage == "value":
                row["catalogue"]["planned"]["future_replay_rows"] = 0
            elif damage == "replacement":
                row["catalogue"]["planned"] = deepcopy(row["catalogue"]["planned"])
            else:
                row["catalogue"] = deepcopy(row["catalogue"])

    harness.on_train = corrupt
    r = harness.run()
    assert r["status"] == "failed" and r["controller"]["completed"] == 1
    failed = r["trials"][1]
    assert failed["status"] == "failed" and "capture_error" in failed["catalogue"]
    assert failed["catalogue"]["planned"]["future_replay_rows"] == 36
    assert harness.train_calls == 3 and r["application"] is None


def test_should_preserve_unknown_joint_fit_without_counting_it_as_completed(harness):
    def lose(owner):
        if harness.train_calls == 3:
            raise RuntimeError("lost joint fit return")

    harness.on_train = lose
    r = harness.run()
    assert r["status"] == "failed" and "lost joint" in r["error"]
    assert r["trials"][0]["status"] == "complete"
    assert r["trials"][1]["catalogue"]["committed"] == dict(
        verified_updates=0,
        current_rows=0,
        replay_rows=0,
        native_rows=0,
        unverified_committed_updates=0,
    )
    assert r["trials"][1]["window"]["attempted_wakes"] == 1
    assert r["controller"]["stopped"] and r["application"] is None


def test_should_record_unknown_preparation_after_work_without_fitting_or_utility(
    harness, monkeypatch
):
    original = app.compose_arrived_cue_catalogue
    calls = []

    def lose(*args, **kwargs):
        prepared = original(*args, **kwargs)
        calls.append(prepared.input_array_bytes)
        raise RuntimeError("lost prepared bank return")

    monkeypatch.setattr(app, "compose_arrived_cue_catalogue", lose)
    r = harness.run()
    assert r["status"] == "failed" and "lost prepared" in r["error"]
    failed = r["trials"][0]["catalogue"]
    assert calls == [90 * 297 * 8]
    assert (
        failed["preparation_attempts"],
        failed["preparation_returns"],
        failed["unreturned_preparation_unknown"],
        failed["planned"],
    ) == (1, 0, True, None)
    assert failed["committed"]["native_rows"] == 0
    assert harness.train_calls == 2 and r["controller"]["completed"] == 0
    assert r["controller"]["stopped"] and r["application"] is None


@pytest.mark.parametrize("n", range(5))
def test_should_dispatch_actual_warm_plus_joint_prefix_to_unchanged_readout(monkeypatch, n):
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy

    rows = [9 if n < 1 else 18, 9 if n < 2 else 18]
    heads = [
        CategoricalLinearFit(296, 7, count, ((0.0,) * 7,) * 297, 0, (0.0,) * count, 0.0)
        for count in rows
    ]
    state = native.ContextLinearTrainingSnapshot(
        ContextLinearPolicy(), (heads[0], heads[1]), 2 + n, 18 + 18 * n
    )
    calls = []

    def dispatch(source, owned, **kwargs):
        calls.append((owned, kwargs))
        kwargs["extra_check"]()
        return dict(status="literal_dispatch")

    monkeypatch.setattr(evaluation, "_run_checkpoint", dispatch)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        native.ContextLinearLearner(),
        state,
        budget=ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            lambda: 0.0,
        ),
        check_boundary=lambda: None,
        reference_window=True,
        reference_catalogue=True,
    )
    assert result["status"] == "literal_dispatch" and len(calls) == 1
    owned, options = calls[0]
    assert owned == state and owned is not state and options["updates"] == n + 2
    assert options["role"] == "knownTRAIN_A3A4_reference_catalogue_prefix"
