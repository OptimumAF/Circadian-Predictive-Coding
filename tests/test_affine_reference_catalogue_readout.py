"""Joint-prefix geometry and existing loop with scripted actions, no native games."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    ContextLinearTrainingSnapshot,
)
from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from tests.test_minigrid_invariant_encoding import frame

WORK = dict(
    constructors=0, predictions=0, prediction_rows=0, environments=0, steps=0, closes=0, forbidden=0
)
HOOKS: dict[str, Any] = {}
TOTALS = (0, 9, 18, 36, 54, 72, 90)
HEADS = ((None, None), (9, None), (9, 9), (18, 9), (18, 18), (18, 18), (18, 18))


def literal_state(count):
    heads = tuple(
        None
        if rows is None
        else CategoricalLinearFit(296, 7, rows, ((0.0,) * 7,) * 297, 0, (0.0,) * rows, 0.0)
        for rows in HEADS[count]
    )
    return ContextLinearTrainingSnapshot(
        ContextLinearPolicy(), (heads[0], heads[1]), count, TOTALS[count]
    )


def budget(clock=None):
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=0,
            max_replay_examples=0,
            max_wall_seconds=150,
            max_process_rss_bytes=512 * 1024**2,
        ),
        clock or (lambda: 0.0),
    )


@pytest.fixture(autouse=True)
def scripted_only(monkeypatch):
    HOOKS.clear()
    original = ContextLinearLearner.__init__

    def construct(owner):
        WORK["constructors"] += 1
        original(owner)

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine fitting/prediction forbidden")

    monkeypatch.setattr(ContextLinearLearner, "__init__", construct)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", forbidden)
    monkeypatch.setattr(ContextLinearLearner, "predict", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    yield
    HOOKS.clear()


def invoke(monkeypatch, state, **options):
    calls = []

    def dispatch(source, owned, **kwargs):
        calls.append((owned, kwargs))
        kwargs["extra_check"]()
        assert kwargs["memory_copier"](owned) == owned
        return dict(status="literal_dispatch")

    monkeypatch.setattr(evaluation, "_run_checkpoint", dispatch)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        ContextLinearLearner(), state, budget=budget(), check_boundary=lambda: None, **options
    )
    return result, calls


@pytest.mark.parametrize("count", range(7))
def test_should_read_only_exact_joint_prefix_and_report_owned_current_and_replay_exposures(
    monkeypatch, count
):
    state = literal_state(count)
    result, calls = invoke(monkeypatch, state, reference_window=True, reference_catalogue=True)
    assert result["status"] == "literal_dispatch" and len(calls) == 1
    owned, options = calls[0]
    assert owned == state and owned is not state
    assert options["role"] == "knownTRAIN_A3A4_reference_catalogue_prefix"
    assert options["require_initial_source"] is options["public_history"] is True
    assert options["sources"] == options["layouts"] == ()
    assert options["representation_feature_dim"] == 298
    assert [(l.branch_row, c, r) for l, c, r in options["live_plan"]] == [
        (3, "blue", "train"),
        (3, "red", "train"),
        (4, "blue", "train"),
        (4, "red", "train"),
    ]
    metadata = options["return_metadata"]
    assert metadata["reference_catalogue"] is True and metadata["source_fit_calls"] == count
    assert metadata["source_fit_rows"] == TOTALS[count]
    assert metadata["source_current_row_exposures"] == count * 9
    assert metadata["source_replay_row_exposures"] == TOTALS[count] - count * 9
    assert metadata["new_fit_calls"] == metadata["new_optimizer_updates"] == 0


@pytest.mark.parametrize(
    "fault",
    [
        "count",
        "rows",
        "blue_rows",
        "red_rows",
        "missing_red",
        "wrong_cue",
        "failed",
        "current_only",
    ],
)
def test_should_refuse_foreign_geometry_and_failed_evidence_before_dispatch(monkeypatch, fault):
    state = literal_state(3)
    if fault == "count":
        state = replace(state, completed_updates=7, training_examples=108)
    elif fault == "rows":
        state = replace(state, training_examples=35)
    elif fault == "blue_rows":
        state = replace(
            state,
            heads=(replace(state.heads[0], fit_rows=9, singular_values=(0.0,) * 9), state.heads[1]),
        )
    elif fault == "red_rows":
        state = replace(
            state,
            heads=(
                state.heads[0],
                replace(state.heads[1], fit_rows=18, singular_values=(0.0,) * 18),
            ),
        )
    elif fault == "missing_red":
        state = replace(state, heads=(state.heads[0], None))
    elif fault == "wrong_cue":
        state = replace(literal_state(1), heads=(None, literal_state(1).heads[0]))
    elif fault == "failed":
        state = replace(state, failed_work=(1, 18, 0, 0, True))
    else:
        state = replace(state, training_examples=27, heads=literal_state(2).heads)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(),
            state,
            budget=budget(),
            check_boundary=lambda: None,
            reference_window=True,
            reference_catalogue=True,
        )


@pytest.mark.parametrize("flag", [1, None, np.bool_(True)])
def test_should_require_exact_boolean_catalogue_option(monkeypatch, flag):
    with pytest.raises(ValueError, match="reference catalogue"):
        invoke(monkeypatch, literal_state(2), reference_window=True, reference_catalogue=flag)


@pytest.mark.parametrize(
    "mode",
    [
        None,
        "retained_catalogue",
        "after_c_capacity",
        "recurring_catalogue",
        "development_a4",
        "a4_training",
        "a4_current_only",
        "a4_current_repeated",
    ],
)
def test_should_require_explicit_reference_window_and_refuse_other_recipes(monkeypatch, mode):
    options = dict(reference_catalogue=True)
    if mode is not None:
        options.update(reference_window=True, **{mode: True})
    with pytest.raises(ValueError):
        invoke(monkeypatch, literal_state(2), **options)


@pytest.mark.parametrize("count", [2, 4, 6])
def test_should_preserve_original_a4_checkpoint_geometry_and_metadata(monkeypatch, count):
    _, calls = invoke(monkeypatch, literal_state(count), a4_training=True)
    options = calls[0][1]
    assert options["role"] == "knownTRAIN_A3A4_adaptation"
    assert options["return_metadata"]["source_fit_rows"] == TOTALS[count]
    assert "reference_catalogue" not in options["return_metadata"]
    assert "source_replay_row_exposures" not in options["return_metadata"]


@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_should_not_silently_relax_original_a4_admission_to_partial_prefixes(monkeypatch, count):
    with pytest.raises(ValueError, match="geometry"):
        invoke(monkeypatch, literal_state(count), a4_training=True)


class Environment:
    def __init__(self, layout, cue, seed=17, max_steps=16, *, role="train", episode_id="scripted"):
        WORK["environments"] += 1
        assert seed == 17 and max_steps == 16 and role == "train"
        self.layout, self.cue, self.seed, self.max_steps, self.role, self.episode_id = (
            layout,
            cue,
            seed,
            max_steps,
            role,
            episode_id,
        )
        self.actions = []
        self.transitions = []
        self.closed = False
        self.terminal = False
        self.observation = self.view(0)

    def view(self, step):
        color = 2 if self.cue == "blue" else 0
        return frame(
            cue_color=color, key_color=color, direction=step % 4, door_state=2 if step == 0 else 0
        )

    @property
    def legal_actions(self):
        return () if self.terminal else tuple(range(7))

    @property
    def report(self):
        return dict(
            episode_id=self.episode_id,
            role=self.role,
            cue=self.cue,
            closed=self.closed,
            environment_steps_attempted=len(self.actions),
            environment_steps=len(self.actions),
            terminated=self.terminal,
            truncated=False,
            total_return=0.5 if self.terminal else 0.0,
            status="complete" if self.terminal else "running",
            transitions=deepcopy(self.transitions),
            failure_work_may_be_unobserved=False,
        )

    def step(self, action):
        WORK["steps"] += 1
        before = self.observation
        self.actions.append(action)
        self.terminal = len(self.actions) == 2
        self.observation = self.view(len(self.actions))
        result = dict(
            observation=self.observation,
            reward=0.5 if self.terminal else 0.0,
            terminated=self.terminal,
            truncated=False,
        )
        self.transitions.append(
            dict(
                observation=before,
                next_observation=self.observation,
                action=action,
                reward=result["reward"],
                terminated=self.terminal,
                truncated=False,
                raw_next_observation={"scripted": True},
            )
        )
        return result

    def close(self):
        WORK["closes"] += 1
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def run_loop(monkeypatch, state, session, source=None):
    def predict(owner, x):
        WORK["predictions"] += 1
        WORK["prediction_rows"] += len(x)
        if "predict" in HOOKS:
            HOOKS["predict"](owner)
        return np.full((len(x), 7), 1 / 7, dtype=np.float64)

    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", Environment)
    return evaluation.evaluate_context_linear_stream_checkpoint(
        source or ContextLinearLearner(),
        state,
        budget=session,
        check_boundary=lambda: None,
        reference_window=True,
        reference_catalogue=True,
    )


@pytest.mark.parametrize("count", range(7))
def test_should_execute_original_loop_for_unfitted_odd_and_complete_joint_prefixes(
    monkeypatch, count
):
    source, state, session = ContextLinearLearner(), literal_state(count), budget()
    before = pickle.dumps(
        (source.snapshot_state(), state, np.random.get_state(), random.getstate())
    )
    work = dict(WORK)
    result = run_loop(monkeypatch, state, session, source)
    assert result["status"] == "complete" and result["state_unchanged"]
    assert (
        result["source_fit_rows"] == TOTALS[count]
        and result["source_replay_row_exposures"] == TOTALS[count] - count * 9
    )
    episodes = result["episodes"]
    assert len(episodes) == 4
    assert len({e["environment"]["episode_id"] for e in episodes}) == 4
    assert all(e["environment"]["closed"] for e in episodes)
    assert (
        WORK["environments"] - work["environments"],
        WORK["closes"] - work["closes"],
        WORK["steps"] - work["steps"],
        WORK["predictions"] - work["predictions"],
    ) == (4, 4, 8, 8)
    assert (
        pickle.dumps((source.snapshot_state(), state, np.random.get_state(), random.getstate()))
        == before
    )
    assert session.updates_completed == session.replay_examples_completed == 0


@pytest.mark.parametrize("fault", ["updates", "replay", "clock", "limits"])
def test_should_refuse_changed_original_zero_work_budget_before_more_actions(monkeypatch, fault):
    session = budget()

    def dispatch(*a, **options):
        if fault == "updates":
            session.updates_completed = 1
        elif fault == "replay":
            session.replay_examples_completed = 1
        elif fault == "clock":
            session.clock = lambda: 1.0
        else:
            object.__setattr__(session.budget, "max_wall_seconds", 151)
        options["extra_check"]()
        pytest.fail("changed budget accepted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", dispatch)
    with pytest.raises(ValueError, match="budget binding"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(),
            literal_state(3),
            budget=session,
            check_boundary=lambda: None,
            reference_window=True,
            reference_catalogue=True,
        )


@pytest.mark.parametrize("fault", ["late", "scorer", "source"])
def test_should_stop_after_failed_literal_prediction_preserve_paid_attempt_and_close(
    monkeypatch, fault
):
    now = [0.0]
    source = ContextLinearLearner()
    session = budget(lambda: now[0])

    def mutate(owner):
        if fault == "late":
            now[0] = 151.0
        elif fault == "source":
            source.restore_state(literal_state(2))
        else:
            owner._state = replace(owner._state, training_examples=55)

    HOOKS["predict"] = mutate
    before = dict(WORK)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as error:
        run_loop(monkeypatch, literal_state(3), session, source)
    assert error.value.report["status"] == "failed"
    assert WORK["predictions"] - before["predictions"] == 1
    assert WORK["steps"] == before["steps"]
    assert WORK["environments"] - before["environments"] == WORK["closes"] - before["closes"] == 1
