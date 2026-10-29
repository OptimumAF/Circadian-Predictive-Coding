"""Literal prefix admission/dispatch only; no solver, prediction or native games."""

from dataclasses import replace

import pytest
import numpy as np

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import (
    ContextLinearLearner,
    ContextLinearTrainingSnapshot,
)
from src.adapters.minigrid_context_linear_policy import ContextLinearPolicy
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit

WORK = dict(empty_owners=0, forbidden=0)


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    original = ContextLinearLearner.__init__

    def empty(owner):
        WORK["empty_owners"] += 1
        assert WORK["empty_owners"] <= 64
        original(owner)

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("no native model work in metadata admission checks")

    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", forbidden)
    monkeypatch.setattr(ContextLinearLearner, "predict", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)


def literal_state(updates):
    fit = CategoricalLinearFit(296, 7, 9, ((0.0,) * 7,) * 297, 1, (1.0,) * 9, 0.0)
    return ContextLinearTrainingSnapshot(
        ContextLinearPolicy(),
        (fit if updates else None, fit if updates >= 2 else None),
        updates,
        updates * 9,
    )


def invoke(monkeypatch, state, **options):
    calls = []

    def dispatch(source, owned, **kwargs):
        calls.append((owned, kwargs))
        kwargs["extra_check"]()
        assert kwargs["memory_copier"](owned) == owned
        return dict(status="literal_dispatch")

    monkeypatch.setattr(evaluation, "_run_checkpoint", dispatch)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        ContextLinearLearner(),
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
        **options,
    )
    return result, calls


@pytest.mark.parametrize("updates", range(7))
def test_should_dispatch_exact_observed_prefix_without_inventing_unfitted_heads(
    monkeypatch, updates
):
    before = literal_state(updates)
    result, calls = invoke(monkeypatch, before, reference_window=True)
    assert result["status"] == "literal_dispatch" and len(calls) == 1
    observed, options = calls[0]
    assert observed == before and observed is not before
    assert options["updates"] == updates and options["require_initial_source"] is True
    assert options["role"] == "knownTRAIN_A3A4_reference_prefix" and options["sources"] == ()
    assert [(layout.branch_row, cue, role) for layout, cue, role in options["live_plan"]] == [
        (3, "blue", "train"),
        (3, "red", "train"),
        (4, "blue", "train"),
        (4, "red", "train"),
    ]
    assert options["return_metadata"]["source_fit_rows"] == 9 * updates
    assert options["return_metadata"]["new_fit_calls"] == 0


@pytest.mark.parametrize(
    "conflict",
    [
        "retained_catalogue",
        "after_c_capacity",
        "recurring_catalogue",
        "development_a4",
        "a4_training",
        "a4_current_only",
    ],
)
def test_should_refuse_other_recipe_modes_before_dispatch(monkeypatch, conflict):
    with pytest.raises(ValueError, match="distinct exact"):
        invoke(monkeypatch, literal_state(2), reference_window=True, **{conflict: True})


@pytest.mark.parametrize("flag", [1, np.bool_(True)])
def test_should_refuse_non_native_boolean(monkeypatch, flag):
    with pytest.raises(ValueError, match="distinct exact"):
        invoke(monkeypatch, literal_state(2), reference_window=flag)


@pytest.mark.parametrize("fault", ["count", "rows", "cue", "failed", "fit_rows"])
def test_should_refuse_incompatible_prefix_geometry_or_failed_evidence(monkeypatch, fault):
    state = literal_state(7 if fault == "count" else 2)
    if fault == "rows":
        state = replace(state, training_examples=19)
    elif fault == "cue":
        state = replace(literal_state(1), heads=(None, literal_state(2).heads[1]))
    elif fault == "failed":
        state = replace(state, failed_work=(1, 9, 0, 0, True))
    elif fault == "fit_rows":
        state = replace(
            literal_state(4),
            heads=(
                replace(state.heads[0], fit_rows=18, singular_values=(1.0,) * 18),
                state.heads[1],
            ),
        )
    with pytest.raises(ValueError, match="reference prefix"):
        invoke(monkeypatch, state, reference_window=True)


def test_should_preserve_original_recipe_admission_and_dispatch(monkeypatch):
    with pytest.raises(ValueError, match="geometry"):
        invoke(monkeypatch, literal_state(1))
    _, calls = invoke(monkeypatch, literal_state(2))
    assert calls[0][1]["role"] == "knownTRAIN_ABA_current_phase"
    assert len(calls[0][1]["live_plan"]) == 2


@pytest.mark.parametrize("updates", [0, 3])
def test_should_distinguish_all_four_reference_episode_identities(updates):
    from src.core.minigrid_key_door import MiniGridKeyDoorLayout

    ids = [
        evaluation._live_episode_id(
            "affine",
            updates,
            MiniGridKeyDoorLayout("A", branch_row=row),
            cue,
            "knownTRAIN_A3A4_reference_prefix",
        )
        for row in (3, 4)
        for cue in ("blue", "red")
    ]
    assert len(set(ids)) == 4
    assert ids == [
        f"live-affine-{updates}-A-row{row}-{cue}" for row in (3, 4) for cue in ("blue", "red")
    ]


def test_should_preserve_old_episode_names_exactly():
    from src.core.minigrid_key_door import MiniGridKeyDoorLayout

    layout = MiniGridKeyDoorLayout("A", branch_row=4)
    for role in ("knownTRAIN_A3A4_adaptation", "knownTRAIN_A3A4_current_only"):
        assert (
            evaluation._live_episode_id("affine", 2, layout, "blue", role)
            == "live-affine-2-A-row4-blue"
        )
    assert (
        evaluation._live_episode_id("affine", 2, layout, "blue", "knownTRAIN_ABA_current_phase")
        == "live-affine-2-A-blue"
    )
