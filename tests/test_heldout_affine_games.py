"""Literal final-only affine admission; no native prediction or held-out outcomes."""

from dataclasses import replace
from typing import Any

import pytest

from src.adapters import minigrid_context_linear_learner as port
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.minigrid_key_door import predefined_layouts
from test_context_linear_causal_games import budget
from test_ordinary_fourth_task import state


def test_should_bind_fixed_affine_endpoint_only_to_original_final_d_plan(monkeypatch):
    source, saved, session = port.ContextLinearLearner(), state(), budget()

    def run(owner, memory, **options):
        assert owner is source and memory == saved and memory is not saved
        assert options["role"] == "final_test" and options["updates"] == 10
        assert options["sources"] == options["layouts"] == ()
        assert options["live_plan"] == (
            (predefined_layouts()[3], "blue", "final_test"),
            (predefined_layouts()[3], "red", "final_test"),
        )
        assert options["require_initial_source"] is options["public_history"] is True
        assert options["representation_feature_dim"] == 298
        assert options["return_metadata"]["source_fit_calls"] == 10
        assert options["return_metadata"]["source_fit_rows"] == 252
        assert options["return_metadata"]["new_fit_calls"] == 0
        assert options["state_equal"](memory, options["memory_copier"](memory))
        assert options["memory_copier"](source.snapshot_state()).completed_updates == 0
        options["extra_check"]()
        return dict(status="complete", public_history={})

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    report = evaluation.evaluate_heldout_context_linear_checkpoint(
        source, saved, budget=session, check_boundary=lambda: None
    )
    assert report["status"] == "complete"
    assert "held-out D final frame" in report["public_history"]["storage_scope"]


@pytest.mark.parametrize("fault", ["intermediate", "rows", "head", "failed", "owner", "source"])
def test_should_refuse_nonqualified_affine_data_before_final_loop(monkeypatch, fault):
    source: Any = port.ContextLinearLearner()
    saved = state()
    if fault == "intermediate":
        saved = replace(saved, completed_updates=8, training_examples=180)
    elif fault == "rows":
        saved = replace(saved, training_examples=253)
    elif fault == "head":
        head = replace(saved.heads[0], fit_rows=35, singular_values=(0.0,) * 35)
        saved = replace(saved, heads=(head, saved.heads[1]))
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 9, 0, 0, True))
    elif fault == "owner":
        source = object()
    else:
        source.restore_state(saved)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("final reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_heldout_context_linear_checkpoint(
            source, saved, budget=budget(), check_boundary=lambda: None
        )


@pytest.mark.parametrize(
    "field,value",
    [("max_training_updates", 1), ("max_replay_examples", 1), ("max_wall_seconds", 151)],
)
def test_should_refuse_nonoriginal_final_budget_before_loop(monkeypatch, field, value):
    session = budget()
    session.budget = replace(session.budget, **{field: value})
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("final reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_heldout_context_linear_checkpoint(
            port.ContextLinearLearner(), state(), budget=session, check_boundary=lambda: None
        )


@pytest.mark.parametrize("fault", ["clock", "charges", "state"])
def test_should_refuse_changed_original_authority_or_checkpoint_before_more_work(
    monkeypatch, fault
):
    source, saved, session = port.ContextLinearLearner(), state(), budget()

    def run(owner, memory, **options):
        if fault == "clock":
            session.clock = lambda: 1.0
        elif fault == "charges":
            session.replay_examples_completed = 1
        else:
            object.__setattr__(memory, "training_examples", 253)
            options["memory_copier"](memory)
        options["extra_check"]()
        pytest.fail("changed authority or data accepted")

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(ValueError):
        evaluation.evaluate_heldout_context_linear_checkpoint(
            source, saved, budget=session, check_boundary=lambda: None
        )
    assert saved.training_examples == 252


def test_should_retain_failed_native_prefix_and_label_final_storage(monkeypatch):
    failed: dict[str, Any] = dict(
        status="failed", reason="scripted native refusal", public_history={}, episodes=[]
    )

    def run(*args, **kwargs):
        raise evaluation.MiniGridEvaluationFailed(failed)

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluation.evaluate_heldout_context_linear_checkpoint(
            port.ContextLinearLearner(), state(), budget=budget(), check_boundary=lambda: None
        )
    assert caught.value.report is not failed  # The existing exception owns a detached copy.
    assert caught.value.report["status"] == "failed" and caught.value.report["episodes"] == []
    assert "held-out D final frame" in caught.value.report["public_history"]["storage_scope"]
    assert failed["public_history"] == {}
