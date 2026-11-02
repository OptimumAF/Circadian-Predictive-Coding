"""Paired ordinary TRAIN selection; bounded fixtures, not comparative science."""

from scripts import context_head_live as live
from scripts import context_training_streams as streams
from scripts import recurring_ticket_workflow_example as entry
from scripts.context_head_serving import PAIRED_TASKS
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID,
)
import pytest


@pytest.mark.parametrize(
    "change",
    [
        dict(paired_serving="current"),
        dict(paired_serving="same_context_joint"),
        dict(paired_serving="retained_joint_publish"),
        dict(paired_serving="retained_joint_probe"),
        dict(method="perceptron"),
        dict(replay="recent"),
        dict(adaptive_serving=True),
        dict(incident_stream="stationary"),
        dict(probe_update="minimum_change"),
    ],
)
def test_should_refuse_other_public_compositions_before_source(change, monkeypatch):
    monkeypatch.setattr(entry, "training_cases", lambda **kw: pytest.fail("source opened"))
    options = dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="joint",
        train_update="minimum_change",
    )
    options.update(change)
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        entry.run_example(**options)


@pytest.mark.parametrize(
    "change",
    [
        dict(control="current"),
        dict(control="same_context_joint"),
        dict(control="retained_joint_publish"),
        dict(adaptive=True),
        dict(calibrated=True),
        dict(incident_stream="stationary"),
        dict(catalog_action="wait"),
        dict(catalog_context="clean"),
        dict(probe_update="minimum_change"),
        dict(probe_rehearsal="joint"),
    ],
)
def test_should_refuse_other_owned_compositions_before_transport(change, monkeypatch):
    from scripts import context_head_rehearsal as transport

    monkeypatch.setattr(transport, "prepare_owned_source", lambda *a, **kw: pytest.fail("source"))
    options = dict(control="joint", linear_fit=True, train_update="minimum_change")
    options.update(change)
    with pytest.raises(ValueError, match="minimum-change TRAIN"):
        live.run_owned_case(None, None, None, None, check_boundary=None, memory=None, **options)


@pytest.mark.parametrize(
    "change",
    [
        dict(stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[0]),
        dict(stream_recipe=streams.CONTEXT_REHEARSAL_RECIPES[1]),
        dict(reference_mode="drift"),
        dict(publish_after_learning=False),
        dict(_owned_source=None),
        dict(_linear_fit=False),
        dict(_retained_joint=True),
        dict(_probe_retained=True),
        dict(_publish_retained=True),
        dict(_probe_update="minimum_change"),
        dict(_reference_validation=True),
        dict(contextual_policy=object()),
        dict(catalog_trial=object()),
        dict(candidate_feedback=True),
        dict(task_feedback=True),
        dict(acknowledge_correct=True),
        dict(arrived_shift=True),
    ],
)
def test_should_refuse_foreign_child_compositions_before_any_source_access(change, monkeypatch):
    monkeypatch.setattr(live, "_source_bytes", lambda *a: pytest.fail("source read"))

    def guard():
        pytest.fail("owned source entered")

    options = dict(
        stream_recipe=streams.CONTEXT_PAIRED_REHEARSAL_RECIPES[1],
        publish_after_learning=True,
        _owned_source=(object(), guard),
        _linear_fit=True,
        _train_update="minimum_change",
    )
    options.update(change)
    with pytest.raises(ValueError):
        live.run_case(None, **options)


@pytest.mark.parametrize("selected", [None, "minimum_change"])
def test_should_forward_paired_cli_and_preserve_omitted_default(selected, tmp_path, monkeypatch):
    from scripts import context_catalog_learning as writer

    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(complete=True)

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", lambda *a, **kw: True)
    flags = [
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "joint",
        "--output",
        str(tmp_path / "paired.json"),
    ]
    if selected is not None:
        flags += ["--train-update", selected]
    assert entry.main(flags) == 0
    assert calls == [
        dict(
            method="linear_fit",
            context_return=True,
            separate_context_heads=True,
            paired_serving="joint",
            **(dict(train_update=selected) if selected is not None else {}),
        )
    ]


def run_paired():
    return entry.run_example(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="joint",
        train_update="minimum_change",
    )


def test_should_execute_paired_train_with_guarded_publication_and_rollback(
    monkeypatch, record_property
):
    from scripts import context_head_probe as private
    from scripts import context_head_rehearsal as rehearsal

    monkeypatch.setattr(private, "build_head_cohorts", lambda *a, **kw: pytest.fail("probe data"))
    monkeypatch.setattr(rehearsal, "prepare_cohorts", lambda *a, **kw: pytest.fail("probe bank"))
    result = run_paired()
    child = result.get("paired_serving", {})
    record_property(
        "paired_train",
        dict(
            complete=result["complete"],
            error=result.get("error"),
            source_work=result.get("training_work"),
            child_work=child.get("training_work"),
            guard=child.get("learning_boundary_guard"),
            child_error=child.get("error"),
            publication_tick=child.get("publication_tick"),
            outcomes=[
                (e["phase"], e["policy_version"], e["concept"], e["episode"]["success"])
                for e in child.get("episodes", [])
            ],
        ),
    )
    assert result["complete"], result.get("error")
    assert child["complete"], child.get("error")
    assert result["training_work"] == dict(
        attempted_updates=12,
        attempted_rows=72,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    )
    assert child["training_work"] == dict(
        attempted_updates=6, attempted_rows=48, returned_updates=6
    )
    assert child["declared_tasks"] == PAIRED_TASKS
    assert child["train_task_family"] == "paired_head_tasks_v1"
    assert child["candidate_state"]["completed_updates"] == 18
    assert child["candidate_state"]["training_examples"] == 120
    assert result["source_unchanged"] and result["actor_unchanged"]
    assert result["numpy_rng_unchanged"] and result["python_rng_unchanged"]
    assert child["source_unchanged"] and child["owned_source_verified"]
    assert child["rollback_equal_source"] and child["rollback_generation"] == 2
    assert child["final_actor"]["generation"] == 2
    assert child["final_actor"]["model_state"] == child["rollback"]["model_state"]
    assert child["live_serving"]["closed"]
    assert child["learning_boundary_tick"] == child["publication_tick"] == 10
    assert child["budget"]["updates"] == child["budget"]["limits"]["max_training_updates"] == 6
    assert result["budget"]["limits"]["max_wall_seconds"] == 30
    assert child["budget"]["limits"]["max_wall_seconds"] == 60
    for report in (result, child):
        assert report["linear_train_update"] == "minimum_change"
        assert report["final_examples"] == report["probe_updates"] == report["replay_examples"] == 0
        assert all(
            u["diagnostic"]["definition"] == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
            for u in report["applied_updates"]
        )
    assert result["costs"]["objective"] == CONTEXT_MINIMUM_CHANGE_RESIDUAL_ID
    # These task slots describe the actual paired lessons, not the legacy probe pair.
    for phase in ("published", "rollback"):
        assert [
            (e["policy_version"], e["concept"]) for e in child["episodes"] if e["phase"] == phase
        ] == list(PAIRED_TASKS)


def test_should_keep_lost_first_paired_update_paid_without_publication(
    monkeypatch, record_property
):
    original = ContextLinearFitLearner.train_batch_minimum_change
    calls = []

    def lost(owner, features, targets):
        calls.append(len(features))
        diagnostic = original(owner, features, targets)
        if len(calls) == 13:
            raise RuntimeError("lost paired minimum-change update")
        return diagnostic

    monkeypatch.setattr(ContextLinearFitLearner, "train_batch_minimum_change", lost)
    result = run_paired()
    child = result.get("paired_serving", {})
    record_property(
        "lost_paired",
        dict(
            complete=result["complete"],
            error=result.get("error"),
            source_work=result.get("training_work"),
            child_work=child.get("training_work"),
            budget=child.get("budget"),
            native_calls=len(calls),
        ),
    )
    assert not result["complete"] and not child["complete"]
    assert "lost paired minimum-change update" in child["error"]
    assert calls == [6] * 12 + [9]
    assert child["training_work"] == dict(attempted_updates=1, attempted_rows=9, returned_updates=0)
    assert child["budget"]["updates"] == 0 and child["applied_updates"] == []
    assert "promotion_receipt" not in child
    assert child["source_unchanged"] and result["source_unchanged"] and result["actor_unchanged"]
