"""Fresh same-context dispatch and causal windows; no native model work."""

from time import monotonic

import pytest

from scripts import context_head_live as live
from scripts import context_head_rehearsal as source
from scripts import recurring_ticket_workflow_example as entry
from scripts.context_training_streams import (
    CONTEXT_PAIRED_REHEARSAL_RECIPES,
    CONTEXT_REHEARSAL_RECIPES,
)
from scripts.ticket_serving_timeline import FollowupTrainingPlan
from src.app.toy_execution_budget import ToyBudgetSession
from src.shared.process_memory import ProcessRssSampler


def options():
    return dict(
        method="linear_fit",
        context_return=True,
        separate_context_heads=True,
        paired_serving="same_context_joint",
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(method="perceptron"),
        dict(context_return=False),
        dict(separate_context_heads=False),
        dict(private_adaptation=True),
        dict(adaptive_serving=True, method="perceptron"),
        dict(incident_stream="shift"),
        dict(catalog_study=True),
        dict(catalog_action="wake1", catalog_context="mistakes"),
        dict(replay="recent"),
        dict(record_train_fit=True),
    ],
)
def test_should_refuse_incompatible_same_context_mode_before_data(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", lambda **kw: pytest.fail("data opened"))
    args = options()
    args.update(change)
    with pytest.raises(ValueError):
        entry.run_example(**args)


def flags(path):
    return [
        "--output",
        str(path),
        "--method",
        "linear_fit",
        "--context-return",
        "--separate-context-heads",
        "--paired-serving",
        "same_context_joint",
    ]


def test_should_forward_normal_cli_and_preserve_existing_output(tmp_path, monkeypatch):
    calls = []

    def example(**kwargs):
        calls.append(kwargs)
        return dict(complete=True, training_work=dict(attempted_updates=0, attempted_rows=0))

    monkeypatch.setattr(entry, "run_example", example)
    path = tmp_path / "out.json"
    assert entry.main(flags(path)) == 0
    assert calls == [options()]
    with pytest.raises(FileExistsError):
        entry.main(flags(path))
    assert len(calls) == 1


@pytest.mark.parametrize("extra", [["--method", "perceptron"], ["--incident-stream", "shift"]])
def test_should_reject_cli_conflicts_before_source_or_output(extra, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", lambda **kw: pytest.fail("source opened"))
    path = tmp_path / "never.json"
    with pytest.raises(SystemExit):
        entry.main([*flags(path), *extra])
    assert not path.exists()


@pytest.mark.parametrize("control", ["current", "joint", "same_context_joint"])
def test_should_bind_exact_existing_recipe_and_original_source_lifetime(control, monkeypatch):
    state = object()
    guard = lambda: None
    calls = []
    qualifiers = []

    def qualify(value, *args, **kwargs):
        qualifiers.append(kwargs)
        return value, guard

    def run(path, **kwargs):
        calls.append((path, kwargs))
        kwargs["_owned_source"][1]()
        return dict(complete=True)

    monkeypatch.setattr(source, "prepare_owned_source", qualify)
    monkeypatch.setattr(live, "run_case", run)
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        budget = ToyBudgetSession(source._limits(12), monotonic)
        budget.attach_memory(memory)
        birth = budget.started_at
        result = live.run_owned_case(
            state,
            None,
            {},
            budget,
            control=control,
            check_boundary=guard,
            memory=memory,
            linear_fit=True,
        )
    assert result["complete"] and result["owned_source_verified"]
    assert budget.started_at == birth and budget.updates_completed == 0
    assert budget.budget.max_wall_seconds == 30
    assert qualifiers[0]["linear_fit"] and "catalog_study" not in qualifiers[0]
    path, kw = calls[0]
    expected = (
        CONTEXT_REHEARSAL_RECIPES[1]
        if control == "same_context_joint"
        else CONTEXT_PAIRED_REHEARSAL_RECIPES[control == "joint"]
    )
    assert path is None and kw["stream_recipe"] is expected
    assert kw["_owned_source"] == (state, guard) and kw["_memory"] is memory
    assert kw["_linear_fit"] and kw["publish_after_learning"]


@pytest.mark.parametrize(
    "change", [dict(linear_fit=False), dict(adaptive=True, incident_stream="shift")]
)
def test_should_refuse_wrong_owned_family_before_qualification(change, monkeypatch):
    monkeypatch.setattr(
        source, "prepare_owned_source", lambda *a, **kw: pytest.fail("source qualified")
    )
    args = dict(linear_fit=True, control="same_context_joint")
    args.update(change)
    result = live.run_owned_case(
        None, None, {}, None, check_boundary=lambda: None, memory=None, **args
    )
    assert not result["complete"] and result["training_work"]["attempted_updates"] == 0


@pytest.mark.parametrize("same_context", [False, True])
def test_should_select_existing_causal_plan_before_source_access(same_context, monkeypatch):
    calls = []

    class StopBeforeSource(Exception):
        pass

    def plan(first, start):
        calls.append(FollowupTrainingPlan(first, start))
        raise StopBeforeSource

    monkeypatch.setattr(live, "FollowupTrainingPlan", plan)
    with pytest.raises(StopBeforeSource):
        live.run_case(
            None,
            stream_recipe=(
                CONTEXT_REHEARSAL_RECIPES[1]
                if same_context
                else CONTEXT_PAIRED_REHEARSAL_RECIPES[1]
            ),
            publish_after_learning=True,
            _linear_fit=True,
            _owned_source=(object(), lambda: None),
            _memory=object(),
        )
    actual = calls[0]
    first = 2 if same_context else 4
    count = first + (1 if same_context else 2)
    assert actual.first_batches == first and actual.start_tick == 12
    assert [actual.index_at(t, count) for t in range(first)] == list(range(first))
    assert all(actual.index_at(t, count) is None for t in range(first, 12))
    assert actual.index_at(12, count) == first
    assert actual.label_index_at(13, count, 1, 1) == first
