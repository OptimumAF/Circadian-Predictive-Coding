"""Fixed task/head selection on disposable LS forks; no scientific cohort."""

from copy import deepcopy
from dataclasses import replace
import pytest

from scripts import context_head_probe as head
from scripts import probe_ticket_policy_learning as recipe
from scripts.ticket_linear_probe import bind_source, bind_readouts
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_LINEAR_RESIDUAL_ID,
)
from src.adapters.linear_fit_learner import _rng_signature
from src.app.actor_shadow import StableActor
from test_context_linear_catalog_study import literal_state


@pytest.fixture(scope="module")
def inputs():
    spec = head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)
    return (
        spec,
        head.build_head_cohorts(spec, head_task_transfer=True),
        head.build_head_cohorts(spec),
    )


def donor():
    # Literal zero-coefficient metadata fixture, NOT an observed qualified donor.
    # Different native counters prove this API does not assume source12/72.
    state = literal_state()
    state = replace(
        state,
        heads=(
            replace(state.heads[0], completed_updates=19, training_examples=120),
            state.heads[1],
        ),
        completed_updates=23,
        training_examples=144,
    )
    seed = ContextLinearFitLearner(state.policy)
    seed.restore_state(state)
    return seed


def run(seed, inputs, **kwargs):
    return recipe.run_arm(
        seed,
        seed.snapshot_state(),
        inputs[1],
        inputs[0],
        lambda: None,
        None,
        "continued",
        separate_context_heads=True,
        linear_fit=True,
        head_tasks=recipe.TRANSFER_HEAD_TASKS,
        **kwargs,
    )


def test_should_bind_actual_transfer_semantics_and_owned_source_without_fit(inputs, monkeypatch):
    seed = donor()
    state = seed.snapshot_state()
    monkeypatch.setattr(seed, "fork", lambda: pytest.fail("binding must not fork"))
    guard = bind_source(
        seed, state, inputs[1], inputs[0], lambda: None, head_tasks=recipe.TRANSFER_HEAD_TASKS
    )
    guard()
    assert seed.snapshot_state() == state


@pytest.mark.parametrize(
    "fault",
    [
        "legacy",
        "role",
        "context",
        "labels",
        "order",
        "head_bit",
        "short",
        "test_labels",
        "old_labels",
        "return_labels",
        "foreign_metadata",
    ],
)
def test_should_refuse_wrong_transfer_cohorts_before_first_fork(inputs, fault, monkeypatch):
    seed = donor()
    groups = [list(group) for group in deepcopy(inputs[1])]
    if fault == "legacy":
        groups = [list(group) for group in deepcopy(inputs[2])]
    elif fault == "role":
        groups[0][0] = replace(groups[0][0], role="final_test")
    elif fault == "context":
        groups[0][0] = replace(groups[0][0], context="parcel-v2")
    elif fault == "labels":
        groups[0][0].targets[1, 0] = 0
    elif fault == "order":
        groups[0][0] = replace(
            groups[0][0],
            features=groups[0][0].features[::-1].copy(),
            targets=groups[0][0].targets[::-1].copy(),
        )
    elif fault == "head_bit":
        groups[0][0].features[:, 4:6] = groups[0][0].features[:, 4:6][:, ::-1]
    elif fault == "short":
        groups[0] = groups[0][:-1]
    elif fault == "test_labels":
        groups[1][0].targets[1, 0] = 0
    elif fault == "old_labels":
        groups[2][0].targets[1, 0] = 0
    elif fault == "return_labels":
        groups[3][0].targets[1, 0] = 0
    else:
        groups[0][0] = replace(groups[0][0], packet_episodes=(object(),))
    monkeypatch.setattr(seed, "fork", lambda: pytest.fail("fork before semantic refusal"))
    cohorts = tuple(tuple(group) for group in groups)
    with pytest.raises(ValueError):
        recipe.run_arm(
            seed,
            seed.snapshot_state(),
            cohorts,
            inputs[0],
            lambda: None,
            None,
            "continued",
            separate_context_heads=True,
            linear_fit=True,
            head_tasks=recipe.TRANSFER_HEAD_TASKS,
        )


@pytest.mark.parametrize(
    "tasks",
    [(), (("v1", "legacy"),), list(recipe.TRANSFER_HEAD_TASKS), recipe.TRANSFER_HEAD_TASKS[::-1]],
)
def test_should_refuse_noncanonical_task_selection_before_binding(inputs, tasks, monkeypatch):
    seed = donor()
    monkeypatch.setattr(seed, "fork", lambda: pytest.fail("fork before task refusal"))
    with pytest.raises(ValueError, match="fixed transfer tasks"):
        recipe.run_arm(
            seed,
            seed.snapshot_state(),
            inputs[1],
            inputs[0],
            lambda: None,
            None,
            "continued",
            separate_context_heads=True,
            linear_fit=True,
            head_tasks=tasks,
        )


def test_should_pin_selected_tasks_as_preceding_paid_readout_evidence():
    row = dict(
        mode="continued",
        prior_native_updates=23,
        prior_native_rows=144,
        readout_work=[],
        episodes=[],
        selected_head_tasks=recipe.TRANSFER_HEAD_TASKS,
    )
    guard, _ = bind_readouts(row, [], lambda: None)
    row["selected_head_tasks"] = recipe.TRANSFER_HEAD_TASKS[::-1]
    with pytest.raises(ValueError, match="paid endpoint"):
        guard()
    assert row["selected_head_tasks"] == recipe.TRANSFER_HEAD_TASKS


def test_should_keep_default_ls_admission_without_selected_task_constraints(inputs, monkeypatch):
    seed = donor()
    monkeypatch.setattr(seed, "fork", lambda: pytest.fail("admission must not fork"))
    bind_source(seed, seed.snapshot_state(), inputs[2], inputs[0], lambda: None)()


def test_should_refuse_stale_task_metadata_in_a_default_invocation(inputs, monkeypatch):
    seed = donor()
    monkeypatch.setattr(seed, "fork", lambda: pytest.fail("fork before stale metadata refusal"))
    with pytest.raises(ValueError, match="row task selection"):
        recipe.run_arm(
            seed,
            seed.snapshot_state(),
            inputs[2],
            inputs[0],
            lambda: None,
            None,
            "continued",
            separate_context_heads=True,
            linear_fit=True,
            row=dict(selected_head_tasks=recipe.TRANSFER_HEAD_TASKS),
        )


def test_should_execute_original_private_probe_and_return_with_selected_tasks(inputs, monkeypatch):
    seed = donor()
    deployed, benchmark = StableActor(seed, version="fixture-deployed"), seed.fork()
    state = seed.snapshot_state()
    rng = _rng_signature()
    pin = deepcopy(inputs[1])
    calls = []
    native = ContextLinearFitLearner.train_batch

    def train(learner, x, y):
        assert all(learner is not owner for owner in (seed, deployed, benchmark))
        calls.append(len(x))
        return native(learner, x, y)

    monkeypatch.setattr(ContextLinearFitLearner, "train_batch", train)
    row = run(seed, inputs)
    recipe._require_arm(
        row, separate_context_heads=True, linear_fit=True, head_tasks=recipe.TRANSFER_HEAD_TASKS
    )
    assert calls == [3] * 16 + [6] * 16
    assert row["selected_head_tasks"] == recipe.TRANSFER_HEAD_TASKS
    assert (row["prior_native_updates"], row["prior_native_rows"]) == (23, 144)
    assert [(x["native_updates"], x["native_rows"]) for x in row["readout_work"]] == [
        (23, 144),
        (39, 192),
        (55, 288),
    ]
    assert [x["updates"] for x in row["sessions"]] == [16, 16]
    assert all(x["limits"] == recipe.asdict(recipe.HEAD_LIMITS) for x in row["sessions"])
    assert [(x["phase"], x["policy_version"], x["concept"]) for x in row["episodes"]] == [
        (phase, v, c)
        for phase in ("before_probe", "after_probe", "after_return")
        for v, c in recipe.TRANSFER_HEAD_TASKS
    ]
    assert all(
        x["role"] == "probe_test"
        and x["episode"]["complete"]
        and x["episode"]["training_updates"] == 0
        for x in row["episodes"]
    )
    assert all(
        d["definition"] == CONTEXT_LINEAR_RESIDUAL_ID
        for phase in (row["probe"], row["probe"]["return_phase"])
        for d in phase["diagnostics"]
    )
    assert (
        seed.snapshot_state()
        == deployed.snapshot_state().state
        == benchmark.snapshot_state()
        == state
    )
    assert _rng_signature() == rng and recipe._same_value(inputs[1], pin)
    # Perceptron's original fixed warm/fresh source-origin contract is intact.
    with pytest.raises(ValueError, match="source origin"):
        recipe._require_arm(row, separate_context_heads=True, head_tasks=recipe.TRANSFER_HEAD_TASKS)
