"""Actual-buffer summaries; only two tests create/train a native learner."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from src.adapters import torch_recurrent_learner as native
from src.adapters import minigrid_public_cue_recurrent_learner as public
from src.adapters import torch_recurrent_gradient_diagnostics as diagnostics
from src.app.coupled_attribution import AttributionStream
from src.adapters.minigrid_learning_pilot import _batch
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.torch_runtime import require_torch
from test_public_cue_recurrent_learner import episode


def literal(*, update=0, momentum=0.5, dimension=2):
    torch = require_torch()
    policy = native.TorchRecurrentPolicy(
        dimension, 1, 7 if dimension == 148 else 2, 17, learning_rate=0.25, momentum=momentum
    )
    direction = (1 if update == 1 else 1.5) if momentum else 1
    value = 1 if update == 0 else 0.75 if update == 1 else 0.375
    shapes = native._shapes(policy)
    parameters = {k: torch.full(s, value, dtype=torch.float32) for k, s in shapes.items()}
    gradients = {k: None if update == 0 else torch.ones(s) for k, s in shapes.items()}
    state = (
        {}
        if not update or not momentum
        else {
            i: {"momentum_buffer": torch.full(s, float(direction))}
            for i, s in enumerate(shapes.values())
        }
    )
    optimizer = dict(
        state=state, param_groups=[dict(native._sgd_options(policy), params=list(range(5)))]
    )
    return native.TorchRecurrentSnapshot(
        1, policy, parameters, optimizer, gradients, update, 2 * update
    )


@pytest.mark.parametrize("momentum", [0.0, 0.5])
def test_should_report_known_first_step_without_executing_native_work(momentum, monkeypatch):
    before, after = literal(momentum=momentum), literal(update=1, momentum=momentum)

    def forbidden(*args, **kwargs):
        raise AssertionError("summary attempted native work")

    for name in ("_optimizer", "_restore", "_forward"):
        monkeypatch.setattr(native, name, forbidden)
    result = diagnostics.summarize_recurrent_update(before, after, rows=2)
    group = result["groups"]["bias_hidden"]
    assert group["gradient_l2"] == 1
    assert group["parameter_delta_l2"] == group["nominal_sgd_delta_l2"] == 0.25
    assert group["sgd_max_abs_residual"] == 0
    assert result["work"]["snapshot_validations"] == 2
    count = sum(v.numel() for v in before.parameters.values())
    assert result["work"]["momentum_scalars"] == (count if momentum else 0)
    assert result["work"]["synthetic_zero_momentum_scalars"] == count * (1 if momentum else 2)
    json.dumps(result, allow_nan=False)


def test_should_read_continued_momentum_and_actual_delta():
    result = diagnostics.summarize_recurrent_update(literal(update=1), literal(update=2), rows=2)
    group = result["groups"]["bias_hidden"]
    assert group["momentum_before_l2"] == group["gradient_l2"] == 1
    assert group["momentum_after_l2"] == 1.5
    assert group["nominal_sgd_delta_l2"] == group["parameter_delta_l2"] == 0.375
    assert group["sgd_max_abs_residual"] == 0


def test_should_expose_inconsistent_sgd_delta_without_claiming_provenance():
    after = literal(update=1)
    after.parameters["bias_hidden"].fill_(5)
    result = diagnostics.summarize_recurrent_update(literal(), after, rows=2)
    assert result["groups"]["bias_hidden"]["sgd_max_abs_residual"] == 4.25


@pytest.mark.parametrize("rows", [True, 0, 17, 2.0])
def test_should_reject_invalid_rows_before_snapshot_validation(rows, monkeypatch):
    monkeypatch.setattr(native, "_snapshot", lambda _: pytest.fail("snapshot called"))
    with pytest.raises(ValueError, match="rows"):
        diagnostics.summarize_recurrent_update(None, None, rows=rows)
    with pytest.raises(ValueError, match="rows"):
        diagnostics.summarize_public_cue_update(None, None, rows=rows)


@pytest.mark.parametrize("change", ["policy", "updates", "rows", "nonfinite"])
def test_should_refuse_incompatible_or_corrupt_snapshots(change):
    after = literal(update=1)
    if change == "policy":
        after = replace(after, policy=replace(after.policy, seed=18))
    elif change == "updates":
        after = replace(after, completed_updates=2, training_examples=4)
    elif change == "rows":
        after = replace(after, training_examples=3)
    else:
        after.parameters["weight_recurrent"].fill_(float("inf"))
    with pytest.raises(ValueError):
        diagnostics.summarize_recurrent_update(literal(), after, rows=2)


def test_should_leave_owned_inputs_and_return_only_detached_scalars():
    before, after = literal(), literal(update=1)
    saved = deepcopy((before, after))
    result = diagnostics.summarize_recurrent_update(before, after, rows=2)
    result["groups"]["weight_input"]["gradient_l2"] = -1
    assert diagnostics._same_state(saved, (before, after))
    assert (
        diagnostics.summarize_recurrent_update(before, after, rows=2)["groups"]["weight_input"][
            "gradient_l2"
        ]
        > 0
    )


def public_pair(index=0):
    cold, trained = literal(dimension=148), literal(update=1, dimension=148)
    policy = public.PublicCueRecurrentPolicy(cold.policy)
    before = public.PublicCueRecurrentSnapshot(policy, (cold, deepcopy(cold)))
    heads = deepcopy(list(before.heads))
    heads[index] = trained
    return before, public.PublicCueRecurrentSnapshot(policy, tuple(heads))


@pytest.mark.parametrize("index", [0, 1])
def test_should_summarize_only_the_changed_public_head(index):
    before, after = public_pair(index)
    result = diagnostics.summarize_public_cue_update(before, after, rows=2)
    assert result["head_index"] == index
    assert result["before_updates"] == 0 and result["after_updates"] == 1
    assert result["selected_head"]["after_updates"] == 1
    assert result["work"] == dict(public_snapshot_copies=2, head_comparisons=2)


@pytest.mark.parametrize("change", ["other_head", "failed_before", "failed_after", "rows"])
def test_should_refuse_unobserved_public_changes_and_failed_owners(change):
    before, after = public_pair()
    if change == "other_head":
        after.heads[1].parameters["bias_hidden"].fill_(7)
    elif change == "failed_before":
        before = replace(before, failed_work=public.FailureWork("train", 0))
    elif change == "failed_after":
        after = replace(after, failed_work=public.FailureWork("train", 0))
    else:
        after = replace(after, heads=(replace(after.heads[0], training_examples=3), after.heads[1]))
    with pytest.raises(ValueError):
        diagnostics.summarize_public_cue_update(before, after, rows=2)


def test_should_measure_two_real_native_steps_without_additional_learning():
    learner = native.TorchRecurrentLearner(native.TorchRecurrentPolicy(2, 2, 2, 17))
    features = np.array([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=np.float64)
    targets = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    previous = learner.snapshot_state()
    for _ in range(2):
        learner.train_batch(features, targets)
        current = learner.snapshot_state()
        summary = diagnostics.summarize_recurrent_update(previous, current, rows=3)
        assert all(g["gradient_l2"] > 0 for g in summary["groups"].values())
        assert max(g["sgd_max_abs_residual"] for g in summary["groups"].values()) < 1e-7
        assert diagnostics._same_state(current, learner.snapshot_state())
        previous = current


def test_should_observe_existing_delayed_stream_without_changing_source_or_actor():
    from time import monotonic

    source = public.PublicCueRecurrentLearner()
    original = source.snapshot_state()
    packets = []
    for i in range(4):
        item = replace(episode(("blue", "red")[i % 2]), episode_id=f"literal-{i}")
        packets.append(_batch(item))
    previous, records = [], []

    def observe(updates, state, batch):
        if updates:
            records.append(
                diagnostics.summarize_public_cue_update(
                    previous[0], state, rows=len(batch.features)
                )
            )
        previous[:] = [state]

    budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=4, max_wall_seconds=60),
        monotonic,
    )
    stream = AttributionStream(
        source,
        tuple(packets),
        (),
        checkpoint_updates=(0, 1, 2, 3, 4),
        label_delay=1,
        budget=budget,
        observe_checkpoint=observe,
        training_only=True,
    )
    actor = deepcopy(stream._runtime.actor.snapshot_state())
    stream.run_until(4)
    result = stream.finish()
    assert [r["head_index"] for r in records] == [0, 1, 0, 1]
    assert [r["selected_head"]["after_updates"] for r in records] == [1, 1, 2, 2]
    assert result["completed_updates"] == budget.updates_completed == 4
    assert diagnostics._same_state(original, source.snapshot_state())
    assert diagnostics._same_state(actor, stream._runtime.actor.snapshot_state())
