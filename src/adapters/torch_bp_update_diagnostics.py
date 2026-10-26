"""Bounded TRAIN-only realized BP wake/replay readout on a disposable learner.

Inputs are an original invariant BP snapshot, B-blue wake, old-A TRAIN pair and
fresh diagnostic budget. Outputs are scalar fit/direction/work records only.
No deployed update, task selection, final/probe data, tuning or stream runner.
Why: prior gradients alone omit the optimizer momentum in an actual native step.
"""

from copy import deepcopy
from math import isfinite
import pickle
import random
from typing import Any, Callable, cast

import numpy as np

from src.adapters import torch_feature_gradient_diagnostics as prior
from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureLearner,
    InvariantTorchFeaturePolicy,
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_policy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, episode_arrays
from src.core.gradient_geometry import compare_gradient_geometry
from src.shared.torch_runtime import require_torch


DEFINITION = "torch_invariant_bp_realized_train_wake_replay_v1"


def _sources(snapshot, wake, replay):
    prior._exact_record(snapshot, InvariantTorchFeatureSnapshot)
    policy = copy_invariant_torch_feature_policy(snapshot.policy)
    expected = InvariantTorchFeaturePolicy(
        native.TorchFeaturePolicy("bp", 296, 8, 17, num_classes=7)
    )
    if policy != expected:
        raise ValueError("BP readout requires the original invariant H8/seed17 BP recipe")
    if type(replay) is not tuple or len(replay) != 2:
        raise ValueError("BP readout requires the original A-blue/A-red TRAIN pair")
    sources = (wake, *replay)
    for episode, context, cue in zip(
        sources, ("B", "A", "A"), ("blue", "blue", "red"), strict=True
    ):
        prior._exact_record(episode, DemonstrationEpisode)
        if (
            episode.role != "train"
            or episode.context != context
            or episode.episode_id != f"minigrid-source-{context}-{cue}"
            or type(episode.transitions) is not tuple
            or len(episode.transitions) != 9
        ):
            raise ValueError(
                "BP readout requires original nine-step B-blue/A-blue/A-red TRAIN episodes"
            )
    return sources


def _binding(budget):
    return (
        id(budget),
        id(budget.budget),
        pickle.dumps(budget.budget, protocol=5),
        id(budget.clock),
        budget.started_at,
        id(budget.progress),
        id(budget.process_rss_sampler),
    )


def _fresh_budget(budget):
    if (
        type(budget) is not ToyBudgetSession
        or type(budget.budget) is not ToyExecutionBudget
        or type(budget.budget.max_training_updates) is not int
        or budget.budget.max_training_updates != 3
        or type(budget.budget.max_replay_examples) is not int
        or budget.budget.max_replay_examples != 18
        or type(budget.updates_completed) is not int
        or budget.updates_completed != 0
        or type(budget.replay_examples_completed) is not int
        or budget.replay_examples_completed != 0
    ):
        raise ValueError("BP readout requires a fresh original three-update/18-replay-row budget")
    prior._exact_record(budget.budget, ToyExecutionBudget)
    # Validate every bound limit without installing a replacement authority.
    ToyExecutionBudget(**vars(budget.budget))
    return _binding(budget)


def _old_ce(snapshot, episodes, torch, report):
    # Reuse exactly the prior diagnostic's raw projection and native logits.
    losses = []
    for episode in episodes:
        features, targets = episode_arrays(episode)
        work = report["work"]
        work["projection_calls_attempted"] += 1
        work["projection_rows_attempted"] += len(features)
        projected = np.array(
            [
                prior.canonicalize_public_history(tuple(float(value) for value in row))
                for row in features
            ],
            dtype=np.float64,
        )
        work["projection_calls_completed"] += 1
        work["projection_rows_completed"] += len(features)
        work["projection_array_bytes_returned"] += int(projected.nbytes)
        work["forward_calls_attempted"] += 1
        work["forward_rows_attempted"] += len(features)
        with torch.no_grad():
            logits = prior._prior_logits(
                snapshot.native_state.head_state, native._features(projected, 296), torch
            )
            loss = torch.nn.functional.cross_entropy(
                logits, torch.from_numpy(targets[:, 0].astype(np.int64))
            )
        work["forward_calls_completed"] += 1
        work["forward_rows_completed"] += len(features)
        value = float(loss.item())
        if not isfinite(value):
            raise FloatingPointError("nonfinite old-A mean CE")
        losses.append(value)
    return sum(losses) / 2


def _arrays(state):
    return tuple(
        state.native_state.head_state[name].detach().numpy().copy() for name in native._PARAMETERS
    )


def _directions(before, after, gradients):
    parameters = _arrays(before)
    delta = tuple(new - old for old, new in zip(parameters, _arrays(after), strict=True))
    buffers = before.native_state.optimizer_state["state"]
    momentum = tuple(
        np.zeros_like(parameters[index])
        if index not in buffers
        else buffers[index]["momentum_buffer"].detach().numpy().copy()
        for index in range(4)
    )
    policy = before.policy.native_policy
    expected = tuple(
        gradient + np.float32(policy.momentum) * previous
        for gradient, previous in zip(gradients, momentum, strict=True)
    )
    actual = tuple(
        after.native_state.optimizer_state["state"][i]["momentum_buffer"].detach().numpy()
        for i in range(4)
    )
    component = tuple(
        -np.float32(policy.learning_rate * policy.momentum) * previous for previous in momentum
    )
    return (
        delta,
        component,
        max(float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)),
        max(
            float(np.max(np.abs(d + np.float32(policy.learning_rate) * e)))
            for d, e in zip(delta, expected, strict=True)
        ),
    )


def _step(learner, episode, old_a, report, torch, charge):
    before = learner.snapshot_state()
    offset = len(report["losses"])
    with torch.enable_grad():
        update_gradient = prior._episode_gradients(before, episode, torch, report)
        old_gradients = tuple(prior._episode_gradients(before, old, torch, report) for old in old_a)
    old_gradient = tuple(
        (left + right) / np.float32(2) for left, right in zip(*old_gradients, strict=True)
    )
    old_before = sum(row["mean_ce"] for row in report["losses"][offset + 1 :]) / 2
    features, targets = episode_arrays(episode)
    charge()
    with torch.enable_grad():
        learner.train_batch(features, targets)
    work = report["work"]
    work["native_train_calls_returned"] += 1
    work["native_training_rows_returned"] += len(features)
    work["optimizer_steps_returned"] += 1
    work["optimizer_steps"] += 1
    work["native_optimizer_constructions_observed"] += 1
    after = learner.snapshot_state()
    delta, momentum, momentum_error, step_error = _directions(before, after, update_gradient)
    old_after = _old_ce(after, old_a, torch, report)
    return dict(
        phase="wake" if not report["steps"] else "replay",
        episode_id=episode.episode_id,
        rows=len(features),
        native_before_updates=before.completed_updates,
        native_after_updates=after.completed_updates,
        native_before_training_rows=before.training_examples,
        native_after_training_rows=after.training_examples,
        old_A_ce_before=old_before,
        old_A_ce_after=old_after,
        old_A_ce_change=old_after - old_before,
        old_A_gradient_delta_geometry=compare_gradient_geometry(old_gradient, delta),
        old_A_gradient_momentum_geometry=compare_gradient_geometry(old_gradient, momentum),
        update_gradient_delta_geometry=compare_gradient_geometry(update_gradient, delta),
        momentum_expected_max_error=momentum_error,
        parameter_step_expected_max_error=step_error,
        per_parameter_geometry={
            name: dict(
                old_A_gradient_delta_geometry=compare_gradient_geometry(
                    (old_gradient[i],), (delta[i],)
                ),
                old_A_gradient_momentum_geometry=compare_gradient_geometry(
                    (old_gradient[i],), (momentum[i],)
                ),
                update_gradient_delta_geometry=compare_gradient_geometry(
                    (update_gradient[i],), (delta[i],)
                ),
            )
            for i, name in enumerate(native._PARAMETERS)
        },
    )


def diagnose_bp_updates(
    snapshot: object,
    wake: DemonstrationEpisode,
    replay: tuple[DemonstrationEpisode, ...],
    *,
    budget: ToyBudgetSession,
    check_boundary: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """Observe three actual native SGD calls on an owned counterfactual snapshot.

    Charge before every native attempt, including replay exposures. No optimizer
    state is reset between calls. A failed attempt retains its paid prefix. This
    does not authenticate historical acquisition or explain reserved F causally.
    """
    sources = _sources(snapshot, wake, replay)
    binding = _fresh_budget(budget)
    if not callable(check_boundary):
        raise ValueError("BP readout boundary check must be callable")
    witness, rows = _validate_sources(sources, 1)
    torch = require_torch()
    if torch.is_inference_mode_enabled():
        raise ValueError("BP readout does not support torch inference_mode")
    rng, numpy_rng, python_rng = (
        torch.get_rng_state().clone(),
        cast(tuple[Any, ...], deepcopy(np.random.get_state())),
        random.getstate(),
    )
    grad_mode = torch.is_grad_enabled()
    report = prior._report()
    report.update(
        definition=DEFINITION,
        interpretation="TRAIN-only realized BP SGD/momentum; negative gradient dot delta is local descent, not causal acquisition/retention or F benefit",
        steps=[],
        work_scope="forward/backward/projection fields count diagnostic work outside native train calls; each returned native BP call adds one forward/backward/optimizer step and nine projected rows; failed native internals may be unobserved; validation/copies/RSS/time additional",
    )
    report["work"].update(
        admission_projection_rows=2 * rows,
        admission_projection_array_equivalent_bytes=2 * rows * 296 * 8,
        native_train_calls_attempted=0,
        native_train_calls_returned=0,
        native_training_rows_attempted=0,
        native_training_rows_returned=0,
        charged_updates=0,
        charged_replay_rows=0,
        optimizer_steps_returned=0,
        native_optimizer_constructions_observed=0,
    )
    work = report["work"]
    owned = marks = None
    last_clock = budget.last_clock

    def guard(callback=True):
        nonlocal last_clock
        if callback:
            check_boundary()
        if (
            _binding(budget) != binding
            or type(budget.updates_completed) is not int
            or budget.updates_completed != work["charged_updates"]
            or type(budget.replay_examples_completed) is not int
            or budget.replay_examples_completed != work["charged_replay_rows"]
            or type(budget.last_clock) is not float
            or not isfinite(budget.last_clock)
            or budget.last_clock < last_clock
        ):
            raise ValueError("original diagnostic budget, counters or clock changed")
        last_clock = budget.last_clock
        if budget.progress is not None and (
            budget.progress.updates_completed != work["charged_updates"]
            or budget.progress.replay_examples_completed != work["charged_replay_rows"]
        ):
            raise ValueError("original diagnostic progress counters changed")
        witness.check(sources)
        now = cast(tuple[Any, ...], np.random.get_state())
        if (
            owned is not None
            and not _same_state(owned, snapshot)
            or marks is not None
            and not _same_state(marks, prior._tensor_marks(snapshot, torch))
            or not torch.equal(rng, torch.get_rng_state())
            or python_rng != random.getstate()
            or numpy_rng[0] != now[0]
            or not np.array_equal(numpy_rng[1], now[1])
            or numpy_rng[2:] != now[2:]
            or torch.is_grad_enabled() != grad_mode
            or torch.is_inference_mode_enabled()
        ):
            raise ValueError("diagnostic original snapshot, inputs, RNG or mode changed")

    failure = None
    try:
        guard(callback=False)
        work["snapshot_validations_attempted"] += 1
        owned = copy_invariant_torch_feature_snapshot(cast(InvariantTorchFeatureSnapshot, snapshot))
        work["snapshot_validations_completed"] += 1
        marks = prior._tensor_marks(snapshot, torch)
        guard()
        learner = InvariantTorchFeatureLearner(owned.policy)
        work["native_optimizer_constructions_observed"] += 1
        learner.restore_state(owned)
        work["native_optimizer_constructions_observed"] += 1
        guard()
        for index, episode in enumerate(sources):
            budget.before_sleep()
            guard()

            def charge():
                budget.before_update()
                guard()
                if index:
                    budget.record_replay(9)
                    work["charged_replay_rows"] += 9
                budget.record_update()
                work["charged_updates"] += 1
                work["native_train_calls_attempted"] += 1
                work["native_training_rows_attempted"] += 9

            step = _step(learner, episode, sources[1:], report, torch, charge)
            report["steps"].append(step)
            guard()
        budget.before_sleep()
        guard()
    except Exception as error:
        failure = error
    try:
        guard(callback=False)
    except Exception as error:
        failure = failure or error
    work["budget_binding_preserved"] = _binding(budget) == binding
    if failure is not None:
        report.update(status="incomplete", reason=f"{type(failure).__name__}: {failure}")
        work["failure_work_may_be_unobserved"] = True
        raise GradientDiagnosticFailed(report) from failure
    report.update(
        status="complete",
        snapshot_unchanged=True,
        inputs_unchanged=True,
        rng_unchanged=True,
        old_A_ce_change=report["steps"][-1]["old_A_ce_after"]
        - report["steps"][0]["old_A_ce_before"],
    )
    return report
