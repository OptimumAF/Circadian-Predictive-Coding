"""Read original recurrent mean-CE gradient geometry from detached TRAIN pairs.

Inputs are one exact native snapshot and two complete same-context episodes.
Output is JSON scalar diagnostics and paid work, never tensors or graphs. There
is no learner/update, environment, final/probe access, tuning or causal claim.
Whole-episode alignment can hide conflict on a few cue-dependent decisions.
"""

from copy import deepcopy
from dataclasses import fields
import random
from typing import Any, cast

import numpy as np

from src.adapters import torch_recurrent_learner as native
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.gradient_geometry import compare_gradient_geometry
from src.shared.torch_runtime import require_torch

DEFINITION = "torch_recurrent_train_mean_ce_gradient_geometry_v1"


class GradientDiagnosticFailed(RuntimeError):
    """A runtime diagnostic failure retaining a detached, paid prefix report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = deepcopy(report)
        super().__init__(f"gradient diagnostic incomplete: {self.report['reason']}")


def _exact_record(value: Any, kind: type) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


def _pair(snapshot: Any, episodes: Any) -> tuple[DemonstrationEpisode, ...]:
    _exact_record(snapshot, native.TorchRecurrentSnapshot)
    policy = native._policy(snapshot.policy)
    if type(episodes) is not tuple or len(episodes) != 2:
        raise ValueError("gradient diagnostics require an exact pair of TRAIN episodes")
    owned = []
    for episode in episodes:
        _exact_record(episode, DemonstrationEpisode)
        if type(episode.transitions) is not tuple or not (
            1 <= len(episode.transitions) <= policy.max_episode_steps
        ):
            raise ValueError("episode transitions must be a bounded nonempty exact tuple")
        for transition in episode.transitions:
            _exact_record(transition, DemonstrationTransition)
        copied = copy_episode(episode)
        if (
            copied.role != "train"
            or copied.num_classes != policy.num_classes
            or len(copied.transitions) > policy.max_episode_steps
            or len(copied.transitions[0].observation) != policy.feature_dim
        ):
            raise ValueError("TRAIN episode geometry must match the original native policy")
        owned.append(copied)
    if owned[0].context != owned[1].context or owned[0].episode_id == owned[1].episode_id:
        raise ValueError("TRAIN gradient pair requires same context and distinct episode IDs")
    return tuple(owned)


def _report() -> dict[str, Any]:
    return dict(
        status="running",
        reason=None,
        definition=DEFINITION,
        objective=native.TORCH_RECURRENT_CE_ID,
        losses=[],
        joint_geometry=None,
        per_parameter_geometry=None,
        work=dict(
            snapshot_validations_attempted=0,
            snapshot_validations_completed=0,
            validation_temporary_optimizers=0,
            forward_calls_attempted=0,
            forward_calls_completed=0,
            forward_rows_attempted=0,
            forward_rows_completed=0,
            backward_calls_attempted=0,
            backward_calls_completed=0,
            backward_rows_attempted=0,
            backward_rows_completed=0,
            optimizer_steps=0,
            published_updates=0,
            failure_work_may_be_unobserved=False,
        ),
    )


def _episode_gradients(
    owned: Any, episode: DemonstrationEpisode, torch: Any, report: dict[str, Any]
) -> tuple[np.ndarray, ...]:
    features, targets = episode_arrays(episode)
    values = native._features(features, owned.policy)
    parameters = {
        name: owned.parameters[name].detach().clone().requires_grad_(True) for name in native._NAMES
    }
    work, rows = report["work"], len(episode.transitions)
    work["forward_calls_attempted"] += 1
    work["forward_rows_attempted"] += rows
    logits, _ = native._forward(
        parameters,
        values,
        torch.zeros((1, owned.policy.hidden_dim), dtype=torch.float32, device="cpu"),
        check_finite=True,
    )
    work["forward_calls_completed"] += 1
    work["forward_rows_completed"] += rows
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("nonfinite recurrent gradient logits")
    loss = torch.nn.functional.cross_entropy(
        logits, torch.from_numpy(targets[:, 0].astype(np.int64))
    )
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError("nonfinite recurrent mean CE")
    report["losses"].append(
        dict(
            episode_id=episode.episode_id,
            context=episode.context,
            rows=rows,
            mean_ce=float(loss.item()),
        )
    )
    work["backward_calls_attempted"] += 1
    work["backward_rows_attempted"] += rows
    gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=False)
    work["backward_calls_completed"] += 1
    work["backward_rows_completed"] += rows
    if type(gradients) is not tuple or len(gradients) != 5:
        raise ValueError("native gradient readout must contain all five parameter gradients")
    arrays = []
    for name, gradient in zip(native._NAMES, gradients, strict=True):
        if (
            type(gradient) is not torch.Tensor
            or gradient.layout != torch.strided
            or gradient.is_sparse
        ):
            raise ValueError("native gradients must be dense CPU float32 tensors")
        native._tensor(gradient, tuple(parameters[name].shape), f"diagnostic gradient {name}")
        arrays.append(gradient.detach().cpu().numpy().astype(np.float32, copy=True))
    return tuple(arrays)


def diagnose_episode_gradients(
    snapshot: object, episodes: tuple[DemonstrationEpisode, ...]
) -> dict[str, Any]:
    """Measure two original mean-CE gradients without applying or accumulating them.

    Native snapshot validation constructs one temporary SGD configuration
    validator. No optimizer step/restoration is used. enable_grad supports a
    no_grad caller and restores its mode; inference-mode tensors are refused.
    """
    pair = _pair(snapshot, episodes)
    torch = require_torch()
    if torch.is_inference_mode_enabled():
        raise ValueError("gradient diagnostics do not support torch inference_mode")
    rng = torch.get_rng_state().clone()
    numpy_rng = cast(tuple[Any, ...], deepcopy(np.random.get_state()))
    python_rng = random.getstate()
    grad_mode = torch.is_grad_enabled()
    report = _report()
    failure = None
    before = None

    def check_inputs() -> None:
        current = _pair(snapshot, episodes)
        current_numpy_rng = cast(tuple[Any, ...], np.random.get_state())
        same_numpy_rng = (
            numpy_rng[0] == current_numpy_rng[0]
            and np.array_equal(numpy_rng[1], current_numpy_rng[1])
            and numpy_rng[2:] == current_numpy_rng[2:]
        )
        if (
            current != pair
            or (before is not None and not _same_state(before, snapshot))
            or not torch.equal(rng, torch.get_rng_state())
            or not same_numpy_rng
            or python_rng != random.getstate()
            or torch.is_grad_enabled() != grad_mode
            or torch.is_inference_mode_enabled()
        ):
            raise ValueError("diagnostic input, snapshot, RNG or mode changed")

    try:
        report["work"]["snapshot_validations_attempted"] += 1
        owned = native._snapshot(cast(native.TorchRecurrentSnapshot, snapshot))
        report["work"]["snapshot_validations_completed"] += 1
        report["work"]["validation_temporary_optimizers"] = 1
        # The existing read-only validator bounds every nested field before it
        # copies. Never deepcopy an unvalidated caller payload. Episode work
        # uses further parameter clones, leaving this validated baseline owned.
        before = owned
        check_inputs()
        with torch.enable_grad():
            gradients = tuple(_episode_gradients(owned, episode, torch, report) for episode in pair)
        joint = compare_gradient_geometry(*gradients)
        per_parameter = {
            name: compare_gradient_geometry((gradients[0][i],), (gradients[1][i],))
            for i, name in enumerate(native._NAMES)
        }
    except Exception as error:
        failure = error
    try:
        check_inputs()
    except Exception as error:
        failure = failure or error
    if failure is not None:
        report.update(status="incomplete", reason=f"{type(failure).__name__}: {failure}")
        report["work"]["failure_work_may_be_unobserved"] = True
        raise GradientDiagnosticFailed(report) from failure
    report.update(status="complete", joint_geometry=joint, per_parameter_geometry=per_parameter)
    return report
