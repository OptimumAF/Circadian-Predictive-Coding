"""Read bound feature-head prior mean-CE gradients from detached TRAIN pairs.

Inputs are one complete encoding-bound snapshot and two complete TRAIN episodes.
Outputs are scalar geometry and paid work, never tensors, graphs or updates.
This common prediction objective is not a PC latent-energy update direction;
alignment is descriptive and supplies no causal interference or advantage claim.
"""

from copy import deepcopy
from dataclasses import fields
import random
from typing import Any, cast

import numpy as np

from src.adapters import torch_feature_learner as native
from src.adapters.invariant_torch_feature_learner import (
    InvariantTorchFeatureSnapshot,
    copy_invariant_torch_feature_policy,
    copy_invariant_torch_feature_snapshot,
)
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_recurrent_gradient_diagnostics import GradientDiagnosticFailed
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.core.demonstration_episode import (
    DemonstrationEpisode,
    DemonstrationTransition,
    copy_episode,
    episode_arrays,
)
from src.core.gradient_geometry import compare_gradient_geometry
from src.core.minigrid_invariant_encoding import canonicalize_public_history
from src.shared.torch_runtime import require_torch

DEFINITION = "torch_invariant_feature_prior_mean_ce_gradient_geometry_v1"
ROW_DEFINITION = "torch_invariant_feature_selected_rows_prior_ce_gradient_geometry_v1"


def _exact_record(value: Any, kind: type) -> None:
    if type(value) is not kind or set(vars(value)) != {field.name for field in fields(kind)}:
        raise ValueError(f"an exact complete {kind.__name__} is required")


def _pair(snapshot: Any, episodes: Any) -> tuple[DemonstrationEpisode, ...]:
    _exact_record(snapshot, InvariantTorchFeatureSnapshot)
    policy = copy_invariant_torch_feature_policy(snapshot.policy)
    if type(episodes) is not tuple or len(episodes) != 2:
        raise ValueError("prior gradient diagnostics require an exact pair of TRAIN episodes")
    owned = []
    for episode in episodes:
        _exact_record(episode, DemonstrationEpisode)
        if type(episode.transitions) is not tuple or not 1 <= len(episode.transitions) <= 16:
            raise ValueError("complete episode transitions require an exact tuple of1..16 rows")
        for transition in episode.transitions:
            _exact_record(transition, DemonstrationTransition)
        copied = copy_episode(episode)
        if (
            copied.role != "train"
            or copied.num_classes != policy.num_classes
            or len(copied.transitions[0].observation) != policy.feature_dim
        ):
            raise ValueError("TRAIN episode geometry must match the bound296/K7 policy")
        owned.append(copied)
    if owned[0].context != owned[1].context or owned[0].episode_id == owned[1].episode_id:
        raise ValueError("TRAIN gradient pair requires same context and distinct episode IDs")
    return tuple(owned)


def _report() -> dict[str, Any]:
    return dict(
        status="running",
        reason=None,
        definition=DEFINITION,
        objective=native.TORCH_CE_ID,
        interpretation="prior_not_native_pc_energy",
        losses=[],
        joint_geometry=None,
        per_parameter_geometry=None,
        work=dict(
            snapshot_validations_attempted=0,
            snapshot_validations_completed=0,
            validation_temporary_optimizers=0,
            projection_calls_attempted=0,
            projection_calls_completed=0,
            projection_rows_attempted=0,
            projection_rows_completed=0,
            projection_array_bytes_returned=0,
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


def _prior_logits(parameters: dict[str, Any], values: Any, torch: Any) -> Any:
    hidden = torch.tanh(values @ parameters["weight_feature_hidden"] + parameters["bias_hidden"])
    return hidden @ parameters["weight_hidden_output"] + parameters["bias_output"]


def _episode_gradients(
    owned: InvariantTorchFeatureSnapshot,
    episode: DemonstrationEpisode,
    torch: Any,
    report: dict[str, Any],
    *,
    row_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, ...]:
    features, targets = episode_arrays(episode)
    if row_indices is not None:
        features = features[list(row_indices)].copy()
        targets = targets[list(row_indices)].copy()
    work, rows = report["work"], len(features)
    work["projection_calls_attempted"] += 1
    work["projection_rows_attempted"] += rows
    projected = np.array(
        [canonicalize_public_history(tuple(float(v) for v in row)) for row in features],
        dtype=np.float64,
    )
    work["projection_calls_completed"] += 1
    work["projection_rows_completed"] += rows
    work["projection_array_bytes_returned"] += int(projected.nbytes)
    values = native._features(projected, owned.policy.feature_dim)
    parameters = {
        name: owned.native_state.head_state[name].detach().clone().requires_grad_(True)
        for name in native._PARAMETERS
    }
    work["forward_calls_attempted"] += 1
    work["forward_rows_attempted"] += rows
    logits = _prior_logits(parameters, values, torch)
    work["forward_calls_completed"] += 1
    work["forward_rows_completed"] += rows
    if not bool(torch.isfinite(logits).all().item()):
        raise FloatingPointError("nonfinite bound feature prior logits")
    loss = torch.nn.functional.cross_entropy(
        logits, torch.from_numpy(targets[:, 0].astype(np.int64))
    )
    if not bool(torch.isfinite(loss).all().item()):
        raise FloatingPointError("nonfinite bound feature prior mean CE")
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
    if type(gradients) is not tuple or len(gradients) != 4:
        raise ValueError("prior gradient readout must contain all four parameter gradients")
    arrays = []
    for name, gradient in zip(native._PARAMETERS, gradients, strict=True):
        if (
            type(gradient) is not torch.Tensor
            or gradient.layout != torch.strided
            or gradient.is_sparse
        ):
            raise ValueError("prior gradients must be dense CPU float32 tensors")
        native._tensor(gradient, tuple(parameters[name].shape), torch.float32, f"gradient {name}")
        arrays.append(
            gradient.detach().numpy().astype(np.float32, copy=True).reshape(parameters[name].shape)
        )
    return tuple(arrays)


def _tensor_marks(value: Any, torch: Any) -> list[tuple[bool, Any]]:
    """Retain autograd metadata that ordinary complete-state value equality omits."""
    if type(value) is torch.Tensor:
        gradient = value.grad if value.is_leaf or value.retains_grad else None
        return [(value.requires_grad, None if gradient is None else gradient.detach().clone())]
    if type(value) is dict:
        return [mark for child in value.values() for mark in _tensor_marks(child, torch)]
    if type(value) in (list, tuple):
        return [mark for child in value for mark in _tensor_marks(child, torch)]
    if type(value) in (InvariantTorchFeatureSnapshot, native.TorchFeatureSnapshot):
        return [
            mark
            for field in fields(value)
            for mark in _tensor_marks(getattr(value, field.name), torch)
        ]
    return []


def diagnose_prior_gradients(
    snapshot: object,
    episodes: tuple[DemonstrationEpisode, ...],
    *,
    row_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Measure two prior mean-CE gradients without optimizer construction or updates.

    Why: detached clones and autograd.grad preserve the deployed snapshot's
    parameters, optimizer, gradient buffers and chemistry. enable_grad restores
    no_grad callers; inference mode is refused before native snapshot work.
    """
    pair = _pair(snapshot, episodes)
    if row_indices is not None and (
        type(row_indices) is not tuple
        or not 1 <= len(row_indices) <= min(len(e.transitions) for e in pair)
        or any(
            type(index) is not int or not 0 <= index < min(len(e.transitions) for e in pair)
            for index in row_indices
        )
        or any(left >= right for left, right in zip(row_indices, row_indices[1:]))
    ):
        raise ValueError(
            "row_indices must be an exact nonempty increasing tuple of native indices valid in both complete TRAIN episodes"
        )
    if row_indices is not None:
        # Complete current/next causal admission prevents hiding a corrupt
        # unselected row behind a numerically valid selected-row gradient.
        _validate_sources(pair, 1)
    torch = require_torch()
    if torch.is_inference_mode_enabled():
        raise ValueError("prior gradient diagnostics do not support torch inference_mode")
    rng = torch.get_rng_state().clone()
    numpy_rng = cast(tuple[Any, ...], deepcopy(np.random.get_state()))
    python_rng, grad_mode = random.getstate(), torch.is_grad_enabled()
    report, failure, before, marks = _report(), None, None, None
    if row_indices is not None:
        report.update(
            definition=ROW_DEFINITION,
            row_indices=list(row_indices),
            interpretation="selected_original_TRAIN_rows_prior_not_native_pc_energy_or_causal_interference",
        )
        report["work"].update(
            admission_projection_rows=2 * sum(len(e.transitions) for e in pair),
            admission_projection_array_equivalent_bytes=2
            * sum(len(e.transitions) for e in pair)
            * 296
            * 8,
        )

    def check_inputs() -> None:
        if row_indices is not None and (
            report.get("definition") != ROW_DEFINITION
            or type(report.get("row_indices")) is not list
            or any(type(index) is not int for index in report["row_indices"])
            or tuple(report["row_indices"]) != row_indices
        ):
            raise ValueError("selected TRAIN gradient row declaration changed")
        current = _pair(snapshot, episodes)
        now = cast(tuple[Any, ...], np.random.get_state())
        same_numpy_rng = (
            numpy_rng[0] == now[0]
            and np.array_equal(numpy_rng[1], now[1])
            and numpy_rng[2:] == now[2:]
        )
        if (
            current != pair
            or (before is not None and not _same_state(before, snapshot))
            or (marks is not None and not _same_state(marks, _tensor_marks(snapshot, torch)))
            or not torch.equal(rng, torch.get_rng_state())
            or not same_numpy_rng
            or python_rng != random.getstate()
            or torch.is_grad_enabled() != grad_mode
            or torch.is_inference_mode_enabled()
        ):
            raise ValueError("diagnostic input, snapshot, gradient buffers, RNG or mode changed")

    try:
        report["work"]["snapshot_validations_attempted"] += 1
        owned = copy_invariant_torch_feature_snapshot(cast(InvariantTorchFeatureSnapshot, snapshot))
        report["work"]["snapshot_validations_completed"] += 1
        # The full bounded validator runs before copying any caller tensor metadata.
        before, marks = owned, _tensor_marks(snapshot, torch)
        check_inputs()
        with torch.enable_grad():
            gradients = tuple(
                _episode_gradients(owned, episode, torch, report)
                if row_indices is None
                else _episode_gradients(owned, episode, torch, report, row_indices=row_indices)
                for episode in pair
            )
        joint = compare_gradient_geometry(*gradients)
        per_parameter = {
            name: compare_gradient_geometry((gradients[0][i],), (gradients[1][i],))
            for i, name in enumerate(native._PARAMETERS)
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
