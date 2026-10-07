"""Train one fixed v12 structural-ranking factor cell without final data.

The app applies counterfactual factor corrections after each existing core
update, then invokes one existing core sleep event. It returns unscored
model states and audit facts; final release and file IO belong elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Any

import numpy as np

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead
from src.infra.continual_roles import PhaseDecisionRoles
from src.infra.datasets import LabeledData

PROTOCOL_ID = "structural_rank_factors_v12"


@dataclass(frozen=True)
class StructuralRankManifest:
    protocol_id: str = PROTOCOL_ID
    seeds: tuple[int, ...] = (23, 29)
    backends: tuple[str, ...] = ("numpy", "torch_cpu")
    wake_modulation: tuple[bool, ...] = (False, True)
    reward_weight_history: tuple[bool, ...] = (False, True)
    rank_importance: tuple[bool, ...] = (False, True)
    hidden_dim: int = 8
    epochs_per_phase: int = 8
    learning_rate: float = 0.02
    inference_steps: int = 2
    inference_learning_rate: float = 0.1
    inner_guard_fraction: float = 0.2
    outer_selection_fraction: float = 0.2
    final_count: int = 40
    split_threshold: float = 0.0
    prune_threshold: float = 1.0
    split_cap: int = 1
    prune_cap: int = 1


@dataclass(frozen=True)
class RankFactors:
    wake_modulation: bool
    reward_weight_history: bool
    rank_importance: bool


@dataclass(frozen=True)
class RankWork:
    wake_updates: int
    wake_examples: int
    inference_loops: int
    example_inference_loops: int
    replay_updates: int
    sleep_events: int


@dataclass(frozen=True)
class UnscoredRankFacts:
    seed: int
    backend: str
    factors: RankFactors
    role_hashes: dict[str, str]
    initial_parameter_digest: str
    pre_sleep_parameter_digest: str
    post_a_parameter_digest: str
    post_b_parameter_digest: str
    reward_scales: tuple[float, ...]
    a_work: RankWork
    total_work: RankWork
    split_pairs: tuple[tuple[int, int], ...]
    removed_prune_ids: tuple[int, ...]
    initial_width: int
    post_sleep_width: int
    final_width: int
    initial_parameter_count: int
    post_sleep_parameter_count: int
    final_parameter_count: int


@dataclass
class TrainedRankTrial:
    facts: UnscoredRankFacts
    post_a_state: Any
    post_b_model: Any


def fixed_structural_rank_manifest() -> StructuralRankManifest:
    """Return the prospectively declared v12 protocol without tuning knobs."""
    return StructuralRankManifest()


def make_structural_model(
    manifest: StructuralRankManifest, backend: str, seed: int, factors: RankFactors
) -> Any:
    """Build a backend head; reward signal is observed in every factor cell."""
    split_mix = 0.20 if factors.rank_importance else 0.0
    prune_mix = 0.35 if factors.rank_importance else 0.0
    if backend == "numpy":
        numpy_config = CircadianConfig(
            use_reward_modulated_learning=True,
            use_adaptive_plasticity_sensitivity=False,
            split_importance_mix=split_mix,
            prune_importance_mix=prune_mix,
            max_split_per_sleep=manifest.split_cap,
            max_prune_per_sleep=manifest.prune_cap,
            split_threshold=manifest.split_threshold,
            prune_threshold=manifest.prune_threshold,
            prune_decay_steps=1,
            sleep_mode="components",
            sleep_enable_replay=False,
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
            replay_steps=0,
            replay_memory_size=0,
        )
        return CircadianPredictiveCodingNetwork(
            2,
            manifest.hidden_dim,
            seed,
            circadian_config=numpy_config,
            min_hidden_dim=manifest.hidden_dim - 1,
            max_hidden_dim=manifest.hidden_dim + 1,
        )
    if backend == "torch_cpu":
        import torch

        torch_config = CircadianHeadConfig(
            use_reward_modulated_learning=True,
            use_adaptive_plasticity_sensitivity=False,
            split_importance_mix=split_mix,
            prune_importance_mix=prune_mix,
            max_split_per_sleep=manifest.split_cap,
            max_prune_per_sleep=manifest.prune_cap,
            split_threshold=manifest.split_threshold,
            prune_threshold=manifest.prune_threshold,
            sleep_mode="components",
            sleep_enable_homeostasis=False,
            sleep_enable_chemical_reset=False,
        )
        return CircadianPredictiveCodingHead(
            2,
            manifest.hidden_dim,
            2,
            torch.device("cpu"),
            seed,
            config=torch_config,
            min_hidden_dim=manifest.hidden_dim - 1,
            max_hidden_dim=manifest.hidden_dim + 1,
        )
    raise ValueError("unknown v12 structural backend")


def _parameter_arrays(model: Any, backend: str) -> tuple[Any, ...]:
    first = "weight_input_hidden" if backend == "numpy" else "weight_feature_hidden"
    return tuple(
        getattr(model, name)
        for name in (first, "bias_hidden", "weight_hidden_output", "bias_output")
    )


def _detached(values: Any, backend: str) -> Any:
    return values.copy() if backend == "numpy" else values.detach().clone()


def parameter_digest(model: Any, backend: str) -> str:
    """Hash the shallow head parameters, not adaptive or final-role state."""
    digest = sha256(f"structural_rank_parameters_v12/{backend}".encode("ascii"))
    dtype = "<f8" if backend == "numpy" else "<f4"
    for tensor in _parameter_arrays(model, backend):
        values = tensor if backend == "numpy" else tensor.detach().cpu().numpy()
        canonical = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def parameter_count(model: Any, backend: str) -> int:
    return (
        int(sum(tensor.size for tensor in _parameter_arrays(model, backend)))
        if backend == "numpy"
        else int(model.parameter_count())
    )


def _set_counterfactual_state(
    model: Any,
    backend: str,
    factors: RankFactors,
    previous_parameters: tuple[Any, ...],
    previous_importance: Any,
    scale: float,
) -> None:
    if not factors.wake_modulation:
        for current, previous in zip(
            _parameter_arrays(model, backend), previous_parameters, strict=True
        ):
            if backend == "numpy":
                np.copyto(current, previous + (current - previous) / scale)
            else:
                import torch

                with torch.no_grad():
                    current.copy_(previous + (current - previous) / scale)
    if not factors.reward_weight_history:
        decay = model.config.importance_ema_decay
        baseline = decay * previous_importance
        corrected = baseline + (model._importance_ema - baseline) / scale
        if backend == "numpy":
            np.copyto(model._importance_ema, corrected)
        else:
            model._importance_ema = corrected


def train_rank_update(
    model: Any,
    backend: str,
    factors: RankFactors,
    batch: LabeledData,
    manifest: StructuralRankManifest,
) -> float:
    """Apply one existing core update, then isolate the two reward factors."""
    previous_parameters = tuple(
        _detached(parameter, backend) for parameter in _parameter_arrays(model, backend)
    )
    previous_importance = _detached(model._importance_ema, backend)
    if backend == "numpy":
        model.train_epoch(
            batch.input,
            batch.target,
            manifest.learning_rate,
            manifest.inference_steps,
            manifest.inference_learning_rate,
        )
        scale = float(model.get_last_reward_scale())
    else:
        import torch

        model.train_step(
            torch.as_tensor(batch.input, dtype=torch.float32),
            torch.as_tensor(batch.target[:, 0], dtype=torch.long),
            manifest.learning_rate,
            manifest.inference_steps,
            manifest.inference_learning_rate,
        )
        scale = float(model.last_reward_scale())
    if not isfinite(scale) or not 0.75 <= scale <= 1.5:
        raise ValueError("v12 core reward scale is outside predeclared bounds")
    _set_counterfactual_state(
        model, backend, factors, previous_parameters, previous_importance, scale
    )
    return scale


def _work(model: Any, manifest: StructuralRankManifest) -> RankWork:
    clocks = model.get_sleep_clocks()
    return RankWork(
        wake_updates=clocks.wake_batches,
        wake_examples=clocks.wake_examples,
        inference_loops=clocks.wake_batches * manifest.inference_steps,
        example_inference_loops=clocks.wake_examples * manifest.inference_steps,
        replay_updates=clocks.replay_updates,
        sleep_events=clocks.sleep_events,
    )


def train_structural_rank_trial(
    manifest: StructuralRankManifest,
    roles: dict[str, PhaseDecisionRoles],
    *,
    seed: int,
    backend: str,
    factors: RankFactors,
) -> TrainedRankTrial:
    """Train A, apply one capped sleep, then train B; return no score."""
    model = make_structural_model(manifest, backend, seed, factors)
    initial_digest = parameter_digest(model, backend)
    initial_parameter_count = parameter_count(model, backend)
    scales: list[float] = []
    for _ in range(manifest.epochs_per_phase):
        scales.append(train_rank_update(model, backend, factors, roles["a"].train, manifest))
    pre_sleep_digest = parameter_digest(model, backend)
    event = model.sleep_event(
        force_sleep=True,
        current_step=manifest.epochs_per_phase,
        total_steps=2 * manifest.epochs_per_phase,
    )
    if not event.performed or event.telemetry is None or event.telemetry.outcome != "applied":
        raise ValueError("v12 forced A-boundary sleep did not commit")
    changes = event.telemetry.changes
    post_sleep_width = model.hidden_dim
    post_sleep_parameter_count = parameter_count(model, backend)
    post_a_digest = parameter_digest(model, backend)
    post_a_state = model.snapshot_state()
    a_work = _work(model, manifest)
    for _ in range(manifest.epochs_per_phase):
        scales.append(train_rank_update(model, backend, factors, roles["b"].train, manifest))
    facts = UnscoredRankFacts(
        seed=seed,
        backend=backend,
        factors=factors,
        role_hashes={
            f"{phase}/{role}": phase_roles.split_hashes[role]
            for phase, phase_roles in roles.items()
            for role in ("train", "inner_guard", "outer_selection")
        },
        initial_parameter_digest=initial_digest,
        pre_sleep_parameter_digest=pre_sleep_digest,
        post_a_parameter_digest=post_a_digest,
        post_b_parameter_digest=parameter_digest(model, backend),
        reward_scales=tuple(scales),
        a_work=a_work,
        total_work=_work(model, manifest),
        split_pairs=changes.applied_split_pairs,
        removed_prune_ids=changes.applied_removed_prune_ids,
        initial_width=manifest.hidden_dim,
        post_sleep_width=post_sleep_width,
        final_width=model.hidden_dim,
        initial_parameter_count=initial_parameter_count,
        post_sleep_parameter_count=post_sleep_parameter_count,
        final_parameter_count=parameter_count(model, backend),
    )
    return TrainedRankTrial(facts, post_a_state, model)
