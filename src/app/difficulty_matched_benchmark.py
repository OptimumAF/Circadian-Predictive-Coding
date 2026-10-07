"""Run the fixed v11 matched modulation comparison with a global final seal.

The app owns scheduling, matched-work preflight, and final-role release.
Core models own updates; infra supplies deterministic phase data. No
decision role or post-update diagnostic selects a setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isclose, isfinite
from typing import Any, Callable

import numpy as np

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.difficulty_diagnostics import measure_train_batch
from src.core.resnet50_variants import CircadianHeadConfig, CircadianPredictiveCodingHead
from src.infra.continual_roles import (
    PhaseDecisionRoles,
    PhaseSource,
    release_final_test,
    split_phase_decision_roles,
)
from src.infra.datasets import LabeledData
from src.infra.difficulty_streams import DifficultyPhaseSource

PROTOCOL_ID = "difficulty_matched_modulation_v11"


@dataclass(frozen=True)
class DifficultyManifest:
    protocol_id: str = PROTOCOL_ID
    seeds: tuple[int, ...] = (17, 19)
    conditions: tuple[str, ...] = ("clean", "label_flip", "feature_outlier")
    backends: tuple[str, ...] = ("numpy", "torch_cpu")
    modulation_arms: tuple[bool, ...] = (False, True)
    hidden_dim: int = 8
    epochs_per_phase: int = 2
    learning_rate: float = 0.02
    inference_steps: int = 2
    inference_learning_rate: float = 0.1
    inner_guard_fraction: float = 0.2
    outer_selection_fraction: float = 0.2
    final_count: int = 40


@dataclass(frozen=True)
class UpdateDiagnostic:
    phase: str
    epoch: int
    actual_reward_scale: float
    mean_absolute_error_before: float
    clipped_error_before: float
    clipped_error_prior_ema_ratio: float
    prior_step_loss_improvement: float | None
    train_bce_before: float
    train_bce_after: float


@dataclass(frozen=True)
class WakeWork:
    optimizer_updates: int
    examples: int
    inference_iterations: int
    inference_example_iterations: int
    replay_updates: int
    sleep_events: int


@dataclass(frozen=True)
class DifficultyOutcome:
    seed: int
    condition: str
    backend: str
    modulated: bool
    development_role_hashes: dict[str, str]
    effective_train_hashes: dict[str, str]
    final_role_hashes: dict[str, str]
    diagnostics: tuple[UpdateDiagnostic, ...]
    work: WakeWork
    a_after_a_accuracy: float
    a_after_b_accuracy: float
    b_after_b_accuracy: float
    forgetting: float


@dataclass(frozen=True)
class DifficultyComparison:
    protocol_id: str
    manifest: DifficultyManifest
    manifest_digest: str
    outcomes: tuple[DifficultyOutcome, ...]


@dataclass
class _TrainedTrial:
    seed: int
    condition: str
    backend: str
    modulated: bool
    development_role_hashes: dict[str, str]
    effective_train_hashes: dict[str, str]
    diagnostics: tuple[UpdateDiagnostic, ...]
    work: WakeWork
    after_a_work: WakeWork
    after_a_state: Any
    after_b_model: Any


def fixed_difficulty_manifest() -> DifficultyManifest:
    """Return the only predeclared v11 configuration; no selection knobs."""
    return DifficultyManifest()


def _validate_manifest(manifest: DifficultyManifest) -> None:
    if type(manifest) is not DifficultyManifest or manifest != fixed_difficulty_manifest():
        raise ValueError("v11 difficulty manifest differs from the predeclared fixed protocol")


def _manifest_digest(manifest: DifficultyManifest) -> str:
    canonical = json.dumps(asdict(manifest), sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _train_digest(ids: tuple[str, ...], batch: LabeledData) -> str:
    digest = sha256(b"difficulty_effective_train_v11\0")
    for sample_id in ids:
        digest.update(sample_id.encode("ascii") + b"\0")
    for values in (batch.input, batch.target):
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _effective_train(roles: PhaseDecisionRoles, condition: str) -> LabeledData:
    inputs = roles.train.input.copy()
    targets = roles.train.target.copy()
    if roles.phase == "b" and condition != "clean":
        # Why this: perturb one declared B train ID after role partitioning,
        # so development decision and sealed final roles stay identical.
        row = int(np.flatnonzero(targets[:, 0] == 0.0)[0])
        if condition == "label_flip":
            targets[row, 0] = 1.0
        elif condition == "feature_outlier":
            inputs[row, 1] = 4.0
        else:
            raise ValueError("unknown difficulty condition")
    elif condition != "clean" and condition not in {"label_flip", "feature_outlier"}:
        raise ValueError("unknown difficulty condition")
    return LabeledData(inputs, targets)


def _make_model(backend: str, seed: int, modulated: bool) -> Any:
    if backend == "numpy":
        numpy_config = CircadianConfig(
            use_reward_modulated_learning=modulated,
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            replay_steps=0,
            replay_memory_size=0,
            sleep_mode="disabled",
        )
        return CircadianPredictiveCodingNetwork(
            2, 8, seed, circadian_config=numpy_config, min_hidden_dim=8, max_hidden_dim=8
        )
    if backend == "torch_cpu":
        import torch

        torch_config = CircadianHeadConfig(
            use_reward_modulated_learning=modulated,
            max_split_per_sleep=0,
            max_prune_per_sleep=0,
            sleep_mode="disabled",
        )
        return CircadianPredictiveCodingHead(
            2,
            8,
            2,
            torch.device("cpu"),
            seed,
            config=torch_config,
            min_hidden_dim=8,
            max_hidden_dim=8,
        )
    raise ValueError("unknown difficulty backend")


def _probabilities(model: Any, backend: str, inputs: np.ndarray) -> np.ndarray:
    if backend == "numpy":
        return np.asarray(model.predict_proba(inputs), dtype=np.float64)
    import torch

    with torch.no_grad():
        features = torch.as_tensor(inputs, dtype=torch.float32)
        logits = model.predict_logits(features)
        return torch.softmax(logits, dim=1)[:, 1:2].cpu().numpy().astype(np.float64)


def _update(model: Any, backend: str, batch: LabeledData, manifest: DifficultyManifest) -> float:
    if backend == "numpy":
        model.train_epoch(
            batch.input,
            batch.target,
            manifest.learning_rate,
            manifest.inference_steps,
            manifest.inference_learning_rate,
        )
        return float(model.get_last_reward_scale())
    import torch

    features = torch.as_tensor(batch.input, dtype=torch.float32)
    targets = torch.as_tensor(batch.target[:, 0], dtype=torch.long)
    model.train_step(
        features,
        targets,
        manifest.learning_rate,
        manifest.inference_steps,
        manifest.inference_learning_rate,
    )
    return float(model.last_reward_scale())


def _wake_work(model: Any, manifest: DifficultyManifest) -> WakeWork:
    clocks = model.get_sleep_clocks()
    return WakeWork(
        optimizer_updates=clocks.wake_batches,
        examples=clocks.wake_examples,
        inference_iterations=clocks.wake_batches * manifest.inference_steps,
        inference_example_iterations=clocks.wake_examples * manifest.inference_steps,
        replay_updates=clocks.replay_updates,
        sleep_events=clocks.sleep_events,
    )


def _train_trial(
    manifest: DifficultyManifest,
    roles: dict[str, PhaseDecisionRoles],
    *,
    seed: int,
    condition: str,
    backend: str,
    modulated: bool,
) -> _TrainedTrial:
    model = _make_model(backend, seed, modulated)
    diagnostics: list[UpdateDiagnostic] = []
    effective_hashes: dict[str, str] = {}
    prior_clipped_ema: float | None = None
    prior_improvement: float | None = None
    after_a_state: Any = None
    after_a_work: WakeWork | None = None
    for phase in ("a", "b"):
        batch = _effective_train(roles[phase], condition)
        effective_hashes[phase] = _train_digest(roles[phase].sample_ids["train"], batch)
        for epoch in range(manifest.epochs_per_phase):
            before = measure_train_batch(_probabilities(model, backend, batch.input), batch.target)
            ratio = (
                1.0
                if prior_clipped_ema is None or prior_clipped_ema == 0.0
                else before.clipped_error / prior_clipped_ema
            )
            scale = _update(model, backend, batch, manifest)
            after = measure_train_batch(_probabilities(model, backend, batch.input), batch.target)
            diagnostics.append(
                UpdateDiagnostic(
                    phase=phase,
                    epoch=epoch,
                    actual_reward_scale=scale,
                    mean_absolute_error_before=before.mean_absolute_error,
                    clipped_error_before=before.clipped_error,
                    clipped_error_prior_ema_ratio=ratio,
                    prior_step_loss_improvement=prior_improvement,
                    train_bce_before=before.cross_entropy,
                    train_bce_after=after.cross_entropy,
                )
            )
            prior_clipped_ema = (
                before.clipped_error
                if prior_clipped_ema is None
                else 0.95 * prior_clipped_ema + 0.05 * before.clipped_error
            )
            prior_improvement = before.cross_entropy - after.cross_entropy
        if phase == "a":
            after_a_state = model.snapshot_state()
            after_a_work = _wake_work(model, manifest)
    if after_a_work is None:
        raise ValueError("difficulty A-phase checkpoint was not captured")
    return _TrainedTrial(
        seed=seed,
        condition=condition,
        backend=backend,
        modulated=modulated,
        development_role_hashes={
            f"{phase}/{role}": phase_roles.split_hashes[role]
            for phase, phase_roles in roles.items()
            for role in ("train", "inner_guard", "outer_selection")
        },
        effective_train_hashes=effective_hashes,
        diagnostics=tuple(diagnostics),
        work=_wake_work(model, manifest),
        after_a_work=after_a_work,
        after_a_state=after_a_state,
        after_b_model=model,
    )


def _preflight(
    manifest: DifficultyManifest,
    trials: list[_TrainedTrial],
    roles_by_seed: dict[int, dict[str, PhaseDecisionRoles]],
) -> None:
    expected = {
        (seed, condition, backend, modulated)
        for seed in manifest.seeds
        for condition in manifest.conditions
        for backend in manifest.backends
        for modulated in manifest.modulation_arms
    }
    actual = [(t.seed, t.condition, t.backend, t.modulated) for t in trials]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("difficulty trials do not cover the exact fixed manifest")
    for seed, phases in roles_by_seed.items():
        if seed not in manifest.seeds or set(phases) != {"a", "b"}:
            raise ValueError("difficulty phase roles are incomplete")
        if any(role.final_released or role.final_test is not None for role in phases.values()):
            raise ValueError("difficulty final role opened before global preflight")
    for trial in trials:
        phases = roles_by_seed[trial.seed]
        expected_roles = {
            f"{phase}/{role}": phase_roles.split_hashes[role]
            for phase, phase_roles in phases.items()
            for role in ("train", "inner_guard", "outer_selection")
        }
        expected_train = {
            phase: _train_digest(
                phase_roles.sample_ids["train"], _effective_train(phase_roles, trial.condition)
            )
            for phase, phase_roles in phases.items()
        }
        expected_updates = 2 * manifest.epochs_per_phase
        expected_examples = expected_updates * len(phases["a"].train.input)
        if len(phases["a"].train.input) != len(phases["b"].train.input):
            raise ValueError("difficulty phases have unequal train exposure")
        expected_work = WakeWork(
            optimizer_updates=expected_updates,
            examples=expected_examples,
            inference_iterations=expected_updates * manifest.inference_steps,
            inference_example_iterations=expected_examples * manifest.inference_steps,
            replay_updates=0,
            sleep_events=0,
        )
        expected_a_work = WakeWork(
            optimizer_updates=manifest.epochs_per_phase,
            examples=manifest.epochs_per_phase * len(phases["a"].train.input),
            inference_iterations=manifest.epochs_per_phase * manifest.inference_steps,
            inference_example_iterations=(
                manifest.epochs_per_phase * len(phases["a"].train.input) * manifest.inference_steps
            ),
            replay_updates=0,
            sleep_events=0,
        )
        if (
            trial.development_role_hashes != expected_roles
            or trial.effective_train_hashes != expected_train
            or trial.work != expected_work
            or trial.after_a_work != expected_a_work
            or trial.after_a_state is None
            or len(trial.diagnostics) != expected_updates
        ):
            raise ValueError("difficulty role, train, state, or work preflight failed")
        previous_improvement: float | None = None
        previous_clipped_ema: float | None = None
        for index, diagnostic in enumerate(trial.diagnostics):
            expected_ratio = (
                1.0
                if previous_clipped_ema is None or previous_clipped_ema == 0.0
                else diagnostic.clipped_error_before / previous_clipped_ema
            )
            if (
                (diagnostic.phase, diagnostic.epoch)
                != (
                    ("a", "b")[index // manifest.epochs_per_phase],
                    index % manifest.epochs_per_phase,
                )
                or any(
                    not isfinite(value)
                    for value in (
                        diagnostic.actual_reward_scale,
                        diagnostic.mean_absolute_error_before,
                        diagnostic.clipped_error_before,
                        diagnostic.clipped_error_prior_ema_ratio,
                        diagnostic.train_bce_before,
                        diagnostic.train_bce_after,
                    )
                )
                or not isclose(
                    diagnostic.clipped_error_prior_ema_ratio,
                    expected_ratio,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not 0.0 <= diagnostic.clipped_error_before <= 0.5
                or not 0.0 <= diagnostic.mean_absolute_error_before <= 1.0
                or diagnostic.train_bce_before < 0.0
                or diagnostic.train_bce_after < 0.0
                or (diagnostic.prior_step_loss_improvement is None)
                != (previous_improvement is None)
                or (
                    previous_improvement is not None
                    and diagnostic.prior_step_loss_improvement is not None
                    and not isclose(
                        diagnostic.prior_step_loss_improvement,
                        previous_improvement,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                or (not trial.modulated and diagnostic.actual_reward_scale != 1.0)
                or (trial.modulated and not 0.75 <= diagnostic.actual_reward_scale <= 1.5)
            ):
                raise ValueError("difficulty train diagnostic preflight failed")
            previous_improvement = diagnostic.train_bce_before - diagnostic.train_bce_after
            previous_clipped_ema = (
                diagnostic.clipped_error_before
                if previous_clipped_ema is None
                else 0.95 * previous_clipped_ema + 0.05 * diagnostic.clipped_error_before
            )


def _accuracy(model: Any, backend: str, batch: LabeledData) -> float:
    probabilities = _probabilities(model, backend, batch.input)
    return float(np.mean((probabilities >= 0.5) == batch.target))


def run_difficulty_comparison(
    manifest: DifficultyManifest,
    *,
    source_factory: Callable[[int, str], PhaseSource] = DifficultyPhaseSource,
) -> DifficultyComparison:
    """Train and preflight all arms, then open final roles and score all."""
    _validate_manifest(manifest)
    roles_by_seed: dict[int, dict[str, PhaseDecisionRoles]] = {}
    for seed in manifest.seeds:
        roles_by_seed[seed] = {}
        for phase in ("a", "b"):
            source = source_factory(seed, phase)
            roles_by_seed[seed][phase] = split_phase_decision_roles(
                source,
                phase=phase,
                seed=seed,
                split_seed=seed * 10 + (1 if phase == "a" else 2),
                inner_guard_fraction=manifest.inner_guard_fraction,
                outer_selection_fraction=manifest.outer_selection_fraction,
                expected_final_count=manifest.final_count,
            )
    trials = [
        _train_trial(
            manifest,
            roles_by_seed[seed],
            seed=seed,
            condition=condition,
            backend=backend,
            modulated=modulated,
        )
        for seed in manifest.seeds
        for condition in manifest.conditions
        for backend in manifest.backends
        for modulated in manifest.modulation_arms
    ]
    _preflight(manifest, trials, roles_by_seed)
    released = {
        seed: {phase: release_final_test(roles) for phase, roles in phases.items()}
        for seed, phases in roles_by_seed.items()
    }
    outcomes: list[DifficultyOutcome] = []
    for trial in trials:
        final_a = released[trial.seed]["a"].final_test
        final_b = released[trial.seed]["b"].final_test
        if final_a is None or final_b is None:
            raise ValueError("difficulty final release did not bind both phases")
        after_a_model = _make_model(trial.backend, trial.seed, trial.modulated)
        after_a_model.restore_state(trial.after_a_state)
        a_after_a = _accuracy(after_a_model, trial.backend, final_a)
        a_after_b = _accuracy(trial.after_b_model, trial.backend, final_a)
        b_after_b = _accuracy(trial.after_b_model, trial.backend, final_b)
        outcomes.append(
            DifficultyOutcome(
                seed=trial.seed,
                condition=trial.condition,
                backend=trial.backend,
                modulated=trial.modulated,
                development_role_hashes=trial.development_role_hashes,
                effective_train_hashes=trial.effective_train_hashes,
                final_role_hashes={
                    phase: released[trial.seed][phase].split_hashes["final_test"]
                    for phase in ("a", "b")
                },
                diagnostics=trial.diagnostics,
                work=trial.work,
                a_after_a_accuracy=a_after_a,
                a_after_b_accuracy=a_after_b,
                b_after_b_accuracy=b_after_b,
                forgetting=a_after_a - a_after_b,
            )
        )
    return DifficultyComparison(PROTOCOL_ID, manifest, _manifest_digest(manifest), tuple(outcomes))
