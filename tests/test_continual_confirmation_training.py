"""Development fixtures verify composition before any reserved source runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_combined_factor_preflight as combined
from src.app import continual_confirmation_training as training
from src.app import continual_gating_pilot as gating
from src.app import continual_parent_factor_preflight as parent
from src.app import continual_replay_factor_pilot as replay
from src.app import continual_schedule_factor_preflight as schedule
from src.app import continual_shift_benchmark as base
from src.app import continual_sleep_factor_preflight as sleep
from src.app.continual_confirmation_manifest import ConfirmationFamily, fixed_confirmation_manifest
from src.app.continual_combined_factor_manifest import fixed_combined_manifest
from src.app.continual_parent_factor_manifest import fixed_parent_manifest
from src.app.continual_confirmation_state import (
    HeldSeed,
    capture_models,
    finite_json,
    require_held_seed,
)
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


NAMES = ("gating", "replay", "sleep", "schedule", "combined", "parent")
LEGACY = {
    "gating": (gating, "_new_models", gating.fixed_gating_pilot_manifest),
    "replay": (replay, "_models", replay.fixed_replay_pilot_manifest),
    "sleep": (sleep, "_new_models", sleep.fixed_sleep_factor_manifest),
    "schedule": (schedule, "_new_models", schedule.fixed_schedule_factor_manifest),
    "combined": (combined, "_new_progress", fixed_combined_manifest),
    "parent": (parent, "_new_progress", fixed_parent_manifest),
}


def _development_families() -> tuple[ConfirmationFamily, ...]:
    # First existing seed by manifest order is chosen before fixture outcomes.
    return tuple(
        replace(family, seeds=(family.development_seeds[0],))
        for family in fixed_confirmation_manifest().families
    )


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("confirmation composition opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("confirmation composition opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("confirmation composition opened outer inputs")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("confirmation composition opened outer labels")


@dataclass(frozen=True)
class _OmittedOuter:
    # Original gating runner passes these to its stubbed scorer. No actual
    # outer value or accuracy is available in the reference fixture.
    input: Any = None
    target: Any = None


def _seal_sources(monkeypatch: pytest.MonkeyPatch, *, blocked_outer: bool = True) -> None:
    def generator(**kwargs: Any) -> _SealedSource:
        data = generate_two_cluster_dataset_with_transform(**kwargs)
        return _SealedSource(data.train_input, data.train_target)

    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles

    def roles_a(*args: Any) -> Any:
        outer: Any = _BlockedOuter() if blocked_outer else _OmittedOuter()
        return replace(original_a(*args), outer_selection=outer)

    def roles_b(*args: Any) -> Any:
        outer: Any = _BlockedOuter() if blocked_outer else _OmittedOuter()
        return replace(original_b(*args), outer_selection=outer)

    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", generator)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", roles_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", roles_b)


def _no_scoring(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("confirmation composition called a scoring runner/helper")


@pytest.fixture(scope="module")
def legacy_references() -> dict[str, dict[str, Any]]:
    references = {}
    for family in _development_families():
        module, factory_name, manifest_factory = LEGACY[family.name]
        original_factory = getattr(module, factory_name)
        captured: dict[str, Any] = {}
        with pytest.MonkeyPatch.context() as patch:
            _seal_sources(patch, blocked_outer=False)
            patch.setattr(gating, "_accuracy", lambda *args: 0.0)
            patch.setattr(replay, "_accuracy", lambda *args: 0.0)

            def factory(*args: Any, **kwargs: Any) -> Any:
                made = original_factory(*args, **kwargs)
                models = (
                    {
                        "ordinary_pc": made.ordinary,
                        "neutral_circadian": made.neutral,
                        "chemical_gating": made.gating,
                    }
                    if family.name == "gating"
                    else made
                    if isinstance(made, dict)
                    else made.models
                )
                captured["models"] = models
                captured["initial"] = capture_models(models, family.arms)
                return made

            original_b = arrived._build_phase_b_roles

            def capture_a(*args: Any) -> Any:
                captured["after_a"] = capture_models(captured["models"], family.arms)
                return original_b(*args)

            patch.setattr(module, factory_name, factory)
            patch.setattr(arrived, "_build_phase_b_roles", capture_a)
            manifest = manifest_factory()
            legacy = finite_json(module._run_seed(manifest, family.seeds[0]))
            for method in legacy.get("methods", []):
                method.pop("development", None)
            captured["after_b"] = capture_models(captured["models"], family.arms)
            captured["legacy"] = legacy
            del captured["models"]
            references[family.name] = captured
    return references


@pytest.fixture(scope="module")
def composed() -> tuple[HeldSeed, ...]:
    with pytest.MonkeyPatch.context() as patch:
        _seal_sources(patch)
        for module, _, _ in LEGACY.values():
            patch.setattr(module, "_run_seed", _no_scoring)
        patch.setattr(gating, "_accuracy", _no_scoring)
        patch.setattr(replay, "_accuracy", _no_scoring)
        return training._train_families(_development_families())


@pytest.mark.parametrize("family", NAMES)
def test_should_preserve_exact_legacy_trajectory_and_unscored_costs(
    family: str,
    composed: tuple[HeldSeed, ...],
    legacy_references: dict[str, dict[str, Any]],
) -> None:
    held = next(item for item in composed if item.facts.family == family)
    reference = legacy_references[family]
    assert held.facts.initial == reference["initial"]
    assert held.facts.after_a == reference["after_a"]
    assert held.facts.after_b == reference["after_b"]
    assert held.facts.legacy_train_facts == reference["legacy"]
    assert held.facts.final_released is held.facts.outer_selection_scored is False
    require_held_seed(held)


def test_should_complete_all_family_a_work_before_any_b_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = _development_families()
    expected = [(family.name, family.seeds[0]) for family in families]
    completed = []
    original_a = training._train_a
    _seal_sources(monkeypatch)
    original_b = arrived._build_phase_b_roles

    def train_a(*args: Any) -> Any:
        item = original_a(*args)
        completed.append((item.family, item.seed))
        return item

    def source_b(*args: Any) -> Any:
        assert completed == expected
        return original_b(*args)

    monkeypatch.setattr(training, "_train_a", train_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", source_b)
    held = training._train_families(families)
    assert [(item.facts.family, item.facts.seed) for item in held] == expected
    assert not (
        {seed for _, seed in expected}
        & {seed for family in fixed_confirmation_manifest().families for seed in family.seeds}
    )


@pytest.mark.parametrize("change", ["partial_family", "development_seed", "budget", "seal"])
def test_should_refuse_changed_production_scope_before_models_or_sources(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    manifest = fixed_confirmation_manifest()
    if change == "partial_family":
        manifest = replace(manifest, families=manifest.families[:-1])
    elif change == "development_seed":
        family = replace(manifest.families[0], seeds=(41,))
        manifest = replace(manifest, families=(family, *manifest.families[1:]))
    elif change == "budget":
        manifest = replace(manifest, max_optimizer_updates=16001)
    else:
        manifest = replace(manifest, global_train_gate_required=False)
    monkeypatch.setattr(training, "_train_a", _no_scoring)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _no_scoring)
    with pytest.raises(ValueError, match="frozen complete scope"):
        training.train_confirmation(manifest)


def test_should_refuse_resolved_factory_drift_before_any_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = _development_families()[0]
    original = training._TRAJECTORIES["gating"]

    def changed(seed: int) -> Any:
        trajectory = original(seed)
        trajectory.manifest = replace(
            trajectory.manifest,
            source=replace(trajectory.manifest.source, guard_drop_tolerance=0.1),
        )
        return trajectory

    monkeypatch.setitem(training._TRAJECTORIES, "gating", changed)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _no_scoring)
    with pytest.raises(ValueError, match="trajectory settings"):
        training._train_a(family, family.seeds[0])


def test_should_refuse_undeclared_seed_before_any_model(monkeypatch: pytest.MonkeyPatch) -> None:
    family = _development_families()[0]
    monkeypatch.setitem(training._TRAJECTORIES, "gating", _no_scoring)
    with pytest.raises(ValueError, match="trajectory seed"):
        training._train_a(family, 43)


@pytest.mark.parametrize("target", ["held", "live"])
@pytest.mark.parametrize("corruption", ["parameters", "traffic", "noise_rng", "selector_rng"])
def test_should_refuse_late_a_checkpoint_before_first_b_source(
    monkeypatch: pytest.MonkeyPatch, target: str, corruption: str
) -> None:
    _seal_sources(monkeypatch)
    original = training._train_a

    def changed(*args: Any) -> Any:
        item = original(*args)
        if item.family == "parent":
            models = item.models if target == "held" else item.trajectory.models
            if corruption == "traffic":
                models["pc_13_off"]._traffic_steps += 1
            else:
                model = models["random_growth"]
                assert isinstance(model, ParentControlledCircadianNetwork)
                if corruption == "parameters":
                    model.weight_input_hidden[0, 0] += 1
                elif corruption == "noise_rng":
                    model._rng.random()
                else:
                    model._parent_selection_rng.random()
        return item

    monkeypatch.setattr(training, "_train_a", changed)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _no_scoring)
    with pytest.raises(ValueError, match="complete checkpoint changed"):
        training._train_families(_development_families())


@pytest.mark.parametrize("checkpoint", ["a", "b"])
def test_should_recheck_late_all_family_checkpoint_after_b(
    monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    _seal_sources(monkeypatch)
    original = training._train_b

    def changed(item: Any) -> HeldSeed:
        held = original(item)
        if held.facts.family == "parent":
            models = held.models_after_a if checkpoint == "a" else held.models_after_b
            model = models["scheduled_growth"]
            assert isinstance(model, ParentControlledCircadianNetwork)
            model._parent_selection_cursor += 1
        return held

    monkeypatch.setattr(training, "_train_b", changed)
    with pytest.raises(ValueError, match="complete checkpoint changed"):
        training._train_families(_development_families())


def test_should_refuse_late_a_guard_label_change_before_b_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_sources(monkeypatch)
    original = training._train_a

    def changed(*args: Any) -> Any:
        item = original(*args)
        if item.family == "parent":
            item.roles.inner_guard.target[0, 0] = 1 - item.roles.inner_guard.target[0, 0]
        return item

    monkeypatch.setattr(training, "_train_a", changed)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _no_scoring)
    with pytest.raises(ValueError, match="train/guard content"):
        training._train_families(_development_families())


def test_should_serialize_every_fixture_checkpoint_and_raw_unscored_row(
    composed: tuple[HeldSeed, ...],
) -> None:
    from dataclasses import asdict

    encoded = finite_json(
        {"implementation_fixture": True, "seed_results": [asdict(item.facts) for item in composed]}
    )
    assert [row["family"] for row in encoded["seed_results"]] == list(NAMES)
    assert sum(len(row["after_b"]) for row in encoded["seed_results"]) == 56
    for row in encoded["seed_results"]:
        assert row["outer_selection_scored"] is row["final_released"] is False
        assert len(row["initial"]) == len(row["after_a"]) == len(row["after_b"])
        assert all(
            "development" not in method for method in row["legacy_train_facts"].get("methods", [])
        )


@pytest.mark.parametrize("family", NAMES)
def test_should_hold_independent_a_arrays_when_b_or_reference_changes(
    family: str, composed: tuple[HeldSeed, ...]
) -> None:
    item = next(item for item in composed if item.facts.family == family)
    for name in item.models_after_a:
        assert not np.shares_memory(
            item.models_after_a[name].weight_input_hidden,
            item.models_after_b[name].weight_input_hidden,
        )
    held = replace(
        item,
        facts=deepcopy(item.facts),
        models_after_a=deepcopy(item.models_after_a),
        models_after_b=deepcopy(item.models_after_b),
    )
    first = next(iter(held.models_after_b.values()))
    first.weight_hidden_output[0, 0] += 1
    with pytest.raises(ValueError, match="complete checkpoint changed"):
        require_held_seed(held)
    require_held_seed(item)


def _reject_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[CircadianPredictiveCodingNetwork, int] = {}

    def accuracy(model: CircadianPredictiveCodingNetwork, *args: Any) -> float:
        count = calls.get(model, 0)
        calls[model] = count + 1
        return 1.0 if count % 2 == 0 else 0.0

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", accuracy)


@pytest.mark.parametrize("family", ["sleep", "schedule"])
def test_should_witness_complete_rejections_and_retain_executed_replay(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    _seal_sources(monkeypatch)
    _reject_guards(monkeypatch)
    original_step = CircadianPredictiveCodingNetwork._run_training_step
    replay_executions = 0

    def step(model: CircadianPredictiveCodingNetwork, *args: Any, **kwargs: Any) -> float:
        nonlocal replay_executions
        replay_executions += int(kwargs.get("update_epoch_state") is False)
        return original_step(model, *args, **kwargs)

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "_run_training_step", step)
    scope = next(item for item in _development_families() if item.name == family)
    held = training._train_families((scope,))[0]
    guards = held.facts.supplemental_guards
    rejected = [w for w in guards if w.outcome == "rolled_back"]
    assert len(rejected) == (5 if family == "sleep" else 6)
    assert all(w.before == w.after for w in guards)
    methods = held.facts.legacy_train_facts.get("methods", [])
    assert replay_executions == (0 if family == "sleep" else 12)
    assert replay_executions == sum(row["rejected_executed_replay_updates"] for row in methods)
    assert all(row["applied_replay_updates"] == 0 for row in methods)


@pytest.mark.parametrize("family", ["sleep", "schedule"])
def test_should_refuse_full_state_corruption_after_old_parameter_only_guard(
    monkeypatch: pytest.MonkeyPatch, family: str
) -> None:
    _seal_sources(monkeypatch)
    _reject_guards(monkeypatch)
    if family == "sleep":
        original = sleep._apply_guarded_sleep

        def broken(models: Any, *args: Any) -> Any:
            facts = original(models, *args)
            models["homeostasis_only"]._hidden_chemical[0] += 1
            return facts

        monkeypatch.setattr(sleep, "_apply_guarded_sleep", broken)
    else:
        original_decision = schedule._decide_and_apply

        def broken_decision(models: Any, roles: Any, manifest: Any, policy: str, *args: Any) -> Any:
            facts = original_decision(models, roles, manifest, policy, *args)
            if facts.outcome == "rolled_back":
                models[f"neutral_{policy}"]._rng.random()
            return facts

        monkeypatch.setattr(schedule, "_decide_and_apply", broken_decision)
    scope = next(item for item in _development_families() if item.name == family)
    with pytest.raises(ValueError, match="changed complete state"):
        training._train_families((scope,))
