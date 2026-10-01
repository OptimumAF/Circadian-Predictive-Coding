"""Frozen parent controls preserve arrived roles and complete proposed/rollback state."""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_parent_factor_preflight as study
from src.app import continual_shift_benchmark as base
from src.app.continual_parent_factor_manifest import (
    GROWTH_ARMS,
    fixed_parent_manifest,
    validate_parent_manifest,
)
from src.app.continual_parent_factor_validation import verify_parent_payload
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra.datasets import generate_two_cluster_dataset_with_transform


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        raise AssertionError("parent factor opened final inputs")

    @property
    def test_target(self) -> np.ndarray:
        raise AssertionError("parent factor opened final labels")


@dataclass(frozen=True)
class _BlockedOuter:
    @property
    def input(self) -> np.ndarray:
        raise AssertionError("parent factor opened outer inputs")

    @property
    def target(self) -> np.ndarray:
        raise AssertionError("parent factor opened outer labels")


def _sealed_generator(**kwargs: Any) -> _SealedSource:
    data = generate_two_cluster_dataset_with_transform(**kwargs)
    return _SealedSource(data.train_input, data.train_target)


def test_should_freeze_parent_modes_and_work_before_source_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixed_parent_manifest()
    assert validate_parent_manifest(manifest) == 576
    assert manifest.seeds == (347, 349, 353) and len(manifest.arms) == 8
    assert manifest.add_epochs == (4, 8, 12, 16, 20)
    assert not set(manifest.seeds) & set(manifest.confirmation_seeds)
    arms = [arm for arm in manifest.arms if arm.name in GROWTH_ARMS]
    assert [arm.parent_mode for arm in arms] == ["usage", "scheduled", "random"]
    assert arms[0].config == arms[1].config == arms[2].config
    assert {(arm.width, arm.min_width, arm.max_width) for arm in arms} == {(8, 8, 13)}

    def forbidden_source(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("changed manifest opened source")

    monkeypatch.setattr(arrived, "_build_phase_a_roles", forbidden_source)
    for changed in (
        replace(manifest, seeds=(347,)),
        replace(manifest, add_epochs=(4, 24)),
        replace(manifest, arms=manifest.arms[:-1]),
        replace(manifest, selector_seed_offset=5002),
        replace(manifest, max_optimizer_updates=599),
    ):
        with pytest.raises(ValueError, match="frozen manifest"):
            study.run_parent_preflight(changed)


def test_should_repeat_every_cell_with_outer_final_and_arrival_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(arrived, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    monkeypatch.setattr(base, "generate_two_cluster_dataset_with_transform", _sealed_generator)
    original_a, original_b, new_progress = (
        arrived._build_phase_a_roles,
        arrived._build_phase_b_roles,
        study._new_progress,
    )
    live: dict[int, study._Progress] = {}
    wake_calls: list[int] = []

    def capture_progress(manifest: Any, seed: int) -> Any:
        progress = new_progress(manifest, seed)
        live[seed] = progress
        return progress

    def phase_a(*args: Any) -> Any:
        return replace(original_a(*args), outer_selection=cast(Any, _BlockedOuter()))

    def phase_b(*args: Any) -> Any:
        progress = live[args[1]]
        assert len(progress.opportunities) == 12
        assert all(work.wake_updates == 12 for work in progress.work.values())
        assert len(progress.models) == 8
        return replace(original_b(*args), outer_selection=cast(Any, _BlockedOuter()))

    def spy_training(cls: Any) -> None:
        original = cls.train_epoch

        def train(self: Any, inputs: Any, targets: Any, *args: Any, **kwargs: Any) -> Any:
            wake_calls.append(len(inputs))
            return original(self, inputs, targets, *args, **kwargs)

        monkeypatch.setattr(cls, "train_epoch", train)

    for cls in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        spy_training(cls)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", phase_a)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", phase_b)
    monkeypatch.setattr(study, "_new_progress", capture_progress)
    manifest = fixed_parent_manifest()
    first = study.json_value(study.run_parent_preflight(manifest))
    assert len(wake_calls) == 576 and sum(wake_calls) == 31104
    second = study.json_value(study.run_parent_preflight(manifest))
    assert first == second and len(wake_calls) == 1152
    assert first["outer_selection_scored"] is False and first["final_released"] is False
    for seed in first["seed_results"]:
        assert seed["executed_optimizer_updates"] == 192
        assert len(seed["methods"]) == 8 and len(seed["opportunities"]) == 24
        for name in GROWTH_ARMS:
            due = [
                row
                for opportunity in seed["opportunities"]
                for row in opportunity["decisions"]
                if row["name"] == name and row["event"]["guard"] is not None
            ]
            assert len(due) == 6 and [row["requested_add_count"] for row in due] == [
                1,
                1,
                1,
                1,
                1,
                0,
            ]
            assert all(row["after"] == row["proposed"] for row in due)
            assert due[-1]["proposed"]["selector"] == due[-1]["before"]["selector"]
    verify_parent_payload(first, study.json_value(manifest))


def _prepared_progress() -> tuple[study._Progress, Any]:
    manifest = fixed_parent_manifest()
    roles = arrived._build_phase_a_roles(manifest.source, manifest.seeds[0])
    progress = study._new_progress(manifest, manifest.seeds[0])
    for _ in range(4):
        study._train_wake(progress, roles)
        progress.shared.observe_train_batch(roles.train.input, roles.train.target)
    return progress, roles


@pytest.mark.parametrize("case", ["content", "order"])
def test_should_check_all_retained_contents_and_order_without_replay(case: str) -> None:
    progress, _ = _prepared_progress()
    model = progress.models["random_growth"]
    assert isinstance(model, ParentControlledCircadianNetwork)
    assert model.config.replay_prioritized and not model.config.sleep_enable_replay
    available = progress.shared.select_recent(8)
    study._check_retained_supply(model, progress.shared, available)
    if case == "content":
        model._replay_memory[0].input_batch[0, 0] += 0.1
    else:
        model._replay_memory = deque(reversed(model._replay_memory))
    with pytest.raises(ValueError, match="retained supply"):
        study._check_retained_supply(model, progress.shared, available)


@pytest.mark.parametrize("name", GROWTH_ARMS)
def test_should_retain_proposed_selector_and_restore_guard_rejection_before_retry(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress, roles = _prepared_progress()
    clean, clean_roles = _prepared_progress()
    model = progress.models[name]
    assert isinstance(model, ParentControlledCircadianNetwork)
    arm = next(arm for arm in progress.manifest.arms if arm.name == name)
    saved = study._capture(model)
    core_calls = []
    original_sleep = ParentControlledCircadianNetwork.sleep_event

    def core(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_sleep(self, *args, **kwargs)
        core_calls.append(result.split_indices)
        return result

    values = iter((1.0, 0.0))
    with monkeypatch.context() as patch:
        patch.setattr(ParentControlledCircadianNetwork, "sleep_event", core)
        patch.setattr(
            ParentControlledCircadianNetwork, "compute_accuracy", lambda *args: next(values)
        )
        rejected = study._apply_growth_sleep(progress, roles, arm, 4, 4)
    assert len(core_calls) == 1 and len(core_calls[0]) == 1
    assert rejected["event"]["outcome"] == "rolled_back"
    assert rejected["before"] == rejected["after"] == saved
    assert rejected["proposed"]["selector"]["selection_calls"] == 1
    assert rejected["proposed"]["state_sha256"] != saved["state_sha256"]
    retry = study._apply_growth_sleep(progress, roles, arm, 4, 4)
    reference = study._apply_growth_sleep(clean, clean_roles, arm, 4, 4)
    assert retry == reference
    assert retry["proposed"]["selector"] == rejected["proposed"]["selector"]
    assert progress.work[name].attempts == 2 and progress.work[name].accepted == 1


@pytest.mark.parametrize("case", ["pre_raise", "pre_nan", "core_raise", "post_raise", "post_nan"])
def test_should_restore_full_state_on_guard_or_core_failure(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress, roles = _prepared_progress()
    arm = next(arm for arm in progress.manifest.arms if arm.name == "random_growth")
    model = progress.models[arm.name]
    assert isinstance(model, ParentControlledCircadianNetwork)
    before = study._capture(model)
    original_sleep = ParentControlledCircadianNetwork.sleep_event
    calls = 0

    def accuracy(*args: Any) -> float:
        nonlocal calls
        calls += 1
        if (calls == 1 and case == "pre_raise") or (calls == 2 and case == "post_raise"):
            raise RuntimeError("injected guard failure")
        if (calls == 1 and case == "pre_nan") or (calls == 2 and case == "post_nan"):
            return float("nan")
        return 1.0

    def core(self: Any, *args: Any, **kwargs: Any) -> Any:
        original_sleep(self, *args, **kwargs)
        raise RuntimeError("injected post-core failure")

    monkeypatch.setattr(ParentControlledCircadianNetwork, "compute_accuracy", accuracy)
    if case == "core_raise":
        monkeypatch.setattr(ParentControlledCircadianNetwork, "sleep_event", core)
    with pytest.raises((ValueError, RuntimeError)):
        study._apply_growth_sleep(progress, roles, arm, 4, 4)
    assert study._capture(model) == before


@pytest.fixture(scope="module")
def payload() -> dict[str, Any]:
    return study.json_value(study.run_parent_preflight(fixed_parent_manifest()))


@pytest.mark.parametrize(
    "case",
    [
        "count",
        "parent",
        "rng",
        "cursor",
        "calls",
        "lineage",
        "clocks",
        "guard",
        "work",
        "checkpoint",
        "before_checkpoint",
        "supply",
        "capacity",
        "role",
        "cell",
        "seed",
        "scored",
        "final",
        "extra_score",
    ],
)
def test_should_reject_forged_parent_guard_checkpoint_and_work_facts(
    case: str, payload: dict[str, Any]
) -> None:
    changed = deepcopy(payload)
    seed = changed["seed_results"][-1]
    row = seed["opportunities"][3]["decisions"][-1]
    if case == "count":
        row["requested_add_count"] = 0
    elif case == "parent":
        row["proposed"]["selector"]["last_decision"]["selected_parent_ids"] = [99]
    elif case == "rng":
        row["proposed"]["selector"]["rng_sha256"] = "0" * 64
    elif case == "cursor":
        row["proposed"]["selector"]["cursor_id"] = 99
    elif case == "calls":
        row["proposed"]["selector"]["selection_calls"] = 2
    elif case == "lineage":
        row["proposed"]["lineage"]["parent_ids"][-1] = 99
    elif case == "clocks":
        row["after"]["clocks"]["sleep_events"] = 0
    elif case == "guard":
        row["event"]["guard"]["tolerance"] = 0.1
    elif case == "work":
        seed["methods"][-1]["wake_updates"] = 23
    elif case == "checkpoint":
        seed["methods"][3]["after_a_state_sha256"] = "0" * 64
    elif case == "before_checkpoint":
        row["before"]["state_sha256"] = "0" * 64
    elif case == "supply":
        seed["opportunities"][3]["retained_bytes"] = 168
    elif case == "capacity":
        seed["opportunities"][3]["after_epoch_widths"]["random_growth"] = 14
    elif case == "role":
        seed["role_counts"]["b_inner_guard"] = 24
    elif case == "cell":
        seed["opportunities"][-1]["decisions"].pop()
    elif case == "seed":
        changed["seed_results"].pop()
    elif case == "scored":
        changed["outer_selection_scored"] = True
    elif case == "final":
        seed["final_released"] = True
    else:
        seed["methods"][0]["accuracy"] = 0.9
    with pytest.raises(ValueError, match="P6.3 parent"):
        verify_parent_payload(changed, study.json_value(fixed_parent_manifest()))
