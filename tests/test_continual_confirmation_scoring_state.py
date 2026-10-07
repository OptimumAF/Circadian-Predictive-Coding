"""Verify live training without reserved construction or any final accuracy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.app import continual_arrived_benchmark as arrived
from src.app import continual_confirmation_scoring_state as gate
from src.app import continual_confirmation_training as training
from src.app import continual_shift_benchmark as base
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.app.continual_confirmation_state import HeldSeed, FamilySeedFacts
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from src.infra import continual_roles
from src.infra.datasets import generate_two_cluster_dataset_with_transform


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("state proof opened a source/model/train/score/final/RNG/file")


@dataclass(frozen=True)
class _SealedSource:
    train_input: np.ndarray
    train_target: np.ndarray

    @property
    def test_input(self) -> np.ndarray:
        return _forbid()

    @property
    def test_target(self) -> np.ndarray:
        return _forbid()


@dataclass(frozen=True)
class _SealedOuter:
    @property
    def input(self) -> np.ndarray:
        return _forbid()

    @property
    def target(self) -> np.ndarray:
        return _forbid()


@pytest.fixture(scope="module")
def development_training() -> tuple[training.TrainedConfirmation, Any]:
    original = fixed_confirmation_manifest()
    families = tuple(replace(f, seeds=(f.development_seeds[0],)) for f in original.families)
    manifest = replace(original, families=families)
    allowed = {f.seeds[0] for f in families}
    arrivals = []
    original_a, original_b = arrived._build_phase_a_roles, arrived._build_phase_b_roles

    def source(**kwargs: Any) -> _SealedSource:
        data = generate_two_cluster_dataset_with_transform(**kwargs)
        return _SealedSource(data.train_input, data.train_target)

    def roles_a(config: Any, seed: int) -> Any:
        assert seed in allowed
        arrivals.append(("a", seed))
        blocked: Any = _SealedOuter()
        return replace(original_a(config, seed), outer_selection=blocked)

    def roles_b(config: Any, seed: int) -> Any:
        assert seed in allowed
        arrivals.append(("b", seed))
        blocked: Any = _SealedOuter()
        return replace(original_b(config, seed), outer_selection=blocked)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(arrived, "generate_two_cluster_dataset_with_transform", source)
        patch.setattr(base, "generate_two_cluster_dataset_with_transform", source)
        patch.setattr(arrived, "_build_phase_a_roles", roles_a)
        patch.setattr(arrived, "_build_phase_b_roles", roles_b)
        patch.setattr(continual_roles, "release_final_test", _forbid)
        held = training._train_families(families)
    assert [phase for phase, _ in arrivals] == ["a"] * 6 + ["b"] * 6
    assert not (allowed & {seed for f in original.families for seed in f.seeds})
    trained = training.TrainedConfirmation(
        training.ConfirmationTrainingFacts(manifest, tuple(item.facts for item in held)), held
    )
    # Independent original writer encoding, including exact LF terminator.
    encoded = (
        json.dumps(asdict(trained.facts), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    identity = gate.TrainingArtifactIdentity(sha256(encoded).hexdigest(), len(encoded))
    return trained, identity


def _copy_training(trained: training.TrainedConfirmation) -> training.TrainedConfirmation:
    facts = deepcopy(trained.facts)
    held = tuple(
        HeldSeed(
            row,
            item.roles_a,
            item.roles_b,
            deepcopy(item.models_after_a),
            deepcopy(item.models_after_b),
        )
        for row, item in zip(facts.seed_results, trained.held, strict=True)
    )
    return training.TrainedConfirmation(facts, held)


def _seal_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
        monkeypatch.setattr(model, "predict_proba", _forbid)
    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "compute_accuracy", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    monkeypatch.setattr(continual_roles, "release_final_test", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(Path, "read_bytes", _forbid)
    monkeypatch.setattr(Path, "read_text", _forbid)


def test_should_match_complete_existing_writer_bytes_without_deepcopy_or_join(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, expected = development_training
    _seal_verification(monkeypatch)
    # The serializer must traverse shallow dataclass fields, never asdict.
    monkeypatch.setattr("dataclasses.asdict", _forbid)
    assert gate.training_artifact_identity(trained.facts) == expected
    proof = gate._verify_reproduced_state(trained, trained.facts.manifest, expected)
    assert proof.training_artifact == expected
    assert proof.family_seed_rows == 6 and proof.cells == 56 and proof.held_checkpoints == 112
    assert proof.source_provenance_verified is False
    assert proof.final_release_authorized is False
    assert proof.scoring_manifest_sha256 is None
    assert proof.validation_scope == "training_facts_and_live_held_state_only"


@pytest.mark.parametrize("phase", ["a", "b"])
@pytest.mark.parametrize(
    "corruption", ["parameters", "traffic", "chemistry", "rng", "selector_rng", "width"]
)
def test_should_reject_last_family_live_corruption_with_unchanged_fact_bytes(
    phase: str,
    corruption: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    last = trained.held[-1]
    models = last.models_after_a if phase == "a" else last.models_after_b
    model = next(
        item for item in models.values() if isinstance(item, ParentControlledCircadianNetwork)
    )
    if corruption == "parameters":
        model.bias_output[0, 0] += 0.1
    elif corruption == "traffic":
        model._traffic_steps += 1
    elif corruption == "chemistry":
        model._hidden_chemical[0] += 0.1
    elif corruption == "rng":
        model._rng.random()
    elif corruption == "selector_rng":
        model._parent_selection_rng.random()
    else:
        model.weight_hidden_output = model.weight_hidden_output[:-1]
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


@pytest.mark.parametrize("corruption", ["phase", "seed", "final", "hash", "train_content", "count"])
def test_should_refuse_last_original_role_drift_without_outer_or_final_access(
    corruption: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    role = trained.held[-1].roles_b
    if corruption == "phase":
        role = replace(role, phase="a")
    elif corruption == "seed":
        role = replace(role, seed=role.seed + 1)
    elif corruption == "final":
        role = replace(role, final_released=True)
    elif corruption == "hash":
        role = replace(role, split_hashes={**role.split_hashes, "outer_selection": "0" * 64})
    elif corruption == "train_content":
        data = replace(role.train, input=role.train.input.copy())
        data.input[0, 0] += 1
        role = replace(role, train=data)
    else:
        role = replace(role, expected_final_count=39)
    trained.held[-1].roles_b = role
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="role"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


@pytest.mark.parametrize(
    "corruption", ["cost", "nested_checkpoint", "seal", "arrival", "metric", "numeric_type"]
)
def test_should_refuse_any_changed_fact_in_the_entire_artifact(
    corruption: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    if corruption == "cost":
        trained.facts.seed_results[-1].legacy_train_facts["methods"][0]["wake_updates"] += 1
    elif corruption == "nested_checkpoint":
        row = trained.facts.seed_results[-1]
        first = next(iter(row.after_b))
        row.after_b[first] = replace(
            row.after_b[first], parameter_count=row.after_b[first].parameter_count + 1
        )
    elif corruption == "metric":
        trained.facts.seed_results[-1].legacy_train_facts["a_after_b"] = None
    elif corruption == "numeric_type":
        row = trained.facts.seed_results[-1]
        method = row.legacy_train_facts["methods"][0]
        method["wake_updates"] = float(method["wake_updates"])
    else:
        updates: Any = (
            {"final_released": True}
            if corruption == "seal"
            else {"all_a_completed_before_first_b": False}
        )
        trained.facts = replace(trained.facts, **updates)
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="training artifact"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "order", "list", "detached_fact", "missing_arm"]
)
def test_should_refuse_wrong_held_inventory_or_detached_fact_attachment(
    corruption: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    if corruption == "missing":
        trained.held = trained.held[:-1]
    elif corruption == "duplicate":
        trained.held = (*trained.held[:-1], trained.held[0])
    elif corruption == "order":
        trained.held = trained.held[::-1]
    elif corruption == "list":
        held: Any = list(trained.held)
        trained.held = held
    elif corruption == "detached_fact":
        trained.held[-1].facts = deepcopy(trained.held[-1].facts)
    else:
        trained.held[-1].models_after_a.pop(next(iter(trained.held[-1].models_after_a)))
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="inventory|attachment|checkpoint"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


def test_should_recheck_all_fact_bytes_after_last_live_verification(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    original_check = gate.require_held_seed

    def corrupt_after_check(item: HeldSeed) -> None:
        original_check(item)
        if item is trained.held[-1]:
            trained.facts.seed_results[0].legacy_train_facts["unexpected_late_fact"] = True

    monkeypatch.setattr(gate, "require_held_seed", corrupt_after_check)
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="training artifact"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


def test_should_refuse_last_check_inventory_drift_before_returning_a_proof(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original, identity = development_training
    trained = _copy_training(original)
    original_check = gate.require_held_seed
    last = trained.held[-1]

    def corrupt_after_check(item: HeldSeed) -> None:
        original_check(item)
        if item is last:
            trained.held = trained.held[:-1]

    monkeypatch.setattr(gate, "require_held_seed", corrupt_after_check)
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="inventory"):
        gate._verify_reproduced_state(trained, trained.facts.manifest, identity)


def test_should_refuse_partial_development_training_at_the_public_gate_before_live_checks(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, _ = development_training
    _seal_verification(monkeypatch)
    monkeypatch.setattr(gate, "require_held_seed", _forbid)
    with pytest.raises(ValueError, match="inventory"):
        gate.verify_scoring_training_state(trained, fixed_scoring_manifest())


def _metadata_training() -> training.TrainedConfirmation:
    # Deliberately invalid scientific bodies: only test sixty-row delegation.
    manifest = fixed_confirmation_manifest()
    omitted_roles: Any = ()
    rows = tuple(
        FamilySeedFacts(f.name, seed, omitted_roles, {}, {}, {}, {}, ())
        for f in manifest.families
        for seed in f.seeds
    )
    role: Any = None
    held = tuple(HeldSeed(row, role, role, {}, {}) for row in rows)
    return training.TrainedConfirmation(training.ConfirmationTrainingFacts(manifest, rows), held)


@pytest.mark.parametrize("late_failure", [False, True])
def test_should_dispatch_the_complete_public_inventory_before_returning_any_proof(
    late_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = _metadata_training()
    manifest = fixed_scoring_manifest()
    identity = gate.TrainingArtifactIdentity(
        manifest.training_bundles[0].result_sha256, manifest.training_bundles[0].result_bytes
    )
    # These spies are explicit metadata-only test seams, never production data.
    monkeypatch.setattr(gate, "training_artifact_identity", lambda facts: identity)
    checked = []

    def check(item: HeldSeed) -> None:
        checked.append((item.facts.family, item.facts.seed))
        if item is trained.held[-1] and late_failure:
            raise ValueError("late complete checkpoint differs")

    monkeypatch.setattr(gate, "require_held_seed", check)
    _seal_verification(monkeypatch)
    if late_failure:
        with pytest.raises(ValueError, match="late complete checkpoint"):
            gate.verify_scoring_training_state(trained, manifest)
    else:
        proof = gate.verify_scoring_training_state(trained, manifest)
        assert (
            proof.family_seed_rows == 60 and proof.cells == 560 and proof.held_checkpoints == 1120
        )
        assert not proof.final_release_authorized and not proof.source_provenance_verified
        assert proof.scoring_manifest_sha256 == scoring_manifest_digest(manifest)
    assert checked == [(f.name, seed) for f in manifest.train_manifest.families for seed in f.seeds]


def test_should_refuse_unbound_whole_scope_metadata_without_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = _metadata_training()
    monkeypatch.setattr(gate, "require_held_seed", _forbid)
    _seal_verification(monkeypatch)
    with pytest.raises(ValueError, match="training artifact"):
        gate.verify_scoring_training_state(trained, fixed_scoring_manifest())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_should_fail_loudly_on_nonfinite_or_unencodable_artifact_state(value: Any) -> None:
    trained = _metadata_training()
    trained.facts.seed_results[-1].legacy_train_facts["bad_value"] = value
    with pytest.raises(ValueError, match="training artifact encoding"):
        gate.training_artifact_identity(trained.facts)
