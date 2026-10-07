"""Development training/fabricated finals prove orchestration, not confirmation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test_continual_confirmation_scoring_state import (
    development_training as development_training,
    _copy_training,
)
from test_continual_confirmation_final_adapter import _FabricatedSource
from src.app import continual_confirmation_scoring as scoring
from src.app import continual_confirmation_scoring_state as state
from src.app import continual_arrived_benchmark as arrived
from src.app.continual_confirmation_analysis import analyze_confirmation
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.app.continual_confirmation_state import FamilySeedFacts, HeldSeed, RoleFacts
from src.app.continual_confirmation_training import ConfirmationTrainingFacts, TrainedConfirmation
from src.core.confirmation_final_roles import (
    EndpointFailure,
    EndpointResult,
    FinalRole,
    final_content_digest,
)
from src.core.controlled_parent_selection import ParentControlledCircadianNetwork
from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra import continual_confirmation_final as adapter


def _forbid(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("scoring fixture opened reserved/original-final/outer/train access")


def _fabricated_role(phase: str, seed: int, ids: tuple[str, ...]) -> FinalRole:
    inputs = np.arange(80, dtype=np.float64).reshape(40, 2) / 100 + seed / 10000
    targets = (np.arange(40) % 2).astype(np.float64).reshape(40, 1)
    role = FinalRole(phase, seed, ids, inputs, targets, "")
    return replace(role, sha256=final_content_digest(role))


def _fixture_training(development_training: tuple[Any, Any]) -> tuple[Any, Any, list[Any]]:
    original, identity = development_training
    trained = _copy_training(original)
    sources = []
    for item in trained.held:
        for phase, name in (("a", "roles_a"), ("b", "roles_b")):
            original_role = getattr(item, name)
            final = _fabricated_role(phase, item.facts.seed, original_role.sample_ids["final_test"])
            source = _FabricatedSource(final.input, final.target)
            sources.append(source)
            setattr(item, name, replace(original_role, _source=source))
    return trained, identity, sources


def _seal_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(arrived, "_build_phase_a_roles", _forbid)
    monkeypatch.setattr(arrived, "_build_phase_b_roles", _forbid)
    for model in (
        BackpropMLP,
        PredictiveCodingNetwork,
        CircadianPredictiveCodingNetwork,
        ParentControlledCircadianNetwork,
    ):
        monkeypatch.setattr(model, "__init__", _forbid)
        monkeypatch.setattr(model, "train_epoch", _forbid)
    monkeypatch.setattr(np.random, "default_rng", _forbid)
    monkeypatch.setattr(Path, "read_bytes", _forbid)
    monkeypatch.setattr(Path, "read_text", _forbid)


def _run_fixture(
    trained: Any,
    identity: Any,
    *,
    release: Any = adapter.release_confirmation_final,
    evaluate: Any = adapter.evaluate_confirmation_final,
    checkpoint: Any = lambda stage: None,
) -> Any:
    return scoring._evaluate_trained(
        trained,
        trained.facts.manifest,
        lambda: state._verify_reproduced_state(trained, trained.facts.manifest, identity),
        release,
        evaluate,
        checkpoint,
    )


def test_should_release_all_fabricated_roles_before_any_development_model_evaluation(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    releases, calls = [], []
    stages: list[str] = []
    expected: list[tuple[int, str]] = []
    families = {family.name: family for family in trained.facts.manifest.families}
    for item in trained.held:
        for arm in families[item.facts.family].arms:
            expected.extend(
                (
                    (id(item.models_after_a[arm]), "a"),
                    (id(item.models_after_b[arm]), "a"),
                    (id(item.models_after_b[arm]), "b"),
                )
            )

    def release(item: HeldSeed, phase: str) -> FinalRole:
        releases.append((item.facts.family, item.facts.seed, phase))
        return adapter.release_confirmation_final(item, phase)

    def evaluate(model: Any, role: FinalRole) -> EndpointResult:
        assert len(releases) == 12
        calls.append((id(model), role.phase))
        return adapter.evaluate_confirmation_final(model, role)

    result = _run_fixture(
        trained, identity, release=release, evaluate=evaluate, checkpoint=stages.append
    )
    assert calls == expected
    assert stages[0] == "before_final_release" and stages[-1] == "after_final_evaluation"
    assert stages.count("before_endpoint") == stages.count("after_endpoint") == 168
    assert all((source.reads_input, source.reads_target) == (1, 1) for source in sources)
    assert asdict(result.totals) == {
        "release_calls": 12,
        "final_role_views": 12,
        "endpoint_calls": 168,
        "example_count": 6720,
        "successful_endpoints": 168,
        "failed_endpoints": 0,
        "successful_cells": 56,
        "failed_cells": 0,
    }
    assert (
        result.training_before_release
        == result.training_after_release
        == result.training_after_evaluation
    )
    assert result.training_after_evaluation.training_artifact == identity
    assert result.outer_selection_scored is result.external_execution_verified is False
    assert result.scoring_manifest_sha256 is None
    assert all(item.roles_a.final_test is item.roles_b.final_test is None for item in trained.held)


def test_should_repeat_every_fabricated_metric_count_role_and_state_proof_exactly(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, identity, _ = _fixture_training(development_training)
    second, other_identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    assert asdict(_run_fixture(first, identity)) == asdict(_run_fixture(second, other_identity))


def test_should_reject_late_training_state_before_first_final_access(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    last = next(
        model
        for model in trained.held[-1].models_after_a.values()
        if isinstance(model, ParentControlledCircadianNetwork)
    )
    last._parent_selection_rng.random()
    _seal_scoring(monkeypatch)
    with pytest.raises(ValueError, match="checkpoint"):
        _run_fixture(trained, identity, release=_forbid, evaluate=_forbid)
    assert all(source.reads_input == source.reads_target == 0 for source in sources)


@pytest.mark.parametrize("change", ["ids", "phase", "seed", "hash", "dtype", "count"])
def test_should_reject_last_released_role_before_first_evaluation(
    change: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)

    def release(item: HeldSeed, phase: str) -> FinalRole:
        role = adapter.release_confirmation_final(item, phase)
        if item is not trained.held[-1] or phase != "b":
            return role
        updates: Any = {
            "ids": {"sample_ids": role.sample_ids[::-1]},
            "phase": {"phase": "a"},
            "seed": {"seed": role.seed + 1},
            "hash": {"sha256": "0" * 64},
            "dtype": {"input": role.input.astype(np.float32)},
            "count": {"sample_ids": role.sample_ids[:-1]},
        }[change]
        changed = replace(role, **updates)
        if change in {"ids", "phase", "seed"}:
            changed = replace(changed, sha256=final_content_digest(changed))
        return changed

    with pytest.raises(ValueError, match="final role"):
        _run_fixture(trained, identity, release=release, evaluate=_forbid)


def test_should_reject_training_mutation_in_last_release_before_first_evaluation(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)

    def release(item: HeldSeed, phase: str) -> FinalRole:
        role = adapter.release_confirmation_final(item, phase)
        if item is trained.held[-1] and phase == "b":
            trained.held[0].models_after_a["ordinary_pc"].bias_output[0, 0] += 1
        return role

    with pytest.raises(ValueError, match="checkpoint"):
        _run_fixture(trained, identity, release=release, evaluate=_forbid)


@pytest.mark.parametrize("rehash", [False, True])
def test_should_recheck_earlier_final_content_after_last_release_before_evaluation(
    rehash: bool,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    first: list[FinalRole] = []

    def release(item: HeldSeed, phase: str) -> FinalRole:
        role = adapter.release_confirmation_final(item, phase)
        if not first:
            first.append(role)
        if item is trained.held[-1] and phase == "b":
            first[0].input[-1, 0] += 1
            if rehash:
                object.__setattr__(first[0], "sha256", final_content_digest(first[0]))
        return role

    with pytest.raises(ValueError, match="digest|content changed"):
        _run_fixture(trained, identity, release=release, evaluate=_forbid)


def test_should_refuse_different_final_values_for_shared_gating_replay_source(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)

    def release(item: HeldSeed, phase: str) -> FinalRole:
        role = adapter.release_confirmation_final(item, phase)
        if item.facts.family == "replay":
            role.input[-1, 0] += 1
            role = replace(role, sha256=final_content_digest(role))
        return role

    with pytest.raises(ValueError, match="shared source"):
        _run_fixture(trained, identity, release=release, evaluate=_forbid)


@pytest.mark.parametrize(
    "change",
    ["a_checkpoint", "selector_rng", "facts", "original_role", "released_data", "retained_count"],
)
def test_should_refuse_late_evaluation_drift_before_returning_a_result(
    change: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    calls = []

    def evaluate(model: Any, role: FinalRole) -> EndpointResult:
        result = adapter.evaluate_confirmation_final(model, role)
        calls.append(result)
        if len(calls) == 168:
            if change == "a_checkpoint":
                trained.held[0].models_after_a["ordinary_pc"].bias_output[0, 0] += 1
            elif change == "selector_rng":
                last = next(
                    m
                    for m in trained.held[-1].models_after_b.values()
                    if isinstance(m, ParentControlledCircadianNetwork)
                )
                last._parent_selection_rng.random()
            elif change == "facts":
                trained.facts.seed_results[-1].legacy_train_facts["unknown_late_fact"] = True
            elif change == "original_role":
                trained.held[-1].roles_a = replace(trained.held[-1].roles_a, final_released=True)
            elif change == "released_data":
                role.input[-1, 0] += 1
                object.__setattr__(role, "sha256", final_content_digest(role))
            else:
                assert calls[0].correct_count is not None
                object.__setattr__(calls[0], "correct_count", (calls[0].correct_count + 1) % 41)
        return result

    with pytest.raises(ValueError, match="checkpoint|artifact|role|content|retained endpoint"):
        _run_fixture(trained, identity, evaluate=evaluate)
    assert len(calls) == 168


@pytest.mark.parametrize("failed", ["first", "all"])
def test_should_attempt_every_endpoint_and_keep_complete_numerical_failures_null(
    failed: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    calls = []

    def evaluate(model: Any, role: FinalRole) -> EndpointResult:
        calls.append(role.phase)
        return (
            EndpointResult(None, EndpointFailure("nonfinite_predictions"))
            if failed == "all" or len(calls) == 1
            else EndpointResult(20)
        )

    result = _run_fixture(trained, identity, evaluate=evaluate)
    assert len(calls) == 168 and len(result.evaluations) == 168 and len(result.cells) == 56
    assert result.totals.example_count == 6720
    assert result.totals.failed_endpoints == (168 if failed == "all" else 1)
    assert result.totals.failed_cells == (56 if failed == "all" else 1)
    assert result.cells[0].accuracy is None and result.cells[0].roles is not None
    assert (
        result.cells[0].failure is not None
        and "a_after_a:nonfinite_predictions:none" in result.cells[0].failure
    )
    if failed == "first":
        assert (
            result.evaluations[1].result.correct_count == 20
            and result.evaluations[2].result.correct_count == 20
        )
    else:
        assert all(cell.accuracy is None and cell.failure is not None for cell in result.cells)


@pytest.mark.parametrize(
    "result",
    [
        EndpointResult(41),
        EndpointResult(True),
        EndpointResult(None),
        EndpointResult(None, EndpointFailure("unknown")),
        None,
    ],
)
def test_should_abort_invalid_endpoint_result_without_partial_success(
    result: Any,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    with pytest.raises(ValueError, match="endpoint result"):
        _run_fixture(trained, identity, evaluate=lambda model, role: result)


@pytest.mark.parametrize(
    "stage", ["before_final_release", "before_final_evaluation", "after_final_evaluation"]
)
def test_should_propagate_boundary_failures_at_each_global_barrier(
    stage: str,
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    calls = []

    def checkpoint(current: str) -> None:
        if current == stage:
            raise RuntimeError("fabricated boundary limit")

    def evaluate(model: Any, role: FinalRole) -> EndpointResult:
        calls.append(role.phase)
        return EndpointResult(20)

    with pytest.raises(RuntimeError, match="boundary limit"):
        _run_fixture(
            trained,
            identity,
            evaluate=evaluate,
            checkpoint=checkpoint,
        )
    assert len(calls) == (168 if stage == "after_final_evaluation" else 0)
    assert sum(source.reads_input for source in sources) == (
        0 if stage == "before_final_release" else 12
    )


def test_should_recheck_state_after_final_boundary_callback(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)

    def checkpoint(stage: str) -> None:
        if stage == "after_final_evaluation":
            trained.held[0].models_after_a["ordinary_pc"].bias_output[0, 0] += 1

    with pytest.raises(ValueError, match="checkpoint"):
        _run_fixture(trained, identity, checkpoint=checkpoint)


def test_should_refuse_partial_development_training_at_public_gate(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained, _, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    with pytest.raises(ValueError, match="inventory"):
        scoring.evaluate_confirmation(
            trained, fixed_scoring_manifest(), _forbid, _forbid, lambda stage: None
        )
    assert all(source.reads_input == source.reads_target == 0 for source in sources)


@dataclass(frozen=True)
class _MetadataModel:
    family: str
    seed: int
    arm: str
    checkpoint: str


def _whole_metadata() -> TrainedConfirmation:
    manifest = fixed_confirmation_manifest()
    held = []
    for family in manifest.families:
        for seed in family.seeds:
            roles = tuple(
                RoleFacts(
                    phase,
                    seed,
                    {},
                    {},
                    {
                        "final_test": tuple(
                            f"phase_{phase}/seed_{seed}/final/{i}" for i in range(40)
                        )
                    },
                    40,
                )
                for phase in ("a", "b")
            )
            placeholder: Any = roles
            row = FamilySeedFacts(family.name, seed, placeholder, {}, {}, {}, {}, ())
            models_a: Any = {
                arm: _MetadataModel(family.name, seed, arm, "a") for arm in family.arms
            }
            models_b: Any = {
                arm: _MetadataModel(family.name, seed, arm, "b") for arm in family.arms
            }
            omitted: Any = None
            held.append(HeldSeed(row, omitted, omitted, models_a, models_b))
    rows = tuple(held)
    return TrainedConfirmation(
        ConfirmationTrainingFacts(manifest, tuple(row.facts for row in rows)), rows
    )


@pytest.mark.parametrize("numerical_failure", [False, True])
def test_should_dispatch_complete_560_cell_scope_with_fabricated_roles_and_gate_spy_only(
    numerical_failure: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trained = _whole_metadata()
    manifest = fixed_scoring_manifest()
    identity = state.TrainingArtifactIdentity(
        manifest.training_bundles[0].result_sha256, manifest.training_bundles[0].result_bytes
    )
    proof = state.TrainingStateProof(identity, 60, 560, 1120, scoring_manifest_digest(manifest))
    gates: list[int] = []
    releases: list[tuple[str, int, str]] = []
    calls: list[tuple[str, int, str, str]] = []

    def verify(*args: Any) -> Any:
        gates.append(len(calls))
        return proof

    def release(item: HeldSeed, phase: str) -> FinalRole:
        releases.append((item.facts.family, item.facts.seed, phase))
        expected = item.facts.roles[0 if phase == "a" else 1]
        return _fabricated_role(phase, item.facts.seed, expected.sample_ids["final_test"])

    def evaluate(model: Any, role: FinalRole) -> EndpointResult:
        assert len(releases) == 120
        endpoint = {("a", "a"): "a_after_a", ("b", "a"): "a_after_b", ("b", "b"): "b_after_b"}[
            model.checkpoint, role.phase
        ]
        calls.append((model.family, model.seed, model.arm, endpoint))
        return (
            EndpointResult(None, EndpointFailure("nonfinite_predictions"))
            if numerical_failure and len(calls) == 1
            else EndpointResult(len(calls) % 41)
        )

    _seal_scoring(monkeypatch)
    monkeypatch.setattr(scoring, "verify_scoring_training_state", verify)
    result = scoring.evaluate_confirmation(trained, manifest, release, evaluate, lambda stage: None)
    expected = [
        (f.name, seed, arm, endpoint)
        for f in manifest.train_manifest.families
        for seed in f.seeds
        for arm in f.arms
        for endpoint in manifest.evaluation_endpoints
    ]
    assert calls == expected and gates == [0, 0, 1680]
    assert (
        len(result.cells) == 560
        and len(result.final_roles) == 60
        and len(result.evaluations) == 1680
    )
    assert result.totals.endpoint_calls == 1680 and result.totals.example_count == 67200
    assert result.totals.release_calls == result.totals.final_role_views == 120
    assert result.scoring_manifest_sha256 == scoring_manifest_digest(manifest)
    assert result.external_execution_verified is False
    analyzed = analyze_confirmation(result.cells, manifest.analysis_contract)
    assert analyzed.successful_cells + analyzed.failed_cells == 560
    assert sum(len(family.contrasts) for family in analyzed.families) == 58
    assert (
        sum(
            len(pair.metrics[0].summary.observations)
            for family in analyzed.families
            for pair in family.contrasts
        )
        == 580
    )
    assert analyzed.primary_statement_count == 116 and analyzed.failed_cells == int(
        numerical_failure
    )


def test_should_refuse_whole_unbound_metadata_without_the_explicit_gate_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seal_scoring(monkeypatch)
    with pytest.raises(ValueError, match="training artifact"):
        scoring.evaluate_confirmation(
            _whole_metadata(), fixed_scoring_manifest(), _forbid, _forbid, lambda stage: None
        )
