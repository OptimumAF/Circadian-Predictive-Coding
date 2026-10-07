"""Observe genuine development models on fabricated final fields only."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from typing import Any
from types import MappingProxyType

import pytest
import numpy as np

from test_continual_confirmation_scoring_state import development_training as development_training
from test_continual_confirmation_scoring import (
    _fabricated_role,
    _fixture_training,
    _run_fixture,
    _seal_scoring,
)
from test_continual_confirmation_final_adapter import _FabricatedSource
from src.app import continual_confirmation_scoring as scoring
from src.app.continual_confirmation_execution import json_value
from src.app.continual_confirmation_final_observation import (
    _expected_observation,
    verify_final_observation,
)
from src.app.continual_confirmation_manifest import fixed_confirmation_manifest
from src.app.continual_confirmation_scoring_manifest import (
    fixed_scoring_manifest,
    scoring_manifest_digest,
)
from src.app.continual_confirmation_scoring_state import (
    TrainingArtifactIdentity,
    TrainingStateProof,
    _verify_reproduced_state,
)
from src.app.continual_confirmation_state import HeldSeed
from src.app.continual_confirmation_training import ConfirmationTrainingFacts, TrainedConfirmation
from src.core.confirmation_final_roles import EndpointResult, final_content_digest
from src.core.backprop_mlp import BackpropMLP
from src.core.predictive_coding import PredictiveCodingNetwork
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.infra import continual_confirmation_final_runtime as runtime
from src.infra.continual_confirmation_final_runtime import FinalExecutionObserver
from src.infra.continual_confirmation_runtime import ConfirmationStopped, ExecutionObserver
from src.shared.process_memory import ProcessRssSampler


def test_should_observe_every_actual_source_read_and_prediction_without_changing_state(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    original_roles = tuple((item.roles_a, item.roles_b) for item in trained.held)
    _seal_scoring(monkeypatch)
    with ProcessRssSampler(read_rss_bytes=lambda: 100) as sampler:
        budget = ExecutionObserver(fixed_confirmation_manifest(), sampler, clock=lambda: 0.0)
        observer = FinalExecutionObserver(trained, budget.checkpoint)
        with observer.observe():
            scored = _run_fixture(
                trained,
                identity,
                release=observer.release,
                evaluate=observer.evaluate,
                checkpoint=observer.checkpoint,
            )
            observer.verify_result(scored)
            actual = observer.observations()
            assert actual == _expected_observation(scored, trained.facts.manifest)
            assert actual["release_attempts"] == actual["release_successes"] == 12
            assert actual["input_reads"] == actual["target_reads"] == 12
            assert actual["prediction_attempts"] == actual["prediction_returns"] == 168
            assert actual["prediction_examples"] == 6720
            assert actual["prediction_numerical_errors"] == 0
            assert actual["by_model_kind"] == {"backprop": 42, "pc": 45, "circadian": 81}
            assert json_value(asdict(scored))["external_execution_verified"] is False
    assert budget.executed_updates == 0 and budget.attempted_updates == 0
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    assert tuple((item.roles_a, item.roles_b) for item in trained.held) == original_roles
    assert all(
        item.roles_a is before[0] and item.roles_b is before[1]
        for item, before in zip(trained.held, original_roles, strict=True)
    )


def test_should_restore_owned_guards_and_methods_on_incomplete_context(
    development_training: tuple[Any, Any],
) -> None:
    trained, _, sources = _fixture_training(development_training)
    original_roles = tuple((item.roles_a, item.roles_b) for item in trained.held)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError, match="incomplete"):
        with observer.observe():
            pass
    assert all(source.reads_input == source.reads_target == 0 for source in sources)
    assert all(
        item.roles_a is before[0] and item.roles_b is before[1]
        for item, before in zip(trained.held, original_roles, strict=True)
    )


def test_should_recheck_content_after_the_last_budget_callback_before_result_verification(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, _ = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    armed = False
    observer: Any = None

    def budget() -> None:
        nonlocal armed
        if armed:
            armed = False
            role = next(iter(observer.views.values()))[0]
            role.target[0, 0] = 1.0 - role.target[0, 0]

    observer = FinalExecutionObserver(trained, budget)
    with pytest.raises(ValueError, match="content"):
        with observer.observe():
            scored = _run_fixture(
                trained,
                identity,
                release=observer.release,
                evaluate=observer.evaluate,
                checkpoint=observer.checkpoint,
            )
            armed = True
            with pytest.raises(ValueError, match="content") as caught:
                observer.verify_result(scored)
            # Propagate the expected gate failure instead of requesting normal
            # completion from this deliberately corrupted context.
            raise caught.value


def _release_all(observer: FinalExecutionObserver, trained: Any) -> None:
    for item in trained.held:
        for phase in ("a", "b"):
            observer.release(item, phase)


def _first(observer: FinalExecutionObserver) -> tuple[Any, Any]:
    request = observer.predictions[0]
    role = next(iter(observer.views.values()))[0]
    return request.model, role


def _methods() -> tuple[Any, ...]:
    return (
        BackpropMLP.predict_proba,
        PredictiveCodingNetwork.predict_proba,
        CircadianPredictiveCodingNetwork.predict_proba,
        BackpropMLP.train_epoch,
        PredictiveCodingNetwork.train_epoch,
        CircadianPredictiveCodingNetwork._run_training_step,
    )


@pytest.mark.parametrize(
    "kind",
    [
        "release_order",
        "release_extra",
        "predict_early",
        "predict_stray",
        "outer_input",
        "outer_labels",
        "source_input",
        "source_labels",
        "train_input",
        "train_labels",
        "optimizer_bp",
        "optimizer_pc",
        "optimizer_circadian",
    ],
)
def test_should_block_unplanned_calls_before_the_actual_operation_and_restore_every_hook(
    development_training: tuple[Any, Any], kind: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    roles = tuple((item.roles_a, item.roles_b) for item in trained.held)
    before = _methods()
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError):
        with observer.observe():
            item = trained.held[0]
            if kind == "release_order":
                observer.release(trained.held[-1], "b")
            elif kind == "release_extra":
                _release_all(observer, trained)
                observer.release(item, "a")
            elif kind == "predict_early":
                role = observer.release(item, "a")
                observer.evaluate(observer.predictions[0].model, role)
            elif kind == "predict_stray":
                observer.predictions[0].model.predict_proba(np.zeros((40, 2)))
            elif kind.startswith("outer"):
                getattr(
                    item.roles_a.outer_selection, "input" if kind == "outer_input" else "target"
                )
            elif kind.startswith("source"):
                getattr(
                    item.roles_a._source, "test_input" if kind == "source_input" else "test_target"
                )
            elif kind.startswith("train"):
                getattr(
                    item.roles_a._source, "train_input" if kind == "train_input" else "train_target"
                )
            else:
                model_kind = {
                    "optimizer_bp": BackpropMLP,
                    "optimizer_pc": PredictiveCodingNetwork,
                    "optimizer_circadian": CircadianPredictiveCodingNetwork,
                }[kind]
                model: Any = next(
                    request.model
                    for request in observer.predictions
                    if type(request.model) is model_kind
                )
                if kind == "optimizer_circadian":
                    model._run_training_step(None, None, 0.01, 1, 0.01, True)
                else:
                    model.train_epoch(None, None, 0.01)
    assert observer.prediction_attempts == 0 and observer.blocked_calls == 1
    assert _methods() == before
    expected_reads = 12 if kind == "release_extra" else 1 if kind == "predict_early" else 0
    assert sum(source.reads_input for source in sources) == expected_reads
    assert sum(source.reads_target for source in sources) == expected_reads
    assert all(
        item.roles_a is original[0] and item.roles_b is original[1]
        for item, original in zip(trained.held, roles, strict=True)
    )


@pytest.mark.parametrize("kind", ["rss", "wall"])
@pytest.mark.parametrize("stage", ["before_source", "after_source", "after_prediction"])
def test_should_enforce_original_live_limits_and_keep_completed_operations_before_a_stop(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str, stage: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    status: dict[str, Any] = {"rss": 100, "clock": 0.0}

    def stop() -> None:
        status["rss" if kind == "rss" else "clock"] = (
            512 * 1024 * 1024 + 1 if kind == "rss" else 600.0
        )

    if stage == "after_source":
        original_source = sources[0]

        class LimitedSource:
            @property
            def train_input(self) -> Any:
                raise AssertionError("unused fabricated training input")

            @property
            def train_target(self) -> Any:
                raise AssertionError("unused fabricated training labels")

            @property
            def test_input(self) -> Any:
                value = original_source.test_input
                stop()
                return value

            @property
            def test_target(self) -> Any:
                return original_source.test_target

        trained.held[0].roles_a = replace(trained.held[0].roles_a, _source=LimitedSource())
    first_type = type(trained.held[0].models_after_a[trained.facts.manifest.families[0].arms[0]])
    if stage == "after_prediction":
        original_prediction = first_type.predict_proba

        def prediction(model: Any, inputs: Any) -> Any:
            value = original_prediction(model, inputs)
            stop()
            return value

        monkeypatch.setattr(first_type, "predict_proba", prediction)
    before = _methods()
    roles = tuple((item.roles_a, item.roles_b) for item in trained.held)
    with ProcessRssSampler(read_rss_bytes=lambda: status["rss"]) as sampler:
        budget = ExecutionObserver(
            fixed_confirmation_manifest(), sampler, clock=lambda: status["clock"]
        )
        observer = FinalExecutionObserver(trained, budget.checkpoint)
        with pytest.raises(ConfirmationStopped, match="rss_limit|wall_limit"):
            with observer.observe():
                if stage == "before_source":
                    stop()
                    observer.release(trained.held[0], "a")
                elif stage == "after_source":
                    observer.release(trained.held[0], "a")
                else:
                    _release_all(observer, trained)
                    observer.evaluate(*_first(observer))
        facts = observer.observations()
    assert budget.executed_updates == 0 and budget.attempted_updates == 0
    assert (
        facts["prediction_attempts"]
        == facts["prediction_returns"]
        == (1 if stage == "after_prediction" else 0)
    )
    assert len(facts["prediction_events"]) == (1 if stage == "after_prediction" else 0)
    assert facts["input_reads"] == (
        0 if stage == "before_source" else 1 if stage == "after_source" else 12
    )
    assert facts["target_reads"] == (12 if stage == "after_prediction" else 0)
    assert _methods() == before
    assert all(
        item.roles_a is original[0] and item.roles_b is original[1]
        for item, original in zip(trained.held, roles, strict=True)
    )


@pytest.mark.parametrize("kind", ["nonfinite", "floating_point"])
@pytest.mark.parametrize("all_calls", [False, True])
def test_should_observe_each_numerical_failure_and_attempt_every_scheduled_endpoint(
    development_training: tuple[Any, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    all_calls: bool,
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    calls = 0
    for model_type in (BackpropMLP, PredictiveCodingNetwork, CircadianPredictiveCodingNetwork):
        original = model_type.predict_proba

        def failing(model: Any, inputs: Any, original: Any = original) -> Any:
            nonlocal calls
            calls += 1
            if all_calls or calls == 1:
                if kind == "floating_point":
                    raise FloatingPointError("fabricated numerical result")
                return np.full((len(inputs), 1), np.nan)
            return original(model, inputs)

        monkeypatch.setattr(model_type, "predict_proba", failing)
    before = _methods()
    observer = FinalExecutionObserver(trained, lambda: None)
    with observer.observe():
        scored = _run_fixture(
            trained,
            identity,
            release=observer.release,
            evaluate=observer.evaluate,
            checkpoint=observer.checkpoint,
        )
        observer.verify_result(scored)
        facts = observer.observations()
    failed = 168 if all_calls else 1
    assert calls == facts["prediction_attempts"] == 168 and facts["prediction_examples"] == 6720
    assert scored.totals.failed_endpoints == failed and scored.totals.failed_cells == (
        56 if all_calls else 1
    )
    assert facts["prediction_numerical_errors"] == (failed if kind == "floating_point" else 0)
    assert facts["prediction_returns"] == 168 - facts["prediction_numerical_errors"]
    assert _methods() == before
    assert all(source.reads_input == source.reads_target == 1 for source in sources)


@pytest.mark.parametrize("kind", ["shape", "dtype", "range", "runtime_error", "value_error"])
def test_should_propagate_unplanned_prediction_errors_without_numerical_substitution(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    first_type = type(trained.held[0].models_after_a[trained.facts.manifest.families[0].arms[0]])

    def bad(model: Any, inputs: Any) -> Any:
        if kind == "runtime_error":
            raise RuntimeError("fabricated prediction contract")
        if kind == "value_error":
            raise ValueError("fabricated prediction contract")
        return (
            np.zeros((40,))
            if kind == "shape"
            else np.zeros((40, 1), dtype=np.float32)
            if kind == "dtype"
            else np.full((40, 1), 1.1)
        )

    monkeypatch.setattr(first_type, "predict_proba", bad)
    before = _methods()
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises((ValueError, RuntimeError)):
        with observer.observe():
            _release_all(observer, trained)
            observer.evaluate(*_first(observer))
    facts = observer.observations()
    assert facts["prediction_attempts"] == 1 and facts["unexpected_prediction_errors"] == 1
    assert facts["prediction_numerical_errors"] == 0 and not facts["prediction_events"]
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


@pytest.mark.parametrize(
    "kind", ["no_prediction", "changed_count", "extra_prediction", "input_copy"]
)
def test_should_reject_forged_adapter_calls_or_results_against_actual_probabilities(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    original = runtime.evaluate_confirmation_final

    def forged(model: Any, role: Any) -> EndpointResult:
        if kind == "no_prediction":
            return EndpointResult(20)
        if kind == "input_copy":
            model.predict_proba(role.input.copy())
            return EndpointResult(20)
        result = original(model, role)
        if kind == "extra_prediction":
            model.predict_proba(role.input)
            return result
        assert result.correct_count is not None
        return EndpointResult((result.correct_count + 1) % 41)

    monkeypatch.setattr(runtime, "evaluate_confirmation_final", forged)
    before = _methods()
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError):
        with observer.observe():
            _release_all(observer, trained)
            observer.evaluate(*_first(observer))
    assert observer.prediction_attempts == (
        1 if kind in {"changed_count", "extra_prediction"} else 0
    )
    assert _methods() == before and all(
        source.reads_input == source.reads_target == 1 for source in sources
    )


@pytest.mark.parametrize(
    "kind", ["array_copy", "substituted_array", "duplicate_field", "reversed_fields"]
)
def test_should_bind_released_views_to_the_single_observed_source_field_objects(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    original = runtime.release_confirmation_final

    def forged(item: Any, phase: str) -> Any:
        if kind == "duplicate_field":
            item.roles_a._source.test_input
        elif kind == "reversed_fields":
            item.roles_a._source.test_target
        role = original(item, phase)
        changed = replace(
            role, input=role.input.copy() + (1.0 if kind == "substituted_array" else 0.0)
        )
        return replace(changed, sha256=final_content_digest(changed))

    monkeypatch.setattr(runtime, "release_confirmation_final", forged)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError):
        with observer.observe():
            observer.release(trained.held[0], "a")
    assert observer.prediction_attempts == 0
    assert sum(source.reads_input for source in sources) == (0 if kind == "reversed_fields" else 1)
    assert sum(source.reads_target for source in sources) == (
        1 if kind in {"array_copy", "substituted_array"} else 0
    )


@pytest.mark.parametrize(
    "kind",
    [
        "input",
        "labels",
        "resealed_labels",
        "source_cache",
        "original_role",
        "model_binding",
        "observed_count",
        "app_count",
        "counter_total",
    ],
)
def test_should_recheck_every_retained_view_and_actual_endpoint_link_after_serialization(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError):
        with observer.observe():
            scored = _run_fixture(
                trained,
                identity,
                release=observer.release,
                evaluate=observer.evaluate,
                checkpoint=observer.checkpoint,
            )
            observer.verify_result(scored)
            json.dumps(asdict(scored), sort_keys=True, allow_nan=False)
            role = next(iter(observer.views.values()))[0]
            if kind == "input":
                role.input[0, 0] += 1.0
            elif kind in {"labels", "resealed_labels"}:
                role.target[0, 0] = 1.0 - role.target[0, 0]
                if kind == "resealed_labels":
                    object.__setattr__(role, "sha256", final_content_digest(role))
            elif kind == "source_cache":
                observer.bindings[0].observed.values["test_input"] = role.input.copy()
            elif kind == "original_role":
                trained.held[0].roles_a = replace(trained.held[0].roles_a, seed=999)
            elif kind == "model_binding":
                trained.held[-1].models_after_b[trained.facts.manifest.families[-1].arms[-1]] = (
                    observer.predictions[0].model
                )
            elif kind == "observed_count":
                before = observer.prediction_events[-1]["result"]["correct_count"]
                observer.prediction_events[-1]["result"]["correct_count"] = (before + 1) % 41
            elif kind == "app_count":
                before = scored.evaluations[-1].result.correct_count
                assert before is not None and scored.cells[-1].accuracy is not None
                count = (before + 1) % 41
                object.__setattr__(scored.evaluations[-1].result, "correct_count", count)
                object.__setattr__(scored.cells[-1].accuracy, "b_after_b", count / 40)
            else:
                observer.prediction_examples += 1
            observer.verify_result(scored)
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    if kind == "original_role":
        assert trained.held[0].roles_a.seed == 999
        assert trained.held[0].roles_a._source is sources[0]


def test_should_require_the_separate_global_live_gate_for_late_parameter_drift(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError, match="checkpoint"):
        with observer.observe():
            scored = _run_fixture(
                trained,
                identity,
                release=observer.release,
                evaluate=observer.evaluate,
                checkpoint=observer.checkpoint,
            )
            observer.verify_result(scored)
            json.dumps(asdict(scored), sort_keys=True, allow_nan=False)
            trained.held[-1].models_after_b[
                trained.facts.manifest.families[-1].arms[-1]
            ].bias_output[0, 0] += 1.0
            _verify_reproduced_state(trained, trained.facts.manifest, identity)
    assert all(source.reads_input == source.reads_target == 1 for source in sources)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_should_restore_all_hooks_sources_and_outer_guards_after_cancellation(
    development_training: tuple[Any, Any], error: type[BaseException]
) -> None:
    trained, _, sources = _fixture_training(development_training)
    before = _methods()
    roles = tuple((item.roles_a, item.roles_b) for item in trained.held)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(error):
        with observer.observe():
            observer.release(trained.held[0], "a")
            raise error("fabricated interrupted worker")
    assert _methods() == before and observer.active is False
    assert all(
        item.roles_a is original[0] and item.roles_b is original[1]
        for item, original in zip(trained.held, roles, strict=True)
    )
    assert (
        sum(source.reads_input for source in sources)
        == sum(source.reads_target for source in sources)
        == 1
    )


def _whole_development_models(
    development_training: tuple[Any, Any],
) -> tuple[TrainedConfirmation, list[Any]]:
    # Full reserved-seed TAGS only. Reuse genuine already-trained development
    # model objects and fabricated source fields; no reserved source/model.
    original, _ = development_training
    templates = {row.facts.family: row for row in original.held}
    manifest = fixed_confirmation_manifest()
    held, sources = [], []
    for family in manifest.families:
        template = templates[family.name]
        for seed in family.seeds:
            role_objects, role_facts = [], []
            for phase, old_role, old_facts in zip(
                ("a", "b"), (template.roles_a, template.roles_b), template.facts.roles, strict=True
            ):
                ids = {
                    name: tuple(
                        value.replace(f"seed_{template.facts.seed}/", f"seed_{seed}/")
                        for value in values
                    )
                    for name, values in old_facts.sample_ids.items()
                }
                final = _fabricated_role(phase, seed, ids["final_test"])
                source = _FabricatedSource(final.input, final.target)
                sources.append(source)
                role_objects.append(
                    replace(old_role, seed=seed, sample_ids=MappingProxyType(ids), _source=source)
                )
                role_facts.append(replace(old_facts, seed=seed, sample_ids=ids))
            facts = replace(template.facts, seed=seed, roles=tuple(role_facts))
            held.append(
                HeldSeed(
                    facts,
                    role_objects[0],
                    role_objects[1],
                    dict(template.models_after_a),
                    dict(template.models_after_b),
                )
            )
    rows = tuple(held)
    return TrainedConfirmation(
        ConfirmationTrainingFacts(manifest, tuple(item.facts for item in rows)), rows
    ), sources


@pytest.mark.parametrize("numerical_failure", [False, True])
def test_should_observe_the_complete_fake_matrix_using_development_models_and_a_global_gate_spy_only(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, numerical_failure: bool
) -> None:
    trained, sources = _whole_development_models(development_training)
    manifest = fixed_scoring_manifest()
    reference = manifest.training_bundles[0]
    proof = TrainingStateProof(
        TrainingArtifactIdentity(reference.result_sha256, reference.result_bytes),
        60,
        560,
        1120,
        scoring_manifest_digest(manifest),
    )
    _seal_scoring(monkeypatch)
    original, identity = development_training
    observer = FinalExecutionObserver(trained, lambda: None)
    calls = []

    def spy(*args: Any) -> Any:
        calls.append(observer.prediction_attempts)
        return proof

    monkeypatch.setattr(scoring, "verify_scoring_training_state", spy)
    if numerical_failure:
        first_type = type(observer.predictions[0].model)
        method = first_type.predict_proba
        first = True

        def fail_once(model: Any, inputs: Any) -> Any:
            nonlocal first
            if first:
                first = False
                raise FloatingPointError("fabricated first whole-metadata endpoint")
            return method(model, inputs)

        monkeypatch.setattr(first_type, "predict_proba", fail_once)
    before = _methods()
    with observer.observe():
        scored = scoring.evaluate_confirmation(
            trained, manifest, observer.release, observer.evaluate, observer.checkpoint
        )
        observer.verify_result(scored)
        observed = observer.observations()
        assert verify_final_observation(observed, json_value(asdict(scored)), manifest) == observed
    assert observed["release_attempts"] == observed["release_successes"] == 120
    assert observed["input_reads"] == observed["target_reads"] == 120
    assert observed["prediction_attempts"] == 1680 and observed["prediction_examples"] == 67200
    assert observed["prediction_numerical_errors"] == int(numerical_failure)
    assert observed["by_model_kind"] == {"backprop": 420, "pc": 450, "circadian": 810}
    assert calls == [0, 0, 1680] and len(scored.cells) == 560
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    assert (
        observed["source_provenance_verified"] is False
        and scored.external_execution_verified is False
    )
    assert _methods() == before
    # Reused development models retain every full state/selector/clock/RNG;
    # this proves no prediction mutation, not reserved training reproduction.
    _verify_reproduced_state(original, original.facts.manifest, identity)


def test_should_refuse_whole_fake_training_without_the_explicit_global_gate_spy(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, sources = _whole_development_models(development_training)
    _seal_scoring(monkeypatch)
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(ValueError, match="artifact"):
        with observer.observe():
            scoring.evaluate_confirmation(
                trained,
                fixed_scoring_manifest(),
                observer.release,
                observer.evaluate,
                observer.checkpoint,
            )
    assert all(source.reads_input == source.reads_target == 0 for source in sources)


@pytest.mark.parametrize("kind", ["rss", "wall"])
def test_should_check_live_budgets_after_serialization_before_accepting_a_complete_observation(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    status: dict[str, Any] = {"rss": 100, "clock": 0.0}
    before = _methods()
    with ProcessRssSampler(read_rss_bytes=lambda: status["rss"]) as sampler:
        budget = ExecutionObserver(
            fixed_confirmation_manifest(), sampler, clock=lambda: status["clock"]
        )
        observer = FinalExecutionObserver(trained, budget.checkpoint)
        with pytest.raises(ConfirmationStopped, match="rss_limit|wall_limit"):
            with observer.observe():
                scored = _run_fixture(
                    trained,
                    identity,
                    release=observer.release,
                    evaluate=observer.evaluate,
                    checkpoint=observer.checkpoint,
                )
                observer.verify_result(scored)
                json.dumps(asdict(scored), sort_keys=True, allow_nan=False)
                status["rss" if kind == "rss" else "clock"] = (
                    512 * 1024 * 1024 + 1 if kind == "rss" else 600.0
                )
                observer.verify_result(scored)
    assert observer.prediction_attempts == observer.prediction_returns == 168
    assert observer.prediction_examples == 6720 and budget.executed_updates == 0
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    assert _methods() == before


def test_should_reject_a_caught_forbidden_operation_even_when_all_planned_calls_later_succeed(
    development_training: tuple[Any, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    trained, identity, sources = _fixture_training(development_training)
    _seal_scoring(monkeypatch)
    observer = FinalExecutionObserver(trained, lambda: None)
    before = _methods()
    with pytest.raises(ValueError, match="blocked_calls"):
        with observer.observe():
            with pytest.raises(ValueError, match="outer"):
                trained.held[0].roles_a.outer_selection.input
            scored = _run_fixture(
                trained,
                identity,
                release=observer.release,
                evaluate=observer.evaluate,
                checkpoint=observer.checkpoint,
            )
            observer.verify_result(scored)
    assert observer.blocked_calls == 1 and observer.prediction_attempts == 168
    assert all(source.reads_input == source.reads_target == 1 for source in sources)
    assert _methods() == before


@pytest.mark.parametrize("field", ["test_input", "test_target"])
def test_should_keep_attempts_and_restore_every_hook_on_source_read_failure(
    development_training: tuple[Any, Any], field: str
) -> None:
    trained, _, sources = _fixture_training(development_training)
    original_source = sources[0]

    class FailedSource:
        @property
        def train_input(self) -> Any:
            raise AssertionError("unused training source")

        @property
        def train_target(self) -> Any:
            raise AssertionError("unused training source")

        @property
        def test_input(self) -> Any:
            if field == "test_input":
                raise FloatingPointError("fabricated input-source failure")
            return original_source.test_input

        @property
        def test_target(self) -> Any:
            raise FloatingPointError("fabricated label-source failure")

    trained.held[0].roles_a = replace(trained.held[0].roles_a, _source=FailedSource())
    role = trained.held[0].roles_a
    before = _methods()
    observer = FinalExecutionObserver(trained, lambda: None)
    with pytest.raises(FloatingPointError, match="source failure"):
        with observer.observe():
            observer.release(trained.held[0], "a")
    facts = observer.observations()
    assert facts["release_attempts"] == 1 and facts["release_successes"] == 0
    assert facts["input_attempts"] == 1 and facts["target_attempts"] == int(field == "test_target")
    assert facts["input_reads"] == int(field == "test_target") and facts["target_reads"] == 0
    assert facts["prediction_numerical_errors"] == facts["prediction_attempts"] == 0
    assert trained.held[0].roles_a is role and _methods() == before
