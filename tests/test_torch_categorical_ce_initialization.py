"""Small software fixtures; no genuine source12/task/adaptation experiment."""

from copy import deepcopy
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest

from src.adapters import torch_categorical_ce_initialization as init
from src.adapters import torch_categorical_ce_readout as ce
from src.adapters import torch_feature_learner as native
from src.adapters.torch_array_weighted import _ownership, _rng
from src.adapters.torch_decision import TorchDecisionPredictor, _same_values
from src.app.coupled_attribution import AttributionBatch
from src.app.decision_reference_evaluation import DecisionCase
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate, DecisionRequest
from src.shared.torch_runtime import require_torch


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected source/model work")


@pytest.fixture(scope="module")
def trained():
    catalog = tuple(DecisionCandidate(n, n) for n in ("a", "b", "c"))
    spec = DecisionFeatureSpec(("alpha", "beta"), catalog)
    cases = tuple(
        DecisionCase(DecisionRequest(str(i), q, catalog), t, "fixture", "train")
        for i, (q, t) in enumerate((("alpha", "a"), ("beta", "b"), ("alpha beta", "c")))
    )
    source = native.TorchFeatureLearner(native.TorchFeaturePolicy("bp", 2, 3, 17, num_classes=3))
    batch = ce.build_decision_training_batch(cases, spec, "fixture", allow_subsets=True)
    source.train_batch(batch.features, batch.targets)
    fit = ce.fit_categorical_ce_readout(source, cases, spec)
    return source, fit, batch, cases


def child(state):
    learner = native.TorchFeatureLearner(state.policy)
    learner.restore_state(state)
    return learner


def test_derivation_preserves_full_source_sgd_rng_and_has_no_model_work(trained, monkeypatch):
    source, fit, _, _ = trained
    before, binding, rng = source.snapshot_state(), _ownership(source), _rng(require_torch())
    monkeypatch.setattr(source, "snapshot_state", forbidden)
    monkeypatch.setattr(source, "fork", forbidden)
    monkeypatch.setattr(source, "train_batch", forbidden)
    monkeypatch.setattr(native, "_components", forbidden)
    result = init.initialize_categorical_ce_probe(source, fit)
    assert _same_values(result.fit, fit)
    assert result.fit is not fit and result.fit.work is not fit.work
    assert result.state is not result.fit.source
    assert _same_values(result.state.optimizer_state, before.optimizer_state)
    assert _same_values(result.state.gradients, before.gradients)
    assert result.state.policy == before.policy
    assert (result.state.completed_updates, result.state.training_examples) == (1, 3)
    for n in ("weight_feature_hidden", "bias_hidden"):
        assert _same_values(result.state.head_state[n], before.head_state[n])
    assert _ownership(source) == binding and _same_values(rng, _rng(require_torch()))
    assert _same_values(native.TorchFeatureLearner.snapshot_state(source), before)
    record = init.ce_initialization_record(result)
    assert record["fit_work"] == fit.work and record["copied_output_values"] == 12
    assert record["additional_native_updates"] == record["additional_native_rows"] == 0
    assert record["additional_model_constructions"] == 0


def test_native_typed_predictions_equal_ce_reader_without_rescaling(trained):
    source, fit, batch, cases = trained
    result = init.initialize_categorical_ce_probe(source, fit)
    learner = child(result.state)
    assert np.array_equal(learner.predict_logits(batch.features), np.array(fit.training_logits))
    direct = TorchDecisionPredictor(fit.spec, learner.snapshot_state())
    control = ce.CategoricalCEReadoutPredictor(fit)
    for case in cases:
        assert direct(case.request) == control(case.request)


def test_private_update_continues_original_momentum_and_keeps_source_untouched(
    trained, monkeypatch
):
    source, fit, batch, _ = trained
    before, rng = source.snapshot_state(), _rng(require_torch())
    result = init.initialize_categorical_ce_probe(source, fit)
    private, reference = child(result.state), child(before)
    torch = require_torch()
    with torch.no_grad():
        coefficients = torch.tensor(fit.coefficients, dtype=torch.float32, device="cpu")
        reference._head.weight_hidden_output.copy_(coefficients[:-1])
        reference._head.bias_output.copy_(coefficients[-1:])
    assert _same_values(private.snapshot_state(), reference.snapshot_state())
    monkeypatch.setattr(source, "fork", forbidden)
    monkeypatch.setattr(source, "train_batch", forbidden)
    private.train_batch(batch.features, batch.targets)
    reference.train_batch(batch.features, batch.targets)
    assert _same_values(private.snapshot_state(), reference.snapshot_state())
    assert private.completed_updates == 2 and private.training_examples == 6
    assert _same_values(source.snapshot_state(), before) and _same_values(rng, _rng(torch))
    assert not _same_values(private.snapshot_state(), result.state)
    assert _same_values(init.copy_ce_initialization(result).state, result.state)


def test_existing_fresh_probe_trains_only_detached_ce_descendant(trained):
    source, fit, batch, _ = trained
    before = source.snapshot_state()
    result = init.initialize_categorical_ce_probe(source, fit)
    seed = child(result.state)
    training = (
        AttributionBatch(
            "probe_train",
            "f",
            ("f0", "f1", "f2"),
            batch.features.copy(),
            batch.targets.copy(),
            num_classes=3,
        ),
    )
    evaluation = (
        AttributionBatch(
            "probe_test",
            "f",
            ("e0", "e1", "e2"),
            batch.features.copy(),
            batch.targets.copy(),
            num_classes=3,
        ),
    )
    session = ToyBudgetSession(ToyExecutionBudget(max_training_updates=1), lambda: 0.0)
    row = run_fresh_task_probe(
        seed,
        result.state,
        training,
        evaluation,
        budget=session,
        checkpoint_updates=(0, 1),
        timer=lambda: 0.0,
    )
    assert row["status"] == "complete", row
    assert session.updates_completed == 1
    assert _same_values(seed.snapshot_state(), result.state)
    assert _same_values(source.snapshot_state(), before)


@pytest.mark.parametrize(
    "fault", ["hidden", "momentum", "gradient", "counter", "coefficient", "work"]
)
def test_inconsistent_initialization_is_refused_without_models(trained, monkeypatch, fault):
    source, fit, _, _ = trained
    result = init.initialize_categorical_ce_probe(source, fit)
    torch = require_torch()
    with torch.no_grad():
        if fault == "hidden":
            result.state.head_state["bias_hidden"].add_(1)
        elif fault == "momentum":
            assert result.state.optimizer_state is not None
            result.state.optimizer_state["state"][0]["momentum_buffer"].add_(1)
        elif fault == "gradient":
            assert result.state.gradients is not None
            result.state.gradients["bias_output"].add_(1)
        elif fault == "counter":
            object.__setattr__(result.state, "training_examples", 4)
        elif fault == "coefficient":
            object.__setattr__(result.fit, "coefficients", tuple((0.0, 0.0, 0.0) for _ in range(4)))
        else:
            result.fit.work["step_returns"] = 0
    monkeypatch.setattr(native, "_components", forbidden)
    with pytest.raises(ValueError):
        init.copy_ce_initialization(result)


def test_spoofed_public_snapshot_cannot_admit_different_actual_source(trained, monkeypatch):
    source, fit, _, _ = trained
    other = child(source.snapshot_state())
    with require_torch().no_grad():
        other._head.bias_hidden.add_(1)
    before = native.TorchFeatureLearner.snapshot_state(other)
    monkeypatch.setattr(other, "snapshot_state", lambda: deepcopy(fit.source))
    with pytest.raises(ValueError, match="actual source"):
        init.initialize_categorical_ce_probe(other, fit)
    assert _same_values(native.TorchFeatureLearner.snapshot_state(other), before)


@pytest.mark.parametrize("fault", ["source", "fit", "copy", "replaced_fit"])
def test_callback_or_copy_mutations_are_refused_and_preserved(trained, monkeypatch, fault):
    source, fit, _, _ = trained
    other, external = child(source.snapshot_state()), deepcopy(fit)
    before = other.snapshot_state()
    calls: list[int] = []

    def mutate():
        if calls:
            return
        calls.append(1)
        if fault in ("source", "copy"):
            with require_torch().no_grad():
                other._head.bias_hidden.add_(1)
        elif fault == "fit":
            external.work["step_returns"] = 0
        else:
            object.__setattr__(external, "work", deepcopy(external.work))

    if fault == "copy":
        original = init.deepcopy

        def changed(value):
            mutate()
            return original(value)

        monkeypatch.setattr(init, "deepcopy", changed)
        boundary = None
    else:
        boundary = mutate
    with pytest.raises(ValueError, match="changed"):
        init.initialize_categorical_ce_probe(other, external, check_boundary=boundary)
    if fault in ("fit", "replaced_fit"):
        assert external.work["step_returns"] == (0 if fault == "fit" else 1)
        assert _same_values(other.snapshot_state(), before)
    else:
        assert not _same_values(other.snapshot_state(), before)


def test_foreign_receipt_fails_before_native_work(monkeypatch):
    monkeypatch.setattr(native, "_components", forbidden)
    with pytest.raises(ValueError, match="exact CE initialization"):
        init.copy_ce_initialization(cast(init.CategoricalCEInitialization, None))


@pytest.mark.parametrize(
    "mode", ["default", "ce", "replaced_fit_work", "copy_replaced_fit_work", "foreign_source"]
)
def test_scripted_probe_handoff_keeps_original_checkpoint_separate(trained, monkeypatch, mode):
    """Scripted counters/clock/data seams ONLY; never native source12 evidence."""
    from scripts import ticket_native_probe as probe

    source, fit, _, _ = trained
    # Simulate the script's fixed checkpoint admission without a new real study.
    state = replace(fit.source, completed_updates=12, training_examples=72)
    owned_fit = replace(fit, source=deepcopy(state))
    initialization = init.CategoricalCEInitialization(
        init.INITIALIZATION, init.CONTINUATION, owned_fit, init._derived_state(owned_fit), 12
    )
    if mode == "foreign_source":
        initialization.fit.source.head_state["bias_hidden"].add_(1)
        initialization = replace(initialization, state=init._derived_state(initialization.fit))
    memory = object()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=12), lambda: 0.0)
    budget.updates_completed = 12
    budget.process_rss_sampler = cast(Any, memory)
    monkeypatch.setattr(ToyBudgetSession, "before_final", lambda s: None)
    monkeypatch.setattr(ToyBudgetSession, "attach_memory", lambda s, m: None)
    monkeypatch.setattr(probe.native, "TorchFeaturePolicy", lambda *a, **kw: state.policy)
    monkeypatch.setattr(probe, "CONTEXT_FEATURES", fit.spec.vocabulary)
    monkeypatch.setattr(probe, "CATALOG", fit.spec.catalog)
    seen: list[str] = []
    work_owner = initialization.fit.work
    if mode == "copy_replaced_fit_work":
        original_copy = probe.deepcopy

        def changed_copy(value):
            object.__setattr__(initialization.fit, "work", deepcopy(initialization.fit.work))
            return original_copy(value)

        monkeypatch.setattr(probe, "deepcopy", changed_copy)

    def cohorts(*args, **kwargs):
        seen.append("cohorts")
        if mode == "replaced_fit_work":
            object.__setattr__(initialization.fit, "work", deepcopy(initialization.fit.work))
        return ((), (), (), ())

    def enter(seed, initial_state, *args, **kwargs):
        seen.append("probe")
        expected = state if mode == "default" else initialization.state
        assert _same_values(initial_state, expected)
        assert _same_values(seed.snapshot_state(), expected)
        assert seed is not source
        raise RuntimeError("scripted stop at paid-probe entry")

    monkeypatch.setattr(probe, "build_head_cohorts", cohorts)
    monkeypatch.setattr(probe, "run_fresh_task_probe", enter)
    monkeypatch.setattr(probe, "_read_tasks", lambda *a, **kw: None)
    report = dict(
        definition="native_context_return_ticket_workflow_v1",
        method="bp",
        reserved_probe_requested=True,
        reserved_probe_source_training=dict(completed_updates=12, applied_update_count=12),
        checkpoints=[dict(state=probe.native.snapshot_record(state))],
    )
    before = deepcopy(report)
    row = probe.run_owned_probe(
        state,
        fit.spec,
        report,
        budget,
        check_boundary=lambda: None,
        memory=memory,
        ce_initialization=None if mode == "default" else initialization,
    )
    assert row["complete"] is False
    assert report == before and budget.updates_completed == 12
    assert row["preserved_training_work"] == dict(
        attempted_updates=0, attempted_rows=0, returned_updates=0
    )
    if mode in ("default", "ce"):
        assert seen == ["cohorts", "probe"] and "scripted stop" in row["error"]
        if mode == "ce":
            assert row["initialization"]["source_native_updates"] == 12
            assert row["initialization"]["fit_work"] == fit.work
            assert row["initialization"]["additional_native_updates"] == 0
        else:
            assert "initialization" not in row
    else:
        assert seen == ([] if mode in ("foreign_source", "copy_replaced_fit_work") else ["cohorts"])
        assert "changed" in row["error"] or "differs" in row["error"]
        if mode in ("replaced_fit_work", "copy_replaced_fit_work"):
            assert initialization.fit.work is not work_owner
