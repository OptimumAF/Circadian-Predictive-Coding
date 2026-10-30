"""Typed literal source admission; no training/probability/teacher execution."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from scripts import context_catalog_replay as replay
from scripts.context_catalog_control import ContextCatalogControl, context_catalog
from src.adapters.context_linear_fit_learner import (
    ContextLinearFitLearner,
    CONTEXT_LINEAR_RESIDUAL_ID,
)
from src.adapters.context_linear_serving import snapshot_digest
from src.adapters.context_perceptron import ContextPerceptronLearner
from src.app.counterfactual_controller import CatalogCounterfactualRequest
from src.app.coupled_attribution import _same_value
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy, LinearFitSnapshot
from src.core.context_linear_state import ContextLinearFitPolicy, ContextLinearFitSnapshot
from src.core.context_perceptron import ContextPerceptronPolicy


def literal_state():
    # Literal complete metadata only, never claimed as observed qualification.
    model = CategoricalLinearFit(7, 6, 6, ((0.0,) * 6,) * 8, 0, (0.0,) * 6, 0.0)
    policy = LinearFitPolicy(7, 6)
    return ContextLinearFitSnapshot(
        ContextLinearFitPolicy(9, 6, (4, 5)),
        (LinearFitSnapshot(policy, model, 8, 48), LinearFitSnapshot(policy, model, 4, 24)),
        12,
        72,
    )


def request(state, version="context-ls-12"):
    return CatalogCounterfactualRequest(
        version, snapshot_digest(state), "mistakes", "replay6-wake1", 0, catalog=context_catalog()
    )


def test_should_bind_complete_ls_source_without_a_fabricated_mistake_counter():
    state = literal_state()
    assert not hasattr(state, "mistake_updates")
    owner = ContextCatalogControl(request(state), base_snapshot=state, linear_fit=True)
    owner.require_source(state, deepcopy(state), digest=snapshot_digest, same_state=_same_value)
    assert owner.report["source_binding_verified"]
    assert owner.report["learner_binding"]["family"] == "context_linear_fit"


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_should_refuse_nonboolean_family_before_record_access(flag):
    with pytest.raises(ValueError, match="Boolean"):
        ContextCatalogControl(None, linear_fit=flag)
    with pytest.raises(ValueError, match="Boolean"):
        replay.compose_context_replay(None, None, None, None, None, linear_fit=flag)


@pytest.mark.parametrize("ls,version", [(False, "context-ls-12"), (True, "qualified-12")])
def test_should_refuse_cross_family_version_before_binding(ls, version):
    state = literal_state()
    with pytest.raises(ValueError, match="family"):
        ContextCatalogControl(request(state, version), base_snapshot=state, linear_fit=ls)


@pytest.mark.parametrize(
    "fault", ["swapped_heads", "wrong_policy", "changed_source", "changed_base", "family_pin"]
)
def test_should_refuse_changed_complete_source_history_before_work(fault):
    state = literal_state()
    if fault == "swapped_heads":
        state = replace(state, heads=tuple(reversed(state.heads)))
    elif fault == "wrong_policy":
        state = replace(state, policy=ContextLinearFitPolicy(9, 6, (0, 1)))
    owner = ContextCatalogControl(request(state), base_snapshot=state, linear_fit=True)
    base = deepcopy(state)
    if fault == "changed_source":
        state = replace(
            state,
            heads=(replace(state.heads[0], completed_updates=9), state.heads[1]),
            completed_updates=13,
        )
    elif fault == "changed_base":
        base = replace(base, heads=tuple(reversed(base.heads)))
    elif fault == "family_pin":
        owner._linear_fit = False
    with pytest.raises(ValueError):
        owner.require_source(state, base, digest=snapshot_digest, same_state=_same_value)
    assert owner.report["source_binding_verified"] is False


def test_should_refuse_digest_callback_source_mutation():
    state = literal_state()
    owner = ContextCatalogControl(request(state), base_snapshot=state, linear_fit=True)

    def changed(value):
        digest = snapshot_digest(value)
        object.__setattr__(value, "completed_updates", 13)
        return digest

    with pytest.raises(ValueError):
        owner.require_source(state, deepcopy(state), digest=changed, same_state=_same_value)
    assert not owner.report["source_binding_verified"]


@pytest.mark.parametrize("linear", [False, True])
def test_should_reuse_bank_and_executor_with_family_native_diagnostic(linear, monkeypatch):
    calls = []
    bank = object()
    monkeypatch.setattr(replay, "compile_context_bank", lambda *a: bank)

    def executor(*args, **kwargs):
        calls.append((args, kwargs))
        return "bound"

    monkeypatch.setattr(replay, "CategoricalReplayAction", executor)
    learner = (
        ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))
        if linear
        else ContextPerceptronLearner(ContextPerceptronPolicy(9, 6, (4, 5)))
    )
    assert (
        replay.compose_context_replay(None, learner, None, None, None, linear_fit=linear) == "bound"
    )
    args, kwargs = calls[0]
    assert args[3] is bank
    assert kwargs["states_equal"] is _same_value
    assert kwargs["definition"] == (
        CONTEXT_LINEAR_RESIDUAL_ID if linear else replay.PERCEPTRON_MISTAKES_DIAGNOSTIC
    )
    assert args[4].max_examples == 2 and args[4].max_bytes == 4096


@pytest.mark.parametrize(
    "linear,learner",
    [
        (True, ContextPerceptronLearner(ContextPerceptronPolicy(9, 6, (4, 5)))),
        (False, ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (4, 5)))),
        (True, ContextLinearFitLearner(ContextLinearFitPolicy(9, 6, (0, 1)))),
    ],
)
def test_should_refuse_wrong_replay_family_or_policy_before_teacher(linear, learner, monkeypatch):
    monkeypatch.setattr(replay, "compile_context_bank", lambda *a: pytest.fail("bank opened"))
    with pytest.raises(ValueError):
        replay.compose_context_replay(None, learner, None, None, None, linear_fit=linear)


@pytest.mark.parametrize("fault", ["source", "candidate", "version"])
def test_should_refuse_foreign_runtime_with_equal_aggregate_history_before_bank(fault, monkeypatch):
    from scripts import context_catalog_control as control

    state = literal_state()
    owner = ContextCatalogControl(request(state), base_snapshot=state, linear_fit=True)
    owner.require_source(state, deepcopy(state), digest=snapshot_digest, same_state=_same_value)
    changed_model = replace(state.heads[0].model, squared_error=1.0)
    changed = replace(state, heads=(replace(state.heads[0], model=changed_model), state.heads[1]))
    wake = SimpleNamespace(budget=object())
    monkeypatch.setattr(owner, "_wake", wake)
    monkeypatch.setattr(owner, "_wake_limits", wake.budget)
    monkeypatch.setattr(owner, "replay_budget", object())
    runtime = SimpleNamespace(
        _budget=wake,
        _base_actor_version="foreign" if fault == "version" else "context-ls-12",
        candidate_snapshot=lambda: SimpleNamespace(
            state=changed if fault == "candidate" else state
        ),
    )
    source = SimpleNamespace(snapshot_state=lambda: changed if fault == "source" else state)
    gate = SimpleNamespace(
        snapshot=lambda: SimpleNamespace(
            limits=SimpleNamespace(max_admitted_updates=owner.admissions)
        )
    )
    monkeypatch.setattr(owner, "before_work", lambda: None)
    monkeypatch.setattr(
        control, "compose_context_replay", lambda *a, **kw: pytest.fail("bank opened")
    )
    with pytest.raises(ValueError, match="source|initial"):
        owner.bind(runtime, source, gate)


@pytest.mark.parametrize("mistakes", [16, 17])
def test_should_preserve_original_perceptron_mistake_history_requirement(mistakes):
    from src.adapters.context_serving import snapshot_digest as perceptron_digest
    from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
    from src.core.context_perceptron import ContextPerceptronSnapshot

    heads = (
        PerceptronSnapshot(PerceptronPolicy(7, 6), ((0.0,) * 6,) * 8, 8, 48, mistakes - 4),
        PerceptronSnapshot(PerceptronPolicy(7, 6), ((0.0,) * 6,) * 8, 4, 24, 4),
    )
    state = ContextPerceptronSnapshot(
        ContextPerceptronPolicy(9, 6, (4, 5)), heads, 12, 72, mistakes
    )
    req = CatalogCounterfactualRequest(
        "qualified-12", perceptron_digest(state), "mistakes", "wake1", 0, catalog=context_catalog()
    )
    owner = ContextCatalogControl(req, base_snapshot=state)
    if mistakes == 17:
        with pytest.raises(ValueError):
            owner.require_source(state, state, digest=perceptron_digest, same_state=_same_value)
    else:
        owner.require_source(state, state, digest=perceptron_digest, same_state=_same_value)
        assert owner.report["source_binding_verified"] and "learner_binding" not in owner.report
