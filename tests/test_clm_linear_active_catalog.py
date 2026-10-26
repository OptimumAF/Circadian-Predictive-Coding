"""Typed ordered active tools with original full-model and paid-work guards."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
import src.adapters.clm_linear_decision as typed
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder, text_recipe
from src.adapters.clm_linear_fit_control import CLMLinearFitControl, predict_clm_linear_control
from src.core.clm_action_features import FrozenActionFeatures
from src.core.clm_head import FrozenEncoderIdentity, FrozenChoiceBatch
from src.core.clm_probe import FrozenProbeChoiceBatch
from src.core.temporal_decision_sequence import CATALOG, policy_for_phase
from src.core.typed_decision import DecisionRequest
from test_wide_categorical_active_readout import model


class CatalogLiteralBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=2)
        self.fault = None

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        values = torch.tensor([0.6, 0.8]).expand(*input_ids.shape, 2).clone()
        if self.fault is not None:
            self.fault()
        return SimpleNamespace(last_hidden_state=values)


class Tokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens):
        return dict(input_ids=[2])


def prepared(phase="baseline", role="train"):
    identity = FrozenEncoderIdentity("literal", "fixed", "fixed", "last", text_recipe(2048), 2)
    source = CLMLinearFitControl(
        identity,
        tuple(c.candidate_id for c in CATALOG),
        role,
        model((10000.0, 1.0, 2.0, 3.0) if phase == "baseline" else (1.0, 2.0, 10000.0, 3.0)),
    )
    actions = FrozenActionFeatures(
        identity, CATALOG, torch.tensor([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    )
    active = tuple(
        replace(c) for c in CATALOG if c.candidate_id in policy_for_phase(phase)["allowed_ids"]
    )
    backbone = CatalogLiteralBackbone().eval()
    encoder = FrozenCLMEncoder(
        backbone, Tokenizer(), identity=identity, max_sequences=1, max_forward_calls=2
    )
    return source, actions, active, encoder, backbone


def provider(parts):
    source, actions, active, encoder, _ = parts
    return typed.CLMLinearDecisionPredictor(
        encoder, source, actions, instructions="choose", max_requests=2, active_candidates=active
    )


@pytest.mark.parametrize("phase", ["baseline", "after_B"])
def test_should_bind_both_contexts_with_full_source_unchanged_and_honest_projection(phase):
    parts = prepared(phase)
    source, actions, active, encoder, _ = parts
    before = asdict(source), actions.features.clone(), torch.get_rng_state().clone()
    predict = provider(parts)
    assert predict._active_action_values.untyped_storage().nbytes() == 24
    assert predict._active_action_values.data_ptr() != predict._action_values.data_ptr()
    result = predict(DecisionRequest("request", "invoice", active))
    values = np.exp(np.array([1.0, 2.0, 3.0]) - 3)
    np.testing.assert_allclose(
        [p for _, p in result.probabilities], values / values.sum(), atol=1e-14
    )
    assert tuple(key for key, _ in result.probabilities) == tuple(c.candidate_id for c in active)
    assert result.selected_id == "route" and result.reported_input_tokens == 1
    assert result.reported_confidence == pytest.approx(
        values[-1] / values.sum() - (values[0] + values[1]) / (2 * values.sum())
    )
    assert asdict(source) == before[0] and torch.equal(actions.features, before[1])
    assert torch.equal(torch.get_rng_state(), before[2])
    assert encoder.work["forward_returns"] == 1
    work = predict.work
    assert (
        work["action_feature_copy_elements"] == 8
        and work["active_action_feature_copy_elements"] == 6
    )
    assert work["source_catalog_columns"] == 4 and work["active_catalog_columns"] == 3
    assert work["active_catalog_readout_attempts"] == work["active_catalog_readout_returns"] == 1
    row = predict.last_readout["linear_prediction"]
    assert row["projection_is_read_only"] and row["source_coefficient_array_bytes"] == 96
    assert row["coefficient_array_bytes"] == 72 and row["classes"] == 3
    assert row["active_candidate_ids"] == [c.candidate_id for c in active]
    assert source.model.squared_error == 7.0 and source.model.num_classes == 4


@pytest.mark.parametrize("role", ["train", "probe_train"])
def test_should_keep_readonly_adapter_role_without_inventing_a_projected_fit(role):
    source, actions, active, _, _ = prepared(role=role)
    ids = tuple(c.candidate_id for c in active)
    indices = [source.candidate_ids.index(k) for k in ids]
    kind = FrozenChoiceBatch if role == "train" else FrozenProbeChoiceBatch
    batch = kind(
        source.identity,
        torch.tensor([[0.6, 0.8]]),
        actions.features[indices].clone(),
        role="outer_selection" if role == "train" else "probe_test",
    )
    report: dict[str, Any] = {}
    before = asdict(source)
    p = predict_clm_linear_control(
        source, batch, source.candidate_ids, report, active_candidate_ids=ids
    )
    assert p.shape == (1, 3) and p.argmax() == 2 and asdict(source) == before
    assert report["linear_prediction"]["optimizer_steps"] == 0


@pytest.mark.parametrize(
    "fault", ["empty", "single", "duplicate", "reverse", "unknown", "description"]
)
def test_should_refuse_bad_active_binding_before_query_encoding(fault):
    source, actions, active, encoder, _ = prepared()
    if fault == "empty":
        active = ()
    elif fault == "single":
        active = active[:1]
    elif fault == "duplicate":
        active = (active[0], active[0])
    elif fault == "reverse":
        active = tuple(reversed(active))
    elif fault == "unknown":
        active = (replace(active[0], candidate_id="unknown"), active[1])
    else:
        active = (replace(active[0], description="different"), *active[1:])
    with pytest.raises(ValueError):
        typed.CLMLinearDecisionPredictor(
            encoder, source, actions, instructions="choose", active_candidates=active
        )
    assert encoder.work["forward_attempts"] == 0


def test_should_refuse_probe_before_action_access_or_encoding():
    source, _, active, encoder, _ = prepared(role="probe_train")
    with pytest.raises(ValueError, match="probe"):
        typed.CLMLinearDecisionPredictor(
            encoder, source, object(), instructions="choose", active_candidates=active
        )
    assert encoder.work["forward_attempts"] == 0


@pytest.mark.parametrize(
    "fault",
    ["request", "declaration", "inactive_weight", "active_owner", "active_value", "columns"],
)
def test_should_refuse_changed_source_binding_before_more_paid_work(fault):
    parts = prepared()
    source, actions, active, encoder, _ = parts
    predict = provider(parts)
    request = DecisionRequest("request", "invoice", active)
    if fault == "request":
        request = replace(request, allowed_candidates=actions.candidates)
    elif fault == "declaration":
        object.__setattr__(active[0], "description", "changed")
    elif fault == "inactive_weight":
        coefficients = source.model.coefficients
        object.__setattr__(
            source.model, "coefficients", (*coefficients[:-1], (9000.0, 1.0, 2.0, 3.0))
        )
    elif fault == "active_owner":
        predict._active_action_values = predict._active_action_values.clone()
    elif fault == "active_value":
        predict._active_action_values.add_(0.1)
    else:
        predict._active_columns = tuple(list(predict._active_columns))
    with pytest.raises(ValueError):
        predict(request)
    assert encoder.work["forward_attempts"] == predict.work["request_attempts"] == 0


@pytest.mark.parametrize("fault", ["declaration_after_encoder", "lost_readout", "active_counter"])
def test_should_retain_uncertain_paid_prefix_and_stop(monkeypatch, fault):
    parts = prepared()
    _, _, active, encoder, backbone = parts
    predict = provider(parts)
    if fault == "lost_readout":
        real = typed.predict_clm_linear_control

        def lost(*args, **kwargs):
            real(*args, **kwargs)
            raise RuntimeError("lost active readout")

        monkeypatch.setattr(typed, "predict_clm_linear_control", lost)
    elif fault == "active_counter":

        def changed():
            predict._work["active_catalog_readout_attempts"] += 1

        backbone.fault = changed
    else:

        def changed():
            object.__setattr__(active[0], "description", "changed")

        backbone.fault = changed
    request = DecisionRequest("request", "invoice", active)
    with pytest.raises((ValueError, RuntimeError)):
        predict(request)
    assert predict.work["request_attempts"] == 1
    # A forged counter stops accounting before the provider can admit the
    # encoder return; the encoder's own completed return remains separately paid.
    assert predict.work["encoder_returns"] == (0 if fault == "active_counter" else 1)
    assert predict.work["unreturned_work_unknown"] and encoder.work["forward_returns"] == 1
    if fault == "lost_readout":
        assert predict.last_readout["linear_prediction"]["returns"] == 1
        assert (
            predict.work["active_catalog_readout_attempts"] == 1
            and predict.work["active_catalog_readout_returns"] == 0
        )
    with pytest.raises(ValueError):
        predict(request)
    assert encoder.work["forward_attempts"] == 1
