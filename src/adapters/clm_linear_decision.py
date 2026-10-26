"""TRAIN-only fixed-class linear scores through the existing typed decision port.

Uses one frozen query encoding per request, original raw action declarations and
the detached categorical control. No fitting, source learner, projection cache,
promotion, probe deployment, final labels or external tool effects.
"""

from copy import deepcopy
from dataclasses import asdict
from typing import Any

import torch

from src.adapters.clm_decision import _action_feature_pin, _require_action_feature_pin
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.adapters.clm_linear_fit_control import (
    CLMLinearFitControl,
    predict_clm_linear_control,
    _tensor,
    _active_candidate_columns,
)
from src.core.clm_head import FrozenChoiceBatch, _exact
from src.core.typed_decision import (
    DecisionRequest,
    DecisionCandidate,
    ProbabilityDecisionResult,
    _text,
    validate_probability_decision_result,
)


def _same_work(actual, expected):
    return (
        type(actual) is dict
        and actual.keys() == expected.keys()
        and all(type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items())
    )


def _encoder_return_work(encoder, before, keeps):
    """Admit only counters explained by this observed nonempty encoding return."""
    observed = encoder.last_readout
    count = len(keeps)
    if (
        type(observed) is not dict
        or observed.get("keeps") != keeps
        or observed.get("forward_returned") is not True
    ):
        raise ValueError("linear operation lacks its original encoder return")
    original, retained = observed.get("original_token_lengths"), observed.get("retained_lengths")
    if (
        type(original) is not list
        or type(retained) is not list
        or len(original) != count
        or len(retained) != count
        or any(
            type(a) is not int or type(b) is not int or not 1 <= b <= a
            for a, b in zip(original, retained, strict=True)
        )
        or observed.get("padded_tokens") != count * max(retained)
        or len(observed.get("token_ids", ())) != count
        or any(
            len(ids) != length for ids, length in zip(observed["token_ids"], retained, strict=True)
        )
    ):
        raise ValueError("linear operation requires bounded nonempty observed token rows")
    expected = dict(before)
    for key, delta in dict(
        embedding_attempts=1,
        embedding_returns=1,
        tokenization_attempts=count,
        tokenization_returns=count,
        token_ids_visited=sum(original),
        retained_token_ids=sum(retained),
        forward_attempts=1,
        forward_returns=1,
        sequences_attempted=count,
        sequences_returned=count,
        padded_tokens_attempted=count * max(retained),
        pooled_values=count * encoder.identity.embedding_dim,
    ).items():
        expected[key] += delta
    expected["unreturned_work_unknown"] = False
    if not _same_work(encoder.work, expected):
        raise ValueError("linear operation work differs from its observed encoder return")
    return expected, sum(retained)


class CLMLinearDecisionPredictor:
    def __init__(
        self,
        encoder,
        control,
        action_features,
        *,
        instructions,
        max_requests=18,
        active_candidates=None,
    ):
        if type(encoder) is not FrozenCLMEncoder:
            raise ValueError("linear provider requires an exact frozen CLM encoder")
        _exact(control, CLMLinearFitControl)
        if control.role != "train":
            raise ValueError("a disposable probe fit cannot become a serving provider")
        _text(instructions, "instructions", 4096)
        if type(max_requests) is not int or not 1 <= max_requests <= 36:
            raise ValueError("linear provider requires1..36 requests")
        pin = _action_feature_pin(encoder.identity, action_features)
        ids = tuple(c.candidate_id for c in action_features.candidates)
        if control.identity != encoder.identity or ids != control.candidate_ids or len(ids) > 8:
            raise ValueError("linear provider encoder or fixed typed catalog differs")
        _tensor(action_features.features, len(ids), control.identity.embedding_dim)
        active = action_features.candidates if active_candidates is None else active_candidates
        if type(active) is not tuple:
            raise ValueError("active candidates require an immutable original catalog subset")
        for candidate in active:
            _exact(candidate, DecisionCandidate)
        active_ids = tuple(candidate.candidate_id for candidate in active)
        columns = _active_candidate_columns(ids, active_ids)
        if any(
            candidate != action_features.candidates[index]
            for candidate, index in zip(active, columns, strict=True)
        ):
            raise ValueError("active candidate descriptions differ from the original catalog")
        encoder._check()
        self._encoder, self._source, self._actions = encoder, control, action_features
        self._control = deepcopy(control)
        self._action_values = action_features.features.detach().clone()
        self._action_pin = pin
        self._action_storage = self._action_values.untyped_storage().data_ptr()
        self._action_version = self._action_values._version
        self._source_fields, self._owned_fields = dict(vars(control)), dict(vars(self._control))
        self._model_proof = asdict(control)
        self._active_candidates, self._active_ids, self._active_columns = (
            active,
            active_ids,
            columns,
        )
        self._active_enabled = active_candidates is not None
        self._active_action_values = (
            self._action_values
            if not self._active_enabled
            # Advanced row selection already returns owned storage.
            else self._action_values[list(columns)]
        )
        self._active_storage = self._active_action_values.untyped_storage().data_ptr()
        self._active_version = self._active_action_values._version
        self._active_fields = tuple(dict(vars(candidate)) for candidate in active)
        self._active_proof = tuple(asdict(candidate) for candidate in active)
        self._instructions, self._max_requests = instructions, max_requests
        self._birth = (
            encoder,
            control,
            action_features,
            self._control,
            self._action_values,
            instructions,
            max_requests,
            self._active_candidates,
            self._active_ids,
            self._active_columns,
            self._active_enabled,
            self._active_action_values,
        )
        self._encoder_expected = encoder.work
        self._failed = False
        self._last_readout: dict[str, Any] | None = None
        self._work = dict(
            request_attempts=0,
            request_returns=0,
            encoder_attempts=0,
            encoder_returns=0,
            linear_readout_attempts=0,
            linear_readout_returns=0,
            linear_rows=0,
            action_feature_copy_elements=self._action_values.numel(),
            head_forward_attempts=0,
            optimizer_steps=0,
            unreturned_work_unknown=False,
        )
        if self._active_enabled:
            self._work.update(
                source_catalog_columns=len(ids),
                active_catalog_columns=len(active_ids),
                active_action_feature_copy_elements=self._active_action_values.numel(),
                active_catalog_readout_attempts=0,
                active_catalog_readout_returns=0,
            )
        self._expected = dict(self._work)
        self._work_owner = self._work
        self._check()

    @property
    def work(self):
        return dict(self._work)

    @property
    def last_readout(self):
        return deepcopy(self._last_readout)

    def _charge(self, **changes):
        if self._work is not self._work_owner or not _same_work(self._work, self._expected):
            raise ValueError("linear provider paid counters changed")
        for key, value in changes.items():
            self._work[key] = value if type(value) is bool else self._work[key] + value
            self._expected[key] = self._work[key]

    def _check(self):
        if self._failed:
            raise ValueError("linear provider stopped after uncertain paid work")
        now = (
            self._encoder,
            self._source,
            self._actions,
            self._control,
            self._action_values,
            self._instructions,
            self._max_requests,
            self._active_candidates,
            self._active_ids,
            self._active_columns,
            self._active_enabled,
            self._active_action_values,
        )
        if any(a is not b for a, b in zip(now, self._birth, strict=True)):
            raise ValueError("original linear provider owners or limits changed")
        for model, fields in (
            (self._source, self._source_fields),
            (self._control, self._owned_fields),
        ):
            _exact(model, CLMLinearFitControl)
            if (
                any(getattr(model, k) is not v for k, v in fields.items())
                or asdict(model) != self._model_proof
            ):
                raise ValueError("original fitted linear control changed")
        _require_action_feature_pin(self._encoder.identity, self._actions, self._action_pin)
        if (
            self._action_values.untyped_storage().data_ptr() != self._action_storage
            or self._action_values._version != self._action_version
            or not torch.equal(self._action_values, self._actions.features)
        ):
            raise ValueError("owned raw action features changed")
        for candidate, fields, proof in zip(
            self._active_candidates, self._active_fields, self._active_proof, strict=True
        ):
            _exact(candidate, DecisionCandidate)
            if (
                any(getattr(candidate, k) is not v for k, v in fields.items())
                or asdict(candidate) != proof
            ):
                raise ValueError("original active candidate declaration changed")
        if (
            self._active_action_values.untyped_storage().data_ptr() != self._active_storage
            or self._active_action_values._version != self._active_version
            or (
                self._active_enabled
                and any(
                    not torch.equal(self._active_action_values[index], self._action_values[column])
                    for index, column in enumerate(self._active_columns)
                )
            )
        ):
            raise ValueError("owned active action features changed")
        self._encoder._check()
        if not _same_work(self._encoder.work, self._encoder_expected):
            raise ValueError("original linear provider encoder counters changed")
        if self._work is not self._work_owner or not _same_work(self._work, self._expected):
            raise ValueError("linear provider paid counters changed")

    def _accept_encoder_return(self):
        expected, retained = _encoder_return_work(self._encoder, self._encoder_expected, ("tail",))
        self._encoder_expected = expected
        return retained

    def __call__(self, request):
        _exact(request, DecisionRequest)
        for candidate in request.allowed_candidates:
            _exact(candidate, DecisionCandidate)
        self._check()
        if request.allowed_candidates != self._active_candidates:
            raise ValueError(
                "request IDs, descriptions or order differ from the fixed fitted catalog"
            )
        if self._work["request_attempts"] >= self._max_requests:
            raise ValueError("linear provider request allowance exhausted")
        owned = deepcopy(request)
        text = owned.query.strip() + "\n\n" + self._instructions.strip()
        self._last_readout = dict(
            request_id=owned.request_id,
            state_text=text,
            definition=(
                "clm_frozen_feature_linear_active_catalog_v1"
                if self._active_enabled
                else "clm_frozen_feature_linear_fit_control_v1"
            ),
            linear_prediction=None,
        )
        if self._active_enabled:
            self._last_readout["active_catalog"] = dict(
                source_ids=list(self._control.candidate_ids),
                active_ids=list(self._active_ids),
                columns=list(self._active_columns),
                projection_is_read_only=True,
            )
        self._charge(request_attempts=1, unreturned_work_unknown=True)
        local: dict[str, Any] = {}
        try:
            self._charge(encoder_attempts=1)
            features = self._encoder.embed_texts((text,), ("tail",))
            retained = self._accept_encoder_return()
            _tensor(features, 1, self._control.identity.embedding_dim)
            self._charge(encoder_returns=1)
            self._check()
            self._charge(linear_readout_attempts=1)
            options = {}
            if self._active_enabled:
                self._charge(active_catalog_readout_attempts=1)
                options["active_candidate_ids"] = self._active_ids
            probabilities = predict_clm_linear_control(
                self._control,
                FrozenChoiceBatch(
                    self._control.identity,
                    features,
                    self._active_action_values,
                    role="outer_selection",
                ),
                self._control.candidate_ids,
                local,
                **options,
            )[0].tolist()
            self._charge(linear_readout_returns=1, linear_rows=1)
            if self._active_enabled:
                self._charge(active_catalog_readout_returns=1)
            winner = max(range(len(probabilities)), key=probabilities.__getitem__)
            rest = [p for i, p in enumerate(probabilities) if i != winner]
            confidence = max(0.0, min(1.0, probabilities[winner] - sum(rest) / len(rest)))
            if request != owned:
                raise ValueError("linear prediction changed its original typed request")
            self._check()
            ids = self._active_ids
            result = ProbabilityDecisionResult(
                owned.request_id,
                ids[winner],
                tuple(zip(ids, probabilities, strict=True)),
                reported_confidence=confidence,
                reported_input_tokens=retained,
            )
            validate_probability_decision_result(owned, result)
            self._charge(request_returns=1, unreturned_work_unknown=False)
            return result
        except Exception:
            self._failed = True
            raise
        finally:
            self._last_readout["linear_prediction"] = deepcopy(local.get("linear_prediction"))
