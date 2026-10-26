"""Single-use catalog learning to frozen typed inference binding.

Admits a current trained source and existing frozen encoder, copies heads through
the ordinary predictor, and retains only a scalar historical receipt after use.
No encoder/head initialization, training, scoring or promotion is performed.
"""

from copy import deepcopy
from dataclasses import asdict, fields

import torch

from src.adapters.clm_catalog_learner import CLMCatalogLearner, CLMCatalogSnapshot
from src.adapters.clm_decision import (
    CLMDecisionPredictor,
    _action_feature_pin,
    _require_action_feature_pin,
)
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.adapters.clm_head_learner import CLMHeadLearner, _equal, _modes, _owners, _params
from src.core.clm_catalog import CATALOG_OBJECTIVE
from src.core.typed_decision import _text


def _tensor_pins(value):
    if isinstance(value, torch.Tensor):
        return ((value, value.untyped_storage().data_ptr(), value._version),)
    if isinstance(value, dict):
        return tuple(pin for item in value.values() for pin in _tensor_pins(item))
    if isinstance(value, (list, tuple)):
        return tuple(pin for item in value for pin in _tensor_pins(item))
    return ()


def _tensor_elements(value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(_tensor_elements(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_elements(item) for item in value)
    return 0


def _native_owners(source):
    optimizer = source._bundle[3]
    return (
        source,
        source._bundle,
        *_owners(source._bundle),
        source._rng,
        source._work,
        source._identity,
        source._policy,
        optimizer.state,
        optimizer.param_groups,
        *optimizer.param_groups,
        *optimizer.state.values(),
    )


def _native_tensors(source):
    parameters = _params(source._bundle)
    return _tensor_pins(
        (parameters, tuple(p.grad for p in parameters), source._bundle[3].state, source._rng)
    )


def _same_owners(actual, original):
    return len(actual) == len(original) and all(
        a is b for a, b in zip(actual, original, strict=True)
    )


def _copy_predictor(
    encoder,
    state_head,
    action_head,
    logit_scale,
    *,
    instructions,
    head_revision,
    action_features=None,
):
    extra = {} if action_features is None else {"action_features": action_features}
    return CLMDecisionPredictor(
        encoder,
        state_head,
        action_head,
        logit_scale,
        instructions=instructions,
        head_revision=head_revision,
        **extra,
    )


class CLMCatalogPredictorBinding:
    def __init__(self, source, encoder, *, instructions, head_revision, action_features=None):
        if type(source) is not CLMCatalogLearner or type(encoder) is not FrozenCLMEncoder:
            raise ValueError("binding requires exact catalog learner and frozen encoder")
        _text(instructions, "instructions", 2048)
        _text(head_revision, "head_revision", 128)
        source._check()
        encoder._check()
        if (
            source.completed_updates < 1
            or source.work["commits"] < 1
            or source.work["unreturned_work_unknown"] is not False
        ):
            raise ValueError("binding requires a successfully trained catalog source")
        if source.identity != encoder.identity:
            raise ValueError("source and frozen encoder identities differ")
        if action_features is not None:
            self._action_features = action_features
            self._action_pin = _action_feature_pin(encoder.identity, action_features)
        self._source, self._encoder = source, encoder
        self._instructions, self._revision = instructions, head_revision
        self._text_declarations = (instructions, head_revision)
        self._identity, self._policy = deepcopy(source._identity), deepcopy(source._policy)
        self._counts = (source.completed_updates, source.training_examples, source.generation)
        self._source_work = dict(source._work)
        self._source_owners = _native_owners(source)
        self._source_tensors = _native_tensors(source)
        self._source_modes = _modes(source._bundle)
        self._scheduler = deepcopy(source._bundle[4].state_dict())
        self._encoder_owners = (
            encoder,
            encoder._model,
            encoder._tokenizer,
            encoder._identity,
            encoder._limits,
            encoder._work,
            encoder._last_readout,
        )
        self._encoder_work = dict(encoder._work)
        self._attempted = False
        self._receipt = None
        self._work = dict.fromkeys(
            (
                "binding_attempts",
                "binding_returns",
                "snapshot_attempts",
                "snapshot_returns",
                "guard_capture_attempts",
                "guard_capture_returns",
                "snapshot_tensor_elements_copied",
                "head_copy_attempts",
                "head_copy_returns",
                "head_parameter_elements_attempted",
                "head_parameter_elements_returned",
                "scale_copy_elements_attempted",
                "scale_copy_elements_returned",
            ),
            0,
        )
        self._work["unreturned_work_unknown"] = False
        if action_features is not None:
            self._work.update(
                action_feature_copy_attempts=0,
                action_feature_copy_returns=0,
                action_feature_elements_copied=0,
            )
        self._require_birth()

    @property
    def work(self):
        return dict(self._work)

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    def _require_birth(self):
        source, encoder = self._source, self._encoder
        if hasattr(self, "_action_features"):
            _require_action_feature_pin(encoder.identity, self._action_features, self._action_pin)
        if (
            type(self._instructions) is not str
            or type(self._revision) is not str
            or (self._instructions, self._revision) != self._text_declarations
        ):
            raise ValueError("binding instructions or head revision changed")
        source._check()
        encoder._check()
        expected = dict(self._source_work)
        expected["snapshot_attempts"] += self._work["snapshot_attempts"]
        expected["snapshot_returns"] += self._work["snapshot_returns"]
        if (
            not _same_owners(_native_owners(source), self._source_owners)
            or _modes(source._bundle) != self._source_modes
            or not _equal(source._identity, self._identity)
            or not _equal(source._policy, self._policy)
            or source.identity != self._identity
            or source.policy != self._policy
            or (source.completed_updates, source.training_examples, source.generation)
            != self._counts
            or not _equal(source.work, expected)
            or not _equal(source._work, expected)
            or not _equal(source._bundle[4].state_dict(), self._scheduler)
        ):
            raise ValueError("catalog source owner, declarations, progress or work changed")
        current = _native_tensors(source)
        if len(current) != len(self._source_tensors) or any(
            now[0] is not before[0] or now[1:] != before[1:]
            for now, before in zip(current, self._source_tensors, strict=True)
        ):
            raise ValueError("catalog source tensor owner or value version changed")
        owners = (
            encoder,
            encoder._model,
            encoder._tokenizer,
            encoder._identity,
            encoder._limits,
            encoder._work,
            encoder._last_readout,
        )
        if (
            not _same_owners(owners, self._encoder_owners)
            or encoder.identity != self._identity
            or not _equal(encoder.work, self._encoder_work)
            or not _equal(encoder._work, self._encoder_work)
        ):
            raise ValueError("frozen encoder owner, identity or work changed")

    def _capture(self):
        self._work["guard_capture_attempts"] += 1
        state = CLMHeadLearner._capture(self._source)
        self._work["guard_capture_returns"] += 1
        # Includes both private guard snapshots and the separate public snapshot.
        self._work["snapshot_tensor_elements_copied"] += _tensor_elements(vars(state))
        return state

    def _require_current(self, before):
        self._require_birth()
        current = self._capture()
        self._require_birth()
        if not _equal(vars(current), vars(before)):
            raise ValueError("catalog source full native state changed during binding")

    def _require_predictor(self, predictor, snapshot):
        if type(predictor) is not CLMDecisionPredictor:
            raise ValueError("binding must return the existing exact CLMDecisionPredictor")
        predictor._check()
        if (
            hasattr(self, "_action_features")
            and getattr(predictor, "_action_features", None) is not self._action_features
        ):
            raise ValueError("predictor raw action source differs from binding authority")
        if not hasattr(self, "_action_features") and hasattr(predictor, "_action_features"):
            raise ValueError("uncached binding cannot acquire raw action authority")
        if (
            predictor._encoder is not self._encoder
            or predictor._instructions != self._instructions
            or predictor._revision != self._revision
            or not _equal(predictor._state_head.state_dict(), snapshot.state_head)
            or not _equal(predictor._action_head.state_dict(), snapshot.action_head)
            or not torch.equal(predictor._scale, snapshot.logit_scale)
        ):
            raise ValueError("predictor copies differ from admitted catalog heads")
        original = _params(self._source._bundle)
        copied = (
            *predictor._state_head.parameters(),
            *predictor._action_head.parameters(),
            predictor._scale,
        )
        storage = {value.untyped_storage().data_ptr() for value in original}
        if any(value.untyped_storage().data_ptr() in storage for value in copied):
            raise ValueError("predictor must own independent frozen head storage")

    def bind(self):
        if self._attempted:
            raise ValueError("catalog predictor binding has one permanent attempt")
        self._attempted = True
        self._work["binding_attempts"] += 1
        before = None
        try:
            with torch.random.fork_rng(devices=[]):
                self._require_birth()
                self._work["unreturned_work_unknown"] = True
                before = self._capture()
                self._work["snapshot_attempts"] += 1
                snapshot = self._source.snapshot_state()
                self._work["snapshot_returns"] += 1
                if isinstance(snapshot, CLMCatalogSnapshot):
                    self._work["snapshot_tensor_elements_copied"] += _tensor_elements(
                        vars(snapshot)
                    )
                if type(snapshot) is not CLMCatalogSnapshot or set(vars(snapshot)) != {
                    field.name for field in fields(CLMCatalogSnapshot)
                }:
                    raise ValueError("expected fresh exact complete CLMCatalogSnapshot")
                self._require_current(before)
                if not _equal(vars(snapshot), vars(before)):
                    raise ValueError(
                        "public catalog snapshot differs from current full native state"
                    )
                head_values = sum(
                    parameter.numel() for parameter in _params(self._source._bundle)[:-1]
                )
                self._work["head_copy_attempts"] += 2
                self._work["head_parameter_elements_attempted"] += head_values
                self._work["scale_copy_elements_attempted"] += 2
                extra = {}
                if hasattr(self, "_action_features"):
                    extra["action_features"] = self._action_features
                    self._work["action_feature_copy_attempts"] += 1
                predictor = _copy_predictor(
                    self._encoder,
                    *self._source._bundle[:3],
                    instructions=self._instructions,
                    head_revision=self._revision,
                    **extra,
                )
                if extra:
                    self._work["action_feature_copy_returns"] += 1
                    self._work["action_feature_elements_copied"] += predictor.work[
                        "action_feature_elements_copied"
                    ]
                self._work["head_copy_returns"] += 2
                self._work["head_parameter_elements_returned"] += head_values
                self._work["scale_copy_elements_returned"] += 2
                self._require_predictor(predictor, snapshot)
                self._require_current(before)
                self._work["binding_returns"] += 1
                self._work["unreturned_work_unknown"] = False
                self._receipt = dict(
                    objective=CATALOG_OBJECTIVE,
                    source_completed_updates=self._counts[0],
                    source_training_examples=self._counts[1],
                    source_generation=self._counts[2],
                    identity=asdict(self._identity),
                    policy=asdict(self._policy),
                    instructions=self._instructions,
                    head_revision=self._revision,
                    work=self.work,
                )
                return predictor
        except Exception as error:
            self._receipt = None
            if before is not None:
                try:
                    with torch.random.fork_rng(devices=[]):
                        self._require_current(before)
                except Exception as guard_error:
                    raise ValueError(
                        "binding failed and original-state guard refused: " + str(guard_error)
                    ) from error
            raise
        finally:
            # One attempt is spent. Do not retain old native heads/optimizer snapshots
            # after the source later trains independently of its frozen predictor.
            del self._source, self._encoder
            del self._source_owners, self._source_tensors, self._encoder_owners
            del self._scheduler
            if hasattr(self, "_action_features"):
                del self._action_features, self._action_pin
