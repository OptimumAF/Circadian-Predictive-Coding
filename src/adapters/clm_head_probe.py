"""Bounded disposable warm-weight CLM probe using the native update engine.

Owns a fresh AdamW/OneCycle state and separate feature-role admission. Source
state is checked only during construction and is never retained. No public
snapshot, restore, fork, installation or export interface is supplied. Private
Python objects remain trusted; this is not malicious-code containment.
"""

from copy import deepcopy
from dataclasses import fields

import torch

from src.adapters.clm_head_learner import CLMHeadLearner, _params
from src.core.clm_head import CLMHeadPolicy, _exact
from src.core.clm_probe import FrozenProbeChoiceBatch, FrozenProbeEmbeddingBatch


class _ProbeHeadLearner(CLMHeadLearner):
    def _batch(self, batch, targets=None):
        return self._paired_embeddings(batch, targets, kind=FrozenProbeEmbeddingBatch)

    def _choice_batch(self, batch):
        return self._choice_embeddings(batch, kind=FrozenProbeChoiceBatch)


def _admit_probe(source, policy, max_readouts):
    if type(source) is not CLMHeadLearner:
        raise ValueError("a probe requires an exact original CLMHeadLearner source")
    _admit_probe_policy(source, policy, max_readouts)


def _admit_probe_policy(source, policy, max_readouts):
    """Shared limits after each public probe admits its exact source family."""
    _exact(policy, CLMHeadPolicy)
    if not 1 <= policy.max_updates <= 16 or policy.max_updates == 10 or policy.max_restores != 0:
        # Native OneCycle pct_start=.1 has a zero-length initial phase at10.
        # Refuse prospectively instead of changing the native schedule.
        raise ValueError("probe policy requires1..16 updates except10 and zero restores")
    if type(max_readouts) is not int or not 1 <= max_readouts <= 64:
        raise ValueError("max_readouts must be a native integer in1..64")
    source._check()
    original_policy = source.policy
    for field in fields(CLMHeadPolicy):
        if field.name not in ("max_updates", "max_restores"):
            supplied, original = getattr(policy, field.name), getattr(original_policy, field.name)
            if type(supplied) is not type(original) or supplied != original:
                raise ValueError("probe policy must preserve every other source setting")


def _require_separate(source, probe):
    source_parameters, probe_parameters = _params(source._bundle), _params(probe._bundle)
    if len(source_parameters) != len(probe_parameters):
        raise ValueError("probe parameter schema differs from source")
    source_storage = {parameter.untyped_storage().data_ptr() for parameter in source_parameters}
    for original, copied in zip(source_parameters, probe_parameters, strict=True):
        if (
            copied is original
            or copied.untyped_storage().data_ptr() in source_storage
            or not torch.equal(original, copied)
        ):
            raise ValueError("probe must own separate exact warm-weight parameters")
    source_modules = {id(module) for head in source._bundle[:2] for module in head.modules()}
    if any(id(module) in source_modules for head in probe._bundle[:2] for module in head.modules()):
        raise ValueError("probe must own its projection modules")
    if (
        probe._bundle[3] is source._bundle[3]
        or probe._bundle[4] is source._bundle[4]
        or probe._bundle[3].state
    ):
        raise ValueError("probe optimizer and scheduler must be fresh native owners")


class CLMHeadProbe:
    def __init__(self, source, *, policy, max_readouts):
        _admit_probe(source, policy, max_readouts)
        self._initialize(
            source, policy=policy, max_readouts=max_readouts, learner_kind=_ProbeHeadLearner
        )

    def _initialize(self, source, *, policy, max_readouts, learner_kind):
        # Family-specific admission precedes this shared owned-copy lifecycle.
        owner, before = source._bundle, source._capture()
        source_work = source.work
        source_counts = (source.completed_updates, source.training_examples, source.generation)
        rng_owner, work_owner = source._rng, source._work
        with torch.random.fork_rng(devices=[]):
            try:
                learner = learner_kind(
                    owner[0], owner[1], owner[2], identity=source.identity, policy=policy
                )
                _require_separate(source, learner)
                for parameter in _params(learner._bundle):
                    parameter.grad = None
                learner._rng = source._rng.clone()
            finally:
                source._require_original(owner, before)
                if (
                    source.work != source_work
                    or source._work is not work_owner
                    or source._rng is not rng_owner
                    or (source.completed_updates, source.training_examples, source.generation)
                    != source_counts
                ):
                    raise ValueError(
                        "probe construction changed original source progress or physical work"
                    )
        self._learner = learner
        self._max_readouts = max_readouts
        self._birth = (learner, deepcopy(learner.identity), deepcopy(learner.policy), max_readouts)
        self._source_counts = source_counts
        self._failed = False

    @property
    def completed_updates(self):
        return self._learner.completed_updates

    @property
    def training_examples(self):
        return self._learner.training_examples

    @property
    def generation(self):
        return self._learner.generation

    @property
    def identity(self):
        return self._learner.identity

    @property
    def policy(self):
        return self._learner.policy

    @property
    def work(self):
        return self._learner.work

    @property
    def source_completed_updates(self):
        return self._source_counts[0]

    @property
    def source_training_examples(self):
        return self._source_counts[1]

    @property
    def source_generation(self):
        return self._source_counts[2]

    def _check(self):
        if self._failed:
            raise ValueError("disposable CLM probe stopped after uncertain native work")
        if (
            self._learner is not self._birth[0]
            or self._learner.identity != self._birth[1]
            or self._learner.policy != self._birth[2]
            or type(self._max_readouts) is not int
            or self._max_readouts != self._birth[3]
        ):
            raise ValueError("probe native owner, identity, policy or readout limit changed")
        self._learner._check()

    def train_batch(self, batch, targets):
        self._check()
        try:
            return self._learner.train_batch(batch, targets)
        finally:
            if self._learner._failed:
                self._failed = True

    def train_batch_observed(self, batch, targets):
        """Same disposable update authority with an immutable scalar return."""
        self._check()
        try:
            return self._learner.train_batch_observed(batch, targets)
        finally:
            if self._learner._failed:
                self._failed = True

    def predict_choices(self, batch):
        self._check()
        if self._learner.work["choice_prediction_attempts"] >= self._max_readouts:
            raise ValueError("disposable probe readout allowance exhausted")
        try:
            return self._learner.predict_choices(batch)
        finally:
            if self._learner._failed:
                self._failed = True
