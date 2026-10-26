"""Owned paired-CLM TRAIN candidate with a protected original source.

Copies warm weights into explicit fresh native optimizer/schedule authority.
Uses only ordinary TRAIN/readout records; no data consent, replay selection,
encoder loading, installation, export or promotion belongs to this adapter.
"""

from copy import deepcopy
from dataclasses import asdict, fields

import torch

from src.adapters.clm_head_learner import CLMHeadLearner, OBJECTIVE, _equal, _params
from src.adapters.clm_head_probe import _require_separate
from src.core.clm_head import CLMHeadPolicy, _exact


def _elements(value):
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, dict):
        return sum(_elements(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_elements(item) for item in value)
    return 0


def _admit(source, policy, max_readouts):
    if type(source) is not CLMHeadLearner:
        raise ValueError("replay candidate requires exact paired TRAIN CLMHeadLearner")
    _exact(policy, CLMHeadPolicy)
    if not 1 <= policy.max_updates <= 16 or policy.max_updates == 10 or policy.max_restores != 0:
        raise ValueError("candidate needs1..16 updates except10, and zero restores")
    if type(max_readouts) is not int or not 1 <= max_readouts <= 64:
        raise ValueError("candidate readout bound must be an integer in1..64")
    source._check()
    if (
        source.completed_updates < 1
        or source.work["commits"] < 1
        or source.work["unreturned_work_unknown"]
    ):
        raise ValueError("candidate needs a successfully trained, certain source")
    for field in fields(CLMHeadPolicy):
        if field.name not in ("max_updates", "max_restores"):
            before, after = getattr(source.policy, field.name), getattr(policy, field.name)
            if type(before) is not type(after) or before != after:
                raise ValueError("candidate must preserve source native settings and dimensions")


class CLMReplayCandidate:
    """Fresh finite TRAIN owner; a private source guard is not export authority."""

    definition = OBJECTIVE

    def __init__(self, source, *, policy, max_readouts=2):
        _admit(source, policy, max_readouts)
        self._source = source
        self._source_bundle = source._bundle
        self._source_work_owner, self._source_rng_owner = source._work, source._rng
        self._source_work = source.work
        self._source_counts = (
            source.completed_updates,
            source.training_examples,
            source.generation,
        )
        self._copy_work = dict(
            construction_attempts=1,
            construction_returns=0,
            source_capture_attempts=0,
            source_capture_returns=0,
            source_capture_tensor_elements=0,
            child_rng_copy_elements=0,
        )
        self._source_pin = self._capture_source()
        self._failed = False
        with torch.random.fork_rng(devices=[]):
            try:
                learner = CLMHeadLearner(
                    *source._bundle[:3], identity=source.identity, policy=policy
                )
                # This shared check grants no probe role or probe conversion.
                _require_separate(source, learner)
                for parameter in _params(learner._bundle):
                    parameter.grad = None
                self._copy_work["child_rng_copy_elements"] += source._rng.numel()
                learner._rng = source._rng.clone()
            finally:
                self._require_source()
        self._learner = learner
        self._max_readouts = max_readouts
        self._birth = (learner, deepcopy(learner.identity), deepcopy(policy), max_readouts)
        self._copy_work["construction_returns"] = 1

    def _capture_source(self):
        self._copy_work["source_capture_attempts"] += 1
        result = self._source._capture()
        self._copy_work["source_capture_returns"] += 1
        self._copy_work["source_capture_tensor_elements"] += _elements(vars(result))
        return result

    def _require_source(self):
        source = self._source
        source._check()
        if (
            source._bundle is not self._source_bundle
            or source._work is not self._source_work_owner
            or source._rng is not self._source_rng_owner
            or source.work != self._source_work
            or (source.completed_updates, source.training_examples, source.generation)
            != self._source_counts
            or not _equal(vars(self._capture_source()), vars(self._source_pin))
        ):
            raise ValueError("original replay source state, work, owner or progress changed")

    def _check(self):
        if self._failed:
            raise ValueError("replay candidate stopped after uncertain native work")
        self._require_source()
        if (
            self._learner is not self._birth[0]
            or self.identity != self._birth[1]
            or self.policy != self._birth[2]
            or type(self._max_readouts) is not int
            or self._max_readouts != self._birth[3]
        ):
            raise ValueError("candidate native owner, policy or readout authority changed")
        self._learner._check()

    @property
    def identity(self):
        return self._learner.identity

    @property
    def policy(self):
        return self._learner.policy

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
    def work(self):
        return self._learner.work

    @property
    def copy_work(self):
        return dict(self._copy_work)

    @property
    def source_metadata(self):
        return dict(
            identity=asdict(self._source.identity),
            completed_updates=self._source_counts[0],
            training_examples=self._source_counts[1],
            generation=self._source_counts[2],
            work=dict(self._source_work),
        )

    def train_batch(self, batch, targets):
        self._check()
        try:
            return self._learner.train_batch(batch, targets)
        finally:
            self._failed = self._learner._failed
            self._require_source()

    def predict_choices(self, batch):
        self._check()
        if self.work["choice_prediction_attempts"] >= self._max_readouts:
            raise ValueError("candidate readout allowance exhausted")
        try:
            return self._learner.predict_choices(batch)
        finally:
            self._failed = self._learner._failed
            self._require_source()
