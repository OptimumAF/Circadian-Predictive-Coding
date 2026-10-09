"""Separately budgeted actual CPC fork/memo observation; no holder authority."""

from unittest.mock import patch
from typing import Callable
import numpy as np
import pytest

import src.adapters.numpy_learners as adapter
from src.core.native_model_copy import ModelCopy, ModelCopyLimits, observe_model_copies
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork, ReplaySnapshot
from test_native_replay_capture_origins import setup
from test_native_managed_replay_origins import native_work  # noqa: F401


def test_should_observe_real_constructor_fork_memo_and_preserve_default_graph():
    owner, life, runtime, ledger = setup()
    learner = runtime._candidate
    model = learner._model
    row = model._replay_memory[0]
    fields, learner_fields = tuple(vars(model)), tuple(vars(learner))
    policy = (learner._learning_rate, learner._inference_steps, learner._inference_learning_rate)
    receipt = runtime._inbox._applied[("e1", "s1")]
    old_accounting = ledger.accounting()
    old_raw_charge = life._copy_budget._charged
    handles: list[tuple[Callable[[], ModelCopy], Callable[[object], object]]] = []
    stages: list[str] = []
    copied_rows: list[ReplaySnapshot] = []

    def observer(stage, read, lookup):
        actual = read()
        assert actual.source is model
        if stage == "before_copy":
            assert actual.target is None
            assert copying.call_count == 0
        else:
            target = actual.target
            assert isinstance(target, CircadianPredictiveCodingNetwork)
            assert lookup(model) is target
            copied = target._replay_memory[0]
            assert lookup(row) is copied
            assert lookup(row.input_batch) is copied.input_batch
            assert lookup(row.target_batch) is copied.target_batch
            assert copied is not row and copied.input_batch is not row.input_batch
            assert copied.target_batch is not row.target_batch
            assert lookup(model.config) is target.config
            assert lookup(model._replay_memory) is target._replay_memory
            assert tuple(vars(target)) == fields
            copied_rows.append(copied)
        handles.append((read, lookup))
        stages.append(stage)

    def refuse(stage, read, lookup):
        assert stage == "before_copy" and read().source is model
        raise ValueError("original before-copy admission refused")

    with patch.object(adapter, "deepcopy", wraps=adapter.deepcopy) as refused_copy:
        with observe_model_copies(model, refuse, ModelCopyLimits(1, 2, 8)):
            with pytest.raises(ValueError, match="before-copy admission"):
                learner.fork()
        assert refused_copy.call_count == 0
    baseline = learner.fork()
    with patch.object(adapter, "deepcopy", wraps=adapter.deepcopy) as copying:
        with observe_model_copies(model, observer, ModelCopyLimits(1, 2, 8)):
            copied_learner = learner.fork()
        assert copying.call_count == 1 and len(copying.call_args.args) == 2
    assert stages == ["before_copy", "copied"]
    assert tuple(vars(copied_learner)) == learner_fields
    assert (
        copied_learner._learning_rate,
        copied_learner._inference_steps,
        copied_learner._inference_learning_rate,
    ) == policy
    for key, value in vars(model).items():
        if type(value) is np.ndarray:
            np.testing.assert_array_equal(vars(baseline._model)[key], value)
            np.testing.assert_array_equal(vars(copied_learner._model)[key], value)
    np.testing.assert_array_equal(copied_rows[0].input_batch, row.input_batch)
    assert ledger.origins()[0].key == ("e1", "s1")
    assert ledger.accounting() == old_accounting
    assert life._copy_budget._charged == old_raw_charge
    assert runtime._inbox._applied[("e1", "s1")] is receipt
    assert runtime._budget.updates_completed == 1
    for read, lookup in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()
        with pytest.raises(ValueError, match="synchronous"):
            lookup(model)
