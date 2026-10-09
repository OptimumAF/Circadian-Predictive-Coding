"""Separately budgeted actual managed checkpoint policy-failure copy witnesses."""

from contextlib import ExitStack
from hashlib import sha256
import json
import pickle
from unittest.mock import patch
import numpy as np
import pytest

from src.adapters.numpy_learners import CircadianLearner, ManagedNumpyBuilder
from src.adapters.numpy_replay_copies import replay_builder_source
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_replay_copies import ManagedReplayCopies
from test_native_replay_capture_origins import setup
from test_native_managed_replay_origins import native_work  # noqa: F401


@pytest.fixture(autouse=True)
def checkpoint_work(tmp_path_factory):
    counts = dict(learner_snapshots=0, learner_restores=0)
    with ExitStack() as stack:
        for method, key in [
            ("snapshot_state", "learner_snapshots"),
            ("restore_state", "learner_restores"),
        ]:
            original = getattr(CircadianLearner, method)

            def call(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(CircadianLearner, method, call))
        try:
            yield
        finally:
            (
                tmp_path_factory.mktemp("checkpoint-copy-work") / "checkpoint-copy-work.json"
            ).write_text(json.dumps(counts), encoding="utf8")
            assert counts == dict(learner_snapshots=2, learner_restores=0)


def test_should_bind_original_managed_copy_to_actual_retained_failed_checkpoint_preparation():
    owner, life, runtime, ledger = setup()
    source = runtime._candidate
    row = source._model._replay_memory[0]
    fields = tuple(vars(source._model))
    policy = (source._learning_rate, source._inference_steps, source._inference_learning_rate)
    builder = ManagedNumpyBuilder(source)
    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=builder,
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        policy_digest=lambda learner: "0" * 64 if learner is source else "1" * 64,
    )
    token = controller.capture()
    copies = ManagedReplayCopies(ledger, builder_source=replay_builder_source)
    before = ledger.accounting()
    raw = life._copy_budget._charged
    with pytest.raises(ValueError, match="prepared native policy"):
        copies.restore(controller, token)
    learner = controller._models[0]
    assert isinstance(learner, CircadianLearner)
    copied = learner._model._replay_memory[0]
    assert copied is not row and copied.input_batch is not row.input_batch
    assert copied.target_batch is not row.target_batch
    np.testing.assert_array_equal(copied.input_batch, row.input_batch)
    assert copies.origins(controller, learner)[0].key == ("e1", "s1")
    assert ledger.accounting().records_created == before.records_created + 1
    assert ledger.accounting().live_records == before.live_records + 1
    assert ledger.accounting().invocations_started == before.invocations_started + 2
    assert life._copy_budget._charged == raw + 96 + 32
    assert tuple(vars(learner._model)) == fields
    assert (
        learner._learning_rate,
        learner._inference_steps,
        learner._inference_learning_rate,
    ) == policy
    assert runtime._budget.updates_completed == 1 and ledger.origins()[0].key == ("e1", "s1")
    assert not runtime._retired and owner._shared._runtime is runtime
    copied.input_batch[0, 0] = 0.0
    with pytest.raises(ValueError, match="integrity changed"):
        copies.origins(controller, learner)
