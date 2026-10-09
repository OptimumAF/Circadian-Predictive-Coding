"""Tiny actual checkpoint/native/inbox memos; observation grants no authority."""

from collections import deque
from contextlib import ExitStack
from hashlib import sha256
import json
import pickle
from unittest.mock import patch
from typing import Any

import numpy as np
import pytest

from src.adapters.numpy_learners import CircadianLearner, ManagedNumpyBuilder
from src.adapters.numpy_replay_copies import replay_builder_source
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_replay_copies import ManagedReplayCopies
from src.core.circadian_predictive_coding import CircadianNetworkSnapshot
from src.core.inbox_cursor import InboxCursor
from src.core.native_graph_copy import observe_graph_copies
from src.core.native_model_copy import ModelCopyLimits
from src.core.native_state_copy import observe_state_copies
from test_native_replay_capture_origins import setup
from test_native_managed_replay_origins import native_work  # noqa: F401

REPORTS: list[dict[str, Any]] = []
KINDS = [
    "native_snapshot",
    "checkpoint_capture",
    "inbox_capture",
    "native_snapshot",
    "inbox_capture",
    "checkpoint_build_state",
    "checkpoint_restore_state",
    "native_restore",
    "native_snapshot",
    "inbox_materialize",
    "native_snapshot",
    "inbox_capture",
    "native_snapshot",
]


@pytest.fixture(scope="module", autouse=True)
def checkpoint_work(tmp_path_factory):
    counts = dict(learner_snapshots=0, learner_restores=0)
    REPORTS.clear()
    with ExitStack() as stack:
        for method, key in [
            ("snapshot_state", "learner_snapshots"),
            ("restore_state", "learner_restores"),
        ]:
            original = getattr(CircadianLearner, method)

            def call(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                assert counts[_key] <= dict(learner_snapshots=7, learner_restores=2)[_key]
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(CircadianLearner, method, call))
        try:
            yield
        finally:
            assert counts == dict(learner_snapshots=7, learner_restores=2)
            (
                tmp_path_factory.mktemp("checkpoint-graph-work") / "checkpoint-graph-work.json"
            ).write_text(json.dumps({"checkpoint": counts, "traces": REPORTS}), encoding="utf8")
            REPORTS.clear()


def array_bytes(value):
    seen = set()

    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return item.nbytes
        if isinstance(item, dict):
            return sum(size(v) for v in item.values())
        if isinstance(item, (list, tuple, deque)):
            return sum(size(v) for v in item)
        if hasattr(item, "__dict__"):
            return size(vars(item))
        return 0

    return size(value)


def checkpoint(owner, runtime):
    source = runtime._candidate

    def policy(learner):
        if type(learner) is not CircadianLearner:
            raise ValueError("native policy probe requires exact CPC learner")
        return sha256(
            pickle.dumps(
                (learner._learning_rate, learner._inference_steps, learner._inference_learning_rate)
            )
        ).hexdigest()

    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=ManagedNumpyBuilder(source),
        state_digest=lambda state: sha256(pickle.dumps(state)).hexdigest(),
        policy_digest=policy,
    )
    return controller


class Trace:
    def __init__(self, runtime, controller, *, fail_restore=False):
        self.runtime, self.controller = runtime, controller
        self.fail_restore = fail_restore
        self.records = []
        self.readers = []
        self.last_native_state = None
        self.restore_input = None
        self.source_array_bytes = 0
        self.state_memo_target = None

    def observe(self, producer, kind, stage, read, lookup):
        for old_read, old_lookup in self.readers:
            with pytest.raises(ValueError, match="synchronous"):
                old_read()
            with pytest.raises(ValueError, match="synchronous"):
                old_lookup(None)
        original = read().source
        if kind.startswith("native_"):
            models = [self.runtime._candidate._model] + [m._model for m in self.controller._models]
            assert any(producer is model for model in models)
            assert type(original) is dict
            if kind == "native_snapshot":
                assert original is producer.__dict__
            else:
                assert self.restore_input is not None and original is self.restore_input.state
        elif kind.startswith("checkpoint_"):
            assert producer is self.controller and type(original) is CircadianNetworkSnapshot
            if kind == "checkpoint_capture":
                assert original.state is self.last_native_state
            else:
                pending = next(iter(self.controller._pending.values()))
                assert original is pending.view.state
        else:
            assert producer is self.runtime._inbox and type(original) is InboxCursor
            if kind == "inbox_capture":
                assert original.experiences[0] is self.runtime._inbox._experiences[("e1", "s1")]
                assert original.labels[0] is self.runtime._inbox._labels[("e1", "s1")]
                assert original.applied[0] is self.runtime._inbox._applied[("e1", "s1")]
            else:
                assert original is next(iter(self.controller._pending.values())).view.inbox
        if stage == "before_copy":
            self.source_array_bytes += array_bytes(original)
            assert self.source_array_bytes <= 1048576
        else:
            target = read().target
            assert lookup(original) is target and target is not original
            if kind.startswith("native_") or kind.startswith("checkpoint_"):
                state = original if type(original) is dict else original.state
                copied_state = target if type(target) is dict else target.state
                assert lookup(state) is copied_state
                original_row, copied_row = (
                    state["_replay_memory"][0],
                    copied_state["_replay_memory"][0],
                )
                assert lookup(original_row) is copied_row and copied_row is not original_row
                assert lookup(original_row.input_batch) is copied_row.input_batch
                assert lookup(original_row.target_batch) is copied_row.target_batch
                if kind == "native_snapshot":
                    self.last_native_state = target
                    if self.state_memo_target is not None:
                        assert self.state_memo_target is target
                if kind == "checkpoint_restore_state":
                    self.restore_input = target
            else:
                for field in ("experiences", "labels", "applied"):
                    for source_item, copied_item in zip(
                        getattr(original, field), getattr(target, field)
                    ):
                        assert copied_item is not source_item and lookup(source_item) is copied_item
                        for payload in ("features", "targets"):
                            if hasattr(source_item, payload):
                                assert lookup(getattr(source_item, payload)) is getattr(
                                    copied_item, payload
                                )
            self.records.append((kind, original, target))
            assert len(self.records) <= 32
            if self.fail_restore and kind == "native_restore":
                raise RuntimeError("observed native restore refused")
        self.readers.append((read, lookup))

    def report(self):
        REPORTS.append(
            {
                "kinds": [kind for kind, _, _ in self.records],
                "source_array_bytes": self.source_array_bytes,
                "count": len(self.records),
            }
        )


def test_should_trace_actual_checkpoint_restore_handoff_and_all_copied_inbox_receipts():
    owner, life, runtime, ledger = setup()
    controller = checkpoint(owner, runtime)
    copies = ManagedReplayCopies(ledger, builder_source=replay_builder_source)
    trace = Trace(runtime, controller)
    original_fields = tuple(vars(runtime._candidate._model))
    original_receipt = runtime._inbox._applied[("e1", "s1")]
    raw = life._copy_budget._charged

    def state_observer(stage, read, lookup):
        if stage == "copied":
            trace.state_memo_target = read().target
            assert (
                lookup(runtime._candidate._model._replay_memory[0])
                is trace.state_memo_target["_replay_memory"][0]
            )

    with observe_graph_copies(trace.observe, ModelCopyLimits(32, 64, 2048)):
        with observe_state_copies(
            runtime._candidate._model.__dict__, state_observer, ModelCopyLimits(1, 2, 8)
        ):
            token = controller.capture()
        trace.state_memo_target = None
        prepared = copies.restore(controller, token)
    assert [kind for kind, _, _ in trace.records] == KINDS
    assert owner._shared._runtime is prepared and runtime._retired
    assert runtime._inbox._historical_completed_updates == 1
    assert prepared._budget is runtime._budget and prepared._inbox._clock is runtime._inbox._clock
    assert prepared._payload_lineage is runtime._payload_lineage
    assert prepared._candidate is controller._models[0]
    assert tuple(vars(prepared._candidate._model)) == original_fields
    owned_cursor = next(target for kind, _, target in trace.records if kind == "inbox_materialize")
    assert prepared._inbox._experiences[("e1", "s1")] is owned_cursor.experiences[0]
    assert prepared._inbox._labels[("e1", "s1")] is owned_cursor.labels[0]
    assert prepared._inbox._applied[("e1", "s1")] is owned_cursor.applied[0]
    assert owned_cursor.applied[0] is not original_receipt
    assert prepared._budget.updates_completed == 1 and life._copy_budget._charged > raw
    assert not controller._pending
    # Observing the new identities does not renew or transfer original authority.
    with pytest.raises(ValueError, match="original runtime/model"):
        ledger.origins()
    with pytest.raises(ValueError, match="original runtime/model|builder differs"):
        copies.origins(controller, prepared._candidate)
    trace.report()


def test_should_preserve_original_owner_and_retained_fork_after_observed_native_restore_fault():
    owner, life, runtime, ledger = setup()
    controller = checkpoint(owner, runtime)
    copies = ManagedReplayCopies(ledger, builder_source=replay_builder_source)
    trace = Trace(runtime, controller, fail_restore=True)
    before = ledger.accounting()
    raw = life._copy_budget._charged
    with observe_graph_copies(trace.observe, ModelCopyLimits(32, 64, 2048)) as sequence:
        token = controller.capture()
        with pytest.raises(RuntimeError, match="restore refused"):
            copies.restore(controller, token)
        assert sequence._window is not None and sequence._window._failed
    assert [kind for kind, _, _ in trace.records] == KINDS[:8]
    assert owner._shared._runtime is runtime and not runtime._retired
    assert runtime._inbox._historical_completed_updates is None
    learner = controller._models[0]
    attempted_state = trace.records[-1][2]
    assert learner._model._replay_memory[0] is not attempted_state["_replay_memory"][0]
    assert copies.origins(controller, learner)[0].key == ("e1", "s1")
    assert ledger.origins()[0].key == ("e1", "s1")
    assert ledger.accounting().records_created == before.records_created + 1
    assert life._copy_budget._charged > raw and controller._attempts == 1
    assert token in controller._pending
    trace.report()
