"""One prospectively budgeted trained CPC graph and retained checkpoint fork.

Constructor-copy lineage qualifies this retained failed preparation only. Its
unwitnessed pending snapshot is discarded normally; replay is never cleared.
No native operation runs after setup or inside complete capture.
"""

from hashlib import sha256
import json
import pickle

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import CircadianLearner, ManagedNumpyBuilder
from src.adapters.numpy_replay_copies import replay_builder_source
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_replay_copies import ManagedReplayCopies
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from test_managed_composite_capture import LIMITS
from test_native_replay_capture_origins import setup
from test_numpy_retained_capture import assert_owned_graph, holder, values


def _digest(state):
    return sha256(pickle.dumps(state)).hexdigest()


def _setup_retained_graph():
    owner, life, runtime, ledger = setup()
    source = runtime._candidate
    builder = ManagedNumpyBuilder(source)
    controller = CandidateCheckpointController(
        owner._shared,
        build_learner=builder,
        state_digest=_digest,
        policy_digest=lambda learner: "0" * 64 if learner is source else "1" * 64,
    )
    token = controller.capture()
    copies = ManagedReplayCopies(ledger, builder_source=replay_builder_source)
    with pytest.raises(ValueError, match="prepared native policy"):
        copies.restore(controller, token)
    assert len(controller._models) == controller._attempts == 1
    # Snapshot lineage is a separate unfinished path. Ordinary discard does not
    # refund any charge or remove the observed failed-preparation model.
    controller.discard(token)
    assert not controller._pending
    return dict(
        owner=owner,
        life=life,
        runtime=runtime,
        ledger=ledger,
        copies=copies,
        controller=controller,
        target=controller._models[0],
        builder=builder,
    )


def _forbidden(*args, **kwargs):
    pytest.fail("native operation after bounded setup")


def test_should_capture_original_retained_nonempty_checkpoint_fork_and_keep_refusals(
    monkeypatch, record_testsuite_property
):
    graph = _setup_retained_graph()
    owner, life, runtime, ledger, copies = (
        graph[key] for key in ("owner", "life", "runtime", "ledger", "copies")
    )
    target, controller = graph["target"], graph["controller"]
    models = dict(
        actor=runtime.actor._learner._model,
        candidate=runtime._candidate._model,
        retained=target._model,
    )
    native_before = {name: pickle.dumps(vars(model)) for name, model in models.items()}
    assert all(len(raw) < 65536 for raw in native_before.values())
    source_row = models["candidate"]._replay_memory[0]
    retained_row = models["retained"]._replay_memory[0]
    assert source_row is not retained_row
    assert not np.shares_memory(source_row.input_batch, retained_row.input_batch)
    assert runtime._budget.updates_completed == 1
    identities = tuple(
        id(item)
        for item in (
            owner,
            life,
            runtime,
            ledger,
            copies,
            controller,
            target,
            graph["builder"],
            runtime._budget,
            runtime._inbox._clock,
            life._copy_budget,
        )
    )
    for kind in (CircadianLearner, CircadianPredictiveCodingNetwork, PredictiveCodingNetwork):
        for name in (
            "__init__",
            "fork",
            "train_batch",
            "train_epoch",
            "predict",
            "predict_proba",
            "snapshot_state",
            "restore_state",
            "infer_hidden_states",
            "_infer_hidden_states",
            "sleep_event",
            "_execute_sleep_event",
        ):
            if hasattr(kind, name):
                monkeypatch.setattr(kind, name, _forbidden)

    def capture(include_copies=True):
        return adapter.capture_numpy_managed_composite(
            owner,
            limits=LIMITS,
            replay_origins=ledger,
            replay_copies=copies if include_copies else None,
        )

    raw_before = life.payload_byte_snapshot()
    accounting_before = ledger.accounting()
    with monkeypatch.context() as guard:
        guard.setattr(adapter, "project_numpy_native", _forbidden)
        guard.setattr(adapter, "deepcopy", _forbidden)
        with pytest.raises(ValueError, match="holder lineage"):
            capture(False)
    assert life._copy_budget._charged == raw_before.charged_bytes
    assert ledger.accounting().invocations_started == accounting_before.invocations_started + 1

    passes = []
    original_copy = adapter.copy_numpy_composite

    def copying(state, limits, memo):
        assert life._copy_budget._charged == (
            raw_before.charged_bytes + raw_before.observed_retained_bytes
        )
        passes.append(memo)
        return original_copy(state, limits, memo)

    with monkeypatch.context() as observe:
        observe.setattr(adapter, "copy_numpy_composite", copying)
        captured = capture()
    assert len(passes) == 2 and passes[0] is passes[1]
    graph_for_holder = dict(life=life)
    candidate = values(holder(captured, graph_for_holder, runtime)["_candidate"])["_model"]
    checkpoint = holder(captured, graph_for_holder, controller)
    retained = values(checkpoint["_models"][0])["_model"]
    actor = values(holder(captured, graph_for_holder, runtime.actor)["_learner"])["_model"]
    for native, model in (
        (candidate, models["candidate"]),
        (retained, models["retained"]),
        (actor, models["actor"]),
    ):
        assert native.state.keys() == vars(model).keys()
        assert_owned_graph(native.state, vars(model))
        assert native.config is native.state["config"]
    for original, detached in (
        (source_row, candidate.state["_replay_memory"][0]),
        (retained_row, retained.state["_replay_memory"][0]),
    ):
        assert passes[0][id(original)] is detached
        assert passes[0][id(original.input_batch)] is detached.input_batch
        assert passes[0][id(original.target_batch)] is detached.target_batch
    assert candidate.state["_replay_memory"][0] is not retained.state["_replay_memory"][0]
    assert not checkpoint["_pending"]
    after_positive = life.payload_byte_snapshot()
    assert (
        after_positive.charged_bytes
        == raw_before.charged_bytes + captured.state.retained_payload_bytes
    )
    assert after_positive.observed_retained_bytes == raw_before.observed_retained_bytes
    after_accounting = ledger.accounting()
    assert after_accounting.records_created == accounting_before.records_created
    assert after_accounting.invocations_started == accounting_before.invocations_started + 2
    assert (
        after_accounting.metadata_bytes_charged == accounting_before.metadata_bytes_charged + 2048
    )
    assert native_before == {name: pickle.dumps(vars(model)) for name, model in models.items()}

    # A legitimate checkpoint witness cannot authorize another holder role.
    with monkeypatch.context() as borrowed:
        borrowed.setattr(runtime.actor, "_learner", target)
        borrowed.setattr(adapter, "project_numpy_native", _forbidden)
        borrowed.setattr(adapter, "deepcopy", _forbidden)
        with pytest.raises(ValueError):
            capture()
    assert life._copy_budget._charged == after_positive.charged_bytes

    late_passes = []
    saved = retained_row.input_batch[0, 0]

    def changed_after_copy(state, limits, memo):
        result = original_copy(state, limits, memo)
        late_passes.append(memo)
        retained_row.input_batch[0, 0] = 0.125
        return result

    try:
        with monkeypatch.context() as late:
            late.setattr(adapter, "copy_numpy_composite", changed_after_copy)
            with pytest.raises(ValueError, match="integrity"):
                capture()
    finally:
        retained_row.input_batch[0, 0] = saved
    assert len(late_passes) == 1
    assert life._copy_budget._charged == (
        after_positive.charged_bytes + raw_before.observed_retained_bytes
    )
    assert runtime._budget.updates_completed == 1 and not runtime._retired
    assert owner._shared._runtime is runtime and controller._models == [target]
    assert native_before == {name: pickle.dumps(vars(model)) for name, model in models.items()}
    assert identities == tuple(
        id(item)
        for item in (
            owner,
            life,
            runtime,
            ledger,
            copies,
            controller,
            target,
            graph["builder"],
            runtime._budget,
            runtime._inbox._clock,
            life._copy_budget,
        )
    )
    record_testsuite_property(
        "retained_replay_observations",
        json.dumps(
            dict(
                native_before={
                    name: sha256(raw).hexdigest() for name, raw in native_before.items()
                },
                native_after={
                    name: sha256(pickle.dumps(vars(model))).hexdigest()
                    for name, model in models.items()
                },
                complete_native_fields={
                    name: sorted(vars(model)) for name, model in models.items()
                },
                original_identities_preserved=True,
                copy_before=raw_before.charged_bytes,
                copy_after_positive=after_positive.charged_bytes,
                copy_after_late_failure=life._copy_budget._charged,
                retained_bytes=raw_before.observed_retained_bytes,
                native_updates=runtime._budget.updates_completed,
                capture_attempts=4,
                graph_copy_passes=3,
                qualified_checkpoint_replay_rows=1,
                copied_snapshot_qualified=False,
            ),
            sort_keys=True,
        ),
    )
