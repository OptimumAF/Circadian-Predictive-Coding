"""One original trained graph and genuinely issued nonempty checkpoint snapshot.

Original manager.capture observes the entire actual native/inbox copy chain.
No native work runs after that bounded setup or during complete capture.
"""

from hashlib import sha256
import json
import pickle

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from test_managed_composite_capture import LIMITS
from test_native_managed_replay_checkpoints import setup
from test_numpy_retained_capture import assert_owned_graph, holder, values


def _setup_snapshot_graph():
    owner, life, runtime, ledger, controller, manager = setup()
    token = manager.capture(controller)
    pending = controller._pending[token]
    assert pending.view is manager._checkpoints[id(token)].view()
    assert pending.view.state.state["_replay_memory"]
    return dict(
        owner=owner,
        life=life,
        runtime=runtime,
        ledger=ledger,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=pending.view,
        snapshot=pending.view.state,
    )


def _identities(graph):
    runtime = graph["runtime"]
    return tuple(
        id(value)
        for value in (
            graph["owner"],
            graph["life"],
            runtime,
            graph["ledger"],
            graph["controller"],
            graph["manager"],
            graph["manager"]._copies,
            graph["token"],
            graph["pending"],
            graph["view"],
            graph["snapshot"],
            graph["view"].inbox,
            runtime._candidate,
            runtime.actor,
            runtime._budget,
            runtime._inbox._clock,
            graph["life"]._copy_budget,
        )
    )


def _forbidden(*args, **kwargs):
    pytest.fail("native work or unqualified projection/copy after bounded setup")


def test_should_capture_actual_issued_snapshot_and_preserve_original_paid_boundaries(
    monkeypatch, record_testsuite_property
):
    graph = _setup_snapshot_graph()
    owner, life, runtime, ledger, controller, manager, token, pending, view = (
        graph[key]
        for key in (
            "owner",
            "life",
            "runtime",
            "ledger",
            "controller",
            "manager",
            "token",
            "pending",
            "view",
        )
    )
    models = dict(actor=runtime.actor._learner._model, candidate=runtime._candidate._model)
    native_before = {name: pickle.dumps(vars(model)) for name, model in models.items()}
    snapshot_before = pickle.dumps(view)
    assert len(snapshot_before) + sum(len(raw) for raw in native_before.values()) < 65536
    original_ids = _identities(graph)
    source_row = models["candidate"]._replay_memory[0]
    pending_row = view.state.state["_replay_memory"][0]
    assert pending_row is not source_row
    assert not np.shares_memory(pending_row.input_batch, source_row.input_batch)
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

    def capture(*, checkpoints=True, copies=False):
        return adapter.capture_numpy_managed_composite(
            owner,
            limits=LIMITS,
            replay_origins=ledger,
            replay_copies=manager._copies if copies else None,
            replay_checkpoints=manager if checkpoints else None,
        )

    before = life.payload_byte_snapshot()
    accounting = ledger.accounting()
    for constructor_only in (False, True):
        with monkeypatch.context() as refused:
            refused.setattr(adapter, "project_numpy_native", _forbidden)
            refused.setattr(adapter, "deepcopy", _forbidden)
            with pytest.raises(ValueError):
                capture(checkpoints=False, copies=constructor_only)
        assert life._copy_budget._charged == before.charged_bytes

    passes = []
    original_copy = adapter.copy_numpy_composite

    def copying(state, limits, memo):
        assert life._copy_budget._charged == before.charged_bytes + before.observed_retained_bytes
        passes.append(memo)
        return original_copy(state, limits, memo)

    with monkeypatch.context() as observed:
        observed.setattr(adapter, "copy_numpy_composite", copying)
        captured = capture()
    assert len(passes) == 2 and passes[0] is passes[1]
    copied_controller = holder(captured, dict(life=life), controller)
    assert not copied_controller["_models"] and len(copied_controller["_pending"]) == 1
    copied_view = values(copied_controller["_pending"][0][1])["view"]
    assert_owned_graph(copied_view, view)
    detached_row = copied_view.state.state["_replay_memory"][0]
    assert passes[0][id(pending_row)] is detached_row
    assert passes[0][id(pending_row.input_batch)] is detached_row.input_batch
    assert passes[0][id(pending_row.target_batch)] is detached_row.target_batch
    assert detached_row is not pending_row and detached_row is not source_row
    for original_holder, field, original_model in (
        (runtime, "_candidate", models["candidate"]),
        (runtime.actor, "_learner", models["actor"]),
    ):
        native = values(holder(captured, dict(life=life), original_holder)[field])["_model"]
        assert_owned_graph(native.state, vars(original_model))
        assert native.config is native.state["config"]
    positive = life.payload_byte_snapshot()
    assert positive.charged_bytes == before.charged_bytes + captured.state.retained_payload_bytes
    assert positive.observed_retained_bytes == before.observed_retained_bytes
    paid = ledger.accounting()
    assert paid.records_created == accounting.records_created
    assert paid.invocations_started == accounting.invocations_started + 3
    assert paid.metadata_bytes_charged == accounting.metadata_bytes_charged + 3072
    assert controller._pending[token] is pending and pending.view is view
    assert snapshot_before == pickle.dumps(view)
    assert native_before == {name: pickle.dumps(vars(model)) for name, model in models.items()}

    with monkeypatch.context() as borrowed:
        borrowed.setattr(runtime.actor, "_learner", view.state)
        borrowed.setattr(adapter, "project_numpy_native", _forbidden)
        borrowed.setattr(adapter, "deepcopy", _forbidden)
        with pytest.raises(ValueError):
            capture()
    assert life._copy_budget._charged == positive.charged_bytes

    late_charges = []
    for change in ("payload", "membership"):
        late_passes = []
        saved = pending_row.input_batch[0, 0]

        def changed_after_copy(state, limits, memo):
            result = original_copy(state, limits, memo)
            late_passes.append(memo)
            if change == "payload":
                pending_row.input_batch[0, 0] = 0.125
            else:
                del controller._pending[token]
            return result

        original_charge = life._copy_budget._charged
        try:
            with monkeypatch.context() as late:
                late.setattr(adapter, "copy_numpy_composite", changed_after_copy)
                with pytest.raises(ValueError):
                    capture()
        finally:
            pending_row.input_batch[0, 0] = saved
            controller._pending[token] = pending
        assert len(late_passes) == 1
        assert life._copy_budget._charged == original_charge + before.observed_retained_bytes
        late_charges.append(life._copy_budget._charged)
    assert _identities(graph) == original_ids
    assert snapshot_before == pickle.dumps(view)
    assert native_before == {name: pickle.dumps(vars(model)) for name, model in models.items()}
    assert runtime._budget.updates_completed == 1 and not runtime._retired
    assert controller._attempts == 0 and not controller._models
    assert owner._shared._runtime is runtime and controller._pending[token] is pending
    record_testsuite_property(
        "issued_snapshot_observations",
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
                snapshot_before=sha256(snapshot_before).hexdigest(),
                snapshot_after=sha256(pickle.dumps(view)).hexdigest(),
                original_identities_preserved=len(original_ids),
                original_token_still_pending=True,
                native_updates=runtime._budget.updates_completed,
                copy_before=before.charged_bytes,
                copy_after_positive=positive.charged_bytes,
                late_charges=late_charges,
                retained_bytes=before.observed_retained_bytes,
                original_copy_limit=before.limits.max_lifetime_owned_bytes,
                capture_attempts=6,
                graph_copy_passes=4,
                qualified_issued_snapshot_rows=1,
                native_restoration_qualified=False,
            ),
            sort_keys=True,
        ),
    )
