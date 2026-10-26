"""One genuine managed restore and full current/retired replay capture."""

from hashlib import sha256
import json
import pickle
from typing import Any

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.predictive_coding import PredictiveCodingNetwork
from test_native_managed_replay_checkpoints import setup
from test_numpy_retained_capture import LIMITS, assert_owned_graph, holder, values


def _setup_restored_graph():
    owner, life, original, ledger, controller, manager = setup(retain_capture_history=True)
    token = manager.capture(controller)
    pending = controller._pending[token]
    view = pending.view
    before = ledger.accounting()
    prepared = manager.restore(controller, token)
    assert owner._shared._runtime is prepared and original._retired
    assert original._inbox._historical_completed_updates == 1
    assert not controller._pending and controller._models == [prepared._candidate]
    return dict(
        owner=owner,
        life=life,
        old=original,
        new=prepared,
        ledger=ledger,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=view,
        before_publication=before,
    )


def _identities(graph):
    current = graph["new"]
    return tuple(
        id(value)
        for value in (
            graph["owner"],
            graph["life"],
            graph["old"],
            current,
            graph["ledger"],
            graph["controller"],
            graph["manager"],
            graph["manager"]._copies,
            graph["token"],
            graph["pending"],
            graph["view"],
            current.actor,
            current._candidate,
            graph["old"]._candidate,
            current._inbox,
            graph["old"]._inbox,
            current._budget,
            current._inbox._clock,
            graph["life"]._copy_budget,
        )
    )


def _forbidden(*args, **kwargs):
    pytest.fail("native operation or unqualified projection/copy after setup")


def test_should_capture_genuine_restored_current_and_retired_graph_without_renewing_limits(
    monkeypatch, record_testsuite_property
):
    graph = _setup_restored_graph()
    owner, life, old, new, ledger, controller, manager = (
        graph[key] for key in ("owner", "life", "old", "new", "ledger", "controller", "manager")
    )
    models = dict(
        actor=new.actor._learner._model,
        retired=old._candidate._model,
        current=new._candidate._model,
    )
    before_bytes = {name: pickle.dumps(vars(model)) for name, model in models.items()}
    original_view = pickle.dumps(graph["view"])
    original_ids = _identities(graph)
    publication = manager._publications[id(models["retired"])]
    publication_charge = 1024 + len(
        json.dumps(
            publication.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )
    assert publication.charged_bytes == publication_charge > 1024
    assert (
        ledger.accounting().metadata_bytes_charged
        >= graph["before_publication"].metadata_bytes_charged + publication_charge
    )
    old_row, new_row = models["retired"]._replay_memory[0], models["current"]._replay_memory[0]
    assert old_row is not new_row
    for name in ("input_batch", "target_batch"):
        assert not np.shares_memory(getattr(old_row, name), getattr(new_row, name))
    assert sum(map(len, before_bytes.values())) + len(pickle.dumps(graph["view"])) < 65536
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
    for copies in (False, True):
        with monkeypatch.context() as refused:
            refused.setattr(adapter, "project_numpy_native", _forbidden)
            refused.setattr(adapter, "deepcopy", _forbidden)
            with pytest.raises(ValueError):
                capture(checkpoints=False, copies=copies)
        assert life._copy_budget._charged == before.charged_bytes

    original_copy = adapter.copy_numpy_composite
    passes: list[dict[int, Any]] = []

    def copying(state, limits, memo):
        assert life._copy_budget._charged == before.charged_bytes + before.observed_retained_bytes
        passes.append(memo)
        return original_copy(state, limits, memo)

    with monkeypatch.context() as observed:
        observed.setattr(adapter, "copy_numpy_composite", copying)
        captured = capture()
    assert len(passes) == 2 and passes[0] is passes[1]
    detached_old = holder(captured, dict(life=life), old)
    detached_new = holder(captured, dict(life=life), new)
    detached_cp = holder(captured, dict(life=life), controller)
    assert detached_old["_retired"] is True and detached_new["_retired"] is False
    assert len(detached_cp["_models"]) == 1 and not detached_cp["_pending"]
    assert detached_cp["_models"][0] is detached_new["_candidate"]
    for runtime, detached, label in (
        (old, detached_old, "retired"),
        (new, detached_new, "current"),
    ):
        native = values(detached["_candidate"])["_model"]
        assert_owned_graph(native.state, vars(models[label]))
        assert native.config is native.state["config"]
        inbox = values(detached["_inbox"])
        for name in ("_experiences", "_labels", "_applied", "_erased"):
            assert_owned_graph(inbox[name], getattr(runtime._inbox, name))
        row = runtime._candidate._model._replay_memory[0]
        copied = native.state["_replay_memory"][0]
        assert passes[0][id(row)] is copied
        for name in ("input_batch", "target_batch"):
            assert passes[0][id(getattr(row, name))] is getattr(copied, name)
    actor = values(holder(captured, dict(life=life), new.actor)["_learner"])["_model"]
    assert_owned_graph(actor.state, vars(models["actor"]))
    after = life.payload_byte_snapshot()
    assert after.charged_bytes == before.charged_bytes + captured.state.retained_payload_bytes
    assert after.observed_retained_bytes == before.observed_retained_bytes
    paid = ledger.accounting()
    assert paid.invocations_started == accounting.invocations_started + 3
    assert paid.records_created == accounting.records_created
    assert paid.metadata_bytes_charged == accounting.metadata_bytes_charged + 3072

    for learner in (old._candidate, new._candidate):
        with monkeypatch.context() as alias:
            alias.setattr(new.actor, "_learner", learner)
            alias.setattr(adapter, "project_numpy_native", _forbidden)
            alias.setattr(adapter, "deepcopy", _forbidden)
            with pytest.raises(ValueError):
                capture()
        assert life._copy_budget._charged == after.charged_bytes

    late = []
    saved = old_row.input_batch[0, 0]

    def changed_after_copy(state, limits, memo):
        result = original_copy(state, limits, memo)
        late.append(memo)
        old_row.input_batch[0, 0] = 0.125
        return result

    try:
        with monkeypatch.context() as changed:
            changed.setattr(adapter, "copy_numpy_composite", changed_after_copy)
            with pytest.raises(ValueError):
                capture()
    finally:
        old_row.input_batch[0, 0] = saved
    assert len(late) == 1
    final = life.payload_byte_snapshot()
    assert final.charged_bytes == after.charged_bytes + before.observed_retained_bytes
    assert final.observed_retained_bytes == before.observed_retained_bytes
    assert _identities(graph) == original_ids
    assert before_bytes == {name: pickle.dumps(vars(model)) for name, model in models.items()}
    assert original_view == pickle.dumps(graph["view"])
    assert old._retired and not new._retired and owner._shared._runtime is new
    assert old._inbox._historical_completed_updates == new._budget.updates_completed == 1
    assert controller._attempts == len(controller._models) == 1 and not controller._pending
    record_testsuite_property(
        "restored_capture_observations",
        json.dumps(
            dict(
                native_before={name: sha256(raw).hexdigest() for name, raw in before_bytes.items()},
                native_after={
                    name: sha256(pickle.dumps(vars(model))).hexdigest()
                    for name, model in models.items()
                },
                complete_native_fields={
                    name: sorted(vars(model)) for name, model in models.items()
                },
                original_identities_preserved=len(original_ids),
                historical_updates=1,
                current_updates=1,
                preparation_attempts=1,
                retired=True,
                current_alias_in_checkpoint=True,
                copy_before=before.charged_bytes,
                copy_after_positive=after.charged_bytes,
                copy_after_late_failure=final.charged_bytes,
                retained_bytes=before.observed_retained_bytes,
                original_copy_limit=4096,
                capture_attempts=6,
                copy_passes=3,
                publication_metadata_before=graph["before_publication"].metadata_bytes_charged,
                publication_metadata_after=accounting.metadata_bytes_charged,
                publication_witness_charge=publication_charge,
                complete_capture_is_data_only=True,
            ),
            sort_keys=True,
        ),
    )
