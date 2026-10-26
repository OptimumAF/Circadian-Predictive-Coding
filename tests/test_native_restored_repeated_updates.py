"""Three original managed wakes, native eviction, then complete retained capture."""

from hashlib import sha256
import json
import pickle
from typing import Any
from weakref import ref

import numpy as np
import pytest

import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import CircadianLearner
from src.core.circadian_predictive_coding import CircadianPredictiveCodingNetwork
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.predictive_coding import PredictiveCodingNetwork
from test_managed_experience import declaration
from test_native_managed_replay_checkpoints import setup
from test_native_restored_replay_capture import _identities
from test_numpy_retained_capture import LIMITS, assert_owned_graph, holder, values


def _setup_repeated_graph():
    owner, life, old, ledger, controller, manager = setup(
        retain_inbox_origins=True,
        retain_capture_history=True,
        live_records=32,
        max_updates=3,
        metadata_bytes=256 * 1024,
    )
    token = manager.capture(controller)
    pending = controller._pending[token]
    view = pending.view
    new = manager.restore(controller, token)
    initial_row = ref(new._candidate._model._replay_memory[0])
    retired_before = pickle.dumps(vars(old._candidate._model))
    old_view = pickle.dumps(view)
    metadata_before = ledger.accounting()
    owner.declare(declaration("s2", "person-2"))
    owner.record_experience(
        Experience(
            "s2",
            "e1",
            4,
            "actor-0",
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival(
            "label-s2",
            "s2",
            "e1",
            6,
            "actor-0",
            np.array([[1.0], [0.0]], dtype=np.float64),
        )
    )
    new._inbox._clock.advance_to(6)
    owner._shared._sharing.resume()
    poll = ledger.train_ready()
    owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 2
    assert old._retired and not new._retired and owner._shared._runtime is new
    assert old._inbox._historical_completed_updates == 1
    assert new._budget.updates_completed == new._candidate._model._epoch_count == 2
    assert old._candidate._model._epoch_count == 1
    retained = new._candidate._model._replay_memory
    assert len(retained) == 2 and initial_row() is None
    assert all(row.input_batch.shape == (1, 3) for row in retained)
    assert all(row.target_batch.shape == (1, 1) for row in retained)
    del retained  # Do not keep an evicted native row alive through the test's old deque.
    previous_success = ref(manager._current_success)
    # The declared default content-hash policy evicts row 1 for these fixed inputs.
    second_row = ref(new._candidate._model._replay_memory[1])
    owner.declare(declaration("s3", "person-3"))
    owner.record_experience(
        Experience(
            "s3",
            "e1",
            8,
            "actor-0",
            np.array([[1.0, 1.0, 0.0]], dtype=np.float64),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival("label-s3", "s3", "e1", 10, "actor-0", np.array([[1.0]], dtype=np.float64))
    )
    new._inbox._clock.advance_to(10)
    owner._shared._sharing.resume()
    poll = ledger.train_ready()
    owner._shared._sharing.pause()
    assert len(poll.updates) == 1 and poll.updates[0].update_number == 3
    assert new._budget.updates_completed == new._candidate._model._epoch_count == 3
    assert old._candidate._model._epoch_count == 1
    assert len(new._candidate._model._replay_memory) == 2 and second_row() is None
    assert previous_success() is None
    assert retired_before == pickle.dumps(vars(old._candidate._model))
    assert old_view == pickle.dumps(view)
    return dict(
        owner=owner,
        life=life,
        old=old,
        new=new,
        ledger=ledger,
        controller=controller,
        manager=manager,
        token=token,
        pending=pending,
        view=view,
        metadata_before_update=metadata_before,
        committed=poll.updates[0],
        evicted_current_row=initial_row,
        previous_success=previous_success,
        evicted_second_row=second_row,
    )


def _forbidden(*args, **kwargs):
    pytest.fail("operational native callback or unqualified projection/copy after setup")


def test_should_capture_successive_real_updates_and_keep_original_retired_history(
    monkeypatch, record_testsuite_property
):
    graph = _setup_repeated_graph()
    owner, life, old, new, ledger, controller, manager = (
        graph[key] for key in ("owner", "life", "old", "new", "ledger", "controller", "manager")
    )
    identities = _identities(graph)
    binding = manager._update_binding
    assert ledger._capture_update_binding is ledger._original_capture_update_binding is binding
    binding_identity = id(binding)
    success = manager._current_success
    assert success is manager._current_success_sealed and success is not None
    metadata = json.dumps(
        success.metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf8")
    assert success.charged_bytes == len(metadata) + 1024
    assert (
        ledger.accounting().metadata_bytes_charged
        >= graph["metadata_before_update"].metadata_bytes_charged + success.charged_bytes
    )
    models = dict(
        actor=new.actor._learner._model,
        retired=old._candidate._model,
        current=new._candidate._model,
    )
    native_before = {name: pickle.dumps(vars(model)) for name, model in models.items()}
    view_before = pickle.dumps(graph["view"])
    assert sum(map(len, native_before.values())) + len(view_before) < 65536
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
            replay_checkpoints=manager if checkpoints else None,
            replay_copies=manager._copies if copies else None,
        )

    before = life.payload_byte_snapshot()
    accounting = ledger.accounting()
    assert before.observed_retained_bytes == 256
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
    detached_current = holder(captured, dict(life=life), new)
    detached_cp = holder(captured, dict(life=life), controller)
    assert detached_cp["_models"][0] is detached_current["_candidate"]
    assert not detached_cp["_pending"]
    for runtime, label in ((old, "retired"), (new, "current")):
        detached = holder(captured, dict(life=life), runtime)
        assert detached["_retired"] is (runtime is old)
        native = values(detached["_candidate"])["_model"]
        assert_owned_graph(native.state, vars(models[label]))
        assert native.config is native.state["config"]
        inbox = values(detached["_inbox"])
        for name in ("_experiences", "_labels", "_applied", "_erased"):
            assert_owned_graph(inbox[name], getattr(runtime._inbox, name))
        assert inbox["_historical_completed_updates"] == (1 if runtime is old else None)
        for row, copied in zip(
            runtime._candidate._model._replay_memory, native.state["_replay_memory"]
        ):
            assert passes[0][id(row)] is copied
            for name in ("input_batch", "target_batch"):
                assert passes[0][id(getattr(row, name))] is getattr(copied, name)
    actor = values(holder(captured, dict(life=life), new.actor)["_learner"])["_model"]
    assert_owned_graph(actor.state, vars(models["actor"]))
    after = life.payload_byte_snapshot()
    assert after.charged_bytes == before.charged_bytes + captured.state.retained_payload_bytes
    assert after.observed_retained_bytes == before.observed_retained_bytes
    assert graph["evicted_current_row"]() is None
    assert ledger.accounting().invocations_started == accounting.invocations_started + 3
    with monkeypatch.context() as alias:
        alias.setattr(new.actor, "_learner", new._candidate)
        alias.setattr(adapter, "project_numpy_native", _forbidden)
        alias.setattr(adapter, "deepcopy", _forbidden)
        with pytest.raises(ValueError):
            capture()
    assert life._copy_budget._charged == after.charged_bytes
    row = models["current"]._replay_memory[0]
    saved = row.input_batch[0, 0]
    late = []

    def changed(state, limits, memo):
        result = original_copy(state, limits, memo)
        late.append(memo)
        row.input_batch[0, 0] = 0.125
        return result

    try:
        with monkeypatch.context() as refused:
            refused.setattr(adapter, "copy_numpy_composite", changed)
            with pytest.raises(ValueError):
                capture()
    finally:
        row.input_batch[0, 0] = saved
    assert len(late) == 1
    charged_final = life._copy_budget._charged
    assert charged_final == after.charged_bytes + before.observed_retained_bytes
    # Native eviction does not release the original retired source's consent.
    owner._revoked_keys.add(("e1", "s1"))
    with monkeypatch.context() as refused:
        refused.setattr(adapter, "project_numpy_native", _forbidden)
        refused.setattr(adapter, "deepcopy", _forbidden)
        with pytest.raises(ValueError):
            capture()
    assert life._copy_budget._charged == charged_final
    assert _identities(graph) == identities and id(manager._update_binding) == binding_identity
    assert manager._current_success is success
    assert native_before == {name: pickle.dumps(vars(model)) for name, model in models.items()}
    assert view_before == pickle.dumps(graph["view"])
    assert old._inbox._historical_completed_updates == 1 and new._budget.updates_completed == 3
    record_testsuite_property(
        "restored_repeated_observations",
        json.dumps(
            dict(
                original_identities_preserved=len(identities),
                original_binding_identity_preserved=True,
                complete_native_fields={
                    name: sorted(vars(model)) for name, model in models.items()
                },
                native_before={
                    name: sha256(raw).hexdigest() for name, raw in native_before.items()
                },
                native_after={
                    name: sha256(pickle.dumps(vars(model))).hexdigest()
                    for name, model in models.items()
                },
                consumed_view_unchanged=True,
                original_copy_limit=4096,
                historical_updates=1,
                current_updates=3,
                evicted_current_row_expired=True,
                evicted_second_row_expired=graph["evicted_second_row"]() is None,
                previous_success_expired=graph["previous_success"]() is None,
                current_retained_rows=len(models["current"]._replay_memory),
                success_metadata_bytes=len(metadata),
                success_metadata_charge=success.charged_bytes,
                retained_bytes=before.observed_retained_bytes,
                copy_before=before.charged_bytes,
                copy_after=after.charged_bytes,
                copy_after_late_failure=charged_final,
                old_consent_refusal=True,
                capture_attempts=6,
                copy_passes=3,
            ),
            sort_keys=True,
        ),
    )
