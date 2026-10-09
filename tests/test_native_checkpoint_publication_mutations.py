"""Eight separately budgeted native refusals from the LAST opaque fingerprint."""

import inspect
from _thread import LockType
from threading import Lock
from typing import Any

import pytest

from src.adapters.numpy_replay_origins import replay_payload_fingerprint
from test_native_managed_replay_checkpoints import (
    assert_unpublished,
    observe_native_work,
    setup,
)


@pytest.fixture(scope="module", autouse=True)
def bounded_mutation_work(tmp_path_factory):
    limits = dict(
        models=8,
        learners=32,
        forks=24,
        wakes=8,
        steps=16,
        stores=8,
        predicts=8,
        array_copies=40,
        graph_events=104,
        source_array_bytes=1024 * 1024,
        captures=8,
        preparations=8,
        native_restores=8,
        handoffs=0,
    )
    yield from observe_native_work(tmp_path_factory, limits)


@pytest.mark.parametrize(
    "mutation",
    [
        "replay_array",
        "weights",
        "rng",
        "inbox_array",
        "event_index",
        "work_progress",
        "custom_sequence",
        "held_gates",
    ],
)
def test_should_refuse_last_callback_mutation_before_publication(mutation):
    assert_last_mutation_refused(mutation)


def assert_last_mutation_refused(mutation):
    armed = [False]
    visits = [0]
    current_frame: list[Any] = [None]
    fired = [False]
    callbacks: list[str] = []
    acquired_gates: list[LockType] = []
    replacements: list[tuple[Any, str, LockType]] = []

    class ForbiddenSequence:
        def __len__(self):
            callbacks.append("len")
            raise AssertionError("custom sequence was traversed")

        def __iter__(self):
            callbacks.append("iter")
            raise AssertionError("custom sequence was traversed")

    def fingerprint(snapshot, maximum):
        result = replay_payload_fingerprint(snapshot, maximum)
        if not armed[0] or fired[0]:
            return result
        frame = inspect.currentframe()
        publication = active_current = None
        while frame is not None:
            if frame.f_code.co_name == "_current":
                active_current = frame
            elif frame.f_code.co_name == "_publication_lease":
                publication = frame
                break
            frame = frame.f_back
        if publication is None or active_current is None:
            return result
        if active_current is not current_frame[0]:
            visits[0] += 1
            current_frame[0] = active_current
        # Three actual final-lease _current calls. Mutate while returning the
        # correct fingerprint for the original source after the third begins.
        if visits[0] != 3:
            return result
        fired[0] = True
        prepared = publication.f_locals["prepared"]
        model = prepared._candidate._model
        if mutation == "replay_array":
            model._replay_memory[0].input_batch[0, 0] += 0.25
        elif mutation == "weights":
            model.weight_input_hidden[0, 0] += 0.25
        elif mutation == "rng":
            state = model._rng.bit_generator.state
            state["state"]["state"] = (state["state"]["state"] + 1) % (2**128)
            model._rng.bit_generator.state = state
        elif mutation == "inbox_array":
            next(iter(prepared._inbox._experiences.values())).features[0, 0] += 0.25
        elif mutation == "event_index":
            prepared._inbox._event_ids.add("untracked-event")
        elif mutation == "work_progress":
            runtime._budget.replay_examples_completed += 1
        elif mutation == "custom_sequence":
            model._replay_memory = ForbiddenSequence()
        elif mutation == "nan_clock":
            runtime._budget.last_clock = float("nan")
        elif mutation == "renewed_tick":
            key = next(iter(ledger._rows.values())).data.key
            owner._declaration_ticks[key] += 1
        elif mutation == "weakref_callback":
            row = next(iter(ledger._rows.values()))
            original_ref = row.declaration

            def forbidden_reference():
                callbacks.append("weakref callback")
                return original_ref()

            object.__setattr__(row, "declaration", forbidden_reference)
        elif mutation == "history_both":
            runtime._attempted_ids.add("replacement-consolidation")
            prepared._attempted_ids.add("replacement-consolidation")
        else:
            # Original guards are held. Their cleanup must release these exact
            # acquired objects without restoring any authority field.
            for holder, name in [
                (controller, "_gate"),
                (runtime, "_write_gate"),
                (owner, "_gate"),
                (ledger, "_gate"),
                (manager, "_gate"),
                (manager._copies, "_gate"),
                (life._registry, "_gate"),
                (life, "_time_gate"),
                (life._copy_budget, "_gate"),
                (owner._shared._sharing, "_gate"),
                (runtime._actor, "_read_gate"),
            ]:
                gate = getattr(holder, name)
                acquired_gates.append(gate)
                replacement = Lock()
                replacements.append((holder, name, replacement))
                setattr(holder, name, replacement)
        return result

    owner, life, runtime, ledger, controller, manager = setup(fingerprint=fingerprint)
    token = manager.capture(controller)
    anchors = ledger._anchors
    charge = life._copy_budget._charged
    armed[0] = True
    try:
        with pytest.raises(ValueError):
            manager.restore(controller, token)
    finally:
        armed[0] = False
        current_frame[0] = None
    assert fired[0] and visits[0] == 3
    assert_unpublished(owner, runtime, ledger, controller, token, anchors)
    assert life._copy_budget._charged > charge and ledger._copy_slots > 0
    assert callbacks == []
    assert all(not gate.locked() for gate in acquired_gates)
    assert all(
        getattr(holder, name) is gate and not gate.locked() for holder, name, gate in replacements
    )
