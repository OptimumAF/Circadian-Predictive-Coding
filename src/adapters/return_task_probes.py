"""Compose existing disposable E-task probes with owned return checkpoints.

Inputs are the fixed reserved packets and original completed stream boundary.
Outputs are existing probe slots and permanent independent charges. No benchmark
training, policy selection, final-data collection, promotion or persistence.
"""

from copy import deepcopy
from time import monotonic
from typing import Any, Callable

from src.adapters.key_door_learnability import _bindings, run_checkpoint_probe
from src.adapters.return_replay_accounting import replay_evidence


def run_return_checkpoint_probe(
    row: dict[str, Any],
    source: Any,
    state: Any,
    data: Any,
    *,
    checkpoint: int,
    stream: Any,
    wake: Any,
    replay: Any,
    helper: Any,
    deadline: float,
    sampler: Any,
    same_state: Callable[[Any, Any], bool],
    isolation: Callable[[], Any],
    probe_runner: Callable[..., dict[str, Any]],
) -> None:
    """Check the sealed benchmark around each external probe callback.

    Why: probe budget checks own their clock; observing outer budget clocks here
    would change the exact unchanged bindings required by the existing helper.
    """
    binding = _bindings(wake, replay)
    expected_state = deepcopy(state)
    candidate = deepcopy(stream._runtime._candidate.snapshot_state())
    receipts = deepcopy(stream._runtime.applied_updates)
    states = deepcopy(stream._states)
    replay_before = None if helper is None else replay_evidence(helper, replay)

    def guard() -> None:
        if monotonic() > deadline:
            raise ValueError("return probe exceeded its original invocation deadline")
        isolation()
        stream._check_original()
        authoritative = [saved for n, saved in stream._states if n == checkpoint]
        if (
            stream.completed_updates != checkpoint
            or len(authoritative) != 1
            or not same_state(state, authoritative[0])
            or binding != _bindings(wake, replay)
            or not same_state(expected_state, state)
            or not same_state(candidate, stream._runtime._candidate.snapshot_state())
            or not same_state(receipts, stream._runtime.applied_updates)
            or not same_state(states, stream._states)
            or (helper is not None and replay_evidence(helper, replay) != replay_before)
        ):
            raise ValueError("return probe changed original benchmark evidence or budget")

    guard()
    run_checkpoint_probe(
        row,
        source,
        state,
        data,
        checkpoint=checkpoint,
        wake=wake,
        replay=replay,
        deadline=deadline,
        sampler=sampler,
        same_state=same_state,
        probe_runner=probe_runner,
        decision_diagnostics=True,
        check_boundary=guard,
    )
    guard()
