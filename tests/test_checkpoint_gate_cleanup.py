"""Release the actual acquired lock even if a failing callback replaces its field."""

from threading import Lock

import pytest

from src.app.actor_shadow import ActorShadowRuntime
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.app.managed_replay_checkpoints import ManagedReplayCheckpoints
from src.app.managed_replay_copies import ManagedReplayCopies
from src.app.managed_replay_origins import ManagedReplayOrigins
from src.app.resource_sharing import ServingPriorityGate


@pytest.mark.parametrize(
    "kind,field,method",
    [
        (ActorShadowRuntime, "_write_gate", "_exclusive"),
        (CandidateCheckpointController, "_gate", "_exclusive"),
        (ManagedExperienceOwner, "_gate", "_operation"),
        (ManagedReplayOrigins, "_gate", "_exclusive"),
        (ManagedReplayCopies, "_gate", "_exclusive"),
        (ManagedReplayCheckpoints, "_gate", "_exclusive"),
    ],
)
def test_should_preserve_failure_and_release_acquired_original_gate(kind, field, method):
    holder = object.__new__(kind)
    original, replacement = Lock(), Lock()
    setattr(holder, field, original)
    failure = ValueError("original checkpoint refusal")
    with pytest.raises(ValueError, match="original checkpoint refusal") as caught:
        with getattr(holder, method)():
            assert original.locked()
            setattr(holder, field, replacement)
            raise failure
    assert caught.value is failure
    assert not original.locked() and not replacement.locked()
    assert getattr(holder, field) is replacement


def test_should_clear_checkpoint_flag_under_original_sharing_gate_on_failure():
    holder = object.__new__(ServingPriorityGate)
    original, replacement = Lock(), Lock()
    holder._gate = original
    holder._checkpointing = holder._training = False
    holder._serving = 0
    holder._paused = True
    holder._retention_hold = None
    failure = ValueError("original sharing refusal")
    with pytest.raises(ValueError, match="original sharing refusal") as caught:
        with holder._checkpoint_lease():
            assert holder._checkpointing
            holder._gate = replacement
            raise failure
    assert caught.value is failure and not holder._checkpointing
    assert not original.locked() and not replacement.locked()
    assert holder._gate is replacement
