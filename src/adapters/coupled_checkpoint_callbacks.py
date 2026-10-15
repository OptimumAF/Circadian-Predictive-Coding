"""Policy-aware callbacks for the existing same-process candidate controller.

Inputs are a fixed owned policy/binding/budget and trusted native snapshots or
exact coupled learners. Outputs are bounded complete-state/policy digests and
independently restored learners. Why reuse the full envelope: omitting update
policy would permit a different future learning rule under the same native state.
These trusted callbacks add no ownership authority, controller, IO, training,
prediction, runtime-resource restoration or recovery protocol.
"""

from dataclasses import asdict
from hashlib import sha256

from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_learner_checkpoint import (
    CoupledLearnerCheckpointCodec,
    _controls,
    restore_coupled_learner,
)
from src.adapters.numpy_checkpoint_frames import canonical_json
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_learner_state import (
    CoupledLearnerCheckpoint,
    CoupledUpdatePolicy,
    copy_update_policy,
)
from src.core.coupled_predictive_coding import CoupledSnapshot


def _policy_bytes(policy: CoupledUpdatePolicy, limits: CodecLimits) -> bytes:
    raw = canonical_json(asdict(policy))
    if len(raw) > limits.max_encoded_bytes:
        raise ValueError("complete coupled policy exceeds the encoded byte bound")
    return raw


class CoupledCheckpointCallbacks:
    """Bind complete native callback operations to detached fixed metadata."""

    def __init__(
        self, policy: CoupledUpdatePolicy, *, binding: CodecBinding, limits: CodecLimits
    ) -> None:
        self._policy = copy_update_policy(policy)
        self._binding, self._limits = _controls(binding, limits)
        _policy_bytes(self._policy, self._limits)

    def _settings(self) -> tuple[CoupledUpdatePolicy, CodecBinding, CodecLimits]:
        # Revalidate at every callback: frozen records are not tamper-proof, and
        # callers must not share the guarded record or alter this operation's caps.
        policy = copy_update_policy(self._policy)
        binding, limits = _controls(self._binding, self._limits)
        _policy_bytes(policy, limits)
        return policy, binding, limits

    def _envelope(self, state: CoupledSnapshot) -> tuple[bytes, CodecBinding, CodecLimits]:
        policy, binding, limits = self._settings()
        checkpoint = CoupledLearnerCheckpoint(1, policy, state)
        raw = CoupledLearnerCheckpointCodec().encode(checkpoint, binding=binding, limits=limits)
        return raw, binding, limits

    def state_digest(self, state: CoupledSnapshot) -> str:
        raw, _, _ = self._envelope(state)
        return sha256(raw).hexdigest()

    def build(self, state: CoupledSnapshot) -> CoupledLearner:
        raw, binding, limits = self._envelope(state)
        return restore_coupled_learner(
            raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
        )

    def policy_digest(self, learner: object) -> str:
        if type(learner) is not CoupledLearner:
            raise ValueError("policy callback requires an exact trusted CoupledLearner")
        owned_policy, _, limits = self._settings()
        policy = copy_update_policy(learner.checkpoint_policy())
        if policy != owned_policy:
            raise ValueError("coupled learner policy differs from the original callback policy")
        return sha256(_policy_bytes(policy, limits)).hexdigest()
