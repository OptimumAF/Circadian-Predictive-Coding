"""Bounded complete learner policy plus unchanged native checkpoint bytes.

Metadata is admitted before native codec work. Exact outer overhead is deducted
from the inner byte allowance, so policy cannot grant a second encoded budget.
Restore validates all bytes first, then uses explicit native constructors/full
restore and the normal owned learner constructor. No IO, generic object loading,
runtime budgets, authority, provenance or automatic retry is provided.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json
from typing import Any

from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec, _configuration
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.numpy_checkpoint_frames import canonical_json, exact_object, unique_pairs
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.coupled_circadian_predictive_coding import (
    CoupledCircadianConfig,
    CoupledCircadianPredictiveCodingNetwork,
)
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import (
    CoupledLearnerCheckpoint,
    CoupledUpdatePolicy,
    copy_update_policy,
    validate_learner_checkpoint,
)
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork

_ENVELOPE = {"codec_version", "kind", "binding", "policy", "native"}
_POLICY = {"learning_rate", "inference_steps", "inference_learning_rate", "inference_policy"}
_GUARDED = {"min_steps", "gradient_tolerance", "relative_energy_tolerance"}


def _controls(binding: CodecBinding, limits: CodecLimits) -> tuple[CodecBinding, CodecLimits]:
    _configuration(binding, limits)
    return replace(binding), replace(limits)


def _policy(value: object) -> CoupledUpdatePolicy:
    record = exact_object(value, _POLICY)
    guarded = record["inference_policy"]
    if guarded is not None:
        guarded = CoupledInferencePolicy(**exact_object(guarded, _GUARDED))
    return CoupledUpdatePolicy(
        record["learning_rate"],
        record["inference_steps"],
        record["inference_learning_rate"],
        guarded,
    )


def _parts(
    binding: CodecBinding, policy: CoupledUpdatePolicy, limits: CodecLimits
) -> tuple[bytes, bytes, CodecLimits]:
    empty = canonical_json(
        dict(
            codec_version=1,
            kind="coupled_learner",
            binding=asdict(binding),
            policy=asdict(policy),
            native={},
        )
    )
    before, marker, after = empty.partition(b'"native":{}')
    assert marker  # Fixed schema and canonical key order provide one placeholder.
    prefix = before + b'"native":'
    remainder = limits.max_encoded_bytes - len(prefix) - len(after)
    if remainder <= 0:
        raise ValueError("learner checkpoint metadata exhausts the total encoded byte bound")
    return prefix, after, replace(limits, max_encoded_bytes=remainder)


def capture_coupled_learner(learner: CoupledLearner) -> CoupledLearnerCheckpoint:
    if type(learner) is not CoupledLearner:
        raise ValueError("learner checkpoint capture requires an exact trusted CoupledLearner")
    policy = copy_update_policy(learner.checkpoint_policy())
    return CoupledLearnerCheckpoint(1, policy, learner.snapshot_state())


def _nonfinite(value: str) -> Any:
    raise ValueError(f"nonfinite learner checkpoint JSON number: {value}")


class CoupledLearnerCheckpointCodec:
    def encode(
        self, state: CoupledLearnerCheckpoint, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        binding, limits = _controls(binding, limits)
        validate_learner_checkpoint(state)
        policy = copy_update_policy(state.policy)
        prefix, suffix, native_limits = _parts(binding, policy, limits)
        native = CoupledCheckpointCodec().encode(
            state.native_state, binding=binding, limits=native_limits
        )
        if type(native) is not bytes or len(native) > native_limits.max_encoded_bytes:
            raise ValueError("native checkpoint exceeds the remaining total byte allowance")
        raw = prefix + native + suffix
        if len(raw) > limits.max_encoded_bytes:
            raise ValueError("complete learner checkpoint exceeds total encoded byte bound")
        return raw

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> CoupledLearnerCheckpoint:
        binding, limits = _controls(binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("learner checkpoint requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("learner checkpoint differs from independently expected digest")
        try:
            body = json.loads(raw, object_pairs_hook=unique_pairs, parse_constant=_nonfinite)
            exact_object(body, _ENVELOPE)
            if type(body["codec_version"]) is not int or body["codec_version"] != 1:
                raise ValueError("unsupported exact learner checkpoint codec version")
            if type(body["kind"]) is not str or body["kind"] != "coupled_learner":
                raise ValueError("unsupported learner checkpoint kind")
            declared = CodecBinding(
                **exact_object(body["binding"], {"source_sha256", "policy_sha256"})
            )
            if declared != binding:
                raise ValueError("learner checkpoint differs from independently expected binding")
            policy = _policy(body["policy"])
            _, _, native_limits = _parts(binding, policy, limits)
            if type(body["native"]) is not dict or canonical_json(body) != raw:
                raise ValueError("learner checkpoint JSON must be complete and canonical")
            native = canonical_json(body["native"])
            if len(native) > native_limits.max_encoded_bytes:
                raise ValueError("native checkpoint exceeds remaining total encoded byte bound")
            state = CoupledCheckpointCodec().decode(
                native,
                binding=binding,
                expected_sha256=sha256(native).hexdigest(),
                limits=native_limits,
            )
            return CoupledLearnerCheckpoint(1, copy_update_policy(policy), state)
        except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError) as error:
            raise ValueError("invalid/corrupted complete coupled learner checkpoint") from error


def restore_coupled_learner(
    raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
) -> CoupledLearner:
    checkpoint = CoupledLearnerCheckpointCodec().decode(
        raw, binding=binding, expected_sha256=expected_sha256, limits=limits
    )
    snapshot, policy = checkpoint.native_state, checkpoint.policy
    # The unchanged learner normalizes rates to float. Encoding may preserve
    # native integers, but restoration must not silently round their policy.
    for name in ("learning_rate", "inference_learning_rate"):
        value = getattr(policy, name)
        if float(value) != value:
            raise ValueError(f"restoring {name} requires a lossless conversion to native float")
    input_dim = snapshot.parameters.hidden_weights[0].shape[0]
    hidden_dims = tuple(weight.shape[1] for weight in snapshot.parameters.hidden_weights)
    model: CoupledPredictiveCodingNetwork
    if snapshot.wake_state is None:
        model = CoupledPredictiveCodingNetwork(
            input_dim, hidden_dims, seed=0, precisions=snapshot.precisions
        )
    else:
        wake = snapshot.wake_state
        config = CoupledCircadianConfig(
            wake.chemical_decay,
            wake.chemical_buildup_rate,
            wake.plasticity_sensitivity,
            wake.min_plasticity,
        )
        model = CoupledCircadianPredictiveCodingNetwork(
            input_dim, hidden_dims, seed=0, precisions=snapshot.precisions, config=config
        )
    model.restore_state(snapshot)
    return CoupledLearner(
        model,
        learning_rate=policy.learning_rate,
        inference_steps=policy.inference_steps,
        inference_learning_rate=policy.inference_learning_rate,
        inference_policy=policy.inference_policy,
    )
