"""Complete supported NumPy inbox byte codec through the inner component port.

Inputs: exact detached cursors/bounded bytes and independently expected original
shape policy/bindings/content. Outputs: canonical bytes or detached cursors.
No model work, IO, consent/owner attestation, budget reset or live/disk restore.
"""

from dataclasses import asdict
from hashlib import sha256
import json

from typing import Any
from numpy.typing import NDArray

from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.inbox_codec_policy import InboxCodecPolicy, DTYPE_FIELDS
from src.core.inbox_cursor import InboxCursor
from src.adapters.numpy_checkpoint_frames import canonical_json, exact_object, unique_pairs
from src.adapters.inbox_checkpoint_schema import native_data, metadata_cursor, materialize_cursor
from src.adapters.inbox_checkpoint_payloads import (
    specifications,
    size_preflight,
    encode_payloads,
    decode_payloads,
)

Array = NDArray[Any]


def validate_configuration(policy, binding, limits):
    if (
        type(policy) is not InboxCodecPolicy
        or type(binding) is not CodecBinding
        or type(limits) is not CodecLimits
    ):
        raise ValueError("inbox codec requires exact original policy/binding/limits")
    policy.__post_init__()
    exact_object(vars(binding), {"source_sha256", "policy_sha256"})
    exact_object(
        vars(limits), {"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"}
    )
    binding.__post_init__()
    limits.__post_init__()
    if policy.input_dim > limits.max_dimension:
        raise ValueError("original payload dimension exceeds codec allowance")


class NumpyInboxCheckpointCodec:
    def __init__(self, policy: InboxCodecPolicy) -> None:
        if type(policy) is not InboxCodecPolicy:
            raise ValueError("inbox codec requires original exact shape policy")
        policy.__post_init__()
        self._policy = policy

    def encode(
        self, state: InboxCursor[Array, Array], *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        validate_configuration(self._policy, binding, limits)
        try:
            data = native_data(state, self._policy)
            metadata_cursor(data, self._policy)
            body = dict(
                codec_version=2,
                kind="numpy_inbox_v2",
                binding=asdict(binding),
                schema=asdict(self._policy),
                cursor=data,
            )
            specs = specifications(data, self._policy, limits, False)
            size_preflight(body, specs, limits)
            encode_payloads(specs)
            return canonical_json(body)
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OverflowError,
            UnicodeError,
            RecursionError,
        ) as error:
            raise ValueError("invalid/unsupported complete NumPy inbox cursor") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> InboxCursor[Array, Array]:
        validate_configuration(self._policy, binding, limits)
        require_codec_digest(expected_sha256)
        if type(raw) is not bytes or len(raw) > limits.max_encoded_bytes:
            raise ValueError("inbox codec requires bounded exact bytes")
        if sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("content differs from independently expected digest")
        try:
            body = exact_object(
                json.loads(raw, object_pairs_hook=unique_pairs),
                {"codec_version", "kind", "binding", "schema", "cursor"},
            )
            schema = exact_object(body["schema"], vars(self._policy).keys()).copy()
            for name in DTYPE_FIELDS:
                if type(schema[name]) is not list:
                    raise ValueError("wire dtype policy requires exact list")
                schema[name] = tuple(schema[name])
            encoded_policy = InboxCodecPolicy(**schema)
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 2
                or body["kind"] != "numpy_inbox_v2"
                or body["binding"] != asdict(binding)
                or encoded_policy != self._policy
            ):
                raise ValueError("version/kind/original source/policy/shape binding differs")
            meta = metadata_cursor(body["cursor"], self._policy)
            specs = specifications(body["cursor"], self._policy, limits, True)
            size_preflight(body, specs, limits)
            if canonical_json(body) != raw:
                raise ValueError("inbox checkpoint bytes must be canonical")
            decode_payloads(specs)
            return materialize_cursor(meta, body["cursor"])
        except (
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            OverflowError,
            UnicodeError,
            RecursionError,
        ) as error:
            raise ValueError("invalid/corrupted complete NumPy inbox bytes") from error
