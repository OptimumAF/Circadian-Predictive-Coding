"""Complete bounded lifecycle bytes through the inner CheckpointCodec port.

Original policy/source/content/reference bindings are supplied independently.
Decode detaches metadata and retains the original supplied capture's live tuple.
Matching tags do not attest ownership or authorize restore. No clocks,ports or IO.
"""

from hashlib import sha256
import json
from src.adapters.lifecycle_checkpoint_schema import (
    materialize_metadata,
    metadata_aliases,
    preflight_aliases,
    preflight_metadata,
    project_metadata,
)
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.lifecycle_codec_policy import LifecycleCodecPolicy
from src.core.managed_lifecycle_state import AUTHORITY_PATHS, ManagedLifecycleCapture
from src.core.managed_lifecycle_validation import validate_lifecycle_capture, require_state_record

ENVELOPE = frozenset(
    {
        "codec_version",
        "kind",
        "binding",
        "policy",
        "authority_sha256",
        "reference_manifest",
        "metadata",
        "metadata_aliases",
    }
)
BINDING_FIELDS = frozenset({"source_sha256", "policy_sha256"})
LIMIT_FIELDS = frozenset({"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"})


def _object(value, fields):
    if type(value) is not dict or value.keys() != fields:
        raise ValueError("lifecycle wire differs from complete explicit envelope")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate lifecycle JSON key")
        result[key] = value
    return result


def _json(value, bound):
    parts, size = [], 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for fragment in encoder.iterencode(value):
        part = fragment.encode("utf8")
        size += len(part)
        if size > bound:
            raise ValueError("lifecycle wire exceeds original byte bound")
        parts.append(part)
    return b"".join(parts)


def _references(original):
    return {item.path: item.value for item in original.authority}


def _reference_manifest(original):
    refs = _references(original)
    first: dict[int, int] = {}
    return [
        {
            "path": path,
            "present": refs[path] is not None,
            "first": first.setdefault(id(refs[path]), index),
        }
        for index, path in enumerate(sorted(AUTHORITY_PATHS))
    ]


def _reference_observation(metadata, original):
    refs = _references(original)
    driver = metadata["driver"]
    if driver is not None and (
        (refs["driver._thread"] is not None) != driver["thread_present"]
        or (driver["held"] and refs["driver._token"] is not refs["sharing._retention_hold"])
    ):
        raise ValueError("lifecycle thread/token observations differ from original references")


class LifecycleCheckpointCodec:
    def __init__(
        self,
        policy: LifecycleCodecPolicy,
        *,
        original: ManagedLifecycleCapture,
        authority_sha256: str,
    ) -> None:
        if type(policy) is not LifecycleCodecPolicy:
            raise ValueError("lifecycle codec requires independent original exact policy")
        LifecycleCodecPolicy.__post_init__(policy)
        require_codec_digest(authority_sha256)
        validate_lifecycle_capture(original, policy.capture_limits)
        self._policy, self._original, self._authority = policy, original, authority_sha256

    def _configuration(self, binding, limits):
        require_state_record(binding, CodecBinding)
        require_state_record(limits, CodecLimits)
        CodecBinding.__post_init__(binding)
        CodecLimits.__post_init__(limits)
        LifecycleCodecPolicy.__post_init__(self._policy)
        require_codec_digest(self._authority)
        validate_lifecycle_capture(self._original, self._policy.capture_limits)
        preflight_metadata(
            project_metadata(self._original.metadata), self._policy, self._canonical(limits)
        )

    def _canonical(self, limits):
        return lambda value: _json(value, limits.max_encoded_bytes)

    def _preflight(self, metadata, aliases, limits):
        canonical = self._canonical(limits)
        nodes = preflight_metadata(metadata, self._policy, canonical)
        preflight_aliases(nodes, aliases, canonical, copy_present=metadata["copy"] is not None)
        _reference_observation(metadata, self._original)

    def encode(
        self, state: ManagedLifecycleCapture, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        try:
            self._configuration(binding, limits)
            validate_lifecycle_capture(state, self._policy.capture_limits)
            refs, original = _references(state), _references(self._original)
            if any(refs[path] is not original[path] for path in AUTHORITY_PATHS):
                raise ValueError("lifecycle encoding requires original complete live references")
            metadata = project_metadata(state.metadata)
            aliases = metadata_aliases(state.metadata)
            self._preflight(metadata, aliases, limits)
            return _json(
                dict(
                    codec_version=1,
                    kind="lifecycle_full_v1",
                    binding={
                        "source_sha256": binding.source_sha256,
                        "policy_sha256": binding.policy_sha256,
                    },
                    policy=project_metadata(self._policy, LifecycleCodecPolicy),
                    authority_sha256=self._authority,
                    reference_manifest=_reference_manifest(self._original),
                    metadata=metadata,
                    metadata_aliases=aliases,
                ),
                limits.max_encoded_bytes,
            )
        except (
            TypeError,
            AttributeError,
            KeyError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise ValueError("unsupported or invalid complete lifecycle capture") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> ManagedLifecycleCapture:
        try:
            self._configuration(binding, limits)
            require_codec_digest(expected_sha256)
            if type(raw) is not bytes or not raw or len(raw) > limits.max_encoded_bytes:
                raise ValueError("lifecycle bytes exceed original wire bound")
            if sha256(raw).hexdigest() != expected_sha256:
                raise ValueError("lifecycle independently expected content digest differs")
            body = _object(json.loads(raw, object_pairs_hook=_pairs), ENVELOPE)
            _object(body["binding"], BINDING_FIELDS)
            canonical = self._canonical(limits)
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 1
                or type(body["kind"]) is not str
                or body["kind"] != "lifecycle_full_v1"
                or canonical(body["binding"])
                != canonical(
                    {"source_sha256": binding.source_sha256, "policy_sha256": binding.policy_sha256}
                )
                or canonical(body["policy"])
                != canonical(project_metadata(self._policy, LifecycleCodecPolicy))
                or type(body["authority_sha256"]) is not str
                or body["authority_sha256"] != self._authority
                or canonical(body["reference_manifest"])
                != canonical(_reference_manifest(self._original))
            ):
                raise ValueError(
                    "lifecycle version or independently expected original bindings differ"
                )
            if canonical(body) != raw:
                raise ValueError("lifecycle wire is not canonical")
            self._preflight(body["metadata"], body["metadata_aliases"], limits)
            metadata = materialize_metadata(body["metadata"], body["metadata_aliases"])
            result = ManagedLifecycleCapture(metadata, self._original.authority)
            validate_lifecycle_capture(result, self._policy.capture_limits)
            return result
        except (
            TypeError,
            AttributeError,
            KeyError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise ValueError("corrupt or unsupported complete lifecycle bytes") from error
