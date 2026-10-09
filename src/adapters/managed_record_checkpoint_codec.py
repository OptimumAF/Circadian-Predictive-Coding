"""Complete bounded paired bytes with independent original capture bindings.

Input: exact original typed policies/capture, source/policy/content/authority
tags and bounded bytes. Output: detached metadata plus the original live tuple.
No port/clock/native/IO operation, provenance seal, authority renewal or restore.
"""

from hashlib import sha256
import json

from src.adapters.lifecycle_checkpoint_codec import (
    BINDING_FIELDS,
    ENVELOPE,
    _json,
    _object,
    _pairs,
    _reference_observation,
)
from src.adapters.lifecycle_checkpoint_schema import (
    materialize_metadata,
    metadata_aliases,
    preflight_aliases,
    project_metadata,
)
from src.adapters.managed_record_checkpoint_schema import PAIR_SCHEMAS, preflight_pair
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.managed_lifecycle_state import AUTHORITY_PATHS, ManagedLifecycleCapture
from src.core.managed_lifecycle_validation import require_state_record
from src.core.managed_record_codec_policy import ManagedRecordCodecPolicy
from src.core.managed_record_state import (
    ManagedRecordCapture,
    ManagedRecordMetadata,
    RECORD_AUTHORITY_PATHS,
    validate_managed_record_capture,
)


def _references(original):
    return {item.path: item.value for item in original.authority}


def _manifest(original):
    refs = _references(original)
    first: dict[int, int] = {}
    return [
        dict(
            path=path, present=refs[path] is not None, first=first.setdefault(id(refs[path]), index)
        )
        for index, path in enumerate(sorted(RECORD_AUTHORITY_PATHS))
    ]


def _project(metadata):
    return project_metadata(metadata, ManagedRecordMetadata, schemas=PAIR_SCHEMAS)


def _aliases(metadata):
    return metadata_aliases(metadata, ManagedRecordMetadata, schemas=PAIR_SCHEMAS)


class ManagedRecordCheckpointCodec:
    def __init__(
        self,
        policy: ManagedRecordCodecPolicy,
        *,
        original: ManagedRecordCapture,
        authority_sha256: str,
    ) -> None:
        require_state_record(policy, ManagedRecordCodecPolicy)
        ManagedRecordCodecPolicy.__post_init__(policy)
        require_codec_digest(authority_sha256)
        validate_managed_record_capture(original, policy.lifecycle.capture_limits)
        self._policy, self._original, self._authority = policy, original, authority_sha256

    def _configuration(self, binding, limits):
        require_state_record(binding, CodecBinding)
        require_state_record(limits, CodecLimits)
        CodecBinding.__post_init__(binding)
        CodecLimits.__post_init__(limits)
        ManagedRecordCodecPolicy.__post_init__(self._policy)
        require_codec_digest(self._authority)
        validate_managed_record_capture(self._original, self._policy.lifecycle.capture_limits)
        self._preflight(
            _project(self._original.metadata), _aliases(self._original.metadata), limits
        )

    def _canonical(self, limits):
        return lambda value: _json(value, limits.max_encoded_bytes)

    def _preflight(self, metadata, aliases, limits):
        canonical = self._canonical(limits)
        nodes = preflight_pair(metadata, self._policy, canonical)
        preflight_aliases(
            nodes,
            aliases,
            canonical,
            copy_present=metadata["lifecycle"]["copy"] is not None,
            prefix="metadata.lifecycle",
        )
        original_life = ManagedLifecycleCapture(
            self._original.metadata.lifecycle,
            tuple(item for item in self._original.authority if item.path in AUTHORITY_PATHS),
        )
        _reference_observation(metadata["lifecycle"], original_life)
        # Bind the entire original observation, not a subset of revision/budget
        # counters: another coherent capture can have different elapsed epochs.
        if canonical(metadata) != canonical(_project(self._original.metadata)) or canonical(
            aliases
        ) != canonical(_aliases(self._original.metadata)):
            raise ValueError(
                "paired complete observation differs from independently supplied original capture"
            )

    def _policy_data(self):
        return project_metadata(self._policy, ManagedRecordCodecPolicy, schemas=PAIR_SCHEMAS)

    def _bindings(self, body, binding, limits):
        _object(body, ENVELOPE)
        _object(body["binding"], BINDING_FIELDS)
        canonical = self._canonical(limits)
        if (
            type(body["codec_version"]) is not int
            or body["codec_version"] != 1
            or type(body["kind"]) is not str
            or body["kind"] != "managed_record_pair_v1"
            or canonical(body["binding"])
            != canonical(
                dict(source_sha256=binding.source_sha256, policy_sha256=binding.policy_sha256)
            )
            or canonical(body["policy"]) != canonical(self._policy_data())
            or type(body["authority_sha256"]) is not str
            or body["authority_sha256"] != self._authority
            or canonical(body["reference_manifest"]) != canonical(_manifest(self._original))
        ):
            raise ValueError("paired version or independently expected original bindings differ")

    def encode(
        self, state: ManagedRecordCapture, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        try:
            self._configuration(binding, limits)
            validate_managed_record_capture(state, self._policy.lifecycle.capture_limits)
            refs, original = _references(state), _references(self._original)
            if any(refs[path] is not original[path] for path in RECORD_AUTHORITY_PATHS):
                raise ValueError("paired encoding requires the original complete live references")
            metadata, aliases = _project(state.metadata), _aliases(state.metadata)
            self._preflight(metadata, aliases, limits)
            return _json(
                dict(
                    codec_version=1,
                    kind="managed_record_pair_v1",
                    binding=dict(
                        source_sha256=binding.source_sha256, policy_sha256=binding.policy_sha256
                    ),
                    policy=self._policy_data(),
                    authority_sha256=self._authority,
                    reference_manifest=_manifest(self._original),
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
            raise ValueError("unsupported or invalid complete paired capture") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> ManagedRecordCapture:
        try:
            self._configuration(binding, limits)
            require_codec_digest(expected_sha256)
            if type(raw) is not bytes or not raw or len(raw) > limits.max_encoded_bytes:
                raise ValueError("paired bytes exceed original wire bound")
            if sha256(raw).hexdigest() != expected_sha256:
                raise ValueError("paired independently expected content digest differs")
            body = json.loads(raw, object_pairs_hook=_pairs)
            self._bindings(body, binding, limits)
            if self._canonical(limits)(body) != raw:
                raise ValueError("paired wire is not canonical")
            self._preflight(body["metadata"], body["metadata_aliases"], limits)
            metadata = materialize_metadata(
                body["metadata"],
                body["metadata_aliases"],
                ManagedRecordMetadata,
                schemas=PAIR_SCHEMAS,
            )
            result = ManagedRecordCapture(metadata, self._original.authority)
            validate_managed_record_capture(result, self._policy.lifecycle.capture_limits)
            return result
        except (
            TypeError,
            AttributeError,
            KeyError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise ValueError("corrupt or unsupported complete paired bytes") from error
