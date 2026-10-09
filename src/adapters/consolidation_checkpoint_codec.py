"""Explicit complete consolidation bytes through the inner checkpoint port.

Inputs are exact immutable observations or bounded canonical bytes with original
policy/source/content/authority references. Outputs are detached observations.
Preserves shared diagnostic identities. No native calls,IO,live owner or restore.
"""

from dataclasses import asdict, replace
from hashlib import sha256
import json

from src.core.actor_ports import AppliedConsolidation
from src.core.checkpoint_codec import CodecBinding, CodecLimits, require_codec_digest
from src.core.consolidation_codec_policy import ConsolidationCodecPolicy, POLICY_FIELDS
from src.core.consolidation_cursor import (
    CURSOR_FIELDS,
    DIAGNOSTIC_FIELDS,
    RECEIPT_FIELDS,
    ConsolidationCursor,
)
from src.core.learner_ports import TrainingDiagnostic

ENVELOPE = frozenset(
    {
        "codec_version",
        "kind",
        "binding",
        "policy",
        "authority_sha256",
        "cursor",
        "diagnostic_aliases",
    }
)
BINDING_FIELDS = frozenset({"source_sha256", "policy_sha256"})
LIMIT_FIELDS = frozenset({"max_encoded_bytes", "max_array_bytes", "max_layers", "max_dimension"})


def _object(value, names):
    if type(value) is not dict or value.keys() != names:
        raise ValueError("consolidation wire fields differ from complete supported schema")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate consolidation JSON key")
        result[key] = value
    return result


def _json(value, bound):
    # Bound aggregate encoded fragments before joining a complete wire copy.
    parts, size = [], 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for fragment in encoder.iterencode(value):
        part = fragment.encode("utf8")
        size += len(part)
        if size > bound:
            raise ValueError("consolidation wire exceeds original byte bound")
        parts.append(part)
    return b"".join(parts)


def _string(value, policy):
    if (
        type(value) is not str
        or not value
        or len(value) > policy.max_identifier_bytes
        or len(value.encode("utf8")) > policy.max_identifier_bytes
    ):
        raise ValueError("consolidation metadata exceeds original UTF8 string bound")


def _bounds(data, policy, *, native):
    _object(data, CURSOR_FIELDS)
    expected = tuple if native else list
    if (
        type(data["consolidation_limit"]) is not int
        or data["consolidation_limit"] != policy.consolidation_limit
    ):
        raise ValueError("consolidation attempt quota differs from original policy")
    for name in ("attempted_ids", "consolidations"):
        if type(data[name]) is not expected or len(data[name]) > policy.consolidation_limit:
            raise ValueError("consolidation history exceeds original count bound")
    for value in (data["actor_version"], data["learner_version"], *data["attempted_ids"]):
        _string(value, policy)


def _diagnostic_bounds(receipt, policy):
    _object(receipt, RECEIPT_FIELDS)
    for name in ("event_id", "actor_version", "learner_version"):
        _string(receipt[name], policy)
    _object(receipt["diagnostic"], DIAGNOSTIC_FIELDS)
    _string(receipt["diagnostic"]["definition"], policy)


def _native_data(state, policy):
    if type(state) is not ConsolidationCursor:
        raise ValueError("consolidation codec requires exact supported cursor")
    data = vars(state).copy()
    _bounds(data, policy, native=True)
    ConsolidationCursor.__post_init__(state)
    receipts, aliases = [], []
    first: dict[int, int] = {}
    for index, receipt in enumerate(state.consolidations):
        row = vars(receipt).copy()
        row["diagnostic"] = vars(receipt.diagnostic).copy()
        _diagnostic_bounds(row, policy)
        receipts.append(row)
        aliases.append(first.setdefault(id(receipt.diagnostic), index))
    data["attempted_ids"] = list(state.attempted_ids)
    data["consolidations"] = receipts
    return data, aliases


def _wire_cursor(data, policy):
    _bounds(data, policy, native=False)
    receipts = []
    for row in data["consolidations"]:
        _diagnostic_bounds(row, policy)
        receipts.append(
            AppliedConsolidation(**{**row, "diagnostic": TrainingDiagnostic(**row["diagnostic"])})
        )
    return ConsolidationCursor(
        **{**data, "attempted_ids": tuple(data["attempted_ids"]), "consolidations": tuple(receipts)}
    )


def _apply_aliases(cursor, aliases, data, limits):
    if type(aliases) is not list or len(aliases) != len(cursor.consolidations):
        raise ValueError("consolidation diagnostic alias count differs")
    receipts: list[AppliedConsolidation] = []
    for index, target in enumerate(aliases):
        if type(target) is not int or not 0 <= target <= index or aliases[target] != target:
            raise ValueError("consolidation diagnostic alias must name its original first record")
        if _json(data["consolidations"][index]["diagnostic"], limits.max_encoded_bytes) != _json(
            data["consolidations"][target]["diagnostic"], limits.max_encoded_bytes
        ):
            raise ValueError("shared consolidation diagnostic bits/types differ")
        diagnostic = (
            cursor.consolidations[index].diagnostic
            if target == index
            else receipts[target].diagnostic
        )
        receipts.append(replace(cursor.consolidations[index], diagnostic=diagnostic))
    return replace(cursor, consolidations=tuple(receipts))


class ConsolidationCheckpointCodec:
    def __init__(self, policy: ConsolidationCodecPolicy, *, authority_sha256: str) -> None:
        if type(policy) is not ConsolidationCodecPolicy:
            raise ValueError("consolidation codec requires original exact policy")
        ConsolidationCodecPolicy.__post_init__(policy)
        require_codec_digest(authority_sha256)
        self._policy, self._authority = policy, authority_sha256

    def _configuration(self, binding, limits):
        if (
            type(binding) is not CodecBinding
            or type(limits) is not CodecLimits
            or vars(binding).keys() != BINDING_FIELDS
            or vars(limits).keys() != LIMIT_FIELDS
        ):
            raise ValueError("consolidation codec requires original exact binding and limits")
        CodecBinding.__post_init__(binding)
        CodecLimits.__post_init__(limits)
        ConsolidationCodecPolicy.__post_init__(self._policy)
        require_codec_digest(self._authority)

    def encode(
        self, state: ConsolidationCursor, *, binding: CodecBinding, limits: CodecLimits
    ) -> bytes:
        try:
            self._configuration(binding, limits)
            cursor, aliases = _native_data(state, self._policy)
            return _json(
                dict(
                    codec_version=1,
                    kind="consolidation_full_v1",
                    binding=asdict(binding),
                    policy=asdict(self._policy),
                    authority_sha256=self._authority,
                    cursor=cursor,
                    diagnostic_aliases=aliases,
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
            raise ValueError("unsupported or invalid complete consolidation observation") from error

    def decode(
        self, raw: bytes, *, binding: CodecBinding, expected_sha256: str, limits: CodecLimits
    ) -> ConsolidationCursor:
        try:
            self._configuration(binding, limits)
            require_codec_digest(expected_sha256)
            if type(raw) is not bytes or not raw or len(raw) > limits.max_encoded_bytes:
                raise ValueError("consolidation bytes exceed original wire bound")
            if sha256(raw).hexdigest() != expected_sha256:
                raise ValueError("consolidation independent content digest differs")
            body = _object(json.loads(raw, object_pairs_hook=_pairs), ENVELOPE)
            _object(body["binding"], BINDING_FIELDS)
            _object(body["policy"], POLICY_FIELDS)
            original_policy = ConsolidationCodecPolicy(**body["policy"])
            if (
                type(body["codec_version"]) is not int
                or body["codec_version"] != 1
                or type(body["kind"]) is not str
                or body["kind"] != "consolidation_full_v1"
                or CodecBinding(**body["binding"]) != binding
                or original_policy != self._policy
                or type(body["authority_sha256"]) is not str
                or body["authority_sha256"] != self._authority
            ):
                raise ValueError("consolidation version or independently expected bindings differ")
            cursor = _wire_cursor(body["cursor"], self._policy)
            result = _apply_aliases(cursor, body["diagnostic_aliases"], body["cursor"], limits)
            if _json(body, limits.max_encoded_bytes) != raw:
                raise ValueError("consolidation wire is not canonical")
            return result
        except (
            TypeError,
            AttributeError,
            KeyError,
            UnicodeError,
            OverflowError,
            RecursionError,
        ) as error:
            raise ValueError("corrupt or unsupported complete consolidation bytes") from error
