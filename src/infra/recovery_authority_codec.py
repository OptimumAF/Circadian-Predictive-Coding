"""Bounded exact canonical JSON for authority metadata, never model/payload pickle."""

from dataclasses import asdict, fields
import json

from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.core.recovery_admission import (
    RecoveryMetadata,
    RecoveryManifest,
    RecoveryLimits,
    RecoveryUsage,
)
from src.core.recovery_observation import RecoveryProcessIdentity

MAX_RECORD_BYTES = 16 * 1024


def _object(value, record_type):
    if type(value) is not dict or set(value) != {field.name for field in fields(record_type)}:
        raise ValueError("authority JSON record fields differ from exact schema")
    return value.copy()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate authority JSON field")
        result[key] = value
    return result


def encode_authority(record: AuthorityRecord) -> bytes:
    validate_authority_record(record)
    raw = json.dumps(
        asdict(record), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("authority record exceeds byte bound")
    return raw


def decode_authority(raw: bytes) -> AuthorityRecord:
    if type(raw) is not bytes or len(raw) > MAX_RECORD_BYTES:
        raise ValueError("authority JSON requires bounded exact bytes")
    try:
        body = _object(json.loads(raw, object_pairs_hook=_pairs), AuthorityRecord)
        metadata = _object(body["metadata"], RecoveryMetadata)
        for name, kind in (
            ("manifest", RecoveryManifest),
            ("limits", RecoveryLimits),
            ("usage", RecoveryUsage),
        ):
            metadata[name] = kind(**_object(metadata[name], kind))
        record = AuthorityRecord(
            RecoveryMetadata(**metadata),
            RecoveryProcessIdentity(**_object(body["anchor"], RecoveryProcessIdentity)),
            RecoveryProcessIdentity(**_object(body["worker"], RecoveryProcessIdentity)),
        )
        if encode_authority(record) != raw:
            raise ValueError("authority JSON is not canonical")
        return record
    except (ValueError, TypeError, RecursionError, UnicodeDecodeError) as error:
        raise ValueError("invalid/corrupted authority JSON") from error
