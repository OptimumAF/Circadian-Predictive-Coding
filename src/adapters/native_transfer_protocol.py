"""Fixed native transfer fixtures and bounded completion-proof data conversion.

The six-field wire preserves the original admission and actual completion, not
just their final hashes. Why strict reconstruction: reports cannot manufacture
completion or renew original costs. These records do not authenticate a peer,
prove native execution or bootstrap authority from disk. No models or IO run here.
"""

from dataclasses import asdict
from hashlib import sha256
from typing import Any

from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coordinator_transfer import _authority, _host, _schema
from src.core.recovery_authority import AuthorityChange, AuthorityRecord
from src.core.recovery_completion import AuthorityCompletion, validate_authority_completion
from src.core.recovery_observation import RecoveryHostObservation, RecoveryProcessIdentity
from src.infra.recovery_authority_codec import MAX_RECORD_BYTES, decode_authority, encode_authority

BINDING = CodecBinding(
    sha256(b"native coordinator transfer engineering source v1").hexdigest(),
    sha256(b"native coordinator transfer engineering policy v1").hexdigest(),
)
LIMITS = CodecLimits(65536, 8192, 4, 16)
FEATURES = ((0.2, -0.3), (-0.4, 0.6))
TARGETS = ((1.0,), (0.0,))
FAMILIES = ("pc", "neutral_cpc", "active_cpc")
_WIRE_FIELDS = {
    "admission_expected_utf8",
    "admission_proposed_utf8",
    "admission_observation",
    "expected_utf8",
    "proposed_utf8",
    "observation",
}


def _object(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != keys:
        raise ValueError("native completion wire differs from exact supported schema")
    return value


def _identity(value: object) -> RecoveryProcessIdentity:
    record = _object(value, {"pid", "created_filetime"})
    return RecoveryProcessIdentity(record["pid"], record["created_filetime"])


def _observation(value: object) -> RecoveryHostObservation:
    record = _object(
        value,
        {
            "clock_epoch",
            "now_ns",
            "rss_bytes",
            "peak_rss_bytes",
            "observer",
            "previous_owner",
            "previous_owner_ended",
        },
    )
    return RecoveryHostObservation(
        record["clock_epoch"],
        record["now_ns"],
        record["rss_bytes"],
        record["peak_rss_bytes"],
        _identity(record["observer"]),
        None if record["previous_owner"] is None else _identity(record["previous_owner"]),
        record["previous_owner_ended"],
    )


def _record(value: object) -> AuthorityRecord:
    if type(value) is not str or len(value) > MAX_RECORD_BYTES:
        raise ValueError("native completion requires bounded canonical authority text")
    try:
        return decode_authority(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError("native completion authority text is not UTF-8") from error


def _validate(change: AuthorityCompletion) -> None:
    _schema(change, AuthorityCompletion)
    _schema(change.admission, AuthorityChange)
    for record in (
        change.admission.expected,
        change.admission.proposed,
        change.expected,
        change.proposed,
    ):
        _authority(record)
    _host(change.admission.observation)
    _host(change.observation)
    validate_authority_completion(change)


def completion_to_wire(change: AuthorityCompletion) -> dict[str, Any]:
    """Serialize validated actual accounting; the caller supplies its trusted origin."""
    _validate(change)
    return dict(
        admission_expected_utf8=encode_authority(change.admission.expected).decode("utf-8"),
        admission_proposed_utf8=encode_authority(change.admission.proposed).decode("utf-8"),
        admission_observation=asdict(change.admission.observation),
        expected_utf8=encode_authority(change.expected).decode("utf-8"),
        proposed_utf8=encode_authority(change.proposed).decode("utf-8"),
        observation=asdict(change.observation),
    )


def completion_from_wire(value: object) -> AuthorityCompletion:
    """Reconstruct and validate all six fields, without granting live authority."""
    record = _object(value, _WIRE_FIELDS)
    admission = AuthorityChange(
        _record(record["admission_expected_utf8"]),
        _record(record["admission_proposed_utf8"]),
        _observation(record["admission_observation"]),
    )
    change = AuthorityCompletion(
        admission,
        _record(record["expected_utf8"]),
        _record(record["proposed_utf8"]),
        _observation(record["observation"]),
    )
    _validate(change)
    return change
