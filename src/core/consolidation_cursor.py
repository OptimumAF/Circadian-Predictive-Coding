"""Complete immutable consolidation ledger; no native state or restore authority.

Contains consumed attempt identities, committed native diagnostics and original
owner flags/counters. Attempt identities are a set in the owner; their canonical
ordering does not invent a timeline for failed transforms. No IO or copying here.
"""

from dataclasses import dataclass

from src.core.actor_ports import AppliedConsolidation
from src.core.experience import require_identifier
from src.core.learner_ports import TrainingDiagnostic

CURSOR_FIELDS = frozenset(
    {
        "format_version",
        "actor_version",
        "learner_version",
        "consolidation_limit",
        "attempted_ids",
        "consolidations",
        "stopped",
        "retired",
        "revision",
        "payload_ready",
    }
)
RECEIPT_FIELDS = frozenset(
    {"event_id", "actor_version", "learner_version", "attempt_number", "diagnostic"}
)
DIAGNOSTIC_FIELDS = frozenset({"definition", "value"})


def require_consolidation_record(value, kind, names) -> None:
    if type(value) is not kind or vars(value).keys() != names:
        raise ValueError("consolidation record differs from complete supported schema")


def require_consolidation_counter(value) -> None:
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("consolidation counter requires a bounded nonnegative exact integer")


def validate_consolidation_diagnostic(diagnostic) -> None:
    require_consolidation_record(diagnostic, TrainingDiagnostic, DIAGNOSTIC_FIELDS)
    # Native isfinite accepts wider Python integers;keep their exact value/type.
    if type(diagnostic.value) is int and diagnostic.value.bit_length() > 1024:
        raise ValueError("diagnostic integer exceeds native finite range")
    try:
        TrainingDiagnostic.__post_init__(diagnostic)
    except OverflowError as error:
        raise ValueError("diagnostic integer exceeds native finite range") from error


def _validate_receipts(cursor) -> None:
    seen, previous = set(), 0
    for receipt in cursor.consolidations:
        require_consolidation_record(receipt, AppliedConsolidation, RECEIPT_FIELDS)
        require_identifier(receipt.event_id, "consolidation event_id")
        require_consolidation_counter(receipt.attempt_number)
        validate_consolidation_diagnostic(receipt.diagnostic)
        if (
            receipt.event_id not in cursor.attempted_ids
            or receipt.event_id in seen
            or not previous < receipt.attempt_number <= len(cursor.attempted_ids)
            or type(receipt.actor_version) is not str
            or receipt.actor_version != cursor.actor_version
            or type(receipt.learner_version) is not str
            or receipt.learner_version != cursor.learner_version
        ):
            raise ValueError("consolidation references,versions or attempt numbering differ")
        seen.add(receipt.event_id)
        previous = receipt.attempt_number


@dataclass(frozen=True)
class ConsolidationCursor:
    format_version: int
    actor_version: str
    learner_version: str
    consolidation_limit: int
    attempted_ids: tuple[str, ...]
    consolidations: tuple[AppliedConsolidation, ...]
    stopped: bool
    retired: bool
    revision: int
    payload_ready: bool

    def __post_init__(self) -> None:
        require_consolidation_record(self, ConsolidationCursor, CURSOR_FIELDS)
        if type(self.format_version) is not int or self.format_version != 1:
            raise ValueError("unsupported consolidation cursor version")
        for value in (self.actor_version, self.learner_version):
            require_identifier(value, "consolidation version")
        if self.actor_version == self.learner_version:
            raise ValueError("actor and candidate versions must differ")
        for counter in (self.consolidation_limit, self.revision):
            require_consolidation_counter(counter)
        if any(
            type(value) is not bool for value in (self.stopped, self.retired, self.payload_ready)
        ):
            raise ValueError("consolidation owner flags require exact booleans")
        if (
            type(self.attempted_ids) is not tuple
            or type(self.consolidations) is not tuple
            or len(self.attempted_ids) > self.consolidation_limit
            or len(self.consolidations) > len(self.attempted_ids)
        ):
            raise ValueError("consolidation histories exceed original consumed allowance")
        for event in self.attempted_ids:
            require_identifier(event, "attempted consolidation")
        if self.attempted_ids != tuple(sorted(set(self.attempted_ids))):
            raise ValueError("consumed consolidation identities must be unique and canonical")
        _validate_receipts(self)
