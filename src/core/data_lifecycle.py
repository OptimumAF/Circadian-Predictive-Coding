"""Declared local consent/provenance and lifetime admission limits.

Metadata only: no payload storage, provenance attestation, deletion or unlearning.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.experience import SampleKey, require_identifier


@dataclass(frozen=True)
class DataProvenance:
    source_id: str
    subject_id: str
    verified: bool
    synthetic: bool

    def __post_init__(self) -> None:
        require_identifier(self.source_id, "source_id")
        require_identifier(self.subject_id, "subject_id")
        if type(self.verified) is not bool or type(self.synthetic) is not bool:
            raise ValueError("provenance flags must be exact booleans")


@dataclass(frozen=True)
class DataConsent:
    training: bool
    replay: bool

    def __post_init__(self) -> None:
        if type(self.training) is not bool or type(self.replay) is not bool:
            raise ValueError("consent flags must be exact booleans")


@dataclass(frozen=True)
class LifecycleDeclaration:
    key: SampleKey
    provenance: DataProvenance
    consent: DataConsent
    retention: Literal["replay", "transient", "audit_only"]

    def __post_init__(self) -> None:
        if type(self.key) is not tuple or len(self.key) != 2:
            raise ValueError("declaration requires an episode/sample key")
        for value in self.key:
            require_identifier(value, "sample key")
        if type(self.provenance) is not DataProvenance or type(self.consent) is not DataConsent:
            raise ValueError("declaration requires typed provenance and consent")
        if type(self.retention) is not str or self.retention not in (
            "replay",
            "transient",
            "audit_only",
        ):
            raise ValueError("unknown retention category")


@dataclass(frozen=True)
class LifecycleLimits:
    max_approved_records: int
    max_replay_records: int
    allow_synthetic: bool = False
    allow_unverified: bool = False

    def __post_init__(self) -> None:
        for value in (self.max_approved_records, self.max_replay_records):
            if type(value) is not int or value < 0:
                raise ValueError("lifetime quotas must be nonnegative integers")
        if type(self.allow_synthetic) is not bool or type(self.allow_unverified) is not bool:
            raise ValueError("policy flags must be exact booleans")

    def require_supported(self, declaration: LifecycleDeclaration) -> None:
        if type(declaration) is not LifecycleDeclaration:
            raise ValueError("admission requires a typed lifecycle declaration")
        if declaration.retention != "replay":
            raise ValueError("only replay retention is supported before native purge semantics")
        if not declaration.consent.training or not declaration.consent.replay:
            raise ValueError("training and replay consent are both required")
        if declaration.provenance.synthetic and not self.allow_synthetic:
            raise ValueError("synthetic admission is disabled")
        if not declaration.provenance.verified and not self.allow_unverified:
            raise ValueError("unverified admission is disabled")
