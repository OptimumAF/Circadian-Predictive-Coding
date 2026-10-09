"""Inner bounded OS observation values and port; no native APIs or authority IO."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RecoveryProcessIdentity:
    pid: int
    created_filetime: int

    def __post_init__(self) -> None:
        if type(self.pid) is not int or not 0 < self.pid < 2**32:
            raise ValueError("process pid requires a positive exact DWORD")
        if type(self.created_filetime) is not int or not 0 < self.created_filetime < 2**64:
            raise ValueError("process creation identity requires an exact positive FILETIME")


def validate_process_identity(value: object) -> None:
    if type(value) is not RecoveryProcessIdentity:
        raise ValueError("process identity requires an exact typed record")
    RecoveryProcessIdentity.__post_init__(value)


def anchored_clock_epoch(anchor: RecoveryProcessIdentity) -> str:
    validate_process_identity(anchor)
    return f"win-interrupt-anchor-v1:{anchor.pid}:{anchor.created_filetime}"


class RecoveryProcessProbe(Protocol):
    @property
    def identity(self) -> RecoveryProcessIdentity: ...

    def is_ended(self) -> bool: ...


@dataclass(frozen=True)
class RecoveryHostObservation:
    clock_epoch: str
    now_ns: int
    rss_bytes: int
    peak_rss_bytes: int
    observer: RecoveryProcessIdentity
    previous_owner: RecoveryProcessIdentity | None
    previous_owner_ended: bool | None

    def __post_init__(self) -> None:
        if type(self.clock_epoch) is not str or not 0 < len(self.clock_epoch) <= 128:
            raise ValueError("observation requires a bounded epoch identifier")
        if self.clock_epoch.strip() != self.clock_epoch:
            raise ValueError("epoch identifier cannot contain surrounding whitespace")
        for name in ("now_ns", "rss_bytes", "peak_rss_bytes"):
            value = getattr(self, name)
            if type(value) is not int or not int(name != "now_ns") <= value < 2**63:
                raise ValueError(f"{name} requires a bounded exact observation")
        if self.peak_rss_bytes < self.rss_bytes:
            raise ValueError("RSS peak cannot be below current RSS")
        validate_process_identity(self.observer)
        if self.previous_owner is None:
            if self.previous_owner_ended is not None:
                raise ValueError("unregistered previous owner has no death observation")
        else:
            validate_process_identity(self.previous_owner)
            if type(self.previous_owner_ended) is not bool:
                raise ValueError("registered previous owner requires an exact ended flag")


class RecoveryObservationPort(Protocol):
    def observe(
        self, previous_owner: RecoveryProcessProbe | None = None
    ) -> RecoveryHostObservation: ...
