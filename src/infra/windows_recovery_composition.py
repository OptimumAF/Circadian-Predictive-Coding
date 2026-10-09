"""Compose original retained Windows registrations and existing private authority.

Inputs are trusted independently known state, journal path and concrete retained
registrations. Owns those registrations only after record/type validation. Does
not launch/repin/terminate processes, create authority, restore payloads or resume
uncertain work. Native API injection remains a private deterministic test hook.
"""

from pathlib import Path
import sys

from src.app.recovery_coordinator import RecoveryCoordinator
from src.core.recovery_authority import AuthorityRecord, validate_authority_record
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_process_handles import WindowsProcessHandle
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


def _require_original(
    known: AuthorityRecord, anchor: WindowsProcessHandle, worker: WindowsProcessHandle
) -> None:
    if sys.platform != "win32":
        raise OSError("Windows recovery composition is unavailable on this platform")
    if known.metadata.stopped or known.metadata.uncertain_work:
        raise ValueError("stopped or uncertain original authority cannot compose resumed work")
    if anchor.identity != known.anchor or worker.identity != known.worker:
        raise ValueError("composition retained original process identity differs")
    if worker._api is not anchor._api:
        raise ValueError("composition requires one supported original observation API instance")
    if anchor._api.current_pid() != known.anchor.pid:
        raise ValueError("composition anchor must be the physical surviving coordinator")
    if anchor.is_ended() is not False or worker.is_ended() is not False:
        raise ValueError("composition requires original live registered coordinator and worker")


def _close_failed_registrations(
    original: BaseException, anchor: WindowsProcessHandle, worker: WindowsProcessHandle
) -> None:
    errors = [original]
    seen: set[int] = set()
    for registration in (anchor, worker):
        if id(registration) in seen:
            continue
        seen.add(id(registration))
        try:
            registration.close()
        except BaseException as error:
            errors.append(error)
    if len(errors) > 1:
        raise BaseExceptionGroup(
            "Windows recovery composition and registration cleanup failed", errors
        )


def compose_windows_recovery(
    known_authority: AuthorityRecord,
    journal_path: str | Path,
    anchor: object,
    worker: object,
) -> RecoveryCoordinator:
    """Transfer exact retained registrations; close owned handles on any later failure.

    Why this: the composition root binds existing native adapters to inner ports
    without allowing persisted/checkpoint metadata to mint a new original session.
    Invalid initial records/types leave registration ownership with the caller.
    """
    validate_authority_record(known_authority)
    if type(anchor) is not WindowsProcessHandle or type(worker) is not WindowsProcessHandle:
        raise ValueError("composition requires exact retained Windows process registrations")
    try:
        _require_original(known_authority, anchor, worker)
        # Preserve independently saved high water when a fresh native reading is lower.
        observer = WindowsRecoveryObserver(
            anchor, peak_rss_bytes=known_authority.metadata.usage.peak_rss_bytes
        )
        if observer._identity != known_authority.anchor:
            raise ValueError("composition observing physical process differs from original anchor")
        journal = SqliteRecoveryJournal(journal_path, known_authority)
        if journal.read() != known_authority:
            raise ValueError(
                "composition existing journal differs from independent original witness"
            )
        return RecoveryCoordinator(known_authority, journal, observer, anchor, worker)
    except BaseException as error:
        _close_failed_registrations(error, anchor, worker)
        raise
