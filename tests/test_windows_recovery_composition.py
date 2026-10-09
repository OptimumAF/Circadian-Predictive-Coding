"""Concrete Windows adapters with deterministic documented-API fakes/local SQLite."""

from dataclasses import replace
import sqlite3
import sys

import pytest

from src.core.recovery_coordination import RecoveryCosts
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.sqlite_recovery_journal import SqliteRecoveryJournal
from src.infra.windows_recovery_composition import compose_windows_recovery
from src.infra.windows_recovery_observer import WindowsRecoveryObserver
from test_recovery_authority import fixture as authority_fixture
from test_windows_recovery_observer import Kernel

windows_composition = pytest.mark.skipif(
    sys.platform != "win32", reason="supported Windows composition runtime only"
)


def fixture(tmp_path):
    record, _ = authority_fixture()
    kernel = Kernel()
    kernel.GetCurrentProcessId.callback = lambda: 10
    kernel.now = 6
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: kernel.rss)
    anchor = WindowsProcessHandle.pin(10, expected=record.anchor, api=api)
    worker = WindowsProcessHandle.pin(20, expected=record.worker, api=api)
    journal = SqliteRecoveryJournal.create(tmp_path / "authority.db", record)
    return record, kernel, api, anchor, worker, journal


@pytest.mark.parametrize("string_path", [False, True])
@windows_composition
def test_should_compose_concrete_adapters_and_publish_under_original_durable_authority(
    tmp_path, string_path
):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    coordinator = compose_windows_recovery(
        record, str(journal.path) if string_path else journal.path, anchor, worker
    )
    assert len(kernel.handles) == 2
    published = []

    def publish(value):
        current = journal.read()
        assert current.anchor == record.anchor and current.worker == record.worker
        assert current.metadata.started_ns == record.metadata.started_ns
        assert current.metadata.limits == record.metadata.limits
        assert current.metadata.manifest == record.metadata.manifest
        assert current.metadata.observed_ns == kernel.now * 100
        assert current.metadata.usage.copied_bytes == 41
        published.append(value)

    assert (
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: "prepared", publish) == "prepared"
    )
    assert published == ["prepared"]
    coordinator.close()
    coordinator.close()
    stopped = journal.read()
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 41
    assert stopped.metadata.usage.updates_admitted == stopped.metadata.usage.updates_completed == 3
    assert not kernel.handles
    assert all(rights == 0x101000 and inherit is False for rights, inherit, _ in kernel.opens)


@pytest.mark.parametrize(
    "fault",
    ["anchor", "worker", "current", "api", "ended-anchor", "ended-worker", "unknown", "creation"],
)
@windows_composition
def test_should_refuse_changed_or_unknown_original_registrations_and_close_owned_handles(
    tmp_path, fault
):
    record, kernel, api, anchor, worker, journal = fixture(tmp_path)
    expected = record
    if fault == "anchor":
        expected = replace(
            record,
            anchor=replace(record.anchor, pid=99),
            metadata=replace(record.metadata, clock_epoch="win-interrupt-anchor-v1:99:100"),
        )
    elif fault == "worker":
        expected = replace(record, worker=replace(record.worker, created_filetime=999))
    elif fault == "current":
        kernel.GetCurrentProcessId.callback = lambda: 30
    elif fault == "api":
        worker._api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: kernel.rss)
    elif fault == "ended-anchor":
        kernel.processes[10][1] = True
    elif fault == "ended-worker":
        kernel.processes[20][1] = True
    elif fault == "unknown":
        kernel.wait_override = 0xFFFFFFFF
    else:
        kernel.processes[20][0] = 999
    with pytest.raises((ValueError, OSError)):
        compose_windows_recovery(expected, journal.path, anchor, worker)
    assert not kernel.handles and journal.read() == record


def test_should_refuse_unsupported_platform_and_close_owned_registrations(tmp_path, monkeypatch):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    monkeypatch.setattr("src.infra.windows_recovery_composition.sys.platform", "unsupported")
    with pytest.raises(OSError, match="Windows"):
        compose_windows_recovery(record, journal.path, anchor, worker)
    assert not kernel.handles and journal.read() == record


@pytest.mark.parametrize("bad_record", [False, True])
def test_should_leave_registration_ownership_with_caller_when_initial_types_are_invalid(
    tmp_path, bad_record
):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    if bad_record:
        object.__setattr__(record.metadata.usage, "copied_bytes", False)
    try:
        with pytest.raises(ValueError):
            compose_windows_recovery(
                record, journal.path, anchor, object() if not bad_record else worker
            )
        assert len(kernel.handles) == 2
    finally:
        anchor.close()
        worker.close()
    assert not kernel.handles


@pytest.mark.parametrize("fault", ["missing", "stale", "stopped", "uncertain"])
@windows_composition
def test_should_require_existing_exact_independent_journal_and_never_bootstrap(tmp_path, fault):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    expected = record
    path = journal.path
    if fault == "missing":
        path = tmp_path / "missing.db"
    elif fault == "stale":
        from src.core.recovery_reporting import plan_report

        assert journal.report(plan_report(record, None, terminal=True))
    elif fault == "stopped":
        expected = replace(record, metadata=replace(record.metadata, stopped=True))
    else:
        expected = replace(record, metadata=replace(record.metadata, uncertain_work=True))
    before = journal.path.read_bytes()
    with pytest.raises((ValueError, OSError, sqlite3.Error)):
        compose_windows_recovery(expected, path, anchor, worker)
    assert not kernel.handles and journal.path.read_bytes() == before
    assert not (tmp_path / "missing.db").exists()


@pytest.mark.parametrize("fault", ["expired", "rss", "clock", "observe-exception"])
@windows_composition
def test_should_close_failed_initial_composition_and_preserve_terminal_negative_facts(
    tmp_path, fault
):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    if fault == "expired":
        kernel.now = 11
    elif fault == "rss":
        kernel.rss = 1025
    elif fault == "clock":
        kernel.now = 0
    else:

        def fail(counter):
            raise KeyboardInterrupt("clock unavailable")

        kernel.QueryInterruptTimePrecise.callback = fail
    with pytest.raises((ValueError, KeyboardInterrupt)):
        compose_windows_recovery(record, journal.path, anchor, worker)
    assert not kernel.handles
    stopped = journal.read()
    assert stopped.metadata.stopped and stopped.metadata.usage.copied_bytes == 40
    assert stopped.metadata.usage.updates_admitted == stopped.metadata.usage.updates_completed == 3
    if fault == "expired":
        assert stopped.metadata.observed_ns == 1100
    if fault == "rss":
        assert stopped.metadata.usage.peak_rss_bytes == 1025


@windows_composition
def test_should_close_duplicate_owned_registration_once_on_binding_failure(tmp_path):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        compose_windows_recovery(record, journal.path, anchor, anchor)
    assert len(kernel.handles) == 1 and len(kernel.closed) == 1
    worker.close()
    assert not kernel.handles and journal.read() == record


@windows_composition
def test_should_refuse_observing_process_identity_changed_during_composition(tmp_path):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    calls = []

    def current():
        calls.append(True)
        return 10 if len(calls) == 1 else 30

    kernel.GetCurrentProcessId.callback = current
    with pytest.raises(ValueError, match="observing physical"):
        compose_windows_recovery(record, journal.path, anchor, worker)
    assert not kernel.handles and len(kernel.closed) == 3
    assert journal.read() == record


@windows_composition
def test_should_report_primary_and_every_cleanup_failure_without_hiding_uncertain_handles(tmp_path):
    record, kernel, api, anchor, worker, journal = fixture(tmp_path)
    kernel.GetCurrentProcessId.callback = lambda: 30
    kernel.close_ok = False
    with pytest.raises(BaseExceptionGroup) as caught:
        compose_windows_recovery(record, journal.path, anchor, worker)
    assert len(caught.value.exceptions) == 3
    assert len(kernel.handles) == 2
    assert journal.read() == record
    # These are injected fake close failures: clear fake handles explicitly after the assertions.
    kernel.close_ok = True
    for handle in list(kernel.handles):
        api.close_handle(handle)
    assert not kernel.handles


@windows_composition
def test_should_preserve_independent_saved_rss_peak_when_fresh_composition_reads_lower_rss(
    tmp_path,
):
    record, kernel, _, anchor, worker, journal = fixture(tmp_path)
    known = replace(
        record,
        metadata=replace(record.metadata, usage=replace(record.metadata.usage, peak_rss_bytes=640)),
    )
    path = tmp_path / "known-peak.db"
    saved = SqliteRecoveryJournal.create(path, known)
    with compose_windows_recovery(known, path, anchor, worker) as coordinator:
        assert saved.read().metadata.usage.peak_rss_bytes == 640
        kernel.rss = 480
        coordinator.execute(RecoveryCosts(copy_bytes=1), lambda: None, lambda v: None)
        assert saved.read().metadata.usage.peak_rss_bytes == 640
    assert saved.read().metadata.stopped and not kernel.handles


@pytest.mark.parametrize("value", [False, -1, 2**63, "640"])
def test_should_refuse_invalid_observer_peak_seed_without_taking_registration_ownership(
    tmp_path, value
):
    _, kernel, _, anchor, worker, _ = fixture(tmp_path)
    try:
        with pytest.raises(ValueError):
            WindowsRecoveryObserver(anchor, peak_rss_bytes=value)
        assert len(kernel.handles) == 2
    finally:
        anchor.close()
        worker.close()
    assert not kernel.handles
