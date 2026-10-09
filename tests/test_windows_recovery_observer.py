"""Deterministic documented-API faults; no real process or native model work."""

import ctypes
from dataclasses import replace

import pytest

from src.core.recovery_observation import (
    RecoveryProcessIdentity,
    anchored_clock_epoch,
)
from src.infra.windows_process_handles import WindowsProcessHandle, WindowsRecoveryApi
from src.infra.windows_recovery_observer import WindowsRecoveryObserver


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class Kernel:
    def __init__(self):
        self.processes = {10: [100, False], 20: [200, False], 30: [300, False]}
        self.handles = {}
        self.next_handle = 500
        self.closed = []
        self.wait_override = None
        self.times_ok = True
        self.close_ok = True
        self.now = 1000
        self.rss = 512
        self.opens = []
        self.OpenProcess = Function(self.open)
        self.CloseHandle = Function(self.close)
        self.GetProcessTimes = Function(self.times)
        self.WaitForSingleObject = Function(self.wait)
        self.GetCurrentProcessId = Function(lambda: 30)
        self.QueryInterruptTimePrecise = Function(self.clock)

    def open(self, rights, inherit, pid):
        self.opens.append((rights, inherit, pid))
        if pid not in self.processes:
            return 0
        self.next_handle += 1
        self.handles[self.next_handle] = self.processes[pid]
        return self.next_handle

    def close(self, handle):
        if not self.close_ok:
            return 0
        self.closed.append(handle)
        del self.handles[handle]
        return 1

    def times(self, handle, creation, exit_time, kernel, user):
        if not self.times_ok:
            return 0
        value = self.handles[handle][0]
        creation._obj.low = value & 0xFFFFFFFF
        creation._obj.high = value >> 32
        return 1

    def wait(self, handle, timeout):
        assert timeout == 0
        if self.wait_override is not None:
            return self.wait_override
        return 0 if self.handles[handle][1] else 258

    def clock(self, counter):
        counter._obj.value = self.now


def fixture():
    kernel = Kernel()
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: kernel.rss)
    anchor = WindowsProcessHandle.pin(10, api=api)
    owner = WindowsProcessHandle.pin(20, api=api)
    return kernel, api, anchor, owner


def test_should_pin_noninherited_query_sync_handles_and_observe_original_epoch():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        result = observer.observe(owner)
        assert result.clock_epoch == "win-interrupt-anchor-v1:10:100"
        assert result.now_ns == 100000 and result.rss_bytes == result.peak_rss_bytes == 512
        assert result.observer == RecoveryProcessIdentity(30, 300)
        assert result.previous_owner == owner.identity and not result.previous_owner_ended
        assert all(rights == 0x101000 and not inherit for rights, inherit, _ in kernel.opens)
        assert ctypes.sizeof(api.filetime_type) == 8
    finally:
        owner.close()
        anchor.close()
    assert not kernel.handles


def test_should_keep_pinned_identity_after_exit_and_pid_reuse():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        original = owner.identity
        kernel.processes[20][1] = True
        kernel.processes[20] = [900, False]
        result = observer.observe(owner)
        assert result.previous_owner == original and result.previous_owner_ended
        with pytest.raises(ValueError, match="identity"):
            WindowsProcessHandle.pin(20, expected=original, api=api)
        assert len(kernel.handles) == 2
    finally:
        owner.close()
        anchor.close()


@pytest.mark.parametrize("pid", [0, -1, True, 2**32, None])
def test_should_refuse_invalid_pid_before_open(pid):
    kernel = Kernel()
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: 512)
    with pytest.raises(ValueError):
        WindowsProcessHandle.pin(pid, api=api)
    assert kernel.opens == []


@pytest.mark.parametrize(
    "api_name",
    [
        "OpenProcess",
        "CloseHandle",
        "GetProcessTimes",
        "WaitForSingleObject",
        "GetCurrentProcessId",
        "QueryInterruptTimePrecise",
    ],
)
def test_should_refuse_missing_api_before_any_handle_open(api_name):
    kernel = Kernel()
    delattr(kernel, api_name)
    with pytest.raises(OSError, match="unavailable"):
        WindowsRecoveryApi(_kernel=kernel)
    assert not kernel.opens and not kernel.handles


def test_should_refuse_unsupported_host_without_loading_windows(monkeypatch):
    monkeypatch.setattr("src.infra.windows_process_handles.sys.platform", "unsupported")
    with pytest.raises(OSError, match="Windows"):
        WindowsRecoveryApi()


def test_should_bind_precise_clock_from_documented_api_set_without_changing_clock_policy(
    monkeypatch,
):
    kernel = Kernel()
    clock = kernel.QueryInterruptTimePrecise
    del kernel.QueryInterruptTimePrecise
    names = []

    class Timing:
        QueryInterruptTimePrecise = clock

    def load(name="kernel32"):
        names.append(name)
        return kernel if name == "kernel32" else Timing()

    monkeypatch.setattr("src.infra.windows_process_handles._load_kernel", load)
    api = WindowsRecoveryApi(rss_reader=lambda: 512)
    assert names == ["kernel32", "api-ms-win-core-realtime-l1-1-1.dll"]
    assert api.interrupt_ns() == kernel.now * 100
    assert not kernel.opens


@pytest.mark.parametrize("failure", ["missing", "times", "ended", "wait"])
def test_should_close_partial_registration_and_refuse_unknown_or_ended_process(failure):
    kernel = Kernel()
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: 512)
    if failure == "missing":
        del kernel.processes[20]
    if failure == "times":
        kernel.times_ok = False
    if failure == "ended":
        kernel.processes[20][1] = True
    if failure == "wait":
        kernel.wait_override = 0xFFFFFFFF
    with pytest.raises((ValueError, OSError)):
        WindowsProcessHandle.pin(20, api=api)
    assert not kernel.handles


@pytest.mark.parametrize("status", [128, 0xFFFFFFFF, 55])
def test_should_refuse_unknown_wait_state_without_inventing_death(status):
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        kernel.wait_override = status
        with pytest.raises(OSError):
            observer.observe(owner)
        kernel.wait_override = None
        with pytest.raises(ValueError, match="failed"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


@pytest.mark.parametrize("rss", [None, 0, -1, True, 2**63, 1.5])
def test_should_poison_failed_or_invalid_rss_observation_without_zero_default(rss):
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        kernel.rss = rss
        with pytest.raises((ValueError, OSError)):
            observer.observe(owner)
        kernel.rss = 512
        with pytest.raises(ValueError, match="failed"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


def test_should_preserve_clock_and_observed_rss_high_water_without_restarting_origin():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        first = observer.observe(owner)
        kernel.now += 1000000
        kernel.rss = 1024
        second = observer.observe(owner)
        kernel.now += 100
        kernel.rss = 600
        third = observer.observe(owner)
        assert second.now_ns - first.now_ns == 100000000
        assert third.peak_rss_bytes == 1024 and third.rss_bytes == 600
        assert first.clock_epoch == second.clock_epoch == third.clock_epoch
    finally:
        owner.close()
        anchor.close()


def test_should_refuse_backwards_time_and_keep_failure_terminal():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        observer.observe(owner)
        kernel.now -= 1
        with pytest.raises(ValueError, match="backwards"):
            observer.observe(owner)
        kernel.now += 2
        with pytest.raises(ValueError, match="failed"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


@pytest.mark.parametrize("during", [False, True])
def test_should_refuse_ended_anchor_before_or_during_observation(during):
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        if during:

            def reading(counter):
                counter._obj.value = kernel.now
                kernel.processes[10][1] = True

            kernel.QueryInterruptTimePrecise.callback = reading
        else:
            kernel.processes[10][1] = True
        with pytest.raises(ValueError, match="anchor"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


def test_should_refuse_anchor_identity_change_and_corrupt_expected_identity():
    kernel, api, anchor, owner = fixture()
    try:
        expected = replace(owner.identity)
        object.__setattr__(expected, "pid", True)
        before = len(kernel.opens)
        with pytest.raises(ValueError):
            WindowsProcessHandle.pin(20, expected=expected, api=api)
        assert len(kernel.opens) == before
        observer = WindowsRecoveryObserver(anchor)
        kernel.processes[10][0] = 101
        with pytest.raises(ValueError, match="identity"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


def test_should_refuse_closed_foreign_or_anchor_owner_handles():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        with pytest.raises(ValueError):
            observer.observe(anchor)
    finally:
        owner.close()
        anchor.close()
    with pytest.raises(ValueError, match="closed"):
        owner.is_ended()
    owner.close()  # Idempotent successful close.
    other_kernel, other_api, other_anchor, other_owner = fixture()
    try:
        with pytest.raises(ValueError):
            WindowsRecoveryObserver(object())
        with pytest.raises(ValueError, match="closed"):
            WindowsRecoveryObserver(anchor)
        observer = WindowsRecoveryObserver(other_anchor)
        with pytest.raises(ValueError, match="registered"):
            observer.observe(owner)
    finally:
        other_owner.close()
        other_anchor.close()


def test_should_report_close_failure_and_deny_further_handle_use():
    kernel, api, anchor, owner = fixture()
    try:
        kernel.close_ok = False
        with pytest.raises(OSError):
            owner.close()
        with pytest.raises(ValueError, match="closed"):
            owner.is_ended()
        kernel.close_ok = True
    finally:
        # Fixture cleanup of simulated native failure; real failure is reported.
        kernel.handles.pop(owner._handle, None)
        anchor.close()


@pytest.mark.parametrize("bad", [0, 2**63 // 100 + 1])
def test_should_refuse_unavailable_or_unrepresentable_native_clock(bad):
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)
        kernel.now = bad
        with pytest.raises(ValueError, match="clock"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


def test_should_refuse_raw_handle_construction_and_subclasses_before_api_access():
    kernel = Kernel()
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: 512)
    with pytest.raises(ValueError, match="registration"):
        WindowsProcessHandle(123, RecoveryProcessIdentity(10, 100), api)

    class Unsupported(WindowsProcessHandle):
        pass

    with pytest.raises(ValueError, match="subclass"):
        Unsupported.pin(10, api=api)
    assert not kernel.opens


def test_should_close_owned_registration_handle_on_base_exception():
    kernel = Kernel()
    api = WindowsRecoveryApi(_kernel=kernel, rss_reader=lambda: 512)

    def failed(*args):
        raise KeyboardInterrupt("interrupted query")

    kernel.GetProcessTimes.callback = failed
    with pytest.raises(KeyboardInterrupt):
        WindowsProcessHandle.pin(20, api=api)
    assert not kernel.handles


def test_should_keep_observation_failure_terminal_and_release_nested_gate():
    kernel, api, anchor, owner = fixture()
    try:
        observer = WindowsRecoveryObserver(anchor)

        def nested(counter):
            observer.observe(owner)

        kernel.QueryInterruptTimePrecise.callback = nested
        with pytest.raises(ValueError, match="busy"):
            observer.observe(owner)
        assert not observer._gate.locked()
        with pytest.raises(ValueError, match="failed"):
            observer.observe(owner)
    finally:
        owner.close()
        anchor.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("now_ns", True),
        ("now_ns", -1),
        ("rss_bytes", 0),
        ("peak_rss_bytes", 1),
        ("clock_epoch", " epoch"),
        ("previous_owner_ended", 1),
        ("previous_owner", None),
    ],
)
def test_should_validate_every_inner_observation_field(field, value):
    kernel, api, anchor, owner = fixture()
    try:
        observation = WindowsRecoveryObserver(anchor).observe(owner)
        with pytest.raises(ValueError):
            replace(observation, **{field: value})
    finally:
        owner.close()
        anchor.close()


def test_should_return_no_death_fact_without_a_registered_previous_owner():
    kernel, api, anchor, owner = fixture()
    try:
        observation = WindowsRecoveryObserver(anchor).observe()
        assert observation.previous_owner is None and observation.previous_owner_ended is None
        with pytest.raises(ValueError):
            replace(observation, previous_owner_ended=True)
        corrupt = RecoveryProcessIdentity(10, 100)
        object.__setattr__(corrupt, "created_filetime", False)
        with pytest.raises(ValueError):
            anchored_clock_epoch(corrupt)
    finally:
        owner.close()
        anchor.close()
