"""Original elapsed clock validates its prior before invoking opaque providers.

Five original numeric graphs retain birth-installed providers and cleanup ports.
Provider counts include construction, queueing and the original training update.
No final clock sample, native learner or new factory is introduced.
"""

from dataclasses import asdict, dataclass
import json
from typing import Callable

import pytest

import src.app.expiry_inbox_observation as observation
from test_automatic_inbox_expiry import (
    AutomaticWork,
    FIRST,
    _assert_authority,
    _assert_successful_raw_cleanup,
    _assert_unpublished,
    _authority,
    _fresh_automatic,
)
from test_expiry_release_proof import _assert_released


@pytest.fixture(scope="module")
def clock_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("clock-prior-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2), encoding="utf8"
        )
    assert work.graphs == work.updates == 5
    assert work.learners == 15 and work.forks == 10
    assert work.arrays == 30 and work.array_bytes == 360
    assert work.consumed_arrays == work.native_copy_arrays == 10
    assert work.consumed_bytes == work.native_copy_bytes == 120
    assert work.cleanup_calls == 4 and work.post_erase_probes == 2
    assert work.pending_consumed is None


@dataclass
class ClockPort:
    calls: int = 0
    armed_calls: int = 0
    action: Callable[[], float] | None = None

    def __call__(self):
        self.calls += 1
        if self.action is not None:
            self.armed_calls += 1
            return self.action()
        return 10.0


def _assert_due_error_preserves_raw(graph, before):
    assert graph.runtime._inbox._experiences[FIRST].features is not None
    assert graph.runtime._inbox._labels[FIRST].targets is not None
    assert graph.runtime._candidate.model.rows
    assert not graph.runtime._inbox._erased and not graph.owner._revoked_keys
    assert graph.life._retention_fault is True
    assert graph.life._failed is False and graph.runtime._stopped is False
    assert graph.ledger._poisoned is False
    _assert_authority(graph, before)
    if graph.history is not None:
        _assert_unpublished(graph, before)
    assert observation._CURRENT.get() is None and observation._ACCESS.get() is None
    assert not graph.life._time_gate.locked()
    _assert_released(graph)


def test_should_publish_original_timed_expiry_without_an_extra_elapsed_read(clock_work):
    provider = ClockPort()
    graph, ports = _fresh_automatic(clock_work, wall_clock=provider, max_retention_seconds=1000)
    assert provider.calls == 31
    before = _authority(graph)
    provider.action = lambda: 11.0
    graph.clock.advance_to(122)

    report = graph.life.expire()

    _assert_successful_raw_cleanup(graph, report, before)
    assert FIRST in graph.history._erased_records and not graph.history._records
    assert graph.life._last_seconds == 11.0
    assert provider.calls == 32 and provider.armed_calls == 1
    assert ports.fired == 0
    assert not graph.life._time_gate.locked()
    _assert_released(graph)


def test_should_reject_backwards_provider_even_when_it_clears_the_live_prior(clock_work):
    provider = ClockPort()
    graph, _ = _fresh_automatic(clock_work, wall_clock=provider, max_retention_seconds=1000)
    assert provider.calls == 31
    assert graph.life._elapsed() == 10.0 and provider.calls == 32
    before = _authority(graph)

    def hide_backwards():
        graph.life._last_seconds = None
        return 9.0

    provider.action = hide_backwards
    graph.clock.advance_to(122)

    with pytest.raises(ValueError, match="invalid or backwards"):
        graph.life.expire()

    assert provider.calls == 33 and provider.armed_calls == 1
    _assert_due_error_preserves_raw(graph, before)


def test_should_preserve_same_original_provider_error_with_history_off(clock_work):
    provider = ClockPort()
    graph, _ = _fresh_automatic(
        clock_work, history=False, wall_clock=provider, max_retention_seconds=1000
    )
    assert provider.calls == 29
    before = _authority(graph)
    primary = RuntimeError("original timed provider failed")

    def fail_provider():
        raise primary

    provider.action = fail_provider
    graph.clock.advance_to(122)

    with pytest.raises(RuntimeError) as failure:
        graph.life.expire()

    assert failure.value is primary
    assert provider.calls == 30 and provider.armed_calls == 1
    _assert_due_error_preserves_raw(graph, before)


def test_should_clean_untimed_history_off_without_calling_the_armed_provider(clock_work):
    provider = ClockPort()
    graph, _ = _fresh_automatic(clock_work, history=False, wall_clock=provider)
    assert provider.calls == 6
    before = _authority(graph)

    def forbidden_provider():
        raise RuntimeError("untimed expiry must not sample elapsed provider")

    provider.action = forbidden_provider
    graph.clock.advance_to(122)

    report = graph.life.expire()

    _assert_successful_raw_cleanup(graph, report, before)
    assert graph.history is None and graph.life._last_seconds is None
    assert provider.calls == 6 and provider.armed_calls == 0
    assert not graph.life._time_gate.locked()
    _assert_released(graph)


def test_should_refuse_invalid_prior_before_invoking_the_original_provider(clock_work):
    provider = ClockPort()
    graph, _ = _fresh_automatic(clock_work, wall_clock=provider, max_retention_seconds=1000)
    assert provider.calls == 31
    assert graph.life._elapsed() == 10.0 and provider.calls == 32
    before = _authority(graph)
    graph.life._last_seconds = float("nan")

    def forbidden_provider():
        raise RuntimeError("invalid prior must refuse before the opaque callback")

    provider.action = forbidden_provider
    graph.clock.advance_to(122)

    with pytest.raises(ValueError, match="invalid or backwards"):
        graph.life.expire()

    assert provider.calls == 32 and provider.armed_calls == 0
    _assert_due_error_preserves_raw(graph, before)
