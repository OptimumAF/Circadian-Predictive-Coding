"""Host RSS observation checks for benchmark memory reporting."""

from __future__ import annotations

from os import getpid

import pytest

from src.shared.process_memory import ProcessRssSampler, read_process_rss_bytes


def test_process_sampler_observes_live_allocation_beyond_feature_bytes() -> None:
    if read_process_rss_bytes() is None:
        pytest.skip("Process RSS is unsupported on this host")

    with ProcessRssSampler() as sampler:
        assert sampler.start_bytes is not None
        tiny_feature_bytes = 64
        held_allocation = bytearray(32 * 1024 * 1024)
        for offset in range(0, len(held_allocation), 4096):
            held_allocation[offset] = 1
        sampler.sample()
        observed_peak = sampler.peak_bytes

    assert observed_peak is not None
    assert observed_peak - sampler.start_bytes >= 1024 * 1024
    assert observed_peak - sampler.start_bytes > tiny_feature_bytes


def test_process_sampler_reports_unavailable_without_a_zero_peak() -> None:
    with ProcessRssSampler(read_rss_bytes=lambda: None) as sampler:
        assert sampler.sample() is None
        with pytest.raises(RuntimeError, match="unavailable"):
            sampler.snapshot()

    assert sampler.start_bytes is None
    assert sampler.peak_bytes is None
    assert sampler.sample_count == 0
    for invalid_interval in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive and finite"):
            ProcessRssSampler(interval_seconds=invalid_interval)


def test_process_sampler_snapshot_keeps_one_segment_baseline_and_peak() -> None:
    readings = iter((100, 140, 120))
    with ProcessRssSampler(interval_seconds=60.0, read_rss_bytes=lambda: next(readings)) as sampler:
        sampler.sample()
        segment = sampler.snapshot()

    assert segment.pid == getpid()
    assert segment.start_bytes == 100
    assert segment.peak_bytes == 140
    assert segment.sample_count == 2
    assert segment.interval_seconds == 60.0
