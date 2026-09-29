"""Unit gate for fixed-feature CUDA checkpoint allocator boundaries."""

from __future__ import annotations

from os import getpid
from types import SimpleNamespace

from src.app import matched_head_benchmark


def test_cuda_checkpoint_segment_reads_starts_before_one_peak_reset() -> None:
    calls: list[str] = []

    class FakeCuda:
        def synchronize(self, device: str) -> None:
            calls.append(f"sync:{device}")

        def memory_allocated(self, device: str) -> int:
            calls.append(f"allocated:{device}")
            return 1024

        def memory_reserved(self, device: str) -> int:
            calls.append(f"reserved:{device}")
            return 1536

        def reset_peak_memory_stats(self, device: str) -> None:
            calls.append(f"reset:{device}")

        def max_memory_allocated(self, device: str) -> int:
            calls.append(f"peak_allocated:{device}")
            return 2048

        def max_memory_reserved(self, device: str) -> int:
            calls.append(f"peak_reserved:{device}")
            return 4096

    fake_torch = SimpleNamespace(cuda=FakeCuda(), device=lambda value: value)
    start = matched_head_benchmark._begin_cuda_allocator_segment(fake_torch, "cuda:0")
    segment = matched_head_benchmark._snapshot_cuda_allocator_segment(fake_torch, "cuda:0", start)

    assert start == (1024, 1536)
    assert segment is not None
    assert segment.pid == getpid()
    assert segment.device == "cuda:0"
    assert segment.allocated_start_bytes == 1024
    assert segment.reserved_start_bytes == 1536
    assert segment.allocated_peak_bytes == 2048
    assert segment.reserved_peak_bytes == 4096
    assert calls == [
        "sync:cuda:0",
        "allocated:cuda:0",
        "reserved:cuda:0",
        "reset:cuda:0",
        "sync:cuda:0",
        "peak_allocated:cuda:0",
        "peak_reserved:cuda:0",
    ]
