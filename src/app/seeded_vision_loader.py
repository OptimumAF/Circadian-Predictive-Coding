"""Replay a seeded image DataLoader from an epoch/batch checkpoint.

Inputs are a map-style DataLoader with a local sampler generator and a
trusted cursor. The caller owns dataset identity, model state, and storage;
this module never reads validation, guard, or final-test examples.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import random
from time import perf_counter
from typing import Any

import numpy as np


class SeededEpochTrainLoader:
    """Replay shuffle and augmentation streams for each model and epoch."""

    def __init__(self, torch: Any, loader: Any, seed: int) -> None:
        if getattr(loader, "generator", None) is None:
            raise ValueError("Seeded vision protocol requires a train loader generator.")
        self.torch = torch
        self.loader = loader
        self.seed = seed
        self.epoch = 0
        self._active_epoch: int | None = None
        self._epoch_entry_torch_state: Any | None = None
        self._next_batch_index = 0
        self._resume_state: SeededEpochLoaderState | None = None
        self.replay_seconds = 0.0

    def snapshot_state(self) -> SeededEpochLoaderState:
        """Record the next batch and the sampler's observed state."""
        self._require_replayable_loader()
        active = self._active_epoch is not None
        entry_state = self._epoch_entry_torch_state
        assert not active or entry_state is not None
        return SeededEpochLoaderState(
            format_version=1,
            seed=self.seed,
            epoch=self._active_epoch if self._active_epoch is not None else self.epoch,
            next_batch_index=self._next_batch_index if active else 0,
            batch_count=len(self.loader),
            batch_size=self.loader.batch_size,
            num_workers=self.loader.num_workers,
            prefetch_factor=self.loader.prefetch_factor,
            drop_last=self.loader.drop_last,
            sampler_type=f"{type(self.loader.sampler).__module__}.{type(self.loader.sampler).__qualname__}",
            generator_state=(self.loader.generator.get_state().clone() if active else None),
            epoch_entry_torch_state=(entry_state.clone() if entry_state is not None else None),
            torch_cpu_random_state=self.torch.get_rng_state().clone() if active else None,
            python_random_state=deepcopy(random.getstate()) if active else None,
            numpy_random_state=deepcopy(np.random.get_state()) if active else None,
        )

    def restore_state(self, state: SeededEpochLoaderState) -> None:
        """Stage a cursor; replay and verify skipped batches on first iteration."""
        self._require_replayable_loader()
        if self._active_epoch is not None:
            raise ValueError("cannot restore an active seeded loader")
        if (
            not isinstance(state, SeededEpochLoaderState)
            or type(state.format_version) is not int
            or state.format_version != 1
        ):
            raise ValueError("incompatible seeded loader cursor format")
        if (
            type(state.epoch) is not int
            or state.epoch < 0
            or type(state.next_batch_index) is not int
            or state.next_batch_index < 0
            or state.next_batch_index > len(self.loader)
        ):
            raise ValueError("incompatible seeded loader batch index or epoch")
        if (
            state.seed != self.seed
            or state.batch_count != len(self.loader)
            or state.batch_size != self.loader.batch_size
            or state.num_workers != self.loader.num_workers
            or state.prefetch_factor != self.loader.prefetch_factor
            or state.drop_last != self.loader.drop_last
            or state.sampler_type
            != f"{type(self.loader.sampler).__module__}.{type(self.loader.sampler).__qualname__}"
        ):
            raise ValueError("incompatible seeded loader configuration")
        if state.next_batch_index == 0:
            if any(
                value is not None
                for value in (
                    state.generator_state,
                    state.epoch_entry_torch_state,
                    state.torch_cpu_random_state,
                    state.python_random_state,
                    state.numpy_random_state,
                )
            ):
                raise ValueError("incompatible seeded loader generator state")
        else:
            for value in (
                state.generator_state,
                state.epoch_entry_torch_state,
                state.torch_cpu_random_state,
            ):
                if (
                    value is None
                    or not self.torch.is_tensor(value)
                    or value.dtype != self.torch.uint8
                ):
                    raise ValueError("incompatible seeded loader generator state")
                try:
                    self.torch.Generator(device="cpu").set_state(value.detach().clone())
                except RuntimeError as exc:
                    raise ValueError("incompatible seeded loader generator state") from exc
            python_state = state.python_random_state
            numpy_state = state.numpy_random_state
            if python_state is None or numpy_state is None:
                raise ValueError("incompatible seeded loader augmentation state")
            try:
                random.Random().setstate(deepcopy(python_state))
                np.random.RandomState().set_state(deepcopy(numpy_state))
            except (TypeError, ValueError) as exc:
                raise ValueError("incompatible seeded loader augmentation state") from exc
        self.epoch = state.epoch
        self._resume_state = replace(
            state,
            generator_state=(
                state.generator_state.detach().clone()
                if state.generator_state is not None
                else None
            ),
            epoch_entry_torch_state=(
                state.epoch_entry_torch_state.detach().clone()
                if state.epoch_entry_torch_state is not None
                else None
            ),
            torch_cpu_random_state=(
                state.torch_cpu_random_state.detach().clone()
                if state.torch_cpu_random_state is not None
                else None
            ),
            python_random_state=deepcopy(state.python_random_state),
            numpy_random_state=deepcopy(state.numpy_random_state),
        )

    def _require_replayable_loader(self) -> None:
        if (
            isinstance(self.loader.dataset, self.torch.utils.data.IterableDataset)
            or self.loader.batch_size is None
            or not isinstance(self.loader.sampler, self.torch.utils.data.RandomSampler)
            or self.loader.sampler.generator is not self.loader.generator
            or self.loader.persistent_workers
            or self.loader.worker_init_fn is not None
            or not getattr(self.loader, "in_order", True)
        ):
            raise ValueError("seeded loader checkpoint requires an ordered, resettable sampler")

    def __iter__(self) -> Any:
        if self._active_epoch is not None:
            raise RuntimeError("seeded loader already has an active iterator")
        epoch = self.epoch
        resume_state = self._resume_state
        replay_pending = resume_state is not None and resume_state.next_batch_index > 0
        replay_started = perf_counter() if replay_pending else 0.0
        previous_random_state = (
            (
                self.torch.get_rng_state().clone(),
                deepcopy(random.getstate()),
                deepcopy(np.random.get_state()),
                self.loader.generator.get_state().clone(),
            )
            if replay_pending
            else None
        )
        self.epoch += 1
        self._active_epoch = epoch
        self._next_batch_index = 0
        self._resume_state = None
        if resume_state is not None and resume_state.next_batch_index:
            self.torch.set_rng_state(resume_state.epoch_entry_torch_state)
        self._epoch_entry_torch_state = self.torch.get_rng_state().clone()
        self.loader.generator.manual_seed(self.seed + epoch)
        # Transforms with zero workers draw from the CPU default generator;
        # workers instead derive seeds from the reset DataLoader generator.
        try:
            with self.torch.random.fork_rng(devices=[]):
                self.torch.random.default_generator.manual_seed(self.seed + 100_000 + epoch)
                for batch_index, batch in enumerate(self.loader, start=1):
                    self._next_batch_index = batch_index
                    if resume_state is not None and batch_index <= resume_state.next_batch_index:
                        # Why this: seeded replay reconstructs worker streams and
                        # zero-worker transforms without persisting prefetched batches.
                        if batch_index == resume_state.next_batch_index and not self.torch.equal(
                            self.loader.generator.get_state(), resume_state.generator_state
                        ):
                            raise ValueError("seeded loader generator state changed during replay")
                        if batch_index == resume_state.next_batch_index:
                            python_state = resume_state.python_random_state
                            numpy_state = resume_state.numpy_random_state
                            assert python_state is not None and numpy_state is not None
                            self.torch.set_rng_state(resume_state.torch_cpu_random_state)
                            random.setstate(deepcopy(python_state))
                            np.random.set_state(deepcopy(numpy_state))
                            self.replay_seconds += perf_counter() - replay_started
                            replay_pending = False
                        continue
                    yield batch
        except Exception:
            if replay_pending and previous_random_state is not None:
                torch_state, python_state, numpy_state, generator_state = previous_random_state
                self.torch.set_rng_state(torch_state)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                self.loader.generator.set_state(generator_state)
                self.epoch = epoch
                self._resume_state = resume_state
            raise
        finally:
            self._active_epoch = None
            self._epoch_entry_torch_state = None
            self._next_batch_index = 0


@dataclass(frozen=True)
class SeededEpochLoaderState:
    """Next logical batch plus sampler state for a seeded image epoch."""

    format_version: int
    seed: int
    epoch: int
    next_batch_index: int
    batch_count: int
    batch_size: int | None
    num_workers: int
    prefetch_factor: int | None
    drop_last: bool
    sampler_type: str
    generator_state: Any | None
    epoch_entry_torch_state: Any | None
    torch_cpu_random_state: Any | None
    python_random_state: Any | None
    numpy_random_state: Any | None
