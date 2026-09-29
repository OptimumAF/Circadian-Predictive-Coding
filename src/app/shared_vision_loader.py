"""Replay a shared, unseeded image DataLoader from a logical batch cursor.

The v1/v2 unmatched protocols share one mutable sampler across models. This
wrapper records its epoch-entry state and replays consumed batches without
resetting the sampler or enclosing process RNG during ordinary training.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import random
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SharedEpochLoaderState:
    """Mutable shared sampler and logical next-batch position."""

    format_version: int
    epoch: int
    next_batch_index: int
    batch_count: int
    batch_size: int | None
    num_workers: int
    prefetch_factor: int | None
    drop_last: bool
    sampler_type: str
    epoch_entry_generator_state: Any | None
    generator_state: Any
    epoch_entry_torch_state: Any | None
    torch_cpu_random_state: Any | None
    python_random_state: Any | None
    numpy_random_state: Any | None


class SharedEpochTrainLoader:
    """Keep the legacy shared stream while enabling fresh-loader replay."""

    def __init__(self, torch: Any, loader: Any) -> None:
        self.torch = torch
        self.loader = loader
        self.epoch = 0
        self._active_epoch: int | None = None
        self._next_batch_index = 0
        self._epoch_entry_generator_state: Any | None = None
        self._epoch_entry_torch_state: Any | None = None
        self._resume_state: SharedEpochLoaderState | None = None
        self.replay_seconds = 0.0

    def snapshot_state(self) -> SharedEpochLoaderState:
        """Capture the shared sampler without advancing it."""
        self._require_replayable_loader()
        active = self._active_epoch is not None
        entry_generator = self._epoch_entry_generator_state
        entry_torch = self._epoch_entry_torch_state
        assert not active or (entry_generator is not None and entry_torch is not None)
        return SharedEpochLoaderState(
            format_version=1,
            epoch=self._active_epoch if self._active_epoch is not None else self.epoch,
            next_batch_index=self._next_batch_index if active else 0,
            batch_count=len(self.loader),
            batch_size=self.loader.batch_size,
            num_workers=self.loader.num_workers,
            prefetch_factor=self.loader.prefetch_factor,
            drop_last=self.loader.drop_last,
            sampler_type=self._sampler_type(),
            epoch_entry_generator_state=(
                entry_generator.clone() if entry_generator is not None and active else None
            ),
            generator_state=self.loader.generator.get_state().clone(),
            epoch_entry_torch_state=(
                entry_torch.clone() if entry_torch is not None and active else None
            ),
            torch_cpu_random_state=self.torch.get_rng_state().clone() if active else None,
            python_random_state=deepcopy(random.getstate()) if active else None,
            numpy_random_state=deepcopy(np.random.get_state()) if active else None,
        )

    def restore_state(self, state: SharedEpochLoaderState) -> None:
        """Validate the cursor before its first replaying iteration."""
        self._require_replayable_loader()
        if self._active_epoch is not None:
            raise ValueError("cannot restore an active shared vision loader")
        if (
            not isinstance(state, SharedEpochLoaderState)
            or type(state.format_version) is not int
            or state.format_version != 1
            or type(state.epoch) is not int
            or state.epoch < 0
            or type(state.next_batch_index) is not int
            or not 0 <= state.next_batch_index <= len(self.loader)
        ):
            raise ValueError("incompatible shared vision loader cursor")
        if (
            state.batch_count != len(self.loader)
            or state.batch_size != self.loader.batch_size
            or state.num_workers != self.loader.num_workers
            or state.prefetch_factor != self.loader.prefetch_factor
            or state.drop_last != self.loader.drop_last
            or state.sampler_type != self._sampler_type()
        ):
            raise ValueError("incompatible shared vision loader configuration")
        self._validate_generator_state(state.generator_state)
        if state.next_batch_index == 0:
            if any(
                value is not None
                for value in (
                    state.epoch_entry_generator_state,
                    state.epoch_entry_torch_state,
                    state.torch_cpu_random_state,
                    state.python_random_state,
                    state.numpy_random_state,
                )
            ):
                raise ValueError("incompatible idle shared vision loader state")
        else:
            self._validate_generator_state(state.epoch_entry_generator_state)
            for value in (state.epoch_entry_torch_state, state.torch_cpu_random_state):
                self._validate_generator_state(value)
            python_state = state.python_random_state
            numpy_state = state.numpy_random_state
            if python_state is None or numpy_state is None:
                raise ValueError("incompatible shared vision augmentation state")
            try:
                random.Random().setstate(deepcopy(python_state))
                np.random.RandomState().set_state(deepcopy(numpy_state))
            except (TypeError, ValueError) as exc:
                raise ValueError("incompatible shared vision augmentation state") from exc
        self.epoch = state.epoch
        self._resume_state = replace(
            state,
            epoch_entry_generator_state=(
                state.epoch_entry_generator_state.detach().clone()
                if state.epoch_entry_generator_state is not None
                else None
            ),
            generator_state=state.generator_state.detach().clone(),
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

    def __iter__(self) -> Any:
        if self._active_epoch is not None:
            raise RuntimeError("shared vision loader already has an active iterator")
        epoch = self.epoch
        resume_state = self._resume_state
        replay_pending = resume_state is not None and resume_state.next_batch_index > 0
        replay_started = perf_counter() if replay_pending else 0.0
        previous = (
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
        if resume_state is not None:
            self.loader.generator.set_state(
                resume_state.epoch_entry_generator_state
                if replay_pending
                else resume_state.generator_state
            )
            if replay_pending:
                self.torch.set_rng_state(resume_state.epoch_entry_torch_state)
        self._epoch_entry_generator_state = self.loader.generator.get_state().clone()
        self._epoch_entry_torch_state = self.torch.get_rng_state().clone()
        try:
            for batch_index, batch in enumerate(self.loader, start=1):
                self._next_batch_index = batch_index
                if resume_state is not None and batch_index <= resume_state.next_batch_index:
                    if batch_index == resume_state.next_batch_index:
                        if not self.torch.equal(
                            self.loader.generator.get_state(), resume_state.generator_state
                        ):
                            raise ValueError(
                                "shared vision checkpoint sampler changed during replay"
                            )
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
            if replay_pending and previous is not None:
                torch_state, python_state, numpy_state, generator_state = previous
                self.torch.set_rng_state(torch_state)
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                self.loader.generator.set_state(generator_state)
                self.epoch = epoch
                self._resume_state = resume_state
            raise
        finally:
            self._active_epoch = None
            self._next_batch_index = 0
            self._epoch_entry_generator_state = None
            self._epoch_entry_torch_state = None

    def _sampler_type(self) -> str:
        sampler = type(self.loader.sampler)
        return f"{sampler.__module__}.{sampler.__qualname__}"

    def _require_replayable_loader(self) -> None:
        if (
            getattr(self.loader, "generator", None) is None
            or isinstance(self.loader.dataset, self.torch.utils.data.IterableDataset)
            or self.loader.batch_size is None
            or not isinstance(self.loader.sampler, self.torch.utils.data.RandomSampler)
            or self.loader.sampler.generator is not self.loader.generator
            or self.loader.persistent_workers
            or self.loader.worker_init_fn is not None
            or not getattr(self.loader, "in_order", True)
        ):
            raise ValueError("shared vision checkpoint requires an ordered, resettable sampler")

    def _validate_generator_state(self, value: Any) -> None:
        if not self.torch.is_tensor(value) or value.dtype != self.torch.uint8:
            raise ValueError("incompatible shared vision generator state")
        try:
            self.torch.Generator(device="cpu").set_state(value.detach().clone())
        except RuntimeError as exc:
            raise ValueError("incompatible shared vision generator state") from exc
