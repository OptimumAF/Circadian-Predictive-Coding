"""Fixed tiny native state copies; no checkpoint/handoff or scientific work."""

from contextlib import ExitStack
from collections import deque
from dataclasses import replace
import json
import pickle
import sys
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.native_model_copy import ModelCopyLimits
from src.core.native_state_copy import observe_state_copies

LIMITS = ModelCopyLimits(1, 2, 16)


@pytest.fixture(scope="module", autouse=True)
def native_work(tmp_path_factory):
    from src.core import native_state_copy

    counts = dict(
        models=0,
        wakes=0,
        stores=0,
        predictions=0,
        snapshots=0,
        restore_attempts=0,
        state_copies=0,
        source_array_bytes=0,
        inference_steps_requested=0,
        array_copies=0,
    )
    previous = sys.getprofile()

    def array_bytes(source):
        seen = set()

        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            if isinstance(value, np.ndarray):
                return value.nbytes
            if isinstance(value, dict):
                return sum(size(v) for v in value.values())
            if isinstance(value, (list, tuple, deque)):
                return sum(size(v) for v in value)
            if hasattr(value, "__dict__"):
                return size(vars(value))
            return 0

        return size(source)

    def profile(frame, event, function):
        if (
            event == "c_call"
            and frame.f_code.co_filename.endswith("circadian_predictive_coding.py")
            and getattr(function, "__name__", None) == "copy"
            and isinstance(getattr(function, "__self__", None), np.ndarray)
        ):
            counts["array_copies"] += 1
            assert counts["array_copies"] <= 10
        if previous is not None:
            previous(frame, event, function)

    with ExitStack() as stack:
        copier = native_state_copy.deepcopy

        def copy_state(source, *args):
            counts["state_copies"] += 1
            counts["source_array_bytes"] += array_bytes(source)
            assert counts["state_copies"] <= 12 and counts["source_array_bytes"] <= 1048576
            return copier(source, *args)

        stack.enter_context(patch.object(native_state_copy, "deepcopy", copy_state))
        for method, key in [
            ("__init__", "models"),
            ("train_epoch", "wakes"),
            ("_store_replay_snapshot", "stores"),
            ("predict_proba", "predictions"),
            ("snapshot_state", "snapshots"),
            ("restore_state", "restore_attempts"),
        ]:
            original = getattr(CircadianPredictiveCodingNetwork, method)

            def call(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                if _key == "wakes":
                    counts["inference_steps_requested"] += args[4]
                    assert counts["inference_steps_requested"] <= 4
                assert (
                    counts[_key]
                    <= dict(
                        models=2, wakes=2, stores=2, predictions=2, snapshots=4, restore_attempts=6
                    )[_key]
                )
                return _original(*args, **kwargs)

            stack.enter_context(patch.object(CircadianPredictiveCodingNetwork, method, call))
        try:
            sys.setprofile(profile)
            yield counts
        finally:
            sys.setprofile(previous)
            (tmp_path_factory.mktemp("native-state-work") / "work.json").write_text(
                json.dumps(counts), encoding="utf8"
            )


def model_with_replay():
    model = CircadianPredictiveCodingNetwork(
        3,
        2,
        min_hidden_dim=2,
        max_hidden_dim=2,
        seed=23,
        circadian_config=CircadianConfig(sleep_mode="disabled", replay_memory_size=2),
    )
    model.train_epoch(np.array([[1.0, 0.0, 1.0]]), np.array([[1.0]]), 0.01, 2, 0.05)
    return model


def test_should_bind_native_snapshot_and_published_restore_rows_through_actual_memos(native_work):
    model = model_with_replay()
    source = model.__dict__
    fields = tuple(source)
    row = model._replay_memory[0]
    row_fields = tuple(vars(row))
    expected = pickle.dumps(source)
    events: list[Any] = []
    handles: list[Any] = []

    def snapshot_observer(stage, read, lookup):
        assert read().source is source
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            assert lookup(source) is target
            copied_row = target["_replay_memory"][0]
            assert lookup(row) is copied_row and copied_row is not row
            assert lookup(row.input_batch) is copied_row.input_batch
            assert lookup(row.target_batch) is copied_row.target_batch
            assert copied_row.input_batch is not row.input_batch
            assert tuple(vars(copied_row)) == row_fields
        events.append(stage)
        handles.append((read, lookup))

    with observe_state_copies(source, snapshot_observer, LIMITS):
        snapshot = model.snapshot_state()
    assert pickle.dumps(snapshot.state) == expected
    saved_row = snapshot.state["_replay_memory"][0]
    restored_rows = []

    def restore_observer(stage, read, lookup):
        assert read().source is snapshot.state
        # Copy completion precedes native validation and final publication.
        assert model._replay_memory[0] is row
        if stage == "copied":
            target = read().target
            assert isinstance(target, dict)
            copied_row = target["_replay_memory"][0]
            assert lookup(snapshot.state) is target
            assert lookup(saved_row) is copied_row
            assert lookup(saved_row.input_batch) is copied_row.input_batch
            assert lookup(saved_row.target_batch) is copied_row.target_batch
            restored_rows.append(copied_row)
        events.append(stage)
        handles.append((read, lookup))

    with observe_state_copies(snapshot.state, restore_observer, LIMITS):
        model.restore_state(snapshot)
    assert model.__dict__ is source and tuple(source) == fields
    assert model._replay_memory[0] is restored_rows[0]
    assert model._replay_memory[0] is not row and model._replay_memory[0] is not saved_row
    assert pickle.dumps(source) == expected
    assert events == ["before_copy", "copied"] * 2
    for read, lookup in handles:
        with pytest.raises(ValueError, match="synchronous"):
            read()
        with pytest.raises(ValueError, match="synchronous"):
            lookup(row)
    assert native_work["restore_attempts"] == 1


def test_should_keep_native_state_unpublished_after_admission_observer_or_validation_fault(
    native_work,
):
    model = model_with_replay()
    snapshot = model.snapshot_state()
    original_row = model._replay_memory[0]
    before = pickle.dumps(model.__dict__)

    for fault_stage in ["before_copy", "copied"]:

        def refuse(stage, read, lookup):
            assert read().source is snapshot.state
            if stage == fault_stage:
                raise RuntimeError("state observation refused")

        with observe_state_copies(snapshot.state, refuse, LIMITS) as window:
            with pytest.raises(RuntimeError, match="observation refused"):
                model.restore_state(snapshot)
            assert window._copies == 1 and window._failed
        assert model._replay_memory[0] is original_row
        assert pickle.dumps(model.__dict__) == before

    # Version validation still precedes copying, even with an active observer.
    with observe_state_copies(snapshot.state, lambda *_: pytest.fail("copy entered"), LIMITS):
        with pytest.raises(ValueError, match="version"):
            model.restore_state(replace(snapshot, format_version=99))

    # An actual copied graph can fail topology validation; it is not authority.
    original_weights = snapshot.state["weight_hidden_output"]
    snapshot.state["weight_hidden_output"] = original_weights[:1]
    stages = []
    with observe_state_copies(snapshot.state, lambda stage, *_: stages.append(stage), LIMITS):
        with pytest.raises(ValueError, match="topology"):
            model.restore_state(snapshot)
    assert stages == ["before_copy", "copied"]
    assert model._replay_memory[0] is original_row
    assert pickle.dumps(model.__dict__) == before
    assert native_work["restore_attempts"] == 5
