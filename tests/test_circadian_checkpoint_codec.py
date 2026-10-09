"""Full native bytes and fixed local continuity controls, not recovery admission."""

from dataclasses import replace
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pytest

from src.adapters.circadian_checkpoint_codec import CircadianCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.circadian_predictive_coding import (
    CircadianConfig,
    CircadianPredictiveCodingNetwork,
    ReplayRetentionBudget,
    ReplaySnapshot,
    WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY,
)
from src.core.replay_retention import ReplayRetentionPolicy

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(65536, 8192, 8, 64)
FEATURES = np.array([[0.3, -0.2], [-0.5, 0.4], [0.1, 0.6], [-0.7, -0.1]])
TARGETS = np.array([[1.0], [0.0], [1.0], [0.0]])
WORK = dict(wake_updates=0, replay_updates=0, sleep_calls=0, restores=0, structural=0)


@pytest.fixture(scope="session", autouse=True)
def should_record_fixed_native_fixture_work(tmp_path_factory):
    yield
    assert WORK["wake_updates"] + WORK["replay_updates"] <= 64
    assert WORK["sleep_calls"] <= 16 and WORK["restores"] <= 32 and WORK["structural"] <= 4
    path = tmp_path_factory.mktemp("native-work") / "native-work.json"
    path.write_text(json.dumps(WORK, sort_keys=True), encoding="utf8")


def model(retention="base", wake_only=False):
    config = CircadianConfig(
        sleep_mode="components",
        sleep_enable_split=False,
        sleep_enable_prune=False,
        replay_steps=1,
        replay_memory_size=4,
        prune_decay_steps=3,
    )
    value = CircadianPredictiveCodingNetwork(
        2,
        4,
        seed=23,
        hidden_dims=(3, 4),
        min_hidden_dim=2,
        max_hidden_dim=8,
        circadian_config=config,
    )
    if retention != "base":
        value.configure_replay_retention(
            ReplayRetentionBudget(4, 1024),
            policy=ReplayRetentionPolicy(
                retention, seed=23 if retention == "seeded_reservoir" else None
            ),
        )
    if wake_only:
        value.configure_replay_side_effect_policy(WAKE_ONLY_REPLAY_SIDE_EFFECT_POLICY)
    return value


def encode(state):
    return CircadianCheckpointCodec().encode(state, binding=BINDING, limits=LIMITS)


def decode(raw, binding=BINDING, limits=LIMITS):
    return CircadianCheckpointCodec().decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


def train(value):
    result = value.train_epoch(FEATURES, TARGETS, 0.03, 2, 0.2)
    WORK["wake_updates"] += 1
    return result


@pytest.mark.parametrize("retention", ["base", "content_hash", "recent_fifo", "seeded_reservoir"])
@pytest.mark.parametrize("wake_only", [False, True])
def test_should_round_trip_full_nonzero_native_variant_and_continue_identically(
    retention, wake_only
):
    original = model(retention, wake_only)
    train(original)
    train(original)
    WORK["sleep_calls"] += 1
    before = original._replay_updates
    original.sleep_event()
    WORK["replay_updates"] += original._replay_updates - before
    original._rng.integers(0, 100, size=3, dtype=np.uint32)
    saved = original.snapshot_state()
    raw = encode(saved)
    restored = decode(raw)
    assert encode(restored) == raw
    assert saved.config == restored.config and saved.state.keys() == restored.state.keys()
    assert restored.state["_rng"] is not saved.state["_rng"]
    assert restored.state["_rng"].bit_generator.state == saved.state["_rng"].bit_generator.state
    assert restored.state["_replay_memory"] is not saved.state["_replay_memory"]
    assert restored.state["_replay_memory"].maxlen == saved.state["_replay_memory"].maxlen
    for name, left in saved.state.items():
        right = restored.state[name]
        if type(left) is np.ndarray:
            assert left.dtype == right.dtype and left.shape == right.shape
            assert left.tobytes() == right.tobytes() and not np.shares_memory(left, right)
        elif type(left) is list and left and type(left[0]) is np.ndarray:
            for a, b in zip(left, right):
                assert a.tobytes() == b.tobytes() and not np.shares_memory(a, b)
    for a, b in zip(saved.state["_replay_memory"], restored.state["_replay_memory"]):
        assert a.input_batch.tobytes() == b.input_batch.tobytes()
        assert a.target_batch.tobytes() == b.target_batch.tobytes()
        assert a.priority == b.priority and a.positive_fraction == b.positive_fraction
        assert not np.shares_memory(a.input_batch, b.input_batch)
        assert not np.shares_memory(a.target_batch, b.target_batch)
    candidate = model(retention, wake_only)
    candidate.restore_state(restored)
    WORK["restores"] += 1
    first, second = train(original), train(candidate)
    assert first.energy == second.energy
    assert encode(original.snapshot_state()) == encode(candidate.snapshot_state())
    assert encode(saved) == raw
    restored.state["weight_input_hidden"][:] = 999
    assert encode(saved) == raw


def test_should_preserve_dynamic_lineage_and_pending_prune_state():
    value = model()
    value._split_neurons((0,))
    WORK["structural"] += 1
    value._remove_neurons((0,))
    WORK["structural"] += 1
    value._prune_marked[0] = True
    value._prune_ttl[0] = 2
    value._split_cooldown[1] = 3
    value._prune_cooldown[1] = 2
    saved = value.snapshot_state()
    raw = encode(saved)
    restored = decode(raw)
    assert encode(restored) == raw
    assert restored.state["_next_neuron_id"] == 5
    assert 0 in restored.state["_parent_ids"] and 0 not in restored.state["_neuron_ids"]


@pytest.mark.parametrize(
    "name",
    sorted(model().snapshot_state().state),
)
def test_should_refuse_missing_native_field(name):
    saved = model().snapshot_state()
    del saved.state[name]
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "dtype",
        "shape",
        "nan",
        "negative",
        "alias",
        "lineage",
        "prune",
        "clock",
        "boolcounter",
        "rng",
        "config",
        "optional",
        "metadata",
        "version",
    ],
)
def test_should_refuse_inconsistent_native_schema(fault):
    saved = model().snapshot_state()
    state = saved.state
    if fault == "extra":
        state["future"] = object()
    elif fault == "dtype":
        state["_neuron_ids"] = state["_neuron_ids"].astype(np.int32)
    elif fault == "shape":
        state["bias_output"] = np.zeros((2, 1))
    elif fault == "nan":
        state["_hidden_chemical"][0] = np.nan
    elif fault == "negative":
        state["_traffic_sum"][0] = -1
    elif fault == "alias":
        state["_hidden_chemical_fast"] = state["_hidden_chemical"]
    elif fault == "lineage":
        state["_parent_ids"][0] = 0
    elif fault == "prune":
        state["_prune_marked"][0] = True
    elif fault == "clock":
        state["_epochs_since_sleep"] = 1
    elif fault == "boolcounter":
        state["_traffic_steps"] = True
    elif fault == "rng":
        state["_rng"] = np.random.Generator(np.random.MT19937(23))
    elif fault == "config":
        state["config"] = replace(saved.config, chemical_decay=0.2)
    elif fault == "optional":
        state["_replay_observed_ids"] = set()
    elif fault == "metadata":
        saved = replace(saved, input_dim=3)
    else:
        saved = replace(saved, format_version=3)
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize(
    "fault",
    [
        "version",
        "field",
        "binding",
        "config",
        "shape",
        "dtype",
        "bytes",
        "rng",
        "rng_bool",
        "metadata",
        "deque",
        "counter",
    ],
)
def test_should_refuse_corrupt_bytes_even_with_matching_independent_digest(fault):
    body = json.loads(encode(model().snapshot_state()))
    if fault == "version":
        body["codec_version"] = True
    elif fault == "field":
        body["state"]["extra"] = 0
    elif fault == "binding":
        body["binding"]["source_sha256"] = "c" * 64
    elif fault == "config":
        del body["state"]["config"]["chemical_decay"]
    elif fault == "shape":
        body["state"]["weight_input_hidden"]["shape"] = [999999999, 4]
    elif fault == "dtype":
        body["state"]["_neuron_ids"]["dtype"] = "<i4"
    elif fault == "bytes":
        body["state"]["bias_output"]["data"] = "AA=="
    elif fault == "rng":
        body["state"]["_rng"]["state"]["inc"] = 2
    elif fault == "rng_bool":
        body["state"]["_rng"]["has_uint32"] = True
    elif fault == "metadata":
        body["metadata"]["input_dim"] = True
    elif fault == "deque":
        body["state"]["_replay_memory"]["maxlen"] = 99
    else:
        body["state"]["_traffic_steps"] = -1
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        decode(raw)


def test_should_refuse_incomplete_exposure_and_retention_variants():
    for field in ("_replay_duplicate_ids", "_replay_exposure_updates", "_replay_retention_budget"):
        saved = model("recent_fifo").snapshot_state()
        del saved.state[field]
        with pytest.raises(ValueError):
            encode(saved)
    saved = model("recent_fifo").snapshot_state()
    saved.state["_replay_exposed_ids"].add("a" * 64)
    with pytest.raises(ValueError):
        encode(saved)


def test_should_refuse_resource_exhaustion_before_native_array_allocation(monkeypatch):
    saved = model().snapshot_state()
    raw = encode(saved)
    for limits in (
        replace(LIMITS, max_encoded_bytes=len(raw) - 1),
        replace(LIMITS, max_array_bytes=1),
        replace(LIMITS, max_layers=1),
        replace(LIMITS, max_dimension=3),
    ):
        with pytest.raises(ValueError):
            CircadianCheckpointCodec().encode(saved, binding=BINDING, limits=limits)
        monkeypatch.setattr(
            np, "frombuffer", lambda *a, **kw: pytest.fail("array allocation before preflight")
        )
        with pytest.raises(ValueError):
            decode(raw, limits=limits)


@pytest.mark.parametrize("raw", [b"{}", b"\xff", b"{", b'{"a":1,"a":2}', b"[" * 1500])
def test_should_refuse_malformed_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_should_refuse_noncanonical_or_wrong_content_or_exact_input_types():
    raw = encode(model().snapshot_state())
    codec = CircadianCheckpointCodec()
    with pytest.raises(ValueError):
        decode(raw + b" ")
    bad: tuple[Any, ...] = (bytearray(raw), None, raw + b" ")
    for value in bad:
        with pytest.raises(ValueError):
            codec.decode(
                value, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
            )
    with pytest.raises(ValueError):
        decode(raw, binding=replace(BINDING, policy_sha256="c" * 64))


@pytest.mark.parametrize("record", ["snapshot", "config", "budget", "policy", "replay"])
def test_should_refuse_undeclared_nested_native_field_without_omission(record):
    saved = model("recent_fifo").snapshot_state()
    row = ReplaySnapshot(FEATURES[:1].copy(), TARGETS[:1].copy(), 0.2, 1.0)
    saved.state["_replay_memory"].append(row)
    value = {
        "snapshot": saved,
        "config": saved.config,
        "budget": saved.state["_replay_retention_budget"],
        "policy": saved.state["_replay_retention_policy"],
        "replay": row,
    }[record]
    object.__setattr__(value, "future_native_field", object())
    with pytest.raises(ValueError):
        encode(saved)


def test_should_refuse_unsupported_config_value_without_executing_object_copy():
    class Foreign:
        def __deepcopy__(self, memo):
            pytest.fail("unsupported value copied before scalar schema validation")

    saved = model().snapshot_state()
    object.__setattr__(saved.config, "chemical_decay", Foreign())
    with pytest.raises(ValueError):
        encode(saved)


def test_should_leave_original_snapshot_unchanged_after_partial_encoding_failure(monkeypatch):
    import src.adapters.circadian_checkpoint_codec as module

    saved = model().snapshot_state()
    original = encode(saved)
    with monkeypatch.context() as patch:

        def fail(*args):
            raise ValueError("injected encoding failure")

        patch.setattr(module, "encode_frame", fail)
        with pytest.raises(ValueError):
            encode(saved)
    assert encode(saved) == original


@pytest.mark.parametrize("fault", ["nan", "bool", "targets", "fraction", "priority"])
def test_should_refuse_invalid_native_replay_or_boolean_bytes(fault):
    saved = model().snapshot_state()
    saved.state["_replay_memory"].append(ReplaySnapshot(FEATURES.copy(), TARGETS.copy(), 0.2, 0.5))
    body = json.loads(encode(saved))
    from base64 import b64encode

    if fault in ("fraction", "priority"):
        name = "positive_fraction" if fault == "fraction" else "priority"
        body["state"]["_replay_memory"]["items"][0][name] = -1.0
    elif fault == "bool":
        frame = body["state"]["_prune_marked"]
        frame["data"] = b64encode(bytes([2, 0, 0, 0])).decode()
    else:
        frame = body["state"]["_replay_memory"]["items"][0]["target_batch"]
        values = np.full((4, 1), np.nan if fault == "nan" else 2.0, dtype="<f8")
        frame["data"] = b64encode(values.tobytes()).decode()
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        decode(raw)
