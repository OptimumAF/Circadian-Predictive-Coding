"""Independent complete learner wire and fake-factory restoration controls."""

from base64 import b64encode
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_learner_checkpoint as checkpoint_module
import src.adapters.coupled_checkpoint_codec as native_module
from src.adapters.coupled_learner_checkpoint import (
    CoupledLearnerCheckpointCodec,
    capture_coupled_learner,
    restore_coupled_learner,
)
from src.adapters.coupled_checkpoint_codec import CoupledCheckpointCodec
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.coupled_inference import CoupledInferencePolicy
from src.core.coupled_learner_state import CoupledLearnerCheckpoint, CoupledUpdatePolicy
from src.infra.checkpoint_files import BoundedCheckpointFileStore
from test_coupled_checkpoint_codec import arrays, assert_same, snapshot

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(32768, 8192, 8, 64)


def policy(guarded=True):
    return CoupledUpdatePolicy(
        0.025, 3, 0.07, CoupledInferencePolicy(1, 0.1, None) if guarded else None
    )


def state(kind="active", order="C", guarded=True):
    return CoupledLearnerCheckpoint(1, policy(guarded), snapshot(kind, order))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def wire(value):
    native = CoupledCheckpointCodec().encode(value.native_state, binding=BINDING, limits=LIMITS)
    return dict(
        codec_version=1,
        kind="coupled_learner",
        binding=asdict(BINDING),
        policy=asdict(value.policy),
        native=json.loads(native),
    )


def encode(value=None, *, limits=LIMITS, binding=BINDING):
    return CoupledLearnerCheckpointCodec().encode(
        state() if value is None else value, binding=binding, limits=limits
    )


def decode(raw, *, limits=LIMITS, binding=BINDING, digest=None):
    return CoupledLearnerCheckpointCodec().decode(
        raw,
        binding=binding,
        expected_sha256=sha256(raw).hexdigest() if digest is None else digest,
        limits=limits,
    )


@pytest.mark.parametrize("kind", ["pc", "neutral", "active", "saturated"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("guarded", [False, True])
def test_should_match_independent_nested_wire_and_detach_complete_native_and_policy(
    kind, order, guarded
):
    original = state(kind, order, guarded)
    raw = encode(original)
    assert raw == canonical(wire(original))
    restored = decode(raw)
    assert type(restored) is CoupledLearnerCheckpoint and restored.format_version == 1
    assert restored.policy == original.policy and restored.policy is not original.policy
    assert_same(restored.native_state, original.native_state)
    if guarded:
        assert restored.policy.inference_policy is not original.policy.inference_policy
    for actual, expected in zip(
        arrays(restored.native_state), arrays(original.native_state), strict=True
    ):
        assert not np.shares_memory(actual, expected)
        assert np.array_equal(np.signbit(actual), np.signbit(expected))
    arrays(restored.native_state)[0].fill(99)
    assert encode(original) == raw


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", True),
        ("learning_rate", 0),
        ("learning_rate", -1),
        ("learning_rate", float("inf")),
        ("learning_rate", np.float64(0.1)),
        ("inference_learning_rate", False),
        ("inference_learning_rate", float("nan")),
        ("inference_steps", True),
        ("inference_steps", 0),
        ("inference_steps", 10001),
        ("inference_steps", np.int64(3)),
        ("inference_policy", {}),
        ("inference_policy", CoupledInferencePolicy(4)),
        (
            "inference_policy",
            SimpleNamespace(min_steps=1, gradient_tolerance=0.1, relative_energy_tolerance=None),
        ),
    ],
)
def test_should_refuse_invalid_native_policy_metadata(field, value):
    kwargs: dict[str, Any] = dict(
        learning_rate=0.025, inference_steps=3, inference_learning_rate=0.07, inference_policy=None
    )
    kwargs[field] = value
    with pytest.raises(ValueError):
        CoupledUpdatePolicy(**kwargs)


@pytest.mark.parametrize(
    "fault",
    [
        "format_bool",
        "format_version",
        "policy_mapping",
        "policy_extra",
        "policy_missing",
        "rate_tampered",
        "guard_extra",
        "guard_missing",
        "guard_bool",
        "guard_minimum",
        "guard_tolerance",
        "native_mapping",
    ],
)
def test_should_revalidate_all_metadata_before_any_native_array_encoding(monkeypatch, fault):
    value = state()
    if fault == "format_bool":
        object.__setattr__(value, "format_version", True)
    elif fault == "format_version":
        object.__setattr__(value, "format_version", 2)
    elif fault == "policy_mapping":
        object.__setattr__(value, "policy", asdict(value.policy))
    elif fault == "policy_extra":
        object.__setattr__(value.policy, "extra", 1)
    elif fault == "policy_missing":
        object.__delattr__(value.policy, "learning_rate")
    elif fault == "rate_tampered":
        object.__setattr__(value.policy, "learning_rate", False)
    elif fault == "guard_extra":
        object.__setattr__(value.policy.inference_policy, "extra", 1)
    elif fault == "guard_missing":
        object.__delattr__(value.policy.inference_policy, "min_steps")
    elif fault == "guard_bool":
        object.__setattr__(value.policy.inference_policy, "min_steps", True)
    elif fault == "guard_minimum":
        object.__setattr__(value.policy.inference_policy, "min_steps", 4)
    elif fault == "guard_tolerance":
        object.__setattr__(value.policy.inference_policy, "gradient_tolerance", -1)
    else:
        object.__setattr__(value, "native_state", asdict(value.native_state))
    touched: list[bool] = []

    def forbidden(*args, **kwargs):
        touched.append(True)
        raise AssertionError("invalid metadata reached native encoding")

    monkeypatch.setattr(CoupledCheckpointCodec, "encode", forbidden)
    with pytest.raises(ValueError):
        encode(value)
    assert touched == []


@pytest.mark.parametrize("guarded", [False, True])
def test_should_share_one_exact_encoded_allowance_with_native_and_refuse_before_array_work(
    monkeypatch, guarded
):
    value = state(guarded=guarded)
    raw = canonical(wire(value))
    exact = replace(LIMITS, max_encoded_bytes=len(raw))
    assert encode(value, limits=exact) == raw
    assert_same(decode(raw, limits=exact).native_state, value.native_state)
    overhead = len(canonical(dict(wire(value), native={}))) - 2
    captured: list[CodecLimits] = []
    native_encode, native_decode = CoupledCheckpointCodec.encode, CoupledCheckpointCodec.decode

    def encoding(codec, native, **kwargs):
        captured.append(kwargs["limits"])
        return native_encode(codec, native, **kwargs)

    def decoding(codec, native, **kwargs):
        captured.append(kwargs["limits"])
        return native_decode(codec, native, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(CoupledCheckpointCodec, "encode", encoding)
        patch.setattr(CoupledCheckpointCodec, "decode", decoding)
        assert encode(value, limits=exact) == raw
        decode(raw, limits=exact)
    assert captured == [replace(exact, max_encoded_bytes=len(raw) - overhead)] * 2

    def forbidden(*args, **kwargs):
        raise AssertionError("byte excess reached array work")

    monkeypatch.setattr(native_module, "encode_frame", forbidden)
    monkeypatch.setattr(native_module, "decode_frame", forbidden)
    with pytest.raises(ValueError):
        encode(value, limits=replace(exact, max_encoded_bytes=len(raw) - 1))
    with pytest.raises(ValueError):
        decode(raw, limits=replace(exact, max_encoded_bytes=len(raw) - 1))
    monkeypatch.setattr(CoupledCheckpointCodec, "encode", forbidden)
    with pytest.raises(ValueError):
        encode(value, limits=replace(exact, max_encoded_bytes=overhead))


@pytest.mark.parametrize(
    "fault",
    [
        "kind",
        "version_bool",
        "version",
        "extra",
        "missing",
        "policy_extra",
        "policy_missing",
        "policy_bool",
        "guard_missing",
        "guard_extra",
        "guard_bool",
        "guard_minimum",
        "native_kind",
        "native_missing",
        "binding",
    ],
)
def test_should_refuse_malformed_envelope_before_native_array_decode_or_model_construction(
    monkeypatch, fault
):
    body = wire(state())
    if fault == "kind":
        body["kind"] = "coupled_pc"
    elif fault == "version_bool":
        body["codec_version"] = True
    elif fault == "version":
        body["codec_version"] = 2
    elif fault == "extra":
        body["extra"] = None
    elif fault == "missing":
        body.pop("native")
    elif fault == "policy_extra":
        body["policy"]["extra"] = 1
    elif fault == "policy_missing":
        body["policy"].pop("learning_rate")
    elif fault == "policy_bool":
        body["policy"]["inference_steps"] = True
    elif fault == "guard_missing":
        body["policy"]["inference_policy"].pop("min_steps")
    elif fault == "guard_extra":
        body["policy"]["inference_policy"]["extra"] = None
    elif fault == "guard_bool":
        body["policy"]["inference_policy"]["min_steps"] = True
    elif fault == "guard_minimum":
        body["policy"]["inference_policy"]["min_steps"] = 4
    elif fault == "native_kind":
        body["native"]["kind"] = "untrusted"
    elif fault == "native_missing":
        body["native"].pop("arrays")
    else:
        body["binding"]["source_sha256"] = "f" * 64
    raw = canonical(body)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid envelope reached native array or model")

    monkeypatch.setattr(native_module, "decode_frame", forbidden)
    monkeypatch.setattr(checkpoint_module, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(checkpoint_module, "CoupledCircadianPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError):
        decode(raw)
    with pytest.raises(ValueError):
        restore_coupled_learner(
            raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize("representation", ["space", "duplicate", "utf8", "hash", "binding"])
def test_should_refuse_corruption_and_noncanonical_outer_bytes_before_native_decode(
    monkeypatch, representation
):
    raw = encode()
    expected = sha256(raw).hexdigest()
    binding = BINDING
    if representation == "space":
        raw += b" "
        expected = sha256(raw).hexdigest()
    elif representation == "duplicate":
        raw = b'{"kind":"coupled_learner",' + raw[1:]
        expected = sha256(raw).hexdigest()
    elif representation == "utf8":
        raw = b"\xff"
        expected = sha256(raw).hexdigest()
    elif representation == "hash":
        raw = raw[:-1] + b"!"
    else:
        binding = CodecBinding("f" * 64, BINDING.policy_sha256)

    def forbidden(*args, **kwargs):
        raise AssertionError("outer refusal reached native decoder")

    monkeypatch.setattr(CoupledCheckpointCodec, "decode", forbidden)
    with pytest.raises(ValueError):
        decode(raw, binding=binding, digest=expected)


class SnapshotModel:
    def __init__(self, value):
        self.value = deepcopy(value)
        self.snapshots = 0

    def snapshot_state(self):
        self.snapshots += 1
        return deepcopy(self.value)


@pytest.mark.parametrize("guarded", [False, True])
def test_should_capture_exact_trusted_learner_with_detached_fixed_policy_and_no_native_work(
    guarded,
):
    from src.adapters.coupled_learners import CoupledLearner

    expected = state(guarded=guarded)
    model = SnapshotModel(expected.native_state)
    learner: Any = object.__new__(CoupledLearner)
    learner._learning_rate = expected.policy.learning_rate
    learner._inference_steps = expected.policy.inference_steps
    learner._inference_learning_rate = expected.policy.inference_learning_rate
    learner._inference_policy = deepcopy(expected.policy.inference_policy)
    learner._model = model
    got = learner.checkpoint_policy()
    assert got == expected.policy and model.snapshots == 0
    assert got.inference_policy is not learner._inference_policy or got.inference_policy is None
    captured = capture_coupled_learner(learner)
    assert captured.policy == expected.policy and model.snapshots == 1
    assert_same(captured.native_state, expected.native_state)
    original = encode(captured)
    captured.native_state.parameters.hidden_weights[0].fill(99)
    object.__setattr__(captured.policy, "learning_rate", 99.0)
    assert encode(capture_coupled_learner(learner)) == original
    learner._learning_rate = True
    with pytest.raises(ValueError):
        learner.checkpoint_policy()


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
@pytest.mark.parametrize("guarded", [False, True])
def test_should_restore_only_after_full_decode_with_exact_native_constructor_and_policy_arguments(
    monkeypatch, kind, guarded
):
    expected = state(kind, "F", guarded)
    raw = encode(expected)
    events: list[Any] = []
    created: list[Any] = []

    class PC:
        def __init__(self, *args, **kwargs):
            events.append(("pc", args, kwargs))
            created.append(self)

        def restore_state(self, native):
            events.append(("restore",))
            assert_same(native, expected.native_state)
            self.native = deepcopy(native)

    class CPC(PC):
        def __init__(self, *args, **kwargs):
            events.append(("cpc", args, kwargs))
            created.append(self)

    class Learner:
        def __init__(self, model, **kwargs):
            assert model is created[0]
            assert_same(model.native, expected.native_state)
            assert kwargs == dict(
                learning_rate=0.025,
                inference_steps=3,
                inference_learning_rate=0.07,
                inference_policy=expected.policy.inference_policy,
            )
            self.value = deepcopy(model.native)
            self.settings = deepcopy(kwargs)
            events.append(("learner",))

    monkeypatch.setattr(checkpoint_module, "CoupledPredictiveCodingNetwork", PC)
    monkeypatch.setattr(checkpoint_module, "CoupledCircadianPredictiveCodingNetwork", CPC)
    monkeypatch.setattr(checkpoint_module, "CoupledLearner", Learner)
    restored: Any = restore_coupled_learner(
        raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    assert [event[0] for event in events] == ["pc" if kind == "pc" else "cpc", "restore", "learner"]
    args, kwargs = events[0][1:]
    assert args == (3, (4, 2))
    assert kwargs["seed"] == 0 and kwargs["precisions"] == (1e-6, 1e6)
    if kind == "pc":
        assert set(kwargs) == {"seed", "precisions"}
    else:
        assert set(kwargs) == {"seed", "precisions", "config"}
        wake = expected.native_state.wake_state
        assert asdict(kwargs["config"]) == {
            name: getattr(wake, name)
            for name in (
                "chemical_decay",
                "chemical_buildup_rate",
                "plasticity_sensitivity",
                "min_plasticity",
            )
        }
    assert_same(restored.value, expected.native_state)
    created[0].native.parameters.output_bias.fill(99)
    assert encode(CoupledLearnerCheckpoint(1, expected.policy, restored.value)) == raw


@pytest.mark.parametrize("kind", ["pc", "neutral", "active"])
def test_should_use_existing_bounded_file_store_with_complete_learner_state(tmp_path, kind):
    expected = state(kind)
    path = tmp_path / "learner.bin"
    store = BoundedCheckpointFileStore(path, CoupledLearnerCheckpointCodec())
    digest = store.save(expected, binding=BINDING, limits=LIMITS)
    raw = path.read_bytes()
    assert digest == sha256(raw).hexdigest() and len(raw) <= LIMITS.max_encoded_bytes
    restored = store.load(binding=BINDING, expected_sha256=digest, limits=LIMITS)
    assert restored.policy == expected.policy
    assert_same(restored.native_state, expected.native_state)
    arrays(restored.native_state)[0].fill(99)
    assert path.read_bytes() == raw and encode(expected) == raw


@pytest.mark.parametrize("family", ["mapping", "subclass"])
def test_should_refuse_untrusted_capture_family_before_invoking_its_methods(family):
    from src.adapters.coupled_learners import CoupledLearner

    def forbidden(*args, **kwargs):
        raise AssertionError("untrusted capture reached a callback")

    class Other(CoupledLearner):
        checkpoint_policy = forbidden
        snapshot_state = forbidden

    value: Any = (
        SimpleNamespace(checkpoint_policy=forbidden, snapshot_state=forbidden)
        if family == "mapping"
        else object.__new__(Other)
    )
    with pytest.raises(ValueError):
        capture_coupled_learner(value)


def test_should_validate_all_native_finite_arrays_before_constructing_restore_models(monkeypatch):
    body = wire(state())
    body["native"]["arrays"]["output_bias"]["data"] = b64encode(
        np.array([[np.nan]], dtype="<f8").tobytes()
    ).decode()
    raw = canonical(body)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid native array reached model construction")

    monkeypatch.setattr(checkpoint_module, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(checkpoint_module, "CoupledCircadianPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError):
        restore_coupled_learner(
            raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )


@pytest.mark.parametrize("field", ["learning_rate", "inference_learning_rate"])
def test_should_preserve_integer_policy_bytes_but_refuse_lossy_native_factory_normalization(
    monkeypatch, field
):
    value = state()
    value = replace(value, policy=replace(value.policy, **{field: 2**53 + 1}))
    raw = encode(value)
    decoded = decode(raw)
    assert type(getattr(decoded.policy, field)) is int
    assert getattr(decoded.policy, field) == 2**53 + 1
    assert encode(decoded) == raw

    def forbidden(*args, **kwargs):
        raise AssertionError("lossy policy reached native construction")

    monkeypatch.setattr(checkpoint_module, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(checkpoint_module, "CoupledCircadianPredictiveCodingNetwork", forbidden)
    with pytest.raises(ValueError, match=f"restoring {field} requires a lossless conversion"):
        restore_coupled_learner(
            raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )
