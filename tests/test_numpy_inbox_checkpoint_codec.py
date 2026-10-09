"""Complete supported inbox bytes; all fixtures perform zero native work."""

from base64 import b64encode
from dataclasses import fields, replace
from hashlib import sha256
import json
from typing import Any

import numpy as np
import pytest

from src.adapters.numpy_inbox_checkpoint_codec import NumpyInboxCheckpointCodec
from src.core.inbox_codec_policy import InboxCodecPolicy, NUMERIC_DTYPES
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.inbox_cursor import InboxCursor
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, AppliedExperience
from src.core.data_erasure import ErasedExperience
from src.core.learner_ports import TrainingDiagnostic

BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(65536, 8192, 8, 64)
POLICY = InboxCodecPolicy(3, 4, 8, 128, 4)


@pytest.fixture(scope="session", autouse=True)
def should_record_zero_native_work(tmp_path_factory):
    path = tmp_path_factory.mktemp("native-work") / "native-work.json"
    path.write_text(
        json.dumps(dict(updates=0, restores=0, sleeps=0, structural=0, predictions=0, workers=0)),
        encoding="utf8",
    )


def source(sample, order="F", replay=True, evaluation=False):
    array = np.array([[-0.0, 0.3, -0.2], [0.6, -0.5, 0.1]], order=order)
    return Experience(
        sample,
        "episode",
        1,
        "actor",
        array,
        "train",
        ExperiencePermissions(True, replay, evaluation),
        ("left", "right"),
        "left",
        -0.0,
    )


def label(sample):
    return LabelArrival("event-" + sample, sample, "episode", 2, "actor", np.array([[1.0], [0.5]]))


def cursor(version=2, stopped=False, order="F", replay=True, evaluation=False):
    sources = tuple(source(name, order, replay, evaluation) for name in ("paired", "source-only"))
    labels = tuple(label(name) for name in ("paired", "label-only"))
    consumed = "gone" if version == 2 else "paired"
    receipt = AppliedExperience(
        consumed,
        "episode",
        "event-" + consumed,
        "actor",
        "learner",
        1,
        2,
        4,
        1,
        TrainingDiagnostic("numpy_binary_bce_preupdate_v1", 0.7),
    )
    erased = (
        (ErasedExperience(("episode", "gone"), "actor", 1, "event-gone", 2, 7, "opt_out"),)
        if version == 2
        else ()
    )
    return InboxCursor(
        version, "learner", 8, sources, labels, (receipt,), 8, stopped, 3 if stopped else 1, erased
    )


def encode(saved=None, policy=POLICY, limits=LIMITS):
    return NumpyInboxCheckpointCodec(policy).encode(
        cursor() if saved is None else saved, binding=BINDING, limits=limits
    )


def decode(raw, policy=POLICY, limits=LIMITS, binding=BINDING):
    return NumpyInboxCheckpointCodec(policy).decode(
        raw, binding=binding, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("evaluation", [False, True])
def test_should_preserve_full_live_unmatched_consumed_erased_cursor(
    version, stopped, order, replay, evaluation
):
    saved = cursor(version, stopped, order, replay, evaluation)
    raw = encode(saved)
    detached = decode(raw)
    assert encode(detached) == raw and vars(detached).keys() == vars(saved).keys()
    assert detached.completed_updates == (3 if stopped else 1)
    assert detached.erased == saved.erased and detached.applied == saved.applied
    for left, right in zip(saved.experiences, detached.experiences):
        assert left.permissions == right.permissions and left.candidate_ids == right.candidate_ids
        assert left.action_id == right.action_id and np.signbit(right.reward)
        assert_array(left.features, right.features)
    for left, right in zip(saved.labels, detached.labels):
        assert_array(left.targets, right.targets)
    detached.experiences[0].features[:] = 9
    assert encode(saved) == raw


def assert_array(left, right):
    assert (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )
    assert left.flags.c_contiguous == right.flags.c_contiguous
    assert left.flags.f_contiguous == right.flags.f_contiguous
    assert right.flags.owndata and not np.shares_memory(left, right)


RECORDS = {
    "cursor": InboxCursor,
    "source": Experience,
    "permissions": ExperiencePermissions,
    "label": LabelArrival,
    "applied": AppliedExperience,
    "diagnostic": TrainingDiagnostic,
    "erased": ErasedExperience,
}


def record(saved, name):
    return {
        "cursor": saved,
        "source": saved.experiences[0],
        "permissions": saved.experiences[0].permissions,
        "label": saved.labels[0],
        "applied": saved.applied[0],
        "diagnostic": saved.applied[0].diagnostic,
        "erased": saved.erased[0],
    }[name]


@pytest.mark.parametrize(
    "name,field", [(name, field.name) for name, kind in RECORDS.items() for field in fields(kind)]
)
def test_should_refuse_every_missing_supported_record_field(name, field):
    saved = cursor()
    object.__delattr__(record(saved, name), field)
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize("name", list(RECORDS))
def test_should_refuse_unknown_nested_native_record_fields(name):
    saved = cursor()
    object.__setattr__(record(saved, name), "future", object())
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize("role", ["inner_guard", "outer_selection", "final_test"])
@pytest.mark.parametrize("kind", ["source", "label"])
def test_should_refuse_forbidden_role_before_payload_inspection(role, kind, monkeypatch):
    body = json.loads(encode())
    saved = cursor()
    row = saved.experiences[0] if kind == "source" else saved.labels[0]
    object.__setattr__(row, "role", role)

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden-role payload was inspected")

    monkeypatch.setattr("src.adapters.numpy_inbox_checkpoint_codec.specifications", forbidden)
    with pytest.raises(ValueError):
        encode(saved)
    body["cursor"]["experiences" if kind == "source" else "labels"][0]["role"] = role
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "fault",
    [
        "permission",
        "duplicate_source",
        "duplicate_label",
        "duplicate_event",
        "duplicate_receipt",
        "pair_version",
        "pair_time",
        "receipt_version",
        "receipt_number",
        "receipt_time",
        "erased_overlap",
        "erased_event",
        "erased_future",
        "completed",
        "bool_tick",
        "identifier",
        "candidate",
        "diagnostic",
        "capacity",
    ],
)
def test_should_refuse_corrupt_complete_metadata_before_arrays(fault, monkeypatch):
    body = json.loads(encode())
    data = body["cursor"]
    if fault == "permission":
        data["experiences"][0]["permissions"]["training"] = False
    elif fault == "duplicate_source":
        data["experiences"].append(data["experiences"][0])
    elif fault == "duplicate_label":
        data["labels"].append(data["labels"][0])
    elif fault == "duplicate_event":
        data["labels"][1]["event_id"] = data["labels"][0]["event_id"]
    elif fault == "duplicate_receipt":
        data["applied"].append(data["applied"][0])
    elif fault == "pair_version":
        data["labels"][0]["model_version"] = "other"
    elif fault == "pair_time":
        data["labels"][0]["arrived_at"] = 0
    elif fault == "receipt_version":
        data["applied"][0]["learner_version"] = "other"
    elif fault == "receipt_number":
        data["applied"][0]["update_number"] = 2
    elif fault == "receipt_time":
        data["applied"][0]["applied_at"] = 9
    elif fault == "erased_overlap":
        data["erased"][0]["key"] = ["episode", "paired"]
    elif fault == "erased_event":
        data["erased"][0]["event_id"] = "event-paired"
    elif fault == "erased_future":
        data["erased"][0]["erased_at"] = 9
    elif fault == "completed":
        data["completed_updates"] = 0
    elif fault == "bool_tick":
        data["last_tick"] = True
    elif fault == "identifier":
        data["learner_version"] = "x" * 129
    elif fault == "candidate":
        data["experiences"][0]["candidate_ids"] *= 3
    elif fault == "diagnostic":
        data["applied"][0]["diagnostic"]["value"] = float("nan")
    else:
        data["capacity"] = 9

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid metadata reached arrays")

    monkeypatch.setattr("src.adapters.numpy_inbox_checkpoint_codec.specifications", forbidden)
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


@pytest.mark.parametrize(
    "fault", ["dtype", "nan", "range", "strided", "alias", "shape", "pair_rows", "foreign"]
)
def test_should_refuse_unsupported_native_payload_graph(fault):
    saved = cursor()
    if fault == "dtype":
        object.__setattr__(saved.labels[0], "targets", np.ones((2, 1), dtype=np.complex64))
    elif fault == "nan":
        saved.experiences[0].features[0, 0] = np.nan
    elif fault == "range":
        saved.labels[-1].targets[0, 0] = 2
    elif fault == "strided":
        object.__setattr__(saved.experiences[0], "features", np.zeros((2, 6))[:, ::2])
    elif fault == "alias":
        object.__setattr__(saved.experiences[1], "features", saved.experiences[0].features)
    elif fault == "shape":
        object.__setattr__(saved.experiences[0], "features", np.zeros((2, 4)))
    elif fault == "pair_rows":
        object.__setattr__(saved.labels[0], "targets", np.ones((3, 1)))
    else:
        object.__setattr__(saved.experiences[0], "features", object())
    with pytest.raises(ValueError):
        encode(saved)


@pytest.mark.parametrize(
    "fault",
    [
        "kind",
        "version",
        "extra",
        "schema",
        "schema_bool",
        "missing_order",
        "order",
        "ambiguous",
        "dtype",
        "shape",
        "bytes",
        "nan",
        "range",
    ],
)
def test_should_refuse_corrupt_wire_before_native_allocation(fault, monkeypatch):
    body = json.loads(encode())
    frame = body["cursor"]["labels"][-1]["targets"]
    if fault == "kind":
        body["kind"] = "other"
    elif fault == "version":
        body["codec_version"] = True
    elif fault == "extra":
        body["future"] = None
    elif fault == "schema":
        body["schema"]["input_dim"] = 4
    elif fault == "schema_bool":
        body["schema"]["max_records"] = True
    elif fault == "missing_order":
        del frame["order"]
    elif fault == "order":
        frame["order"] = "K"
    elif fault == "ambiguous":
        frame["order"] = "F"
    elif fault == "dtype":
        frame["dtype"] = "<f4"
    elif fault == "shape":
        frame["shape"] = [99999999999, 1]
    elif fault == "bytes":
        frame["data"] = "!" * len(frame["data"])
    else:
        frame["data"] = b64encode(
            np.full((2, 1), np.nan if fault == "nan" else 2.0).tobytes()
        ).decode()
    monkeypatch.setattr(
        np, "frombuffer", lambda *a, **kw: pytest.fail("invalid input allocated an array")
    )
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def test_should_enforce_exact_aggregate_wire_and_independent_policy_bounds(monkeypatch):
    saved = cursor()
    raw = encode(saved)
    size = sum(s.features.nbytes for s in saved.experiences) + sum(
        s.targets.nbytes for s in saved.labels
    )
    limits = replace(LIMITS, max_array_bytes=size, max_encoded_bytes=len(raw))
    assert encode(saved, limits=limits) == raw and encode(decode(raw, limits=limits)) == raw
    monkeypatch.setattr(
        np, "frombuffer", lambda *a, **kw: pytest.fail("resource refusal allocated arrays")
    )
    for small in (
        replace(limits, max_array_bytes=size - 1),
        replace(limits, max_encoded_bytes=len(raw) - 1),
        replace(limits, max_dimension=2),
    ):
        with pytest.raises(ValueError):
            encode(saved, limits=small)
        with pytest.raises(ValueError):
            decode(raw, limits=small)
    with pytest.raises(ValueError):
        decode(raw, policy=replace(POLICY, input_dim=4))


@pytest.mark.parametrize("field", ["source_sha256", "policy_sha256"])
def test_should_require_independent_original_binding(field):
    with pytest.raises(ValueError):
        decode(encode(), binding=replace(BINDING, **{field: "c" * 64}))


@pytest.mark.parametrize("raw", [b"{}", b"\xff", b"{", b'{"a":1,"a":2}', b"[" * 1500])
def test_should_refuse_malformed_duplicate_or_unbounded_json(raw):
    with pytest.raises(ValueError):
        decode(raw)


def test_should_require_canonical_exact_bytes_and_independent_content():
    raw = encode()
    with pytest.raises(ValueError):
        decode(raw + b" ")
    codec = NumpyInboxCheckpointCodec(POLICY)
    with pytest.raises(ValueError):
        codec.decode(
            raw + b" ", binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )
    mutable: Any = bytearray(raw)
    with pytest.raises(ValueError):
        codec.decode(
            mutable, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
        )


def test_should_round_trip_empty_cursor_without_creating_payloads(monkeypatch):
    saved: InboxCursor[Any, Any] = InboxCursor(1, "learner", 8, (), (), (), 8, False, 0)
    monkeypatch.setattr(
        np, "frombuffer", lambda *a, **kw: pytest.fail("empty state allocated an array")
    )
    assert decode(encode(saved)) == saved


@pytest.mark.parametrize("kind", ["source_only", "label_only", "pair"])
@pytest.mark.parametrize("reason", ["deleted", "expired", "opt_out"])
@pytest.mark.parametrize("stopped", [False, True])
def test_should_preserve_all_payload_free_tombstone_variants(kind, reason, stopped, monkeypatch):
    tombstone = ErasedExperience(
        ("episode", "gone"),
        "actor",
        None if kind == "label_only" else 1,
        None if kind == "source_only" else "event-gone",
        None if kind == "source_only" else 2,
        7,
        reason,
    )
    saved: InboxCursor[Any, Any] = InboxCursor(
        2, "learner", 8, (), (), (), 8, stopped, 2 if stopped else 0, (tombstone,)
    )
    monkeypatch.setattr(
        np, "frombuffer", lambda *a, **kw: pytest.fail("tombstones allocated payloads")
    )
    assert decode(encode(saved)) == saved


@pytest.mark.parametrize("field", [field.name for field in fields(InboxCodecPolicy)])
@pytest.mark.parametrize("value", [True, 0, -1, 2**63])
def test_should_refuse_invalid_independent_policy(field, value):
    with pytest.raises(ValueError):
        replace(POLICY, **{field: value})


@pytest.mark.parametrize("target", ["policy", "binding", "limits"])
def test_should_refuse_unknown_original_configuration_fields(target):
    policy, binding, limits = replace(POLICY), replace(BINDING), replace(LIMITS)
    object.__setattr__(
        {"policy": policy, "binding": binding, "limits": limits}[target], "future", 1
    )
    with pytest.raises(ValueError):
        NumpyInboxCheckpointCodec(policy).encode(cursor(), binding=binding, limits=limits)


def test_should_detach_metadata_and_preserve_source_on_partial_serialization_failure(monkeypatch):
    import src.adapters.inbox_checkpoint_payloads as payloads

    saved = cursor()
    expected = encode(saved)
    original = payloads.encode_frame
    calls = []

    def fail_late(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("fixed serialization failure")
        return original(*args)

    monkeypatch.setattr(payloads, "encode_frame", fail_late)
    with pytest.raises(ValueError):
        encode(saved)
    monkeypatch.setattr(payloads, "encode_frame", original)
    assert encode(saved) == expected


@pytest.mark.parametrize("dtype", NUMERIC_DTYPES)
@pytest.mark.parametrize("order", ["C", "F"])
def test_should_preserve_every_supported_real_numeric_width_and_byte_order(dtype, order):
    saved = cursor(order=order)
    for row in saved.experiences:
        array = np.array(np.arange(6).reshape(2, 3), dtype=dtype, order=order)
        if array.dtype.kind == "f":
            array[0, 0] = -0.0
        object.__setattr__(row, "features", array)
    for row in saved.labels:
        object.__setattr__(row, "targets", np.array([[1], [0]], dtype=dtype, order=order))
    detached = decode(encode(saved))
    for left, right in zip(saved.experiences, detached.experiences):
        assert_array(left.features, right.features)
    for left, right in zip(saved.labels, detached.labels):
        assert_array(left.targets, right.targets)
    assert encode(detached) == encode(saved)


def test_should_require_independent_feature_and_target_dtype_policies():
    saved = cursor()
    limited = replace(POLICY, feature_dtypes=("<f8",), target_dtypes=("<f8",))
    raw = encode(saved, policy=limited)
    assert encode(decode(raw, policy=limited), policy=limited) == raw
    with pytest.raises(ValueError):
        decode(raw, policy=POLICY)
    object.__setattr__(saved.labels[0], "targets", saved.labels[0].targets.astype(np.float32))
    with pytest.raises(ValueError):
        encode(saved, policy=limited)


def test_should_refuse_prior_unaccepted_f8_wire_version():
    body = json.loads(encode())
    assert body["codec_version"] == 2 and body["kind"] == "numpy_inbox_v2"
    body.update(codec_version=1, kind="numpy_inbox_v1")
    with pytest.raises(ValueError):
        decode(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())


def test_should_round_trip_actual_record_capture_without_native_work(monkeypatch):
    from src.adapters.numpy_learners import BackpropLearner
    from src.app.experience_inbox import ExperienceInbox
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
    from src.core.backprop_mlp import BackpropMLP
    from src.core.experience import LogicalClock

    def forbidden(*args, **kwargs):
        raise AssertionError("zero-native fixture called a native operation")

    for name in ("train_epoch", "predict_proba"):
        monkeypatch.setattr(BackpropMLP, name, forbidden)
    monkeypatch.setattr(BackpropLearner, "restore_state", forbidden)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    inbox = ExperienceInbox(
        BackpropLearner(BackpropMLP(3, 4, seed=23), learning_rate=0.03),
        clock=LogicalClock(8),
        budget=budget,
        learner_version="learner",
        max_experiences=8,
    )
    inbox.record_label(label("paired"))
    inbox.record_experience(source("paired"))
    inbox.record_experience(source("source-only"))
    saved = inbox.capture_cursor()
    raw = encode(saved)
    assert encode(decode(raw)) == raw
    assert budget.updates_completed == 0 and budget.started_at == 0.0
    assert encode(inbox.capture_cursor()) == raw
