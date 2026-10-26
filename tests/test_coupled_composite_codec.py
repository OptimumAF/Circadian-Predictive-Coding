"""Independent wire, complete data graphs, and bounded refusal controls.

These fixtures contain synthetic native records. Decoding grants no live owner,
callback, model, checkpoint reuse or restored lifecycle authority.
"""

from base64 import b64decode, b64encode
from collections import deque
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
from hashlib import sha256
import json
from typing import Any, Literal

import numpy as np
import pytest

import src.adapters.coupled_composite_codec as module
import src.adapters.coupled_composite_wire as wire_module
from src.adapters.coupled_composite_codec import CoupledCompositeCheckpointCodec
from src.adapters.coupled_composite_capture import capture_coupled_managed_composite
from src.core.checkpoint_codec import CodecBinding, CodecLimits
from src.core.inbox_codec_policy import NUMERIC_DTYPES
from src.core.managed_composite_state import ManagedCompositeCapture
from src.core.managed_lifecycle_state import AuthorityReference
from src.core.serving_ports import PromotionReceipt
from test_coupled_managed_composite_capture import (
    LIMITS as CAPTURE_LIMITS,
    forbid_operations,
    graph_factory as graph_factory,
    values,
)


BINDING = CodecBinding("a" * 64, "b" * 64)
LIMITS = CodecLimits(262144, 65536, 2, 32)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def independent_wire(state, capture_limits=CAPTURE_LIMITS):
    """Literal v1 token/frame oracle; never calls the production serializer."""
    nodes: list[dict[str, Any]] = []
    memo: dict[int, int] = {}
    retained: list[object] = []
    sequence_tags: dict[type[Any], str] = {tuple: "tuple", list: "list", deque: "deque"}

    def visit(value):
        if value is None or type(value) in (bool, int, float, str):
            return {"value": value}
        if id(value) in memo:
            return {"ref": memo[id(value)]}
        number = len(nodes)
        memo[id(value)] = number
        retained.append(value)
        node: dict[str, Any] = {}
        nodes.append(node)
        if type(value) is np.ndarray:
            order: Literal["C", "F"] = (
                "F" if value.flags.f_contiguous and not value.flags.c_contiguous else "C"
            )
            node.update(
                tag="array",
                dtype=value.dtype.str,
                shape=list(value.shape),
                order=order,
                writeable=bool(value.flags.writeable),
                data=b64encode(value.tobytes(order=order)).decode(),
            )
        elif type(value) is np.random.Generator:
            node.update(tag="pcg64", state=visit(value.bit_generator.state))
        elif is_dataclass(value) and not isinstance(value, type):
            node.update(
                tag="record",
                type=type(value).__module__ + "." + type(value).__name__,
                fields=[[field.name, visit(getattr(value, field.name))] for field in fields(value)],
            )
        elif type(value) is dict:
            node.update(tag="dict", items=[[visit(k), visit(v)] for k, v in value.items()])
        elif type(value) in (set, frozenset):
            # Complete normal fixtures contain only scalar/tuple unordered members;
            # the separate ambiguity test covers identity-based record ties.
            ordered = sorted(value, key=lambda item: canonical(item))
            node.update(
                tag="set" if type(value) is set else "frozenset", items=[visit(v) for v in ordered]
            )
        elif type(value) in (tuple, list, deque):
            node.update(
                tag=sequence_tags[type(value)],
                items=[visit(v) for v in value],
            )
            if type(value) is deque:
                node["maxlen"] = value.maxlen
        else:
            raise AssertionError("independent oracle received unsupported data")
        return {"ref": number}

    root = visit(state)
    return canonical(
        dict(
            codec_version=1,
            kind="coupled_managed_composite_v1",
            binding=asdict(BINDING),
            capture_limits=asdict(capture_limits),
            root=root,
            nodes=nodes,
        )
    )


def assert_graph(detached, original, forward=None, reverse=None):
    if forward is None:
        forward, reverse = {}, {}
    assert type(detached) is type(original)
    if original is None or type(original) in (bool, int, float, str):
        assert canonical(detached) == canonical(original)
        return
    if id(original) in forward:
        assert forward[id(original)] is detached
        return
    assert id(detached) not in reverse
    forward[id(original)], reverse[id(detached)] = detached, original
    if not (type(original) is tuple and not original):
        assert detached is not original
    if type(original) is np.ndarray:
        assert not np.shares_memory(detached, original)
        assert detached.dtype.str == original.dtype.str and detached.shape == original.shape
        assert detached.flags.writeable == original.flags.writeable
        assert (detached.flags.c_contiguous, detached.flags.f_contiguous) == (
            original.flags.c_contiguous,
            original.flags.f_contiguous,
        )
        assert detached.tobytes(order="A") == original.tobytes(order="A")
    elif type(original) is np.random.Generator:
        assert type(detached.bit_generator) is np.random.PCG64
        assert detached.bit_generator.state == original.bit_generator.state
    elif is_dataclass(original):
        for field in fields(original):
            assert_graph(
                getattr(detached, field.name), getattr(original, field.name), forward, reverse
            )
    elif type(original) is dict:
        assert list(detached) == list(original)
        for copied_key, key in zip(detached, original):
            assert_graph(copied_key, key, forward, reverse)
            assert_graph(detached[copied_key], original[key], forward, reverse)
    elif type(original) in (set, frozenset):
        assert detached == original
    else:
        assert len(detached) == len(original)
        if type(original) is deque:
            assert detached.maxlen == original.maxlen
        for a, b in zip(detached, original):
            assert_graph(a, b, forward, reverse)


@pytest.fixture
def state_factory(graph_factory, monkeypatch):
    def build(family="pc", guarded=False):
        graph = graph_factory(family, True, guarded)
        captured = capture_coupled_managed_composite(graph["owner"], limits=CAPTURE_LIMITS)
        forbid_operations(monkeypatch)
        return captured.state

    return build


def metadata(state):
    actor = next(
        values(record) for _, record in state.holders if record.kind.endswith(".PromotableActor")
    )
    return values(values(actor["_slot"])["bundle"])["metadata"]


def encode(state, codec=None, limits=LIMITS):
    codec = codec or CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    return codec.encode(state, binding=BINDING, limits=limits)


def decode(raw, codec=None, limits=LIMITS):
    codec = codec or CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    return codec.decode(
        raw, binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=limits
    )


@pytest.mark.parametrize("family", ("pc", "neutral", "active"))
@pytest.mark.parametrize("guarded", (False, True))
def test_should_match_independent_complete_wire_and_preserve_full_graph(
    state_factory, family, guarded
):
    state = state_factory(family, guarded)
    raw = encode(state)
    assert raw == independent_wire(state)
    restored = decode(raw)
    assert_graph(restored, state)
    assert type(restored) is not ManagedCompositeCapture and not hasattr(restored, "authority")
    life = restored.records.lifecycle
    assert life.registry.limits is life.lifecycle.policy.holders
    assert life.copy.limits is life.lifecycle.policy.owned_payload_copies
    assert life.copy.charged_bytes == state.records.lifecycle.copy.charged_bytes


@pytest.mark.parametrize("dtype", (*NUMERIC_DTYPES, "|b1"))
def test_should_preserve_every_admitted_dtype_and_raw_frame_without_conversion(
    state_factory, dtype
):
    state = state_factory()
    dt = np.dtype(dtype)
    if dt.kind == "b":
        array = np.frombuffer(bytes((0, 2, 255, 1)), dtype=dt).reshape(2, 2).copy()
    elif dt.kind == "u":
        array = np.array([[0, 1], [np.iinfo(dt).max, 2]], dtype=dt)
    elif dt.kind == "i":
        array = np.array([[np.iinfo(dt).min, -1], [np.iinfo(dt).max, 0]], dtype=dt)
    else:
        array = np.array([[-0.0, 0.25], [-0.5, 1.0]], dtype=dt)
    metadata(state)["typed"] = array
    raw = encode(state)
    frame = next(
        node
        for node in json.loads(raw)["nodes"]
        if node.get("tag") == "array" and node["dtype"] == dt.str and node["shape"] == [2, 2]
    )
    assert b64decode(frame["data"], validate=True) == array.tobytes(order="C")
    assert frame["order"] == "C" and frame["writeable"] is True
    actual = metadata(decode(raw))["typed"]
    assert actual.dtype.str == dtype and actual.tobytes() == array.tobytes()


@pytest.mark.parametrize(
    "shape,order,writeable",
    (
        ((), "C", True),
        ((0,), "C", True),
        ((0, 3), "C", False),
        ((2, 0), "C", True),
        ((2, 3), "F", False),
        ((1, 3), "F", False),
    ),
)
def test_should_preserve_scalar_empty_fortran_and_readonly_arrays(
    state_factory, shape, order, writeable
):
    state = state_factory()
    array = np.zeros(shape, dtype=">f8", order=order)
    array.flags.writeable = writeable
    metadata(state)["typed"] = array
    raw = encode(state)
    actual = metadata(decode(raw))["typed"]
    assert_graph(actual, array)
    if shape == (1, 3):
        frame = next(
            n
            for n in json.loads(raw)["nodes"]
            if n.get("tag") == "array" and n["shape"] == [1, 3] and n["dtype"] == ">f8"
        )
        assert frame["order"] == "C"


def test_should_preserve_collections_fifo_aliases_distinct_equal_arrays_and_pcg64_without_draws(
    state_factory,
):
    state = state_factory()
    meta = metadata(state)
    array = np.array([[0.25, -0.0]])
    rng = np.random.Generator(np.random.PCG64(23))
    before = deepcopy(rng.bit_generator.state)
    meta["fifo"] = {("episode", "z"): array, ("episode", "a"): array.copy()}
    meta["collections"] = [
        array,
        array,
        (array,),
        deque([array], maxlen=2),
        deque([], maxlen=0),
        {"z", "a"},
        frozenset(("z", "a")),
        rng,
        rng,
    ]
    restored = decode(encode(state))
    assert_graph(restored, state)
    actual = metadata(restored)
    items = actual["collections"]
    assert items[0] is items[1] is items[2][0] is items[3][0]
    assert items[0] is actual["fifo"][("episode", "z")]
    assert actual["fifo"][("episode", "a")] is not items[0]
    assert list(actual["fifo"]) == [("episode", "z"), ("episode", "a")]
    assert (
        items[7] is items[8] and rng.bit_generator.state == before == items[7].bit_generator.state
    )


def test_should_canonicalize_unordered_members_independently_of_insertion_or_identity(
    state_factory,
):
    state = state_factory()
    first = {"z", "a", "m"}
    second = set()
    for value in reversed(sorted(first)):
        second.add(value)
    metadata(state)["unordered"] = first
    other = deepcopy(state)
    metadata(other)["unordered"] = second
    assert encode(state) == encode(other)


def test_should_refuse_ambiguous_equal_identity_members_without_merging_outside_aliases(
    state_factory,
):
    state = state_factory()
    one, two = PromotionReceipt(0, "a", "b"), PromotionReceipt(0, "a", "b")
    metadata(state).update(outside=one, ambiguous={one, two})
    with pytest.raises(ValueError):
        encode(state)


WIRE_FAULTS = (
    "version_bool",
    "kind",
    "missing_envelope",
    "extra_envelope",
    "root_bool_ref",
    "root_negative_ref",
    "root_dangling",
    "root_scalar",
    "unknown_tag",
    "unknown_record",
    "missing_field",
    "extra_field",
    "duplicate_field",
    "field_order",
    "extra_node_field",
    "unknown_dtype",
    "shape_bool",
    "shape_negative",
    "shape_rank",
    "shape_oversize",
    "order",
    "writeable_int",
    "bad_base64",
    "base64_length",
    "nonfinite_array",
    "cycle",
    "unreachable",
    "token_extra",
    "token_both",
    "duplicate_dict_key",
    "duplicate_set_member",
    "deque_bool_maxlen",
    "pcg64_kind",
)


@pytest.mark.parametrize("fault", WIRE_FAULTS)
def test_should_refuse_malformed_complete_wire_with_independent_matching_digest(
    state_factory, fault
):
    state = state_factory()
    metadata(state).update(
        unordered={"one", "two"},
        queue=deque([1], maxlen=2),
        generator=np.random.Generator(np.random.PCG64(23)),
    )
    document = json.loads(encode(state))
    nodes = document["nodes"]
    root = nodes[document["root"]["ref"]]
    array = next(node for node in nodes if node["tag"] == "array")
    if fault == "version_bool":
        document["codec_version"] = True
    elif fault == "kind":
        document["kind"] = "other"
    elif fault == "missing_envelope":
        del document["binding"]
    elif fault == "extra_envelope":
        document["authority"] = None
    elif fault == "root_bool_ref":
        document["root"] = {"ref": True}
    elif fault == "root_negative_ref":
        document["root"] = {"ref": -1}
    elif fault == "root_dangling":
        document["root"] = {"ref": len(nodes)}
    elif fault == "root_scalar":
        document["root"] = {"value": None}
    elif fault == "unknown_tag":
        root["tag"] = "pickle"
    elif fault == "unknown_record":
        root["type"] = "builtins.object"
    elif fault == "missing_field":
        root["fields"].pop()
    elif fault == "extra_field":
        root["fields"].append(["authority", {"value": None}])
    elif fault == "duplicate_field":
        root["fields"].append(root["fields"][0])
    elif fault == "field_order":
        root["fields"].reverse()
    elif fault == "extra_node_field":
        array["owner"] = 1
    elif fault == "unknown_dtype":
        array["dtype"] = "|O"
    elif fault == "shape_bool":
        array["shape"] = [True]
    elif fault == "shape_negative":
        array["shape"] = [-1]
    elif fault == "shape_rank":
        array["shape"] = [1, 1, 1]
    elif fault == "shape_oversize":
        array["shape"] = [33, 1]
    elif fault == "order":
        array["order"] = "A"
    elif fault == "writeable_int":
        array["writeable"] = 1
    elif fault == "bad_base64":
        array["data"] = "!" * len(array["data"])
    elif fault == "base64_length":
        array["data"] += "AAAA"
    elif fault == "nonfinite_array":
        raw = bytearray(b64decode(array["data"]))
        raw[:8] = np.array([np.nan], dtype=array["dtype"]).tobytes()
        array["data"] = b64encode(raw).decode()
    elif fault == "cycle":
        root["fields"][0][1] = {"ref": document["root"]["ref"]}
    elif fault == "unreachable":
        nodes.append({"tag": "list", "items": []})
    elif fault == "token_extra":
        document["root"]["extra"] = None
    elif fault == "token_both":
        document["root"]["value"] = None
    elif fault == "duplicate_dict_key":
        node = next(node for node in nodes if node["tag"] == "dict" and node["items"])
        node["items"].append(deepcopy(node["items"][0]))
    elif fault == "duplicate_set_member":
        node = next(node for node in nodes if node["tag"] == "set" and node["items"])
        node["items"].append(deepcopy(node["items"][0]))
    elif fault == "deque_bool_maxlen":
        next(n for n in nodes if n["tag"] == "deque")["maxlen"] = True
    else:
        node = next(n for n in nodes if n["tag"] == "pcg64")
        node["tag"] = "mt19937"
    with pytest.raises(ValueError):
        decode(canonical(document))


@pytest.mark.parametrize(
    "fault",
    (
        "admitted",
        "copy_charge",
        "ownership_alias",
        "copy_alias",
        "future_anchor",
        "runtime_work",
        "consolidation_revision",
        "policy",
    ),
)
def test_should_refuse_invalid_lifecycle_policy_and_cross_record_relationships(
    state_factory, fault
):
    state = state_factory()
    life = state.records.lifecycle
    if fault == "admitted":
        object.__setattr__(
            life.lifecycle, "admitted_bytes", life.lifecycle.policy.max_lifetime_ingress_bytes + 1
        )
    elif fault == "copy_charge":
        object.__setattr__(life.copy, "charged_bytes", True)
    elif fault == "ownership_alias":
        object.__setattr__(life.registry, "limits", deepcopy(life.registry.limits))
    elif fault == "copy_alias":
        object.__setattr__(life.copy, "limits", deepcopy(life.copy.limits))
    elif fault == "future_anchor":
        object.__setattr__(
            life.owner,
            "declaration_ticks",
            ((life.owner.catalog[0].key, life.lifecycle.last_tick + 1),),
        )
    elif fault == "runtime_work":
        object.__setattr__(
            state.records.owner, "budget_updates", state.records.owner.budget_updates + 1
        )
    elif fault == "consolidation_revision":
        object.__setattr__(
            state.records.consolidation, "revision", state.records.owner.revision + 1
        )
    else:
        object.__setattr__(life.lifecycle.policy, "max_retention_seconds", -1.0)
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize("fault", ("capture", "reference", "opaque"))
def test_should_refuse_live_authority_or_opaque_hooks_without_invocation(state_factory, fault):
    state = state_factory()

    class Poison:
        def __deepcopy__(self, memo):
            pytest.fail("opaque copy")

        def __reduce__(self):
            pytest.fail("opaque reduction")

    value = {
        "capture": lambda: ManagedCompositeCapture(state, ()),
        "reference": lambda: AuthorityReference("original", Poison()),
        "opaque": Poison,
    }[fault]()
    metadata(state)["forbidden"] = value
    with pytest.raises(ValueError):
        encode(state)


@pytest.mark.parametrize("fault", ("raw_type", "raw_subclass", "oversize", "digest", "binding"))
def test_should_refuse_outer_controls_before_hash_or_parser_when_possible(
    state_factory, monkeypatch, fault
):
    raw: Any = encode(state_factory())
    kwargs: dict[str, Any] = dict(
        binding=BINDING, expected_sha256=sha256(raw).hexdigest(), limits=LIMITS
    )
    codec = CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    if fault == "raw_type":
        raw = bytearray(raw)
    elif fault == "raw_subclass":

        class Bytes(bytes):
            pass

        raw = Bytes(raw)
    elif fault == "oversize":
        kwargs["limits"] = replace(LIMITS, max_encoded_bytes=len(raw) - 1)
    elif fault == "digest":
        kwargs["expected_sha256"] = "bad"
    elif fault == "binding":
        kwargs["binding"] = object()
    with monkeypatch.context() as patcher:
        patcher.setattr(module, "sha256", lambda *a: pytest.fail("hash before outer admission"))
        patcher.setattr(
            module.json, "loads", lambda *a, **k: pytest.fail("parse before outer admission")
        )
        with pytest.raises(ValueError):
            codec.decode(raw, **kwargs)


@pytest.mark.parametrize("representation", ("space", "indent", "duplicate", "nan"))
def test_should_refuse_noncanonical_or_duplicate_json(state_factory, representation):
    document = json.loads(encode(state_factory()))
    if representation == "space":
        raw = b" " + canonical(document)
    elif representation == "indent":
        raw = json.dumps(document, indent=2).encode()
    elif representation == "duplicate":
        raw = b'{"codec_version":1,' + canonical(document)[1:]
    else:
        raw = canonical(document).replace(b'"codec_version":1', b'"codec_version":NaN')
    with pytest.raises(ValueError):
        decode(raw)


def test_should_use_native_layer_cap_independently_of_full_graph_depth(state_factory):
    state = state_factory()
    raw = encode(state, limits=replace(LIMITS, max_layers=2))
    assert_graph(decode(raw, limits=replace(LIMITS, max_layers=2)), state)
    with pytest.raises(ValueError):
        encode(state, limits=replace(LIMITS, max_layers=1))
    with pytest.raises(ValueError):
        decode(raw, limits=replace(LIMITS, max_layers=1))


def test_should_admit_exact_wire_array_and_dimension_limits(state_factory):
    state = state_factory()
    raw = encode(state)
    arrays = [node for node in json.loads(raw)["nodes"] if node["tag"] == "array"]
    size = sum(len(b64decode(node["data"])) for node in arrays)
    dimension = max((max(node["shape"], default=0) for node in arrays), default=1)
    tight = replace(
        LIMITS, max_encoded_bytes=len(raw), max_array_bytes=size, max_dimension=dimension
    )
    assert encode(state, limits=tight) == raw
    assert_graph(decode(raw, limits=tight), state)


@pytest.mark.parametrize(
    "field", ("max_encoded_bytes", "max_array_bytes", "max_dimension", "max_layers")
)
def test_should_admit_complete_encoding_before_any_frame_copy(state_factory, monkeypatch, field):
    state = state_factory()
    overrides: dict[str, Any] = {field: 1}
    bad = replace(LIMITS, **overrides)
    monkeypatch.setattr(
        wire_module, "encode_array", lambda *a, **k: pytest.fail("array copied before admission")
    )
    with pytest.raises(ValueError):
        encode(state, limits=bad)


@pytest.mark.parametrize(
    "fault",
    (
        "late_dtype",
        "late_shape",
        "late_base64",
        "late_extra",
        "cycle",
        "unreachable",
        "edges",
        "array_bytes",
        "native_layers",
        "dimension",
    ),
)
def test_should_validate_entire_wire_before_even_the_first_array_allocation(
    state_factory, monkeypatch, fault
):
    raw = encode(state_factory())
    doc = json.loads(raw)
    arrays = [node for node in doc["nodes"] if node["tag"] == "array"]
    kwargs: dict[str, Any] = {}
    if fault == "late_dtype":
        arrays[-1]["dtype"] = "|O"
    elif fault == "late_shape":
        arrays[-1]["shape"] = [True]
    elif fault == "late_base64":
        arrays[-1]["data"] = "!" * len(arrays[-1]["data"])
    elif fault == "late_extra":
        arrays[-1]["opaque"] = None
    elif fault == "cycle":
        doc["nodes"][0]["fields"][0][1] = {"ref": 0}
    elif fault == "unreachable":
        doc["nodes"].append({"tag": "list", "items": []})
    elif fault == "edges":
        node = next(n for n in doc["nodes"] if n["tag"] == "tuple")
        node["items"] = [{"value": None}] * (CAPTURE_LIMITS.max_nodes + 1)
    elif fault == "array_bytes":
        kwargs["limits"] = replace(LIMITS, max_array_bytes=1)
    elif fault == "native_layers":
        kwargs["limits"] = replace(LIMITS, max_layers=1)
    else:
        kwargs["limits"] = replace(LIMITS, max_dimension=1)
    monkeypatch.setattr(
        wire_module, "decode_array", lambda *a, **k: pytest.fail("premature array allocation")
    )
    with pytest.raises(ValueError):
        decode(canonical(doc), **kwargs)


def test_should_refuse_reordered_node_ids_even_when_all_references_are_rebound(state_factory):
    doc = json.loads(encode(state_factory()))
    doc["nodes"][1], doc["nodes"][2] = doc["nodes"][2], doc["nodes"][1]

    def rebind(value):
        if type(value) is dict:
            if set(value) == {"ref"}:
                if value["ref"] == 1:
                    value["ref"] = 2
                elif value["ref"] == 2:
                    value["ref"] = 1
            else:
                for item in value.values():
                    rebind(item)
        elif type(value) is list:
            for item in value:
                rebind(item)

    rebind(doc)
    with pytest.raises(ValueError):
        decode(canonical(doc))


@pytest.mark.parametrize("fault", ("cycle", "opaque"))
def test_should_refuse_invalid_data_graph_before_frame_copy(state_factory, monkeypatch, fault):
    state = state_factory()
    meta = metadata(state)
    meta["invalid"] = meta if fault == "cycle" else object()
    monkeypatch.setattr(
        wire_module, "encode_array", lambda *a, **k: pytest.fail("copy before graph admission")
    )
    with pytest.raises(ValueError):
        encode(state)


def test_should_refuse_wrong_content_digest_without_changing_the_input_graph(state_factory):
    state = state_factory()
    raw = encode(state)
    with pytest.raises(ValueError):
        CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS).decode(
            raw, binding=BINDING, expected_sha256="c" * 64, limits=LIMITS
        )
    assert independent_wire(state) == raw


def test_should_decode_promotion_handles_as_data_without_original_authority(
    graph_factory, monkeypatch
):
    graph = graph_factory("active", True, guarded=True)
    original = graph["promotion"]
    state = capture_coupled_managed_composite(graph["owner"], limits=CAPTURE_LIMITS).state
    forbid_operations(monkeypatch)
    restored = decode(encode(state))
    promotion = next(
        values(record)
        for _, record in restored.holders
        if record.kind.endswith(".ServingPromotionController")
    )
    ticket = promotion["_pending"][0][2]
    receipt = promotion["_latest"][0]
    assert ticket is not next(iter(original._pending)) and receipt is not original._latest[0]
    assert asdict(ticket) == asdict(next(iter(original._pending)))
    assert asdict(receipt) == asdict(original._latest[0])
    with pytest.raises(ValueError, match="foreign"):
        original._require_ticket(ticket)
    with pytest.raises(ValueError, match="foreign"):
        original.rollback(receipt)


def test_should_own_constructor_limits_despite_later_valid_caller_broadening(state_factory):
    caller = deepcopy(CAPTURE_LIMITS)
    codec = CoupledCompositeCheckpointCodec(capture_limits=caller)
    object.__setattr__(caller, "max_nodes", caller.max_nodes + 1)
    object.__setattr__(caller.records, "max_records", caller.records.max_records + 1)
    state = state_factory()
    raw = encode(state, codec)
    assert json.loads(raw)["capture_limits"] == asdict(CAPTURE_LIMITS)
    assert_graph(decode(raw, codec), state)


@pytest.mark.parametrize(
    "field",
    ("max_nodes", "max_depth", "max_array_bytes", "max_dimension", "records", "nested_records"),
)
def test_should_revalidate_its_own_complete_limits_before_array_work(
    state_factory, monkeypatch, field
):
    state = state_factory()
    codec = CoupledCompositeCheckpointCodec(capture_limits=CAPTURE_LIMITS)
    raw = encode(state, codec)
    if field == "nested_records":
        object.__setattr__(codec._capture_limits.records, "max_records", True)
    else:
        object.__setattr__(codec._capture_limits, field, object() if field == "records" else True)
    monkeypatch.setattr(
        wire_module, "encode_array", lambda *a, **k: pytest.fail("frame copy before owned controls")
    )
    monkeypatch.setattr(
        wire_module,
        "decode_array",
        lambda *a, **k: pytest.fail("array allocation before owned controls"),
    )
    with pytest.raises(ValueError):
        encode(state, codec)
    with pytest.raises(ValueError):
        decode(raw, codec)


@pytest.mark.parametrize("invalid", (None, True, {}))
def test_should_require_exact_constructor_limits_without_native_operations(invalid):
    with pytest.raises(ValueError):
        CoupledCompositeCheckpointCodec(capture_limits=invalid)


@pytest.mark.parametrize("field", ("binding", "capture_limits"))
def test_should_bind_wire_to_independent_original_controls_before_allocation(
    state_factory, monkeypatch, field
):
    document = json.loads(encode(state_factory()))
    if field == "binding":
        document["binding"]["source_sha256"] = "c" * 64
    else:
        document["capture_limits"]["max_nodes"] += 1
    monkeypatch.setattr(
        wire_module,
        "decode_array",
        lambda *a, **k: pytest.fail("allocation before independent binding"),
    )
    with pytest.raises(ValueError):
        decode(canonical(document))


@pytest.mark.parametrize("kind", ("capture", "reference"))
def test_should_refuse_authority_objects_as_the_root_data(state_factory, kind):
    state = state_factory()
    root = (
        ManagedCompositeCapture(state, ())
        if kind == "capture"
        else AuthorityReference("root", state)
    )
    with pytest.raises(ValueError):
        encode(root)


@pytest.mark.parametrize("container", (set, frozenset))
def test_should_order_distinct_unordered_generators_by_full_retained_state_without_draws(
    state_factory, container
):
    state = state_factory()
    generators = [np.random.Generator(np.random.PCG64(seed)) for seed in (23, 41, 59)]
    original_states = [deepcopy(generator.bit_generator.state) for generator in generators]
    metadata(state).update(generators=container(generators), outside=generators[0])
    raw = encode(state)
    restored = metadata(decode(raw))
    actual = restored["generators"]
    assert type(actual) is container and len(actual) == 3
    assert sorted(canonical(generator.bit_generator.state) for generator in actual) == sorted(
        canonical(value) for value in original_states
    )
    assert restored["outside"] in actual
    other = deepcopy(state)
    reversed_generators = [np.random.Generator(np.random.PCG64(seed)) for seed in (59, 41, 23)]
    metadata(other).update(
        generators=container(reversed_generators), outside=reversed_generators[-1]
    )
    assert encode(other) == raw
    assert [generator.bit_generator.state for generator in generators] == original_states


def test_should_not_emit_a_shallow_first_alias_graph_that_its_decoder_refuses(
    state_factory, monkeypatch
):
    state = state_factory()

    def height(value, memo):
        if value is None or type(value) in (bool, int, float, str, np.ndarray):
            return 0
        if id(value) in memo:
            return memo[id(value)]
        if is_dataclass(value):
            children = [getattr(value, field.name) for field in fields(value)]
        elif type(value) is dict:
            children = [item for pair in value.items() for item in pair]
        else:
            children = list(value)
        result = max((1 + height(child, memo) for child in children), default=0)
        memo[id(value)] = result
        return result

    baseline = height(state, {})
    shared: Any = np.array([[0.25]])
    for _ in range(8):
        shared = [shared]
    deep = shared
    for _ in range(12):
        deep = [deep]
    metadata(state).update(shallow=shared, deep=deep)
    bound = height(state, {}) - 1
    assert baseline < bound < CAPTURE_LIMITS.max_depth
    codec = CoupledCompositeCheckpointCodec(capture_limits=replace(CAPTURE_LIMITS, max_depth=bound))
    calls: list[object] = []
    original = wire_module.encode_array

    def copying(value, node):
        calls.append(value)
        return original(value, node)

    monkeypatch.setattr(wire_module, "encode_array", copying)
    try:
        raw = encode(state, codec)
    except ValueError:
        assert not calls
    else:
        assert_graph(decode(raw, codec), state)
