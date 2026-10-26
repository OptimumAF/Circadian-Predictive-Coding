"""Actual issued legacy histories, with no native operations during capture.

Two shared constructor-only graphs exercise genuine checkpoint retirement and
prepared promotions. Fake replay controls remain separately scoped; an empty
native replay buffer is never presented as qualified nonempty lineage.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass, replace
import json
from typing import Any

import numpy as np
import pytest

import src.adapters.numpy_capture_example as example
import src.adapters.numpy_composite_capture as adapter
from src.adapters.numpy_learners import BackpropLearner, CircadianLearner, BackpropSnapshot
from src.core.backprop_mlp import BackpropMLP
from src.core.circadian_predictive_coding import (
    CircadianNetworkSnapshot,
    CircadianPredictiveCodingNetwork,
    ReplaySnapshot,
)
from src.core.managed_composite_state import CompositeCaptureLimits, SourceRecord
from src.core.managed_lifecycle_state import LifecycleCaptureLimits


LIMITS = CompositeCaptureLimits(LifecycleCaptureLimits(64, 128), 8192, 64, 65536, 32)


def forbidden(*args, **kwargs):
    pytest.fail("native operation or unadmitted payload copy")


@pytest.fixture(scope="module")
def graphs():
    # Why: actual setup admission/handoff must happen before operational methods
    # are forbidden. Reuse exactly these two original owners throughout probes.
    result = [example._setup_case(family) for family in ("backprop", "circadian")]
    with pytest.MonkeyPatch.context() as guards:
        for kind in (
            BackpropLearner,
            CircadianLearner,
            BackpropMLP,
            CircadianPredictiveCodingNetwork,
        ):
            for name in (
                "__init__",
                "fork",
                "train_batch",
                "train_epoch",
                "predict",
                "predict_proba",
                "snapshot_state",
                "restore_state",
                "infer",
                "infer_latents",
                "sleep",
                "split_neuron",
                "prune_neuron",
            ):
                if hasattr(kind, name):
                    guards.setattr(kind, name, forbidden)
        yield result
    for graph in result:
        assert graph["budget"].updates_completed == 0
        assert graph["sampler"]._thread is None
        assert graph["life"]._copy_budget._charged <= 4096


def values(record):
    assert type(record) is SourceRecord
    return dict(record.fields)


def holder(captured, graph, original):
    number = next(
        number
        for number, (_, reference) in graph["life"]._registry._holders.items()
        if reference() is original
    )
    return values(dict(captured.state.holders)[number])


def capture(graph, limits=LIMITS):
    return adapter.capture_numpy_managed_composite(graph["owner"], limits=limits)


def assert_owned_graph(detached, original, forward=None, reverse=None):
    """Compare every field/edge, permitting deepcopy's immutable tuple reuse."""
    if forward is None:
        forward, reverse = {}, {}
    assert type(detached) is type(original)
    if original is None or type(original) in (bool, int, float, str):
        assert json.dumps(detached, allow_nan=False) == json.dumps(original, allow_nan=False)
        return
    if id(original) in forward:
        assert forward[id(original)] is detached
        return
    assert id(detached) not in reverse
    forward[id(original)], reverse[id(detached)] = detached, original
    # Why: deepcopy may return an unchanged all-immutable tuple. Recursing still
    # requires every reachable mutable child to detach; aliases stay bijective.
    if type(original) not in (tuple, frozenset):
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
        assert type(detached.bit_generator) is type(original.bit_generator) is np.random.PCG64
        assert detached.bit_generator is not original.bit_generator
        assert detached.bit_generator.state == original.bit_generator.state
    elif is_dataclass(original):
        for field in fields(original):
            assert_owned_graph(
                getattr(detached, field.name), getattr(original, field.name), forward, reverse
            )
    elif type(original) is dict:
        assert list(detached) == list(original)
        for copied_key, key in zip(detached, original):
            assert_owned_graph(copied_key, key, forward, reverse)
            assert_owned_graph(detached[copied_key], original[key], forward, reverse)
    elif type(original) in (list, tuple, deque):
        assert len(detached) == len(original)
        if type(original) is deque:
            assert detached.maxlen == original.maxlen
        for copied, value in zip(detached, original):
            assert_owned_graph(copied, value, forward, reverse)
    elif type(original) in (set, frozenset):
        assert detached == original
        for copied, value in zip(sorted(detached, key=repr), sorted(original, key=repr)):
            assert_owned_graph(copied, value, forward, reverse)
    else:
        pytest.fail(f"unexamined native graph type: {type(original).__qualname__}")


def native_snapshot_fields(snapshot, model):
    assert type(snapshot) in (BackpropSnapshot, CircadianNetworkSnapshot)
    assert snapshot.input_dim == model.input_dim
    if type(snapshot) is BackpropSnapshot:
        assert snapshot.format_version == 1 and snapshot.hidden_dims == model.hidden_dims
    else:
        assert snapshot.format_version == 2 and snapshot.initial_hidden_dims == model.hidden_dims
        assert snapshot.min_hidden_dim == model._min_hidden_dim
        assert snapshot.max_hidden_dim == model.max_hidden_dim
    assert snapshot.state.keys() == vars(model).keys()
    assert_owned_graph(snapshot.state, vars(model))
    if type(snapshot) is BackpropSnapshot:
        assert snapshot.state["weight_input_hidden"] is snapshot.state["_hidden_weights"][0]
        assert snapshot.state["bias_hidden"] is snapshot.state["_hidden_biases"][0]
    else:
        assert snapshot.config is snapshot.state["config"]
        assert not snapshot.state["_replay_memory"]


@pytest.mark.parametrize("index", (0, 1))
def test_should_detach_all_original_native_and_populated_issued_histories(
    graphs, monkeypatch, index
):
    graph = graphs[index]
    life, old, new = graph["life"], graph["old"], graph["new"]
    assert old._retired is True and new._retired is False
    assert graph["shared"]._runtime is new and old is not new
    assert old._inbox._historical_completed_updates == 0
    before = life.payload_byte_snapshot()
    passes: list[dict[int, Any]] = []
    original = adapter.copy_numpy_composite

    def copying(state, limits, memo):
        assert life._copy_budget._charged == before.charged_bytes + before.observed_retained_bytes
        passes.append(memo)
        return original(state, limits, memo)

    monkeypatch.setattr(adapter, "copy_numpy_composite", copying)
    result = capture(graph)
    assert len(passes) == 2 and passes[0] is passes[1]
    assert result.state.retained_payload_bytes == before.observed_retained_bytes
    assert result.state.records.lifecycle.copy.charged_bytes == life._copy_budget._charged
    for runtime in (old, new):
        copied = holder(result, graph, runtime)
        assert copied.keys() == vars(runtime).keys()
        assert copied["_retired"] is runtime._retired
        native_snapshot_fields(values(copied["_candidate"])["_model"], runtime._candidate._model)
        inbox = values(copied["_inbox"])
        assert inbox.keys() == vars(runtime._inbox).keys()
        assert_owned_graph(inbox["_experiences"], runtime._inbox._experiences)
        assert_owned_graph(inbox["_labels"], runtime._inbox._labels)
        assert (
            inbox["_historical_completed_updates"] == runtime._inbox._historical_completed_updates
        )
    slot = values(holder(result, graph, graph["actor"])["_slot"])
    bundle = values(slot["bundle"])
    native_snapshot_fields(
        values(bundle["learner"])["_model"], graph["actor"]._slot.bundle.learner._model
    )
    assert_owned_graph(bundle["metadata"], graph["actor"]._slot.bundle.metadata)
    assert_owned_graph(bundle["cache"], graph["actor"]._slot.bundle.cache)
    checkpoint = holder(result, graph, graph["checkpoint"])
    assert checkpoint["_models"][0] is holder(result, graph, new)["_candidate"]
    cp = checkpoint["_pending"]
    pr = holder(result, graph, graph["promotion"])["_pending"]
    assert len(cp) == len(pr) == 1
    view = values(cp[0][1])["view"]
    assert_owned_graph(view, graph["pending_view"])
    pending_bundle = values(values(pr[0][1])["bundle"])
    native_snapshot_fields(
        values(pending_bundle["learner"])["_model"], graph["pending_bundle"].learner._model
    )
    assert_owned_graph(pending_bundle["metadata"], graph["pending_bundle"].metadata)
    assert pr[0][2] is not graph["ticket"]
    # prepare deliberately deep-copies the issued ticket report. Compare both
    # complete report graphs together to preserve the actual source aliases.
    assert_owned_graph(
        (pr[0][2].report, values(pr[0][1])["report"]),
        (graph["ticket"].report, graph["promotion"]._pending[graph["ticket"]].report),
    )
    with pytest.raises(ValueError, match="foreign"):
        graph["promotion"]._require_ticket(pr[0][2])
    with pytest.raises(ValueError, match="foreign"):
        graph["checkpoint"]._require(deepcopy(graph["pending_checkpoint"]))
    assert any(ref.value is graph["pending_checkpoint"] for ref in result.authority)
    assert any(ref.value is graph["ticket"] for ref in result.authority)
    assert len({ref.path for ref in result.authority}) == len(result.authority)
    assert graph["checkpoint"]._require(graph["pending_checkpoint"]).view is graph["pending_view"]
    assert graph["promotion"]._require_ticket(graph["ticket"]).bundle is graph["pending_bundle"]


@pytest.mark.parametrize("index", (0, 1))
def test_should_preserve_raw_order_readonly_aliases_and_rng_without_drawing(
    graphs, monkeypatch, index
):
    graph = graphs[index]
    model = graph["old"]._candidate._model
    weight = np.asfortranarray(model.weight_input_hidden)
    weight.flags.writeable = False
    monkeypatch.setattr(model, "weight_input_hidden", weight)
    if type(model) is BackpropMLP:
        monkeypatch.setattr(model, "_hidden_weights", [weight, *model._hidden_weights[1:]])
    raw = graph["actor"]._slot.bundle.metadata
    name = next(name for name, value in raw.items() if type(value) is np.ndarray)
    typed = np.array([[-0.0, 0.25, 0.5]], dtype=">f8", order="F")
    typed.flags.writeable = False
    monkeypatch.setitem(raw, name, typed)
    rng_states = {
        name: deepcopy(value.bit_generator.state)
        for name, value in vars(model).items()
        if type(value) is np.random.Generator
    }
    if index == 1:
        assert rng_states
    result = capture(graph)
    native = values(holder(result, graph, graph["old"])["_candidate"])["_model"]
    actual = native.state["weight_input_hidden"]
    if type(model) is BackpropMLP:
        assert actual is native.state["_hidden_weights"][0]
    assert actual.flags.f_contiguous and not actual.flags.c_contiguous
    assert not actual.flags.writeable and not np.shares_memory(actual, weight)
    assert actual.tobytes(order="A") == weight.tobytes(order="A")
    copied_raw = values(values(holder(result, graph, graph["actor"])["_slot"])["bundle"])[
        "metadata"
    ][name]
    assert copied_raw.dtype.str == ">f8" and not copied_raw.flags.writeable
    assert copied_raw.tobytes(order="A") == typed.tobytes(order="A")
    for name, state in rng_states.items():
        assert (
            vars(model)[name].bit_generator.state == native.state[name].bit_generator.state == state
        )
        assert native.state[name] is not vars(model)[name]


@pytest.mark.parametrize(
    "fault",
    (
        "not_paused",
        "serving",
        "training",
        "checkpointing",
        "revoked",
        "opted_out",
        "expired_tick",
        "foreign_clock",
        "unknown_field",
        "bytes",
        "nodes",
        "depth",
    ),
)
def test_should_refuse_unavailable_or_unbounded_sources_before_copy_and_charge(
    graphs, monkeypatch, fault
):
    graph = graphs[0]
    life, gate, owner = graph["life"], graph["gate"], graph["owner"]
    limits = LIMITS
    if fault == "not_paused":
        monkeypatch.setattr(gate, "_paused", False)
    elif fault in ("serving", "training", "checkpointing"):
        monkeypatch.setattr(gate, "_" + fault, 1 if fault == "serving" else True)
    elif fault == "revoked":
        monkeypatch.setattr(owner, "_revoked_keys", set(owner._catalog))
    elif fault == "opted_out":
        monkeypatch.setattr(
            owner, "_opted_out", {d.provenance.subject_id for d in owner._catalog.values()}
        )
    elif fault == "expired_tick":
        monkeypatch.setattr(graph["clock"], "_time", life._policy.max_retention_ticks)
    elif fault == "foreign_clock":
        monkeypatch.setattr(graph["budget"], "clock", lambda: 8.0)
    elif fault == "unknown_field":
        monkeypatch.setattr(graph["new"], "unknown", None, raising=False)
    elif fault == "bytes":
        limits = replace(LIMITS, max_array_bytes=1)
    elif fault == "nodes":
        limits = replace(LIMITS, max_nodes=1)
    else:
        limits = replace(LIMITS, max_depth=1)
    charged, last_tick = life._copy_budget._charged, life._last_tick
    monkeypatch.setattr(adapter, "deepcopy", forbidden)
    try:
        with pytest.raises(ValueError):
            capture(graph, limits)
        assert life._copy_budget._charged == charged
    finally:
        if fault == "expired_tick":
            life._last_tick = last_tick  # Restore only the synthetic invalid clock observation.


@pytest.mark.parametrize(
    "fault", ("integrity", "build", "sampler", "owner", "promotion_builder", "ticket_generation")
)
def test_should_refuse_corrupt_issued_bindings_without_copy_or_new_charge(
    graphs, monkeypatch, fault
):
    graph = graphs[1]
    pending = graph["checkpoint"]._pending[graph["pending_checkpoint"]]
    promotion = graph["promotion"]._pending[graph["ticket"]]
    value: Any
    target, name, value = pending, "integrity", "0" * 64
    if fault == "build":
        name, value = "build", object()
    elif fault == "sampler":
        name, value = "sampler_state", (lambda: 100, pending.sampler_state[1])
    elif fault == "owner":
        name, value = "owner", graphs[0]["new"]
    elif fault == "promotion_builder":
        target, name, value = promotion, "builder", object()
    elif fault == "ticket_generation":
        target, name, value = (
            graph["ticket"],
            "actor_generation",
            graph["ticket"].actor_generation + 1,
        )
    original = getattr(target, name)
    before = graph["life"]._copy_budget._charged
    object.__setattr__(target, name, value)
    monkeypatch.setattr(adapter, "deepcopy", forbidden)
    try:
        with pytest.raises(ValueError):
            capture(graph)
        assert graph["life"]._copy_budget._charged == before
    finally:
        object.__setattr__(target, name, original)


@pytest.mark.parametrize("index", (0, 1))
def test_should_keep_original_admission_when_the_payload_copy_fails(graphs, monkeypatch, index):
    graph = graphs[index]
    before = graph["life"].payload_byte_snapshot()
    error = RuntimeError("injected graph copy failure")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(adapter, "deepcopy", fail)
    with pytest.raises(RuntimeError) as failure:
        capture(graph)
    assert failure.value is error
    assert (
        graph["life"]._copy_budget._charged == before.charged_bytes + before.observed_retained_bytes
    )


@pytest.mark.parametrize("index", (0, 1))
def test_should_recheck_original_pending_authority_after_copy_without_refunding(
    graphs, monkeypatch, index
):
    graph = graphs[index]
    pending = graph["checkpoint"]._pending[graph["pending_checkpoint"]]
    before = graph["life"].payload_byte_snapshot()
    original = adapter.copy_numpy_composite
    calls: list[str] = []

    def copying(*args):
        result = original(*args)
        calls.append("copied")
        monkeypatch.setattr(pending, "build", object())
        return result

    monkeypatch.setattr(adapter, "copy_numpy_composite", copying)
    with pytest.raises(ValueError, match="pending checkpoint"):
        capture(graph)
    assert calls == ["copied"]
    assert (
        graph["life"]._copy_budget._charged == before.charged_bytes + before.observed_retained_bytes
    )


@pytest.mark.parametrize("fault", ("alias", "nonfinite", "cpc_shape"))
def test_should_refuse_malformed_full_native_fields_before_copy(graphs, monkeypatch, fault):
    graph = graphs[1 if fault == "cpc_shape" else 0]
    model = graph["old"]._candidate._model
    if fault == "alias":
        monkeypatch.setattr(model, "bias_hidden", model.bias_hidden.copy())
    elif fault == "nonfinite":
        monkeypatch.setattr(model, "bias_output", np.full((1, 1), np.nan))
    else:
        monkeypatch.setattr(model, "_hidden_chemical", np.zeros(1))
    before = graph["life"]._copy_budget._charged
    monkeypatch.setattr(adapter, "deepcopy", forbidden)
    with pytest.raises(ValueError):
        capture(graph)
    assert graph["life"]._copy_budget._charged == before


def test_should_refuse_nonempty_replay_without_original_qualified_lineage_before_projection(
    graphs, monkeypatch
):
    graph = graphs[1]
    model = graph["new"]._candidate._model
    assert not model._replay_memory
    row = ReplaySnapshot(graph["features"], graph["targets"], 1.0, 1.0)
    monkeypatch.setattr(model, "_replay_memory", deque([row]))
    monkeypatch.setattr(adapter, "project_numpy_native", forbidden)
    monkeypatch.setattr(adapter, "deepcopy", forbidden)
    before = graph["life"]._copy_budget._charged
    with pytest.raises(ValueError, match="nonempty replay"):
        capture(graph)
    assert graph["life"]._copy_budget._charged == before


def test_should_describe_fixed_populated_controls_without_setup_or_capture(monkeypatch):
    monkeypatch.setattr(example, "_setup_case", forbidden)
    monkeypatch.setattr(example, "capture_numpy_managed_composite", forbidden)
    description = example.describe_example()
    assert description["protocol_id"] == "numpy_populated_retired_capture_engineering_v1"
    assert description["families"] == ["backprop", "circadian"]
    assert description["native_training_calls"] == 0
    assert description["energy_evaluations"] == description["inference_calls"] == 0
    assert description["capture_limits"] == asdict(LIMITS)
    assert description["expected_setup_charge"] == 184
    assert description["expected_retained_bytes"] == 152
    assert description["expected_capture_charge"] == 336
    assert "nonempty" in description["limitations"] and "remain open" in description["limitations"]


def test_should_preserve_original_resource_identity_and_unspent_work_allowances(graphs):
    for graph in graphs:
        old, new = graph["old"], graph["new"]
        assert old.actor is new.actor is graph["actor"]
        assert old._budget is new._budget is graph["budget"]
        assert old._inbox._clock is new._inbox._clock is graph["clock"]
        assert graph["shared"]._sharing is graph["gate"]
        assert graph["life"]._sampler is graph["sampler"]
        assert graph["sampler"].read_rss_bytes is graph["rss_reader"]
        assert graph["budget"].clock is graph["wall_clock"]
        assert graph["budget"].progress is graph["progress"]
        assert graph["budget"].updates_completed == graph["progress"].updates_completed == 0
        assert graph["gate"]._paused is True and graph["gate"]._admitted == 0
        assert graph["sampler"]._thread is None
        assert graph["life"]._copy_budget._limits.max_lifetime_owned_bytes == 4096


@pytest.mark.parametrize("index", (0, 1))
def test_should_refuse_final_original_copy_exhaustion_without_resetting_any_allowance(
    graphs, monkeypatch, index
):
    graph = graphs[index]
    ledger = graph["life"]._copy_budget
    # This last probe spends genuine original allowance. Nothing restores it.
    ledger.reserve(4096 - ledger._charged)
    monkeypatch.setattr(adapter, "deepcopy", forbidden)
    with pytest.raises(ValueError, match="allowance exhausted"):
        capture(graph)
    assert ledger._charged == 4096
