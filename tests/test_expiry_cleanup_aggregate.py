"""Original cleanup-proof limits before allocation and after native callbacks.

Controller model-list duplication is explicitly adversarial metadata mutation.
Every occurrence aliases an original learner: no additional model/payload work.
"""

from dataclasses import asdict, replace
import json

import pytest

import src.app.expiry_cleanup_proof as proof_module
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.expiry_inbox_observation import ExpiryObservationError
from src.core.payload_ownership import PayloadOwnershipLimits
from test_automatic_inbox_expiry import (
    AutomaticWork,
    FIRST,
    SECOND,
    _assert_authority,
    _assert_unpublished,
    _authority,
    _fresh_automatic,
    _observer_refusal,
    _queue_all_forms,
)


@pytest.fixture(scope="module")
def aggregate_work(tmp_path_factory):
    work = AutomaticWork()
    try:
        yield work
    finally:
        (tmp_path_factory.mktemp("aggregate-work") / "work.json").write_text(
            json.dumps(asdict(work), indent=2), encoding="utf8"
        )
    assert work.graphs == work.updates == 8
    assert work.learners == 24 and work.forks == 16
    assert work.arrays <= 128 and work.array_bytes <= 2048
    assert work.pending_consumed is None


def _controllers(graph, occurrences):
    def no_native(*args):
        pytest.fail("aggregate proof must not invoke checkpoint native ports")

    controllers = tuple(
        CandidateCheckpointController(
            graph.owner._shared,
            build_learner=no_native,
            state_digest=no_native,
            policy_digest=no_native,
            max_pending=1,
            max_prepare_attempts=1,
        )
        for _ in range(2)
    )
    for controller in controllers:
        controller._models.extend([graph.runtime._candidate] * occurrences)
    return controllers


def _assert_controller_cleanup(graph, report, before):
    # The original factory queues only s1. Invalidation counts pending tokens;
    # these controllers retain model aliases but have no pending checkpoints.
    assert report.reason == "expired" and report.requested_keys == (FIRST,)
    assert report.requested_keys == report.revoked_keys
    assert report.model_snapshots_erased == report.inboxes_cleared == 1
    assert report.checkpoints_invalidated == 0 and report.promotions_invalidated == 0
    assert not graph.runtime._candidate.model.rows
    assert not graph.runtime._inbox._experiences and not graph.runtime._inbox._labels
    assert not graph.life._failed and not graph.life._retention_fault
    assert not graph.runtime._stopped and not graph.ledger._poisoned
    _assert_authority(graph, before)
    _assert_unpublished(graph, before)


def _refuse_before_projection(monkeypatch, graph):
    def no_projection(*args, **kwargs):
        pytest.fail("over-budget cleanup proof allocated a graph projection or weak witness")

    with monkeypatch.context() as patch:
        for name in ("ref", "_inventory", "_holder"):
            patch.setattr(proof_module, name, no_projection)
        with pytest.raises(ValueError, match="aggregate original limits"):
            proof_module.observe_cleanup_graph(graph.life, graph.history._content_limits)


def test_should_publish_with_original_aggregate_limits_and_mixed_arrivals(aggregate_work):
    graph, ports = _fresh_automatic(aggregate_work)
    _queue_all_forms(graph)
    original = graph.history._content_limits
    graph.clock.advance_to(122)
    report = graph.life.expire()
    assert report.model_snapshots_erased == report.inboxes_cleared == 1
    assert not graph.history._records
    assert set(graph.history._erased_records) == {FIRST}
    assert set(graph.history._untrained_records) == {SECOND, ("e1", "s3"), ("e1", "s4")}
    assert graph.history._content_limits is original
    assert ports.fired == 0 and not graph.runtime._candidate.model.rows


def test_should_refuse_combined_node_excess_before_projection_and_still_clean(
    aggregate_work, monkeypatch
):
    graph, ports = _fresh_automatic(
        aggregate_work, holder_limits=PayloadOwnershipLimits(16, 512), metadata_bytes=1024 * 1024
    )
    controllers = _controllers(graph, 300)
    assert all(len(controller._models) < 512 for controller in controllers)
    _refuse_before_projection(monkeypatch, graph)
    before = _authority(graph)
    graph.clock.advance_to(122)
    with pytest.raises(ExpiryObservationError) as failure:
        graph.life.expire()
    _assert_controller_cleanup(graph, failure.value.report, before)
    assert ports.fired == 0 and all(not controller._gate.locked() for controller in controllers)


def test_should_refuse_independent_byte_excess_below_node_bound(aggregate_work, monkeypatch):
    graph, ports = _fresh_automatic(aggregate_work, metadata_bytes=32 * 1024)
    controllers = _controllers(graph, 20)
    # 42 occurrences and four holders fit4096 nodes; the original32KiB does not
    # fit their conservative saved-proof plus simultaneous-reproof byte grammar.
    assert 12 + 4 * 24 + 42 * 12 + 4 + 2 * 4 < 4096
    assert graph.history._content_limits.max_metadata_bytes == 32 * 1024
    _refuse_before_projection(monkeypatch, graph)
    before = _authority(graph)
    graph.clock.advance_to(122)
    with pytest.raises(ExpiryObservationError) as failure:
        graph.life.expire()
    _assert_controller_cleanup(graph, failure.value.report, before)
    assert ports.fired == 0 and all(not controller._gate.locked() for controller in controllers)


def test_should_recheck_aggregate_before_final_projection_after_native_probe(
    aggregate_work, monkeypatch
):
    graph, ports = _fresh_automatic(
        aggregate_work, holder_limits=PayloadOwnershipLimits(16, 512), metadata_bytes=1024 * 1024
    )
    controllers = _controllers(graph, 0)
    before = _authority(graph)
    original_inventory = proof_module._inventory
    inventory_calls: list[None] = []

    def bounded_inventory(life):
        inventory_calls.append(None)
        assert len(inventory_calls) <= 2, "final over-budget inventory allocated before preflight"
        return original_inventory(life)

    def inflate_after_native_cleanup():
        _assert_unpublished(graph, before)
        for controller in controllers:
            controller._models.extend([graph.runtime._candidate] * 300)

    monkeypatch.setattr(proof_module, "_inventory", bounded_inventory)
    ports.on_last_footprint = inflate_after_native_cleanup
    graph.clock.advance_to(122)
    with pytest.raises(ExpiryObservationError) as failure:
        graph.life.expire()
    assert ports.fired == 1 and len(inventory_calls) == 2
    _assert_controller_cleanup(graph, failure.value.report, before)
    assert all(not controller._gate.locked() for controller in controllers)


def test_should_reject_equal_value_limit_replacement_after_native_cleanup(aggregate_work):
    graph, ports = _fresh_automatic(aggregate_work)
    original = graph.history._content_limits
    before = _authority(graph)
    ports.on_last_footprint = lambda: setattr(graph.history, "_content_limits", replace(original))
    graph.clock.advance_to(122)
    _observer_refusal(
        graph, before, restore=lambda: setattr(graph.history, "_content_limits", original)
    )
    assert ports.fired == 1


def test_should_reject_changed_original_limit_values_after_native_cleanup(aggregate_work):
    graph, ports = _fresh_automatic(aggregate_work)
    original = graph.history._content_limits
    maximum = original.max_metadata_bytes
    before = _authority(graph)
    ports.on_last_footprint = lambda: object.__setattr__(
        original, "max_metadata_bytes", maximum + 1
    )
    graph.clock.advance_to(122)
    _observer_refusal(
        graph, before, restore=lambda: object.__setattr__(original, "max_metadata_bytes", maximum)
    )
    assert ports.fired == 1


def test_should_preserve_primary_native_error_after_overbudget_observer_refusal(aggregate_work):
    graph, ports = _fresh_automatic(
        aggregate_work, holder_limits=PayloadOwnershipLimits(16, 512), metadata_bytes=1024 * 1024
    )
    controllers = _controllers(graph, 300)
    before = _authority(graph)
    original = ValueError("original bounded native cleanup failure")
    ports.native_error = original
    graph.clock.advance_to(122)
    with pytest.raises(ValueError) as failure:
        graph.life.expire()
    assert failure.value is original and not isinstance(failure.value, ExpiryObservationError)
    assert graph.life._failed and graph.runtime._stopped
    _assert_unpublished(graph, before)
    assert all(not controller._gate.locked() for controller in controllers)


def test_should_bound_saved_projection_even_when_current_graph_is_small(
    aggregate_work, monkeypatch
):
    graph, _ = _fresh_automatic(
        aggregate_work, holder_limits=PayloadOwnershipLimits(16, 512), metadata_bytes=1024 * 1024
    )
    limits = graph.history._content_limits
    original = proof_module.observe_cleanup_graph(graph.life, limits)
    holder = original.holders[0]
    # Deliberately malformed retained proof, not genuine newly enrolled lineage.
    # Only existing weak references are duplicated; current live graph is small.
    oversized = holder[:3] + ((holder[3][0],) * 350,) + holder[4:]
    saved = original._replace(holders=(oversized,) + original.holders[1:])

    def no_projection(*args, **kwargs):
        pytest.fail("saved over-budget proof allocated a current projection")

    with monkeypatch.context() as patch:
        patch.setattr(proof_module, "_inventory", no_projection)
        with pytest.raises(ValueError, match="aggregate original limits"):
            proof_module.require_cleanup_graph(saved, graph.life, limits, final=True, groups=())

        class Foreign:
            def __eq__(self, other):
                pytest.fail("foreign saved proof comparison callback executed")

        foreign = holder[:3] + ((holder[3][0][:1] + (Foreign(),) + holder[3][0][2:],),) + holder[4:]
        saved = original._replace(holders=(foreign,) + original.holders[1:])
        with pytest.raises(ValueError, match="closed exact fields"):
            proof_module.require_cleanup_graph(saved, graph.life, limits, final=True, groups=())
