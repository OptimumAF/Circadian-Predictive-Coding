"""Complete capture of genuinely restored legacy replay and successive updates.

Setup reuses the fixed issued-snapshot example and performs one managed restore.
Capture/report uses borrowed projections and existing data codecs only. Original
weak publication witnesses establish lineage; component digests only deduplicate
report DATA. No new proof, native algorithm, restore authority, IO or recovery.
"""

from dataclasses import asdict
from hashlib import sha256
import json
import os
from typing import Any
from weakref import ref

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

from src.adapters.numpy_capture_example import (
    BINDING,
    CAPTURE_LIMITS,
    NATIVE_LIMITS,
    _plain,
    _policy,
    _require,
    _same_budget,
)
from src.adapters.numpy_checkpoint_frames import canonical_json
from src.adapters.numpy_composite_capture import capture_numpy_managed_composite
from src.adapters.numpy_inbox_checkpoint_codec import NumpyInboxCheckpointCodec
from src.adapters.numpy_snapshot_capture_example import (
    _GRAPH_KEYS as _SNAPSHOT_KEYS,
    _array_detachment,
    _captured_cursor,
    _holder,
    _inbox_policy,
    _native,
    _setup_snapshot_case,
    _view_metadata,
    describe_snapshot_example,
)
from src.app.candidate_checkpoint import _budget_state
from src.core.data_lifecycle import DataConsent, DataProvenance, LifecycleDeclaration
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.replay_origin import ReplayOriginLimits

_GRAPH_KEYS = _SNAPSHOT_KEYS | {"old", "new", "before_publication", "copy_before_publication"}
_SUCCESSIVE_KEYS = _GRAPH_KEYS | {"continuation"}
_CONTINUATION_KEYS = frozenset(
    (
        "observed_paused_poll",
        "initial_row_expired",
        "second_row_expired",
        "previous_success_expired",
        "zero_work_poll_metadata_bytes",
        "zero_work_poll_metadata_charge",
    )
)


def _setup_restored_case(
    *,
    max_updates: int = 2,
    origin_limits: ReplayOriginLimits | None = None,
    retain_inbox_origins: bool = False,
) -> dict[str, Any]:
    """Sole native boundary: one original issue/update, then actual publication."""
    graph = _setup_snapshot_case(
        retain_capture_history=True,
        retain_inbox_origins=retain_inbox_origins,
        max_updates=max_updates,
        origin_limits=origin_limits,
    )
    graph["before_publication"] = graph["ledger"].accounting()
    graph["copy_before_publication"] = graph["life"].payload_byte_snapshot()
    graph["old"] = graph["runtime"]
    graph["new"] = graph["manager"].restore(graph["controller"], graph["token"])
    current, old = graph["new"], graph["old"]
    _require(
        graph["shared"]._runtime is current
        and old._retired
        and old._inbox._historical_completed_updates == current._budget.updates_completed == 1
        and not graph["controller"]._pending
        and len(graph["controller"]._models) == 1
        and graph["controller"]._models[0] is current._candidate,
        "restored setup did not publish its actual original current/retired pair",
    )
    charge = graph["life"].payload_byte_snapshot()
    _require(
        charge.charged_bytes == 608 and charge.observed_retained_bytes == 128,
        "original restored setup accounting changed",
    )
    return graph


def _successive_origin_limits() -> ReplayOriginLimits:
    return ReplayOriginLimits(32, 64, 32, 262144, 120, 4096)


def _record_next_pair(graph, number, features, targets, observed_at, arrived_at):
    owner = graph["owner"]
    sample = f"s{number}"
    owner.declare(
        LifecycleDeclaration(
            ("e1", sample),
            DataProvenance("local", f"person-{number}", True, False),
            DataConsent(True, True),
            "replay",
        )
    )
    owner.record_experience(
        Experience(
            sample,
            "e1",
            observed_at,
            "actor-0",
            np.array(features, dtype=np.float64),
            "train",
            ExperiencePermissions(True, True),
        )
    )
    owner.record_label(
        LabelArrival(
            f"label-{sample}",
            sample,
            "e1",
            arrived_at,
            "actor-0",
            np.array(targets, dtype=np.float64),
        )
    )
    graph["clock"].advance_to(arrived_at)


def _train_next_pair(graph, update):
    graph["gate"].resume()
    try:
        poll = graph["ledger"].train_ready()
    finally:
        graph["gate"].pause()
    _require(
        len(poll.updates) == 1
        and poll.updates[0].update_number == update
        and graph["budget"].updates_completed == update,
        "successive setup did not commit its one original managed update",
    )


def _metadata_charge(record) -> tuple[int, int]:
    size = len(
        json.dumps(
            record.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )
    _require(record.charged_bytes == size + 1024, "original paid metadata charge changed")
    return record.charged_bytes, size


def _paused_poll(graph, previous_success) -> tuple[int, int]:
    ledger, manager = graph["ledger"], graph["manager"]
    before, raw_before = ledger.accounting(), graph["life"].payload_byte_snapshot()
    poll = ledger.train_ready()  # Original gate is paused, with pair three ready.
    observation = manager._zero_work_poll
    _require(observation is not None, "paused poll lacks its original paid observation")
    charge, size = _metadata_charge(observation)
    after, raw_after = ledger.accounting(), graph["life"].payload_byte_snapshot()
    _require(
        not poll.updates
        and poll.deferred_reason == "paused"
        and manager._current_success is previous_success()
        and manager._current_success.completed_updates == graph["budget"].updates_completed == 2
        and observation is manager._zero_work_poll_sealed
        and manager._zero_work_poll_storage == (observation, observation)
        and observation.previous_success() is previous_success()
        and observation.deferred_reason == "paused"
        and after.invocations_started == before.invocations_started
        and after.records_created == before.records_created
        and after.metadata_bytes_charged == before.metadata_bytes_charged + charge
        and raw_before == raw_after,
        "paused poll changed learned success/work or renewed original accounting",
    )
    return charge, size


def _setup_successive_case() -> dict[str, Any]:
    """Only native boundary: fixed restore, two further wakes and one real defer."""
    graph = _setup_restored_case(
        max_updates=3,
        origin_limits=_successive_origin_limits(),
        retain_inbox_origins=True,
    )
    identities = _identity(graph)
    native_before, inbox_before, consumed_before = _components(graph)
    model = graph["new"]._candidate._model
    initial_row = ref(model._replay_memory[0])
    binding_identity = id(graph["manager"]._update_binding)
    _record_next_pair(graph, 2, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[1.0], [0.0]], 4, 6)
    _train_next_pair(graph, 2)
    retained = model._replay_memory
    _require(
        len(retained) == 2 and initial_row() is None, "first current native row did not expire"
    )
    # The fixed default content-hash retention policy evicts index one for the
    # declared inputs. Do not keep its old deque alive across the next wake.
    second_row = ref(retained[1])
    del retained
    previous_success = ref(graph["manager"]._current_success)
    _record_next_pair(graph, 3, [[1.0, 1.0, 0.0]], [[1.0]], 8, 10)
    zero_charge, zero_bytes = _paused_poll(graph, previous_success)
    _train_next_pair(graph, 3)
    native_after, inbox_after, consumed_after = _components(graph)
    _require(
        second_row() is None
        and previous_success() is None
        and len(model._replay_memory) == 2
        and model._epoch_count == 3
        and graph["old"]._candidate._model._epoch_count == 1
        and identities == _identity(graph)
        and id(graph["manager"]._update_binding) == binding_identity
        and graph["manager"]._zero_work_poll is None
        and native_before["actor"] == native_after["actor"]
        and native_before["retired"] == native_after["retired"]
        and inbox_before["retired"] == inbox_after["retired"]
        and consumed_before == consumed_after,
        "successive setup changed immutable state/authority or retained superseded evidence",
    )
    charge = graph["life"].payload_byte_snapshot()
    _require(
        charge.charged_bytes == 800 and charge.observed_retained_bytes == 256,
        "successive setup changed its original cumulative raw accounting",
    )
    graph["continuation"] = dict(
        observed_paused_poll=True,
        initial_row_expired=True,
        second_row_expired=True,
        previous_success_expired=True,
        zero_work_poll_metadata_bytes=zero_bytes,
        zero_work_poll_metadata_charge=zero_charge,
    )
    return graph


def _identity(graph: dict[str, Any]) -> dict[str, int]:
    current, old = graph["new"], graph["old"]
    result = {
        name: id(graph[name])
        for name in (
            "owner",
            "life",
            "old",
            "new",
            "ledger",
            "controller",
            "manager",
            "token",
            "pending",
            "view",
            "actor",
            "budget",
            "clock",
            "shared",
            "gate",
            "wall_clock",
            "builder",
        )
    }
    result.update(
        copies=id(graph["manager"]._copies),
        current_learner=id(current._candidate),
        retired_learner=id(old._candidate),
        current_inbox=id(current._inbox),
        retired_inbox=id(old._inbox),
        copy_budget=id(graph["life"]._copy_budget),
    )
    return result


def _components(graph: dict[str, Any]):
    current, old = graph["new"], graph["old"]
    native = dict(
        actor=_native(current.actor._learner),
        retired=_native(old._candidate),
        current=_native(current._candidate),
    )
    native_frames = {
        name: graph["codec"].encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in native.items()
    }
    codec = NumpyInboxCheckpointCodec(_inbox_policy())
    inbox_frames = {
        name: codec.encode(runtime._inbox._borrow_cursor(), binding=BINDING, limits=NATIVE_LIMITS)
        for name, runtime in (("retired", old), ("current", current))
    }
    consumed = dict(
        native=graph["codec"].encode(graph["view"].state, binding=BINDING, limits=NATIVE_LIMITS),
        inbox=codec.encode(graph["view"].inbox, binding=BINDING, limits=NATIVE_LIMITS),
        metadata=_view_metadata(graph["view"]),
    )
    return native_frames, inbox_frames, consumed


def _detached_components(graph: dict[str, Any], captured: Any):
    actor = _holder(graph, captured, graph["actor"])
    retired = _holder(graph, captured, graph["old"])
    current = _holder(graph, captured, graph["new"])
    checkpoint = _holder(graph, captured, graph["controller"])
    _require(
        retired["_retired"] is True
        and current["_retired"] is False
        and len(checkpoint["_models"]) == 1
        and not checkpoint["_pending"]
        and checkpoint["_models"][0] is current["_candidate"],
        "complete capture lost original retirement or checkpoint/current learner alias",
    )
    native = dict(
        actor=dict(actor["_learner"].fields)["_model"],
        retired=dict(retired["_candidate"].fields)["_model"],
        current=dict(current["_candidate"].fields)["_model"],
    )
    inbox = {
        name: dict(record["_inbox"].fields)
        for name, record in (("retired", retired), ("current", current))
    }
    _require(
        inbox["retired"]["_historical_completed_updates"] == 1
        and inbox["current"]["_historical_completed_updates"] is None,
        "complete capture changed original historical/current work observations",
    )
    native_frames = {
        name: graph["codec"].encode(value, binding=BINDING, limits=NATIVE_LIMITS)
        for name, value in native.items()
    }
    codec = NumpyInboxCheckpointCodec(_inbox_policy())
    inbox_frames = {
        name: codec.encode(_detached_cursor(captured, data), binding=BINDING, limits=NATIVE_LIMITS)
        for name, data in inbox.items()
    }
    return native, inbox, native_frames, inbox_frames


def _detached_cursor(captured, data):
    # Retired inbox DATA observes its own completed history, independently of
    # the still-running candidate's cumulative original budget.
    return _captured_cursor(captured, data, completed_updates=data["_historical_completed_updates"])


def _joint_detachment(graph: dict[str, Any], native, inbox) -> int:
    memo: dict[int, Any] = {}
    arrays = 0
    for name, learner in (
        ("actor", graph["new"].actor._learner),
        ("retired", graph["old"]._candidate),
        ("current", graph["new"]._candidate),
    ):
        arrays += _array_detachment(vars(learner._model), native[name].state, memo)
        _require(
            native[name].config is native[name].state["config"],
            "complete captured native config alias changed",
        )
    for name, runtime in (("retired", graph["old"]), ("current", graph["new"])):
        for field in ("_experiences", "_labels", "_applied", "_erased"):
            arrays += _array_detachment(getattr(runtime._inbox, field), inbox[name][field], memo)
    return arrays


def _paid_publication(graph: dict[str, Any]) -> tuple[int, int]:
    manager = graph["manager"]
    key = id(graph["old"]._candidate._model)
    record = manager._publications[key]
    size = len(
        json.dumps(
            record.metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf8")
    )
    _require(
        manager._retain_capture_history is True
        and manager._publication_sealed[key] is record
        and manager._publication_storage[0] is manager._publications
        and manager._publication_storage[1] is manager._publication_sealed
        and record.charged_bytes == size + 1024
        and graph["ledger"].accounting().metadata_bytes_charged
        >= graph["before_publication"].metadata_bytes_charged + record.charged_bytes,
        "restored publication lacks its exact original paid metadata record",
    )
    return record.charged_bytes, size


def _paid_current_success(graph) -> tuple[int, int]:
    manager = graph["manager"]
    record = manager._current_success
    charge, size = _metadata_charge(record)
    _require(
        record is manager._current_success_sealed
        and manager._current_success_storage == (record, record)
        and record.completed_updates == 3
        and len(record.rows) == 2
        and len(record.history_records) == 3
        and manager._zero_work_poll is None
        and graph["ledger"].accounting().metadata_bytes_charged >= charge,
        "successive capture lacks its exact original paid current success",
    )
    return charge, size


def _capture_restored_case(graph: dict[str, Any], *, successive: bool = False) -> dict[str, Any]:
    """One visible before/admit/after transaction; component helpers own data reads."""
    if type(successive) is not bool:
        raise ValueError("successive replay capture requires an exact boolean")
    expected_keys = _SUCCESSIVE_KEYS if successive else _GRAPH_KEYS
    if type(graph) is not dict or graph.keys() != expected_keys:
        raise ValueError("restored replay report requires the complete original setup graph")
    if successive:
        continuation = graph["continuation"]
        if (
            type(continuation) is not dict
            or continuation.keys() != _CONTINUATION_KEYS
            or any(
                continuation[key] is not True
                for key in (
                    "observed_paused_poll",
                    "initial_row_expired",
                    "second_row_expired",
                    "previous_success_expired",
                )
            )
            or type(continuation["zero_work_poll_metadata_bytes"]) is not int
            or type(continuation["zero_work_poll_metadata_charge"]) is not int
            or continuation["zero_work_poll_metadata_bytes"] <= 0
            or continuation["zero_work_poll_metadata_charge"]
            != continuation["zero_work_poll_metadata_bytes"] + 1024
        ):
            raise ValueError("successive replay report requires original continuation observations")
    setup_charge, after_charge, retained = (800, 1056, 256) if successive else (608, 736, 128)
    updates = 3 if successive else 1
    native_before, inbox_before, consumed_before = _components(graph)
    identities = _identity(graph)
    budget_before = _plain(_budget_state(graph["budget"]))
    copy_before = graph["life"].payload_byte_snapshot()
    origins_before = graph["ledger"].accounting()
    publication_charge, publication_bytes = _paid_publication(graph)
    binding = graph["manager"]._update_binding if successive else None
    current_payment = _paid_current_success(graph) if successive else None
    try:
        capture_numpy_managed_composite(
            graph["owner"], limits=CAPTURE_LIMITS, replay_origins=graph["ledger"]
        )
    except ValueError as error:
        refusal = str(error)
    else:
        raise AssertionError(
            "restored nonempty replay captured without original publication witnesses"
        )
    _require(
        graph["life"]._copy_budget._charged == copy_before.charged_bytes,
        "default restored capture refusal charged a raw copy",
    )
    captured = capture_numpy_managed_composite(
        graph["owner"],
        limits=CAPTURE_LIMITS,
        replay_origins=graph["ledger"],
        replay_checkpoints=graph["manager"],
    )
    native, inbox, native_captured, inbox_captured = _detached_components(graph, captured)
    _require(
        native_captured == native_before and inbox_captured == inbox_before,
        "restored captured complete component bytes changed",
    )
    arrays = _joint_detachment(graph, native, inbox)
    native_after, inbox_after, consumed_after = _components(graph)
    copy_after = graph["life"].payload_byte_snapshot()
    origins_after = graph["ledger"].accounting()
    budget_after = _plain(_budget_state(graph["budget"]))
    _require(
        (native_before, inbox_before, consumed_before)
        == (native_after, inbox_after, consumed_after)
        and identities == _identity(graph),
        "restored capture changed original full state or authority",
    )
    _require(
        copy_before.charged_bytes == setup_charge
        and copy_after.charged_bytes == after_charge
        and copy_before.observed_retained_bytes
        == copy_after.observed_retained_bytes
        == captured.state.retained_payload_bytes
        == retained,
        "restored capture did not preserve cumulative original raw charge",
    )
    _require(
        origins_after.invocations_started == origins_before.invocations_started + 2
        and origins_after.metadata_bytes_charged == origins_before.metadata_bytes_charged + 2048
        and origins_after.records_created == origins_before.records_created,
        "restored capture changed original admission accounting",
    )
    _require(
        _same_budget(_plain(budget_before), _plain(budget_after)),
        "restored capture changed original native budget",
    )
    authority = {item.path: item.value for item in captured.authority}
    for path, key in (
        ("root.owner", "owner"),
        ("root.lifecycle", "life"),
        ("runtime.root", "new"),
        ("runtime.actor", "actor"),
        ("runtime.budget", "budget"),
        ("inbox.clock", "clock"),
    ):
        _require(authority[path] is graph[key], "restored capture replaced original live authority")
    current = graph["new"]
    _require(
        graph["shared"]._runtime is current
        and graph["controller"]._models[0] is current._candidate
        and not graph["controller"]._pending
        and graph["controller"]._attempts == 1
        and graph["old"]._retired
        and graph["old"]._inbox._historical_completed_updates == 1
        and current._budget.updates_completed == updates,
        "restored capture changed publication ownership/history",
    )
    _require(
        _paid_publication(graph) == (publication_charge, publication_bytes),
        "original paid publication record changed during capture",
    )
    payloads = {
        sha256(raw).hexdigest(): raw.decode("utf8")
        for raw in (
            *native_before.values(),
            *inbox_before.values(),
            consumed_before["native"],
            consumed_before["inbox"],
        )
    }
    digest = lambda values: {name: sha256(raw).hexdigest() for name, raw in values.items()}
    result = dict(
        family="circadian",
        policy=_policy(current._candidate, "circadian"),
        native_before=digest(native_before),
        native_after=digest(native_after),
        native_captured=digest(native_captured),
        inbox_before=digest(inbox_before),
        inbox_after=digest(inbox_after),
        inbox_captured=digest(inbox_captured),
        component_payloads=payloads,
        consumed_view_before=dict(
            native=sha256(consumed_before["native"]).hexdigest(),
            inbox=sha256(consumed_before["inbox"]).hexdigest(),
            metadata=consumed_before["metadata"],
        ),
        consumed_view_after=dict(
            native=sha256(consumed_after["native"]).hexdigest(),
            inbox=sha256(consumed_after["inbox"]).hexdigest(),
            metadata=consumed_after["metadata"],
        ),
        copy_before=asdict(copy_before),
        copy_after=asdict(copy_after),
        copy_before_publication=asdict(graph["copy_before_publication"]),
        origins_before=asdict(origins_before),
        origins_after=asdict(origins_after),
        origins_before_publication=asdict(graph["before_publication"]),
        budget_before=budget_before,
        budget_after=budget_after,
        identity=dict(original=identities, current=_identity(graph)),
        publication_metadata_charge=publication_charge,
        publication_metadata_bytes=publication_bytes,
        retired_historical_updates=1,
        current_updates=updates,
        preparation_attempts=1,
        pending_checkpoints=0,
        default_off_refusal=refusal,
        capture_calls=2,
        graph_copy_passes=2,
        detached_array_count=arrays,
        actual_cp_current_alias=True,
        joint_aliases_preserved=True,
        native_unchanged=True,
        original_authority_preserved=True,
        original_consumed_view_unchanged=True,
        data_only=True,
    )
    if successive:
        _require(
            graph["manager"]._update_binding is binding
            and graph["ledger"]._capture_update_binding
            is graph["ledger"]._original_capture_update_binding
            is binding
            and current_payment == _paid_current_success(graph),
            "successive capture changed its original update binding/current paid evidence",
        )
        assert current_payment is not None
        result.update(
            continuation=dict(graph["continuation"]),
            current_success_metadata_charge=current_payment[0],
            current_success_metadata_bytes=current_payment[1],
            current_update_binding_preserved=True,
        )
    _require(
        len(canonical_json(result)) <= 131072,
        "successive replay report exceeds bounded output"
        if successive
        else "restored replay report exceeds bounded output",
    )
    return result


def describe_restored_example() -> dict[str, Any]:
    result = describe_snapshot_example()
    result.update(
        protocol_id="numpy_restored_replay_capture_engineering_v1",
        retain_capture_history=True,
        native_restore_calls=1,
        checkpoint_issues=1,
        checkpoint_restores=1,
        handoffs=1,
        expected_setup_charge=608,
        expected_retained_bytes=128,
        expected_capture_charge=736,
        publication_metadata_charge_rule="canonical UTF-8 JSON(metadata) bytes + 1024",
        limitations="Original genuinely restored current/retired complete capture DATA; no portable authority, later work, chained handoff, other-role lineage, erasure, complete managed bytes, durable/live recovery or scientific benefit.",
    )
    return result


def run_restored_example() -> dict[str, Any]:
    return dict(
        protocol=describe_restored_example(),
        case=_capture_restored_case(_setup_restored_case()),
        restored_replay_capture_verified=True,
        full_managed_process_recovery_verified=False,
    )


def describe_successive_example() -> dict[str, Any]:
    result = describe_restored_example()
    result.update(
        protocol_id="numpy_successive_replay_capture_engineering_v1",
        retain_inbox_origins=True,
        origin_limits=asdict(_successive_origin_limits()),
        budget={**result["budget"], "max_training_updates": 3},
        native_training_calls=3,
        paid_paused_polls=1,
        expected_setup_charge=800,
        expected_retained_bytes=256,
        expected_capture_charge=1056,
        update_sequence=[
            dict(features=[[1.0, 0.0, 0.0]], targets=[[0.0]], observed_at=1, arrived_at=3),
            dict(
                features=[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                targets=[[1.0], [0.0]],
                observed_at=4,
                arrived_at=6,
            ),
            dict(features=[[1.0, 1.0, 0.0]], targets=[[1.0]], observed_at=8, arrived_at=10),
        ],
        current_success_metadata_charge_rule="canonical UTF-8 JSON(metadata) bytes + 1024",
        zero_work_poll_metadata_charge_rule="canonical UTF-8 JSON(metadata) bytes + 1024",
        limitations="Original restored retired/current capture DATA after successive original single-update polls and a paid paused deferral; no chained restoration, other-role lineage, erasure, portable authority, complete managed bytes, durable/live recovery or scientific benefit.",
    )
    return result


def run_successive_example() -> dict[str, Any]:
    return dict(
        protocol=describe_successive_example(),
        case=_capture_restored_case(_setup_successive_case(), successive=True),
        successive_replay_capture_verified=True,
        full_managed_process_recovery_verified=False,
    )
