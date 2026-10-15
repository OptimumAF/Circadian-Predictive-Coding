"""Bounded raw graph sizing and original copy allowances across native-port handoff."""

from dataclasses import replace
from types import SimpleNamespace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as basic_ports
import src.adapters.coupled_payload_copies as copies
import src.adapters.coupled_managed_cleanup as cleanup_demo
from src.adapters.coupled_candidate_handoff import BINDING, LIMITS
from src.adapters.coupled_checkpoint_callbacks import CoupledCheckpointCallbacks
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.candidate_checkpoint import CandidateCheckpointController, CandidateCheckpointView
from src.app.managed_experience import ManagedExperienceOwner
from src.core.coupled_learner_state import CoupledUpdatePolicy
from src.core.data_retention import DataRetentionPolicy
from src.core.experience import Experience, ExperiencePermissions, LabelArrival
from src.core.inbox_cursor import InboxCursor
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from src.core.resource_sharing import SharingLimits, SharingSnapshot
from src.core.serving_ports import CachedPrediction
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


def callbacks():
    return CoupledCheckpointCallbacks(
        CoupledUpdatePolicy(0.025, 3, 0.07), binding=BINDING, limits=LIMITS
    )


def synthetic_learner():
    learner: Any = object.__new__(CoupledLearner)
    learner._model = SimpleNamespace(input_dim=2, hidden_dims=(3, 2))
    return learner


def view():
    x, y = np.ones((2, 2)), np.ones((2, 1))
    inbox = InboxCursor(
        1,
        "c0",
        1,
        (Experience("s0", "e0", 0, "a0", x, "train", ExperiencePermissions(True, True)),),
        (LabelArrival("l0", "s0", "e0", 1, "a0", y),),
        (),
        0,
        False,
        0,
    )
    return CandidateCheckpointView(
        "a0",
        0,
        "c0",
        snapshot(),
        inbox,
        (),
        (),
        0,
        False,
        {},
        SharingSnapshot(SharingLimits(2, 3, 1), 0, False, True, 0, ()),
        0,
    )


def policy(**kwargs):
    return DataRetentionPolicy(
        192,
        100,
        PayloadOwnershipLimits(4, 4),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(576),
        **kwargs,
    )


def test_should_count_arrays_once_per_graph_and_keep_metadata_or_aliases_unmodified(monkeypatch):
    array = np.arange(4, dtype=np.float64).reshape(2, 2)
    shared = [array, None, True, "metadata", 7, 0.5]
    graph = {"first": shared, "again": shared, "cache": CachedPrediction(3, array)}
    raw = array.tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("sizing scanned or converted numeric data")

    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    assert copies.measure_coupled_auxiliary_bytes(graph) == 32
    assert copies.measure_coupled_auxiliary_bytes(graph) == 32
    assert graph["first"] is graph["again"] and array.tobytes() == raw


@pytest.mark.parametrize(
    "kind",
    [
        "list_cycle",
        "dict_cycle",
        "mutual_cycle",
        "opaque",
        "dict_key",
        "subclass",
        "object_array",
        "complex_array",
        "bool_array",
        "datetime_array",
        "timedelta_array",
    ],
)
def test_should_refuse_unsupported_or_cyclic_graph_without_opaque_hooks(kind):
    class Opaque:
        def __deepcopy__(self, memo):
            raise AssertionError("opaque graph was copied")

        def __array__(self, *args, **kwargs):
            raise AssertionError("opaque graph was converted")

    class DerivedList(list):
        def __iter__(self):
            raise AssertionError("container subclass hook was used")

    def object_array():
        value = np.empty(1, dtype=object)
        value[0] = Opaque()
        return value

    if kind == "list_cycle":
        graph: Any = []
        graph.append(graph)
    elif kind == "dict_cycle":
        graph = {}
        graph["self"] = graph
    elif kind == "mutual_cycle":
        graph = []
        other = {"parent": graph}
        graph.append(other)
    else:
        graph = {
            "opaque": lambda: Opaque(),
            "dict_key": lambda: {1: np.zeros(1)},
            "subclass": lambda: DerivedList([1]),
            "object_array": object_array,
            "complex_array": lambda: np.array([1j]),
            "bool_array": lambda: np.array([True]),
            "datetime_array": lambda: np.array(["2026-10-15"], dtype="datetime64[D]"),
            "timedelta_array": lambda: np.array([1], dtype="timedelta64[D]"),
        }[kind]()
    with pytest.raises(ValueError):
        copies.measure_coupled_auxiliary_bytes(graph)


@pytest.mark.parametrize("depth", [31, 32, 33])
def test_should_bound_graph_depth_inclusively(depth):
    graph: Any = np.zeros(1)
    for _ in range(depth):
        graph = [graph]
    if depth <= 32:
        assert copies.measure_coupled_auxiliary_bytes(graph) == 8
    else:
        with pytest.raises(ValueError):
            copies.measure_coupled_auxiliary_bytes(graph)


@pytest.mark.parametrize("nodes", [4096, 4097])
def test_should_bound_distinct_graph_nodes_before_unbounded_walk(nodes):
    graph: list[dict[str, Any]] = [{} for _ in range(nodes - 1)]
    if nodes == 4096:
        assert copies.measure_coupled_auxiliary_bytes(graph) == 0
    else:
        with pytest.raises(ValueError):
            copies.measure_coupled_auxiliary_bytes(graph)


@pytest.mark.parametrize("family", ["pc", "neutral_cpc", "active_cpc"])
def test_should_exclude_learned_parameters_and_chemistry_from_preparation_bytes(
    family, monkeypatch
):
    builder, state = copies.ManagedCoupledBuilder(callbacks()), snapshot(family)

    def forbidden(*args, **kwargs):
        raise AssertionError("preparation sizing copied or scanned learned arrays")

    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    assert copies.coupled_preparation_bytes(builder, state) == 0
    assert builder.build_calls == 0


@pytest.mark.parametrize("bad", ["state", "parameters", "weights", "bias", "wake", "extra"])
def test_should_refuse_unknown_nested_native_structure_before_copy_or_builder(bad):
    builder = copies.ManagedCoupledBuilder(callbacks())
    state: Any = snapshot("active_cpc")

    class Poison:
        def __deepcopy__(self, memo):
            raise AssertionError("unknown native state was copied")

    if bad == "state":
        state = Poison()
    elif bad == "parameters":
        object.__setattr__(state, "parameters", Poison())
    elif bad == "weights":
        object.__setattr__(state.parameters, "hidden_weights", (Poison(),))
    elif bad == "bias":
        object.__setattr__(state.parameters, "output_bias", np.zeros((2, 1)))
    elif bad == "wake":
        object.__setattr__(state, "wake_state", Poison())
    else:
        object.__setattr__(state, "extra", Poison())
    with pytest.raises(ValueError):
        copies.coupled_preparation_bytes(builder, state)
    assert builder.build_calls == 0


def test_should_measure_checkpoint_raw_arrays_and_validate_metadata_without_copying_state():
    captured = view()
    assert copies.measure_coupled_checkpoint_bytes(captured) == 48
    assert captured.state.parameters.hidden_weights[0].nbytes > 0
    object.__setattr__(captured.inbox, "completed_updates", True)
    with pytest.raises(ValueError):
        copies.measure_coupled_checkpoint_bytes(captured)


@pytest.mark.parametrize("mode", ["success", "failure"])
def test_should_count_only_actual_builder_calls_and_forward_the_exact_state(monkeypatch, mode):
    bound, state, sentinel = callbacks(), snapshot(), object()
    calls: list[Any] = []

    def build(self, actual):
        calls.append((self, actual))
        if mode == "failure":
            raise RuntimeError("injected builder failure")
        return sentinel

    monkeypatch.setattr(CoupledCheckpointCallbacks, "build", build)
    builder = copies.ManagedCoupledBuilder(bound)
    assert builder.build_calls == 0 and calls == []
    with pytest.raises(AttributeError):
        setattr(builder, "build_calls", 99)
    assert copies.coupled_preparation_bytes(builder, state) == 0
    if mode == "success":
        assert builder(state) is sentinel
    else:
        with pytest.raises(RuntimeError, match="injected builder"):
            builder(state)
    assert builder.build_calls == 1 and len(calls) == 1 and calls[0][1] is state


@pytest.mark.parametrize("bad", [None, False, object()])
def test_should_refuse_opaque_builder_callbacks_without_calls(bad):
    with pytest.raises(ValueError):
        copies.ManagedCoupledBuilder(bad)


@pytest.mark.parametrize("kind", ["closure", "subclass", "opaque"])
def test_should_refuse_uninspectable_preparation_builder_before_any_call(kind):
    class Derived(copies.ManagedCoupledBuilder):
        pass

    class Opaque:
        def __call__(self, state):
            raise AssertionError("opaque builder was invoked while sizing")

    def closure(state):
        raise AssertionError("closure builder was invoked while sizing")

    selected = {"closure": closure, "subclass": Derived(callbacks()), "opaque": Opaque()}[kind]
    with pytest.raises(ValueError):
        copies.coupled_preparation_bytes(selected, snapshot())


def test_should_measure_growth_and_output_cache_from_metadata_without_model_methods(monkeypatch):
    learner = synthetic_learner()

    def forbidden(*args, **kwargs):
        raise AssertionError("measurement invoked a model operation")

    for name in ("snapshot_state", "predict", "train_batch", "restore_state", "fork"):
        monkeypatch.setattr(CoupledLearner, name, forbidden)
    x, y = np.ones((3, 2)), np.zeros((3, 1))
    assert copies.coupled_native_growth_bytes(learner, x, y) == 0
    assert copies.coupled_prediction_cache_bytes(learner, x) == 24


@pytest.mark.parametrize("kind", ["integer", "nonfinite", "non_task_shapes"])
def test_should_measure_real_typed_storage_without_imposing_native_task_validation(kind):
    learner = synthetic_learner()
    x: Any
    y: Any
    if kind == "integer":
        x, y = np.ones((3, 2), dtype=np.int16), np.zeros((3, 1), dtype=np.uint8)
    elif kind == "nonfinite":
        x, y = np.full((3, 2), np.nan), np.full((3, 1), np.inf)
    else:
        x, y = np.ones(7), np.zeros((2, 4, 1))
    assert copies.coupled_native_growth_bytes(learner, x, y) == 0
    if kind != "non_task_shapes":
        assert copies.coupled_prediction_cache_bytes(learner, x) == 24


@pytest.mark.parametrize("bad", ["learner", "width", "rank", "targets", "dtype"])
def test_should_refuse_invalid_model_or_batch_metadata_without_model_calls(bad):
    learner: Any = synthetic_learner()
    x: Any = np.ones((3, 2))
    y: Any = np.zeros((3, 1))
    if bad == "learner":
        learner = object()
    elif bad == "width":
        x = np.ones((3, 3))
    elif bad == "rank":
        x = np.ones(2)
    elif bad == "targets":
        y = np.zeros((3, 1), dtype=object)
    else:
        x = np.ones((3, 2), dtype=object)
    with pytest.raises(ValueError):
        if bad in {"width", "rank"}:
            copies.coupled_prediction_cache_bytes(learner, x)
        else:
            copies.coupled_native_growth_bytes(learner, x, y)


@pytest.mark.parametrize("bad", ["missing_copy_limit", "elapsed", "policy_type"])
def test_should_refuse_unsupported_copy_policy_before_owner_operations(monkeypatch, bad):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported policy reached lifecycle construction")

    monkeypatch.setattr(copies, "ManagedDataLifecycle", forbidden)
    selected: Any = policy()
    if bad == "missing_copy_limit":
        selected = replace(selected, owned_payload_copies=None)
    elif bad == "elapsed":
        selected = replace(selected, max_retention_seconds=1.0)
    else:
        selected = None
    with pytest.raises(ValueError):
        copies.make_coupled_copy_lifecycle(object.__new__(ManagedExperienceOwner), policy=selected)


def test_should_borrow_raw_cursor_references_while_public_capture_stays_detached():
    from test_resource_sharing import shared
    from test_candidate_checkpoint import queue

    _, runtime, *_ = shared()
    queue(runtime)
    with runtime._exclusive():
        borrowed = runtime._inbox._borrow_cursor()
        source = runtime._inbox._experiences[borrowed.experiences[0].key]
        target = runtime._inbox._labels[borrowed.labels[0].key]
        assert borrowed.experiences[0] is source and borrowed.labels[0] is target
    public = runtime._inbox.capture_cursor()
    public.experiences[0].features.append("outside")
    public.labels[0].targets.append("outside")
    assert "outside" not in source.features and "outside" not in target.targets
    assert borrowed.experiences[0].features is source.features


def test_should_borrow_valid_metadata_without_calling_an_opaque_payload_copy_hook():
    from test_resource_sharing import shared

    class Poison:
        copies = 0

        def __deepcopy__(self, memo):
            self.copies += 1
            raise AssertionError("borrowed cursor copied raw payload")

    _, runtime, *_ = shared()
    payload = Poison()
    source = Experience("s0", "e0", 0, "actor-0", payload, "train", ExperiencePermissions(True))
    target = LabelArrival("l0", "s0", "e0", 1, "actor-0", payload)
    with runtime._exclusive():
        runtime._inbox._experiences[source.key] = source
        runtime._inbox._labels[target.key] = target
        runtime._inbox._event_ids.add(target.event_id)
        borrowed = runtime._inbox._borrow_cursor()
    assert borrowed.experiences == (source,) and borrowed.labels == (target,)
    assert borrowed.experiences[0] is source and borrowed.labels[0] is target
    assert payload.copies == 0


def test_should_refuse_mutated_current_raw_history_before_restoring_borrowed_cursor():
    from test_candidate_checkpoint import prepared

    control, wrapper, old, *_ = prepared()
    token = control.capture()
    before = old._candidate.snapshot_state()
    next(iter(old._inbox._labels.values())).targets.append("changed outside registration")
    with pytest.raises(ValueError, match="stale|changed"):
        control.restore(token)
    assert wrapper._runtime is old and not old._retired
    assert old._candidate.snapshot_state() == before and control._attempts == 0


_ROUTES = {
    None: "example",
    "--repeat-handoffs": "repeated_example",
    "--resource-stops": "resource_stop_example",
    "--endurance": "endurance_example",
    "--managed-consent": "managed_consent_example",
    "--managed-cleanup": "managed_cleanup_example",
    "--managed-copies": "owned_copy_example",
}


@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_owned_copy_optin_without_running_other_modes(monkeypatch, capsys, mode):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[str] = []

    def forbidden():
        raise AssertionError("unselected mode ran")

    def selected():
        calls.append(mode)
        return {"selected": mode}

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(cli, f"{mode}_owned_copy_example", selected)
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", "--managed-copies"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"selected": mode}
    assert calls == [mode]


@pytest.mark.parametrize(
    "other", ["--managed-cleanup", "--managed-consent", "--endurance", "--copy-limit"]
)
def test_should_refuse_conflicting_or_unfrozen_copy_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--managed-copies", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_describe_fixed_copy_policy_without_operational_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed work")

    for name in (
        "_model",
        "CoupledLearner",
        "ManagedExperienceOwner",
        "CandidateCheckpointController",
    ):
        monkeypatch.setattr(cleanup_demo, name, forbidden)
    value = cleanup_demo.describe_owned_copy_example()
    assert value["retention_policy"]["owned_payload_copies"] == {"max_lifetime_owned_bytes": 576}
    assert value["retention_policy"]["max_lifetime_ingress_bytes"] == 192
    assert (
        value["retention_policy"]["max_checkpoint_pending"]
        == value["retention_policy"]["max_checkpoint_preparations"]
        == 2
    )


def test_should_charge_all_owned_copies_and_refuse_before_disallowed_copy_work(monkeypatch):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    controllers: list[Any] = []
    owners: list[Any] = []
    original_resources: list[Any] = []
    restore_refusals: list[Any] = []
    capture_refusals: list[Any] = []
    original_init = FakeLearner.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._model = SimpleNamespace(input_dim=2, hidden_dims=(3, 2))

    def model(family):
        native = snapshot(family)
        if native.wake_state is not None:
            native = replace(
                native,
                wake_state=replace(
                    native.wake_state,
                    chemical_state=np.zeros(2),
                    plasticity_state=np.ones(2),
                    wake_updates=0,
                ),
            )
        return FakeModel(native)

    def restore(raw, **kwargs):
        captured = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        p = captured.policy
        return FakeLearner(
            FakeModel(captured.native_state),
            learning_rate=p.learning_rate,
            inference_steps=p.inference_steps,
            inference_learning_rate=p.inference_learning_rate,
            inference_policy=p.inference_policy,
        )

    def owner(shared, **kwargs):
        actual = ManagedExperienceOwner(shared, **kwargs)
        owners.append(actual)
        return actual

    def controller(shared, **kwargs):
        runtime, managed = shared._runtime, owners[-1]
        original_resources.append(
            (
                runtime.actor,
                runtime._budget,
                runtime._inbox._clock,
                shared._sharing,
                managed,
                managed._lifecycle,
                runtime.actor._payload_registry,
                managed._catalog,
                managed._opted_out,
                managed._revoked_keys,
                managed._limits,
                runtime._inbox._registration_guard,
                runtime._inbox._training_guard,
                managed._lifecycle._copy_budget,
            )
        )
        actual = CandidateCheckpointController(shared, **kwargs)
        controllers.append(actual)
        original_restore, original_capture = actual.restore, actual.capture

        def restore_token(token):
            life = owners[-1]._lifecycle
            if (
                actual._attempts == 1
                and len(actual._pending) == 1
                and life._copy_budget._charged == 576
            ):
                builder: Any = actual._build

                def forbidden(*args, **kwargs):
                    raise AssertionError("refused restore built or materialized a checkpoint")

                with monkeypatch.context() as local:
                    local.setattr(callbacks_module.CoupledCheckpointCallbacks, "build", forbidden)
                    local.setattr(callbacks_module, "restore_coupled_learner", forbidden)
                    import src.app.candidate_checkpoint as checkpoint_module

                    original_copy = checkpoint_module.copy_graph

                    def guarded_copy(owner, phase, value):
                        if phase in {"checkpoint_build_state", "checkpoint_restore_state"}:
                            forbidden()
                        return original_copy(owner, phase, value)

                    local.setattr(checkpoint_module, "copy_graph", guarded_copy)
                    from src.app.experience_inbox import ExperienceInbox

                    local.setattr(ExperienceInbox, "_materialize_cursor", forbidden)
                    local.setattr(ExperienceInbox, "capture_cursor", forbidden)
                    try:
                        return original_restore(token)
                    except ValueError:
                        restore_refusals.append(
                            (actual._attempts, len(actual._models), builder.build_calls)
                        )
                        raise
            return original_restore(token)

        def capture_token():
            life = owners[-1]._lifecycle
            if len(actual._pending) == 1 and life._copy_budget._charged == 576:

                def forbidden(*args, **kwargs):
                    raise AssertionError("refused capture copied native or raw checkpoint state")

                with monkeypatch.context() as local:
                    local.setattr(shared._runtime._candidate, "snapshot_state", forbidden)
                    from src.app.experience_inbox import ExperienceInbox

                    local.setattr(ExperienceInbox, "capture_cursor", forbidden)
                    try:
                        return original_capture()
                    except ValueError:
                        capture_refusals.append((actual._attempts, len(actual._pending)))
                        raise
            return original_capture()

        monkeypatch.setattr(actual, "restore", restore_token)
        monkeypatch.setattr(actual, "capture", capture_token)
        return actual

    monkeypatch.setattr(FakeLearner, "__init__", initialize)
    monkeypatch.setattr(cleanup_demo, "_model", model)
    monkeypatch.setattr(cleanup_demo, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(cleanup_demo, "ManagedExperienceOwner", owner)
    monkeypatch.setattr(cleanup_demo, "CandidateCheckpointController", controller)
    monkeypatch.setattr(basic_ports, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(copies, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = cleanup_demo.run_owned_copy_example()
    assert (
        result["owned_copy_limits_verified"]
        is result["managed_cleanup_handoff_verified"]
        is result["pc_neutral_parameters_exact"]
        is True
    )
    assert result["full_managed_process_recovery_verified"] is False
    assert len(controllers) == len(result["cases"]) == 6 and FakeLearner.train_calls == 36
    assert restore_refusals == [(2, 1, 1)] * 6 and capture_refusals == [(2, 1)] * 6
    for row, control, managed, original in zip(
        result["cases"], controllers, owners, original_resources, strict=True
    ):
        life, runtime = managed._lifecycle, control._shared._runtime
        assert all(
            left is right
            for left, right in zip(
                (
                    runtime.actor,
                    runtime._budget,
                    runtime._inbox._clock,
                    control._shared._sharing,
                    managed,
                    life,
                    runtime.actor._payload_registry,
                    managed._catalog,
                    managed._opted_out,
                    managed._revoked_keys,
                    managed._limits,
                    runtime._inbox._registration_guard,
                    runtime._inbox._training_guard,
                    life._copy_budget,
                ),
                original,
                strict=True,
            )
        )
        assert control._attempts == 2 and len(control._models) == 1 and len(control._pending) == 0
        assert life.admitted_payload_bytes == 144 and life._policy.max_lifetime_ingress_bytes == 192
        assert (
            life.payload_byte_snapshot().charged_bytes == 576
            and life.payload_byte_snapshot().observed_retained_bytes == 0
        )
        assert runtime._inbox._experiences == runtime._inbox._labels == {}
        assert (
            runtime._budget.updates_completed
            == control._shared._sharing.snapshot().admitted_updates
            == 2
        )
        assert not runtime._stopped and len(runtime.applied_updates) == 2
        assert row["copy_budget_identity"]["original"] == row["copy_budget_identity"]["current"]
        assert row["owned_copy_limits_verified"]
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["build_calls"] == 1
        assert row["caller_copy"] == dict(inspected=False, outside_cleanup_guarantee=True)
        assert [
            (entry["phase"], entry["charged_bytes"], entry["observed_retained_bytes"])
            for entry in row["copy_accounting"]
        ] == [
            ("ingress", 144, 144),
            ("first_capture", 288, 288),
            ("restore", 432, 288),
            ("second_capture", 576, 432),
            ("cleanup", 576, 0),
        ]
        assert all(
            entry["limits"] == {"max_lifetime_owned_bytes": 576} for entry in row["copy_accounting"]
        )
        assert len(row["refusals"]) == 7
        for refusal in row["refusals"]:
            assert refusal["preparation_attempts"] == 2
            assert refusal["prepared_models"] == refusal["builder_calls"] == 1
            assert refusal["complete_state_preserved"]
            if refusal["phase"] in {
                "copy_restore",
                "copy_capture",
                "ingress_source",
                "ingress_label",
            }:
                assert "owned payload byte allowance" in refusal["message"]
                assert "ingress byte" not in refusal["message"]
        before, after = row["before_cleanup"], row["after_cleanup"]
        assert before["native_digests"] == after["native_digests"]
        assert before["policy_digests"] == after["policy_digests"]
        assert before["registry"] == after["registry"] == row["final_registry"]
        assert (
            before["lifecycle"]["admitted_payload_bytes"]
            == after["lifecycle"]["admitted_payload_bytes"]
            == 144
        )
        assert (
            before["lifecycle"]["owned_copies"]["charged_bytes"]
            == after["lifecycle"]["owned_copies"]["charged_bytes"]
            == 576
        )
        assert before["lifecycle"]["owned_copies"]["observed_retained_bytes"] == 432
        assert after["lifecycle"]["owned_copies"]["observed_retained_bytes"] == 0
        assert row["waiting_at5"] == row["empty_at9"] == dict(updates=[], deferred_reason=None)
        assert managed._opted_out == {"subject-A"} and len(managed._catalog) == 4
        assert managed._revoked_keys == {("e0", "s0"), ("e1", "s1"), ("e2", "s2")}
        assert len(runtime._inbox.capture_cursor().erased) == 3
        actor, captured, final = (
            json.loads(row[key]) for key in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8")
        )
        assert actor["policy"] == captured["policy"] == final["policy"] == row["policy"]
        if row["family"] != "pc":
            assert [
                value["native"]["wake_state"]["wake_updates"] for value in (actor, captured, final)
            ] == [2, 3, 4]
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3):
        for key in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            assert (
                json.loads(result["cases"][offset][key])["native"]["arrays"]
                == json.loads(result["cases"][offset + 1][key])["native"]["arrays"]
            )
