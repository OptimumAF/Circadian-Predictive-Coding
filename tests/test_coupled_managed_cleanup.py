"""Typed coupled retention ports and conservative cleanup through actual owners."""

from dataclasses import replace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as ports
import src.adapters.coupled_managed_cleanup as cleanup_demo
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.candidate_checkpoint import CandidateCheckpointController
from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


_ROUTES = {
    None: "example",
    "--repeat-handoffs": "repeated_example",
    "--resource-stops": "resource_stop_example",
    "--endurance": "endurance_example",
    "--managed-consent": "managed_consent_example",
    "--managed-cleanup": "managed_cleanup_example",
}


@pytest.mark.parametrize("option", list(_ROUTES))
@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_only_the_explicit_cleanup_or_existing_cli_workflow(
    monkeypatch, capsys, option, mode
):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[Any] = []

    def forbidden():
        raise AssertionError("unselected workflow ran")

    def selected():
        calls.append((option, mode))
        return {"selected": (option, mode)}

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(cli, f"{mode}_{_ROUTES[option]}", selected)
    monkeypatch.setattr(
        sys, "argv", ["handoff", f"--{mode}"] + ([] if option is None else [option])
    )
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == json.loads(
        json.dumps({"selected": (option, mode)})
    )
    assert calls == [(option, mode)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--managed-cleanup"],
        *[
            ["--run", "--managed-cleanup", flag]
            for flag in (
                "--repeat-handoffs",
                "--resource-stops",
                "--endurance",
                "--managed-consent",
            )
        ],
        ["--run", "--managed-cleanup", "--refund-ingress"],
        ["--run", "--managed-cleanup", "--subject-only"],
    ],
)
def test_should_refuse_mutually_exclusive_or_unfrozen_cleanup_controls(monkeypatch, arguments):
    import scripts.run_coupled_candidate_handoff as cli

    def forbidden():
        raise AssertionError("invalid controls reached product work")

    for suffix in _ROUTES.values():
        for verb in ("describe", "run"):
            monkeypatch.setattr(cli, f"{verb}_{suffix}", forbidden)
    monkeypatch.setattr(sys, "argv", ["handoff", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "kind",
    ["features", "targets", "strided", "empty", "scalar", "nonfinite", "integer", "unsigned"],
)
def test_should_measure_real_numeric_ndarray_storage_without_copy_or_finite_scan(monkeypatch, kind):
    factories = {
        "features": lambda: np.ones((2, 2)),
        "targets": lambda: np.zeros((2, 1)),
        "strided": lambda: np.ones((4, 4))[::-2, ::2],
        "empty": lambda: np.empty((0, 2)),
        "scalar": lambda: np.array(-0.0),
        "nonfinite": lambda: np.array([np.inf, np.nan]),
        "integer": lambda: np.array([1, 2], dtype=np.int16),
        "unsigned": lambda: np.array([1, 2], dtype=np.uint64),
    }
    value = factories[kind]()
    expected, raw = value.nbytes, value.tobytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("byte measurement inspected data or converted the array")

    monkeypatch.setattr(np, "isfinite", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    result = ports.measure_coupled_payload_bytes(value)
    assert type(result) is int and result == expected
    assert value.tobytes() == raw


@pytest.mark.parametrize(
    "kind",
    [
        "bool",
        "complex",
        "object",
        "text",
        "datetime",
        "timedelta",
        "structured",
        "subclass",
        "list",
    ],
)
def test_should_refuse_unsupported_payload_types_without_array_conversion(kind):
    class ArraySubclass(np.ndarray):
        pass

    factories = {
        "bool": lambda: np.array([True]),
        "complex": lambda: np.array([1j]),
        "object": lambda: np.array([object()], dtype=object),
        "text": lambda: np.array(["x"]),
        "datetime": lambda: np.array(["2026-10-15"], dtype="datetime64[D]"),
        "timedelta": lambda: np.array([1], dtype="timedelta64[D]"),
        "structured": lambda: np.zeros(1, dtype=[("x", "f8")]),
        "subclass": lambda: np.ones(2).view(ArraySubclass),
        "list": lambda: [1.0, 2.0],
    }
    with pytest.raises(ValueError):
        ports.measure_coupled_payload_bytes(factories[kind]())


def test_should_refuse_opaque_payload_before_any_byte_or_conversion_hook():
    class Opaque:
        @property
        def nbytes(self):
            raise AssertionError("opaque byte property was accessed")

        def __array__(self, *args, **kwargs):
            raise AssertionError("opaque array conversion was invoked")

        def __deepcopy__(self, memo):
            raise AssertionError("opaque data was copied")

    with pytest.raises(ValueError):
        ports.measure_coupled_payload_bytes(Opaque())


@pytest.mark.parametrize("name", ["coupled_replay_footprint", "erase_coupled_replay_payloads"])
def test_should_report_zero_replay_payload_for_exact_learner_without_any_learner_call(
    monkeypatch, name
):
    learner = object.__new__(CoupledLearner)

    def forbidden(*args, **kwargs):
        raise AssertionError("zero replay port touched native state")

    for method in ("snapshot_state", "predict", "train_batch", "restore_state", "fork"):
        monkeypatch.setattr(CoupledLearner, method, forbidden)
    result = getattr(ports, name)(learner)
    assert type(result) is ReplayPayloadErasure and result == ReplayPayloadErasure(0, 0, 0)
    assert vars(learner) == {}


@pytest.mark.parametrize("name", ["coupled_replay_footprint", "erase_coupled_replay_payloads"])
@pytest.mark.parametrize("kind", ["opaque", "subclass", "native_snapshot", "bool"])
def test_should_require_exact_coupled_learner_family_before_state_access(name, kind):
    class Derived(CoupledLearner):
        pass

    values = {
        "opaque": object(),
        "subclass": object.__new__(Derived),
        "native_snapshot": snapshot(),
        "bool": False,
    }
    with pytest.raises(ValueError):
        getattr(ports, name)(values[kind])


def _policy(**kwargs):
    return DataRetentionPolicy(
        144,
        100,
        PayloadOwnershipLimits(4, 4),
        max_checkpoint_pending=1,
        max_checkpoint_preparations=1,
        max_promotion_pending=1,
        **kwargs,
    )


@pytest.mark.parametrize("unsupported", ["copies", "seconds", "both"])
def test_should_refuse_unsupported_policy_before_owner_or_lifecycle_operations(
    monkeypatch, unsupported
):
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported policy touched lifecycle or owner")

    monkeypatch.setattr(ports, "ManagedDataLifecycle", forbidden)
    policy = _policy(
        owned_payload_copies=PayloadCopyLimits(1000) if unsupported in {"copies", "both"} else None,
        max_retention_seconds=1.0 if unsupported in {"seconds", "both"} else None,
    )
    with pytest.raises(ValueError):
        ports.make_coupled_data_lifecycle(object.__new__(ManagedExperienceOwner), policy=policy)


@pytest.mark.parametrize("policy", [None, False, {}])
def test_should_require_exact_typed_retention_policy_before_lifecycle_construction(
    monkeypatch, policy
):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid policy reached construction")

    monkeypatch.setattr(ports, "ManagedDataLifecycle", forbidden)
    with pytest.raises(ValueError):
        ports.make_coupled_data_lifecycle(object.__new__(ManagedExperienceOwner), policy=policy)


def test_should_forward_original_supported_policy_and_three_explicit_ports(monkeypatch):
    owner, policy, sentinel = object.__new__(ManagedExperienceOwner), _policy(), object()
    calls: list[Any] = []

    def lifecycle(actual, **kwargs):
        calls.append((actual, kwargs))
        return sentinel

    monkeypatch.setattr(ports, "ManagedDataLifecycle", lifecycle)
    assert ports.make_coupled_data_lifecycle(owner, policy=policy) is sentinel
    assert len(calls) == 1 and calls[0][0] is owner
    assert calls[0][1] == dict(
        policy=policy,
        measure_payload_bytes=ports.measure_coupled_payload_bytes,
        native_footprint=ports.coupled_replay_footprint,
        native_erase=ports.erase_coupled_replay_payloads,
    )
    assert calls[0][1]["policy"] is policy


def test_should_describe_conservative_cleanup_without_lifecycle_or_native_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description reached operational work")

    for name in (
        "_model",
        "CoupledLearner",
        "ManagedExperienceOwner",
        "make_coupled_data_lifecycle",
        "CandidateCheckpointController",
    ):
        monkeypatch.setattr(cleanup_demo, name, forbidden)
    value = cleanup_demo.describe_managed_cleanup_example()
    assert value["protocol_id"] == "coupled_managed_cleanup_engineering_v1"
    assert value["native_training_calls"] == 36 and value["process_launches"] == 0
    assert value["runtime_update_limit"] == 3 and value["preparation_attempt_limit"] == 1
    assert value["label_arrivals"] == [3, 7, 9] and value["cleanup_at"] == 8
    assert value["lifecycle_limits"] == dict(
        max_approved_records=4, max_replay_records=4, allow_synthetic=True, allow_unverified=True
    )
    assert value["retention_policy"] == dict(
        max_lifetime_ingress_bytes=144,
        max_retention_ticks=100,
        holders=dict(max_live_holders=4, max_lifetime_enrollments=4),
        max_checkpoint_pending=1,
        max_checkpoint_preparations=1,
        max_promotion_pending=1,
        owned_payload_copies=None,
        max_retention_seconds=None,
    )
    assert value["native_energy_evaluation_cap"] == 180 and value["native_constructor_cap"] == 48
    assert value["prediction_calls"] == 42 and value["prediction_cap"] == 64
    assert value["guarded_inference_cap"] == 18


def test_should_cleanup_all_registered_raw_holders_without_resetting_native_or_lifetime_authority(
    monkeypatch,
):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    controllers: list[Any] = []
    owners: list[Any] = []
    original_resources: list[Any] = []
    retired: list[Any] = []

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
        policy = captured.policy
        return FakeLearner(
            FakeModel(captured.native_state),
            learning_rate=policy.learning_rate,
            inference_steps=policy.inference_steps,
            inference_learning_rate=policy.inference_learning_rate,
            inference_policy=policy.inference_policy,
        )

    def owner(shared, **kwargs):
        actual = ManagedExperienceOwner(shared, **kwargs)
        owners.append(actual)
        return actual

    def controller(shared, **kwargs):
        assert kwargs["max_pending"] == kwargs["max_prepare_attempts"] == 1
        runtime, managed_owner = shared._runtime, owners[-1]
        retired.append(runtime)
        original_resources.append(
            (
                runtime.actor,
                runtime._budget,
                runtime._inbox._clock,
                shared._sharing,
                managed_owner,
                managed_owner._lifecycle,
                managed_owner._catalog,
                managed_owner._opted_out,
                managed_owner._revoked_keys,
                managed_owner._limits,
                runtime.actor._payload_registry,
                runtime._inbox._registration_guard,
                runtime._inbox._training_guard,
            )
        )
        actual = CandidateCheckpointController(shared, **kwargs)
        controllers.append(actual)
        return actual

    monkeypatch.setattr(cleanup_demo, "_model", model)
    monkeypatch.setattr(cleanup_demo, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(cleanup_demo, "ManagedExperienceOwner", owner)
    monkeypatch.setattr(cleanup_demo, "CandidateCheckpointController", controller)
    monkeypatch.setattr(ports, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = cleanup_demo.run_managed_cleanup_example()
    assert (
        result["managed_cleanup_handoff_verified"] is result["pc_neutral_parameters_exact"] is True
    )
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(owners) == len(controllers) == 6
    assert FakeLearner.train_calls == 36
    for row, consent, control, old, original in zip(
        result["cases"], owners, controllers, retired, original_resources, strict=True
    ):
        runtime, gate = control._shared._runtime, control._shared._sharing
        life = consent._lifecycle
        assert all(
            left is right
            for left, right in zip(
                (
                    runtime.actor,
                    runtime._budget,
                    runtime._inbox._clock,
                    gate,
                    consent,
                    life,
                    consent._catalog,
                    consent._opted_out,
                    consent._revoked_keys,
                    consent._limits,
                    runtime.actor._payload_registry,
                    runtime._inbox._registration_guard,
                    runtime._inbox._training_guard,
                ),
                original,
                strict=True,
            )
        )
        assert life.admitted_payload_bytes == 144
        assert life._policy == _policy() and life._copy_budget is None
        assert runtime.actor._payload_registry._total == 4
        assert len(runtime.actor._payload_registry._live()) == 4
        assert control._attempts == len(control._models) == 1 and len(control._pending) == 0
        assert old._retired and not runtime._stopped
        assert (
            old._inbox._experiences
            == old._inbox._labels
            == runtime._inbox._experiences
            == runtime._inbox._labels
            == {}
        )
        old_cursor, live_cursor = old._inbox.capture_cursor(), runtime._inbox.capture_cursor()
        assert old_cursor.completed_updates == len(old_cursor.applied) == 1
        assert live_cursor.completed_updates == len(live_cursor.applied) == 2
        assert len(old_cursor.erased) == len(live_cursor.erased) == 3
        assert old._inbox._historical_completed_updates == 1
        assert live_cursor.last_tick == 9 and not live_cursor.stopped
        assert runtime._budget.updates_completed == gate.snapshot().admitted_updates == 2
        assert runtime._budget.budget.max_training_updates == 3
        assert consent._opted_out == {"subject-A"}
        assert consent._revoked_keys == {("e0", "s0"), ("e1", "s1"), ("e2", "s2")}
        assert len(consent.declarations) == 4
        for index, item in enumerate(consent.declarations):
            assert item.key == (f"e{index}", f"s{index}")
            assert (
                item.provenance.subject_id
                == ("subject-A", "subject-B", "subject-A", "subject-C")[index]
            )
            assert item.provenance.source_id == "local-fixed-engineering-fixture"
            assert item.provenance.synthetic is True and item.provenance.verified is False
            assert (
                item.consent.training is item.consent.replay is True and item.retention == "replay"
            )
        assert ports.coupled_replay_footprint(runtime._candidate) == ReplayPayloadErasure(0, 0, 0)
        assert row["actor_unchanged"] and row["complete_continuation_exact"]
        assert row["same_managed_authority"] and row["same_actor_budget_clock_gate"]
        assert row["owned_payloads_cleared"]
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["builds"] == ["correct_policy"]
        assert row["preparation_attempts"] == row["prepared_models"] == 1
        assert row["pending_checkpoints"] == 0
        # The public JSON boundary turns immutable native keys into arrays.
        report = json.loads(json.dumps(row["cleanup_report"]))
        assert report == dict(
            reason="opt_out",
            requested_keys=[["e0", "s0"], ["e2", "s2"]],
            revoked_keys=[["e0", "s0"], ["e1", "s1"], ["e2", "s2"]],
            model_snapshots_erased=0,
            inboxes_cleared=2,
            checkpoints_invalidated=1,
            promotions_invalidated=0,
        )
        before, after = row["before_cleanup"], row["after_cleanup"]
        assert before["budget"] == after["budget"]
        assert before["sharing"] == after["sharing"]
        assert before["registry"] == after["registry"]
        assert before["native_digests"] == after["native_digests"]
        assert before["policy_digests"] == after["policy_digests"]
        assert row["caller_copy"] == dict(
            experiences=3, labels=3, payload_bytes=144, outside_cleanup_guarantee=True
        )
        assert (
            before["lifecycle"]["admitted_payload_bytes"]
            == after["lifecycle"]["admitted_payload_bytes"]
            == row["final_lifecycle"]["admitted_payload_bytes"]
            == 144
        )
        assert (
            before["lifecycle"]["policy"]
            == after["lifecycle"]["policy"]
            == row["final_lifecycle"]["policy"]
        )
        assert before["registry"] == row["final_registry"]
        assert after["old_revision"] == before["old_revision"] + 1
        assert after["new_revision"] == before["new_revision"] + 1
        for key, completed in (("old_cursor", 1), ("new_cursor", 2)):
            assert len(before[key]["experiences"]) == len(before[key]["labels"]) == 3
            assert after[key]["experiences"] == after[key]["labels"] == []
            assert before[key]["applied"] == after[key]["applied"]
            assert before[key]["completed_updates"] == after[key]["completed_updates"] == completed
            assert len(after[key]["erased"]) == 3 and after[key]["last_tick"] == 8
        assert len(row["refusals"]) == 5
        assert {value["phase"] for value in row["refusals"]} == {
            "invalidated_checkpoint",
            "ingress_source",
            "ingress_label",
            "regrant_subject_a",
            "fifth_declaration",
        }
        for refusal in row["refusals"]:
            expected_delta = 1 if refusal["phase"] in {"ingress_source", "ingress_label"} else 0
            assert refusal["revision_delta"] == expected_delta
            assert refusal["revision_after"] == refusal["revision_before"] + expected_delta
            assert refusal["complete_state_preserved"] is True
            assert (
                refusal["preparation_attempts"]
                == refusal["prepared_models"]
                == refusal["builder_calls"]
                == 1
            )
            assert refusal["message"]
        assert row["waiting_at5"] == row["empty_at9"] == dict(updates=[], deferred_reason=None)
        assert [item["sample_id"] for item in row["first_updates"]] == ["s0"]
        assert [item["sample_id"] for item in row["second_updates"]] == ["s1"]
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
