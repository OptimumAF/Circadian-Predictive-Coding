"""Elapsed policy composition and original deadline enforcement without real ML."""

from dataclasses import replace
from types import SimpleNamespace
import json
import sys
from typing import Any

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as basic
import src.adapters.coupled_payload_copies as copies
import src.adapters.coupled_elapsed_retention as elapsed
import src.adapters.coupled_retention_example as example
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.managed_experience import ManagedExperienceOwner
from src.app.payload_ownership import PayloadOwnershipBusy
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


def policy(*, owned=True):
    return DataRetentionPolicy(
        192,
        100,
        PayloadOwnershipLimits(4, 4),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(576) if owned else None,
        max_retention_seconds=2.0,
    )


@pytest.mark.parametrize(
    "bad",
    [
        "type",
        "missing_seconds",
        "zero_seconds",
        "nan_seconds",
        "bool_seconds",
        "holder_type",
        "holder_value",
        "copy_type",
        "copy_value",
        "ticks",
    ],
)
def test_should_revalidate_complete_elapsed_policy_before_owner_or_constructor_hooks(
    monkeypatch, bad
):
    selected: Any = policy()
    if bad == "type":
        selected = object()
    elif bad == "holder_value":
        object.__setattr__(selected.holders, "max_live_holders", True)
    elif bad == "copy_value":
        object.__setattr__(selected.owned_payload_copies, "max_lifetime_owned_bytes", -1)
    else:
        field, value = {
            "missing_seconds": ("max_retention_seconds", None),
            "zero_seconds": ("max_retention_seconds", 0),
            "nan_seconds": ("max_retention_seconds", float("nan")),
            "bool_seconds": ("max_retention_seconds", True),
            "holder_type": ("holders", object()),
            "copy_type": ("owned_payload_copies", object()),
            "ticks": ("max_retention_ticks", 0),
        }[bad]
        object.__setattr__(selected, field, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid policy touched owner or lifecycle")

    monkeypatch.setattr(elapsed, "ManagedDataLifecycle", forbidden)
    with pytest.raises(ValueError):
        elapsed.make_coupled_elapsed_lifecycle(
            object.__new__(ManagedExperienceOwner), policy=selected
        )


@pytest.mark.parametrize("owned", [False, True])
def test_should_compose_original_elapsed_policy_and_all_eight_explicit_ports_without_worker(
    monkeypatch, owned
):
    owner, selected, sentinel = (
        object.__new__(ManagedExperienceOwner),
        policy(owned=owned),
        object(),
    )
    captured: list[Any] = []

    def constructor(actual, **kwargs):
        captured.append((actual, kwargs))
        return sentinel

    monkeypatch.setattr(elapsed, "ManagedDataLifecycle", constructor)
    assert elapsed.make_coupled_elapsed_lifecycle(owner, policy=selected) is sentinel
    assert len(captured) == 1 and captured[0][0] is owner
    assert captured[0][1] == dict(
        policy=selected,
        measure_payload_bytes=basic.measure_coupled_payload_bytes,
        native_footprint=basic.coupled_replay_footprint,
        native_erase=basic.erase_coupled_replay_payloads,
        measure_auxiliary_bytes=copies.measure_coupled_auxiliary_bytes,
        measure_checkpoint_bytes=copies.measure_coupled_checkpoint_bytes,
        native_growth_bytes=copies.coupled_native_growth_bytes,
        prepare_model_bytes=copies.coupled_preparation_bytes,
        prediction_cache_bytes=copies.coupled_prediction_cache_bytes,
    )


@pytest.mark.parametrize(
    "factory", [basic.make_coupled_data_lifecycle, copies.make_coupled_copy_lifecycle]
)
def test_should_keep_older_factories_refusing_elapsed_policy_before_owner_access(factory):
    with pytest.raises(ValueError, match="elapsed|retention"):
        factory(object.__new__(ManagedExperienceOwner), policy=policy())


@pytest.mark.parametrize("kind", ["stable", "promotable"])
@pytest.mark.parametrize("phase", ["deadline", "stopped", "input_copy", "learner", "output_copy"])
def test_should_check_every_direct_actor_prediction_boundary_against_original_deadline(
    monkeypatch, kind, phase
):
    from test_retention_expiry import setup, promotion_fixture
    from test_managed_experience import declaration, record

    if kind == "stable":
        owner, _, driver, _, runtime, _, wall, _, _ = setup()
        owner.declare(declaration())
        record(owner)
        actor, learner = runtime.actor, runtime.actor._learner
    else:
        actor, runtime, _, wall, driver = promotion_fixture()
        learner = actor._slot.bundle.learner
    native_before = learner.snapshot_state()
    calls = dict(input=0, learner=0, output=0)

    class Input:
        def __deepcopy__(self, memo):
            calls["input"] += 1
            if phase in {"deadline", "stopped"}:
                raise AssertionError("closed actor copied input")
            if phase == "input_copy":
                wall.value = 2.0
            return []

    class Output:
        def __deepcopy__(self, memo):
            calls["output"] += 1
            if phase == "learner":
                raise AssertionError("expired actor copied output")
            wall.value = 2.0
            return [0.5]

    def predict(features):
        calls["learner"] += 1
        if phase == "learner":
            wall.value = 2.0
        return Output()

    monkeypatch.setattr(learner, "predict", predict)
    if phase == "deadline":
        wall.value = 2.0
    if phase == "stopped":
        assert driver.stop()
    try:
        with pytest.raises(ValueError, match="expired|retention|terminal|driver"):
            actor.predict(Input())
        assert calls["input"] == (0 if phase in {"deadline", "stopped"} else 1)
        assert calls["learner"] == (1 if phase in {"learner", "output_copy"} else 0)
        assert calls["output"] == (1 if phase == "output_copy" else 0)
        assert learner.snapshot_state() == native_before
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive


def test_should_describe_elapsed_control_without_models_clock_reads_or_threads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    for name in ("_model", "CoupledLearner", "RetentionExpiryDriver", "ManagedCoupledBuilder"):
        monkeypatch.setattr(example, name, forbidden)
    value = example.describe_elapsed_retention_example()
    assert value["protocol_id"] == "coupled_elapsed_retention_engineering_v1"
    assert value["native_training_calls"] == 36
    assert value["retention_policy"]["max_retention_seconds"] == 2.0
    assert value["retention_policy"]["owned_payload_copies"] == {"max_lifetime_owned_bytes": 576}


@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_elapsed_optin_without_other_work(monkeypatch, capsys, mode):
    import scripts.run_coupled_candidate_handoff as cli

    selected: list[str] = []

    def choose():
        selected.append(mode)
        return {"mode": mode}

    monkeypatch.setattr(cli, f"{mode}_elapsed_retention_example", choose)
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", "--elapsed-retention"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"mode": mode}
    assert selected == [mode]


@pytest.mark.parametrize("other", ["--managed-copies", "--managed-cleanup", "--deadline-seconds"])
def test_should_refuse_conflicting_or_unfrozen_elapsed_cli_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--elapsed-retention", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_expire_six_handoffs_with_original_driver_clock_copy_budget_and_joined_workers(
    monkeypatch,
):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    drivers: list[Any] = []
    originals: list[tuple[Any, ...]] = []
    wake_observations: list[dict[str, int]] = []
    original_init = FakeLearner.__init__
    actual_driver = example.RetentionExpiryDriver

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

    def driver(life, **kwargs):
        result = actual_driver(life, **kwargs)
        drivers.append(result)
        observed = dict(wake=0, wait=0, initial_busy=0)
        wake_observations.append(observed)
        actual_wake, actual_wait, actual_poll = result.wake, result.wait_for_purge, result.poll_once

        def poll():
            if observed["initial_busy"] == 0:
                observed["initial_busy"] = 1
                raise PayloadOwnershipBusy("deterministic initial owner contention")
            return actual_poll()

        def wake():
            observed["wake"] += 1
            actual_wake()

        def wait(timeout_seconds):
            # Starting under the operation gate can make the first worker poll
            # busy. Explicit notification must precede the bounded purge wait.
            assert observed["wake"] == 1 and observed["wait"] == 0 and timeout_seconds == 2.0
            observed["wait"] += 1
            return actual_wait(timeout_seconds)

        monkeypatch.setattr(result, "wake", wake)
        monkeypatch.setattr(result, "wait_for_purge", wait)
        monkeypatch.setattr(result, "poll_once", poll)
        owner = life._owner
        runtime = owner._shared._runtime
        originals.append(
            (
                life,
                owner,
                runtime.actor,
                runtime._budget,
                runtime._inbox._clock,
                runtime._budget.clock,
                owner._shared._sharing,
                runtime.actor._payload_registry,
                life._copy_budget,
                owner._catalog,
                owner._opted_out,
                owner._revoked_keys,
                owner._limits,
                runtime._inbox._registration_guard,
                runtime._inbox._training_guard,
                result._limits,
                result._created_at,
            )
        )
        return result

    monkeypatch.setattr(FakeLearner, "__init__", initialize)
    monkeypatch.setattr(example, "_model", model)
    monkeypatch.setattr(example, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(example, "RetentionExpiryDriver", driver)
    monkeypatch.setattr(basic, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(copies, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = example.run_elapsed_retention_example()
    assert result["elapsed_retention_verified"] is result["pc_neutral_parameters_exact"] is True
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(drivers) == 6 and FakeLearner.train_calls == 36
    assert wake_observations == [dict(wake=1, wait=1, initial_busy=1)] * 6
    assert [(row["mode"], row["family"]) for row in result["cases"]] == [
        (mode, family)
        for mode in ("fixed", "guarded")
        for family in ("pc", "neutral_cpc", "active_cpc")
    ]
    for row, worker, original in zip(result["cases"], drivers, originals, strict=True):
        life, owner = worker._lifecycle, worker._lifecycle._owner
        runtime = owner._shared._runtime
        assert all(
            left is right
            for left, right in zip(
                (
                    life,
                    owner,
                    runtime.actor,
                    runtime._budget,
                    runtime._inbox._clock,
                    runtime._budget.clock,
                    owner._shared._sharing,
                    runtime.actor._payload_registry,
                    life._copy_budget,
                    owner._catalog,
                    owner._opted_out,
                    owner._revoked_keys,
                    owner._limits,
                    runtime._inbox._registration_guard,
                    runtime._inbox._training_guard,
                    worker._limits,
                ),
                original[:-1],
                strict=True,
            )
        )
        assert worker._created_at == original[-1]
        assert life._wall_clock is runtime._budget.clock and life._retention_driver is worker
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["driver_origin"] == dict(
            created_at=worker._created_at, limits_id=id(worker._limits), preserved=True
        )
        assert row["automatic_purge_verified"] and row["thread_joined"]
        assert row["complete_continuation_exact"] and row["actor_unchanged"]
        assert worker.snapshot().state == "stopped" and not worker.snapshot().alive
        assert worker._thread is not None and not worker._thread.is_alive()
        assert worker._lifecycle._retention_driver is worker
        assert worker._lifecycle._copy_budget._charged == 576
        assert worker._lifecycle.admitted_payload_bytes == 144
        ready, purged, stopped = (
            row[key] for key in ("driver_ready", "driver_after_purge", "driver_final")
        )
        assert (
            ready["state"] == "ready"
            and ready["polls"] == ready["purges"] == ready["cleanup_attempts"] == 0
        )
        assert ready["alive"] is False and ready["pending_cleanup"] is False
        assert (
            ready["limits"]
            == purged["limits"]
            == stopped["limits"]
            == dict(
                poll_interval_seconds=30.0,
                max_polls=4,
                max_run_seconds=60.0,
                join_timeout_seconds=2.0,
            )
        )
        assert 1 <= purged["polls"] <= stopped["polls"] <= 4
        assert 1 <= purged["purges"] <= stopped["purges"] <= stopped["cleanup_attempts"] <= 6
        assert purged["cleanup_attempts"] <= stopped["cleanup_attempts"]
        assert (
            stopped["state"] == "stopped"
            and stopped["alive"] is stopped["pending_cleanup"] is False
        )
        assert ready["error_type"] is purged["error_type"] is stopped["error_type"] is None
        assert [
            (v["phase"], v["charged_bytes"], v["observed_retained_bytes"])
            for v in row["copy_accounting"]
        ] == [
            ("ingress", 144, 144),
            ("first_capture", 288, 288),
            ("restore", 432, 288),
            ("second_capture", 576, 432),
            ("cleanup", 576, 0),
        ]
        assert all(v["limits"] == {"max_lifetime_owned_bytes": 576} for v in row["copy_accounting"])
        before, after = row["before_expiry"], row["after_stop"]
        assert all(
            after[name] == before[name] + stopped["purges"]
            for name in ("old_revision", "new_revision")
        )
        assert before["budget"] == after["budget"] and before["sharing"] == after["sharing"]
        assert before["registry"] == after["registry"]
        assert {k: v for k, v in before["managed"].items() if k != "revoked_keys"} == {
            k: v for k, v in after["managed"].items() if k != "revoked_keys"
        }
        assert before["managed"]["revoked_keys"] == []
        assert after["managed"]["revoked_keys"] == [[f"e{i}", f"s{i}"] for i in range(3)]
        assert before["managed"]["opted_out"] == after["managed"]["opted_out"] == []
        assert (
            before["owned_copies"]["observed_retained_bytes"] == 432
            and after["owned_copies"]["observed_retained_bytes"] == 0
        )
        assert (
            before["owned_copies"]["charged_bytes"] == after["owned_copies"]["charged_bytes"] == 576
        )
        for name, completed in (("old_cursor", 1), ("new_cursor", 2)):
            previous, current = before[name], after[name]
            assert len(previous["experiences"]) == len(previous["labels"]) == 3
            assert current["experiences"] == current["labels"] == []
            assert current["erased"] == [
                dict(
                    key=[f"e{i}", f"s{i}"],
                    actor_version="a0",
                    observed_at=0,
                    event_id=f"l{i}",
                    arrived_at=arrival,
                    erased_at=8,
                    reason="expired",
                )
                for i, arrival in enumerate((3, 7, 9))
            ]
            assert (
                previous["applied"] == current["applied"] and len(current["applied"]) == completed
            )
            assert (
                current["completed_updates"] == completed
                and current["last_tick"] == 8
                and current["stopped"] is False
            )
        assert after["pending_checkpoints"] == 0 and before["pending_checkpoints"] == 1
        assert (
            before["preparation_attempts"]
            == after["preparation_attempts"]
            == before["prepared_models"]
            == after["prepared_models"]
            == row["build_calls"]
            == 1
        )
        assert (
            runtime._budget.updates_completed
            == owner._shared._sharing.snapshot().admitted_updates
            == 2
        )
        assert row["waiting_at5"] == dict(updates=[], deferred_reason=None)
        for key in ("deadline_refusals", "terminal_refusals"):
            assert [v["phase"] for v in row[key]] == [
                "predict",
                "snapshot",
                "restore",
                "ingress_source",
                "ingress_label",
            ]
            for refusal in row[key]:
                assert refusal["complete_state_preserved"] and refusal["poison_copy_attempts"] == 0
                assert refusal["preparation_attempts"] == refusal["build_calls"] == 1
                expected = (
                    "declared payload retention has expired; cleanup required"
                    if key == "deadline_refusals"
                    else "checkpoint is foreign, copied, discarded or already used"
                    if refusal["phase"] == "restore"
                    else "retention driver terminal; raw admission/access is closed"
                )
                assert refusal["message"] == expected
        assert [v["phase"] for v in row["renewal_refusals"]] == ["restart", "replacement"]
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
