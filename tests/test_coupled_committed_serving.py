"""Committed cache generations, live rollback and terminal raw expiry."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any
import json
import sys

import numpy as np
import pytest

import src.adapters.coupled_managed_serving as example
import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as basic
import src.adapters.coupled_payload_copies as copies
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.payload_ownership import PayloadOwnershipBusy
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot, advance


@pytest.mark.parametrize("terminal", [False, True])
def test_should_keep_rollback_live_until_original_deadline_then_refuse_publication(terminal):
    from test_retention_expiry import promotion_fixture
    from test_serving_promotion import prepare

    actor, runtime, control, wall, driver = promotion_fixture()
    actor.serve([0], now=0)
    ticket = prepare(control, runtime)
    receipt = control.commit(runtime, ticket)
    current, previous = actor._slot.bundle, actor._slot.previous
    assert previous is not None
    wall.value = 1.0
    assert control._latest[0] is receipt and actor._slot.generation == 1
    assert previous.cache and previous.metadata
    assert current.learner.state == [0.75, 0.8] and previous.learner.state == [0.5, 0.8]
    if terminal:
        assert driver.stop()
    else:
        wall.value = 2.0
    slot = actor._slot
    try:
        with pytest.raises(ValueError, match="expired|rollback receipt"):
            control.rollback(receipt)
        assert actor._slot is slot and current.learner.state == [0.75, 0.8]
        if terminal:
            assert control._latest is None and actor._slot.previous is None
            assert previous.metadata == previous.cache == current.metadata == current.cache == {}
        else:
            assert control._latest[0] is receipt and actor._slot.previous is previous
            assert actor._slot.generation == 1 and previous.cache and previous.metadata
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive


def test_should_describe_committed_variant_without_work_and_preserve_pending_controls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    for name in ("_model", "CoupledLearner", "RetentionExpiryDriver", "ManagedCoupledBuilder"):
        monkeypatch.setattr(example, name, forbidden)
    original = example.describe_managed_serving_example()
    committed = example.describe_committed_serving_example()
    assert original["protocol_id"] == "coupled_managed_serving_expiry_engineering_v1"
    assert committed["protocol_id"] == "coupled_committed_serving_expiry_engineering_v1"
    assert original["retention_policy"]["owned_payload_copies"] == {"max_lifetime_owned_bytes": 640}
    assert committed["retention_policy"]["owned_payload_copies"] == {
        "max_lifetime_owned_bytes": 656
    }
    changed = {
        "protocol_id",
        "retention_policy",
        "prediction_calls",
        "serving_requests",
        "copy_charges",
        "retained_raw_bytes",
    }
    for key, value in original.items():
        if key not in changed:
            assert committed[key] == value
    assert {
        k: v for k, v in original["retention_policy"].items() if k != "owned_payload_copies"
    } == {k: v for k, v in committed["retention_policy"].items() if k != "owned_payload_copies"}
    assert committed["prediction_calls"] == 60 and committed["serving_requests"] == 54


@pytest.mark.parametrize("mode", ["describe", "run"])
@pytest.mark.parametrize("route", ["default", "pending", "committed"])
def test_should_route_new_mode_without_changing_default_or_pending_routes(
    monkeypatch, capsys, mode, route
):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[str] = []
    suffixes = {
        "default": "example",
        "pending": "managed_serving_example",
        "committed": "committed_serving_example",
    }
    for name, suffix in suffixes.items():

        def selected(name=name):
            calls.append(name)
            return {"route": (name, mode)}

        monkeypatch.setattr(cli, f"{mode}_{suffix}", selected)
    flag = {
        "default": [],
        "pending": ["--managed-serving-expiry"],
        "committed": ["--committed-serving-expiry"],
    }[route]
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", *flag])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"route": [route, mode]}
    assert calls == [route]


@pytest.mark.parametrize(
    "other", ["--managed-serving-expiry", "--elapsed-retention", "--copy-limit"]
)
def test_should_refuse_conflicting_or_unfrozen_committed_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--committed-serving-expiry", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_expire_committed_current_and_previous_raw_bundles_without_native_rollback(
    monkeypatch,
):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    drivers: list[Any] = []
    previous_bundles: list[Any] = []
    pending_bundles: list[Any] = []
    controls: list[Any] = []
    receipts: list[Any] = []
    originals: list[tuple[Any, ...]] = []
    startup: list[dict[str, int]] = []
    initialize = FakeLearner.__init__
    actual_driver, actual_promotion = (
        example.RetentionExpiryDriver,
        example.ServingPromotionController,
    )

    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self._model = SimpleNamespace(input_dim=2, hidden_dims=(3, 2))

    def model(family):
        state = snapshot(family)
        if state.wake_state is not None:
            state = replace(
                state,
                wake_state=replace(
                    state.wake_state,
                    chemical_state=np.zeros(2),
                    plasticity_state=np.ones(2),
                    wake_updates=0,
                ),
            )
        return FakeModel(state)

    def restore(raw, **kwargs):
        checkpoint = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        p = checkpoint.policy
        return FakeLearner(
            FakeModel(checkpoint.native_state),
            learning_rate=p.learning_rate,
            inference_steps=p.inference_steps,
            inference_learning_rate=p.inference_learning_rate,
            inference_policy=p.inference_policy,
        )

    def driver(life, **kwargs):
        worker = actual_driver(life, **kwargs)
        drivers.append(worker)
        owner = life._owner
        current = owner._shared._runtime
        originals.append(
            (
                life,
                owner,
                current.actor,
                current._budget,
                current._inbox._clock,
                current._budget.clock,
                owner._shared._sharing,
                current.actor._payload_registry,
                life._copy_budget,
                owner._catalog,
                owner._opted_out,
                owner._revoked_keys,
                owner._limits,
                current._inbox._registration_guard,
                current._inbox._training_guard,
                worker._limits,
                worker._created_at,
            )
        )
        observed = dict(busy=0, wake=0, wait=0)
        startup.append(observed)
        poll, wake, wait = worker.poll_once, worker.wake, worker.wait_for_purge

        def poll_once():
            if not observed["busy"]:
                observed["busy"] = 1
                raise PayloadOwnershipBusy("deterministic initial contention")
            return poll()

        def notify():
            observed["wake"] += 1
            wake()

        def wait_for_purge(seconds):
            assert observed["wake"] == 1 and observed["wait"] == 0 and seconds == 2.0
            observed["wait"] += 1
            return wait(seconds)

        monkeypatch.setattr(worker, "poll_once", poll_once)
        monkeypatch.setattr(worker, "wake", notify)
        monkeypatch.setattr(worker, "wait_for_purge", wait_for_purge)
        return worker

    def promotion(*args, **kwargs):
        control = actual_promotion(*args, **kwargs)
        controls.append(control)
        prepare, commit = control.prepare, control.commit

        def prepare_ticket(*args, **kwargs):
            ticket = prepare(*args, **kwargs)
            pending_bundles.append(control._pending[ticket].bundle)
            return ticket

        def publish(*args, **kwargs):
            previous = control._actor._slot.bundle
            receipt = commit(*args, **kwargs)
            previous_bundles.append(previous)
            receipts.append(receipt)
            assert (
                control._latest is not None
                and control._latest[0] is receipt
                and control._actor._slot.previous is previous_bundles[-1]
            )
            current = control._actor._slot.bundle
            assert current.metadata is not previous.metadata and current.cache is not previous.cache
            current_raw, previous_raw = current.metadata["raw"], previous.metadata["raw"]
            assert isinstance(current_raw, np.ndarray) and isinstance(previous_raw, np.ndarray)
            assert not np.shares_memory(current_raw, previous_raw)
            return receipt

        monkeypatch.setattr(control, "prepare", prepare_ticket)
        monkeypatch.setattr(control, "commit", publish)
        return control

    monkeypatch.setattr(FakeLearner, "__init__", init)
    monkeypatch.setattr(example, "_model", model)
    monkeypatch.setattr(example, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(example, "RetentionExpiryDriver", driver)
    monkeypatch.setattr(example, "ServingPromotionController", promotion)
    monkeypatch.setattr(basic, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(copies, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    report = example.run_committed_serving_example()
    assert (
        report["committed_serving_expiry_verified"] is report["pc_neutral_parameters_exact"] is True
    )
    assert report["full_managed_process_recovery_verified"] is False
    assert len(report["cases"]) == len(drivers) == 6 and FakeLearner.train_calls == 36
    assert [(v["mode"], v["family"]) for v in report["cases"]] == [
        (mode, family)
        for mode in ("fixed", "guarded")
        for family in ("pc", "neutral_cpc", "active_cpc")
    ]
    assert startup == [dict(busy=1, wake=1, wait=1)] * 6
    assert len(receipts) == 6
    for row, worker, control, previous, pending, original in zip(
        report["cases"],
        drivers,
        controls,
        previous_bundles,
        pending_bundles,
        originals,
        strict=True,
    ):
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
        assert worker._created_at == original[-1] and life._retention_driver is worker
        assert worker.snapshot().state == "stopped" and not worker.snapshot().alive
        assert (
            control._latest is None
            and control._pending == {}
            and runtime.actor._slot.previous is None
        )
        assert previous.metadata == previous.cache == pending.metadata == pending.cache == {}
        assert runtime.actor._slot.bundle.metadata == runtime.actor._slot.bundle.cache == {}
        assert life._copy_budget._charged == 656 and life.admitted_payload_bytes == 144
        assert row["complete_continuation_exact"]
        assert row["policy"] == dict(
            learning_rate=0.025,
            inference_steps=3,
            inference_learning_rate=0.07,
            inference_policy=None
            if row["mode"] == "fixed"
            else dict(min_steps=1, gradient_tolerance=1e6, relative_energy_tolerance=None),
        )
        assert "actor_unchanged" not in row and "pending_promotion_committed" not in row
        assert (
            row["promotion_committed"]
            is row["rollback_live_before_expiry"]
            is row["rollback_revoked"]
            is True
        )
        assert row["serving_c_unchanged"] and row["previous_a_preserved"]
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["commit_receipt"] == dict(
            generation=1, previous_version="a0", actor_version="c0"
        )
        assert row["commit_evidence"] == dict(
            generation_before=0,
            generation_after=1,
            pending_before=1,
            pending_after=0,
            original_receipt_live=True,
            previous_bundle_retained=True,
            independent_metadata=True,
            independent_cache=True,
        )
        assert row["build_calls"] == 4 and row["guard_callback_counts"] == dict(
            feature_digest=5, state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1
        )
        guard = row["guard_report"]
        assert guard["new_guard_keys"] == [["guard-new", "g0"]] and guard["old_guard_keys"] == [
            ["guard-old", "g0"]
        ]
        assert guard["training_ids"] == [[f"e{i}", f"s{i}"] for i in range(3)]
        assert (
            guard["assessed_at"] == 8
            and guard["label_arrivals"] == [7, 7]
            and guard["decision"] == dict(accepted=True, reasons=[])
        )
        assert guard["evidence"]["candidate_max_prediction_seconds"] == 0.0
        assert [
            (v["phase"], v["charged_bytes"], v["observed_retained_bytes"])
            for v in row["copy_accounting"]
        ] == [
            ("initial", 24, 24),
            ("ingress", 168, 168),
            ("cache", 184, 184),
            ("first_capture", 328, 328),
            ("restore", 472, 328),
            ("second_capture", 616, 472),
            ("promotion", 640, 496),
            ("commit", 640, 496),
            ("committed_cache", 656, 512),
            ("committed_hit", 656, 512),
            ("cleanup", 656, 0),
        ]
        assert all(v["limits"] == {"max_lifetime_owned_bytes": 656} for v in row["copy_accounting"])
        ready, stopped = row["driver_ready"], row["driver_final"]
        assert (
            ready["polls"] == ready["purges"] == ready["cleanup_attempts"] == 0
            and ready["state"] == "ready"
        )
        assert (
            stopped["limits"]
            == ready["limits"]
            == dict(
                poll_interval_seconds=30.0,
                max_polls=4,
                max_run_seconds=60.0,
                join_timeout_seconds=2.0,
            )
        )
        assert (
            1 <= stopped["polls"] <= 4
            and 1 <= stopped["purges"] <= stopped["cleanup_attempts"] <= 6
        )
        assert (
            stopped["pending_cleanup"] is stopped["alive"] is False
            and stopped["error_type"] is None
        )
        assert row["driver_origin"] == dict(
            created_at=worker._created_at, limits_id=id(worker._limits), preserved=True
        )
        before, after = row["before_expiry"], row["after_stop"]
        assert before["rollback_live"] is True and after["rollback_live"] is False
        assert after["previous_serving"] is None
        assert (
            before["serving"]["generation"] == 1
            and after["serving"]["generation"] == 1 + stopped["purges"]
        )
        assert before["serving"]["actor_version"] == after["serving"]["actor_version"] == "c0"
        assert (
            before["previous_serving"]["actor_version"] == "a0"
            and before["previous_serving"]["generation"] == 0
        )
        assert (
            before["serving"]["configuration"]
            == before["previous_serving"]["configuration"]
            == after["serving"]["configuration"]
            == dict(cache_ttl_ticks=20, max_cache_entries=2)
        )
        assert before["serving"]["model_state"] == after["serving"]["model_state"]
        current_cache, old_cache = before["serving"]["cache"], before["previous_serving"]["cache"]
        assert (
            len(current_cache) == len(old_cache) == 1 and current_cache.keys() == old_cache.keys()
        )
        assert next(iter(current_cache.values())) == dict(
            expires_at=28, prediction=row["prediction_c"]
        )
        assert next(iter(old_cache.values())) == dict(expires_at=20, prediction=row["prediction_a"])
        assert row["prediction_a"] != row["prediction_c"]
        assert (
            before["auxiliary_started_at"] == row["original_auxiliary_anchor"] == 0.0
            and after["auxiliary_started_at"] is None
        )
        assert (
            after["previous_owned_metadata"]
            == after["previous_owned_cache"]
            == after["serving"]["metadata"]
            == after["serving"]["cache"]
            == {}
        )
        assert after["prepared_bundle_metadata"] == after["prepared_bundle_cache"] == {}
        assert (
            before["pending_promotions"] == after["pending_promotions"] == 0
            and before["pending_checkpoints"] == 1
            and after["pending_checkpoints"] == 0
        )
        assert (
            before["owned_copies"]["observed_retained_bytes"] == 512
            and after["owned_copies"]["observed_retained_bytes"] == 0
        )
        assert (
            before["owned_copies"]["charged_bytes"] == after["owned_copies"]["charged_bytes"] == 656
        )
        for key in ("budget", "sharing", "registry"):
            assert before[key] == after[key]
        assert {k: v for k, v in before["managed"].items() if k != "revoked_keys"} == {
            k: v for k, v in after["managed"].items() if k != "revoked_keys"
        }
        assert before["managed"]["revoked_keys"] == [] and after["managed"]["revoked_keys"] == [
            [f"e{i}", f"s{i}"] for i in range(3)
        ]
        for key, count in (("old_cursor", 1), ("new_cursor", 2)):
            old, new = before[key], after[key]
            assert len(old["experiences"]) == len(old["labels"]) == 3
            assert new["experiences"] == new["labels"] == [] and new["applied"] == old["applied"]
            assert (
                len(new["applied"]) == new["completed_updates"] == count
                and new["last_tick"] == 8
                and new["stopped"] is False
            )
            assert new["erased"] == [
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
        assert all(
            after[key] == before[key] + stopped["purges"]
            for key in ("old_revision", "new_revision")
        )
        assert [
            (
                v["phase"],
                v["result"]["cache_hit"],
                v["result"]["generation"],
                v["result"]["actor_version"],
            )
            for v in row["cache_observations"]
        ] == [
            ("prime", False, 0, "a0"),
            ("handoff_hit", True, 0, "a0"),
            ("prepared_hit", True, 0, "a0"),
            ("committed_miss", False, 1, "c0"),
            ("committed_hit", True, 1, "c0"),
        ]
        for v in row["cache_observations"]:
            assert (
                v["result"]["prediction"]
                == row["prediction_c" if v["phase"].startswith("committed") else "prediction_a"]
            )
        for key in ("deadline_refusals", "terminal_refusals"):
            assert [v["phase"] for v in row[key]] == [
                "cached_hit",
                "poison_cache",
                "serving_snapshot",
                "promotion_commit",
                "checkpoint_restore",
                "fresh_prepare",
                "rollback",
            ]
            for v in row[key]:
                assert (
                    v["complete_state_preserved"]
                    and v["poison_copy_attempts"] == 0
                    and v["feature_digest_calls"] == 5
                    and v["build_calls"] == 4
                )
                assert v["quota_precedes_deadline"] is False
                assert v["consumed_ticket_precedes_deadline"] is (
                    key == "deadline_refusals" and v["phase"] == "promotion_commit"
                )
                expected = (
                    "promotion ticket is foreign, forged, discarded or consumed"
                    if v["phase"] == "promotion_commit"
                    else "declared payload retention has expired; cleanup required"
                    if key == "deadline_refusals"
                    else "checkpoint is foreign, copied, discarded or already used"
                    if v["phase"] == "checkpoint_restore"
                    else "rollback receipt is foreign, stale or consumed"
                    if v["phase"] == "rollback"
                    else "retention driver terminal; raw admission/access is closed"
                )
                assert v["message"] == expected
        codec = CoupledLearnerCheckpointCodec()
        independent = model(row["family"]).snapshot_state()
        for count, key in ((2, "actor_a_utf8"), (3, "captured_b_utf8"), (4, "continued_c_utf8")):
            independent = advance(advance(independent)) if count == 2 else advance(independent)
            raw = row[key].encode()
            decoded = codec.decode(
                raw,
                expected_sha256=sha256(raw).hexdigest(),
                binding=example.BINDING,
                limits=example.LIMITS,
            )
            assert (
                codec.encode(
                    replace(decoded, native_state=independent),
                    binding=example.BINDING,
                    limits=example.LIMITS,
                )
                == raw
            )
            assert json.loads(raw)["policy"] == row["policy"]
            if row["family"] != "pc":
                assert json.loads(raw)["native"]["wake_state"]["wake_updates"] == count
        for native, key in (
            (previous.learner.snapshot_state(), "actor_a_utf8"),
            (pending.learner.snapshot_state(), "continued_c_utf8"),
            (runtime.actor._slot.bundle.learner.snapshot_state(), "continued_c_utf8"),
        ):
            raw = row[key].encode()
            decoded = codec.decode(
                raw,
                expected_sha256=sha256(raw).hexdigest(),
                binding=example.BINDING,
                limits=example.LIMITS,
            )
            assert (
                codec.encode(
                    replace(decoded, native_state=native),
                    binding=example.BINDING,
                    limits=example.LIMITS,
                )
                == raw
            )
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3):
        for key in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            assert (
                json.loads(report["cases"][offset][key])["native"]["arrays"]
                == json.loads(report["cases"][offset + 1][key])["native"]["arrays"]
            )
