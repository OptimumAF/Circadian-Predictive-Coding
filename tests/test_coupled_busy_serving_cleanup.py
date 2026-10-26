"""Busy terminal retention preserves ownership until explicit cleanup succeeds."""

from dataclasses import asdict, replace
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
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


def test_should_keep_original_tokens_and_raw_bundles_while_terminal_cleanup_is_busy():
    from test_retention_expiry import promotion_fixture
    from test_serving_promotion import prepare
    from test_candidate_checkpoint import controller

    actor, runtime, promotion, wall, driver = promotion_fixture()
    shared = driver._lifecycle._owner._shared
    shared._sharing.pause()
    checkpoint = controller(shared)
    token = checkpoint.capture()
    actor.serve([0], now=0)
    receipt = promotion.commit(runtime, prepare(promotion, runtime))
    actor.serve([0], now=1)
    previous, current = actor._slot.previous, actor._slot.bundle
    assert previous is not None
    before = actor._slot
    wall.value = 2.0
    try:
        with runtime._exclusive():
            assert driver.poll_once().outcome == "busy"
            assert driver.stop() is False
        pending = driver.snapshot()
        assert pending.state == "stopped" and pending.pending_cleanup and not pending.alive
        assert pending.purges == 0 and 1 <= pending.cleanup_attempts <= pending.limits.max_polls + 2
        assert actor._slot is before and previous.cache and current.cache
        assert (
            token in checkpoint._pending
            and promotion._latest is not None
            and promotion._latest[0] is receipt
        )
        for action in (
            lambda: actor.serve([0], now=1),
            actor.serving_snapshot,
            lambda: checkpoint.restore(token),
            lambda: promotion.rollback(receipt),
        ):
            with pytest.raises(ValueError, match="terminal|retention"):
                action()
            assert actor._slot is before and token in checkpoint._pending
        assert driver.finish_cleanup().outcome == "purged"
        assert (
            actor._slot.previous is None and promotion._latest is None and checkpoint._pending == {}
        )
        assert previous.metadata == previous.cache == current.metadata == current.cache == {}
        assert current.learner.state == [0.75, 0.8] and previous.learner.state == [0.5, 0.8]
        assert driver.snapshot().purges == 1 and runtime._budget.updates_completed == 0
        with pytest.raises(ValueError, match="rollback receipt"):
            promotion.rollback(receipt)
        with pytest.raises(ValueError, match="unfinished"):
            driver.finish_cleanup()
    finally:
        assert driver.stop()


def test_should_describe_busy_variant_without_operational_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    for name in ("_model", "CoupledLearner", "RetentionExpiryDriver", "ManagedCoupledBuilder"):
        monkeypatch.setattr(example, name, forbidden)
    old = example.describe_committed_serving_example()
    value = example.describe_busy_committed_serving_example()
    assert value["protocol_id"] == "coupled_busy_committed_serving_cleanup_engineering_v1"
    assert value["prediction_calls"] == 60 and value["serving_requests"] == 66
    for key in (
        "retention_policy",
        "driver_limits",
        "promotion_policy",
        "guard_batches",
        "serving_configuration",
    ):
        assert value[key] == old[key]


@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_busy_variant_without_other_work(monkeypatch, capsys, mode):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[str] = []

    def selected():
        calls.append(mode)
        return {"mode": (mode, "busy")}

    monkeypatch.setattr(cli, f"{mode}_busy_committed_serving_example", selected)
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", "--busy-serving-cleanup"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"mode": [mode, "busy"]}
    assert calls == [mode]


@pytest.mark.parametrize(
    "other", ["--committed-serving-expiry", "--managed-serving-expiry", "--join-timeout"]
)
def test_should_refuse_conflicting_or_unfrozen_busy_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--busy-serving-cleanup", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "committed,busy",
    [(False, True), (True, 1), (True, None), (True, "true"), (True, np.bool_(True))],
)
def test_should_refuse_unsupported_private_busy_modes_before_models(monkeypatch, committed, busy):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid mode constructed model")

    monkeypatch.setattr(example, "_model", forbidden)
    with pytest.raises(ValueError):
        example._case("pc", "fixed", committed=committed, busy_stop=busy)


def test_should_stop_six_actual_workers_busy_then_finish_without_raw_or_native_resurrection(
    monkeypatch,
):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    drivers: list[Any] = []
    stopped: list[Any] = []
    finished: list[Any] = []
    promotions: list[Any] = []
    checkpoints: list[Any] = []
    previous_bundles: list[Any] = []
    originals: list[tuple[Any, ...]] = []
    original_candidates: list[Any] = []
    original_init = FakeLearner.__init__
    original_predict, original_snapshot = FakeLearner.predict, FakeLearner.snapshot_state
    native_work = dict(predict=0, snapshot=0)
    actual_driver, actual_promotion, actual_checkpoint = (
        example.RetentionExpiryDriver,
        example.ServingPromotionController,
        example.CandidateCheckpointController,
    )

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

    def predict(self, features):
        native_work["predict"] += 1
        return original_predict(self, features)

    def snapshot_state(self):
        native_work["snapshot"] += 1
        return original_snapshot(self)

    def driver(life, **kwargs):
        worker = actual_driver(life, **kwargs)
        drivers.append(worker)
        owner = life._owner
        runtime = owner._shared._runtime
        original_candidates.append(runtime._candidate)
        originals.append(
            (
                life,
                owner,
                runtime.actor,
                runtime._budget,
                runtime._inbox._clock,
                runtime._budget.clock,
                owner._shared._sharing,
                life._copy_budget,
                worker._limits,
                worker._created_at,
            )
        )
        stop, finish = worker.stop, worker.finish_cleanup

        def observe_stop():
            before_work = (FakeLearner.train_calls, dict(native_work))
            result = stop()
            if not result:
                assert worker._thread is not None and not worker._thread.is_alive()
                assert before_work == (FakeLearner.train_calls, native_work)
                stopped.append(worker.snapshot())
            return result

        def observe_finish():
            before_work = (FakeLearner.train_calls, dict(native_work))
            result = finish()
            assert before_work == (FakeLearner.train_calls, native_work)
            finished.append(result)
            return result

        monkeypatch.setattr(worker, "stop", observe_stop)
        monkeypatch.setattr(worker, "finish_cleanup", observe_finish)
        return worker

    def promotion(*args, **kwargs):
        control = actual_promotion(*args, **kwargs)
        promotions.append(control)
        commit = control.commit

        def publish(*args, **kwargs):
            previous = control._actor._slot.bundle
            receipt = commit(*args, **kwargs)
            previous_bundles.append(previous)
            return receipt

        monkeypatch.setattr(control, "commit", publish)
        return control

    def checkpoint(*args, **kwargs):
        control = actual_checkpoint(*args, **kwargs)
        checkpoints.append(control)
        return control

    monkeypatch.setattr(FakeLearner, "__init__", initialize)
    monkeypatch.setattr(FakeLearner, "predict", predict)
    monkeypatch.setattr(FakeLearner, "snapshot_state", snapshot_state)
    monkeypatch.setattr(example, "_model", model)
    monkeypatch.setattr(example, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(example, "RetentionExpiryDriver", driver)
    monkeypatch.setattr(example, "ServingPromotionController", promotion)
    monkeypatch.setattr(example, "CandidateCheckpointController", checkpoint)
    for module in (basic, copies, callbacks_module):
        monkeypatch.setattr(module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    report = example.run_busy_committed_serving_example()
    assert len(report["cases"]) == len(drivers) == len(stopped) == len(finished) == 6
    assert FakeLearner.train_calls == 36 and all(v.outcome == "purged" for v in finished)
    assert report["busy_cleanup_verified"] is report["pc_neutral_parameters_exact"] is True
    for (
        row,
        worker,
        promotion_control,
        checkpoint_control,
        previous,
        original,
        old_candidate,
        busy_driver,
    ) in zip(
        report["cases"],
        drivers,
        promotions,
        checkpoints,
        previous_bundles,
        originals,
        original_candidates,
        stopped,
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
                    life._copy_budget,
                    worker._limits,
                ),
                original[:-1],
                strict=True,
            )
        )
        assert worker._created_at == original[-1] and life._retention_driver is worker
        assert (
            worker.snapshot().state == "stopped"
            and worker.snapshot().purges == 1
            and not worker.snapshot().alive
        )
        assert promotion_control._latest is None and checkpoint_control._pending == {}
        assert (
            previous.metadata
            == previous.cache
            == runtime.actor._slot.bundle.metadata
            == runtime.actor._slot.bundle.cache
            == {}
        )
        assert life._copy_budget._charged == 656 and life.admitted_payload_bytes == 144
        assert row["automatic_purge_verified"] is False and row["busy_cleanup_verified"] is True
        assert row["manual_cleanup_verified"] is True and row["busy_stop_returned"] is False
        assert row["busy_poll"]["outcome"] == "busy"
        assert row["driver_busy_stopped"] == asdict(busy_driver)
        assert (
            busy_driver.state == "stopped" and busy_driver.pending_cleanup and not busy_driver.alive
        )
        assert busy_driver.purges == 0 and busy_driver.error_type is None
        assert 1 <= busy_driver.polls <= 4 and 1 <= busy_driver.cleanup_attempts < 6
        final = worker.snapshot()
        assert final.purges == 1 and final.cleanup_attempts == busy_driver.cleanup_attempts + 1 <= 6
        assert final.polls == busy_driver.polls and not final.pending_cleanup
        assert (
            row["driver_ready"]["limits"]
            == row["driver_final"]["limits"]
            == asdict(busy_driver.limits)
        )
        pending = row["pending_cleanup"]
        before, after = row["before_expiry"], row["after_stop"]
        assert pending["driver"] == row["driver_busy_stopped"]
        assert (
            pending["generation"] == 1 and pending["previous_present"] and pending["rollback_live"]
        )
        assert pending["pending_checkpoints"] == 1 and pending["pending_promotions"] == 0
        assert pending["raw_counts"] == dict(
            old_sources=3, old_labels=3, new_sources=3, new_labels=3
        )
        assert pending["cache_entries"] == dict(current=1, previous=1)
        assert pending["metadata_bytes"] == dict(current=24, previous=24)
        assert pending["owned_copies"] == dict(
            limits={"max_lifetime_owned_bytes": 656}, charged_bytes=656, observed_retained_bytes=512
        )
        assert pending["identity"] == row["identity"]["original"] == row["identity"]["current"]
        for key in ("budget", "sharing", "registry"):
            assert pending[key] == before[key] == after[key]
        assert pending["old"]["revision"] == before["old_revision"]
        assert pending["new"]["revision"] == before["new_revision"]
        assert (
            after["old_revision"] == before["old_revision"] + 1
            and after["new_revision"] == before["new_revision"] + 1
        )
        assert after["serving"]["generation"] == 2 and after["previous_serving"] is None
        assert (
            pending["preparation_attempts"] == pending["prepared_models"] == 1
            and pending["build_calls"] == 4
        )
        assert [v["phase"] for v in row["pending_refusals"]] == [
            "cached_hit",
            "poison_cache",
            "serving_snapshot",
            "promotion_commit",
            "checkpoint_restore",
            "fresh_prepare",
            "rollback",
        ]
        for refusal in row["pending_refusals"]:
            assert refusal["complete_state_preserved"] and refusal["poison_copy_attempts"] == 0
            assert refusal["feature_digest_calls"] == 5 and refusal["build_calls"] == 4
            assert refusal["message"] == (
                "promotion ticket is foreign, forged, discarded or consumed"
                if refusal["phase"] == "promotion_commit"
                else "retention driver terminal; raw admission/access is closed"
            )
        terminal = {v["phase"]: v["message"] for v in row["terminal_refusals"]}
        assert (
            terminal["checkpoint_restore"]
            == "checkpoint is foreign, copied, discarded or already used"
        )
        assert terminal["rollback"] == "rollback receipt is foreign, stale or consumed"
        assert row["finish_cleanup"]["outcome"] == "purged"
        cleanup = row["finish_cleanup"]["report"]
        assert cleanup["reason"] == "expired" and cleanup["inboxes_cleared"] == 2
        assert cleanup["checkpoints_invalidated"] == 1 and cleanup["promotions_invalidated"] == 0
        assert cleanup["revoked_keys"] == [[f"e{i}", f"s{i}"] for i in range(3)]
        assert [v["phase"] for v in row["copy_accounting"]][-3:] == [
            "committed_hit",
            "busy_stop",
            "cleanup",
        ]
        assert [(v["charged_bytes"], v["observed_retained_bytes"]) for v in row["copy_accounting"]][
            -3:
        ] == [(656, 512), (656, 512), (656, 0)]
        assert any(
            v["phase"] == "second_finish" and v["message"] == "no unfinished retention cleanup"
            for v in row["renewal_refusals"]
        )
        codec = CoupledLearnerCheckpointCodec()
        for learner, key, wakes in (
            (previous.learner, "actor_a_utf8", 2),
            (old_candidate, "captured_b_utf8", 3),
            (runtime._candidate, "continued_c_utf8", 4),
            (runtime.actor._slot.bundle.learner, "continued_c_utf8", 4),
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
                    replace(decoded, native_state=learner.snapshot_state()),
                    binding=example.BINDING,
                    limits=example.LIMITS,
                )
                == raw
            )
            if row["family"] != "pc":
                assert json.loads(raw)["native"]["wake_state"]["wake_updates"] == wakes
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
