"""Managed cache/promotion expiry boundaries with synthetic coupled native ports."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any
import json
import sys

import numpy as np
import pytest

import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as basic
import src.adapters.coupled_payload_copies as copies
import src.adapters.coupled_managed_serving as example
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.payload_ownership import PayloadOwnershipBusy
from src.app.promotion_guard_evaluation import PromotionGuardEvaluator
from src.core.actor_ports import CandidateState, VersionedState
from src.core.promotion_guard import GuardBatch
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot, advance


class Poison:
    def __init__(self):
        self.copy_attempts = 0

    def __deepcopy__(self, memo):
        self.copy_attempts += 1
        raise AssertionError("unavailable guard/cache payload was copied")


@pytest.mark.parametrize(
    "bad", ["train", "outer_selection", "final_test", "future", "overlap", "training_overlap"]
)
def test_should_refuse_guard_metadata_before_any_payload_state_or_model_callback(bad):
    from test_promotion_guard_evaluation import policy

    payload = Poison()
    calls: list[str] = []

    def forbidden(*args):
        calls.append("callback")
        raise AssertionError("metadata gate reached state/model callback")

    evaluator = PromotionGuardEvaluator(
        build_learner=forbidden,
        utility=forbidden,
        actions=forbidden,
        state_valid=forbidden,
        prediction_valid=forbidden,
        resource_bytes=forbidden,
        state_digest=forbidden,
        clock=forbidden,
    )
    new = GuardBatch("new", (("guard-new", "new"),), 7, payload, payload, "inner_guard")
    old = GuardBatch("old", (("guard-old", "old"),), 7, payload, payload, "inner_guard")
    training: frozenset[tuple[str, str]] = frozenset()
    if bad in {"train", "outer_selection", "final_test"}:
        new = replace(new, role=bad)
    elif bad == "future":
        new = replace(new, labels_arrived_at=8)
    elif bad == "overlap":
        old = replace(old, sample_keys=new.sample_keys)
    else:
        training = frozenset(new.sample_keys)
    with pytest.raises(ValueError):
        evaluator.evaluate(
            policy(),
            VersionedState("a", payload),
            CandidateState("a", "c", payload, 2, 0, 0),
            new,
            old,
            now=7,
            training_ids=training,
        )
    assert payload.copy_attempts == 0 and calls == []


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("operation", ["cached", "poison", "snapshot", "commit", "prepare"])
def test_should_refuse_due_or_stopped_cached_serving_and_promotion_before_copy_hash_or_model(
    monkeypatch, terminal, operation
):
    from test_retention_expiry import promotion_fixture
    from test_serving_promotion import prepare
    from test_promotion_guard_evaluation import guards

    actor, runtime, control, wall, driver = promotion_fixture()
    closed = [False]
    calls = dict(hash=0, model=0, build=0)
    original_hash, original_build = actor._feature_digest, control._build

    def feature_hash(features):
        if closed[0]:
            calls["hash"] += 1
            raise AssertionError("unavailable serving input was hashed")
        return original_hash(features)

    def build(state):
        if closed[0]:
            calls["build"] += 1
            raise AssertionError("unavailable promotion built model")
        return original_build(state)

    monkeypatch.setattr(actor, "_feature_digest", feature_hash)
    monkeypatch.setattr(control, "_build", build)
    monkeypatch.setattr(control._evaluator, "_build", build)
    first = actor.serve([0], now=0)
    ticket = prepare(control, runtime)
    pending = control._pending[ticket].bundle
    if operation == "prepare":
        control.discard(ticket)
    original_actor = list(actor._slot.bundle.learner.state)
    original_candidate = list(runtime._candidate.state)
    if terminal:
        assert driver.stop()
    else:
        wall.value = 2.0
    poison = Poison()

    def forbidden(*args, **kwargs):
        calls["model"] += 1
        raise AssertionError("closed serving used a model method")

    for learner in (actor._slot.bundle.learner, runtime._candidate, pending.learner):
        monkeypatch.setattr(learner, "predict", forbidden)
        monkeypatch.setattr(learner, "snapshot_state", forbidden)
    closed[0] = True
    new, old = guards()
    action = {
        "cached": lambda: actor.serve([0], now=1),
        "poison": lambda: actor.serve(poison, now=1),
        "snapshot": actor.serving_snapshot,
        "commit": lambda: control.commit(runtime, ticket),
        "prepare": lambda: control.prepare(runtime, new, old, now=2, training_ids=frozenset()),
    }[operation]
    try:
        with pytest.raises(ValueError, match="expired|terminal|retention|ticket"):
            action()
        assert not first.cache_hit and first.prediction == [0.5]
        assert calls == dict(hash=0, model=0, build=0) and poison.copy_attempts == 0
        assert actor._slot.bundle.learner.state == original_actor
        assert runtime._candidate.state == original_candidate
        assert len(control._pending) == (0 if terminal or operation == "prepare" else 1)
        assert len(actor._slot.bundle.cache) == (0 if terminal else 1)
    finally:
        assert driver.stop()
    assert not driver.snapshot().alive


def test_should_describe_managed_serving_without_native_or_worker_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    for name in ("_model", "CoupledLearner", "RetentionExpiryDriver", "ManagedCoupledBuilder"):
        monkeypatch.setattr(example, name, forbidden)
    value = example.describe_managed_serving_example()
    assert value["protocol_id"] == "coupled_managed_serving_expiry_engineering_v1"
    assert value["native_training_calls"] == 36
    assert value["serving_configuration"] == dict(cache_ttl_ticks=20, max_cache_entries=2)
    assert value["retention_policy"] == dict(
        max_lifetime_ingress_bytes=192,
        max_retention_ticks=100,
        holders=dict(max_live_holders=5, max_lifetime_enrollments=5),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=dict(max_lifetime_owned_bytes=640),
        max_retention_seconds=2.0,
    )
    assert value["prediction_calls"] == 54 and value["guard_evaluations"] == 6
    assert [value["guard_batches"][key]["role"] for key in ("new", "old")] == ["inner_guard"] * 2
    assert [value["guard_batches"][key]["sample_keys"] for key in ("new", "old")] == [
        [["guard-new", "g0"]],
        [["guard-old", "g0"]],
    ]


@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_managed_serving_mode_without_other_work(monkeypatch, capsys, mode):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[str] = []

    def selected():
        calls.append(mode)
        return {"fixed": ("managed-serving", mode)}

    monkeypatch.setattr(cli, f"{mode}_managed_serving_example", selected)
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", "--managed-serving-expiry"])
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out) == {"fixed": ["managed-serving", mode]}
    assert calls == [mode]


@pytest.mark.parametrize("other", ["--managed-copies", "--elapsed-retention", "--cache-ttl"])
def test_should_refuse_conflicting_or_unfrozen_serving_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--managed-serving-expiry", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_expire_six_real_cache_promotion_transactions_with_owned_native_state(monkeypatch):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    drivers: list[Any] = []
    original_resources: list[tuple[Any, ...]] = []
    promotions: list[Any] = []
    checkpoints: list[Any] = []
    prepared_bundles: list[Any] = []
    original_init, actual_driver = FakeLearner.__init__, example.RetentionExpiryDriver
    actual_promotion = example.ServingPromotionController
    actual_checkpoint = example.CandidateCheckpointController
    starts: list[dict[str, int]] = []

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
        worker = actual_driver(life, **kwargs)
        drivers.append(worker)
        owner = life._owner
        runtime = owner._shared._runtime
        original_resources.append(
            (
                owner,
                life,
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
                worker._created_at,
            )
        )
        observed = dict(busy=0, wake=0, wait=0)
        starts.append(observed)
        actual_poll, actual_wake, actual_wait = worker.poll_once, worker.wake, worker.wait_for_purge

        def poll():
            if not observed["busy"]:
                observed["busy"] = 1
                raise PayloadOwnershipBusy("deterministic startup contention")
            return actual_poll()

        def wake():
            observed["wake"] += 1
            actual_wake()

        def wait(seconds):
            assert observed["wake"] == 1 and observed["wait"] == 0 and seconds == 2.0
            observed["wait"] += 1
            return actual_wait(seconds)

        monkeypatch.setattr(worker, "poll_once", poll)
        monkeypatch.setattr(worker, "wake", wake)
        monkeypatch.setattr(worker, "wait_for_purge", wait)
        return worker

    def promotion(*args, **kwargs):
        control = actual_promotion(*args, **kwargs)
        promotions.append(control)
        original_prepare = control.prepare

        def prepare(*args, **kwargs):
            ticket = original_prepare(*args, **kwargs)
            prepared_bundles.append(control._pending[ticket].bundle)
            return ticket

        monkeypatch.setattr(control, "prepare", prepare)
        return control

    def checkpoint(*args, **kwargs):
        control = actual_checkpoint(*args, **kwargs)
        checkpoints.append(control)
        return control

    monkeypatch.setattr(FakeLearner, "__init__", initialize)
    monkeypatch.setattr(example, "_model", model)
    monkeypatch.setattr(example, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(example, "RetentionExpiryDriver", driver)
    monkeypatch.setattr(example, "ServingPromotionController", promotion)
    monkeypatch.setattr(example, "CandidateCheckpointController", checkpoint)
    monkeypatch.setattr(basic, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(copies, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    result = example.run_managed_serving_example()
    assert (
        result["managed_serving_expiry_verified"] is result["pc_neutral_parameters_exact"] is True
    )
    assert result["full_managed_process_recovery_verified"] is False
    assert len(result["cases"]) == len(drivers) == 6 and FakeLearner.train_calls == 36
    assert starts == [dict(busy=1, wake=1, wait=1)] * 6
    assert len(promotions) == len(checkpoints) == len(prepared_bundles) == 6
    for row, worker, promotion_control, checkpoint_control, pending, original in zip(
        result["cases"],
        drivers,
        promotions,
        checkpoints,
        prepared_bundles,
        original_resources,
        strict=True,
    ):
        life, owner = worker._lifecycle, worker._lifecycle._owner
        runtime, gate = owner._shared._runtime, owner._shared._sharing
        assert all(
            left is right
            for left, right in zip(
                (
                    owner,
                    life,
                    runtime.actor,
                    runtime._budget,
                    runtime._inbox._clock,
                    runtime._budget.clock,
                    gate,
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
        assert (
            promotion_control._build
            is promotion_control._evaluator._build
            is checkpoint_control._build
        )
        assert promotion_control._pending == checkpoint_control._pending == {}
        assert pending.metadata == pending.cache == {}
        assert runtime.actor._slot.bundle.metadata == runtime.actor._slot.bundle.cache == {}
        assert checkpoint_control._attempts == len(checkpoint_control._models) == 1
        assert checkpoint_control._build.build_calls == 4
        assert worker.snapshot().state == "stopped" and not worker.snapshot().alive
        assert worker._thread is not None and not worker._thread.is_alive()
        assert row["actor_unchanged"] and row["complete_continuation_exact"]
        assert worker._lifecycle._copy_budget._charged == 640
        assert worker._lifecycle.admitted_payload_bytes == 144
        assert row["identity"]["original"] == row["identity"]["current"]
        assert row["driver_origin"] == dict(
            created_at=worker._created_at, limits_id=id(worker._limits), preserved=True
        )
        assert [v["phase"] for v in row["renewal_refusals"]] == ["restart", "replacement"]
        ready, purged, stopped = (
            row[key] for key in ("driver_ready", "driver_after_purge", "driver_final")
        )
        assert (
            ready["state"] == "ready"
            and ready["polls"] == ready["purges"] == ready["cleanup_attempts"] == 0
        )
        assert (
            ready["limits"]
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
        assert (
            stopped["state"] == "stopped"
            and stopped["alive"] is stopped["pending_cleanup"] is False
        )
        assert ready["error_type"] is purged["error_type"] is stopped["error_type"] is None
        assert (
            row["pending_promotion_committed"] is False and row["original_auxiliary_anchor"] == 0.0
        )
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
            ("cleanup", 640, 0),
        ]
        assert all(v["limits"] == {"max_lifetime_owned_bytes": 640} for v in row["copy_accounting"])
        observations = row["cache_observations"]
        assert [(v["phase"], v["result"]["cache_hit"]) for v in observations] == [
            ("prime", False),
            ("handoff_hit", True),
            ("prepared_hit", True),
        ]
        assert all(
            v["result"]["actor_version"] == "a0"
            and v["result"]["generation"] == 0
            and v["result"]["prediction"] == row["prediction_a"]
            and v["result"]["metadata"] == {"raw": [0.25, 0.5, 0.75], "source": "synthetic-local"}
            for v in observations
        )
        before, after = row["before_expiry"], row["after_stop"]
        assert (
            before["owned_copies"]["observed_retained_bytes"] == 496
            and after["owned_copies"]["observed_retained_bytes"] == 0
        )
        assert (
            before["owned_copies"]["charged_bytes"] == after["owned_copies"]["charged_bytes"] == 640
        )
        assert before["auxiliary_started_at"] == 0.0 and after["auxiliary_started_at"] is None
        for key in ("budget", "sharing", "registry"):
            assert before[key] == after[key]
        assert len(before["registry"]["holders"]) == before["registry"]["total_enrollments"] == 5
        assert {k: v for k, v in before["managed"].items() if k != "revoked_keys"} == {
            k: v for k, v in after["managed"].items() if k != "revoked_keys"
        }
        assert before["managed"]["revoked_keys"] == [] and after["managed"]["revoked_keys"] == [
            [f"e{i}", f"s{i}"] for i in range(3)
        ]
        for name, completed in (("old_cursor", 1), ("new_cursor", 2)):
            previous, current = before[name], after[name]
            assert len(previous["experiences"]) == len(previous["labels"]) == 3
            assert current["experiences"] == current["labels"] == []
            assert (
                current["applied"] == previous["applied"] and len(current["applied"]) == completed
            )
            assert (
                current["completed_updates"] == completed
                and current["last_tick"] == 8
                and current["stopped"] is False
            )
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
        assert all(
            after[key] == before[key] + stopped["purges"]
            for key in ("old_revision", "new_revision")
        )
        assert before["pending_checkpoints"] == before["pending_promotions"] == 1
        assert after["pending_checkpoints"] == after["pending_promotions"] == 0
        assert (
            before["pending_promotion_metadata"]
            == before["serving"]["metadata"]
            == {"raw": [0.25, 0.5, 0.75], "source": "synthetic-local"}
        )
        assert (
            after["pending_promotion_metadata"]
            == after["pending_promotion_cache"]
            == after["serving"]["metadata"]
            == after["serving"]["cache"]
            == {}
        )
        assert (
            before["serving"]["generation"] == 0
            and after["serving"]["generation"] == stopped["purges"]
        )
        assert before["serving"]["model_state"] == after["serving"]["model_state"]
        assert (
            before["serving"]["configuration"]
            == after["serving"]["configuration"]
            == dict(cache_ttl_ticks=20, max_cache_entries=2)
        )
        assert before["serving"]["last_cache_tick"] == after["serving"]["last_cache_tick"] == 8
        assert len(before["serving"]["cache"]) == 1
        assert next(iter(before["serving"]["cache"].values())) == dict(
            expires_at=20, prediction=row["prediction_a"]
        )
        assert row["guard_callback_counts"] == dict(
            feature_digest=3, state_valid=2, prediction_valid=4, utility=4, actions=4, resource=1
        )
        report = row["guard_report"]
        assert report["decision"] == dict(accepted=True, reasons=[])
        assert report["label_arrivals"] == [7, 7] and report["assessed_at"] == 8
        assert report["new_guard_task_id"] == "new" and report["old_guard_task_id"] == "old"
        assert report["new_guard_keys"] == [["guard-new", "g0"]] and report["old_guard_keys"] == [
            ["guard-old", "g0"]
        ]
        assert report["training_ids"] == [[f"e{i}", f"s{i}"] for i in range(3)]
        assert report["evidence"]["candidate_max_prediction_seconds"] == 0.0
        assert report["evidence"]["numerically_valid"] is report["evidence"]["actions_safe"] is True
        assert runtime._budget.updates_completed == gate.snapshot().admitted_updates == 2
        for key in ("deadline_refusals", "terminal_refusals"):
            assert [v["phase"] for v in row[key]] == [
                "cached_hit",
                "poison_cache",
                "serving_snapshot",
                "promotion_commit",
                "checkpoint_restore",
                "fresh_prepare",
            ]
            for refusal in row[key]:
                assert refusal["complete_state_preserved"] and refusal["poison_copy_attempts"] == 0
                assert refusal["feature_digest_calls"] == 3 and refusal["build_calls"] == 4
                assert refusal["quota_precedes_deadline"] is (
                    key == "deadline_refusals" and refusal["phase"] == "fresh_prepare"
                )
                expected = (
                    "pending promotion ticket quota exhausted; discard obsolete tickets"
                    if refusal["quota_precedes_deadline"]
                    else "declared payload retention has expired; cleanup required"
                    if key == "deadline_refusals"
                    else "promotion ticket is foreign, forged, discarded or consumed"
                    if refusal["phase"] == "promotion_commit"
                    else "checkpoint is foreign, copied, discarded or already used"
                    if refusal["phase"] == "checkpoint_restore"
                    else "retention driver terminal; raw admission/access is closed"
                )
                assert refusal["message"] == expected
        actor, captured, final = (
            json.loads(row[key]) for key in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8")
        )
        assert actor["policy"] == captured["policy"] == final["policy"] == row["policy"]
        assert (
            row["policy"]["learning_rate"] == 0.025
            and row["policy"]["inference_steps"] == 3
            and row["policy"]["inference_learning_rate"] == 0.07
        )
        assert row["policy"]["inference_policy"] == (
            None
            if row["mode"] == "fixed"
            else dict(min_steps=1, gradient_tolerance=1e6, relative_energy_tolerance=None)
        )
        codec = CoupledLearnerCheckpointCodec()
        independent = model(row["family"]).snapshot_state()
        for expected_count, key in (
            (2, "actor_a_utf8"),
            (3, "captured_b_utf8"),
            (4, "continued_c_utf8"),
        ):
            independent = (
                advance(independent) if expected_count > 2 else advance(advance(independent))
            )
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
                ).decode()
                == row[key]
            )
        raw_c = row["continued_c_utf8"].encode()
        full_c = codec.decode(
            raw_c,
            expected_sha256=sha256(raw_c).hexdigest(),
            binding=example.BINDING,
            limits=example.LIMITS,
        )
        assert (
            codec.encode(
                replace(full_c, native_state=pending.learner.snapshot_state()),
                binding=example.BINDING,
                limits=example.LIMITS,
            ).decode()
            == row["continued_c_utf8"]
        )
        if row["family"] != "pc":
            assert [
                v["native"]["wake_state"]["wake_updates"] for v in (actor, captured, final)
            ] == [2, 3, 4]
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
    for offset in (0, 3):
        for stage in ("actor_a_utf8", "captured_b_utf8", "continued_c_utf8"):
            assert (
                json.loads(result["cases"][offset][stage])["native"]["arrays"]
                == json.loads(result["cases"][offset + 1][stage])["native"]["arrays"]
            )
