"""One-shot erasure faults retain failure facts and require terminal cleanup."""

from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast
import json
import logging
import sys

import numpy as np
import pytest

import src.adapters.coupled_cleanup_failure as fault_port
import src.adapters.coupled_managed_serving as example
import src.adapters.coupled_checkpoint_callbacks as callbacks_module
import src.adapters.coupled_data_lifecycle as basic
import src.adapters.coupled_payload_copies as copies
from src.adapters.coupled_learners import CoupledLearner
from src.adapters.coupled_learner_checkpoint import CoupledLearnerCheckpointCodec
from src.app.managed_experience import ManagedExperienceOwner
from src.core.data_erasure import ReplayPayloadErasure
from src.core.data_retention import DataRetentionPolicy
from src.core.payload_bytes import PayloadCopyLimits
from src.core.payload_ownership import PayloadOwnershipLimits
from test_coupled_repeated_handoff import FakeLearner, FakeModel
from test_native_transfer_session import snapshot


def policy():
    return DataRetentionPolicy(
        192,
        100,
        PayloadOwnershipLimits(5, 5),
        max_checkpoint_pending=2,
        max_checkpoint_preparations=2,
        max_promotion_pending=1,
        owned_payload_copies=PayloadCopyLimits(656),
        max_retention_seconds=2.0,
    )


def test_should_bind_one_shot_exact_family_and_delegate_without_any_learner_operation(monkeypatch):
    learner = object.__new__(CoupledLearner)
    delegated: list[Any] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("erasure touched model or replaced delegate")

    def delegate(model):
        delegated.append(model)
        return ReplayPayloadErasure(0, 0, 0)

    for name in ("snapshot_state", "predict", "train_batch", "restore_state", "fork"):
        monkeypatch.setattr(CoupledLearner, name, forbidden)
    monkeypatch.setattr(fault_port, "erase_coupled_replay_payloads", delegate)
    erase = fault_port.OneShotCoupledEraseFailure()
    monkeypatch.setattr(fault_port, "erase_coupled_replay_payloads", forbidden)
    monkeypatch.setattr(fault_port, "CoupledLearner", object)
    with pytest.raises(RuntimeError, match="injected coupled replay erasure failure"):
        erase(learner)
    assert (erase.calls, erase.failures, erase.delegated_calls) == (1, 1, 0) and delegated == []
    assert erase(learner) == erase(learner) == ReplayPayloadErasure(0, 0, 0)
    assert delegated == [learner, learner] and vars(learner) == {}
    assert (erase.calls, erase.failures, erase.delegated_calls) == (3, 1, 2)
    for name in ("calls", "failures", "delegated_calls"):
        with pytest.raises(AttributeError):
            setattr(erase, name, 0)


@pytest.mark.parametrize("kind", ["opaque", "subclass", "snapshot", "bool"])
def test_should_refuse_wrong_family_before_spending_fault_or_delegating(monkeypatch, kind):
    class Derived(CoupledLearner):
        pass

    class Opaque:
        def __getattribute__(self, name):
            raise AssertionError("wrong-family erasure read opaque state")

    def forbidden(*args):
        raise AssertionError("wrong-family erasure delegated")

    monkeypatch.setattr(fault_port, "erase_coupled_replay_payloads", forbidden)
    erase = fault_port.OneShotCoupledEraseFailure()
    values = {
        "opaque": Opaque(),
        "subclass": object.__new__(Derived),
        "snapshot": snapshot(),
        "bool": False,
    }
    with pytest.raises(ValueError):
        erase(values[kind])
    assert (erase.calls, erase.failures, erase.delegated_calls) == (0, 0, 0)


def test_should_install_same_one_shot_callback_and_policy_at_lifecycle_birth(monkeypatch):
    owner = object.__new__(ManagedExperienceOwner)
    selected = policy()
    erase = fault_port.OneShotCoupledEraseFailure()
    sentinel = object()
    calls: list[Any] = []

    def compose(actual, **kwargs):
        calls.append((actual, kwargs))
        return sentinel

    monkeypatch.setattr(fault_port, "_make_coupled_elapsed_lifecycle", compose)
    assert (
        fault_port.make_failing_coupled_elapsed_lifecycle(
            owner, policy=selected, erase_failure=erase
        )
        is sentinel
    )
    assert calls == [(owner, dict(policy=selected, native_erase=erase))]
    assert erase.calls == 0


@pytest.mark.parametrize("kind", ["opaque", "subclass"])
def test_should_refuse_unsupported_fault_callback_before_owner_operations(monkeypatch, kind):
    class Derived(fault_port.OneShotCoupledEraseFailure):
        pass

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported callback reached owner")

    monkeypatch.setattr(fault_port, "_make_coupled_elapsed_lifecycle", forbidden)
    value = object() if kind == "opaque" else Derived()
    with pytest.raises(ValueError):
        fault_port.make_failing_coupled_elapsed_lifecycle(
            object.__new__(ManagedExperienceOwner),
            policy=policy(),
            erase_failure=cast(fault_port.OneShotCoupledEraseFailure, value),
        )


def test_should_describe_fault_without_models_callbacks_or_driver_operations(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("description performed operational work")

    for name in ("_model", "CoupledLearner", "RetentionExpiryDriver", "OneShotCoupledEraseFailure"):
        monkeypatch.setattr(example, name, forbidden)
    value = example.describe_cleanup_failure_example()
    assert value["protocol_id"] == "coupled_cleanup_failure_engineering_v1"
    assert value["native_training_calls"] == 36 and value["prediction_calls"] == 60


@pytest.mark.parametrize("mode", ["describe", "run"])
def test_should_route_fault_variant_without_other_work(monkeypatch, capsys, mode):
    import scripts.run_coupled_candidate_handoff as cli

    calls: list[str] = []

    def selected():
        calls.append(mode)
        return {"mode": (mode, "failure")}

    monkeypatch.setattr(cli, f"{mode}_cleanup_failure_example", selected)
    monkeypatch.setattr(sys, "argv", ["handoff", f"--{mode}", "--cleanup-failure"])
    assert cli.main() == 0 and json.loads(capsys.readouterr().out) == {"mode": [mode, "failure"]}
    assert calls == [mode]


@pytest.mark.parametrize(
    "other", ["--busy-serving-cleanup", "--committed-serving-expiry", "--failure-count"]
)
def test_should_refuse_conflicting_or_unfrozen_fault_controls(monkeypatch, other):
    import scripts.run_coupled_candidate_handoff as cli

    monkeypatch.setattr(sys, "argv", ["handoff", "--run", "--cleanup-failure", other])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "committed,busy,failure",
    [
        (False, False, True),
        (True, True, True),
        (True, False, 1),
        (True, False, None),
        (True, False, np.bool_(True)),
    ],
)
def test_should_refuse_unsupported_fault_modes_before_models(monkeypatch, committed, busy, failure):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid fault mode constructed model")

    monkeypatch.setattr(example, "_model", forbidden)
    with pytest.raises(ValueError):
        example._case("pc", "fixed", committed=committed, busy_stop=busy, cleanup_failure=failure)


def test_should_record_six_manual_failures_then_clean_raw_data_without_reopening_candidates(
    monkeypatch, caplog
):
    FakeLearner.instances, FakeLearner.train_calls = [], 0
    lives: list[Any] = []
    drivers: list[Any] = []
    failed: list[Any] = []
    finished: list[Any] = []
    original_candidates: list[Any] = []
    original_actors: list[Any] = []
    original_init = FakeLearner.__init__
    original_predict, original_snapshot = FakeLearner.predict, FakeLearner.snapshot_state
    work = dict(predict=0, snapshot=0)
    actual_factory, actual_driver = (
        example.make_failing_coupled_elapsed_lifecycle,
        example.RetentionExpiryDriver,
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
        state = CoupledLearnerCheckpointCodec().decode(raw, **kwargs)
        p = state.policy
        return FakeLearner(
            FakeModel(state.native_state),
            learning_rate=p.learning_rate,
            inference_steps=p.inference_steps,
            inference_learning_rate=p.inference_learning_rate,
            inference_policy=p.inference_policy,
        )

    def predict(self, features):
        work["predict"] += 1
        return original_predict(self, features)

    def snapshot_state(self):
        work["snapshot"] += 1
        return original_snapshot(self)

    def factory(owner, *, policy, erase_failure):
        life = actual_factory(owner, policy=policy, erase_failure=erase_failure)
        assert life._erase is erase_failure
        lives.append((life, erase_failure, life._copy_budget, life._wall_clock))
        original_candidates.append(owner._shared._runtime)
        original_actors.append(owner._shared._runtime.actor._slot.bundle.learner)
        return life

    def driver(life, **kwargs):
        worker = actual_driver(life, **kwargs)
        drivers.append(worker)
        poll, finish = worker.poll_once, worker.finish_cleanup

        def observe_poll():
            before = (FakeLearner.train_calls, dict(work))
            result = poll()
            if result.outcome == "failed":
                assert before == (FakeLearner.train_calls, work)
                failed.append(worker.snapshot())
            return result

        def observe_finish():
            before = (FakeLearner.train_calls, dict(work))
            result = finish()
            assert before == (FakeLearner.train_calls, work)
            finished.append(result)
            return result

        monkeypatch.setattr(worker, "poll_once", observe_poll)
        monkeypatch.setattr(worker, "finish_cleanup", observe_finish)
        return worker

    monkeypatch.setattr(FakeLearner, "__init__", initialize)
    monkeypatch.setattr(FakeLearner, "predict", predict)
    monkeypatch.setattr(FakeLearner, "snapshot_state", snapshot_state)
    monkeypatch.setattr(example, "_model", model)
    monkeypatch.setattr(example, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(example, "make_failing_coupled_elapsed_lifecycle", factory)
    monkeypatch.setattr(example, "RetentionExpiryDriver", driver)
    for module in (basic, copies, callbacks_module, fault_port):
        monkeypatch.setattr(module, "CoupledLearner", FakeLearner)
    monkeypatch.setattr(callbacks_module, "restore_coupled_learner", restore)
    with caplog.at_level(logging.ERROR, logger="src.app.retention_expiry"):
        report = example.run_cleanup_failure_example()
    assert (
        len(report["cases"]) == len(drivers) == len(failed) == len(finished) == 6
        and FakeLearner.train_calls == 36
    )
    errors = [
        r
        for r in caplog.records
        if r.name == "src.app.retention_expiry" and r.getMessage() == "retention_cleanup_failed"
    ]
    assert len(errors) == 6 and all(
        r.levelno == logging.ERROR
        and r.error_type == "RuntimeError"
        and r.polls == 1
        and r.exc_info is None
        for r in errors
    )
    assert "injected coupled replay erasure failure" not in caplog.text
    assert report["cleanup_failure_verified"] is report["pc_neutral_parameters_exact"] is True
    for row, worker, broken, (life, erase, copy_budget, clock), old_candidate, actor_a in zip(
        report["cases"], drivers, failed, lives, original_candidates, original_actors, strict=True
    ):
        assert worker._thread is None and not worker.snapshot().alive
        assert (
            worker.snapshot().state == "failed" and worker.snapshot().error_type == "RuntimeError"
        )
        assert (
            worker.snapshot().purges == 1
            and worker.snapshot().cleanup_attempts == 2
            and worker.snapshot().polls == 1
        )
        assert not worker.snapshot().pending_cleanup
        assert (
            broken.state == "failed"
            and broken.pending_cleanup
            and broken.purges == 0
            and broken.cleanup_attempts == broken.polls == 1
        )
        assert (
            life._erase is erase and life._copy_budget is copy_budget and life._wall_clock is clock
        )
        assert copy_budget._charged == 656 and life.admitted_payload_bytes == 144
        assert old_candidate._stopped and old_candidate._retired
        assert life._owner._shared._runtime._stopped
        current = life._owner._shared._runtime
        assert row["driver_after_failure"] == asdict(broken)
        assert (
            row["worker_started"]
            is row["thread_joined"]
            is row["automatic_purge_verified"]
            is False
        )
        assert (
            row["manual_cleanup_verified"]
            and row["terminal_retry_verified"]
            and row["cleanup_failure_verified"]
        )
        assert row["erase_callback"] == dict(
            after_failure=dict(calls=1, failures=1, delegated_calls=0),
            final=dict(calls=5, failures=1, delegated_calls=4),
            original=id(erase),
            current=id(life._erase),
        )
        assert (erase.calls, erase.failures, erase.delegated_calls) == (5, 1, 4)
        before, partial, after = row["before_expiry"], row["after_failure"], row["after_stop"]
        assert (
            partial["generation"] == 1
            and partial["previous_present"]
            and not partial["rollback_live"]
        )
        assert (
            partial["candidate_stopped"] == after["candidate_stopped"] == dict(old=True, new=True)
        )
        assert partial["lifecycle_failed"] is True and after["lifecycle_failed"] is False
        assert partial["pending_checkpoints"] == partial["pending_promotions"] == 0
        assert partial["raw_counts"] == dict(
            old_sources=3, old_labels=3, new_sources=3, new_labels=3
        )
        assert partial["cache_entries"] == dict(current=1, previous=1) and partial[
            "metadata_bytes"
        ] == dict(current=24, previous=24)
        assert partial["owned_copies"] == dict(
            limits={"max_lifetime_owned_bytes": 656}, charged_bytes=656, observed_retained_bytes=368
        )
        assert partial["identity"] == row["identity"]["original"] == row["identity"]["current"]
        for key in ("budget", "sharing", "registry"):
            assert before[key] == partial[key] == after[key]
        for name, completed in (("old", 1), ("new", 2)):
            assert partial[name]["revision"] == before[f"{name}_revision"] + 1
            assert after[f"{name}_revision"] == before[f"{name}_revision"] + 2
            assert (
                partial[name]["applied"]
                == before[f"{name}_cursor"]["applied"]
                == after[f"{name}_cursor"]["applied"]
            )
            assert after[f"{name}_cursor"]["completed_updates"] == completed
            assert after[f"{name}_cursor"]["experiences"] == after[f"{name}_cursor"]["labels"] == []
        assert (
            partial["managed"]["revoked_keys"]
            == after["managed"]["revoked_keys"]
            == [[f"e{i}", f"s{i}"] for i in range(3)]
        )
        assert (
            before["owned_copies"]["observed_retained_bytes"] == 512
            and after["owned_copies"]["observed_retained_bytes"] == 0
        )
        assert after["owned_copies"]["charged_bytes"] == 656
        assert after["serving"]["generation"] == 2 and after["previous_serving"] is None
        assert (
            after["serving"]["metadata"]
            == after["serving"]["cache"]
            == after["previous_owned_metadata"]
            == after["previous_owned_cache"]
            == {}
        )
        assert [
            (v["phase"], v["charged_bytes"], v["observed_retained_bytes"])
            for v in row["copy_accounting"]
        ][-3:] == [("committed_hit", 656, 512), ("failed_cleanup", 656, 368), ("cleanup", 656, 0)]
        cleanup = row["finish_cleanup"]["report"]
        assert cleanup["requested_keys"] == [] and cleanup["revoked_keys"] == [
            [f"e{i}", f"s{i}"] for i in range(3)
        ]
        assert (
            cleanup["checkpoints_invalidated"]
            == cleanup["promotions_invalidated"]
            == cleanup["model_snapshots_erased"]
            == 0
            and cleanup["inboxes_cleared"] == 2
        )
        phases = [
            "cached_hit",
            "poison_cache",
            "serving_snapshot",
            "promotion_commit",
            "checkpoint_restore",
            "fresh_prepare",
            "rollback",
        ]
        for phase_key in ("postfailure_refusals", "terminal_refusals"):
            assert [v["phase"] for v in row[phase_key]] == phases
            for value in row[phase_key]:
                assert (
                    value["complete_state_preserved"]
                    and value["poison_copy_attempts"] == 0
                    and value["feature_digest_calls"] == 5
                    and value["build_calls"] == 4
                )
                expected = (
                    "promotion ticket is foreign, forged, discarded or consumed"
                    if value["phase"] == "promotion_commit"
                    else "checkpoint is foreign, copied, discarded or already used"
                    if value["phase"] == "checkpoint_restore"
                    else "rollback receipt is foreign, stale or consumed"
                    if value["phase"] == "rollback"
                    else "candidate is stopped after uncertain native mutation"
                    if value["phase"] == "fresh_prepare"
                    else "payload cleanup failed; retry cleanup before access"
                    if phase_key == "postfailure_refusals"
                    else "retention driver terminal; raw admission/access is closed"
                )
                assert value["message"] == expected
        expected_fences = [
            dict(
                owner=owner,
                operation=operation,
                message=(
                    "candidate owner is retired after checkpoint handoff"
                    if owner == "old"
                    else "candidate is stopped after uncertain native mutation"
                ),
                native_work=False,
            )
            for owner in ("old", "new")
            for operation in ("train", "snapshot")
        ]
        assert (
            row["candidate_fences_after_failure"]
            == row["candidate_fences_final"]
            == expected_fences
        )
        budget_before = current._budget.updates_completed
        revisions = (old_candidate._revision, current._revision)
        calls_before = (FakeLearner.train_calls, dict(work))
        for runtime, message in ((old_candidate, "retired"), (current, "stopped")):
            for operation in (runtime.train_ready, runtime.candidate_snapshot):
                with pytest.raises(ValueError, match=message):
                    operation()
        assert calls_before == (FakeLearner.train_calls, work)
        assert current._budget.updates_completed == budget_before == 2 and revisions == (
            old_candidate._revision,
            current._revision,
        )
        codec = CoupledLearnerCheckpointCodec()
        for learner, key, wake in (
            (actor_a, "actor_a_utf8", 2),
            (old_candidate._candidate, "captured_b_utf8", 3),
            (current._candidate, "continued_c_utf8", 4),
            (current.actor._slot.bundle.learner, "continued_c_utf8", 4),
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
                assert json.loads(raw)["native"]["wake_state"]["wake_updates"] == wake
        assert len(json.dumps(row, separators=(",", ":"), allow_nan=False).encode()) <= 65536
