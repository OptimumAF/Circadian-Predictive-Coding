"""Actual source consent/receipt handoff plus tiny private callback fixtures."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from scripts.ticket_linear_rehearsal import SourceTrainPackets, LinearProbeWork, LIMITS
from scripts.context_head_probe import build_head_cohorts
from scripts.run_ticket_workflow import CONTEXT_FEATURES, CATALOG
from src.adapters.linear_fit_learner import LinearFitLearner, MINIMUM_CHANGE_RESIDUAL_ID
from src.app.coupled_attribution import AttributionBatch, AttributionStream
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.categorical_linear_state import LinearFitPolicy
from src.core.decision_features import DecisionFeatureSpec
from src.core.learner_ports import TrainingDiagnostic
from src.core.probe_update import ProbeUpdateFailed
from src.shared.process_memory import ProcessRssSampler

SCRIPTED = dict(calls=0, rows=0)


def scripted(owner, x, y):
    SCRIPTED["calls"] += 1
    SCRIPTED["rows"] += len(x)
    before = owner.snapshot_state()
    model = CategoricalLinearFit(9, 6, len(x), ((0.0,) * 6,) * 10, 0, (0.0,) * min(len(x), 10), 0.0)
    owner._state = replace(
        before,
        model=model,
        completed_updates=before.completed_updates + 1,
        training_examples=before.training_examples + len(x),
    )
    return TrainingDiagnostic(MINIMUM_CHANGE_RESIDUAL_ID, 0.0)


class ScriptedSource(LinearFitLearner):
    def train_batch(self, x, y):
        return scripted(self, x, y)

    def fork(self):
        child = ScriptedSource(self.policy)
        child.restore_state(self.snapshot_state())
        return child


def packet(rows, role, identifier, offset=-1.0):
    x = np.zeros((rows, 9), dtype=np.float64)
    x[:, :rows] = np.eye(rows)
    x[:, 8] = offset
    return AttributionBatch(
        role,
        identifier,
        tuple(f"{identifier}-{i}" for i in range(rows)),
        x,
        np.arange(rows, dtype=np.float64).reshape(-1, 1),
        num_classes=6,
    )


def source(*, consent=True, updates=12):
    clock = [0.0]
    memory = ProcessRssSampler(read_rss_bytes=lambda: 1)
    memory.sample()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=12, max_wall_seconds=30, max_process_rss_bytes=512 * 1024**2
        ),
        lambda: clock[0],
    )
    budget.attach_memory(memory)
    owner = ScriptedSource(LinearFitPolicy(9, 6))
    batches = tuple(packet(6, "train", f"source-{i}", i / 12) for i in range(12))
    stream = AttributionStream(
        owner,
        batches,
        (),
        checkpoint_updates=(0, 4, 8, 12),
        label_delay=1,
        budget=budget,
        training_only=True,
        replay_consent=consent,
    )
    stream.run_until(updates)
    return stream, budget, memory, clock


def callback_fixture(*, control="joint"):
    stream, budget, memory, clock = source()
    state = stream.candidate_snapshot()
    certified = SourceTrainPackets(stream, state)
    aux = ToyBudgetSession(LIMITS, budget.clock)
    aux.attach_memory(memory)
    row: dict[str, Any] = {}
    work = LinearProbeWork(row, lambda: budget.before_final(), certified, aux, control=control)
    child = LinearFitLearner(state.policy)
    child.restore_state(state)
    return stream, child, work, aux, row, clock


@pytest.mark.parametrize("consent", [False, True])
def test_only_birth_consented_arrived_source_can_supply_rehearsal(consent):
    stream, *_ = source(consent=consent)
    if not consent:
        with pytest.raises(ValueError, match="consented"):
            SourceTrainPackets(stream, stream.candidate_snapshot())
    else:
        bank = SourceTrainPackets(stream, stream.candidate_snapshot())
        assert len(bank._bank) == 2 and bank.array_bytes["retained_train_bytes"] == 960
        assert all(
            e.permissions.replay is True for e in stream._runtime._inbox._experiences.values()
        )


def test_consent_cannot_be_granted_after_source_learning():
    stream, *_ = source(consent=False)
    stream._replay_consent = True
    with pytest.raises(ValueError, match="consent changed"):
        SourceTrainPackets(stream, stream.candidate_snapshot())


def test_source_requires_preceding_complete_receipts():
    stream, *_ = source(updates=8)
    with pytest.raises(ValueError, match="checkpoint12"):
        SourceTrainPackets(stream, stream.candidate_snapshot())


def test_initial_handoff_refuses_a_receipt_for_a_different_actor():
    stream, *_ = source()
    inbox = stream._runtime._inbox
    key = next(iter(inbox._applied))
    inbox._applied[key] = replace(inbox._applied[key], actor_version="foreign-actor")
    with pytest.raises(ValueError, match="receipt"):
        SourceTrainPackets(stream, stream.candidate_snapshot())


@pytest.mark.parametrize("fault", ["receipt", "label", "permission", "bank", "clock"])
def test_source_and_prior_evidence_changes_refuse_before_further_private_work(fault):
    stream, child, work, aux, _, _ = callback_fixture()
    inbox = stream._runtime._inbox
    if fault == "receipt":
        key = next(iter(inbox._applied))
        inbox._applied[key] = deepcopy(inbox._applied[key])
    elif fault == "label":
        next(iter(inbox._labels.values())).targets[0, 0] = 5.0
    elif fault == "permission":
        key = next(iter(inbox._experiences))
        entry = inbox._experiences[key]
        inbox._experiences[key] = replace(
            entry, permissions=replace(entry.permissions, replay=False)
        )
    elif fault == "bank":
        work._source._bank[0].targets[0, 0] = 5.0
    else:
        stream._clock.advance_to(13)
    with pytest.raises(ValueError):
        work.callback(
            child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=lambda: None
        )
    assert aux.updates_completed == aux.replay_examples_completed == 0


@pytest.mark.parametrize("control", ["joint", "current_only"])
def test_real_tiny_callback_applies_equal_rows_privately_and_keeps_permanent_costs(control):
    stream, child, work, aux, _, _ = callback_fixture(control=control)
    source_pin = stream.candidate_snapshot()
    for phase, current, expected in (
        ("fresh", packet(3, "probe_train", "fresh"), 15),
        ("return", packet(6, "train", "return"), 18),
    ):
        receipt = work.callback(child, current, (), phase=phase, check_boundary=lambda: None)
        assert receipt.applied and receipt.work["native_sgd_updates"] == 0
        assert receipt.work["fit_attempts"] == receipt.work["fit_returns"] == 1
        assert receipt.work["fit_rows"] == expected
        assert receipt.work["retained_train_rows_used"] == (12 if control == "joint" else 0)
    assert child.completed_updates == 14 and child.training_examples == 105
    assert stream.candidate_snapshot() == source_pin and stream._budget.updates_completed == 12
    assert aux.updates_completed == 2 and aux.replay_examples_completed == 24


@pytest.mark.parametrize("role", ["train", "probe_test", "final_test"])
def test_fresh_callback_refuses_wrong_roles_before_charging(role):
    _, child, work, aux, _, _ = callback_fixture()
    with pytest.raises(ValueError):
        work.callback(child, packet(3, role, "bad"), (), phase="fresh", check_boundary=lambda: None)
    assert aux.updates_completed == 0


def test_callback_refuses_source_owner_before_work():
    stream, _, work, aux, _, _ = callback_fixture()
    with pytest.raises(ValueError, match="detached"):
        work.callback(
            stream._runtime._candidate,
            packet(3, "probe_train", "f"),
            (),
            phase="fresh",
            check_boundary=lambda: None,
        )
    assert aux.updates_completed == 0


def test_failed_admitted_callback_keeps_charges_and_closes_more_work():
    _, child, work, aux, _, _ = callback_fixture()

    def fail():
        raise RuntimeError("late boundary")

    with pytest.raises(ProbeUpdateFailed) as failure:
        work.callback(child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=fail)
    assert not failure.value.receipt.applied and failure.value.receipt.work["fit_attempts"] == 0
    assert aux.updates_completed == 1 and aux.replay_examples_completed == 12
    with pytest.raises(ValueError, match="failed"):
        work.callback(
            child, packet(3, "probe_train", "again"), (), phase="fresh", check_boundary=lambda: None
        )


def test_lost_fit_return_keeps_known_application_and_no_native_sgd_credit(monkeypatch):
    stream, child, work, aux, _, _ = callback_fixture()
    before = stream.candidate_snapshot()

    def lose(owner, x, y):
        scripted(owner, x, y)
        raise RuntimeError("return lost")

    monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", lose)
    with pytest.raises(ProbeUpdateFailed) as error:
        work.callback(
            child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=lambda: None
        )
    receipt = error.value.receipt
    assert (
        receipt.applied
        and receipt.work["fit_returns"] == 0
        and receipt.work["native_sgd_updates"] == 0
    )
    assert aux.updates_completed == 1 and stream.candidate_snapshot() == before


def test_public_evidence_refusal_cannot_be_repaired_into_new_authority():
    _, child, work, aux, row, _ = callback_fixture()
    original = row["linear_rehearsal_work"]
    row["linear_rehearsal_work"] = deepcopy(original)
    with pytest.raises(ValueError, match="replaced"):
        work.check()
    row["linear_rehearsal_work"] = original
    with pytest.raises(ValueError, match="permanent"):
        work.callback(
            child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=lambda: None
        )
    assert aux.updates_completed == 0


def test_foreign_public_counter_and_budget_do_not_replace_original_paid_authority():
    _, child, work, aux, row, _ = callback_fixture()
    row["replay_examples"] = 999
    aux.updates_completed = 99
    with pytest.raises(ValueError):
        work.callback(
            child, packet(3, "probe_train", "f"), (), phase="fresh", check_boundary=lambda: None
        )
    preserved = work.preserved
    assert preserved["budget"]["admissions"] == 0 and preserved["budget"]["additional_rows"] == 0
    assert row["replay_examples"] == 999 and aux.updates_completed == 99


def test_existing_generic_runner_keeps_auxiliary_receipts_and_source_isolated(monkeypatch):
    stream, child, work, aux, _, _ = callback_fixture()
    source_pin = stream.candidate_snapshot()
    monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", scripted)
    # Reuse the accepted fixed cohort builder rather than inventing its context contract.
    fresh, tests, monitors, returns = build_head_cohorts(
        DecisionFeatureSpec(CONTEXT_FEATURES, CATALOG), head_task_transfer=True
    )
    sessions = [
        ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=16, max_wall_seconds=30), lambda: 0.0
        )
        for _ in range(2)
    ]
    result = run_fresh_task_probe(
        child,
        child.snapshot_state(),
        fresh,
        tests,
        budget=sessions[0],
        checkpoint_updates=(0, 4, 8, 16),
        training_readouts=monitors,
        return_training=returns,
        return_budget=sessions[1],
        update_callback=work.callback,
    )
    assert result["status"] == "complete", result.get("reason")
    work.complete(result)
    assert result["return_phase"]["status"] == "complete"
    assert aux.updates_completed == 32 and aux.replay_examples_completed == 384
    assert stream.candidate_snapshot() == source_pin and [
        b.updates_completed for b in sessions
    ] == [16, 16]


def test_returned_fit_requires_observed_application_and_unknown_state_stays_unknown(monkeypatch):
    stream, child, positive, _, _, _ = callback_fixture()
    pin = stream.candidate_snapshot()
    for phase, current in (
        ("fresh", packet(3, "probe_train", "positive-f")),
        ("return", packet(6, "train", "positive-r")),
    ):
        assert positive.callback(
            child, current, (), phase=phase, check_boundary=lambda: None
        ).applied
    for unknown in (False, True):
        owner = LinearFitLearner(pin.policy)
        owner.restore_state(pin)
        auxiliary = ToyBudgetSession(LIMITS, stream._budget.clock)
        auxiliary.attach_memory(stream._budget.process_rss_sampler)
        work = LinearProbeWork({}, lambda: None, positive._source, auxiliary, control="joint")

        def no_application(learner, x, y):
            SCRIPTED["calls"] += 1
            SCRIPTED["rows"] += len(x)
            if unknown:

                def lost_state():
                    raise RuntimeError("post-state unavailable")

                monkeypatch.setattr(learner, "snapshot_state", lost_state)
            return TrainingDiagnostic(MINIMUM_CHANGE_RESIDUAL_ID, 0.0)

        monkeypatch.setattr(LinearFitLearner, "train_batch_minimum_change", no_application)
        with pytest.raises(ProbeUpdateFailed) as error:
            work.callback(
                owner,
                packet(3, "probe_train", f"bad-{unknown}"),
                (),
                phase="fresh",
                check_boundary=lambda: None,
            )
        receipt = error.value.receipt
        assert not receipt.applied and receipt.work["fit_returns"] == 1
        assert receipt.work["application_unknown"] == int(unknown)
        assert work.preserved["applied_updates"] == 0
        assert work.preserved["application_unknown_updates"] == int(unknown)
        assert auxiliary.updates_completed == 1 and auxiliary.replay_examples_completed == 12
    assert stream.candidate_snapshot() == pin
