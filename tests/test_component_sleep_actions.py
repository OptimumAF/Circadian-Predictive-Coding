"""One bounded native component event through original consolidation authority."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pickle
import random
from time import monotonic
from typing import Any

import numpy as np
import pytest

from src.adapters.numpy_learners import CircadianLearner
from src.app.resource_sharing import ResourceSharedRuntime, ServingPriorityGate
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget, ToyExecutionStopped
from src.core.actor_ports import ConsolidatedState
from src.core.adaptation_action import AdaptationAction, AdaptationEnvelope
from src.core.circadian_predictive_coding import CircadianConfig, CircadianPredictiveCodingNetwork
from src.core.learner_ports import TrainingDiagnostic
from src.core.resource_sharing import SharingLimits
from test_actor_shadow import PausableLearner, queue, setup


class SleepLearner(PausableLearner):
    def __init__(self, value=None, calls=None, after=None):
        super().__init__(value)
        self.calls = [] if calls is None else calls
        self.after = after

    def fork(self):
        return SleepLearner(self.value, self.calls, self.after)

    def validate_component_sleep(self):
        pass

    def prepare_component_sleep(self, state):
        self.calls.append("sleep")
        state[:] = [v * 0.8 for v in state]
        if self.after:
            self.after()
        return ConsolidatedState(state, TrainingDiagnostic("fixture_component_sleep_v1", 1.0))


def wrapped(
    *, source=None, budget=None, resource=None, action=None, max_sleep=2, max_consolidations=2
):
    r, c, s, b = setup(
        source=source or SleepLearner(), budget=budget, max_consolidations=max_consolidations
    )
    g = ServingPriorityGate(SharingLimits(1, 4, 1), resource_available=resource or (lambda: True))
    envelope = AdaptationEnvelope(
        (
            AdaptationAction("wait"),
            AdaptationAction("wake", 1),
            action or AdaptationAction("sleep", sleep_updates=1),
        ),
        15,
        2,
        max_replay_examples=1,
        max_sleep_updates=max_sleep,
        max_learning_rate_multiplier=2,
        gate_settings=("unchanged", "active"),
    )
    return ResourceSharedRuntime(r, g, action_envelope=envelope), r, c, s, b, g


def native(**changes):
    cfg = CircadianConfig(
        **dict(
            sleep_mode="components",
            sleep_enable_split=False,
            sleep_enable_prune=False,
            sleep_enable_replay=False,
            sleep_enable_homeostasis=True,
            homeostatic_downscale_factor=0.8,
            **changes,
        )
    )
    model = CircadianPredictiveCodingNetwork(
        2, 4, seed=23, circadian_config=cfg, min_hidden_dim=4, max_hidden_dim=4
    )
    return CircadianLearner(
        model, learning_rate=0.03, inference_steps=2, inference_learning_rate=0.2
    )


def batch():
    return np.array([[0.3, -0.2], [-0.5, 0.4]]), np.array([[1.0], [0.0]])


def test_sleep_deferrals_and_wake_interleaving_preserve_original_charges_and_source():
    w, r, c, source, b, g = wrapped()
    queue(r)
    c.advance_to(3)
    w.train_action("wake")
    before = r.candidate_snapshot().state
    g.pause()
    paused = w.train_action("sleep")
    assert (
        paused.requested_sleep_updates == 1
        and paused.sleep_receipt is None
        and paused.training.deferred_reason == "paused"
    )
    g.resume()
    with g.serving():
        assert w.train_action("sleep").training.deferred_reason == "serving_active"
    assert source.calls == []
    event = w.train_action("sleep")
    assert event.sleep_receipt == r.applied_consolidations[0] and event.training.updates == ()
    assert (
        event.progress.sleep_admissions
        == event.progress.sleep_native_attempts
        == event.progress.sleep_native_completions
        == event.progress.sleep_commits
        == 1
    )
    assert r.candidate_snapshot().state == [v * 0.8 for v in before]
    queue(r, "second", [6, 6])
    w.train_action("wake")
    assert r.candidate_snapshot().state == [
        6,
        6,
    ] and source.snapshot_state() == r.actor.snapshot_state().state == [2, 2]
    assert b.updates_completed == 2 and g.snapshot().admitted_updates == 3


def test_sleep_resource_deferral_has_no_native_payload_copy(monkeypatch):
    w, r, c, source, b, g = wrapped(resource=lambda: False)
    monkeypatch.setattr(r._candidate, "snapshot_state", lambda: pytest.fail("payload copy"))
    assert w.train_action("sleep").training.deferred_reason == "resource_unavailable"
    assert source.calls == [] and g.snapshot().admitted_updates == b.updates_completed == 0


@pytest.mark.parametrize("kind", ["missing", "wake", "replay", "rate", "gate", "two"])
def test_unsupported_sleep_refuses_before_resources_and_payloads(kind):
    action = AdaptationAction("sleep", sleep_updates=1)
    if kind == "wake":
        action = replace(action, wake_updates=1)
    if kind == "replay":
        action = replace(action, replay_examples=1)
    if kind == "rate":
        action = replace(action, learning_rate_multiplier=0.5)
    if kind == "gate":
        action = replace(action, gate_setting="active")
    if kind == "two":
        action = replace(action, sleep_updates=2)
    source = PausableLearner() if kind == "missing" else SleepLearner()
    w, r, c, s, b, g = wrapped(
        source=source, action=action, resource=lambda: pytest.fail("resource")
    )
    with pytest.raises(ValueError, match="unsupported"):
        w.train_action("sleep")
    assert (
        g.snapshot().admitted_updates == b.updates_completed == 0 and r.applied_consolidations == ()
    )


def test_sleep_envelope_and_original_consolidation_quota_are_distinct_paid_limits():
    w, r, c, s, b, g = wrapped(max_sleep=1)
    w.train_action("sleep")
    with pytest.raises(ValueError, match="envelope"):
        w.train_action("sleep")
    assert g.snapshot().admitted_updates == 1
    w, r, c, s, b, g = wrapped(max_consolidations=0)
    with pytest.raises(ValueError, match="quota"):
        w.train_action("sleep")
    p = w.adaptation_snapshot().progress
    assert (
        p.sleep_admissions == 1
        and p.sleep_native_attempts == 0
        and g.snapshot().admitted_updates == 1
    )


@pytest.mark.parametrize("changed", ["quota", "receipt", "attempt"])
def test_replaced_sleep_history_refuses_before_more_native_work(changed):
    w, r, c, s, b, g = wrapped()
    w.train_action("sleep")
    if changed == "quota":
        r._consolidation_limit += 1
    if changed == "receipt":
        r._consolidations[0] = replace(
            r._consolidations[0], diagnostic=TrainingDiagnostic("changed", 2.0)
        )
    if changed == "attempt":
        r._attempted_ids.add("unobserved-event")
    with pytest.raises(ValueError, match="(quota|evidence)"):
        w.train_action("wake")
    assert s.calls == ["sleep"] and g.snapshot().admitted_updates == 1


def test_failed_native_sleep_stops_all_further_work_and_preserves_source_and_paid_attempt():
    def fail():
        raise RuntimeError("lost native return")

    w, r, c, s, b, g = wrapped(source=SleepLearner(after=fail))
    before = r.candidate_snapshot().state
    with pytest.raises(RuntimeError, match="lost native"):
        w.train_action("sleep")
    p = w.adaptation_snapshot().progress
    assert (
        p.sleep_admissions == p.sleep_native_attempts == 1
        and p.sleep_native_completions == p.sleep_commits == 0
    )
    assert p.unreturned_work_unknown and p.sleep_publication_unknown
    assert (
        r.candidate_snapshot().state == before
        and s.snapshot_state() == r.actor.snapshot_state().state == [2, 2]
    )
    with pytest.raises(ValueError, match="stopped"):
        w.train_action("wake")
    assert g.snapshot().admitted_updates == 1


def test_restore_uncertainty_is_paid_and_actor_stays_available(monkeypatch):
    w, r, c, s, b, g = wrapped()

    def fail(state):
        r._candidate.value[0] = 99
        raise RuntimeError("partial restore")

    monkeypatch.setattr(r._candidate, "restore_state", fail)
    with pytest.raises(RuntimeError, match="partial restore"):
        w.train_action("sleep")
    p = w.adaptation_snapshot().progress
    assert (
        p.sleep_native_completions == 1
        and p.sleep_commits == 0
        and p.sleep_publication_unknown
        and p.unreturned_work_unknown
    )
    assert r.actor.snapshot_state().state == [2, 2]


def test_post_commit_budget_stop_keeps_real_receipt_and_complete_charge():
    time = [0.0]
    b = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_wall_seconds=1), lambda: time[0]
    )
    w, r, c, s, b, g = wrapped(
        source=SleepLearner(after=lambda: time.__setitem__(0, 2.0)), budget=b
    )
    with pytest.raises(ToyExecutionStopped):
        w.train_action("sleep")
    p = w.adaptation_snapshot().progress
    assert p.sleep_native_completions == p.sleep_commits == len(r.applied_consolidations) == 1
    assert not p.sleep_publication_unknown and not p.unreturned_work_unknown
    assert g.snapshot().admitted_updates == 1 and b.updates_completed == 0


def test_post_commit_metadata_read_failure_is_unknown_and_never_admits_more_work(monkeypatch):
    w, r, c, s, b, g = wrapped()
    original = r.capture_consolidation_cursor
    calls = [0]

    def capture():
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("lost metadata read")
        return original()

    monkeypatch.setattr(r, "capture_consolidation_cursor", capture)
    with pytest.raises(RuntimeError, match="lost metadata"):
        w.train_action("sleep")
    assert len(r.applied_consolidations) == 1
    assert (
        w._action_progress.sleep_publication_unknown and w._action_progress.unreturned_work_unknown
    )
    with pytest.raises(ValueError, match="evidence"):
        w.train_action("wake")
    assert g.snapshot().admitted_updates == 1 and s.calls == ["sleep"]


def test_two_valid_sleep_events_keep_bounded_metadata_and_exact_cumulative_charge():
    w, r, c, s, b, g = wrapped()
    w.train_action("sleep")
    last = w.train_action("sleep")
    assert last.progress.sleep_admissions == last.progress.sleep_commits == 2
    assert len(w._action_sleep_history[3]) == len(w._action_sleep_history[4]) == 2
    assert len(r.applied_consolidations) == 2 and not hasattr(w, "_sleep_model_chain")
    with pytest.raises(ValueError, match="envelope"):
        w.train_action("sleep")
    assert g.snapshot().admitted_updates == 2 and b.updates_completed == 0


def test_native_component_sleep_matches_full_native_reference_without_source_or_rng_mutation():
    source = native()
    x, y = batch()
    source.train_batch(x, y)
    initial = source.snapshot_state()
    before = pickle.dumps(initial)
    numpy_rng = pickle.dumps(np.random.get_state())
    python_rng = random.getstate()
    reference = deepcopy(source._model)
    expected = reference.sleep_event(max_replay_examples=0, max_hidden_width=4)
    result = source.prepare_component_sleep(initial)
    assert expected.performed and result.diagnostic.value == 1.0
    assert pickle.dumps(result.state) == pickle.dumps(reference.snapshot_state())
    assert pickle.dumps(source.snapshot_state()) == before
    assert numpy_rng == pickle.dumps(np.random.get_state()) and python_rng == random.getstate()
    np.testing.assert_array_equal(
        result.state.state["weight_input_hidden"], initial.state["weight_input_hidden"] * 0.8
    )
    assert (
        result.state.state["_sleep_events"] == 1 and result.state.state["_epochs_since_sleep"] == 0
    )


@pytest.mark.parametrize("kind", ["mode", "replay", "split", "prune", "width", "disabled"])
def test_native_incompatible_sleep_metadata_refuses_before_native_operation(kind, monkeypatch):
    source = native()
    model = source._model
    values: dict[str, dict[str, Any]] = {
        "mode": dict(sleep_mode="legacy"),
        "replay": dict(sleep_enable_replay=True),
        "split": dict(sleep_enable_split=True),
        "prune": dict(sleep_enable_prune=True),
        "disabled": dict(sleep_enable_homeostasis=False, sleep_enable_chemical_reset=False),
    }
    if kind == "width":
        model.max_hidden_dim = 5
    else:
        model.config = replace(model.config, **values[kind])
    monkeypatch.setattr(
        CircadianPredictiveCodingNetwork,
        "sleep_event",
        lambda *a, **kw: pytest.fail("native sleep"),
    )
    with pytest.raises(ValueError, match="component sleep"):
        source.validate_component_sleep()


def test_native_malformed_snapshot_and_failed_staged_sleep_leave_source_unchanged(monkeypatch):
    source = native()
    state = source.snapshot_state()
    before = pickle.dumps(state)
    bad = deepcopy(state)
    bad.state.pop("weight_input_hidden")
    with pytest.raises(ValueError):
        source.prepare_component_sleep(bad)
    original = CircadianPredictiveCodingNetwork.sleep_event

    def fail(*args, **kw):
        original(*args, **kw)
        raise RuntimeError("staged sleep failed")

    monkeypatch.setattr(CircadianPredictiveCodingNetwork, "sleep_event", fail)
    with pytest.raises(RuntimeError, match="staged sleep"):
        source.prepare_component_sleep(state)
    assert pickle.dumps(source.snapshot_state()) == before


def test_native_fixed_component_sleep_case():
    output = os.environ.get("COMPONENT_SLEEP_NATIVE_REPORT")
    if not output:
        pytest.skip("separate frozen native engineering case")
    from src.core.experience import Experience, ExperiencePermissions, LabelArrival
    from src.shared.process_memory import ProcessRssSampler

    rng = pickle.dumps(np.random.get_state())
    prng = random.getstate()
    source = native()
    x, y = batch()
    initial = pickle.dumps(source.snapshot_state())
    with ProcessRssSampler(interval_seconds=0.02) as sampler:
        b = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=2, max_wall_seconds=60, max_process_rss_bytes=512 * 1024**2
            ),
            monotonic,
        )
        b.attach_memory(sampler)
        r, c, _, _ = setup(source=source, budget=b, max_consolidations=1)
        g = ServingPriorityGate(SharingLimits(1, 3, 1), resource_available=lambda: True)
        e = AdaptationEnvelope(
            (AdaptationAction("wake", 1), AdaptationAction("sleep", sleep_updates=1)),
            8,
            2,
            max_sleep_updates=1,
        )
        w = ResourceSharedRuntime(r, g, action_envelope=e)
        polls = []
        sleep_weights: dict[str, list[list[float]]] = {}
        for i in range(2):
            c.advance_to(1 + 3 * i)
            r.record_experience(
                Experience(
                    f"s{i}",
                    f"e{i}",
                    1 + 3 * i,
                    "actor-0",
                    x,
                    "train",
                    ExperiencePermissions(training=True),
                )
            )
            c.advance_to(2 + 3 * i)
            r.record_label(LabelArrival(f"l{i}", f"s{i}", f"e{i}", 2 + 3 * i, "actor-0", y))
            if i == 1:
                before = r.candidate_snapshot().state
                g.pause()
                polls.append(w.train_action("sleep"))
                assert polls[-1].training.deferred_reason == "paused"
                g.resume()
                with g.serving():
                    polls.append(w.train_action("sleep"))
                    assert polls[-1].training.deferred_reason == "serving_active"
                polls.append(w.train_action("sleep"))
                after = r.candidate_snapshot().state
                sleep_weights["before"] = before.state["weight_input_hidden"].tolist()
                sleep_weights["after"] = after.state["weight_input_hidden"].tolist()
                np.testing.assert_array_equal(
                    after.state["weight_input_hidden"], before.state["weight_input_hidden"] * 0.8
                )
            polls.append(w.train_action("wake"))
        final = r.candidate_snapshot()
        assert (
            final.wake_updates == b.updates_completed == 2 and final.consolidations_completed == 1
        )
        assert (
            final.state.state["_epoch_count"] == 2
            and final.state.state["_epochs_since_sleep"] == 1
            and final.state.state["_sleep_events"] == 1
        )
        assert (
            pickle.dumps(source.snapshot_state())
            == pickle.dumps(r.actor.snapshot_state().state)
            == initial
        )
        assert pickle.dumps(np.random.get_state()) == rng and random.getstate() == prng
        report = dict(
            complete=True,
            source_unchanged=True,
            actor_unchanged=True,
            numpy_rng_unchanged=True,
            python_rng_unchanged=True,
            candidate_counts=dict(
                wake_updates=final.wake_updates,
                consolidations_completed=final.consolidations_completed,
                native_epochs=final.state.state["_epoch_count"],
                epochs_since_sleep=final.state.state["_epochs_since_sleep"],
                sleep_events=final.state.state["_sleep_events"],
            ),
            native_config=asdict(final.state.config),
            input_weights_before_sleep=sleep_weights["before"],
            input_weights_after_sleep=sleep_weights["after"],
            polls=[asdict(p) for p in polls],
            receipts=[asdict(p) for p in r.applied_updates],
            consolidations=[asdict(p) for p in r.applied_consolidations],
            sharing=asdict(g.snapshot()),
            budget=dict(
                limits=asdict(b.budget),
                updates_completed=b.updates_completed,
                started_at=b.started_at,
                last_clock=b.last_clock,
            ),
            sampled_process_rss=asdict(sampler.snapshot()),
        )
        Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
