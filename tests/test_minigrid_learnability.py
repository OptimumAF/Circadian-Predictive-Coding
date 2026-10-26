"""Frozen MiniGrid stream/probe composition with owned fake native ports."""

from dataclasses import asdict
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_coupled_attribution as cli
from src.adapters import minigrid_learnability as pilot
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_learning_pilot import (
    METHODS,
    POINTS,
    evaluator as evaluator,
    observation,
    pilot_environment,
    same_state,
)
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


class ProbeEnvironment:
    work: dict[str, Any] = {}

    def __init__(self, layout, cue, seed=17, max_steps=16, *, role="train", episode_id="probe"):
        assert layout.context == f"E-{cue}" and role in ("probe_train", "probe_test")
        assert seed == 17 and max_steps == 16
        self.layout, self.cue, self.role, self.episode_id = layout, cue, role, episode_id
        self.observation = observation()
        self.terminal = False
        self.closed = False
        self.transitions: list[DemonstrationTransition] = []
        self.work.setdefault("probe_environments", []).append(self)
        callback = self.work.get("on_probe_collection")
        if callback:
            callback(self)

    @property
    def legal_actions(self):
        return () if self.terminal else tuple(range(7))

    def step(self, action):
        index = len(self.transitions)
        following = observation(index + 1)
        self.terminal = index == 8
        self.transitions.append(
            DemonstrationTransition(
                index,
                self.observation,
                action,
                0.5 if self.terminal else 0.0,
                following,
                self.terminal,
                7,
            )
        )
        self.observation = following
        return dict(
            observation=following,
            reward=0.5 if self.terminal else 0.0,
            terminated=self.terminal,
            truncated=False,
        )

    def to_demonstration(self):
        return DemonstrationEpisode(
            self.episode_id, self.layout.context, self.role, tuple(self.transitions), 7
        )

    @property
    def report(self):
        return dict(
            context=self.layout.context,
            cue=self.cue,
            role=self.role,
            layout=asdict(self.layout),
            reset_calls=1,
            environment_steps_attempted=len(self.transitions),
            environment_steps=len(self.transitions),
            terminated=self.terminal,
            truncated=False,
            total_return=0.5 if self.terminal else 0.0,
            status="complete" if self.terminal else "running",
            closed=self.closed,
            transitions=[asdict(row) for row in self.transitions],
            failure_work_may_be_unobserved=False,
        )

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def environment(monkeypatch, evaluator, *, accepted_arms=False):
    captured = pilot_environment(monkeypatch, evaluator)
    rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    row_factory = pilot._new_row
    score = pilot._score_checkpoint
    probe_runner = pilot.run_fresh_task_probe
    monkeypatch.setattr(ProbeEnvironment, "work", evaluator.work)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", ProbeEnvironment)
    monkeypatch.setattr(
        pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=lambda public: 0)
    )
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "_same_state", same_state)

    def create_row(method):
        row = row_factory(method)
        rows.append(row)
        return row

    def score_checkpoint(source, state, **kwargs):
        assert len(captured["finished"]) == 5
        raw.append(
            dict(source=source, state=state, updates=kwargs["updates"], method=kwargs["method"])
        )
        return score(source, state, **kwargs)

    def run_probe(source, state, training, evaluation, **kwargs):
        assert len(captured["finished"]) == 5
        budget = kwargs["budget"]
        assert budget.updates_completed == 0 and budget.budget.max_training_updates == 16
        assert kwargs["checkpoint_updates"] == (0, 4, 8, 16)
        assert kwargs["decision_diagnostics"] is True
        first_train = len(evaluator.work.get("trains", []))
        call = dict(
            source=source,
            state=state,
            training=training,
            evaluation=evaluation,
            budget=budget,
            source_before=pickle.dumps(source.snapshot_state()),
            state_before=pickle.dumps(state),
            inputs_before=pickle.dumps((training, evaluation)),
        )
        probes.append(call)
        try:
            call["report"] = probe_runner(source, state, training, evaluation, **kwargs)
            return call["report"]
        finally:
            call["trains"] = evaluator.work.get("trains", [])[first_train:]
            call["source_after"] = pickle.dumps(source.snapshot_state())
            call["state_after"] = pickle.dumps(state)
            call["inputs_after"] = pickle.dumps((training, evaluation))

    def predicted(owner):
        assert len(captured["finished"]) == 5
        callback = evaluator.work.get("test_prediction")
        if callback:
            callback(owner)

    def collected(game):
        assert len(captured["finished"]) == 5

    evaluator.work["on_predict"] = predicted
    evaluator.work["on_probe_collection"] = collected
    monkeypatch.setattr(pilot, "_new_row", create_row)
    monkeypatch.setattr(pilot, "_score_checkpoint", score_checkpoint)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", run_probe)
    if accepted_arms:

        def accepted(row, packets, sampler, deadline, check_boundary):
            check_boundary()
            source = pilot.shared._learner(row["method"])
            states: list[tuple[int, Any]] = []
            for point in POINTS:
                state = source.snapshot_state()
                state.completed_updates, state.training_examples = point, 9 * point
                states.append((point, state))
            budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=128), lambda: 0.0)
            budget.updates_completed = 128
            stream = SimpleNamespace(completed_updates=128, _states=states)
            row.update(
                status="trained",
                charged_updates=128,
                native_training_examples=1152,
                source_unchanged=True,
                actor_unchanged=True,
                checkpoint_updates=list(POINTS),
                stream_result=dict(completed_updates=128, training_examples=1152),
                wake_calls_attempted=128,
                wake_calls_returned=128,
            )
            captured["streams"].append((stream, source, packets, budget))
            captured["finished"].append(stream)
            return dict(
                row=row,
                source=source,
                states=states,
                stream=stream,
                budget=budget,
                guard=lambda: None,
            )

        monkeypatch.setattr(pilot.shared, "_train_arm", accepted)
        monkeypatch.setattr(pilot.shared, "_readback", lambda *args: None)
    return SimpleNamespace(
        captured=captured, rows=rows, probes=probes, raw=raw, work=evaluator.work
    )


def test_should_compose_five_original_streams_with_isolated_fixed_dose_probes_and_no_final(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator)
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert [row["method"] for row in report["methods"]] == list(METHODS)
    assert len(case.captured["streams"]) == len(case.captured["finished"]) == 5
    assert case.captured["collection_calls"] == 1
    assert len(case.raw) == 25 and len(case.probes) == 10
    assert len(case.work["trains"]) == 800
    assert sum(len(features) for _, features, _ in case.work["trains"]) == 7200
    assert len(case.work["predicts"]) == 230
    assert sum(len(features) for _, features in case.work["predicts"]) == 2070
    assert not case.work.get("sessions") and not case.work.get("environments")
    assert len(case.work["probe_environments"]) == 4
    assert all(
        game.closed and len(game.transitions) == 9 for game in case.work["probe_environments"]
    )
    assert [(game.layout.context, game.role) for game in case.work["probe_environments"]] == [
        (f"E-{cue}", role) for role in ("probe_train", "probe_test") for cue in ("blue", "red")
    ]
    assert report["probe_data"]["input_array_bytes"] == 193104
    assert report["probe_data"]["physical_episodes"] == 4
    assert len({id(call["budget"]) for call in case.probes}) == 10
    assert [call["state"].completed_updates for call in case.probes] == [0, 128] * 5
    for call in case.probes:
        assert len(call["training"]) == len(call["trains"]) == 16 and len(call["evaluation"]) == 2
        assert len({packet.episode.episode_id for packet in call["training"]}) == 16
        assert [packet.context for packet in call["training"]] == ["E-blue", "E-red"] * 8
        arrays = [
            value
            for packet in (*call["training"], *call["evaluation"])
            for value in (packet.features, packet.targets)
        ]
        assert all(
            not np.shares_memory(left, right)
            for index, left in enumerate(arrays)
            for right in arrays[index + 1 :]
        )
        assert all(
            packet.features.shape == (9, 148)
            and packet.role == packet.episode.role == "probe_train"
            for packet in call["training"]
        )
        assert all(
            packet.role == packet.episode.role == "probe_test" for packet in call["evaluation"]
        )
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
        assert call["budget"].updates_completed == call["report"]["completed_updates"] == 16
        assert call["report"]["completed_training_examples"] == 144
        assert call["report"]["reserved_data_bytes"] == 193104
        assert call["report"]["score_forward_calls_completed"] == 8
        assert [point["updates"] for point in call["report"]["checkpoints"]] == [0, 4, 8, 16]
        assert all(owner is not call["source"] for owner, _, _ in call["trains"])
        assert all(len(features) == 9 for _, features, _ in call["trains"])
    for (_, source, packets, budget), row in zip(
        case.captured["streams"], report["methods"], strict=True
    ):
        assert (
            source.policy.feature_dim,
            source.policy.hidden_dim,
            source.policy.num_classes,
            source.policy.seed,
        ) == (148, 8, 7, 17)
        assert source.policy.learning_rate == 0.03 and source.policy.momentum == 0.9
        assert len(packets) == len({packet.episode.episode_id for packet in packets}) == 128
        assert [packet.context for packet in packets] == ["A"] * 32 + ["B"] * 32 + [
            "C_train"
        ] * 32 + ["A"] * 32
        assert all(packet.role == "train" and packet.num_classes == 7 for packet in packets)
        assert source.state.completed_updates == 0
        assert budget.updates_completed == row["charged_updates"] == 128
        assert budget.replay_examples_completed == 0
        assert row["source_unchanged"] is row["actor_unchanged"] is True
        assert [point["updates"] for point in row["readouts"]] == list(POINTS)
        assert row["final_evaluation"]["status"] == "not_run"
        assert [slot["checkpoint_updates"] for slot in row["probes"]] == [0, 128]
        assert [slot["native_updates_before"] for slot in row["probes"]] == [0, 128]
        assert [slot["native_training_examples_before"] for slot in row["probes"]] == [0, 1152]
        assert all(
            slot["status"] == "complete" and slot["charged_updates"] == 16 for slot in row["probes"]
        )
        assert all(
            slot["source_unchanged"]
            is slot["state_unchanged"]
            is slot["budget_binding_preserved"]
            is True
            for slot in row["probes"]
        )
        assert row["acquisition_conditions"] == dict(A=False, B=False, C_train=False)
        assert row["probe_gain_change"] == 0
        assert all(
            value == 0 for metric in row["signed_metrics"].values() for value in metric.values()
        )
    assert report["work"]["native_train_calls"] == 800
    assert report["work"]["training_rows"] == 7200
    assert report["work"]["known_raw_score_forwards_completed"] == 150
    assert report["work"]["probe_score_forwards_completed"] == 80
    assert report["work"]["source_environment_steps_returned"] == 90
    assert report["work"]["final_episodes"] == report["work"]["learner_live_episodes"] == 0
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("kind", ["raw", "probe"])
@pytest.mark.parametrize("replacement", ["equal_bytes", "invalid_bytes"])
def test_should_refuse_prior_paid_record_replacement_before_later_work_and_decode_only_owned_payload(
    monkeypatch, evaluator, kind, replacement
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    changed: list[bytes] = []
    loads = pilot.pickle.loads

    def private_load(payload, *args, **kwargs):
        assert not changed or payload is not changed[0], "untrusted public record decoded"
        return loads(payload, *args, **kwargs)

    def corrupt(owner):
        key = "readouts" if kind == "raw" else "probes"
        prior = case.rows[0][key][0]
        if type(prior) is bytes and not changed:
            substitute = (
                bytes(bytearray(prior)) if replacement == "equal_bytes" else b"not owned evidence"
            )
            assert substitute is not prior
            case.rows[0][key][0] = substitute
            changed.append(substitute)

    monkeypatch.setattr(pilot.pickle, "loads", private_load)
    case.work["test_prediction"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "replaced" in report["reason"]
    assert len(changed) == 1 and report["comparisons"] is None
    assert all(row["outcomes"] is None for row in report["methods"])
    assert all(row["charged_updates"] == 128 for row in report["methods"])
    if kind == "raw":
        original = report["methods"][0]["readouts"][0]
        assert original["status"] == "complete" and len(original["scores"]) == 6
        assert report["methods"][0]["readouts"][1]["work"]["score_forward_calls_completed"] == 1
        assert len(case.work["predicts"]) == 7 and not case.probes
    else:
        original = report["methods"][0]["probes"][0]
        assert original["status"] == "complete" and original["charged_updates"] == 16
        assert original["report"]["completed_updates"] == 16
        second = report["methods"][0]["probes"][1]
        assert second["status"] == "failed" and second["charged_updates"] == 0
        assert second["report"]["score_forward_calls_completed"] == 1
        assert len(case.probes) == 2 and len(case.work["predicts"]) == 159
        assert len(case.work["trains"]) == 16
    assert len(case.work["probe_environments"]) == 4
    assert all(row["final_evaluation"]["status"] == "not_run" for row in report["methods"])


def test_should_keep_probe_commit_charge_before_corrupted_preceding_raw_evidence_is_refused(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    changed: list[bool] = []

    def corrupt(owner):
        if case.probes and not changed:
            prior = case.rows[0]["readouts"][0]
            assert type(prior) is bytes
            case.rows[0]["readouts"][0] = bytes(bytearray(prior))
            changed.append(True)

    case.work["on_train"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and changed == [True]
    slot = report["methods"][0]["probes"][0]
    assert slot["status"] == "failed" and slot["charged_updates"] == 1
    assert slot["report"]["completed_updates"] == 1
    assert slot["report"]["completed_training_examples"] == 9
    assert len(slot["report"]["diagnostics"]) == 1
    assert slot["report"]["score_forward_calls_completed"] == 2
    assert len(case.work["trains"]) == 1 and len(case.probes) == 1
    assert report["methods"][0]["readouts"][0]["status"] == "complete"
    assert report["methods"][0]["probes"][1]["status"] == "not_run"
    assert report["comparisons"] is None


def test_should_retain_failed_probe_native_attempt_and_known_commit_without_later_scoring(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    case.work["fail_train"] = lambda owner: bool(case.probes) and owner.state.completed_updates == 1
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    slot = report["methods"][0]["probes"][0]
    result = slot["report"]
    assert slot["status"] == "failed" and slot["charged_updates"] == 1
    assert result["status"] == "failed" and "fake training failed" in result["reason"]
    assert result["attempted_updates"] == 2 and result["completed_updates"] == 1
    assert (
        result["attempted_training_examples"] == 18 and result["completed_training_examples"] == 9
    )
    assert [point["updates"] for point in result["checkpoints"]] == [0]
    assert result["score_forward_calls_completed"] == 2
    assert len(case.work["trains"]) == 2 and len(case.probes) == 1
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["comparisons"] is None


def test_should_refuse_probe_source_mutation_after_paid_update_without_touching_next_probe(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)

    def mutate(owner):
        if case.probes:
            case.probes[-1]["source"].state.parameters["weight"] += 1

    case.work["on_train"] = mutate
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    slot = report["methods"][0]["probes"][0]
    assert slot["charged_updates"] == slot["report"]["completed_updates"] == 1
    assert slot["source_unchanged"] is False
    assert slot["status"] == "failed" and slot.get("readback_error")
    assert len(case.probes) == len(case.work["trains"]) == 1
    assert report["methods"][0]["probes"][1]["status"] == "not_run"


def test_should_keep_completed_probe_charges_when_original_probe_budget_readback_fails(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    runner = pilot.run_fresh_task_probe

    def corrupt_budget(*args, **kwargs):
        result = runner(*args, **kwargs)
        object.__setattr__(kwargs["budget"].budget, "max_training_updates", 17)
        return result

    monkeypatch.setattr(pilot, "run_fresh_task_probe", corrupt_budget)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    slot = report["methods"][0]["probes"][0]
    assert slot["status"] == "failed" and slot["charged_updates"] == 16
    assert slot["report"]["status"] == "complete" and slot["report"]["completed_updates"] == 16
    assert "budget" in slot["readback_error"]
    assert len(case.probes) == 1 and len(case.work["trains"]) == 16
    assert report["methods"][0]["probes"][1]["status"] == "not_run"
    assert report["comparisons"] is None


def test_should_preserve_probe_teacher_primary_failure_with_close_error_and_observed_prefix(
    monkeypatch, evaluator
):
    environment(monkeypatch, evaluator, accepted_arms=True)

    class FailingEnvironment(ProbeEnvironment):
        def close(self):
            super().close()
            raise RuntimeError("probe close secondary")

    def fail(public):
        raise RuntimeError("probe teacher primary")

    monkeypatch.setattr(pilot, "MiniGridKeyDoor", FailingEnvironment)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=fail))
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and "probe teacher primary" in report["reason"]
    assert "probe close secondary" not in report["reason"]
    assert len(report["probe_source_reports"]) == 1
    observed = report["probe_source_reports"][0]
    assert "probe teacher primary" in observed["primary_error"]
    assert "probe close secondary" in observed["close_error"]
    assert observed["environment_steps"] == observed["environment_steps_attempted"] == 0
    assert all(row["probes"][0]["status"] == "not_run" for row in report["methods"])


@pytest.mark.parametrize("fault", ["source_count", "probe_count", "equal_probe_container"])
def test_should_refuse_changed_public_source_evidence_and_report_only_original_paid_costs(
    monkeypatch, evaluator, fault
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    captured: list[dict[str, Any]] = []
    changed: list[bool] = []
    builtin_dict = dict

    def construction(*args, **kwargs):
        value = builtin_dict(*args, **kwargs)
        if not captured and {
            "methods",
            "training_sources",
            "source_reports",
            "probe_source_reports",
        } <= set(kwargs):
            captured.append(value)
            # Restore normal global lookup before any caller callbacks/type checks.
            monkeypatch.delattr(pilot, "dict")
        return value

    def corrupt(owner):
        if not changed:
            report = captured[0]
            if fault == "source_count":
                report["source_reports"][0]["environment_steps"] = 999
            elif fault == "probe_count":
                report["probe_source_reports"][0]["environment_steps"] = 999
            else:
                old = report["probe_source_reports"]
                report["probe_source_reports"] = list(old)
                assert (
                    report["probe_source_reports"] == old
                    and report["probe_source_reports"] is not old
                )
            changed.append(True)

    monkeypatch.setattr(pilot, "dict", construction, raising=False)
    case.work["test_prediction"] = corrupt
    report = pilot.run_pilot()
    assert changed == [True] and report["status"] == "incomplete"
    assert report["comparisons"] is None
    assert len(case.work["predicts"]) == 1 and not case.probes
    assert all(entry["environment_steps"] == 9 for entry in report["source_reports"])
    assert all(entry["environment_steps"] == 9 for entry in report["probe_source_reports"])
    assert report["work"]["source_environment_steps_returned"] == 90
    assert report["work"]["known_raw_score_forwards_completed"] == 1
    assert report["work"]["probe_native_train_calls"] == 0
    assert all(row["outcomes"] is None for row in report["methods"])


def test_should_preserve_collected_probe_prefix_and_unknown_constructor_work_without_retry(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)

    class ConstructorFailure(ProbeEnvironment):
        def __init__(self, layout, cue, *args, **kwargs):
            if kwargs.get("role") == "probe_train" and cue == "red":
                raise RuntimeError("probe constructor state unknown")
            super().__init__(layout, cue, *args, **kwargs)

    monkeypatch.setattr(pilot, "MiniGridKeyDoor", ConstructorFailure)
    report = pilot.run_pilot()
    assert (
        report["status"] == "incomplete" and "probe constructor state unknown" in report["reason"]
    )
    observed = report["probe_source_reports"]
    assert len(observed) == 2
    assert observed[0]["environment_steps"] == 9 and observed[0]["status"] == "complete"
    assert observed[1]["failure_work_may_be_unobserved"] is True
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert len(case.work["probe_environments"]) == 1 and case.work["probe_environments"][0].closed
    assert not case.raw and not case.probes and not case.work.get("trains")
    assert all(row["charged_updates"] == 128 for row in report["methods"])


def test_should_keep_original_paid_probe_result_after_post_return_corruption(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, accepted_arms=True)
    original = pilot._same_state
    changed: list[bool] = []

    def corrupt_after_return(left, right):
        if case.probes and case.probes[0].get("report") is not None and not changed:
            case.probes[0]["report"]["completed_training_examples"] = 99999
            changed.append(True)
        return original(left, right)

    monkeypatch.setattr(pilot, "_same_state", corrupt_after_return)
    report = pilot.run_pilot()
    assert changed and report["status"] == "incomplete"
    assert report["comparisons"] is None and len(case.probes) == 1
    slot = report["methods"][0]["probes"][0]
    assert slot["status"] == "failed" and slot["charged_updates"] == 16
    assert slot["report"]["completed_updates"] == 16
    assert slot["report"]["completed_training_examples"] == 144
    assert slot["report"]["score_forward_calls_completed"] == 8
    assert report["work"]["probe_training_rows"] == 144
    assert report["work"]["native_train_calls"] == 656


def test_should_render_original_pending_slot_after_mutation_refusal(monkeypatch, evaluator):
    case = environment(monkeypatch, evaluator, accepted_arms=True)

    def corrupt_pending(owner):
        case.rows[-1]["readouts"][-1]["work"] = dict(
            score_forward_calls_attempted=999,
            score_forward_calls_completed=999,
            score_rows_attempted=8991,
            score_rows_completed=8991,
            failure_work_may_be_unobserved=False,
        )

    case.work["test_prediction"] = corrupt_pending
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and not case.probes
    pending_slot = report["methods"][-1]["readouts"][-1]
    assert pending_slot == dict(updates=128, role="train", status="not_run")
    assert report["work"]["known_raw_score_forwards_completed"] == 1
    assert report["work"]["known_raw_score_rows_completed"] == 9
    assert report["comparisons"] is None


def test_should_keep_paid_benchmark_costs_after_final_readback_corrupts_charge(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator)
    original = pilot.shared._readback

    def corrupt_charge(row, *args):
        original(row, *args)
        if row["method"] == "bp" and len(case.probes) == 10:
            row["charged_updates"] = 0

    monkeypatch.setattr(pilot.shared, "_readback", corrupt_charge)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete"
    assert report["comparisons"] is None
    assert report["methods"][0]["charged_updates"] == 128
    assert report["work"]["benchmark_native_train_calls"] == 640
    assert report["work"]["benchmark_training_rows"] == 5760
    assert report["work"]["probe_native_train_calls"] == 160
    assert report["work"]["native_train_calls"] == 800


def test_should_describe_frozen_fit_and_probe_recipe_without_resources(monkeypatch):
    for name in (
        "require_torch",
        "ProcessRssSampler",
        "MiniGridKeyDoor",
        "_collect_probe_sources",
        "_score_checkpoint",
        "run_fresh_task_probe",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", forbidden)
    protocol = pilot.describe_protocol()
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["methods"] == list(METHODS) and protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["probe"]["source_updates"] == [0, 128]
    assert protocol["probe"]["checkpoint_updates"] == [0, 4, 8, 16]
    assert protocol["probe"]["input_array_bytes"] == 193104
    work = protocol["planned_work"]
    assert work["native_train_calls"] == 800 and work["training_rows"] == 7200
    assert work["score_forwards"] == 230 and work["score_rows"] == 2070
    assert work["final_episodes"] == work["learner_live_episodes"] == work["replay_rows"] == 0
    assert work["source_environment_steps"] == 90 and work["recurrent_unroll_steps"] == 1854


@pytest.mark.parametrize("mode", ["describe", "complete", "incomplete"])
def test_should_dispatch_only_explicit_learnability_cli_and_preserve_failure_exit(
    monkeypatch, capsys, mode
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["pilot", "--scenario", "torch-minigrid-learnability"]
        + (["--describe"] if mode == "describe" else []),
    )
    calls: list[bool] = []

    def selected():
        calls.append(True)
        return dict(
            status="incomplete" if mode == "incomplete" else "complete",
            selected="minigrid-learnability",
        )

    monkeypatch.setattr(
        cli,
        "describe_minigrid_learnability_protocol",
        selected if mode == "describe" else forbidden,
    )
    monkeypatch.setattr(
        cli, "run_minigrid_learnability_pilot", forbidden if mode == "describe" else selected
    )
    monkeypatch.setattr(cli, "run_minigrid_pilot", forbidden)
    if mode == "incomplete":
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 2
    else:
        cli.main()
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["selected"] == "minigrid-learnability"
