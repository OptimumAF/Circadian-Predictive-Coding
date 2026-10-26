"""Finite probe-support admission and guarded fake F-task composition.

The F arrays below are explicit engineering frames, not physics or teacher
evidence. Native methods remain fake ports; no neural updates are scheduled.
"""

from dataclasses import replace
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from src.adapters.minigrid_probe_support import summarize_probe_support
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import change_pixel, frame
from test_minigrid_learning_pilot import POINTS, evaluator as evaluator, observation
from test_minigrid_learnability import ProbeEnvironment
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")
BASE: dict[str, Any] = dict(public_history=True, invariant_history=True)
A_ONLY: dict[str, Any] = dict(**BASE, qualification_method="bp", acquisition_only=True)
FLAGS: dict[str, Any] = dict(**A_ONLY, distinct_probe=True)


def f_frame(cue, step):
    color = 2 if cue == "blue" else 0
    values = frame(wall_color=5, direction=step % 4, cue_color=color, key_color=color)
    # A moving public marker keeps the nine canonical current frames distinct.
    # It supplies no target, role or future frame to the learner.
    return change_pixel(values, 5 + step, (6, color, 0))


def raw_probe_data(*, pair="F", collision=False):
    def packet(role, cue, identity):
        frames = tuple(
            observation(index) if collision else f_frame(cue, index) for index in range(10)
        )
        episode = DemonstrationEpisode(
            identity,
            f"{pair}-{cue}",
            role,
            tuple(
                DemonstrationTransition(
                    index,
                    frames[index],
                    0,
                    0.5 if index == 8 else 0.0,
                    frames[index + 1],
                    index == 8,
                    7,
                )
                for index in range(9)
            ),
            7,
        )
        return pilot._packet(episode)

    train = tuple(
        packet("probe_train", ("blue", "red")[index % 2], f"support-probe-train-{index}")
        for index in range(16)
    )
    evaluation = tuple(
        packet("probe_test", cue, f"support-probe-test-{cue}") for cue in ("blue", "red")
    )
    return train, evaluation


class FProbeEnvironment(ProbeEnvironment):
    collision = False

    def __init__(self, layout, cue, seed=17, max_steps=16, *, role="train", episode_id="probe"):
        assert layout.context == f"F-{cue}" and layout.branch_row == 2
        assert layout.rotation == 0 and layout.swap_keys is False and layout.wall_color == "grey"
        assert role in ("probe_train", "probe_test") and seed == 17 and max_steps == 16
        self.layout, self.cue, self.role, self.episode_id = layout, cue, role, episode_id
        self.observation = self._frame(0)
        self.terminal, self.closed = False, False
        self.transitions = []
        self.work.setdefault("probe_environments", []).append(self)
        callback = self.work.get("on_probe_collection")
        if callback:
            callback(self)

    def _frame(self, step):
        return observation(step) if self.collision else f_frame(self.cue, step)

    def step(self, action):
        index = len(self.transitions)
        following = self._frame(index + 1)
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


def composition(monkeypatch, evaluator, *, collision=False):
    case = environment(monkeypatch, evaluator, method="public_invariant_bp", acquisition_only=True)
    monkeypatch.setattr(FProbeEnvironment, "work", case.native)
    monkeypatch.setattr(FProbeEnvironment, "collision", collision)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", FProbeEnvironment)
    return case


def test_should_measure_actual_raw_and_canonical_support_without_learning_or_mutating_cohorts(
    monkeypatch, evaluator
):
    train = shared._packets(evaluator.sources, context_sequence=("A",) * 4)
    data = raw_probe_data()
    before = pickle.dumps((train, data))
    monkeypatch.setattr(pilot, "_history_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    report = summarize_probe_support(train, data)
    assert (
        report["schema_id"] == "minigrid_probe_input_support_v1" and report["status"] == "complete"
    )
    assert report["training"]["packets"] == 128 and report["training"]["rows"] == 1152
    assert report["training"]["context_rows"] == {"A": 1152}
    assert report["training"]["raw"] == dict(
        unique_input_keys=2,
        conflicting_training_keys=0,
        distinct_input_label_pairs=2,
        logical_key_payload_bytes=2 * 2368,
        logical_label_payload_bytes=16,
    )
    assert report["training"]["canonical"] == dict(
        unique_input_keys=1,
        conflicting_training_keys=0,
        distinct_input_label_pairs=1,
        logical_key_payload_bytes=2368,
        logical_label_payload_bytes=8,
    )
    assert report["probe"]["training_packets"] == 16 and report["probe"]["evaluation_packets"] == 2
    assert report["probe"]["rows"] == 162 and len(report["probe"]["packets"]) == 18
    assert report["all_canonical_inputs_unseen"] is True
    for item in report["probe"]["packets"]:
        assert item["rows"] == 9
        for encoding in ("raw", "canonical"):
            assert item[encoding] == dict(
                matched_input_rows=0,
                unmatched_input_rows=9,
                matched_label_rows=0,
                unsupported_label_rows=9,
                conflicting_training_key_rows=0,
            )
    work = report["work"]
    assert work["native_updates"] == work["native_score_forwards"] == 0
    assert work["history_rows"] == work["canonical_projection_rows"] == 1314
    assert work["canonical_projected_array_equivalent_bytes"] == 1314 * 2368
    assert work["input_array_bytes"] == 1314 * 149 * 8
    assert work["training_logical_key_payload_bytes"] == 3 * 2368
    assert work["training_logical_label_payload_bytes"] == 24
    assert work["elapsed_seconds"] >= 0
    assert pickle.dumps((train, data)) == before
    frozen = json.dumps(report, sort_keys=True, allow_nan=False)
    original = train[0].features[0, 0]
    train[0].features[0, 0] = 0.123
    assert json.dumps(report, sort_keys=True, allow_nan=False) == frozen
    train[0].features[0, 0] = original
    report["training"]["context_rows"]["A"] = 0
    assert pickle.dumps((train, data)) == before


def test_should_preserve_known_e_overlap_and_conflicting_training_label_evidence(evaluator):
    train = list(shared._packets(evaluator.sources, context_sequence=("A",) * 4))
    packet = train[0]
    assert packet.episode is not None
    transitions = list(packet.episode.transitions)
    transitions[0] = replace(transitions[0], action=0)
    train[0] = shared._batch(replace(packet.episode, transitions=tuple(transitions)))
    data = raw_probe_data(pair="E", collision=True)
    report = summarize_probe_support(tuple(train), data)
    assert report["all_canonical_inputs_unseen"] is False
    assert report["training"]["raw"]["conflicting_training_keys"] == 1
    assert report["training"]["raw"]["distinct_input_label_pairs"] == 3
    assert report["training"]["canonical"]["conflicting_training_keys"] == 1
    assert report["training"]["canonical"]["distinct_input_label_pairs"] == 2
    for item in report["probe"]["packets"]:
        assert item["canonical"] == dict(
            matched_input_rows=9,
            unmatched_input_rows=0,
            matched_label_rows=9,
            unsupported_label_rows=0,
            conflicting_training_key_rows=9,
        )
        assert item["raw"] == dict(
            matched_input_rows=9,
            unmatched_input_rows=0,
            matched_label_rows=5,
            unsupported_label_rows=4,
            conflicting_training_key_rows=5,
        )


def test_should_require_every_row_in_every_f_episode_to_be_unseen_not_only_most(evaluator):
    train = shared._packets(evaluator.sources, context_sequence=("A",) * 4)
    data = raw_probe_data()
    packet = data[0][0]
    assert packet.episode is not None
    frames = (observation(0), *(f_frame("blue", step) for step in range(1, 10)))
    changed = replace(
        packet.episode,
        transitions=tuple(
            DemonstrationTransition(
                step, frames[step], 0, 0.5 if step == 8 else 0.0, frames[step + 1], step == 8, 7
            )
            for step in range(9)
        ),
    )
    data = ((pilot._packet(changed), *data[0][1:]), data[1])
    report = summarize_probe_support(train, data)
    assert report["all_canonical_inputs_unseen"] is False
    first, *others = report["probe"]["packets"]
    assert first["canonical"]["matched_input_rows"] == 1
    assert first["canonical"]["unmatched_input_rows"] == 8
    assert all(item["canonical"]["unmatched_input_rows"] == 9 for item in others)


@pytest.mark.parametrize(
    "fault",
    [
        "train_length",
        "train_role",
        "probe_length",
        "mixed_contexts",
        "labels",
        "history_instead_of_raw",
    ],
)
def test_should_refuse_malformed_support_cohorts_without_native_work(monkeypatch, evaluator, fault):
    train: Any = shared._packets(evaluator.sources, context_sequence=("A",) * 4)
    data: Any = raw_probe_data()
    if fault == "train_length":
        train = train[:-1]
    elif fault == "train_role":
        packet = train[-1]
        assert packet.episode is not None
        train = (
            *train[:-1],
            replace(
                packet, role="probe_train", episode=replace(packet.episode, role="probe_train")
            ),
        )
    elif fault == "probe_length":
        data = data[0][:-1], data[1]
    elif fault == "mixed_contexts":
        packet = data[0][-1]
        assert packet.episode is not None
        data = (
            (
                *data[0][:-1],
                replace(packet, context="E-red", episode=replace(packet.episode, context="E-red")),
            ),
            data[1],
        )
    elif fault == "labels":
        data[0][-1].targets[0, 0] = 7
    else:
        train = pilot._history_packets(train)
    monkeypatch.setattr(pilot, "_history_learner", forbidden)
    monkeypatch.setattr(pilot, "run_fresh_task_probe", forbidden)
    with pytest.raises(ValueError):
        summarize_probe_support(train, data)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonbool_distinct_mode_before_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**BASE, distinct_probe=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(**BASE, distinct_probe=value)


@pytest.mark.parametrize(
    "options",
    [
        dict(distinct_probe=True),
        dict(public_history=True, distinct_probe=True),
        dict(invariant_history=True, distinct_probe=True),
        dict(public_history=1, invariant_history=True, distinct_probe=True),
        dict(public_history=True, invariant_history=1, distinct_probe=True),
    ],
)
def test_should_require_exact_public_and_invariant_history_before_resources(monkeypatch, options):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**options)
    with pytest.raises(ValueError):
        pilot.run_pilot(**options)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(public_history=True),
        BASE,
        A_ONLY,
        dict(**BASE, pc_qualification=True),
        dict(**BASE, qualification_method="neutral_cpc", activation_health=True),
    ],
)
def test_should_preserve_previous_descriptors_when_false_is_explicit(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, distinct_probe=False
    )


def test_should_preserve_legacy_e_cohort_bytes_when_false_is_explicit(monkeypatch, evaluator):
    monkeypatch.setattr(ProbeEnvironment, "work", evaluator.work)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", ProbeEnvironment)
    monkeypatch.setattr(
        pilot, "CueKeyDoorTeacher", lambda: SimpleNamespace(choose_action=lambda public: 0)
    )
    default_reports: list[dict[str, Any]] = []
    false_reports: list[dict[str, Any]] = []
    ordinary = pilot._collect_probe_sources(lambda: None, default_reports)
    explicit = pilot._collect_probe_sources(lambda: None, false_reports, distinct_probe=False)
    assert pickle.dumps(ordinary) == pickle.dumps(explicit)
    assert pickle.dumps(default_reports) == pickle.dumps(false_reports)
    assert [packet.context for packet in ordinary[1]] == ["E-blue", "E-red"]


@pytest.mark.parametrize("method", METHODS)
def test_should_select_f_without_changing_the_original_learner_or_fixed_dose(monkeypatch, method):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    old = pilot.describe_protocol(**BASE, qualification_method=method)
    selected = pilot.describe_protocol(**BASE, qualification_method=method, distinct_probe=True)
    assert selected["methods"] == old["methods"] == [f"public_invariant_{method}"]
    for key in (
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "hidden_dim",
        "num_classes",
        "feature_dim_by_method",
        "cpc_configs",
        "task_sequence",
    ):
        assert selected[key] == old[key]
    assert [layout["context"] for layout in selected["probe_layouts"]] == ["F-blue", "F-red"]
    assert all(
        layout["rotation"] == 0
        and layout["branch_row"] == 2
        and layout["swap_keys"] is False
        and layout["wall_color"] == "grey"
        for layout in selected["probe_layouts"]
    )
    assert selected["planned_work"]["native_train_calls"] == 160
    assert selected["planned_work"]["training_rows"] == 1440
    assert selected["probe"]["updates"] == 16 and selected["probe"]["source_updates"] == [0, 128]
    assert selected["planned_work"]["final_episodes"] == 0


@pytest.mark.parametrize("options", [BASE, A_ONLY])
def test_should_preserve_default_bound_and_a_only_native_settings_when_selecting_f(
    monkeypatch, options
):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    old = pilot.describe_protocol(**options)
    selected = pilot.describe_protocol(**options, distinct_probe=True)
    for key in (
        "methods",
        "feature_dim_by_method",
        "trainable_parameters",
        "seed",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "cpc_configs",
        "task_sequence",
        "checkpoint_updates",
    ):
        assert selected[key] == old[key]
    for key in (
        "native_train_calls",
        "training_rows",
        "score_forwards",
        "score_rows",
        "teacher_episodes",
        "final_episodes",
        "learner_live_episodes",
    ):
        assert selected["planned_work"][key] == old["planned_work"][key]
    assert selected["probe"]["updates"] == old["probe"]["updates"] == 16


def test_should_omit_the_new_collector_keyword_on_the_false_route(monkeypatch, evaluator):
    case = environment(monkeypatch, evaluator, method="public_invariant_bp", acquisition_only=True)
    calls = []

    def old_collect(check_boundary, reports):
        calls.append(True)
        check_boundary()
        raise RuntimeError("stop at old collector call shape")

    monkeypatch.setattr(pilot, "_collect_probe_sources", old_collect)
    report = pilot.run_pilot(**A_ONLY, distinct_probe=False)
    assert report["status"] == "incomplete" and "old collector call shape" in report["reason"]
    assert calls == [True] and case.work["probes"] == []
    assert report["methods"][0]["charged_updates"] == 128


def test_should_finish_the_original_benchmark_then_check_support_and_run_separate_f_probes(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    support = report["probe_data"]["support"]
    assert support["status"] == "complete" and support["all_canonical_inputs_unseen"] is True
    assert all(
        packet["canonical"]["unmatched_input_rows"] == packet["rows"] == 9
        for packet in support["probe"]["packets"]
    )
    assert len(report["methods"]) == 1 and report["comparisons"] is None
    row = report["methods"][0]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [point for point, _, _ in case.work["fit"]] == list(POINTS)
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert len(case.native["trains"]) == 32 and len(case.native["predicts"]) == 46
    budgets = [call["budget"] for call in case.work["probes"]]
    assert budgets[0] is not budgets[1]
    for call in case.work["probes"]:
        assert (
            call["budget"] is not case.work["arms"][0]["budget"]
            and call["budget"].updates_completed == 16
        )
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
        assert [packet.context for packet in call["evaluation"]] == ["F-blue", "F-red"]
        assert [packet.context for packet in call["training"]] == ["F-blue", "F-red"] * 8
        assert call["report"]["completed_training_examples"] == 144
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert not case.native.get("environments") and not case.native.get("sessions")
    assert report["scientific_advantage_verified"] is False


def test_should_refuse_seen_support_before_probe_updates_and_preserve_all_benchmark_charges(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator, collision=True)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    support = report["probe_data"]["support"]
    assert support["status"] == "complete" and support["all_canonical_inputs_unseen"] is False
    assert all(item["canonical"]["matched_input_rows"] == 9 for item in support["probe"]["packets"])
    assert case.work["probes"] == [] and not case.native.get("trains")
    row = report["methods"][0]
    assert row["charged_updates"] == case.work["arms"][0]["budget"].updates_completed == 128
    assert row["native_training_examples"] == 1152
    assert all(
        slot["status"] == "not_run" and slot["charged_updates"] is None for slot in row["probes"]
    )
    assert report["comparisons"] is None


def test_should_preserve_prior_paid_f_probe_and_stop_after_corruption_during_the_next_update(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator)

    def corrupt(owner):
        if len(case.native.get("trains", [])) == 17:
            case.work["slots"][0]["report"]["completed_updates"] = 0

    case.native["on_train"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    first, second = row["probes"]
    assert first["status"] == first["report"]["status"] == "complete"
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert (
        second["status"] == "failed"
        and second["charged_updates"] == second["report"]["completed_updates"] == 1
    )
    assert second["report"]["completed_training_examples"] == 9
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert report["comparisons"] is None and len(case.native["trains"]) == 17
    assert report["probe_data"]["support"]["all_canonical_inputs_unseen"] is True


@pytest.mark.parametrize("fault", ["input_signature", "metadata_value", "equal_metadata_list"])
def test_should_refuse_changes_to_preceding_evidence_before_publishing_support(
    monkeypatch, evaluator, fault
):
    case = composition(monkeypatch, evaluator)
    original_loads = pilot.pickle.loads
    actual_summary = pilot.summarize_probe_support
    published_metadata: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    original_metadata: list[dict[str, Any]] = []

    def observe_loads(*args, **kwargs):
        value = original_loads(*args, **kwargs)
        # This is the existing evidence-publication boundary, not a stack hook.
        if (
            type(value) is dict
            and {"packets", "physical_episodes", "history_input_array_bytes"} <= value.keys()
        ):
            published_metadata.append(value)
        return value

    def corrupt_after_summary(training_packets, probe_data):
        summary = actual_summary(training_packets, probe_data)
        completed.append(original_loads(pickle.dumps(summary, protocol=5)))
        assert published_metadata
        owner = published_metadata[-1]
        original_metadata.append(original_loads(pickle.dumps(owner, protocol=5)))
        if fault == "input_signature":
            probe_data[0][0].features[0, 0] = 0.125
        elif fault == "metadata_value":
            owner["scope"] = "corrupt"
        else:
            # Equal values still replace a previously accepted container.
            owner["packets"] = list(owner["packets"])
        return summary

    monkeypatch.setattr(pilot.pickle, "loads", observe_loads)
    monkeypatch.setattr(pilot, "summarize_probe_support", corrupt_after_summary)
    report = pilot.run_pilot(**FLAGS)

    assert report["status"] == "incomplete" and report["reason"]
    assert len(completed) == len(original_metadata) == 1
    assert report["probe_data"] == original_metadata[0]
    assert "support" not in report["probe_data"]
    assert report["unpublished_probe_support"] == completed[0]
    assert report["work"]["probe_support"] == completed[0]["work"]
    assert completed[0]["status"] == "complete"
    assert completed[0]["work"]["history_rows"] == 1314
    assert completed[0]["work"]["native_updates"] == 0
    row = report["methods"][0]
    assert row["charged_updates"] == case.work["arms"][0]["budget"].updates_completed == 128
    assert row["native_training_examples"] == 1152
    assert case.work["fit"] == case.work["probes"] == []
    assert not case.native.get("trains") and not case.native.get("predicts")
    assert all(
        slot["status"] == "not_run" and slot["charged_updates"] is None for slot in row["probes"]
    )
    assert report["comparisons"] is None


def test_should_admit_against_the_captured_support_record_not_the_helpers_mutable_alias(
    monkeypatch, evaluator
):
    case = composition(monkeypatch, evaluator, collision=True)
    actual_summary = pilot.summarize_probe_support
    original_check = shared._check_signature
    aliases: list[dict[str, Any]] = []

    def remember_summary(training_packets, probe_data):
        summary = actual_summary(training_packets, probe_data)
        assert summary["all_canonical_inputs_unseen"] is False
        aliases.append(summary)
        return summary

    def mutate_alias_at_next_boundary(*args, **kwargs):
        result = original_check(*args, **kwargs)
        if aliases:
            aliases[0]["all_canonical_inputs_unseen"] = True
            for packet in aliases[0]["probe"]["packets"]:
                packet["canonical"]["unmatched_input_rows"] = 9
        return result

    monkeypatch.setattr(pilot, "summarize_probe_support", remember_summary)
    monkeypatch.setattr(shared, "_check_signature", mutate_alias_at_next_boundary)
    report = pilot.run_pilot(**FLAGS)

    assert report["status"] == "incomplete" and report["reason"]
    assert len(aliases) == 1 and aliases[0]["all_canonical_inputs_unseen"] is True
    proof = report["probe_data"]["support"]
    assert proof["all_canonical_inputs_unseen"] is False
    assert all(
        packet["canonical"]["matched_input_rows"] == 9
        and packet["canonical"]["unmatched_input_rows"] == 0
        for packet in proof["probe"]["packets"]
    )
    assert "unpublished_probe_support" not in report
    assert report["work"]["probe_support"] == proof["work"]
    row = report["methods"][0]
    assert row["charged_updates"] == case.work["arms"][0]["budget"].updates_completed == 128
    assert row["native_training_examples"] == 1152
    assert case.work["fit"] == case.work["probes"] == []
    assert not case.native.get("trains") and not case.native.get("predicts")
    assert all(
        slot["status"] == "not_run" and slot["charged_updates"] is None for slot in row["probes"]
    )
