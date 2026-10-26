"""Matched A-only recipe availability, not evidence of native learning benefit.

The original accepted TRAIN stub preserves 128 paid updates and all checkpoints.
Real fit, F-support, monitored-probe and A-return use-cases execute against fake
native ports. Historical BP descriptors are reused directly as immutable inputs.
"""

from copy import deepcopy
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from test_minigrid_a_only_acquisition import OLD_CLAIMS
from test_minigrid_distinct_probe import FProbeEnvironment, METHODS
from test_minigrid_learning_pilot import POINTS, evaluator as evaluator
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


BASE: dict[str, Any] = dict(public_history=True, invariant_history=True, acquisition_only=True)
RETURN: dict[str, Any] = dict(**BASE, distinct_probe=True, old_task_readouts=True, return_to_a=True)
ROOT = Path(__file__).resolve().parents[1]


def matched_environment(monkeypatch, evaluator, method):
    case = environment(
        monkeypatch, evaluator, method=f"public_invariant_{method}", acquisition_only=True
    )
    monkeypatch.setattr(FProbeEnvironment, "work", case.native)
    monkeypatch.setattr(FProbeEnvironment, "collision", False)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", FProbeEnvironment)
    return case


@pytest.mark.parametrize("method", METHODS)
def test_should_offer_the_same_fixed_a_curriculum_with_each_original_native_configuration(
    monkeypatch, method
):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    previous = pilot.describe_protocol(
        public_history=True, invariant_history=True, qualification_method=method
    )
    selected = pilot.describe_protocol(**BASE, qualification_method=method)
    assert selected["protocol_id"] == f"minigrid_invariant_{method}_A_only_acquisition_v1"
    assert selected["methods"] == [f"public_invariant_{method}"]
    assert selected["task_sequence"] == ["A"] * 4
    assert selected["stream"]["label_delay"] == previous["stream"]["label_delay"] == 1
    assert selected["checkpoint_updates"] == previous["checkpoint_updates"] == list(POINTS)
    for key in (
        "seed",
        "feature_dim_by_method",
        "hidden_dim",
        "num_classes",
        "learning_rate",
        "momentum",
        "inference_steps",
        "inference_learning_rate",
        "trainable_parameters",
        "cpc_configs",
    ):
        assert selected[key] == previous[key]
    for key in (
        "native_train_calls",
        "training_rows",
        "score_forwards",
        "score_rows",
        "teacher_episodes",
        "final_episodes",
        "learner_live_episodes",
        "maximum_source_environment_steps",
    ):
        assert selected["planned_work"][key] == previous["planned_work"][key]
    assert selected["probe"]["source_stages"] == ["initial", "post_A128"]
    assert selected["primary_comparisons"] == []
    assert selected["cpc_configs"] == (
        {method: shared.describe_protocol()["cpc_configs"][method]}
        if method in {"neutral_cpc", "gated_cpc"}
        else {}
    )


@pytest.mark.parametrize(
    "stage,extra",
    [
        ("minigrid-bp-a-only-20261017", {}),
        ("minigrid-distinct-probe-20261017", {"distinct_probe": True}),
        ("minigrid-probe-old-fit-20261017", {"distinct_probe": True, "old_task_readouts": True}),
        (
            "minigrid-a-return-20261017",
            {"distinct_probe": True, "old_task_readouts": True, "return_to_a": True},
        ),
    ],
)
def test_should_keep_each_accepted_bp_descriptor_exactly_equal_to_its_saved_record(
    monkeypatch, stage, extra
):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    path = ROOT / "artifacts" / "runs" / stage / "protocol.json"
    original = path.read_bytes()
    expected = json.loads(original)
    assert pilot.describe_protocol(**BASE, qualification_method="bp", **extra) == expected
    assert path.read_bytes() == original


@pytest.mark.parametrize("method", METHODS)
def test_should_preserve_original_curricula_and_descriptors_when_acquisition_is_false(method):
    options = dict(public_history=True, invariant_history=True, qualification_method=method)
    ordinary = pilot.describe_protocol(**options)
    explicit = pilot.describe_protocol(**options, acquisition_only=False)
    assert ordinary == explicit
    assert ordinary["task_sequence"] == ["A", "B", "C_train", "A"]


@pytest.mark.parametrize("method", METHODS)
def test_should_run_matched_a_packets_f_probes_and_a_returns_without_changing_exposure_or_isolation(
    monkeypatch, evaluator, method
):
    case = matched_environment(monkeypatch, evaluator, method)
    accepted_train = shared._train_arm
    actual_packets = []
    sources_before = pickle.dumps(evaluator.sources)
    expected = pilot._history_packets(evaluator.sources[:2])

    def check_packets(row, packets, sampler, deadline, check_boundary, **options):
        actual_packets.extend(packets)
        assert row["method"] == f"public_invariant_{method}"
        assert len(packets) == 128
        assert len({identity for packet in packets for identity in packet.row_ids}) == 1152
        for index, packet in enumerate(packets):
            template = expected[index % 2]
            assert packet.role == "train" and packet.context == "A" and packet.num_classes == 7
            assert packet.episode is not None
            assert packet.episode.episode_id == f"minigrid-training-{index}"
            assert packet.row_ids == tuple(
                f"minigrid-training-{index}:step:{step}" for step in range(9)
            )
            np.testing.assert_array_equal(packet.features, template.features)
            np.testing.assert_array_equal(packet.targets, template.targets)
            assert not np.shares_memory(packet.features, template.features)
        return accepted_train(row, packets, sampler, deadline, check_boundary, **options)

    monkeypatch.setattr(shared, "_train_arm", check_packets)
    report = pilot.run_pilot(**RETURN, qualification_method=method)
    assert report["status"] == "complete", report["reason"]
    assert report["protocol"]["methods"] == [f"public_invariant_{method}"]
    assert report["comparisons"] is None and len(actual_packets) == 128
    assert case.work["factory_calls"] == [f"public_invariant_{method}"]
    row = report["methods"][0]
    assert row["method"] == f"public_invariant_{method}"
    assert row["charged_updates"] == case.work["arms"][0]["budget"].updates_completed == 128
    assert row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["A_acquisition"] == dict(A_at32=False, A_at128=False)
    assert row["A_fit_gains"] == dict(ce=0.0, accuracy=0.0, exact_sequence_accuracy=0.0)
    assert not any(key in row for key in OLD_CLAIMS)
    assert [point["updates"] for point in row["outcomes"]] == list(POINTS)
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert [point for point, _, _ in case.work["fit"]] == list(POINTS)
    assert len(case.native["trains"]) == 64 and len(case.native["predicts"]) == 94
    for offset in (0, 32):
        assert all(
            np.all(targets == 0) for _, _, targets in case.native["trains"][offset : offset + 16]
        )
        assert all(
            np.all(targets == 1)
            for _, _, targets in case.native["trains"][offset + 16 : offset + 32]
        )
    for slot in row["probes"]:
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["report"]["completed_training_examples"] == 144
        returned = slot["report"]["return_phase"]
        assert returned["status"] == "complete" and returned["completed_updates"] == 16
        assert returned["completed_training_examples"] == 144
        assert returned["checkpoints"][0]["scores"] == slot["report"]["checkpoints"][-1]["scores"]
        assert (
            returned["training_readouts"]["checkpoints"][0]["scores"]
            == slot["report"]["training_readouts"]["checkpoints"][-1]["scores"]
        )
        assert [point["updates"] for point in slot["return_fit_outcomes"]["A"]["curve"]] == [
            0,
            4,
            8,
            16,
        ]
    assert report["work"]["native_train_calls"] == 192 and report["work"]["training_rows"] == 1728
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    assert report["probe_data"]["support"]["all_canonical_inputs_unseen"] is True
    assert pickle.dumps(evaluator.sources) == sources_before
    for call in case.work["probes"]:
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
        assert call["budget"] is not case.work["arms"][0]["budget"]
    assert not case.native.get("sessions") and not case.native.get("environments")
    assert report["scientific_advantage_verified"] is False


@pytest.mark.parametrize("method", METHODS[1:])
def test_should_refuse_replaced_paid_return_evidence_for_each_new_a_only_method(
    monkeypatch, evaluator, method
):
    case = matched_environment(monkeypatch, evaluator, method)

    def replace_paid(owner):
        if len(case.native.get("trains", [])) == 33:
            slot = case.work["slots"][0]
            slot["report"]["return_phase"] = deepcopy(slot["report"]["return_phase"])

    case.native["on_train"] = replace_paid
    report = pilot.run_pilot(**RETURN, qualification_method=method)
    assert report["status"] == "incomplete" and report["reason"]
    row = report["methods"][0]
    first, second = row["probes"]
    assert row["method"] == f"public_invariant_{method}"
    assert (
        first["status"] == "complete"
        and first["charged_updates"] == first["return_charged_updates"] == 16
    )
    assert first["report"]["return_phase"]["completed_updates"] == 16
    assert second["status"] == "failed" and second["charged_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert second["report"]["return_phase"]["status"] == "not_run"
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert len(case.native["trains"]) == 33 and report["comparisons"] is None
