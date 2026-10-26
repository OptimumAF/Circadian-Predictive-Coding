"""A-only curriculum admission and existing guarded fit/probe composition.

Packet tests use synthetic nine-row sources. Composition reuses the accepted
128-update TRAIN stub plus real fit/probe helpers with fake native ports; no
neural learning, teacher physics or scientific acquisition result is executed.
"""

from dataclasses import replace
import json
import pickle
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_learning_pilot as shared
from test_minigrid_learning_pilot import POINTS, evaluator as evaluator
from test_minigrid_pc_qualification import environment
from test_torch_feature_rehearsal_pilot import forbidden


METHOD = "public_invariant_bp"
BASE: dict[str, Any] = dict(public_history=True, invariant_history=True, qualification_method="bp")
FLAGS: dict[str, Any] = dict(**BASE, acquisition_only=True)
ORIGINAL = ("A", "B", "C_train", "A")
A_ONLY = ("A",) * 4
OLD_CLAIMS = ("signed_metrics", "acquisition_conditions", "initial_A_acquired", "probe_gain_change")


def test_should_keep_default_packets_identical_to_the_original_explicit_curriculum(evaluator):
    before = pickle.dumps(evaluator.sources)
    ordinary = shared._packets(evaluator.sources)
    explicit = shared._packets(evaluator.sources, context_sequence=ORIGINAL)
    assert pickle.dumps(ordinary) == pickle.dumps(explicit)
    assert [packet.context for packet in ordinary] == ["A"] * 32 + ["B"] * 32 + ["C_train"] * 32 + [
        "A"
    ] * 32
    assert pickle.dumps(evaluator.sources) == before


def test_should_own_128_unique_alternating_a_copies_without_touching_the_six_original_sources(
    evaluator,
):
    sources = evaluator.sources
    before = pickle.dumps(sources)
    packets = shared._packets(sources, context_sequence=A_ONLY)
    assert len(packets) == 128
    assert len({shared._episode(packet).episode_id for packet in packets}) == 128
    assert len({identity for packet in packets for identity in packet.row_ids}) == 1152
    arrays: list[Any] = []
    for index, packet in enumerate(packets):
        assert packet.episode is not None
        original = sources[index % 2]
        assert original.episode is not None
        assert packet.role == packet.episode.role == "train"
        assert packet.context == packet.episode.context == "A"
        assert packet.num_classes == packet.episode.num_classes == 7
        assert packet.episode is not original.episode
        assert packet.episode.episode_id == f"minigrid-training-{index}"
        assert packet.row_ids == tuple(
            f"minigrid-training-{index}:step:{step}" for step in range(9)
        )
        assert packet.features.shape == (9, 148) and packet.targets.shape == (9, 1)
        assert packet.features.dtype == packet.targets.dtype == np.float64
        assert packet.features.flags.owndata and packet.targets.flags.owndata
        np.testing.assert_array_equal(packet.features, original.features)
        np.testing.assert_array_equal(packet.targets, original.targets)
        arrays.extend((packet.features, packet.targets))
    for index, left in enumerate(arrays):
        assert all(not np.shares_memory(left, right) for right in arrays[index + 1 :])
        assert all(
            not np.shares_memory(left, value)
            for source in sources
            for value in (source.features, source.targets)
        )
    other_before = packets[2].features.copy()
    packets[0].features[0, 0] = 0.123
    packets[0].targets[0, 0] = 6.0
    np.testing.assert_array_equal(packets[2].features, other_before)
    assert pickle.dumps(sources) == before


class StringSubclass(str):
    pass


class TupleSubclass(tuple):
    pass


@pytest.mark.parametrize(
    "sequence",
    [
        None,
        [],
        list(ORIGINAL),
        TupleSubclass(ORIGINAL),
        (),
        ("A",),
        ("A",) * 5,
        ("B", "A", "C_train", "A"),
        ("A", "B", "C_train", "B"),
        ("A", "A", "A", "D"),
        ("A", True, "A", "A"),
        (StringSubclass("A"), "A", "A", "A"),
    ],
)
def test_should_refuse_unfrozen_sequences_before_packet_validation_or_copies(
    monkeypatch, evaluator, sequence
):
    monkeypatch.setattr(shared, "_validate_batch", forbidden)
    monkeypatch.setattr(shared, "_batch", forbidden)
    with pytest.raises(ValueError):
        shared._packets(evaluator.sources, context_sequence=sequence)


@pytest.mark.parametrize("fault", ["list", "only_a_pair", "reversed", "last_role", "last_shape"])
def test_should_admit_all_six_sources_before_creating_any_a_only_packet(
    monkeypatch, evaluator, fault
):
    sources: Any = evaluator.sources
    if fault == "list":
        sources = list(sources)
    elif fault == "only_a_pair":
        sources = sources[:2]
    elif fault == "reversed":
        sources = tuple(reversed(sources))
    elif fault == "last_role":
        packet = sources[-1]
        assert packet.episode is not None
        sources = (
            *sources[:-1],
            replace(packet, role="probe_test", episode=replace(packet.episode, role="probe_test")),
        )
    else:
        sources = (
            *sources[:-1],
            replace(sources[-1], features=sources[-1].features[:, :147].copy()),
        )
    monkeypatch.setattr(shared, "_batch", forbidden)
    with pytest.raises(ValueError):
        shared._packets(sources, context_sequence=A_ONLY)


def test_should_describe_a_single_task_without_continual_or_generalization_claims(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "_history_learner", "MiniGridKeyDoor"):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol(**FLAGS)
    assert protocol["protocol_id"] == "minigrid_invariant_bp_A_only_acquisition_v1"
    assert protocol["methods"] == [METHOD] and protocol["task_sequence"] == list(A_ONLY)
    assert protocol["checkpoint_updates"] == list(POINTS)
    assert protocol["primary_comparisons"] == [] and protocol["cpc_configs"] == {}
    assert protocol["probe"]["source_updates"] == [0, 128]
    assert protocol["probe"]["updates"] == 16
    assert protocol["probe"]["source_stages"] == ["initial", "post_A128"]
    assert not {
        "initial_A_gain",
        "new_B_gain",
        "new_C_gain",
        "old_A_loss",
        "return_A_gain",
        "return_gap",
    }.intersection(protocol.get("differences") or {})
    for key, expected in dict(
        benchmark_native_train_calls=128,
        benchmark_training_rows=1152,
        probe_native_train_calls=32,
        probe_training_rows=288,
        native_train_calls=160,
        training_rows=1440,
        known_raw_score_forwards=30,
        known_raw_score_rows=270,
        probe_score_forwards=16,
        probe_score_rows=144,
        score_forwards=46,
        score_rows=414,
        teacher_episodes=10,
        maximum_source_environment_steps=160,
        learner_live_episodes=0,
        final_episodes=0,
        replay_rows=0,
    ).items():
        assert protocol["planned_work"][key] == expected
    assert "activation_health" not in protocol
    json.dumps(protocol, allow_nan=False)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], np.bool_(True)])
def test_should_refuse_nonbool_acquisition_mode_before_resources(monkeypatch, value):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(**BASE, acquisition_only=value)
    with pytest.raises(ValueError):
        pilot.run_pilot(**BASE, acquisition_only=value)


@pytest.mark.parametrize(
    "options",
    [
        dict(acquisition_only=True),
        dict(public_history=True, acquisition_only=True),
        dict(public_history=True, invariant_history=True, acquisition_only=True),
        dict(
            public_history=True,
            invariant_history=True,
            qualification_method="recurrent_bc",
            acquisition_only=True,
        ),
        dict(
            public_history=True,
            invariant_history=True,
            qualification_method="unsupported",
            acquisition_only=True,
        ),
        dict(
            public_history=True,
            invariant_history=True,
            qualification_method=1,
            acquisition_only=True,
        ),
        dict(
            public_history=True,
            invariant_history=True,
            pc_qualification=True,
            acquisition_only=True,
        ),
        dict(**FLAGS, activation_health=True),
        dict(**FLAGS, gradient_geometry=True),
    ],
)
def test_should_refuse_unfrozen_routes_before_any_native_resources(monkeypatch, options):
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
        dict(public_history=True, invariant_history=True),
        BASE,
        dict(public_history=True, invariant_history=True, pc_qualification=True),
        dict(**BASE, activation_health=True),
    ],
)
def test_should_preserve_every_previous_descriptor_when_false_is_explicit(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, acquisition_only=False
    )


def test_should_compute_signed_a_only_fit_changes_without_reintroducing_continual_claims():
    points = []
    for point in POINTS:
        scores = []
        for context in ("A", "B", "C_train"):
            for cue in ("blue", "red"):
                ce, accuracy, exact = 0.1, 1.0, 1.0
                if context == "A" and point == 0 and cue == "red":
                    ce, accuracy, exact = 1.5, 0.4, 0.0
                elif context == "A" and point in (64, 96, 128):
                    ce, accuracy, exact = 1.2, 0.4, 0.0
                scores.append(
                    dict(
                        context=context,
                        cue=cue,
                        ce=ce,
                        accuracy=accuracy,
                        exact_sequence_accuracy=exact,
                    )
                )
        points.append(dict(updates=point, scores=scores, status="complete"))
    before = pickle.dumps(points)
    outcome = pilot._a_only_fit_outcomes(points)
    assert outcome["A_acquisition"] == dict(A_at32=True, A_at128=False)
    assert outcome["A_fit_gains"]["ce"] == pytest.approx(-0.4)
    assert outcome["A_fit_gains"]["accuracy"] == pytest.approx(-0.3)
    assert outcome["A_fit_gains"]["exact_sequence_accuracy"] == pytest.approx(-0.5)
    assert [point["updates"] for point in outcome["outcomes"]] == list(POINTS)
    assert all(
        point["B"]["exact_sequence_accuracy"] == point["C_train"]["exact_sequence_accuracy"] == 1
        for point in outcome["outcomes"]
    )
    assert not any(key in outcome for key in OLD_CLAIMS)
    assert (
        outcome["interpretation"] == "single_task_A_fit_only_no_retention_or_generalization_claim"
    )
    assert pickle.dumps(points) == before


def test_should_execute_only_a_training_and_keep_all_context_readouts_and_two_isolated_probes(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, method=METHOD, acquisition_only=True)
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "complete", report["reason"]
    assert report["comparisons"] is None and len(report["methods"]) == 1
    row = report["methods"][0]
    assert row["method"] == METHOD and row["status"] == "complete"
    assert all(packet.context == "A" for packet in case.captured["streams"][0][2])
    assert [source["context"] for source in report["training_sources"]] == [
        "A",
        "A",
        "B",
        "B",
        "C_train",
        "C_train",
    ]
    assert [point["updates"] for point in row["outcomes"]] == list(POINTS)
    assert all(set(point) == {"updates", "A", "B", "C_train"} for point in row["outcomes"])
    assert all(len(readout["scores"]) == 6 for readout in row["readouts"])
    assert not any(key in row for key in OLD_CLAIMS)
    assert row["A_acquisition"] == dict(A_at32=False, A_at128=False)
    assert row["A_fit_gains"] == dict(ce=0.0, accuracy=0.0, exact_sequence_accuracy=0.0)
    assert row["interpretation"] == "single_task_A_fit_only_no_retention_or_generalization_claim"
    assert row["post_A128_probe_gain_change"] == 0
    assert row["probe_source_stages"] == ["initial", "post_A128"]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [call["state"].completed_updates for call in case.work["probes"]] == [0, 128]
    assert len(case.native["trains"]) == 32 and len(case.native["predicts"]) == 46
    budgets = [call["budget"] for call in case.work["probes"]]
    assert budgets[0] is not budgets[1]
    assert all(
        budget.updates_completed == budget.budget.max_training_updates == 16 for budget in budgets
    )
    assert all(budget is not case.work["arms"][0]["budget"] for budget in budgets)
    for call in case.work["probes"]:
        assert call["source_before"] == call["source_after"]
        assert (
            call["state_before"] == call["state_after"]
            and call["inputs_before"] == call["inputs_after"]
        )
    assert report["work"]["native_train_calls"] == 160 and report["work"]["training_rows"] == 1440
    assert report["scientific_advantage_verified"] is False


def test_should_preserve_paid_probe_prefix_and_omit_continual_claims_after_corruption(
    monkeypatch, evaluator
):
    case = environment(monkeypatch, evaluator, method=METHOD, acquisition_only=True)

    def corrupt(owner):
        if len(case.native.get("trains", [])) == 17:
            case.work["slots"][0]["report"]["completed_updates"] = 0

    case.native["on_train"] = corrupt
    report = pilot.run_pilot(**FLAGS)
    assert report["status"] == "incomplete" and report["reason"]
    assert report["comparisons"] is None
    row = report["methods"][0]
    first, second = row["probes"]
    assert first["status"] == first["report"]["status"] == "complete"
    assert first["charged_updates"] == first["report"]["completed_updates"] == 16
    assert second["status"] == "failed"
    assert second["charged_updates"] == second["report"]["completed_updates"] == 1
    assert second["report"]["completed_training_examples"] == 9
    assert len(case.native["trains"]) == 17 and len(case.work["fit"]) == 5
    assert not any(key in row for key in OLD_CLAIMS)
    assert row["A_fit_gains"] is row["A_acquisition"] is None
    assert "post_A128_probe_gain_change" not in row
    assert row["probe_source_stages"] == ["initial", "post_A128"]
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert report["work"]["native_train_calls"] == 145 and report["work"]["training_rows"] == 1305
