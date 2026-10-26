"""Two complete native weighted online cases with synthetic TRAIN/F sources.

No physics/teacher or comparative learning claim. Each case pays128weightedTRAIN
updates plus two independent F16 and A16 paths, and ordinary unweighted readouts.
"""

from copy import deepcopy

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_weighted_feature_route as route
from src.adapters.invariant_torch_feature_learner import InvariantTorchFeatureLearner
from src.adapters.torch_feature_learner import TORCH_PC_ENERGY_ID
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from src.adapters.torch_pc_weighted import TORCH_INVARIANT_WEIGHTED_PC_ID
from src.adapters.torch_train_weighted_learner import TorchTrainWeightedLearner
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_run_weighted_online_train_then_ordinary_isolated_f_and_return(monkeypatch, method):
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    reserved = raw_probe_data()
    arms, templates, calls = [], [], []
    train_arm, make_probe = pilot.shared._train_arm, route.probe_arm
    ordinary_train = InvariantTorchFeatureLearner.train_batch

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic weighted API check forbids teacher and game physics")

    def arm(*args, **kwargs):
        result = train_arm(*args, **kwargs)
        arms.append(result)
        return result

    def template(original, **kwargs):
        result = make_probe(original, **kwargs)
        assert type(result["source"]) is InvariantTorchFeatureLearner
        assert type(original["source"]) is TorchTrainWeightedLearner
        before = deepcopy(result["source"].snapshot_state())
        templates.append((result["source"], before))
        return result

    def unweighted(owner, features, targets):
        result = ordinary_train(owner, features, targets)
        calls.append((owner, len(features), result.definition))
        return result

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(
        pilot, "_collect_probe_sources", lambda boundary, reports, **kwargs: reserved
    )
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    monkeypatch.setattr(route, "probe_arm", template)
    monkeypatch.setattr(InvariantTorchFeatureLearner, "train_batch", unweighted)
    report = pilot.run_pilot(
        public_history=True,
        invariant_history=True,
        qualification_method=method,
        weighted_pc=True,
        distinct_probe=True,
        old_task_readouts=True,
        return_to_a=True,
    )
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == f"public_invariant_{method}"
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 1152
    receipts = row["stream_result"]["applied_updates"]
    assert len(receipts) == 128 and all(
        record["diagnostic"]["definition"] == TORCH_INVARIANT_WEIGHTED_PC_ID for record in receipts
    )
    assert [record["updates"] for record in row["readouts"]] == [0, 32, 64, 96, 128]
    assert all(record["status"] == "complete" for record in row["readouts"])
    for slot in row["probes"]:
        assert slot["status"] == "complete", slot["reason"]
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert len(arms) == 1 and len(templates) == 2 and len(calls) == 64
    assert all(rows == 9 and definition == TORCH_PC_ENERGY_ID for _, rows, definition in calls)
    assert all(owner is not source for owner, _, _ in calls for source, _ in templates)
    for source, before in templates:
        assert _same_state(source.snapshot_state(), before)
        assert before.completed_updates == before.training_examples == 0
    arm_result = arms[0]
    arm_result["guard"]()
    route.check_receipts(arm_result, method=method)
    initial = arm_result["source"].snapshot_state()
    assert initial.completed_updates == initial.training_examples == 0
    assert _same_state(initial, arm_result["stream"]._runtime.actor.snapshot_state().state)
    assert report["work"]["native_train_calls"] == 192
    assert report["work"]["training_rows"] == 1728
    assert row["final_evaluation"]["status"] == "not_run"
    assert report["scientific_advantage_verified"] is False and report["comparisons"] is None
