"""Two bounded public native routes; synthetic complete sources, no physics."""

from copy import deepcopy
import pickle
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import FLAGS
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from test_minigrid_pc_update_route import install_reader, sources
from test_minigrid_learning_pilot import Learner, same_state


@pytest.mark.parametrize("fault", ["work_owner", "work_type", "source_ids_owner", "rows_type"])
def test_should_refuse_post_helper_callback_corruption_and_keep_paid_prefix(monkeypatch, fault):
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(pilot, "_same_state", same_state)
    monkeypatch.setattr(pilot.gradient_readouts, "_same_state", same_state)
    monkeypatch.setattr(
        pilot, "_invariant_snapshot", lambda state, *args, **kwargs: deepcopy(state)
    )
    install_reader(monkeypatch)
    original = pilot.gradient_readouts.run_pc_update_readouts
    published = None
    changed = False

    def observe(report, *args, **kwargs):
        nonlocal published
        result = original(report, *args, **kwargs)
        published = report
        return result

    def boundary():
        nonlocal changed
        if published is None or changed:
            return
        changed = True
        if fault == "work_owner":
            published["work"]["pc_update_diagnostics"] = deepcopy(
                published["work"]["pc_update_diagnostics"]
            )
        elif fault == "work_type":
            published["work"]["pc_update_diagnostics"]["native_train_calls"] = False
        elif fault == "source_ids_owner":
            diagnostic = published["pc_update_diagnostics"][0]["diagnostic"]
            diagnostic["source_episode_ids"] = list(diagnostic["source_episode_ids"])
        else:
            published["pc_update_diagnostics"][0]["diagnostic"]["rows"] = 54.0

    monkeypatch.setattr(pilot.gradient_readouts, "run_pc_update_readouts", observe)
    source = Learner(SimpleNamespace(method="pc"))
    raw = pilot._score_checkpoint(
        source,
        source.snapshot_state(),
        method="public_invariant_pc",
        updates=0,
        sources=sources(),
        budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 1.0),
        check_boundary=boundary,
        pc_update_diagnostics=True,
        catalogue_rows=54,
    )
    assert changed and raw["status"] == "failed" and "evidence changed" in raw["reason"]
    assert len(raw["scores"]) == 6
    assert raw["work"]["score_rows_completed"] == 54
    assert type(raw["work"]["pc_update_diagnostics"]["native_train_calls"]) is int
    assert type(raw["pc_update_diagnostics"][0]["diagnostic"]["rows"]) is int
    assert raw["work"]["pc_update_diagnostics"]["catalogue_calls_completed"] == 1


@pytest.mark.parametrize("method", ["pc", "neutral_cpc"])
def test_should_run_joint_checkpoints_and_isolated_return_with_paid_pc_readouts(
    monkeypatch, method
):
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    probes = raw_probe_data()
    inputs = pickle.dumps((raw, probes))
    owners, originals, arms = [], [], []
    factory, train = pilot._history_learner, pilot.shared._train_arm

    def learner(name):
        owned = factory(name)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def arm(*args, **kwargs):
        assert kwargs["catalogue_rows"] == 54
        result = train(*args, **kwargs)
        arms.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic engineering route forbids physics/teacher")

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda boundary, reports, **kwargs: probes)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    report = pilot.run_pilot(
        **{**FLAGS, "qualification_method": method},
        joint_catalogue=True,
        pc_update_diagnostics=True,
    )
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == "public_invariant_" + method
    assert row["charged_updates"] == 128 and row["native_training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [r["updates"] for r in row["readouts"]] == [0, 32, 64, 96, 128]
    for readout in row["readouts"]:
        assert readout["status"] == "complete" and len(readout["scores"]) == 6
        assert readout["expected_training_examples"] == 54 * readout["updates"]
        if readout["updates"] not in (0, 128):
            assert "pc_update_diagnostics" not in readout
            assert "pc_update_diagnostics" not in readout["work"]
            continue
        records = readout["pc_update_diagnostics"]
        assert len(records) == 1 and records[0]["context"] == "joint_train_catalogue"
        diagnostic = records[0]["diagnostic"]
        assert diagnostic["status"] == "complete" and diagnostic["method"] == method
        assert diagnostic["rows"] == 54 and len(diagnostic["relaxation"]) == 3
        assert diagnostic["source_episode_ids"] == records[0]["episode_ids"]
        work = readout["work"]["pc_update_diagnostics"]
        assert work["catalogue_calls_completed"] == 1 and work["train_rows_admitted"] == 54
        assert work["projection_rows_completed"] == 54
        assert work["prior_forward_rows_completed"] == 108
        assert work["prior_backward_rows_completed"] == 54
        assert work["relaxation_forward_rows_completed"] == 162
        assert (
            work["native_train_calls"] == work["optimizer_steps"] == work["published_updates"] == 0
        )
    for slot in row["probes"]:
        assert slot["status"] == "complete"
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
    work = report["work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["total_score_forwards_completed"] == 94
    assert work["total_score_rows_completed"] == 846
    extra = work["pc_update_diagnostics"]
    assert extra["catalogue_calls_attempted"] == extra["catalogue_calls_completed"] == 2
    assert extra["projection_rows_completed"] == extra["train_rows_admitted"] == 108
    assert extra["prior_forward_rows_completed"] == 216
    assert extra["prior_backward_rows_completed"] == 108
    assert extra["relaxation_forward_rows_completed"] == 324
    assert extra["geometry_reductions_completed"] == 10
    assert extra["failure_work_may_be_unobserved"] is False
    assert row["work"]["pc_update_diagnostics"] == extra
    assert row["final_evaluation"]["status"] == "not_run"
    assert report["scientific_advantage_verified"] is False
    assert len(owners) == 1 and _same_state(owners[0].snapshot_state(), originals[0])
    assert pickle.dumps((raw, probes)) == inputs
    arms[0]["guard"]()
