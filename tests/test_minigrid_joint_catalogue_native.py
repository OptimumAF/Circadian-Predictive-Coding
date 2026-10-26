"""One real BP public API case, with synthetic complete TRAIN/F sources.

No teacher or game executes. All192native updates7488rows and94scores846rows are
engineering work; metrics are not acquisition or scientific benefit evidence.
"""

from copy import deepcopy

from src.adapters import minigrid_learnability as pilot
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import FLAGS


def test_should_run_public_joint_stream_readouts_and_isolated_return_without_final_data(
    monkeypatch,
):
    raw = tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )
    probe = raw_probe_data()
    owners, originals, arms = [], [], []
    build = pilot._history_learner
    train = pilot.shared._train_arm

    def learner(method):
        owned = build(method)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def native_arm(*args, **kwargs):
        assert kwargs["catalogue_rows"] == 54
        arm = train(*args, **kwargs)
        arms.append(arm)
        return arm

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic API test forbids native physics and teacher collection")

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda boundary, reports, **kwargs: probe)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", native_arm)
    report = pilot.run_pilot(**FLAGS, joint_catalogue=True)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == "public_invariant_bp" and row["status"] == "complete"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["partial_stream"]["committed_training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert "signed_metrics" not in row and "acquisition_conditions" not in row
    assert type(row["joint_catalogue_acquired"]) is bool
    assert [r["updates"] for r in row["readouts"]] == [0, 32, 64, 96, 128]
    for raw_score in row["readouts"]:
        assert raw_score["status"] == "complete" and len(raw_score["scores"]) == 6
        assert raw_score["expected_training_examples"] == 54 * raw_score["updates"]
    for slot in row["probes"]:
        assert slot["status"] == "complete"
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["expected_training_examples"] == 54 * slot["checkpoint_updates"]
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
    assert report["catalogue_inputs"]["row_exposures"] == 6912
    assert report["catalogue_inputs"]["input_array_bytes"] == 54 * 297 * 8 * 128
    support = report["probe_data"]["support"]
    assert support["work"]["canonical_projection_rows"] == 216
    assert all(p["canonical"]["unmatched_input_rows"] == 9 for p in support["probe"]["packets"])
    assert report["work"]["native_train_calls"] == 192
    assert report["work"]["training_rows"] == 7488
    assert report["work"]["total_score_forwards_completed"] == 94
    assert report["work"]["total_score_rows_completed"] == 846
    assert report["scientific_advantage_verified"] is False
    assert report["comparisons"] is None
    assert row["final_evaluation"]["status"] == "not_run"
    assert len(owners) == 1 and _same_state(owners[0].snapshot_state(), originals[0])
    arms[0]["guard"]()
