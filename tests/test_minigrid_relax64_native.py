"""Native fixed64 recipe parity and complete synthetic public routes, no physics.

These are engineering checks, not acquisition or comparative benefit evidence.
"""

from copy import deepcopy
import pickle

import pytest

from src.adapters import minigrid_learnability as pilot
from src.adapters import minigrid_relaxation_control as control
from src.adapters.torch_feature_rehearsal_pilot import _same_state
from test_minigrid_distinct_probe import raw_probe_data
from test_minigrid_invariant_learnability import episode
from test_minigrid_joint_catalogue_route import FLAGS


def training_sources():
    return tuple(
        pilot._packet(episode(context, cue, "train", f"minigrid-source-{context}-{cue}"))
        for context in ("A", "B", "C_train")
        for cue in ("blue", "red")
    )


def test_should_preserve_pc_neutral_assignment_parity_at_fixed64():
    import numpy as np

    plan = pilot.joint.compose(pilot._history_packets(training_sources()))
    batch = plan.training[0]
    learners = [pilot._history_learner(method) for method in control.BOUND_METHODS]
    initial = [deepcopy(learner.snapshot_state()) for learner in learners]
    names = ("weight_feature_hidden", "bias_hidden", "weight_hidden_output", "bias_output")
    for name in names:
        assert np.array_equal(
            initial[0].native_state.head_state[name].numpy(),
            initial[1].native_state.head_state[name].numpy(),
        )
    for method, learner in zip(control.BOUND_METHODS, learners, strict=True):
        learner.train_batch(batch.features, batch.targets)
        state = learner.snapshot_state()
        assert state.policy == control.policy(method)
        pilot.shared._invariant_feature_counter(state, 1, method, catalogue_rows=54)
        with pytest.raises(ValueError):
            pilot.shared._invariant_feature_counter(
                state, 1, method.removesuffix("_relax64"), catalogue_rows=54
            )
    final = [learner.snapshot_state().native_state.head_state for learner in learners]
    assert any(
        not np.array_equal(initial[0].native_state.head_state[name].numpy(), final[0][name].numpy())
        for name in names
    )
    for name in names:
        assert np.array_equal(final[0][name].numpy(), final[1][name].numpy())


@pytest.mark.parametrize("method", control.METHODS)
def test_should_complete_fixed64_joint_stream_and_disposable_f_return(monkeypatch, method):
    raw, probes = training_sources(), raw_probe_data()
    inputs = pickle.dumps((raw, probes))
    owners, originals, arms = [], [], []
    build, train = pilot._history_learner, pilot.shared._train_arm

    def learner(name):
        owned = build(name)
        assert owned.policy == control.policy(name)
        owners.append(owned)
        originals.append(deepcopy(owned.snapshot_state()))
        return owned

    def arm(*args, **kwargs):
        assert kwargs["catalogue_rows"] == 54
        result = train(*args, **kwargs)
        arms.append(result)
        return result

    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic fixed64 route forbids physics/teacher")

    monkeypatch.setattr(pilot.shared, "_collect_sources", lambda boundary, reports: raw)
    monkeypatch.setattr(pilot, "_collect_probe_sources", lambda boundary, reports, **kwargs: probes)
    monkeypatch.setattr(pilot, "MiniGridKeyDoor", forbidden)
    monkeypatch.setattr(pilot, "CueKeyDoorTeacher", forbidden)
    monkeypatch.setattr(pilot, "_history_learner", learner)
    monkeypatch.setattr(pilot.shared, "_train_arm", arm)
    report = pilot.run_pilot(**{**FLAGS, "qualification_method": method}, joint_catalogue=True)
    assert report["status"] == "complete", report["reason"]
    row = report["methods"][0]
    assert row["method"] == f"public_invariant_{method}" and row["status"] == "complete"
    assert row["charged_updates"] == row["stream_result"]["completed_updates"] == 128
    assert row["native_training_examples"] == row["stream_result"]["training_examples"] == 6912
    assert row["source_unchanged"] is row["actor_unchanged"] is True
    assert [item["updates"] for item in row["readouts"]] == [0, 32, 64, 96, 128]
    assert all(
        item["status"] == "complete" and len(item["scores"]) == 6 for item in row["readouts"]
    )
    for slot in row["probes"]:
        assert slot["status"] == "complete"
        assert slot["charged_updates"] == slot["return_charged_updates"] == 16
        assert slot["source_unchanged"] is slot["state_unchanged"] is True
    work = report["work"]
    assert work["native_train_calls"] == 192 and work["training_rows"] == 7488
    assert work["total_score_forwards_completed"] == 94
    assert work["total_score_rows_completed"] == 846
    assert work["relaxation_control"] == dict(
        inference_steps=64,
        native_train_calls_returned=192,
        nominal_latent_iterations_returned=12288,
        nominal_latent_row_iterations_returned=479232,
        failure_work_may_be_unobserved=False,
        scope=work["relaxation_control"]["scope"],
    )
    assert report["scientific_advantage_verified"] is False
    assert report["comparisons"] is None and row["final_evaluation"]["status"] == "not_run"
    assert len(owners) == 1 and _same_state(owners[0].snapshot_state(), originals[0])
    assert pickle.dumps((raw, probes)) == inputs
    arms[0]["guard"]()
