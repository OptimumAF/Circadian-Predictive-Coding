"""Selected fixed-input streams reuse delayed releases without empirical fits.

Two cached sequences: two initializers, three appends/rows, one additional
validation normalization, five returned metadata observers. No prediction,
distance query, fit, RNG, SGD, native scene or physics occurs in this cohort.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import pytest

from src.adapters import continuous_stream_demo as demo
from src.adapters import continuous_stream_sequence as sequence
from src.adapters.numpy_nearest_continuous_bc import (
    FixedInputNearestContinuousBCPolicy,
    NumpyNearestContinuousBCLearner,
)
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def episode(name):
    observation = (-2.0, 0.5) if name == "warm" else (4.0, -1.0)
    action = -1.0 if name == "warm" else 1.0
    return ContinuousEpisode(
        name,
        name + "-context",
        "train",
        ContinuousActionSpec(2, ("unit",), (-1.0,), (1.0,)),
        (ContinuousTransition(0, observation, (action,), 1.0, (0.0, 0.0), True, False),),
    )


def assert_state(first, second):
    assert first.policy == second.policy and type(first.policy) is type(second.policy)
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    assert first.prototype_features.tolist() == second.prototype_features.tolist()
    assert first.prototype_actions.tolist() == second.prototype_actions.tolist()


def forbidden(*args, **kwargs):
    pytest.fail("fixed-input admission reached a forbidden fitter, predictor or constructor")


@pytest.fixture(scope="module")
def streams():
    results = {}
    runtime_factory = sequence.ActorShadowRuntime
    train = NumpyNearestContinuousBCLearner.train_batch
    for damage in (False, True):
        warm, new = episode("warm"), episode("new")
        original = asdict(warm), asdict(new)
        captured: dict[str, Any] = dict(readouts=[], trains=[])

        def runtime(source, **options):
            created = runtime_factory(source, **options)
            captured.update(
                source=source, runtime=created, options=options, initial=source.snapshot_state()
            )
            return created

        def append(owner, features, targets):
            captured["trains"].append((features.tolist(), targets.tolist()))
            return train(owner, features, targets)

        def observe(readout, metadata):
            baseline = readout.snapshot_state()
            captured["readouts"].append((readout, deepcopy(metadata), baseline))
            returned = {"phase": metadata["phase"], "scope": "metadata_only_no_predictions"}
            # Detached snapshots and metadata are safe for the callback to edit.
            baseline.prototype_features.fill(99.0)
            metadata["tick"] = -99
            metadata["known_episode_ids"] = ["callback-only"]
            if damage and returned["phase"] == "before_release":
                readout._state.prototype_actions[0, 0] = -0.75
            return returned

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sequence, "ActorShadowRuntime", runtime)
            patch.setattr(NumpyNearestContinuousBCLearner, "train_batch", append)
            patch.setattr(sequence, "fit_training_standardization", forbidden)
            patch.setattr(demo, "fit_training_standardization", forbidden)
            patch.setattr(NumpyNearestContinuousBCLearner, "predict_with_work", forbidden)
            report = demo.run_continuous_demo(
                warm, new, fixed_input=True, checkpoint_observer=observe
            )
        assert (asdict(warm), asdict(new)) == original
        results[damage] = report, captured
    return results


def test_should_keep_labels_unavailable_for_two_ticks_and_charge_only_the_single_arrived_append(
    streams,
):
    report, captured = streams[False]
    assert report["complete"] is True, report
    assert [row["tick"] for row in report["events"]] == [0, 1, 2]
    assert [row["label_available"] for row in report["events"]] == [False, False, True]
    assert [len(row["applied"]) for row in report["events"]] == [0, 0, 1]
    assert [row["candidate_prototype_rows"] for row in report["events"]] == [1, 1, 2]
    assert [row["budget_updates"] for row in report["events"]] == [0, 0, 1]
    assert len(report["applied"]) == 1
    receipt = report["applied"][0]
    assert receipt["observed_at"] == 0
    assert receipt["arrived_at"] == receipt["applied_at"] == 2
    assert receipt["episode_id"] == "new" and receipt["update_number"] == 1
    assert receipt["diagnostic"]["value"] == 1
    assert captured["options"]["budget"].updates_completed == 1
    assert captured["options"]["budget"].replay_examples_completed == 0
    assert captured["trains"] == [([[-2.0, 0.5]], [[-1.0]]), ([[4.0, -1.0]], [[1.0]])]


def test_should_leave_source_and_actor_warm_while_candidate_uses_owned_identity_coordinates(
    streams,
):
    report, captured = streams[False]
    initial = captured["initial"]
    assert type(initial.policy) is FixedInputNearestContinuousBCPolicy
    assert initial.policy.observation_standardization is None
    assert report["policy"]["observation_standardization"] is None
    assert report["fixed_input"] is True
    assert report["input_normalization"] == dict(
        definition="continuous_nearest_explicit_fixed_input_identity_v1",
        empirical_fit=False,
        policy_standardization=None,
        normalization_calls="identity_copy_of_already_encoded_inputs",
    )
    assert_state(initial, captured["source"].snapshot_state())
    assert_state(initial, captured["runtime"].actor.snapshot_state().state)
    candidate = captured["runtime"].candidate_snapshot().state
    assert candidate.completed_updates == candidate.training_examples == 2
    assert candidate.prototype_features.tolist() == [[-2.0, 0.5], [4.0, -1.0]]
    assert candidate.prototype_actions.tolist() == [[-1.0], [1.0]]
    work = report["work"]
    assert work["fit_attempts"] == work["fit_returns"] == work["fit_rows"] == 0
    assert work["warm_append_rows"] == work["runtime_append_rows"] == work["validated_updates"] == 1
    assert work["validation_normalization_rows"] == 1
    assert work["model_initializations"] == 1
    assert work["model_random_draws"] == work["optimizer_updates"] == 0


def test_should_detach_every_observer_and_keep_callback_metadata_out_of_parent_checkpoints(streams):
    report, captured = streams[False]
    observations = captured["readouts"]
    assert len({id(row[0]) for row in observations}) == 3
    assert [row[1]["phase"] for row in observations] == ["warm", "before_release", "after_release"]
    assert [row["tick"] for row in report["checkpoints"]] == [0, 0, 2]
    assert [row["known_episode_ids"] for row in report["checkpoints"]] == [
        ["warm"],
        ["warm"],
        ["warm", "new"],
    ]
    assert [row[0].snapshot_state().training_examples for row in observations] == [1, 1, 2]
    assert (
        report["work"]["observer_attempts"]
        == report["work"]["observer_returns"]
        == report["work"]["observer_validated"]
        == 3
    )


def test_should_retain_a_returned_observer_result_but_refuse_release_after_readout_mutation(
    streams,
):
    report, captured = streams[True]
    assert report["complete"] is False and report["error"]["type"] == "ValueError"
    assert report["checkpoints"][-1]["result"] == dict(
        phase="before_release", scope="metadata_only_no_predictions"
    )
    assert len(captured["readouts"]) == 2 and len(captured["trains"]) == 1
    assert report["events"] == report["applied"] == []
    assert report["work"]["observer_attempts"] == report["work"]["observer_returns"] == 2
    assert report["work"]["observer_validated"] == 1
    assert report["work"]["runtime_append_attempts"] == report["work"]["runtime_append_rows"] == 0
    assert report["experiment_state"]["completed_updates"] == 0
    assert_state(captured["initial"], captured["source"].snapshot_state())
    assert_state(captured["initial"], captured["runtime"].actor.snapshot_state().state)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_should_refuse_nonboolean_selection_before_fitting_or_model_construction(monkeypatch, flag):
    monkeypatch.setattr(sequence, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_demo(episode("warm"), episode("new"), fixed_input=flag)
    with pytest.raises(ValueError):
        sequence.run_sequence(episode("warm"), (episode("new"),), fixed_input=flag)


def test_should_refuse_nontrain_source_before_identity_model_construction(monkeypatch):
    monkeypatch.setattr(sequence, "fit_training_standardization", forbidden)
    monkeypatch.setattr(sequence, "NumpyNearestContinuousBCLearner", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_demo(
            episode("warm"), replace(episode("new"), role="probe_test"), fixed_input=True
        )


@pytest.mark.parametrize("explicit_false", [False, True])
def test_should_preserve_default_sequence_callshape_without_new_false_metadata(
    monkeypatch, explicit_false
):
    calls = []
    returned = {"legacy_schema": True}

    def run(*args, **options):
        calls.append((args, options))
        return returned

    def observe(*args):
        return {}

    monkeypatch.setattr(sequence, "run_sequence", run)
    warm, new = episode("warm"), episode("new")
    options: dict[str, Any] = dict(fixed_input=False) if explicit_false else {}
    assert demo.run_continuous_demo(warm, new, checkpoint_observer=observe, **options) is returned
    assert calls == [((warm, (new,), observe), {})]
    assert "fixed_input" not in returned and "input_normalization" not in returned
