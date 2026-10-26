"""Saved MiniGrid playback preserves public observations and exact provenance."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.minigrid_inspector_projection import project_minigrid_report


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/r21-fixed-a-acquisition-20261016/native-result.json"
)
METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc", "recurrent_bc")


@pytest.fixture(scope="module")
def recorded():
    return json.loads(ARTIFACT.read_text(encoding="utf8"))


def pointer(document, reference):
    value = document
    for token in reference.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if type(value) is list else value[token]
    return value


def decode(observation):
    image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
    for x in range(7):
        for y in range(7):
            offset = (x * 7 + y) * 3
            image[x][y] = [
                round((observation[offset + component] + 1) * scale / 2)
                for component, scale in enumerate((10, 5, 2))
            ]
    return image, round((observation[147] + 1) * 3 / 2)


def test_should_preserve_all_five_methods_twenty_recorded_episodes_and_negative_results(recorded):
    projected = project_minigrid_report(recorded, source_label=ARTIFACT.name)
    assert projected["schema_id"] == "minigrid_inspector_v1"
    assert projected["origin"] == "recorded_artifact" and projected["source_label"] == ARTIFACT.name
    assert projected["source"] == recorded and projected["source"] is not recorded
    assert [condition["key"] for condition in projected["conditions"]] == list(METHODS)
    transitions, successes, negatives, timeouts = 0, 0, 0, 0
    for index, condition in enumerate(projected["conditions"]):
        assert condition["pointer"] == f"/methods/{index}"
        assert pointer(projected["source"], condition["pointer"]) == recorded["methods"][index]
        assert len(condition["episodes"]) == 4
        assert [(slot["updates"], slot["cue"]) for slot in condition["episodes"]] == [
            (0, "blue"),
            (0, "red"),
            (128, "blue"),
            (128, "red"),
        ]
        for slot in condition["episodes"]:
            episode = pointer(projected["source"], slot["pointer"])
            assert episode == slot["record"]
            assert slot["pointer"].startswith("/evaluations/") and "/episodes/" in slot["pointer"]
            assert episode["context"] == "A" and episode["source_role"] == "train"
            environment = episode["environment"]
            transitions += len(environment["transitions"])
            successes += episode["success"] is True
            negatives += episode["return"] < 0
            timeouts += environment["truncated"] is True
            assert len(slot["frames"]) == len(environment["transitions"]) + 1
            for frame in slot["frames"]:
                observation = pointer(projected["source"], frame["pointer"])
                assert frame["observation"] == observation and len(observation) == 148
                image, direction = decode(observation)
                assert frame["image"] == image and frame["direction"] == direction
                decision_pointer = frame["decision_pointer"]
                if decision_pointer is not None:
                    decision = pointer(projected["source"], decision_pointer)
                    assert decision["observation"] == observation
                    assert decision["action"] == decision["probabilities"].index(
                        max(decision["probabilities"])
                    )
                    transition = pointer(projected["source"], frame["transition_pointer"])
                    assert transition["action"] == decision["action"]
                    if condition["key"] == "recurrent_bc":
                        assert frame["features_pointer"] is None and "features" not in decision
                    else:
                        features = pointer(projected["source"], frame["features_pointer"])
                        assert features == episode["predictions"][0]["observation"] + observation
                else:
                    assert frame["features_pointer"] is None
            assert slot["frames"][-1]["decision_pointer"] is None
    assert (transitions, timeouts, negatives, successes) == (288, 16, 2, 2)
    assert all(
        row["joint_acquisition"] is False for row in projected["source"]["acquisition"]["methods"]
    )
    assert (
        projected["source"]["acquisition"]["signed_comparisons"]["gated_cpc-minus-pc"]["ce_gain"]
        < 0
    )


def test_should_keep_native_x_y_orientation_and_cue_removal_without_invented_global_state(recorded):
    projected = project_minigrid_report(recorded, source_label="native")
    slot = projected["conditions"][0]["episodes"][0]
    initial, after = slot["frames"][:2]
    assert initial["image"] == slot["record"]["environment"]["initial_observation"]["image"]
    # MiniGrid's first image axis is x. These asymmetric coordinates catch a transpose.
    assert initial["image"][2][6] == [6, 2, 0]
    assert initial["image"][6][2] == [2, 5, 0]
    assert any(cell[0] == 6 for column in initial["image"] for cell in column)
    assert not any(cell[0] == 6 for column in after["image"] for cell in column)
    assert initial["direction"] == 3


@pytest.mark.parametrize(
    "fault", ["teacher_negative", "teacher_mismatch", "array_bytes", "rss", "seconds"]
)
def test_should_refuse_corrupted_costs_displayed_by_the_inspector(recorded, fault):
    source = deepcopy(recorded)
    if fault == "teacher_negative":
        source["source_reports"][0]["environment_steps"] = -1
    elif fault == "teacher_mismatch":
        source["source_reports"][0]["environment_steps"] = 8
    elif fault == "array_bytes":
        source["retained_input_array_bytes"]["training_raw"] = -1
    elif fault == "rss":
        source["observed_process_rss"]["peak_bytes"] = 0
    else:
        source["seconds"] = -0.1
    with pytest.raises(ValueError):
        project_minigrid_report(source, source_label="invalid displayed cost")


def test_should_dispatch_only_exact_protocol_and_detach_projection_records(recorded):
    source = deepcopy(recorded)
    projected = project_key_door_report(source, source_label="detached")
    assert projected["schema_id"] == "minigrid_inspector_v1"
    original = projected["conditions"][0]["episodes"][0]["record"]["predictions"][0]["action"]
    source["evaluations"][0]["episodes"][0]["predictions"][0]["action"] = 123
    assert (
        projected["conditions"][0]["episodes"][0]["record"]["predictions"][0]["action"] == original
    )
    projected["conditions"][0]["episodes"][0]["frames"][0]["image"][0][0][0] = 99
    assert (
        projected["source"]["evaluations"][0]["episodes"][0]["environment"]["initial_observation"][
            "image"
        ][0][0][0]
        != 99
    )


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "action",
        "argmax",
        "probabilities",
        "steps",
        "prediction_count",
        "observation_chain",
        "history",
        "rnn_features",
        "role",
        "cue",
        "nonterminal",
        "return",
        "closed",
    ],
)
def test_should_refuse_corrupted_saved_decisions_clocks_and_causal_features(recorded, fault):
    source = deepcopy(recorded)
    episode = source["evaluations"][0]["episodes"][0]
    decision, transition = episode["predictions"][0], episode["environment"]["transitions"][0]
    if fault == "protocol":
        source["protocol"]["protocol_id"] = "minigrid_trained_history_live_boundary_v1"
    elif fault == "action":
        decision["action"] = (decision["action"] + 1) % 7
    elif fault == "argmax":
        decision["action"] = transition["action"] = (decision["action"] + 1) % 7
    elif fault == "probabilities":
        decision["probabilities"] = [1.0] * 7
    elif fault == "steps":
        episode["environment"]["environment_steps"] -= 1
    elif fault == "prediction_count":
        episode["prediction_calls_returned"] += 1
    elif fault == "observation_chain":
        episode["predictions"][1]["observation"][0] = 0.2
    elif fault == "history":
        episode["predictions"][1]["features"][0] = 0.2
    elif fault == "rnn_features":
        rnn = next(
            row
            for row in source["evaluations"]
            if row["method"] == "recurrent_bc" and row["updates"] == 0
        )
        rnn["episodes"][0]["predictions"][0]["features"] = [0.0] * 296
    elif fault == "role":
        episode["environment"]["role"] = "final_test"
    elif fault == "cue":
        source["evaluations"][0]["episodes"][1]["cue"] = "blue"
    elif fault == "nonterminal":
        episode["environment"]["truncated"] = False
        episode["environment"]["transitions"][-1]["truncated"] = False
    elif fault == "closed":
        episode["environment"]["closed"] = False
    else:
        episode["return"] += 0.1
    with pytest.raises(ValueError):
        project_minigrid_report(source, source_label="corrupted")


def partial_report(recorded):
    source = deepcopy(recorded)
    source.update(status="incomplete", reason="recorded readout interrupted")
    source["evaluations"] = source["evaluations"][:1]
    for row in source["acquisition"]["methods"]:
        row.update({key: None for key in row if key not in ("method", "status")})
        row["status"] = "unavailable"
    for comparison in source["acquisition"]["signed_comparisons"].values():
        comparison.update({key: None for key in comparison})
    return source


def test_should_keep_missing_methods_empty_and_original_paid_training_costs_unknown_not_zero(
    recorded,
):
    source = partial_report(recorded)
    projected = project_minigrid_report(source, source_label="partial")
    assert projected["source"] == source
    assert len(projected["conditions"][0]["episodes"]) == 2
    assert all(condition["episodes"] == [] for condition in projected["conditions"][1:])
    assert sum(row["charged_updates"] for row in projected["source"]["methods"]) == 640
    assert all(
        row["joint_acquisition"] is None for row in projected["source"]["acquisition"]["methods"]
    )
    assert len(projected["source"]["evaluations"]) == 1


def test_should_preserve_failed_constructor_slot_without_inventing_frames_or_policy_inputs(
    recorded,
):
    source = partial_report(recorded)
    readout = source["evaluations"][0]
    readout.update(
        status="failed", reason="recorded environment construction failed", state_unchanged=None
    )
    episode = readout["episodes"][0]
    readout["episodes"] = [episode]
    episode.update(
        status="failed",
        reason="recorded environment construction failed",
        predictions=[],
        prediction_calls_attempted=0,
        prediction_calls_returned=0,
        environment=None,
        state_unchanged=None,
        success=None,
        **{"return": None},
    )
    for key in readout["work"]:
        if (
            key.startswith(("live_", "environment_", "history_live_"))
            or key == "history_episodes_started"
        ):
            readout["work"][key] = 0
    readout["work"].update(environment_resets_attempted=1, failure_work_may_be_unobserved=True)
    projected = project_minigrid_report(source, source_label="constructor-prefix")
    slot = projected["conditions"][0]["episodes"][0]
    assert slot["record"] == episode and slot["frames"] == [] and slot["initial_image"] is None
    assert slot["record"]["environment"] is None and slot["record"]["success"] is None
    assert projected["source"]["evaluations"][0]["work"]["environment_resets_attempted"] == 1
    assert projected["source"]["evaluations"][0]["work"]["environment_steps_returned"] == 0


def test_should_preserve_two_paid_steps_of_failed_episode_without_claiming_termination(recorded):
    source = partial_report(recorded)
    readout = source["evaluations"][0]
    readout.update(
        status="failed", reason="recorded boundary refused after two steps", state_unchanged=None
    )
    episode = readout["episodes"][0]
    readout["episodes"] = [episode]
    episode.update(
        status="failed",
        reason="recorded boundary refused after two steps",
        prediction_calls_attempted=2,
        prediction_calls_returned=2,
        state_unchanged=None,
        success=None,
        **{"return": None},
    )
    episode["predictions"] = episode["predictions"][:2]
    environment = episode["environment"]
    environment["transitions"] = environment["transitions"][:2]
    environment.update(
        environment_steps_attempted=2,
        environment_steps=2,
        status="failed",
        terminated=False,
        truncated=False,
        total_return=0.0,
    )
    readout["work"].update(
        live_prediction_calls_attempted=2,
        live_prediction_calls_returned=2,
        environment_resets_attempted=1,
        environment_resets_completed=1,
        environment_steps_attempted=2,
        environment_steps_returned=2,
        history_episodes_started=1,
        history_live_feature_rows=2,
        history_live_feature_array_bytes=2 * 296 * 8,
        failure_work_may_be_unobserved=True,
    )
    projected = project_minigrid_report(source, source_label="two-paid-steps")
    slot = projected["conditions"][0]["episodes"][0]
    assert slot["record"] == episode and len(slot["frames"]) == 3
    assert slot["record"]["success"] is slot["record"]["return"] is None
    assert slot["frames"][-1]["decision_pointer"] is None
    assert slot["frames"][-1]["features_pointer"] is None
    assert slot["frames"][-1]["observation"] == environment["transitions"][-1]["next_observation"]
