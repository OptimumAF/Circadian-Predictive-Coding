"""Recorded native-state metadata only; no episode playback or learner work.

Per invocation: 37 cases; two saved JSON reads, eight successful literal
projections / 632 episode callbacks, and 27 refusals before callback admission.
No model, tensor, numerical array, RNG, prediction, fit, update or physics work.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.paired_quota_inspector import project_paired_quota_report


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "complete": "artifacts/runs/paired-wall-allowance-20261020/paired-0-23.json",
    "failed": "artifacts/runs/paired-quota-control-20261020/paired-0-23.json",
}
SHAPES = {
    "weight_input": [8, 8],
    "weight_recurrent": [8, 8],
    "bias_hidden": [1, 8],
    "weight_output": [8, 5],
    "bias_output": [1, 5],
}


@pytest.fixture(scope="module")
def saved():
    return {
        key: json.loads((ROOT / path).read_text(encoding="utf-8")) for key, path in PATHS.items()
    }


def literal_episode(record, reference):
    return {"pointer": reference, "record": deepcopy(record), "frames": []}


def project(report):
    return project_paired_quota_report(
        report, source_label="saved native observations", project_episode=literal_episode
    )


@pytest.fixture(scope="module")
def projected(saved):
    output = {}
    for key, report in saved.items():
        before = deepcopy(report)
        output[key] = project(report)
        assert report == before
    return output


def pointer(document, reference):
    value = document
    for segment in reference.split("/")[1:]:
        value = value[int(segment)] if type(value) is list else value[segment]
    return value


@pytest.mark.parametrize("key", list(PATHS))
def test_should_bind_every_actual_native_observation_and_shape_to_original_stage(
    saved, projected, key
):
    raw, result = saved[key], projected[key]
    for arm_index, condition in enumerate(result["conditions"]):
        for ordinal, stage in enumerate(condition["stages"]):
            assert stage["observation_pointer"] == f"/runs/{arm_index}/observations/{ordinal}"
            observation = pointer(result["source"], stage["observation_pointer"])
            assert observation == raw["runs"][arm_index]["observations"][ordinal]
            assert stage["parameter_shapes"] == SHAPES
            assert observation["native_updates"] == stage["native_updates"]
            assert stage["physical_steps"] == ordinal * 24
            assert observation["architecture"] == "elman_tanh"
            assert observation["head_tensor_bytes"] == 724
            assert "excludes" in observation["memory_scope"]
            assert condition["unknown"] == dict(topology=None, lineage=None, sleep_proposals=None)
    assert all(stage["native_updates"] == 0 for stage in result["conditions"][0]["stages"])
    frozen = result["source"]["runs"][0]["observations"]
    assert all(
        record["native_training_examples"] == record["training_unroll_steps"] == 0
        for record in frozen
    )
    assert all(
        record["optimizer_tensor_bytes"] == record["gradient_tensor_bytes"] == 0
        for record in frozen
    )
    trained = result["source"]["runs"][1]["observations"]
    assert trained[-1]["native_updates"] == 9
    assert trained[-1]["native_training_examples"] == trained[-1]["training_unroll_steps"] == 96
    assert trained[-1]["optimizer_tensor_bytes"] == trained[-1]["gradient_tensor_bytes"] == 724
    # Missing raw physical labels are kept absent, even though the stage has a physical ordinal.
    assert all("physical_steps" not in row for row in trained)
    assert result["source"] == raw


def test_should_show_saved_state_even_when_later_live_episodes_were_unreached(projected):
    result = projected["failed"]
    condition = result["conditions"][1]
    assert condition["status"] == "failed"
    stage = condition["stages"][-1]
    assert pointer(result["source"], stage["observation_pointer"])["native_updates"] == 9
    assert all(row["record"]["status"] == "not_run" for row in condition["episodes"][32:])
    assert result["comparison"] is None


@pytest.mark.parametrize("mode", ["missing", "null", "empty", "slot_null"])
def test_should_leave_unrecorded_observations_unknown_in_complete_older_reports(saved, mode):
    report = deepcopy(saved["complete"])
    for run in report["runs"]:
        if mode == "missing":
            del run["observations"]
        elif mode == "null":
            run["observations"] = None
        elif mode == "empty":
            run["observations"] = []
        else:
            run["observations"][2] = None
    before = deepcopy(report)
    result = project(report)
    for condition in result["conditions"]:
        for index, stage in enumerate(condition["stages"]):
            if mode != "slot_null" or index == 2:
                assert stage["observation_pointer"] is None
                assert stage["parameter_shapes"] is None
            else:
                assert stage["observation_pointer"] is not None
                assert stage["parameter_shapes"] == SHAPES
    assert result["source"] == before and report == before


def test_should_keep_missing_attributes_and_explicit_null_parameters_without_inference(saved):
    report = deepcopy(saved["complete"])
    report["runs"][1]["observations"][1] = {"native_updates": 2}
    report["runs"][1]["observations"][2]["parameters"] = None
    before = deepcopy(report)
    result = project(report)
    first, second = result["conditions"][1]["stages"][1:3]
    assert first["parameter_shapes"] is second["parameter_shapes"] is None
    observation = pointer(result["source"], first["observation_pointer"])
    assert observation == {"native_updates": 2}
    assert observation.get("architecture") is None and observation.get("head_tensor_bytes") is None
    assert pointer(result["source"], second["observation_pointer"])["parameters"] is None
    assert result["source"] == before and report == before


def test_should_keep_unbound_observation_in_source_without_claiming_a_stage_state(saved):
    report = deepcopy(saved["failed"])
    trained = report["runs"][1]
    trained["quota_checkpoints"][4] = None
    trained["stream_result"]["checkpoints"][4] = None
    trained["episodes"][32:] = [None] * 8
    original_observation = deepcopy(trained["observations"][4])
    result = project(report)
    stage = result["conditions"][1]["stages"][4]
    assert stage["observation_pointer"] is None and stage["parameter_shapes"] is None
    assert result["source"]["runs"][1]["observations"][4] == original_observation
    assert original_observation["native_updates"] == 9
    assert stage["native_updates"] is None and stage["physical_steps"] is None
    assert result["source"] == report


def test_should_detach_shape_lists_and_recorded_parameters_from_input_and_other_stages(
    saved, projected
):
    before = deepcopy(saved["complete"])
    result = projected["complete"]
    original = deepcopy(result)
    try:
        result["conditions"][0]["stages"][0]["parameter_shapes"]["weight_input"][0] = 99
        result["source"]["runs"][0]["observations"][0]["parameters"]["weight_input"][0][0] = 99
        assert result["conditions"][0]["stages"][1]["parameter_shapes"] == SHAPES
        assert saved["complete"] == before
    finally:
        result.clear()
        result.update(original)


@pytest.mark.parametrize(
    "fault",
    [
        "native_bool",
        "native_mismatch",
        "physical_bool",
        "physical_mismatch",
        "wake_bool",
        "wake_mismatch",
        "memory_bool",
        "negative_memory",
        "training_float",
        "architecture",
        "scope_empty",
        "scope_type",
        "six_observations",
        "entry_string",
        "missing_native",
        "parameter_missing",
        "parameter_extra",
        "rank_one",
        "ragged",
        "bool_cell",
        "string_cell",
        "nonfinite_cell",
        "transposed_output",
        "feature_bool",
        "hidden_zero",
        "classes_oversize",
        "feature_mismatch",
    ],
)
def test_should_refuse_malformed_native_state_before_any_episode_callback(saved, fault):
    report = deepcopy(saved["complete"])
    observation = report["runs"][1]["observations"][1]
    parameters = observation["parameters"]
    protocol = report["protocol"]["arm_protocol"]
    if fault == "native_bool":
        observation["native_updates"] = True
    elif fault == "native_mismatch":
        observation["native_updates"] = 3
    elif fault == "physical_bool":
        observation["physical_steps"] = True
    elif fault == "physical_mismatch":
        observation["physical_steps"] = 48
    elif fault == "wake_bool":
        observation["wake_updates"] = True
    elif fault == "wake_mismatch":
        observation["wake_updates"] = 3
    elif fault == "memory_bool":
        observation["head_tensor_bytes"] = True
    elif fault == "negative_memory":
        observation["optimizer_tensor_bytes"] = -1
    elif fault == "training_float":
        observation["native_training_examples"] = 24.0
    elif fault == "architecture":
        observation["architecture"] = "linear"
    elif fault == "scope_empty":
        observation["memory_scope"] = ""
    elif fault == "scope_type":
        observation["memory_scope"] = 42
    elif fault == "six_observations":
        report["runs"][1]["observations"].append(deepcopy(observation))
    elif fault == "entry_string":
        report["runs"][1]["observations"][1] = "unavailable"
    elif fault == "missing_native":
        del observation["native_updates"]
    elif fault == "parameter_missing":
        del parameters["bias_output"]
    elif fault == "parameter_extra":
        parameters["unrecorded_weight"] = [[1.0]]
    elif fault == "rank_one":
        parameters["bias_hidden"] = parameters["bias_hidden"][0]
    elif fault == "ragged":
        parameters["weight_input"][0].pop()
    elif fault == "bool_cell":
        parameters["weight_input"][0][0] = True
    elif fault == "string_cell":
        parameters["weight_input"][0][0] = "0.0"
    elif fault == "nonfinite_cell":
        parameters["weight_input"][0][0] = float("inf")
    elif fault == "transposed_output":
        parameters["weight_output"] = [
            list(row) for row in zip(*parameters["weight_output"], strict=True)
        ]
    elif fault == "feature_bool":
        protocol["feature_dim"] = True
    elif fault == "hidden_zero":
        protocol["hidden_dim"] = 0
    elif fault == "classes_oversize":
        protocol["num_classes"] = 257
    else:
        protocol["feature_dim"] = 9

    def forbidden(*args, **kwargs):
        pytest.fail("invalid native observation admitted playback")

    with pytest.raises(ValueError):
        project_paired_quota_report(
            report, source_label="corrupt native state", project_episode=forbidden
        )
