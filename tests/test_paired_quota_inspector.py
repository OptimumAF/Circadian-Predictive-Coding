"""Saved paired quota projection: no learner, environment or numerical kernels.

Three shared public projections cover 240 episode slots / 2692 recorded
transitions. Metadata-refusal cases use a literal callback and do not replay
saved actions again. Original saved files are never changed.

Per invocation: 40 cases, three saved JSON reads/public pair projections, five
valid metadata-only projections (392 literal callbacks), twenty metadata refusals,
and five malformed single-episode attempts (at most 60 additional transitions).
No model initialization, numerical arrays, RNG, learner or environment work occurs.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.app.key_door_inspector import project_key_door_report
from src.app.paired_quota_inspector import project_paired_quota_report


ROOT = Path(__file__).resolve().parents[1]
PATHS = {
    "0-23": "artifacts/runs/paired-wall-allowance-20261020/paired-0-23.json",
    "6-17": "artifacts/runs/paired-wall-allowance-20261020/paired-6-17.json",
    "failed": "artifacts/runs/paired-quota-control-20261020/paired-0-23.json",
}


@pytest.fixture(scope="module")
def saved():
    return {
        key: json.loads((ROOT / path).read_text(encoding="utf-8")) for key, path in PATHS.items()
    }


@pytest.fixture(scope="module")
def projections(saved):
    results = {}
    for key, report in saved.items():
        original = deepcopy(report)
        results[key] = project_key_door_report(report, source_label=PATHS[key])
        assert report == original
    return results


def pointer(document, reference):
    result = document
    for part in reference.split("/")[1:]:
        result = result[int(part)] if type(result) is list else result[part]
    return result


def literal_episode(record, reference):
    return {"pointer": reference, "record": deepcopy(record), "frames": []}


def metadata_projection(report):
    return project_paired_quota_report(
        report, source_label="literal metadata", project_episode=literal_episode
    )


@pytest.mark.parametrize("key", list(PATHS))
def test_should_project_both_original_arms_and_every_episode_with_resolvable_pointers(
    saved, projections, key
):
    raw, result = saved[key], projections[key]
    assert result["schema_id"] == "paired_quota_inspector_v1"
    assert result["origin"] == "recorded_artifact"
    assert result["source"] == raw
    assert [condition["key"] for condition in result["conditions"]] == ["frozen", "trained"]
    for arm_index, condition in enumerate(result["conditions"]):
        assert condition["pointer"] == f"/runs/{arm_index}"
        assert condition["protocol_id"] == raw["protocol"]["protocol_id"]
        assert condition["status"] == raw["runs"][arm_index]["status"]
        assert condition["map_origin"] == "derived_from_recorded_layout_and_applied_actions"
        assert condition["unknown"] == dict(topology=None, lineage=None, sleep_proposals=None)
        assert len(condition["episodes"]) == 40
        for episode_index, episode in enumerate(condition["episodes"]):
            assert episode["pointer"] == f"/runs/{arm_index}/episodes/{episode_index}"
            assert pointer(result["source"], episode["pointer"]) == episode["record"]
            transitions = episode["record"].get("transitions")
            if transitions:
                assert len(episode["frames"]) == len(transitions) + 1
            else:
                assert episode["frames"] == []
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key", ["0-23", "6-17"])
def test_should_bind_five_physical_stages_to_original_quota_and_native_score_records(
    saved, projections, key
):
    raw, result = saved[key], projections[key]
    for arm_index, condition in enumerate(result["conditions"]):
        assert len(condition["stages"]) == 5
        for index, stage in enumerate(condition["stages"]):
            quota = raw["runs"][arm_index]["quota_checkpoints"][index]
            assert stage["physical_steps"] == index * 24 == quota["physical_steps"]
            assert stage["native_updates"] == quota["native_updates"]
            assert stage["pointer"] == f"/runs/{arm_index}/quota_checkpoints/{index}"
            assert pointer(result["source"], stage["pointer"]) == quota
            assert stage["episode_indices"] == list(range(index * 8, index * 8 + 8))
            score = pointer(result["source"], stage["score_pointer"])
            assert score["updates"] == stage["native_updates"]
            assert all(
                raw["runs"][arm_index]["episodes"][ordinal]["checkpoint_updates"]
                == stage["native_updates"]
                for ordinal in stage["episode_indices"]
            )
    assert [stage["native_updates"] for stage in result["conditions"][0]["stages"]] == [0] * 5
    # The trained raw records omitted physical fields; a derived stage must not repair them.
    assert all("checkpoint_physical_steps" not in episode for episode in raw["runs"][1]["episodes"])
    assert result["source"]["runs"][1]["episodes"] == raw["runs"][1]["episodes"]


def test_should_retain_original_failed_unreached_and_unavailable_fields_without_false_zero(
    saved, projections
):
    raw, result = saved["failed"], projections["failed"]
    assert raw["comparison"] is None and result["comparison"] is None
    assert "timing" not in raw and result["timing"] is None
    trained = result["conditions"][1]
    assert trained["status"] == "failed"
    assert [row["record"]["status"] for row in trained["episodes"]].count("not_run") == 8
    failed = trained["episodes"][31]
    assert failed["record"]["status"] == "failed"
    assert failed["record"].get("return") is None
    assert failed["record"].get("steps_executed") is None
    for episode in trained["episodes"][32:]:
        assert episode["frames"] == []
        assert episode["record"].get("return") is None
        assert episode["record"].get("success") is None
        assert "layout" not in episode["record"]
    assert result["source"] == raw


def test_should_derive_stage_from_quota_ordinal_without_replacing_explicit_raw_null(saved):
    report = deepcopy(saved["0-23"])
    for episode in report["runs"][1]["episodes"]:
        episode["checkpoint_physical_steps"] = None
    before = deepcopy(report)
    result = metadata_projection(report)
    assert [stage["physical_steps"] for stage in result["conditions"][1]["stages"]] == [
        0,
        24,
        48,
        72,
        96,
    ]
    assert all(
        episode["record"]["checkpoint_physical_steps"] is None
        for episode in result["conditions"][1]["episodes"]
    )
    assert result["source"] == before and report == before


def test_should_keep_partial_null_stage_score_and_episode_slots_unavailable(saved):
    report = deepcopy(saved["failed"])
    trained = report["runs"][1]
    trained["quota_checkpoints"][4] = None
    trained["stream_result"]["checkpoints"][4] = None
    trained["episodes"][32:] = [None] * 8
    result = metadata_projection(report)
    stage = result["conditions"][1]["stages"][4]
    assert stage["physical_steps"] is None and stage["native_updates"] is None
    assert pointer(result["source"], stage["pointer"]) is None
    assert stage["score_pointer"] is None
    assert stage["episode_indices"] == list(range(32, 40))
    for episode in result["conditions"][1]["episodes"][32:]:
        assert episode["record"] is None and episode["frames"] == []
        assert pointer(result["source"], episode["pointer"]) is None
    assert result["comparison"] is None and result["source"] == report


def test_should_project_admitted_copy_when_callback_mutates_future_original_row(saved):
    report = deepcopy(saved["0-23"])
    admitted = deepcopy(report)

    def mutate_original(record, reference):
        if reference == "/runs/0/episodes/0":
            report["runs"][0]["episodes"][1]["cue"] = 1
        return literal_episode(record, reference)

    result = project_paired_quota_report(
        report, source_label="callback ownership", project_episode=mutate_original
    )
    assert report != admitted
    assert result["source"] == admitted
    future = result["conditions"][0]["episodes"][1]
    assert future["record"] == admitted["runs"][0]["episodes"][1]
    assert pointer(result["source"], future["pointer"]) == future["record"]


def test_should_refuse_boolean_displayed_work_before_any_playback(saved):
    report = deepcopy(saved["0-23"])
    report["runs"][0]["work"]["live_callback_returns"] = True

    def forbidden(*args, **kwargs):
        pytest.fail("boolean work admitted playback")

    with pytest.raises(ValueError):
        project_paired_quota_report(report, source_label="boolean cost", project_episode=forbidden)


@pytest.mark.parametrize("status", ["not_evaluated", "trained"])
def test_should_preserve_unreleased_producer_arm_status_without_inventing_success(saved, status):
    report = deepcopy(saved["failed"])
    report["runs"][1]["status"] = status
    result = metadata_projection(report)
    assert result["conditions"][1]["status"] == status
    assert result["source"] == report
    assert result["comparison"] is None
    assert result["conditions"][1]["episodes"][32]["record"]["status"] == "not_run"


@pytest.mark.parametrize("key", ["0-23", "6-17"])
def test_should_preserve_every_signed_comparison_and_all_seven_inclusive_timing_records(
    saved, projections, key
):
    raw, result = saved[key], projections[key]
    assert len(result["comparison"]) == 40
    for index, cell in enumerate(result["comparison"]):
        assert cell["pointer"] == f"/comparison/{index}"
        assert cell["record"] == pointer(result["source"], cell["pointer"])
    assert result["timing"]["pointer"] == "/timing"
    assert result["timing"]["record"] == raw["timing"]
    timing = result["timing"]["record"]
    assert "overlap" in timing["scope"]
    records = [row for arm in timing["arms"].values() for row in arm.values()]
    records.extend(timing["guards"].values())
    assert len(records) == 7 and all(row["inclusive_seconds"] >= 0 for row in records)
    if key == "0-23":
        assert any(cell["record"]["ce_reduction"] < 0 for cell in result["comparison"])


def test_should_derive_mapped_B_positions_from_applied_actions_and_keep_both_probability_laws(
    saved, projections
):
    episode = projections["0-23"]["conditions"][0]["episodes"][2]
    record = episode["record"]
    first = record["transitions"][0]
    assert record["context"] == "B" and record["layout"]["rotation"] == 0
    assert record["layout"]["movement_action_map"] == [1, 2, 3, 0]
    assert first["action"] == 3 and first["legal_actions"] == [3]
    assert first["probabilities"] == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert all(value > 0 for value in first["raw_probabilities"])
    assert episode["frames"][0]["position"] == [2, 4]
    assert episode["frames"][1]["position"] == [2, 3]
    for index, transition in enumerate(record["transitions"]):
        assert episode["frames"][index]["observation"] == transition["observation"]
        assert episode["frames"][index + 1]["observation"] == transition["next_observation"]
    assert record == saved["0-23"]["runs"][0]["episodes"][2]


def test_should_detach_source_stage_episode_and_timing_views_from_saved_artifact(
    saved, projections
):
    before = deepcopy(saved["0-23"])
    result = projections["0-23"]
    original_projection = deepcopy(result)
    try:
        result["source"]["runs"][0]["status"] = "changed"
        result["conditions"][0]["stages"][0]["episode_indices"][0] = 99
        result["conditions"][0]["episodes"][0]["record"]["transitions"][0]["action"] = 4
        result["conditions"][0]["episodes"][0]["frames"][0]["position"][0] = 99
        result["timing"]["record"]["guards"]["pair"]["attempts"] = 99
        assert result["source"]["runs"][0]["episodes"][0]["transitions"][0]["action"] == 0
        assert saved["0-23"] == before
    finally:
        result.clear()
        result.update(original_projection)


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "one_arm",
        "arm_order",
        "duplicate_arm",
        "physical_bool",
        "physical_boundary",
        "native_bool",
        "frozen_native",
        "episode_native",
        "episode_cue",
        "episode_physical",
        "score_native",
        "comparison_missing",
        "comparison_null",
        "comparison_context",
        "comparison_nan",
        "timing_nan",
        "timing_bool",
    ],
)
def test_should_refuse_corrupt_pair_metadata_before_any_episode_callback(saved, fault):
    report = deepcopy(saved["0-23"])
    frozen, trained = report["runs"]
    if fault == "protocol":
        report["protocol"]["protocol_id"] = "unrelated"
    elif fault == "one_arm":
        report["runs"].pop()
    elif fault == "arm_order":
        report["runs"].reverse()
    elif fault == "duplicate_arm":
        trained["arm"] = "frozen"
    elif fault == "physical_bool":
        frozen["quota_checkpoints"][0]["physical_steps"] = False
    elif fault == "physical_boundary":
        trained["quota_checkpoints"][1]["physical_steps"] = 25
    elif fault == "native_bool":
        trained["quota_checkpoints"][0]["native_updates"] = False
    elif fault == "frozen_native":
        frozen["quota_checkpoints"][1]["native_updates"] = 1
    elif fault == "episode_native":
        trained["episodes"][8]["checkpoint_updates"] = 99
    elif fault == "episode_cue":
        trained["episodes"][1]["cue"] = 1
    elif fault == "episode_physical":
        frozen["episodes"][8]["checkpoint_physical_steps"] = 25
    elif fault == "score_native":
        trained["stream_result"]["checkpoints"][1]["updates"] = 99
    elif fault == "comparison_missing":
        report["comparison"].pop()
    elif fault == "comparison_null":
        report["comparison"] = None
    elif fault == "comparison_context":
        report["comparison"][0]["context"] = "B-red"
    elif fault == "comparison_nan":
        report["comparison"][0]["ce_reduction"] = float("nan")
    elif fault == "timing_nan":
        report["timing"]["arms"]["frozen"]["train"]["inclusive_seconds"] = float("nan")
    else:
        report["timing"]["guards"]["pair"]["attempts"] = True

    def forbidden(*args, **kwargs):
        pytest.fail("corrupt metadata admitted playback")

    with pytest.raises(ValueError):
        project_paired_quota_report(
            report, source_label="corrupt metadata", project_episode=forbidden
        )


def test_should_refuse_claiming_complete_coverage_for_original_unreached_slots(saved):
    report = deepcopy(saved["failed"])
    report["status"] = "complete"
    with pytest.raises(ValueError):
        metadata_projection(report)


@pytest.mark.parametrize(
    "fault", ["map_duplicate", "map_bool", "basis", "raw_missing", "conditional_law"]
)
def test_should_refuse_malformed_mapped_episode_or_probability_law_without_an_environment(
    saved, fault
):
    episode = deepcopy(saved["0-23"]["runs"][0]["episodes"][2])
    if fault == "map_duplicate":
        episode["layout"]["movement_action_map"] = [0, 0, 2, 3]
    elif fault == "map_bool":
        episode["layout"]["movement_action_map"][0] = True
    elif fault == "basis":
        episode["layout"]["observation_basis"] = "button_directions"
    elif fault == "raw_missing":
        del episode["transitions"][0]["raw_probabilities"]
    else:
        episode["transitions"][0]["probabilities"] = episode["transitions"][0][
            "raw_probabilities"
        ].copy()
    report = {
        "status": "complete",
        "protocol": {"protocol_id": "saved_only_fixture"},
        "policy_evaluation": {"episodes": [episode]},
    }
    with pytest.raises(ValueError):
        project_key_door_report(report, source_label="malformed mapped B")
