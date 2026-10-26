"""Saved public-history evidence admission and offline publication, without learning."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

import scripts.build_key_door_inspector as cli
from src.app.key_door_inspector import _episode, project_key_door_report
from src.app.public_history_inspector_projection import project_public_history_report
import src.infra.key_door_inspector_files as files
from tests.test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("bounded-action-memory", "memory_live", ("observation_memory",)),
    ("action-memory-probes", "memory_probe", ("observation_memory",)),
    ("matched-history-live", "matched_live", ("bp", "pc", "neutral_cpc", "gated_cpc")),
)


def artifact(name: str) -> Path:
    return ROOT / "artifacts" / "runs" / (name + "-20261017") / "native.json"


@pytest.fixture(scope="module")
def saved() -> dict[str, Any]:
    return {name: json.loads(artifact(name).read_text(encoding="utf-8")) for name, _, _ in CASES}


def at(document: Any, pointer: str) -> Any:
    value = document
    for token in pointer.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if type(value) is list else value[token]
    return value


def replace(document: Any, pointer: str, value: Any) -> None:
    parent, _, token = pointer.rpartition("/")
    target = at(document, parent)
    if type(target) is list:
        target[int(token)] = value
    else:
        target[token] = value


@pytest.mark.parametrize("name,mode,methods", CASES)
def test_should_project_every_saved_method_episode_and_exact_source_pointer(
    saved: dict[str, Any], name: str, mode: str, methods: tuple[str, ...]
) -> None:
    original = deepcopy(saved[name])
    dto = project_public_history_report(original, source_label=name + ".json")
    assert dto["schema_id"] == "public_history_inspector_v1"
    assert dto["mode"] == mode and dto["origin"] == "recorded_artifact"
    assert dto["source_label"] == name + ".json" and dto["source"] == original
    assert dto["unknown"] == {
        "topology": None,
        "lineage": None,
        "sleep_proposals": None,
        "promotion": None,
    }
    assert [row["method"] for row in dto["runs"]] == list(methods)
    for index, row in enumerate(dto["runs"]):
        assert row["pointer"] == f"/runs/{index}"
        assert row["status"] == original["runs"][index]["status"]
        assert len(row["episodes"]) == (0 if mode == "memory_probe" else 30)
        for episode_index, episode in enumerate(row["episodes"]):
            assert episode["pointer"] == f"/runs/{index}/live_episodes/{episode_index}"
            assert at(original, episode["pointer"]) == episode["record"]
            assert episode["record"]["source_role"] == "train"
            assert len(episode["frames"]) == len(episode["record"]["transitions"]) + 1
    assert original == saved[name]
    assert project_key_door_report(original, source_label=name + ".json") == dto


@pytest.mark.parametrize("name,mode,methods", CASES)
def test_should_detach_the_full_original_and_projected_episode_values(
    saved: dict[str, Any], name: str, mode: str, methods: tuple[str, ...]
) -> None:
    original = deepcopy(saved[name])
    before = deepcopy(original)
    dto = project_public_history_report(original, source_label="owned.json")
    original["runs"][0]["observations"][-1]["native_updates"] = 999
    assert dto["source"] == before
    dto["source"]["runs"][0]["observations"][-1]["native_updates"] = 888
    assert original["runs"][0]["observations"][-1]["native_updates"] == 999
    if mode != "memory_probe":
        record = dto["runs"][0]["episodes"][0]["record"]
        record["transitions"][0]["probabilities"][0] = 7
        assert original["runs"][0]["live_episodes"][0] == before["runs"][0]["live_episodes"][0]
        assert dto["source"]["runs"][0]["live_episodes"][0] == before["runs"][0]["live_episodes"][0]
    assert saved[name] == before


def test_should_preserve_actual_neural_negatives_and_raw_illegal_action_probabilities(saved):
    report = saved["matched-history-live"]
    dto = project_public_history_report(report, source_label="negative.json")
    assert len(dto["runs"]) == 4
    assert any(row["outcomes"]["new_C_gain"] < 0 for row in report["runs"])
    for original, projected in zip(report["runs"], dto["runs"], strict=True):
        assert (
            dto["source"]["runs"][report["runs"].index(original)]["outcomes"]
            == original["outcomes"]
        )
        assert all(point["success_rate"] == 0 for point in original["live_outcomes"])
        illegal = next(
            ep for ep in projected["episodes"] if ep["record"]["outcome"] == "illegal_action"
        )
        decision = illegal["record"]["transitions"][-1]
        assert decision["action"] not in decision["legal_actions"]
        assert any(
            probability > 0
            for action, probability in enumerate(decision["probabilities"])
            if action not in decision["legal_actions"]
        )
        assert decision["applied"] is False
    assert dto["source"]["runs"][-1]["method"] == "gated_cpc"
    assert dto["source"]["runs"][-1]["observations"][-1] == report["runs"][-1]["observations"][-1]


def test_should_preserve_probe_origins_charges_and_nulls_without_inventing_live_playback(saved):
    report = saved["action-memory-probes"]
    dto = project_public_history_report(report, source_label="assay.json")
    row = dto["source"]["runs"][0]
    assert dto["runs"][0]["episodes"] == [] and "live_episodes" not in row
    assert [slot["checkpoint_updates"] for slot in row["probes"]] == [0, 8]
    for slot in row["probes"]:
        assert slot["charged_updates"] == 16
        assert slot["report"]["completed_updates"] == 16
        assert slot["report"]["completed_training_examples"] == 128
        assert slot["report"]["score_forward_calls_completed"] == 8
        assert [point["updates"] for point in slot["report"]["checkpoints"]] == [0, 4, 8, 16]
        assert "logical_payload_bytes" not in slot["report"]
    assert row["observations"][-1]["entries"] == 48
    assert row["observations"][-1]["logical_payload_bytes"] == 8064
    assert dto["source"]["pc_neutral_exact"] is None


def test_should_require_the_explicit_public_history_selector_for_raw_train(saved):
    raw = deepcopy(saved["matched-history-live"]["runs"][0]["live_episodes"][0])
    with pytest.raises(ValueError, match="TRAIN"):
        _episode(raw, "/raw")
    with pytest.raises(ValueError):
        _episode(raw, "/raw", train=True, public_history=True)
    assert _episode(raw, "/raw", public_history=True)["record"] == raw


@pytest.mark.parametrize("status", ["incomplete", "failed", "not_run"])
@pytest.mark.parametrize("missing", [False, True])
def test_should_preserve_partial_rows_and_missing_or_recorded_nulls_without_fabrication(
    saved, status, missing
):
    report = deepcopy(saved["bounded-action-memory"])
    report["status"] = status
    report["runs"] = [{"method": "observation_memory", "status": status, "reason": "not observed"}]
    if not missing:
        report["runs"][0].update(observations=None, stream_result=None, live_episodes=None)
    dto = project_public_history_report(report, source_label="partial.json")
    assert dto["source"] == report
    assert dto["runs"][0]["status"] == status and dto["runs"][0]["episodes"] == []
    assert ("observations" not in dto["source"]["runs"][0]) is missing


def test_should_preserve_failed_and_unreached_episode_slots_and_paid_work(saved):
    report = deepcopy(saved["bounded-action-memory"])
    report["status"] = "incomplete"
    row = report["runs"][0]
    row.update(status="failed", reason="recorded prediction failure", live_outcomes=None)
    failed = deepcopy(row["live_episodes"][0])
    failed.update(status="failed", reason="recorded failure", state_unchanged=None)
    not_run = {
        key: deepcopy(row["live_episodes"][1][key])
        for key in ("checkpoint_updates", "context", "cue", "layout_partition", "source_role")
    }
    not_run.update(status="not_run", reason=None, state_unchanged=None, layout=None)
    row["live_episodes"] = [failed, not_run]
    dto = project_public_history_report(report, source_label="paid-partial.json")
    assert dto["source"]["runs"][0]["live_work"] == row["live_work"]
    assert [ep["record"]["status"] for ep in dto["runs"][0]["episodes"]] == ["failed", "not_run"]
    assert dto["runs"][0]["episodes"][1]["frames"] == []


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/protocol/protocol_id", "local_key_door_bounded_action_memory_v2"),
        ("/status", None),
        ("/runs/0/status", "won"),
        ("/runs/0/method", "bp"),
        ("/protocol/methods", ["observation_memory", "bp"]),
        ("/total_seconds", float("nan")),
        ("/observed_process_rss/peak_bytes", -1),
        ("/runs/0/wake_updates", True),
        ("/runs/0/work/score_forwards_completed", -1),
        ("/runs/0/observations/4/native_updates", False),
        ("/runs/0/observations/4/entries", 65),
        ("/runs/0/observations/4/logical_payload_bytes", 10753),
        ("/runs/0/observations/4/conflicting_keys", -1),
        ("/runs/0/observations/4/snapshot/policy/feature_dim", 8),
        ("/runs/0/observations/4/snapshot/policy/max_entries", 128),
        ("/runs/0/observations/4/snapshot/entries/0/features", [0] * 15),
        ("/runs/0/observations/4/snapshot/entries/0/counts", [True, 0, 0, 0, 0]),
        ("/runs/0/observations/4/snapshot/entries/0/counts/0", -1),
        ("/runs/0/training_points/0/scores/A-red/accuracy", 1.01),
        ("/runs/0/live_episodes/0/source_role", "final_test"),
        ("/runs/0/live_episodes/0/transitions/0/probabilities/0", float("inf")),
    ],
)
def test_should_refuse_malformed_known_fields_in_saved_memory_evidence(saved, pointer, value):
    report = deepcopy(saved["bounded-action-memory"])
    replace(report, pointer, value)
    with pytest.raises(ValueError):
        project_public_history_report(report, source_label="corrupt.json")


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "reordered", "missing"])
def test_should_refuse_complete_matched_reports_with_wrong_method_identity_or_slots(saved, kind):
    report = deepcopy(saved["matched-history-live"])
    if kind == "duplicate":
        report["runs"][1]["method"] = "bp"
    elif kind == "unknown":
        report["runs"][1]["method"] = "inactive_hidden_control"
    elif kind == "reordered":
        report["runs"].reverse()
    else:
        report["runs"].pop()
    with pytest.raises(ValueError):
        project_public_history_report(report, source_label="wrong-methods.json")


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/runs/0/probes/0/charged_updates", False),
        ("/runs/0/probes/0/report/completed_updates", 0),
        ("/runs/0/probes/0/report/completed_training_examples", 127),
        ("/runs/0/probes/0/report/score_forward_calls_completed", 7),
        ("/runs/0/probes/1/checkpoint_updates", 0),
    ],
)
def test_should_refuse_probe_work_or_checkpoint_contradictions_in_complete_reports(
    saved, pointer, value
):
    report = deepcopy(saved["action-memory-probes"])
    replace(report, pointer, value)
    with pytest.raises(ValueError):
        project_public_history_report(report, source_label="corrupt-probe.json")


@pytest.mark.parametrize(
    "field,maximum", [("live_episodes", 30), ("observations", 5), ("training_points", 5)]
)
def test_should_refuse_excess_recorded_arrays_without_truncating_evidence(saved, field, maximum):
    report = deepcopy(saved["bounded-action-memory"])
    row = report["runs"][0]
    row[field] = [deepcopy(row[field][0]) for _ in range(maximum + 1)]
    with pytest.raises(ValueError):
        project_public_history_report(report, source_label="oversized-array.json")


@pytest.mark.parametrize("name,mode,methods", CASES)
def test_should_build_actual_saved_report_offline_without_modifying_its_source(
    saved, tmp_path, name, mode, methods
):
    source = artifact(name)
    before = source.read_bytes()
    output = tmp_path / (name + ".html")
    files.write_key_door_inspector(source, output)
    page = Page(output.read_text(encoding="utf-8"))
    payload = page.payload()
    assert payload == project_public_history_report(saved[name], source_label="native.json")
    assert source.read_bytes() == before
    for _, attributes in page.tags:
        for key in ("src", "href"):
            value = attributes.get(key)
            assert value is None or not value.lower().startswith(("https:", "http:", "//"))
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, output)
    assert source.read_bytes() == before


def test_should_escape_source_labels_and_recorded_reasons_as_inert_json(saved, tmp_path):
    danger = '</script><img id="injected" src="x" onerror="alert(1)"> & < >'
    report = deepcopy(saved["bounded-action-memory"])
    report["runs"][0]["reason"] = danger
    dto = project_public_history_report(report, source_label=danger)
    page = Page(files.render_key_door_inspector(dto).decode("utf-8"))
    assert page.payload() == dto
    assert not any(attributes.get("id") == "injected" for _, attributes in page.tags)
    embedded = "".join(page.payload_parts)
    assert "<" not in embedded and ">" not in embedded and "&" not in embedded


def test_should_refuse_unverified_separate_resources_and_preserve_existing_files(saved, tmp_path):
    source = artifact("bounded-action-memory")
    resources = source.with_name("native-resources.json")
    output = tmp_path / "uncreated" / "inspector.html"
    with pytest.raises(ValueError):
        project_public_history_report(
            saved["bounded-action-memory"],
            source_label="native.json",
            resources={},
            resource_label="other.json",
        )
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, output, resource_artifact=resources)
    assert not output.parent.exists()
    before = source.read_bytes()
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, source)
    assert source.read_bytes() == before


def test_should_refuse_more_than_eight_mebibytes_before_any_output(tmp_path):
    source = tmp_path / "too-large.json"
    with source.open("wb") as stream:
        stream.write(b"{}")
        stream.truncate(8 * 1024**2 + 1)
    output = tmp_path / "uncreated" / "inspector.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, output)
    assert not output.parent.exists()


def test_should_route_the_existing_builder_cli_to_the_new_saved_report_projection(
    tmp_path, monkeypatch, capsys
):
    source = artifact("action-memory-probes")
    output = tmp_path / "probe.html"
    monkeypatch.setattr(
        sys,
        "argv",
        ["build_key_door_inspector", "--source-artifact", str(source), "--output", str(output)],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == str(source.resolve())
    assert result["output"] == str(output.resolve())
    assert result["bytes"] == output.stat().st_size
    assert Page(output.read_text(encoding="utf-8")).payload()["mode"] == "memory_probe"


def test_should_refuse_partial_snapshot_from_a_different_observed_checkpoint(saved):
    report = deepcopy(saved["bounded-action-memory"])
    report["status"] = "incomplete"
    row = report["runs"][0]
    row["status"] = "failed"
    initial_snapshot = deepcopy(row["observations"][0]["snapshot"])
    row["observations"] = [dict(wake_updates=2, snapshot=initial_snapshot)]
    with pytest.raises(ValueError, match="clock"):
        project_public_history_report(report, source_label="wrong-clock.json")
