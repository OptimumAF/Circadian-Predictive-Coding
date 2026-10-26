"""Saved TRAIN report publication through the real projection and offline file CLI."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

import scripts.build_key_door_inspector as cli
import src.infra.key_door_inspector_files as files
from test_key_door_inspector_files import Page


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    "off": ROOT / "artifacts/runs/r21-acquired-task-return-20261016/pilot.json",
    "recent48": ROOT / "artifacts/runs/r21-acquired-replay-20261016/recent48/pilot.json",
    "coverage": ROOT / "artifacts/runs/r21-acquired-replay-20261016/coverage/pilot.json",
}


@pytest.fixture(scope="module")
def accepted():
    return {name: json.loads(path.read_text(encoding="utf8")) for name, path in FIXTURES.items()}


def pointer(document, reference):
    value = document
    for token in reference.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if type(value) is list else value[token]
    return value


@pytest.mark.parametrize("control", ["off", "recent48", "coverage"])
def test_should_publish_all_twelve_saved_train_records_with_exact_pointers_clocks_and_costs(
    tmp_path, accepted, control
):
    source = FIXTURES[control]
    before = source.read_bytes()
    destination = tmp_path / f"{control}-train.html"
    written = files.write_key_door_inspector(source, destination)
    assert written.resolve() == destination.resolve() and source.read_bytes() == before
    assert 0 < len(before) <= files.MAX_INPUT_BYTES
    assert 0 < destination.stat().st_size <= files.MAX_OUTPUT_BYTES
    page = Page(destination.read_text(encoding="utf8"))
    payload = page.payload()
    original = accepted[control]
    assert (
        payload["schema_id"] == "key_door_inspector_v1" and payload["origin"] == "recorded_artifact"
    )
    assert payload["source_label"] == source.name and payload["source"] == original
    assert len(payload["conditions"]) == 1
    condition = payload["conditions"][0]
    assert condition["key"] == "single" and condition["pointer"] == ""
    assert condition["readout_key"] == "acquired_return_diagnostics"
    assert condition["source_role"] == "train"
    assert condition["unknown"] == dict(topology=None, lineage=None, sleep_proposals=None)
    episodes = condition["episodes"]
    original_episodes = original["acquired_return_diagnostics"]["episodes"]
    assert len(episodes) == len(original_episodes) == 12
    assert [episode["record"]["success"] for episode in episodes] == [
        episode["success"] for episode in original_episodes
    ]
    assert [episode["record"]["return"] for episode in episodes] == [
        episode["return"] for episode in original_episodes
    ]
    for index, (episode, recorded) in enumerate(zip(episodes, original_episodes, strict=True)):
        expected_pointer = f"/acquired_return_diagnostics/episodes/{index}"
        assert episode["pointer"] == expected_pointer
        assert episode["record"] == pointer(payload["source"], expected_pointer) == recorded
        assert episode["record"]["source_role"] == "train"
        assert episode["record"]["phase"] == "evaluation"  # Preserve the original episode field.
        assert episode["record"]["learning_active"] is False
        assert len(episode["frames"]) == len(recorded["transitions"]) + 1
        assert episode["frames"][0]["visible_cue"] == recorded["cue"]
        assert all(
            "raw_probabilities" in transition for transition in episode["record"]["transitions"]
        )
    for field in (
        "acquired_return",
        "acquired_initial_condition_met",
        "acquired_interpretation",
        "total_seconds",
        "observed_process_rss",
        "work",
        "partial_stream",
    ):
        assert payload["source"][field] == original[field]
    readout = payload["source"][condition["readout_key"]]
    assert readout["outcomes"] == original["acquired_return_diagnostics"]["outcomes"]
    assert readout["metrics"] == original["acquired_return_diagnostics"]["metrics"]
    assert readout["work"] == original["acquired_return_diagnostics"]["work"]
    if control == "off":
        assert "checkpoint_clocks" not in readout
        assert all("native_checkpoint_updates" not in episode["record"] for episode in episodes)
        assert "replay" not in payload["source"]
    else:
        assert readout["replay_control"] == payload["source"]["replay_control"] == control
        assert [clock["wake_updates"] for clock in readout["checkpoint_clocks"]] == [192, 256, 320]
        assert [clock["native_updates"] for clock in readout["checkpoint_clocks"]] == [
            288,
            384,
            480,
        ]
        assert (
            readout["checkpoint_clocks"]
            == original["acquired_return_diagnostics"]["checkpoint_clocks"]
        )
        assert payload["source"]["replay"] == original["replay"]
        assert payload["source"]["replay"]["charged_examples"] == 1280
        assert (
            payload["source"]["acquired_new_task_condition_met"]
            == original["acquired_new_task_condition_met"]
        )
    for _, attributes in page.tags:
        assert not any(
            (attributes.get(field) or "").lower().startswith(("http:", "https:", "//"))
            for field in ("src", "href")
        )


@pytest.mark.parametrize(
    "fault",
    [
        "mixed_readouts",
        "paired_train",
        "readout_role",
        "episode_role",
        "protocol",
        "control",
        "clocks",
        "missing_clocks",
        "checkpoint",
    ],
)
def test_should_refuse_mixed_or_mislabeled_train_evidence_before_creating_output(
    tmp_path, accepted, fault
):
    report = deepcopy(accepted["coverage"])
    if fault == "mixed_readouts":
        report["policy_evaluation"] = dict(status="complete", episodes=[])
    elif fault == "paired_train":
        report = dict(status="complete", conditions=dict(off=deepcopy(accepted["off"]), on=report))
    elif fault == "readout_role":
        report["acquired_return_diagnostics"]["source_role"] = "final_test"
    elif fault == "episode_role":
        report["acquired_return_diagnostics"]["episodes"][0]["source_role"] = "final_test"
    elif fault == "protocol":
        report["protocol"]["protocol_id"] = "key_door_return_policy_evaluation_v1"
    elif fault == "control":
        report["replay_control"] = "recent48"
    elif fault == "clocks":
        report["acquired_return_diagnostics"]["checkpoint_clocks"][0]["native_updates"] += 1
    elif fault == "missing_clocks":
        del report["acquired_return_diagnostics"]["checkpoint_clocks"]
    else:
        episode = report["acquired_return_diagnostics"]["episodes"][0]
        episode["checkpoint_updates"] = episode["identity"]["checkpoint_updates"] = 193
    source = tmp_path / "invalid-recorded.json"
    before = json.dumps(report, allow_nan=False).encode("utf8")
    assert len(before) <= files.MAX_INPUT_BYTES
    source.write_bytes(before)
    destination = tmp_path / "uncreated" / "inspector.html"
    with pytest.raises(ValueError):
        files.write_key_door_inspector(source, destination)
    assert source.read_bytes() == before and not destination.parent.exists()


def test_should_preserve_incomplete_outer_status_without_hiding_completed_train_readout(
    tmp_path, accepted
):
    report = deepcopy(accepted["off"])
    report.update(
        status="incomplete", reason="recorded resource readback interrupted after live readout"
    )
    source, destination = tmp_path / "partial.json", tmp_path / "partial.html"
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    files.write_key_door_inspector(source, destination)
    payload = Page(destination.read_text(encoding="utf8")).payload()
    assert payload["source"] == report and source.read_bytes() == before
    assert payload["conditions"][0]["status"] == "incomplete"
    assert payload["conditions"][0]["source_role"] == "train"
    assert len(payload["conditions"][0]["episodes"]) == 12


def test_should_build_train_report_through_actual_cli_and_keep_unsafe_source_text_inert(
    tmp_path, accepted, monkeypatch, capsys
):
    source, destination = tmp_path / "recorded-train.json", tmp_path / "actual-cli.html"
    report = deepcopy(accepted["off"])
    unsafe = '</script><img id="injected-train" src="x" onerror="alert(1)"> & < > \u2028\u2029'
    report["recorded_annotation"] = unsafe
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["inspector", "--source-artifact", str(source), "--output", str(destination)]
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == dict(
        output=str(destination.resolve()),
        source=str(source.resolve()),
        bytes=destination.stat().st_size,
    )
    assert source.read_bytes() == before
    page = Page(destination.read_text(encoding="utf8"))
    assert page.payload()["source"]["recorded_annotation"] == unsafe
    assert page.payload()["conditions"][0]["readout_key"] == "acquired_return_diagnostics"
    assert not any(attributes.get("id") == "injected-train" for _, attributes in page.tags)
    assert "<" not in "".join(page.payload_parts) and "&" not in "".join(page.payload_parts)


def test_should_report_actual_cli_validation_error_without_publishing_or_mutating_source(
    tmp_path, accepted, monkeypatch, capsys
):
    report = deepcopy(accepted["off"])
    report["acquired_return_diagnostics"]["source_role"] = "final_test"
    source, destination = tmp_path / "wrong-role.json", tmp_path / "uncreated" / "failed.html"
    source.write_text(json.dumps(report, allow_nan=False), encoding="utf8")
    before = source.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["inspector", "--source-artifact", str(source), "--output", str(destination)]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and source.read_bytes() == before
    output = capsys.readouterr()
    assert output.out == "" and output.err and not destination.parent.exists()


def test_should_preserve_existing_published_html_and_source_for_new_train_selector(tmp_path):
    source, destination = FIXTURES["off"], tmp_path / "existing.html"
    before = source.read_bytes()
    destination.write_bytes(b"user-owned existing dashboard")
    with pytest.raises(FileExistsError):
        files.write_key_door_inspector(source, destination)
    assert (
        source.read_bytes() == before
        and destination.read_bytes() == b"user-owned existing dashboard"
    )
