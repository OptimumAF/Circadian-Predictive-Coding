"""The opt-in return assay reuses one parsed source and its original TRAIN A evidence."""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import pytest

import scripts.run_minigrid_memory_probe as cli
from test_minigrid_memory_probe_cli import SOURCE, forbidden, replace, saved as saved, write_source


def test_should_load_original_a_episodes_and_native_states_from_exactly_one_source_read(
    saved, monkeypatch
):
    original_read = cli._read_source
    calls = []

    def read(path):
        calls.append(path)
        return original_read(path)

    monkeypatch.setattr(cli, "_read_source", read)
    before = SOURCE.read_bytes()
    states, episodes = cli.load_return_inputs(SOURCE)
    assert calls == [SOURCE]
    assert [(s.completed_updates, s.training_examples, len(s.entries)) for s in states] == [
        (0, 0, 0),
        (128, 1152, 54),
    ]
    assert type(episodes) is tuple and len(episodes) == 2
    assert [episode.episode_id for episode in episodes] == [
        "minigrid-source-A-blue",
        "minigrid-source-A-red",
    ]
    assert all(
        episode.context == "A"
        and episode.role == "train"
        and episode.num_classes == 7
        and len(episode.transitions) == 9
        for episode in episodes
    )
    for actual, raw in zip(episodes, saved["training_sources"][:2], strict=True):
        assert json.loads(json.dumps(asdict(actual))) == raw
        assert all(
            len(step.observation) == len(step.next_observation) == 148
            for step in actual.transitions
        )
    object.__setattr__(episodes[0], "role", "final_test")
    assert cli.load_return_inputs(SOURCE)[1][0].role == "train"
    assert SOURCE.read_bytes() == before


@pytest.mark.parametrize(
    "pointer,value",
    [
        ("/training_sources/0/role", "probe_train"),
        ("/training_sources/0/context", "A-blue"),
        ("/training_sources/0/episode_id", "copied-unknown-A-blue"),
        ("/training_sources/0/num_classes", 5),
        ("/training_sources/0/transitions/0/action", 1),
        ("/training_sources/0/transitions/0/reward", 0.25),
        ("/training_sources/0/transitions/0/terminal", True),
        ("/training_sources/0/transitions/0/observation", [0.0] * 296),
        ("/source_reports/0/role", "final_test"),
        ("/source_reports/0/context", "D"),
        ("/source_reports/0/episode_id", "not-original"),
        ("/source_reports/0/status", "failed"),
        ("/source_reports/0/transitions/0/action", 1),
        ("/source_reports/0/transitions/0/reward", 0.25),
        ("/source_reports/0/transitions/0/applied", False),
        ("/source_reports/0/transitions/0/attempted", False),
        ("/source_reports/0/transitions/8/terminated", False),
    ],
)
def test_should_refuse_unbound_or_retyped_return_teacher_evidence(saved, tmp_path, pointer, value):
    document = deepcopy(saved)
    replace(document, pointer, value)
    source = write_source(tmp_path, document)
    before = source.read_bytes()
    with pytest.raises(ValueError):
        cli.load_return_inputs(source)
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "extra_episode",
        "extra_transition",
        "missing_source_report",
        "swapped_source_reports",
        "missing_transition",
    ],
)
def test_should_refuse_ambiguous_or_missing_a_source_record_shapes(saved, tmp_path, fault):
    document = deepcopy(saved)
    if fault == "extra_episode":
        document["training_sources"][0]["extra"] = 1
    elif fault == "extra_transition":
        document["training_sources"][0]["transitions"][0]["extra"] = 1
    elif fault == "missing_source_report":
        document["source_reports"].pop(0)
    elif fault == "swapped_source_reports":
        document["source_reports"][0], document["source_reports"][1] = (
            document["source_reports"][1],
            document["source_reports"][0],
        )
    else:
        document["training_sources"][0]["transitions"].pop()
    with pytest.raises(ValueError):
        cli.load_return_inputs(write_source(tmp_path, document))


def test_should_preserve_original_snapshot_only_loader_without_a_source_report_dependency(
    saved, tmp_path
):
    document = deepcopy(saved)
    document.pop("source_reports")
    path = write_source(tmp_path, document)
    assert cli.load_snapshots(path)[1].completed_updates == 128
    with pytest.raises(ValueError):
        cli.load_return_inputs(path)


def test_should_preview_new_return_protocol_without_loading_or_running_either_mode(
    monkeypatch, capsys
):
    expected = {"protocol_id": "return-preview", "cohorts": 4}
    monkeypatch.setattr(cli, "describe_return_protocol", lambda: deepcopy(expected))
    for name in (
        "load_snapshots",
        "load_return_inputs",
        "run_pilot",
        "run_return_pilot",
        "describe_protocol",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    assert cli.main(["--return-comparison", "--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_should_route_opt_in_return_inputs_once_and_preserve_negative_publication(
    tmp_path, monkeypatch, capsys
):
    output = tmp_path / "return.json"
    states: Any = (object(), object())
    episodes: Any = (object(), object())
    calls: list[tuple[Any, ...]] = []

    def load(path):
        calls.append(("load", path))
        return states, episodes

    def run(received_states, received_episodes):
        assert received_states is states and received_episodes is episodes
        calls.append(("run",))
        return {
            "status": "incomplete",
            "cohorts": [{"status": "failed", "charged_updates": 1}],
            "reason": "expected capacity refusal",
        }

    monkeypatch.setattr(cli, "load_return_inputs", load)
    monkeypatch.setattr(cli, "run_return_pilot", run)
    monkeypatch.setattr(cli, "load_snapshots", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    before = SOURCE.read_bytes()
    assert (
        cli.main(["--return-comparison", "--source-artifact", str(SOURCE), "--output", str(output)])
        == 2
    )
    assert calls == [("load", SOURCE), ("run",)]
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "incomplete" and report["cohorts"][0]["charged_updates"] == 1
    assert json.loads(capsys.readouterr().out) == report and SOURCE.read_bytes() == before


@pytest.mark.parametrize(
    "args",
    [["--return-comparison"], ["--return-comparison", "--describe", "--source-artifact", "x.json"]],
)
def test_should_refuse_invalid_return_cli_arguments_before_both_loaders(monkeypatch, args):
    for name in ("load_snapshots", "load_return_inputs", "run_pilot", "run_return_pilot"):
        monkeypatch.setattr(cli, name, forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(args)
    assert raised.value.code == 2


def test_should_refuse_existing_return_output_before_source_loading(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_bytes(b"prior negative stays")
    monkeypatch.setattr(cli, "load_return_inputs", forbidden)
    monkeypatch.setattr(cli, "run_return_pilot", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--return-comparison", "--source-artifact", str(SOURCE), "--output", str(output)])
    assert raised.value.code == 2 and output.read_bytes() == b"prior negative stays"
