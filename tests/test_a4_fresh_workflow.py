"""Fresh collector→existing stream→game-loop composition, one numeric fixture."""

from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import minigrid_affine_workflow as workflow
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core import minigrid_key_door as layouts
from src.core.demonstration_episode import DemonstrationEpisode
from src.infra import minigrid_key_door as native
from test_a4_adaptation import literal_record, sources
from test_context_linear_causal_games import PhaseEnvironment

FLAG = "--example-affine-a4"


def bind_collection(patch):
    box: dict[str, Any] = dict(environments=[], teacher_calls=0, fault=None)
    records = literal_record()["collection"]

    class Teacher:
        def choose_action(self, observation):
            box["teacher_calls"] += 1
            return 5

    class Environment:
        def __init__(self, layout, cue, *, seed, max_steps, role, episode_id):
            assert asdict(layout) == records[0]["layout"]
            assert seed == 17 and max_steps == 16 and role == "train"
            self.record = deepcopy(records[0 if cue == "blue" else 1])
            assert self.record["episode_id"] == episode_id
            self.steps, self.closed = 0, False
            box["environments"].append(self)

        @property
        def observation(self):
            return tuple(self.record["transitions"][min(self.steps, 8)]["observation"])

        @property
        def terminal(self):
            return self.steps == 9

        def step(self, action):
            assert action == 5
            self.steps += 1
            if box["fault"] == "step":
                raise RuntimeError("lost collection step return")

        def to_demonstration(self):
            transitions = tuple(
                cli._transition(
                    dict(
                        step=i,
                        observation=t["observation"],
                        action=t["action"],
                        reward=t["reward"],
                        next_observation=t["next_observation"],
                        terminal=t["terminated"],
                        num_classes=7,
                    )
                )
                for i, t in enumerate(self.record["transitions"])
            )
            return DemonstrationEpisode(self.record["episode_id"], "A", "train", transitions, 7)

        def close(self):
            self.closed = True
            if box["fault"] == "close":
                raise RuntimeError("lost collection close return")

        @property
        def report(self):
            report = dict(
                deepcopy(self.record),
                seed=17,
                max_steps=16,
                closed=self.closed,
                status="complete" if self.terminal else "running",
                terminated=self.terminal,
                environment_steps=self.steps,
                environment_steps_attempted=self.steps,
                transitions=deepcopy(self.record["transitions"][: self.steps]),
                total_return=0.5 if self.terminal else 0.0,
            )
            for row in report["transitions"]:
                for name in ("observation", "next_observation"):
                    row[name] = tuple(row[name])
            return report

    patch.setattr(layouts, "CueKeyDoorTeacher", Teacher)
    patch.setattr(native, "MiniGridKeyDoor", Environment)
    patch.setattr(app, "_require_minigrid", lambda: None)
    patch.setattr(app, "example_affine_training", lambda: nullcontext(Path("bundled")))
    patch.setattr(cli, "read_training_sources", lambda path: sources()[:2])
    patch.setattr(app, "read_a4_training", lambda *a: pytest.fail("historical source read"))
    return box


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        box = bind_collection(patch)

        class Environment(PhaseEnvironment):
            work: dict[str, Any] = {}

            @property
            def report(self):
                return dict(
                    super().report,
                    layout=asdict(self.layout),
                    role=self.role,
                    cue=self.cue,
                    seed=self.seed,
                    max_steps=self.max_steps,
                    episode_id=self.episode_id,
                )

        patch.setattr(evaluation, "MiniGridKeyDoor", Environment)
        captured: dict[str, Any] = {}
        original = workflow.run_workflow

        def run(**options):
            assert options == dict(a4_adaptation=True)
            captured["report"] = original(**options)
            return captured["report"]

        patch.setattr(workflow, "run_workflow", run)
        output = tmp_path_factory.mktemp("fresh-a4") / "output.json"
        patch.setattr(sys, "argv", ["affine", FLAG, "--output", str(output)])
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            cli.main()
        return captured["report"], box, output


def test_should_collect_then_train_and_read_all_three_stages_through_actual_cli(integrated):
    report, box, output = integrated
    assert report["status"] == "complete", report.get("reason")
    assert output.is_file() and output.stat().st_size < 8 * 1024**2
    assert box["teacher_calls"] == 18
    assert len(box["environments"]) == 2 and all(e.closed for e in box["environments"])
    assert report["training"]["catalogue"]["committed_rows"] == dict(
        arrivals=54, replay=36, native=90
    )
    assert [p["updates"] for p in report["games"]] == [2, 4, 6]
    assert sum(len(p["games"]["episodes"]) for p in report["games"]) == 12
    assert report["summary"]["complete"] is True
    assert report["training"]["source_unchanged"] is True
    assert report["training"]["actor_unchanged"] is True


def test_should_retain_truthful_fresh_provenance_and_single_pre_collection_clock(integrated):
    report = integrated[0]
    source, budget = report["source"], report["readout_budget"]
    assert source["origin"] == "current_run_A4_TRAIN"
    assert (source["teacher_steps"], source["history_rows"], source["environment_episodes"]) == (
        18,
        18,
        2,
    )
    assert source["collection_seconds"] >= 0
    assert "source_sha256" not in source and "source_bytes" not in source
    assert "source_sha256" not in report["protocol"]
    assert budget["started_at"] == budget["observed_started_at"] < budget["ended_at"]
    assert budget["limits"] == budget["initial_limits"]
    assert budget["limits"]["max_wall_seconds"] == 150
    assert budget["updates"] == budget["replay"] == 0
    assert [r["cue"] for r in report["collection"]] == ["blue", "red"]
    assert all(r["closed"] and r["role"] == "train" for r in report["collection"])
    receipts = report["training"]["committed_receipts"]
    assert [(r["observed_at"], r["arrived_at"], r["applied_at"]) for r in receipts] == [
        (i, i + 1, i + 1) for i in range(6)
    ]


@pytest.mark.parametrize("fault", ["step", "close"])
def test_should_preserve_failed_collection_and_stop_before_fit(monkeypatch, fault):
    box = bind_collection(monkeypatch)
    box["fault"] = fault
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: pytest.fail("fit"))
    report = workflow.run_workflow(a4_adaptation=True)
    assert report["status"] == "failed" and report["training"] is None
    assert len(report["collection"]) == len(box["environments"]) == 1
    assert box["environments"][0].closed
    assert report["collection"][0].get("primary_error") or report["collection"][0].get(
        "cleanup_error"
    )
    assert report["summary"]["complete"] is False
    assert report["readout_budget"]["started_at"] == report["readout_budget"]["observed_started_at"]


@pytest.mark.parametrize("fault", ["mutation", "replacement", "incomplete", "overflow"])
def test_should_refuse_corrupt_paid_collection_before_more_work(monkeypatch, fault):
    monkeypatch.setattr(app, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: pytest.fail("fit"))
    original = literal_record()["collection"][0]

    def producer(guard, records):
        records.append(deepcopy(original))
        guard()
        if fault == "mutation":
            records[0]["total_return"] = 0.25
        elif fault == "replacement":
            records[0] = deepcopy(records[0])
        elif fault == "incomplete":
            records.append(dict(status="failed", closed=True))
        else:
            records.extend([deepcopy(original), deepcopy(original)])
        guard()
        pytest.fail("corrupted prefix admitted")

    report = app._run_adaptation(producer, app.describe_fresh_adaptation())
    assert report["status"] == "failed" and report["training"] is None
    assert report["collection"][0] == original
    assert len(report["collection"]) <= 2


@pytest.mark.parametrize("fault", ["packet", "seed", "rows"])
def test_should_refuse_bad_typed_producer_handoff_before_fit(monkeypatch, fault):
    monkeypatch.setattr(app, "_require_minigrid", lambda: None)
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: pytest.fail("fit"))

    def collect(guard, records, *, layout):
        rows = deepcopy(literal_record()["collection"])
        for row in rows:
            row.update(seed=17, max_steps=16)
        if fault == "seed":
            rows[0]["seed"] = 18
        if fault == "rows":
            rows[0]["environment_steps"] = 8
        records.extend(rows)
        guard()
        return sources()[2:][::-1] if fault == "packet" else sources()[2:]

    monkeypatch.setattr(workflow, "_collect_recurring_train_sources", collect)
    report = workflow.run_workflow(a4_adaptation=True)
    assert report["status"] == "failed" and report["training"] is None
    assert len(report["collection"]) == 2


@pytest.mark.parametrize(
    "options",
    [
        dict(a4_adaptation=1),
        dict(a4_adaptation=True, private_adaptation=True),
        dict(a4_adaptation=True, recurring_training=True),
    ],
)
def test_should_refuse_bad_mode_before_collection(options):
    for function in (workflow.describe_workflow, workflow.run_workflow):
        with pytest.raises(ValueError):
            function(**options)


def test_should_describe_fresh_and_original_modes_without_io(monkeypatch, capsys):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work"))
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--describe"])
    cli.main()
    descriptor = json.loads(capsys.readouterr().out)
    assert descriptor == json.loads(json.dumps(workflow.describe_workflow(a4_adaptation=True)))
    assert descriptor["historical_source_reads"] == 0
    assert app.describe_adaptation()["source_sha256"] == app.SOURCE_SHA
    assert app.describe_adaptation(current_only=True)["source_bytes"] == app.SOURCE_BYTES
    assert workflow.describe_workflow()["definition"] == "portable_affine_train_and_play_v1"
    assert workflow.describe_workflow(recurring_training=True)["new_fit_rows"] == 252
    assert workflow.describe_workflow(private_adaptation=True)["new_fit_rows"] == 252


@pytest.mark.parametrize(
    "args",
    [
        ["--source", "missing"],
        ["--retained-source", "missing"],
        ["--fit-updates", "2"],
        ["--cue-relative"],
        ["--example-affine-recurring"],
        ["--describe", "--output", "missing"],
    ],
)
def test_should_refuse_conflicting_cli_before_any_work(monkeypatch, args):
    monkeypatch.setattr(workflow, "run_workflow", lambda **k: pytest.fail("work"))
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, *args])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_should_preserve_exclusive_output_and_save_failure(monkeypatch, tmp_path, capsys):
    output = tmp_path / "failed.json"
    calls = []

    def fail(**kwargs):
        calls.append(kwargs)
        return dict(status="failed", collection=[dict(status="construction_failed")])

    monkeypatch.setattr(workflow, "run_workflow", fail)
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--output", str(output)])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(output.read_text())["status"] == "failed"
    capsys.readouterr()
    with pytest.raises(FileExistsError):
        cli.main()
    assert calls == [dict(a4_adaptation=True)]


def test_should_keep_historical_reader_strict_and_native_record_decoder_compatible(tmp_path):
    packets, provenance = app._admit_a4_record(literal_record())
    assert packets == sources()[2:]
    assert provenance["source_sha256"] == app.SOURCE_SHA
    file = tmp_path / "unqualified.json"
    file.write_text(json.dumps(literal_record()))
    with pytest.raises(ValueError, match="exact qualified170"):
        app.read_a4_training(file)


@pytest.mark.parametrize("fault", ["source", "training", "prior_phase", "clock", "budget"])
def test_should_preserve_cumulative_authority_and_paid_phase_prefix(integrated, monkeypatch, fault):
    initial = integrated[0]
    provenance = deepcopy(initial["source"])
    training = deepcopy(initial["training"])
    monkeypatch.setattr(app, "_require_minigrid", lambda: None)
    monkeypatch.setattr(app, "example_affine_training", lambda: nullcontext(Path("literal")))
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources()[:2])
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: training)

    def producer(guard, records):
        records.extend(deepcopy(initial["collection"]))
        guard()
        return sources()[2:], provenance

    seen: list[dict[str, Any]] = []

    def games(*args, budget, check_boundary, **kwargs):
        if fault == "source":
            provenance["teacher_steps"] += 1
        elif fault == "training":
            training["committed_receipts"][0]["update_number"] = 99
        elif fault == "clock":
            budget.started_at += 1
        elif fault == "budget":
            budget.budget = replace(budget.budget, max_wall_seconds=151)
        elif seen:
            seen[0]["episodes"][0]["success"] = not seen[0]["episodes"][0]["success"]
        check_boundary()
        observed = deepcopy(initial["games"][len(seen)]["games"])
        seen.append(observed)
        return observed

    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", games)
    report = app._run_adaptation(producer, app.describe_fresh_adaptation())
    assert report["status"] == "failed" and report["summary"]["complete"] is False
    assert len(seen) <= 1
    assert report["source"] == initial["source"]
    assert report["training"] == initial["training"]


def test_should_reject_missing_runtime_before_collection_and_training(monkeypatch):
    def unavailable():
        raise RuntimeError("optional runtime unavailable")

    monkeypatch.setattr(app, "_require_minigrid", unavailable)
    monkeypatch.setattr(app, "_collect_fresh_a4", lambda *a: pytest.fail("collection"))
    result = workflow.run_workflow(a4_adaptation=True)
    assert result["status"] == "failed" and result["collection"] == []
    assert result["source"] is result["training"] is None


def test_should_enforce_fresh_size_limit_without_changing_original_limit(monkeypatch, tmp_path):
    output = tmp_path / "oversized.json"
    monkeypatch.setattr(
        workflow, "run_workflow", lambda **k: dict(status="complete", payload="x" * (8 * 1024**2))
    )
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--output", str(output)])
    with pytest.raises(ValueError, match="exceeds8MiB"):
        cli.main()
    assert not output.exists()


def test_should_handoff_actual_in_memory_tuple_records_without_fit(monkeypatch):
    box = bind_collection(monkeypatch)
    records: list[dict[str, Any]] = []
    packets, provenance = app._collect_fresh_a4(lambda: None, records)
    assert packets == sources()[2:]
    assert all(type(r["transitions"][0]["observation"]) is tuple for r in records)
    assert provenance["teacher_steps"] == box["teacher_calls"] == 18
    assert all(e.closed for e in box["environments"])
    assert len(records) == 2 and provenance["origin"] == "current_run_A4_TRAIN"


@pytest.mark.parametrize("fault", ["short", "string", "nan", "tuple_next"])
def test_should_normalize_only_native_tuple_representation_and_keep_codec_checks(fault):
    records = deepcopy(literal_record()["collection"])
    for record in records:
        for transition in record["transitions"]:
            for name in ("observation", "next_observation"):
                transition[name] = tuple(transition[name])
    row = records[0]["transitions"][0]
    if fault == "short":
        row["observation"] = row["observation"][:-1]
    elif fault == "string":
        row["observation"] = "x" * 148
    elif fault == "nan":
        row["observation"] = (float("nan"), *row["observation"][1:])
    else:
        row["next_observation"] = (2.0, *row["next_observation"][1:])
    with pytest.raises(ValueError):
        app._admit_a4_native_records(records)
