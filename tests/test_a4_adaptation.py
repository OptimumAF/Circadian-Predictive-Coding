"""One6fit90row literal integration shared by role/receipt/native-port checks."""

from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_arrived_cue_catalogue import episode
from test_context_linear_causal_games import PhaseEnvironment, budget

CAPTURE: dict[str, Any] = {}


def sources():
    return tuple(
        episode(f"minigrid-source-{identity}-{cue}", color, context="A", mark=mark, action=action)
        for identity, mark, action in (("A", 1, 2), ("A4-train", 3, 5))
        for cue, color in (("blue", 2), ("red", 0))
    )


def literal_record():
    rows = []
    for packet, cue in zip(sources()[2:], ("blue", "red"), strict=True):
        rows.append(
            dict(
                episode_id=packet.episode_id,
                context="A",
                role="train",
                cue=cue,
                layout=dict(
                    context="A", rotation=0, branch_row=4, swap_keys=False, wall_color="grey"
                ),
                status="complete",
                closed=True,
                terminated=True,
                truncated=False,
                environment_steps=9,
                environment_steps_attempted=9,
                total_return=0.5,
                failure_work_may_be_unobserved=False,
                transitions=[
                    dict(
                        observation=list(t.observation[148:]),
                        action=t.action,
                        reward=t.reward,
                        next_observation=list(t.next_observation[148:]),
                        terminated=t.terminal,
                        applied=True,
                    )
                    for t in packet.transitions
                ],
            )
        )
    return dict(
        status="complete",
        new_fits=0,
        new_replay=0,
        history_rows=18,
        collection=rows,
        episodes=json.loads(json.dumps([asdict(p) for p in sources()[2:]])),
        budget=dict(updates=0, replay=0, started_at=1.0, observed_started_at=1.0),
        counts=dict(forbidden=0),
        seconds=1.0,
        sampled_peak_rss_bytes=1000,
    )


def bind_inputs(monkeypatch):
    fresh, provenance = app._admit_a4_record(literal_record())
    monkeypatch.setattr(app, "read_a4_training", lambda path: (fresh, deepcopy(provenance)))
    monkeypatch.setattr(app, "example_affine_training", lambda: nullcontext(Path("literal")))
    monkeypatch.setattr(cli, "read_training_sources", lambda path: sources()[:2])
    monkeypatch.setattr(app, "_require_minigrid", lambda: None)


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        bind_inputs(patch)

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
        original = app.run_adaptation

        def run(path):
            result = original(path)
            CAPTURE["result"] = result
            return result

        patch.setattr(app, "run_adaptation", run)
        destination = tmp_path_factory.mktemp("a4-adaptation") / "output.json"
        patch.setattr(
            sys,
            "argv",
            [
                "affine",
                "--context-linear-a4-adaptation",
                "--source",
                "literal",
                "--output",
                str(destination),
            ],
        )
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            cli.main()
        return CAPTURE["result"], destination


def test_should_execute_actual_cli_six_delayed_fits_and_three_scripted_layout_phases(integrated):
    result, destination = integrated
    assert result["status"] == "complete", result.get("reason")
    assert destination.is_file() and destination.stat().st_size < 8 * 1024**2
    training = result["training"]
    assert training["catalogue"]["committed_rows"] == dict(arrivals=54, replay=36, native=90)
    assert training["observed_checkpoints"] == [0, 2, 4, 6]
    assert len(training["committed_receipts"]) == training["budget"]["updates_completed"] == 6
    for i, receipt in enumerate(training["committed_receipts"]):
        assert (
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
            receipt["update_number"],
        ) == (i, i + 1, i + 1, i + 1)
    assert [phase["updates"] for phase in result["games"]] == [2, 4, 6]
    assert all(len(phase["games"]["episodes"]) == 4 for phase in result["games"])
    assert result["readout_budget"]["started_at"] == result["readout_budget"]["observed_started_at"]
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    assert result["summary"]["complete"] is True


def test_should_bind_causal_joint_rows_and_distinct_a3_a4_identities(integrated):
    training = integrated[0]["training"]
    wakes = training["catalogue"]["wakes"]
    assert [w["current_rows"] + w["replay_rows"] for w in wakes] == [9, 9, 18, 18, 18, 18]
    assert [w["current_episode_id"] for w in wakes] == [
        f"{sources()[index].episode_id}:wake:{tick}"
        for tick, index in enumerate((0, 1, 2, 3, 0, 1))
    ]
    assert [row["state"]["training_examples"] for row in training["endpoints"]] == [0, 18, 54, 90]
    assert training["stream"]["evaluation_scope"] == "training_only"
    assert training["protocol"]["definition"] == cli.describe_context_linear_a4()["definition"]


@pytest.mark.parametrize(
    "fault", ["flag", "retention", "abc", "recurring", "order", "role", "context", "rows"]
)
def test_should_refuse_wrong_a4_protocol_before_fitting(monkeypatch, fault):
    packets = sources()
    options: dict[str, Any] = dict(a4_training=True, retain_arrived=True)
    if fault == "flag":
        options["a4_training"] = 1
    elif fault == "retention":
        options["retain_arrived"] = False
    elif fault == "abc":
        options["abc_catalogue"] = True
    elif fault == "recurring":
        options["recurring_catalogue"] = True
    elif fault == "order":
        packets = tuple(reversed(packets))
    elif fault == "role":
        packets = (replace(packets[0], role="probe_test"), *packets[1:])
    elif fault == "context":
        packets = (replace(packets[0], context="B"), *packets[1:])
    else:
        packets = (episode(packets[0].episode_id, rows=8, context="A"), *packets[1:])
    monkeypatch.setattr(ContextLinearLearner, "train_batch", lambda *a: pytest.fail("fit reached"))
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(packets, **options)


@pytest.mark.parametrize("fault", ["role", "layout", "label", "clock", "incomplete"])
def test_should_refuse_corrupted_from_birth_train_evidence(fault):
    record = literal_record()
    if fault == "role":
        record["collection"][0]["role"] = "probe_test"
    elif fault == "layout":
        record["collection"][0]["layout"]["branch_row"] = 3
    elif fault == "label":
        record["episodes"][0]["transitions"][0]["action"] = 4
    elif fault == "clock":
        record["budget"]["observed_started_at"] = 2
    else:
        record["status"] = "failed"
    with pytest.raises(ValueError):
        app._admit_a4_record(record)


def test_should_admit_exact_native_train_records_and_refuse_replaced_bytes(tmp_path, monkeypatch):
    path = tmp_path / "train.json"
    raw = json.dumps(literal_record()).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(app, "SOURCE_BYTES", len(raw))
    monkeypatch.setattr(app, "SOURCE_SHA", sha256(raw).hexdigest())
    packets, provenance = app.read_a4_training(path)
    assert packets == sources()[2:] and provenance["source_bytes"] == len(raw)
    path.write_bytes(raw.replace(b'"complete"', b'"replaced"', 1))
    with pytest.raises(ValueError, match="exact"):
        app.read_a4_training(path)


@pytest.mark.parametrize("fault", ["flag", "shape", "conflict", "clock"])
def test_should_refuse_wrong_stage_or_readout_authority_before_games(
    integrated, monkeypatch, fault
):
    row = integrated[0]["training"]["endpoints"][2]
    state = cli._context_linear_training_snapshot(row["state"])
    options: dict[str, Any] = dict(a4_training=True)
    session = budget()
    if fault == "flag":
        options["a4_training"] = 1
    elif fault == "shape":
        state = replace(state, training_examples=90)
    elif fault == "conflict":
        options["retained_catalogue"] = True
    else:
        session.budget = replace(session.budget, max_wall_seconds=151)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("game reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(), state, budget=session, check_boundary=lambda: None, **options
        )


@pytest.mark.parametrize("fault", ["training", "prior_phase", "budget", "phase_role"])
def test_should_refuse_changed_paid_prefix_without_admitting_more_work(
    integrated, monkeypatch, fault
):
    bind_inputs(monkeypatch)
    training = deepcopy(integrated[0]["training"])
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: training)
    seen: list[dict[str, Any]] = []

    def games(*args, budget, check_boundary, **options):
        if fault == "training":
            training["committed_receipts"][0]["update_number"] = 99
        if fault == "budget":
            budget.started_at += 1
        if fault == "prior_phase" and seen:
            seen[0]["episodes"][0]["success"] = False
        check_boundary()
        observed = deepcopy(integrated[0]["games"][len(seen)]["games"])
        if fault == "phase_role":
            observed["episodes"][0]["source_role"] = "probe_test"
        seen.append(observed)
        return observed

    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", games)
    result = app.run_adaptation("literal")
    assert result["status"] == "failed" and result["summary"]["complete"] is False
    assert len(seen) <= 1


def test_should_keep_partial_metrics_unknown_and_distinguish_recovery_from_retention():
    def phase(n, outcomes):
        return dict(
            updates=n,
            games=dict(
                episodes=[
                    dict(status="complete", success=v, environment=dict(closed=True))
                    for v in outcomes
                ]
            ),
        )

    phases = [
        phase(2, [True, True, False, False]),
        phase(4, [False, True, True, True]),
        phase(6, [True, True, True, True]),
    ]
    summary = app.summarize_games(phases)
    assert summary["acquisition_gain"] == 1 and summary["old_retention_delta"] == -0.5
    assert summary["return_gain"] == 0.5 and summary["learning_recovery"] == [True, False]
    assert app.summarize_games(phases[:2])["acquisition_gain"] is None


def test_should_describe_without_io_and_refuse_exclusive_or_irrelevant_cli(
    integrated, monkeypatch, capsys
):
    monkeypatch.setattr(app, "run_adaptation", lambda *a: pytest.fail("work reached"))
    monkeypatch.setattr(sys, "argv", ["affine", "--context-linear-a4-adaptation", "--describe"])
    cli.main()
    assert "A3" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affine",
            "--context-linear-a4-adaptation",
            "--source",
            "literal",
            "--output",
            str(integrated[1]),
        ],
    )
    with pytest.raises(FileExistsError):
        cli.main()
    monkeypatch.setattr(
        sys,
        "argv",
        ["affine", "--context-linear-a4-adaptation", "--source", "literal", "--fit-updates", "0"],
    )
    with pytest.raises(SystemExit):
        cli.main()
