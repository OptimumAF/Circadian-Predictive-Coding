"""One6fit54row literal control capture; accepted joint numerics are not rerun."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import os
import sys
from typing import Any

import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from test_a4_adaptation import bind_inputs, sources
from test_arrived_cue_catalogue import episode
from test_context_linear_causal_games import PhaseEnvironment, budget, state

CAPTURE: dict[str, Any] = {}


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
        original_run, original_fit = app.run_adaptation, ContextLinearLearner.train_batch
        rows: list[tuple[Any, Any]] = []

        def fit(owner, x, y):
            rows.append((x.copy(), y.copy()))
            return original_fit(owner, x, y)

        def run(path, **options):
            assert options == dict(current_only=True)
            result = original_run(path, **options)
            CAPTURE["result"] = result
            return result

        patch.setattr(ContextLinearLearner, "train_batch", fit)
        patch.setattr(app, "run_adaptation", run)
        output = tmp_path_factory.mktemp("a4-current-only") / "output.json"
        patch.setattr(
            sys,
            "argv",
            [
                "affine",
                "--context-linear-a4-current-only",
                "--source",
                "literal",
                "--output",
                str(output),
            ],
        )
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            cli.main()
        CAPTURE["rows"] = rows
        return CAPTURE["result"], output, rows


def test_should_execute_actual_cli_six_current_only_updates_and_all_twelve_scripted_games(
    integrated,
):
    result, output, rows = integrated
    assert result["status"] == "complete", result.get("reason")
    training = result["training"]
    assert "catalogue" not in training
    assert training["stream"]["training_examples"] == 54
    assert len(training["committed_receipts"]) == training["budget"]["updates_completed"] == 6
    assert [r["state"]["training_examples"] for r in training["endpoints"]] == [0, 18, 36, 54]
    assert [phase["updates"] for phase in result["games"]] == [2, 4, 6]
    assert all(len(phase["games"]["episodes"]) == 4 for phase in result["games"])
    assert output.is_file() and output.stat().st_size < 8 * 1024**2
    assert result["readout_budget"]["started_at"] == result["readout_budget"]["observed_started_at"]
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    assert len(rows) == 6


def test_should_fit_only_the_current_arrived_episode_with_delay_one_and_no_joint_rows(integrated):
    training, rows = integrated[0]["training"], integrated[2]
    for tick, (index, (x, y), receipt) in enumerate(
        zip((0, 1, 2, 3, 0, 1), rows, training["committed_receipts"], strict=True)
    ):
        packet = sources()[index]
        assert x.tolist() == [list(t.observation) for t in packet.transitions]
        assert y[:, 0].tolist() == [t.action for t in packet.transitions]
        assert len(x) == 9
        assert (
            receipt["observed_at"],
            receipt["arrived_at"],
            receipt["applied_at"],
            receipt["update_number"],
        ) == (tick, tick + 1, tick + 1, tick + 1)
    assert training["protocol"]["catalogue_replay_row_exposures"] == 0
    assert training["protocol"]["fit_rows"] == 54
    assert training["stream"]["evaluation_scope"] == "training_only"


def test_should_preserve_joint_default_and_fixed_shared_data_game_protocol():
    joint, control = app.describe_adaptation(), app.describe_adaptation(current_only=True)
    assert joint["definition"] == "minigrid_a3_a4_a3_adaptation_and_games_v1"
    assert (
        joint["training"]["fit_rows"] == 90
        and joint["training"]["catalogue_replay_row_exposures"] == 36
    )
    assert (
        control["training"]["fit_rows"] == 54
        and control["training"]["catalogue_replay_row_exposures"] == 0
    )
    assert joint["source_sha256"] == control["source_sha256"]
    assert joint["native_plan"] == control["native_plan"]
    assert (
        joint["training"]["checkpoint_updates"]
        == control["training"]["checkpoint_updates"]
        == (0, 2, 4, 6)
    )
    with pytest.raises(ValueError):
        app.describe_adaptation(current_only=1)


@pytest.mark.parametrize("updates", [2, 4, 6])
def test_should_preserve_joint_typed_plan_without_another_joint_fit(monkeypatch, updates):
    original = state(updates)
    head_rows = 9 if updates == 2 else 18
    saved = replace(
        original,
        training_examples={2: 18, 4: 54, 6: 90}[updates],
        heads=tuple(
            replace(h, fit_rows=head_rows, singular_values=(0.0,) * head_rows)
            for h in original.heads
        ),
    )

    def inspect(owner, current, **options):
        assert options["role"] == "knownTRAIN_A3A4_adaptation"
        assert current == saved and options["updates"] == updates
        assert [(layout.branch_row, cue, role) for layout, cue, role in options["live_plan"]] == [
            (row, cue, "train") for row in (3, 4) for cue in ("blue", "red")
        ]
        options["extra_check"]()
        return dict(status="complete")

    monkeypatch.setattr(evaluation, "_run_checkpoint", inspect)
    assert (
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(),
            saved,
            budget=budget(),
            check_boundary=lambda: None,
            a4_training=True,
        )["status"]
        == "complete"
    )


@pytest.mark.parametrize(
    "fault", ["flag", "retention", "joint", "abc", "recurring", "order", "role", "context", "rows"]
)
def test_should_refuse_invalid_current_only_source_or_conflict_before_fitting(monkeypatch, fault):
    packets = sources()
    options: dict[str, Any] = dict(a4_current_only=True)
    if fault == "flag":
        options["a4_current_only"] = 1
    elif fault == "retention":
        options["retain_arrived"] = True
    elif fault == "joint":
        options["a4_training"] = True
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


def test_should_keep_original_joint_retention_required_contract(monkeypatch):
    monkeypatch.setattr(ContextLinearLearner, "train_batch", lambda *a: pytest.fail("fit reached"))
    with pytest.raises(ValueError, match="requires retention"):
        cli.run_context_linear_stream(sources(), a4_training=True)


@pytest.mark.parametrize("fault", ["flag", "shape", "conflict", "clock"])
def test_should_refuse_wrong_control_state_or_readout_authority_before_games(
    integrated, monkeypatch, fault
):
    saved = cli._context_linear_training_snapshot(
        integrated[0]["training"]["endpoints"][2]["state"]
    )
    options: dict[str, Any] = dict(a4_current_only=True)
    session = budget()
    if fault == "flag":
        options["a4_current_only"] = 1
    elif fault == "shape":
        saved = replace(saved, training_examples=54)
    elif fault == "conflict":
        options["a4_training"] = True
    else:
        session.budget = replace(session.budget, max_wall_seconds=151)
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("game reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(), saved, budget=session, check_boundary=lambda: None, **options
        )


@pytest.mark.parametrize("fault", ["training", "prior_phase", "budget", "phase_role"])
def test_should_preserve_paid_control_prefix_and_refuse_drift_before_more_work(
    integrated, monkeypatch, fault
):
    bind_inputs(monkeypatch)
    training = deepcopy(integrated[0]["training"])
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: training)
    seen: list[dict[str, Any]] = []

    def games(*args, budget, check_boundary, **options):
        assert options == dict(a4_current_only=True)
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
    result = app.run_adaptation("literal", current_only=True)
    assert result["status"] == "failed" and result["summary"]["complete"] is False
    assert len(seen) <= 1


def test_should_keep_recovery_and_unknown_metric_definitions_identical_to_joint():
    def phase(n, values):
        return dict(
            updates=n,
            games=dict(
                episodes=[
                    dict(status="complete", success=v, environment=dict(closed=True))
                    for v in values
                ]
            ),
        )

    phases = [
        phase(2, [True, True, False, False]),
        phase(4, [False, True, True, True]),
        phase(6, [True, True, False, False]),
    ]
    result = app.summarize_games(phases)
    assert result["acquisition_gain"] == 1 and result["old_retention_delta"] == -0.5
    assert result["return_gain"] == 0.5 and result["learning_recovery"] == [True, False]
    assert app.summarize_games(phases[:2])["acquisition_gain"] is None


def test_should_describe_without_io_and_refuse_output_replacement_or_conflicting_cli(
    integrated, monkeypatch, capsys
):
    monkeypatch.setattr(app, "run_adaptation", lambda *a, **k: pytest.fail("work reached"))
    monkeypatch.setattr(sys, "argv", ["affine", "--context-linear-a4-current-only", "--describe"])
    cli.main()
    assert "current_only" in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "affine",
            "--context-linear-a4-current-only",
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
        [
            "affine",
            "--context-linear-a4-current-only",
            "--context-linear-a4-adaptation",
            "--source",
            "literal",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()
