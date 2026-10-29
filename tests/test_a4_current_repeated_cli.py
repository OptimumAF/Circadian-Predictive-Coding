"""Real A4 command/stream/evaluator/loops on literal ports, no native learning."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import json
import os
import pickle
import random
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from tests import test_a4_current_repeated as literal
from test_a4_adaptation import bind_inputs
from test_context_linear_causal_games import PhaseEnvironment, budget
from test_minigrid_learning_pilot import Environment

FLAG = "--context-linear-a4-current-repeated"
WORK = {key: 0 for key in literal.WORK}
WORK.update(games=0, steps=0)


def bind_case(patch):
    patch.setattr(literal, "WORK", WORK)
    box = literal.bind_ports(patch)
    bind_inputs(patch)

    class LiteralGame(PhaseEnvironment):
        def __init__(self, *args, **options):
            WORK["games"] += 1
            assert WORK["games"] <= 120
            super().__init__(*args, **options)

        def step(self, action):
            WORK["steps"] += 1
            assert WORK["steps"] <= 256
            return super().step(action)

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

    patch.setattr(Environment, "work", box)
    patch.setattr(evaluation, "MiniGridKeyDoor", LiteralGame)
    return box


@pytest.fixture
def case(monkeypatch):
    return bind_case(monkeypatch)


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        box = bind_case(patch)
        original = app.run_adaptation

        def run(path, **options):
            assert options == dict(current_repeated=True)
            result = original(path, **options)
            box["raw"] = result
            return result

        patch.setattr(app, "run_adaptation", run)
        output = tmp_path_factory.mktemp("repeated-cli") / "result.json"
        patch.setattr(sys, "argv", ["affine", FLAG, "--source", "literal", "--output", str(output)])
        rng = pickle.dumps((random.getstate(), np.random.get_state()))
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            try:
                cli.main()
            except SystemExit as error:
                pytest.fail(f"CLI exit{error.code}: {box.get('raw', {}).get('reason')}")
        assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
        return json.loads(output.read_text()), box, output


def test_should_run_actual_command_through_six_fits_and_twelve_closed_games(complete):
    result, box, output = complete
    assert result["status"] == "complete", result.get("reason")
    assert result["definition"] == "minigrid_a3_a4_a3_current_repeated_and_games_v1"
    assert [len(x) for x, _ in box["fits"]] == [9, 9, 18, 18, 18, 18]
    assert result["training"]["current_repetition"]["committed_rows"] == dict(
        arrivals=54, repeated_current=36, native=90
    )
    assert [phase["updates"] for phase in result["games"]] == [2, 4, 6]
    assert [phase["games"]["source_fit_rows"] for phase in result["games"]] == [18, 54, 90]
    assert all(
        phase["games"]["role"] == "knownTRAIN_A3A4_current_repeated" for phase in result["games"]
    )
    assert len(box["environments"]) == 12 and all(e.closed for e in box["environments"])
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    assert result["readout_budget"]["started_at"] == result["readout_budget"]["observed_started_at"]
    assert 0 < output.stat().st_size < 8 * 1024**2


def test_should_keep_fixed_readouts_and_unchanged_metric_definitions(complete):
    result = complete[0]
    for phase in result["games"]:
        assert [
            (r["environment"]["layout"]["branch_row"], r["cue"], r["source_role"])
            for r in phase["games"]["episodes"]
        ] == [(row, cue, "train") for row in (3, 4) for cue in ("blue", "red")]
        assert all(
            r["environment"]["seed"] == 17 and r["environment"]["max_steps"] == 16
            for r in phase["games"]["episodes"]
        )
    assert result["summary"]["complete"] is True
    assert result["summary"]["acquisition_gain"] == 0.0
    assert result["summary"]["learning_recovery"] == [False, False]


def saved_state(complete, updates=4):
    row = next(r for r in complete[1]["raw"]["training"]["endpoints"] if r["updates"] == updates)
    return cli._context_linear_training_snapshot(row["state"])


@pytest.mark.parametrize("updates", [2, 4, 6])
def test_should_dispatch_distinct_readout_binding_without_new_fits(
    complete, case, monkeypatch, updates
):
    saved = saved_state(complete, updates)

    def dispatch(owner, state, **options):
        assert state == saved and state is not saved
        assert options["role"] == "knownTRAIN_A3A4_current_repeated"
        assert options["return_metadata"]["source_fit_rows"] == {2: 18, 4: 54, 6: 90}[updates]
        assert options["memory_copier"](state) == saved
        options["extra_check"]()
        return dict(status="literal_dispatch")

    monkeypatch.setattr(evaluation, "_run_checkpoint", dispatch)
    result = evaluation.evaluate_context_linear_stream_checkpoint(
        ContextLinearLearner(),
        saved,
        budget=budget(),
        check_boundary=lambda: None,
        a4_current_repeated=True,
    )
    assert result["status"] == "literal_dispatch" and not case["fits"]


@pytest.mark.parametrize(
    "fault",
    ["flag", "rows", "head_rows", "failure", "budget", "joint", "current", "reference", "retained"],
)
def test_should_refuse_wrong_binding_before_games(complete, case, monkeypatch, fault):
    state = saved_state(complete)
    options: dict[str, Any] = dict(a4_current_repeated=True)
    session = budget()
    if fault == "flag":
        options["a4_current_repeated"] = 1
    elif fault == "rows":
        state = replace(state, training_examples=36)
    elif fault == "head_rows":
        state = replace(
            state,
            heads=tuple(replace(h, fit_rows=9, singular_values=(0.0,) * 9) for h in state.heads),
        )
    elif fault == "failure":
        state = replace(state, failed_work=(1, 9, 0, 0, True))
    elif fault == "budget":
        session.budget = replace(session.budget, max_wall_seconds=151)
    else:
        options[
            {
                "joint": "a4_training",
                "current": "a4_current_only",
                "reference": "reference_window",
                "retained": "retained_catalogue",
            }[fault]
        ] = True
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("game reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_stream_checkpoint(
            ContextLinearLearner(), state, budget=session, check_boundary=lambda: None, **options
        )
    assert not case["fits"]


@pytest.mark.parametrize(
    "field",
    [
        "arrival_row_exposures",
        "repeated_current_row_exposures",
        "prior_packet_replay_rows",
        "retained_bank_rows",
        "verified_receipts",
        "protocol",
    ],
)
def test_should_refuse_unqualified_repeat_costs_before_games(complete, case, monkeypatch, field):
    training = deepcopy(complete[1]["raw"]["training"])
    if field == "protocol":
        training["protocol"]["definition"] = "changed"
    elif field == "verified_receipts":
        training["current_repetition"][field] = []
    else:
        training["current_repetition"][field] = False
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: training)
    monkeypatch.setattr(
        app,
        "evaluate_context_linear_stream_checkpoint",
        lambda *a, **k: pytest.fail("game reached"),
    )
    result = app.run_adaptation("literal", current_repeated=True)
    assert result["status"] == "failed" and result["games"] == []
    assert result["training"]["budget"]["updates_completed"] == 6 and not case["fits"]


@pytest.mark.parametrize("fault", ["training", "prior_phase", "phase_replaced", "clock", "role"])
def test_should_preserve_paid_prefix_and_refuse_drift_before_more_games(
    complete, case, monkeypatch, fault
):
    training = deepcopy(complete[1]["raw"]["training"])
    monkeypatch.setattr(cli, "run_context_linear_stream", lambda *a, **k: training)
    seen: list[Any] = []
    original_paid = app.paid_report_guard
    original_phase: Any = None

    def paid(value):
        nonlocal original_phase
        if "updates" in value and "games" in value:
            original_phase = value
        return original_paid(value)

    def games(*args, budget, check_boundary, **options):
        assert options == dict(a4_current_repeated=True)
        if fault == "training":
            training["committed_receipts"][0]["update_number"] = 99
        if fault == "clock":
            budget.started_at += 1
        if seen and fault == "prior_phase":
            original_phase["games"]["episodes"][0]["success"] = False
        if seen and fault == "phase_replaced":
            original_phase["games"] = deepcopy(original_phase["games"])
        check_boundary()
        observed = deepcopy(complete[1]["raw"]["games"][len(seen)]["games"])
        if fault == "role":
            observed["episodes"][0]["source_role"] = "probe_test"
        seen.append(observed)
        return observed

    monkeypatch.setattr(app, "paid_report_guard", paid)
    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", games)
    result = app.run_adaptation("literal", current_repeated=True)
    assert result["status"] == "failed" and result["summary"]["complete"] is False
    assert len(result["games"]) <= 1 and len(seen) <= 1
    assert result["training"]["budget"]["updates_completed"] == 6 and not case["fits"]


def test_should_keep_first_phase_and_close_partial_second_phase(complete, case, monkeypatch):
    monkeypatch.setattr(
        cli, "run_context_linear_stream", lambda *a, **k: deepcopy(complete[1]["raw"]["training"])
    )

    def fail(environment):
        if len(case["environments"]) == 5:
            case["step_error"] = True

    case["on_environment"] = fail
    result = app.run_adaptation("literal", current_repeated=True)
    assert result["status"] == "failed" and len(result["games"]) == 1
    assert result["failed_phase"]["updates"] == 4
    assert all(e.closed for e in case["environments"])
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        ["--context-linear-a4-current-only"],
        ["--context-linear-a4-adaptation"],
        ["--fit-updates", "6"],
        ["--describe", "--source", "literal"],
        [],
    ],
)
def test_should_refuse_unfrozen_cli_before_work(case, monkeypatch, extra):
    monkeypatch.setattr(app, "run_adaptation", lambda *a, **k: pytest.fail("work reached"))
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, *extra])
    with pytest.raises(SystemExit):
        cli.main()
    assert not case["fits"]


def test_should_describe_and_refuse_existing_output_before_work(
    complete, case, monkeypatch, capsys
):
    monkeypatch.setattr(app, "run_adaptation", lambda *a, **k: pytest.fail("work reached"))
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["training"]["repeated_current_row_exposures"] == 36
    before = complete[2].read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["affine", FLAG, "--source", "literal", "--output", str(complete[2])]
    )
    with pytest.raises(FileExistsError):
        cli.main()
    assert complete[2].read_bytes() == before and not case["fits"]


@pytest.mark.parametrize(
    "options",
    [
        dict(current_only=1),
        dict(current_repeated=1),
        dict(current_only=True, current_repeated=True),
    ],
)
def test_should_refuse_bad_public_mode_before_budgets(case, options):
    with pytest.raises(ValueError):
        app.run_adaptation("literal", **options)
    assert not case["fits"]


@pytest.mark.parametrize("current", [False, True])
def test_should_preserve_original_descriptor_and_dispatch(case, monkeypatch, capsys, current):
    called: list[Any] = []

    def run(path, **options):
        called.append(options)
        return dict(status="complete")

    monkeypatch.setattr(app, "run_adaptation", run)
    flag = "--context-linear-a4-current-only" if current else "--context-linear-a4-adaptation"
    monkeypatch.setattr(sys, "argv", ["affine", flag, "--describe"])
    cli.main()
    assert json.loads(capsys.readouterr().out)["training"]["fit_rows"] == (54 if current else 90)
    monkeypatch.setattr(sys, "argv", ["affine", flag, "--source", "literal"])
    cli.main()
    assert called == [dict(current_only=True) if current else {}] and not case["fits"]
