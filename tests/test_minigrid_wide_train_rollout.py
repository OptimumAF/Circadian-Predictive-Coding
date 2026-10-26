"""H32 TRAIN games compose with the existing paid, isolated fit stream.

One public CLI case performs 128 BP calls / 6912 rows, twelve fixed scoring
calls / 108 rows and two actual A games (at most 32 predictions / steps).
Other cases use owned fake learners and evaluator receipts: no native updates,
environment resets, physics, teacher, replay, probe or final work.
"""

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import random
import sys

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_constructive_fit as fit
from src.adapters import minigrid_train_rollout as live
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed
from src.shared.torch_runtime import require_torch
from test_minigrid_constructive_fit import assert_raw_scores, capture_stream
from test_minigrid_learning_pilot import Learner
from test_minigrid_train_fit_bounds import histories, raw_sources
from test_torch_feature_learner import assert_tree
from test_torch_feature_rehearsal_pilot import forbidden


METHODS = ("bp", "pc", "neutral_cpc", "gated_cpc")


@pytest.fixture
def no_resources(monkeypatch):
    for name in ("_learner", "AttributionStream", "ProcessRssSampler", "run_train_rollout"):
        monkeypatch.setattr(fit, name, forbidden)


@pytest.mark.parametrize("method", METHODS)
def test_should_describe_opt_in_known_train_games_without_changing_default(no_resources, method):
    original = fit.describe_protocol(wide_method=method, updates=128)
    assert original == fit.describe_protocol(wide_method=method, updates=128, train_a_rollout=False)
    assert "train_a_rollout" not in original
    selected = fit.describe_protocol(wide_method=method, updates=128, train_a_rollout=True)
    rollout = selected.pop("train_a_rollout")
    for key in ("checkpoint_updates", "training_rows", "score_rows", "score_forward_calls", "seed"):
        assert selected[key] == original[key]
    assert rollout["checkpoint_updates"] == 128
    assert rollout["checkpoint_training_rows"] == 6912
    assert rollout["context"] == "A" and rollout["cues"] == ["blue", "red"]
    assert rollout["role"] == "train"
    assert rollout["learner_live_episodes"] == 2
    assert rollout["maximum_live_prediction_forwards"] == 32
    assert rollout["maximum_live_environment_steps"] == 32
    assert rollout["max_wall_seconds"] == 20
    assert selected["probe_updates"] == selected["final_episodes"] == selected["replay_rows"] == 0


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_nonboolean_rollout_before_resources(no_resources, flag):
    options = dict(wide_method="bp", updates=128, train_a_rollout=flag)
    with pytest.raises(ValueError):
        fit.describe_protocol(**options)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), **options)


@pytest.mark.parametrize(
    "options",
    [
        {},
        dict(perceptron=True),
        dict(kernel_perceptron=True),
        dict(cue_relative=True),
        dict(cue_turn_gradients=True),
        dict(wide_method="bp", updates=1),
        dict(wide_method="pc", updates=127),
        dict(wide_method="neutral_cpc", updates=129),
        dict(wide_method="gated_cpc", updates=512),
    ],
)
def test_should_refuse_other_families_or_clocks_before_resources(no_resources, options):
    options = dict(updates=128) | options | dict(train_a_rollout=True)
    with pytest.raises(ValueError):
        fit.describe_protocol(**options)
    with pytest.raises(ValueError):
        fit.diagnose_train_fit(histories(), **options)


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--constructive-bp", "--fit-updates", "128"],
        ["--perceptron", "--fit-updates", "128"],
        ["--kernel-perceptron", "--fit-updates", "128"],
        ["--bounds-only"],
        ["--wide-neural", "bp"],
        ["--wide-neural", "bp", "--fit-updates", "129"],
        ["--wide-neural", "bp", "--fit-updates", "128", "--cue-turn-gradients"],
        ["--wide-neural", "bp", "--fit-updates", "128", "--cue-relative"],
    ],
)
def test_should_refuse_incompatible_public_options_before_source_io(monkeypatch, options):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(cli, "diagnose_train_fit", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["fit", *options, "--train-a-rollout", "--source", "absent.json"]
    )
    with pytest.raises((ValueError, SystemExit)) as caught:
        cli.main()
    if isinstance(caught.value, SystemExit):
        assert caught.value.code == 2


@pytest.mark.parametrize("method", METHODS)
def test_should_preview_selected_public_recipe_before_any_source_io(monkeypatch, capsys, method):
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    monkeypatch.setattr(fit, "_learner", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit", "--wide-neural", method, "--fit-updates", "128", "--train-a-rollout", "--describe"],
    )
    cli.main()
    assert json.loads(capsys.readouterr().out) == fit.describe_protocol(
        wide_method=method, updates=128, train_a_rollout=True
    )


def paid_report(status="complete"):
    return dict(
        status=status,
        reason=None if status == "complete" else "injected partial live failure",
        episodes=[dict(context="A", cue=cue, source_role="train") for cue in ("blue", "red")],
        work=dict(
            environment_resets_completed=2,
            live_prediction_calls_returned=3,
            environment_steps_returned=3,
            failure_work_may_be_unobserved=False,
        ),
    )


@pytest.fixture
def fake_fit(monkeypatch):
    captured = capture_stream(monkeypatch)
    monkeypatch.setattr(Learner, "work", {})
    monkeypatch.setattr(fit, "InvariantTorchFeatureLearner", lambda policy: Learner(policy))
    monkeypatch.setattr(fit, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    clock = [1.0]
    monkeypatch.setattr(fit, "monotonic", lambda: clock[0])
    monkeypatch.setattr(live, "monotonic", lambda: clock[0])

    def counter(state, point, *, wide_method=None, **unused):
        assert state.policy == fit._wide_policy(wide_method)
        assert state.completed_updates == point and state.training_examples == point * 54

    monkeypatch.setattr(fit, "_counter", counter)
    return captured, clock


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("failed", [False, True])
def test_should_run_existing_helper_after_all_fixed_readouts_and_keep_paid_prefix(
    monkeypatch, fake_fit, method, failed
):
    captured, _ = fake_fit
    calls = []
    paid = paid_report("failed" if failed else "complete")

    def evaluator(source, state, **options):
        assert source is captured["source"]
        assert captured["stream"]._finished is True
        assert state.completed_updates == 128 and state.training_examples == 6912
        assert len(Learner.work["predicts"]) == 12
        assert options["wide_method"] == method and "catalogue_method" not in options
        assert options["updates"] == 128 and options["known_a_only"] is True
        budget = options["budget"]
        assert budget.budget.max_training_updates == budget.budget.max_replay_examples == 0
        assert budget.budget.max_wall_seconds == 20
        assert captured["options"]["budget"].updates_completed == 128
        options["check_boundary"]()
        calls.append(options)
        if failed:
            raise MiniGridEvaluationFailed(paid)
        return paid

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluator)
    result = fit.diagnose_train_fit(
        histories(), wide_method=method, updates=128, train_a_rollout=True
    )
    assert result["status"] == ("incomplete" if failed else "complete"), result["reason"]
    assert len(calls) == 1
    assert (
        result["charged_updates"]
        == result["committed_updates"]
        == result["native_completed_updates"]
        == 128
    )
    assert result["native_training_rows"] == 6912 and len(result["applied_updates"]) == 128
    assert [record["updates"] for record in result["readouts"]] == [0, 128]
    receipt = result["train_a_rollout"]
    assert receipt["status"] == ("failed" if failed else "complete")
    assert receipt["report"] == paid and receipt["report"] is not paid
    assert receipt["charged_updates"] == receipt["replay_examples"] == 0
    work = result["work"]
    assert work["score_forward_calls_completed"] == 12 and work["score_rows_completed"] == 108
    assert work["live_prediction_forwards_completed"] == 3
    assert work["total_prediction_forwards_completed"] == 15
    assert work["total_prediction_rows_completed"] == 111
    assert work["source_environment_steps_returned"] == 0
    assert work["total_environment_steps_returned"] == 3
    assert work["training_work_may_be_unobserved"] is False
    paid["work"]["live_prediction_calls_returned"] = 999
    assert receipt["report"]["work"]["live_prediction_calls_returned"] == 3


@pytest.mark.parametrize(
    "fault",
    ["wall", "source_model", "actor_model", "candidate_model", "source_input", "parent_budget"],
)
def test_should_retain_live_evidence_when_parent_closing_guard_refuses(
    monkeypatch, fake_fit, fault
):
    captured, clock = fake_fit
    sources = histories()

    def evaluator(*args, **options):
        options["check_boundary"]()
        if fault == "wall":
            clock[0] = 1000.0
        elif fault == "source_model":
            captured["source"].state.parameters["weight"][0, 0] = 999
        elif fault == "actor_model":
            captured["stream"]._runtime.actor._learner.state.parameters["weight"][0, 0] = 999
        elif fault == "candidate_model":
            captured["stream"]._runtime._candidate.state.parameters["weight"][0, 0] = 999
        elif fault == "source_input":
            object.__setattr__(sources[0], "context", "foreign")
        else:
            captured["options"]["budget"].started_at = 0.0
        return paid_report()

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluator)
    result = fit.diagnose_train_fit(sources, wide_method="bp", updates=128, train_a_rollout=True)
    assert result["status"] == "incomplete" and result["reason"]
    assert (
        result["native_completed_updates"]
        == result["committed_updates"]
        == result["charged_updates"]
        == 128
    )
    assert len(result["applied_updates"]) == 128
    assert result["train_a_rollout"]["report"]["work"]["environment_steps_returned"] == 3
    assert result["work"]["total_environment_steps_returned"] == 3
    assert result["work"]["training_work_may_be_unobserved"] is False


def test_should_keep_live_budget_binding_after_helper_returns(monkeypatch, fake_fit):
    captured_budget = []
    helper = fit.run_train_rollout

    def evaluator(*args, **options):
        captured_budget.append(options["budget"])
        return paid_report()

    def replace_clock_after_return(*args, **options):
        record, binding = helper(*args, **options)
        captured_budget[0].started_at = 0.0
        return record, binding

    monkeypatch.setattr(live, "evaluate_neural_representation_checkpoint", evaluator)
    monkeypatch.setattr(fit, "run_train_rollout", replace_clock_after_return)
    result = fit.diagnose_train_fit(
        histories(), wide_method="bp", updates=128, train_a_rollout=True
    )
    assert result["status"] == "incomplete" and "binding" in result["reason"]
    assert result["charged_updates"] == result["native_completed_updates"] == 128
    assert result["train_a_rollout"]["report"]["work"]["environment_steps_returned"] == 3


def test_should_mark_unobserved_live_work_when_helper_return_is_lost(monkeypatch, fake_fit):
    def lost(*args, **options):
        raise RuntimeError("injected lost live helper return")

    monkeypatch.setattr(fit, "run_train_rollout", lost)
    result = fit.diagnose_train_fit(
        histories(), wide_method="bp", updates=128, train_a_rollout=True
    )
    assert result["status"] == "incomplete" and "lost live helper" in result["reason"]
    assert result["work"]["live_work_may_be_unobserved"] is True
    assert result["charged_updates"] == result["native_completed_updates"] == 128


def test_should_execute_public_bp128_train_fit_then_two_target_free_games(
    monkeypatch, tmp_path, capsys
):
    torch = require_torch()
    source_path, output_path = tmp_path / "sources.json", tmp_path / "fit.json"
    source_path.write_text(
        json.dumps(
            dict(
                training_sources=[asdict(e) for e in raw_sources()],
                probe_data="unavailable",
                final_test="unavailable",
            )
        )
    )
    before = source_path.read_bytes()
    rng = (random.getstate(), pickle.dumps(np.random.get_state()), torch.get_rng_state().clone())
    captured = {}
    original = fit.AttributionStream

    def stream(source, training, heldout, **options):
        captured.update(
            source=source,
            initial=deepcopy(source.snapshot_state()),
            heldout=heldout,
            options=options,
        )
        result = original(source, training, heldout, **options)
        captured["stream"] = result
        return result

    monkeypatch.setattr(fit, "AttributionStream", stream)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--wide-neural",
            "bp",
            "--train-a-rollout",
            "--fit-updates",
            "128",
            "--fit-seconds",
            "90",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
        ],
    )
    cli.main()
    result = json.loads(capsys.readouterr().out)
    label = os.environ.get("WIDE_CHECK_LABEL")
    destination = (
        Path("artifacts/runs/wide-neural-train-rollout-20261018") / f"{label}-public-bp.json"
        if label
        else tmp_path / "public-bp.json"
    )
    with destination.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    assert result["status"] == "complete", result["reason"]
    assert (
        result["native_completed_updates"]
        == result["committed_updates"]
        == result["charged_updates"]
        == 128
    )
    assert result["native_training_rows"] == 6912 and len(result["applied_updates"]) == 128
    assert result["source_unchanged"] is result["actor_unchanged"] is True
    assert captured["heldout"] == () and captured["options"]["training_only"] is True
    assert captured["options"]["budget"].updates_completed == 128
    assert result["peak_process_rss_bytes"] > 0
    assert_raw_scores(result)
    receipt = result["train_a_rollout"]
    assert receipt["status"] == "complete" and receipt["budget_binding_preserved"] is True
    assert receipt["timing_complete"] is True
    assert receipt["charged_updates"] == receipt["replay_examples"] == 0
    games = receipt["report"]
    assert games["wide_neural_method"] == "bp" and games["hidden_dim"] == 32
    assert games["native_training_examples"] == 6912 and games["updates"] == 128
    assert games["role"] == "knownTRAIN_A_only" and games["scores"] == []
    assert games["state_unchanged"] is True
    assert [(slot["context"], slot["cue"], slot["source_role"]) for slot in games["episodes"]] == [
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    live_work = games["work"]
    assert live_work["environment_resets_completed"] == 2
    assert live_work["raw_score_forwards_completed"] == 0
    assert 0 < live_work["environment_steps_returned"] <= 32
    assert live_work["environment_steps_returned"] == live_work["live_prediction_calls_returned"]
    for slot in games["episodes"]:
        assert slot["status"] == "complete"
        first = slot["predictions"][0]["observation"]
        for row in slot["predictions"]:
            assert row["features"] == first + row["observation"]
            assert row["action"] == int(np.argmax(row["probabilities"]))
            assert "target" not in row and "teacher" not in row
    work = result["work"]
    assert work["score_forward_calls_completed"] == 12 and work["score_rows_completed"] == 108
    assert work["learner_live_episodes"] == 2
    assert (
        work["total_prediction_forwards_completed"]
        == 12 + live_work["live_prediction_calls_returned"]
    )
    assert (
        work["total_prediction_rows_completed"] == 108 + live_work["live_prediction_calls_returned"]
    )
    assert work["total_environment_steps_returned"] == live_work["environment_steps_returned"]
    assert result["protocol"]["probe_updates"] == result["protocol"]["final_episodes"] == 0
    assert source_path.read_bytes() == before and json.loads(output_path.read_text()) == result
    assert_tree(torch, captured["source"].snapshot_state(), captured["initial"])
    assert_tree(
        torch, captured["stream"]._runtime.actor.snapshot_state().state, captured["initial"]
    )
    assert random.getstate() == rng[0] and pickle.dumps(np.random.get_state()) == rng[1]
    assert torch.equal(torch.get_rng_state(), rng[2])
    monkeypatch.setattr(cli, "read_training_sources", forbidden)
    with pytest.raises(FileExistsError):
        cli.main()
