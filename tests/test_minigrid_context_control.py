"""Public-cue routing, owned conventional state and existing stream composition."""

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_control as control
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_perceptron import (
    MiniGridContextPerceptronLearner,
    copy_snapshot,
    project_features,
)
from src.core.context_perceptron import ContextPerceptronPolicy
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_context_encoding import context_public_history
from src.core.minigrid_invariant_encoding import canonicalize_public_frame
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget


def frame(cue=2):
    values = [-0.8, -1.0, -1.0] * 49 + [-1.0]
    if cue is not None:
        values[:3] = [0.2, 2.0 * cue / 5.0 - 1.0, -1.0]
    return tuple(values)


def features(cue=2, rows=2):
    return np.array([frame(cue) + frame(cue)] * rows, dtype=np.float64)


def episodes():
    return tuple(
        DemonstrationEpisode(
            f"minigrid-source-{context}-{cue}",
            context,
            "train",
            tuple(
                DemonstrationTransition(
                    step,
                    frame(color) + frame(color),
                    action,
                    0.0,
                    frame(color) + frame(color),
                    step == 8,
                    7,
                )
                for step in range(9)
            ),
            7,
        )
        for context in ("A", "B", "C_train")
        for cue, color, action in (("blue", 2, 1), ("red", 0, 2))
    )


def test_should_route_from_initial_cue_only_and_keep_current_coordinates():
    blue = context_public_history(frame(2) + frame(0))
    red = context_public_history(frame(0) + frame(2))
    assert len(blue) == 298 and blue[-2:] == (1.0, 0.0) and red[-2:] == (0.0, 1.0)
    assert blue[148:151] == canonicalize_public_frame(frame(0))[:3]
    assert context_public_history(frame(2) + frame(None))[-2:] == (1.0, 0.0)


@pytest.mark.parametrize(
    "fault", ["missing", "multiple", "color", "direction", "current", "length"]
)
def test_should_refuse_malformed_history_before_head_routing(fault):
    history = list(frame(2) + frame(2))
    if fault == "missing":
        history[0] = -0.8
    elif fault == "multiple":
        history[3:6] = history[:3]
    elif fault == "color":
        history[1] = -0.6
    elif fault == "direction":
        history[147] = 0.1
    elif fault == "current":
        history[148] = 0.123
    else:
        history.pop()
    with pytest.raises(ValueError):
        context_public_history(tuple(history))


def test_should_bound_extended_context_capacity_without_changing_default():
    assert ContextPerceptronPolicy(9, 6, (7, 8)).feature_dim == 9
    assert ContextPerceptronPolicy(298, 7, (296, 297)).feature_dim == 298
    with pytest.raises(ValueError):
        ContextPerceptronPolicy(299, 7, (297, 298))


@pytest.fixture(scope="module")
def learned():
    owner = MiniGridContextPerceptronLearner()
    zero = owner.snapshot_state()
    owner.train_batch(features(), np.ones((2, 1)))
    blue = owner.snapshot_state()
    owner.train_batch(features(0), np.full((2, 1), 2.0))
    return owner, zero, blue


def test_should_preserve_inactive_head_and_mix_readonly_rows_in_original_order(learned):
    owner, zero, blue = learned
    both = owner.snapshot_state()
    assert blue.native_state.heads[1] == zero.native_state.heads[1]
    assert both.native_state.heads[0] == blue.native_state.heads[0]
    values = owner.predict(np.vstack((features(0, 1), features(2, 1))))
    assert np.argmax(values, axis=1).tolist() == [2, 1]
    assert both.completed_updates == 2 and both.training_examples == 4
    assert owner.snapshot_state() == both


@pytest.mark.parametrize("fault", ["mixed", "missing", "dtype", "shape", "nan", "target"])
def test_should_refuse_bad_training_before_mutation(learned, fault):
    owner = learned[0].fork()
    before = owner.snapshot_state()
    x, y = features(), np.ones((2, 1))
    if fault == "mixed":
        x[1] = features(0, 1)[0]
    elif fault == "missing":
        x[1, 0] = -0.8
    elif fault == "dtype":
        x = x.astype(np.float32)
    elif fault == "shape":
        x = x[:, :-1]
    elif fault == "nan":
        x[1, 148] = np.nan
    else:
        y[1, 0] = 7
    with pytest.raises(ValueError):
        owner.train_batch(x, y)
    assert owner.snapshot_state() == before


@pytest.mark.parametrize("fault", ["encoding", "policy", "extra", "count"])
def test_should_refuse_foreign_or_corrupt_snapshots_atomically(learned, fault):
    owner = learned[0].fork()
    before = owner.snapshot_state()
    bad = deepcopy(before)
    if fault == "encoding":
        bad = replace(bad, encoding_id="foreign")
    elif fault == "policy":
        object.__setattr__(bad.policy, "extra", 1)
    elif fault == "extra":
        object.__setattr__(bad, "extra", 1)
    else:
        bad = replace(bad, native_state=replace(bad.native_state, completed_updates=99))
    with pytest.raises(ValueError):
        owner.restore_state(bad)
    assert owner.snapshot_state() == before


def test_should_own_complete_forks_and_validate_every_row(learned):
    owner = learned[0]
    before = owner.snapshot_state()
    child = owner.fork()
    child.train_batch(features(), np.zeros((2, 1)))
    assert child.snapshot_state() != before and owner.snapshot_state() == before
    assert copy_snapshot(before) == before and copy_snapshot(before) is not before
    raw = features()
    projected = project_features(raw)
    projected[0, 0] = 999
    assert raw[0, 0] != 999


@pytest.mark.parametrize("role", ["probe_train", "inner_guard", "outer_selection", "final_test"])
def test_should_refuse_nontrain_sources_before_construction(monkeypatch, role):
    sources = list(episodes())
    object.__setattr__(sources[0], "role", role)
    monkeypatch.setattr(control, "MiniGridContextPerceptronLearner", lambda: pytest.fail("model"))
    with pytest.raises(ValueError):
        control.run_control(tuple(sources))


def test_should_build_fixed_owned_train_schedule_with_unique_ids():
    sources = episodes()
    witness, batches = control.prepare_training(sources)
    assert len(batches) == 32 and len({r for b in batches for r in b.row_ids}) == 288
    assert [b.context for b in batches] == [c for c in ("A", "B", "C_train", "A") for _ in range(8)]
    assert all(b.role == "train" and b.features.shape == (9, 296) for b in batches)
    assert not np.shares_memory(batches[0].features, batches[2].features)
    witness.check(sources)


def test_should_execute_original_stream_and_complete_all_work_before_game_reads(monkeypatch):
    calls = []

    def games(source, state, *, updates, budget, check_boundary):
        check_boundary()
        assert budget.updates_completed == 32 and source.completed_updates == 0
        assert state.completed_updates == updates and state.training_examples == updates * 9
        calls.append(updates)
        return dict(status="complete", fixture="no physics", episodes=[])

    monkeypatch.setattr(evaluation, "evaluate_context_checkpoint", games)
    result = control.run_control(episodes())
    assert result["status"] == "complete", result.get("reason")
    assert calls == [0, 8, 16, 24, 32]
    assert (
        result["source_unchanged"] and result["actor_unchanged"] and result["numpy_rng_unchanged"]
    )
    assert result["budget"]["updates_completed"] == 32
    assert result["work"] == dict(
        committed_updates=32,
        committed_rows=288,
        score_calls=30,
        score_rows=270,
        unreturned_work_unknown=False,
    )
    assert len(result["applied_updates"]) == 32
    assert all(len(r["scores"]) == 6 and r["status"] == "complete" for r in result["checkpoints"])


@pytest.mark.parametrize("point", [0, 8, 16, 24, 32])
def test_should_bind_game_facade_to_declared_state_clocks_without_physics(
    monkeypatch, learned, point
):
    state = deepcopy(learned[1])
    native = state.native_state
    head = replace(native.heads[0], completed_updates=point, training_examples=9 * point)
    state = replace(
        state,
        native_state=replace(
            native,
            heads=(head, native.heads[1]),
            completed_updates=point,
            training_examples=9 * point,
        ),
    )
    calls = []

    def run(source, passed, **kwargs):
        calls.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(evaluation, "_run_checkpoint", run)
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=32), lambda: 0.0)
    for _ in range(32):
        budget.record_update()
    assert (
        evaluation.evaluate_context_checkpoint(
            object(), state, updates=point, budget=budget, check_boundary=lambda: None
        )["status"]
        == "complete"
    )
    assert calls[0]["role"] == "knownTRAIN_A_only" and calls[0]["sources"] == ()
    assert calls[0]["representation_feature_dim"] == 298
    assert [r[1:] for r in calls[0]["live_plan"]] == [("blue", "train"), ("red", "train")]
    assert calls[0]["memory_copier"](state) == state
    calls[0]["extra_check"]()
    budget.updates_completed = 31
    with pytest.raises(ValueError, match="spent context budget"):
        calls[0]["extra_check"]()


@pytest.mark.parametrize(
    "flag", ["--fit-updates", "--fit-seconds", "--train-a-rollout", "--cue-relative"]
)
def test_should_refuse_cli_overrides_before_source_or_output(monkeypatch, flag):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    args = ["command", "--context-perceptron", flag]
    if flag in ("--fit-updates", "--fit-seconds"):
        args.append("32")
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_describe_context_recipe_without_source_or_model(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["command", "--context-perceptron", "--describe"])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("source"))
    cli.main()
    assert json.loads(capsys.readouterr().out)["training_rows"] == 288


@pytest.mark.parametrize("state", ["running", "complete", "empty"])
def test_should_mark_failed_readouts_without_replacing_prior_paid_prefixes(state):
    previous = dict(status="complete", scores=[dict(accuracy=0.5)])
    current = dict(status=state, scores=[dict(accuracy=0.25)])
    saved = deepcopy(previous)
    report: dict[str, Any] = dict(
        status="running",
        checkpoints=[] if state == "empty" else [previous, current],
        work=dict(unreturned_work_unknown=False),
    )
    partial = dict(
        status="failed", reason="original lifetime exhausted", episodes=[dict(status="failed")]
    )
    error = evaluation.MiniGridEvaluationFailed(partial)
    control._record_failure(report, error)
    assert report["status"] == "failed" and "original lifetime exhausted" in report["reason"]
    assert report["work"]["unreturned_work_unknown"] is True
    assert previous == saved
    if state != "empty":
        assert report["checkpoints"][0] is previous and report["checkpoints"][1] is current
        assert current["scores"] == [dict(accuracy=0.25)]
        assert current["status"] == ("failed" if state == "running" else "complete")
        assert ("games" in current) == (state == "running")


@pytest.mark.parametrize("unknown", [True, False, None, "missing"])
def test_should_propagate_game_work_uncertainty_without_inventing_known_work(unknown):
    game_work = {} if unknown == "missing" else {"failure_work_may_be_unobserved": unknown}
    error = evaluation.MiniGridEvaluationFailed(dict(reason="stopped", work=game_work))
    report: dict[str, Any] = dict(checkpoints=[], work=dict(unreturned_work_unknown=False))
    control._record_failure(report, error)
    assert report["work"]["unreturned_work_unknown"] is (unknown is not False)
