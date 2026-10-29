"""Existing private two-arm loop with literal fits; no numerical/task benefit claim."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from src.adapters import minigrid_affine_rehearsal as study
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingDiagnostic, TrainingUpdateFailed
from tests.test_a4_adaptation import sources
from tests.test_retained_context_linear_games import state
from test_affine_private_rehearsal import probe_sources, closed_value

WORK = dict(predicts=0, predict_rows=0)


def literal_train(owner, features, targets):
    before = owner.snapshot_state()
    from src.adapters.minigrid_context_linear_learner import _project

    _, projected = _project(features)
    cue = int(projected[0, 297])
    assert np.all(projected[:, 296 + cue] == 1) and len(features) in (27, 36)
    marker = tuple(float(i in targets[:, 0]) for i in range(7))
    probability = tuple(v / sum(marker) for v in marker)
    fit = CategoricalLinearFit(
        296, 7, len(features), ((0.0,) * 7,) * 296 + (probability,), 0, (0.0,) * len(features), 0.0
    )
    heads = list(before.heads)
    heads[cue] = fit
    owner._state = replace(
        before,
        heads=tuple(heads),
        completed_updates=before.completed_updates + 1,
        training_examples=before.training_examples + len(features),
    )
    return TrainingDiagnostic("literal", 0.0)


def literal_predict(owner, features):
    from src.adapters.minigrid_context_linear_learner import _project

    WORK["predicts"] += 1
    WORK["predict_rows"] += len(features)
    _, projected = _project(features)
    result = np.zeros((len(features), 7), dtype=np.float64)
    for i, head in enumerate(owner.snapshot_state().heads):
        result[projected[:, 296 + i] == 1] = head.coefficients[-1]
    return result


@pytest.fixture
def literal(monkeypatch):
    monkeypatch.setattr(ContextLinearLearner, "train_batch", literal_train)
    monkeypatch.setattr(ContextLinearLearner, "predict", literal_predict)


def run(**options):
    return study.compare_private_rehearsal(
        sources(), state(), probe_sources(), retained_a4=True, **options
    )


def prepared():
    _, monitors, spans = study.original._prepare_monitors(sources(), retained_a4=True)
    training, evaluation = study.prepare_selected_probe(probe_sources())
    return monitors, spans, training, evaluation


def test_should_describe_the_explicit_two_old_context_recipe_and_preserve_defaults():
    p = study.describe_protocol(retained_a4=True)
    assert p["arms"] == ("current_only", "joint_A3_A4_F")
    assert (p["source_native_fits"], p["source_native_rows"]) == (4, 54)
    assert (p["fit_rows_per_update"], p["total_new_fit_rows"]) == (27, 108)
    assert p["joint_old_TRAIN_reuse_exposures"] == p["current_only_extra_F_exposures"] == 36
    assert p["upstream_paid_fits"] is p["upstream_paid_rows"] is None
    assert study.describe_protocol()["total_new_fit_rows"] == 144
    assert study.describe_protocol(fresh_reference=True)["upstream_paid_rows"] == 108


@pytest.mark.parametrize("arm", study.RETAINED_ARMS)
def test_should_compose_equal_owned_rows_with_exact_original_roles_and_spans(arm):
    monitors, spans, training, evaluation = prepared()
    before = pickle.dumps((monitors, spans, training, evaluation))
    batches, plan = study.compose_training(training, monitors, spans, arm, retained_a4=True)
    study._validate_batches(batches, evaluation, monitors)
    assert plan["input_array_bytes"] == 54 * 297 * 8
    for cue, b in enumerate(batches):
        assert (
            b.features.shape == (27, 296) and b.role == "probe_train" and len(set(b.row_ids)) == 27
        )
        expected = (
            np.tile(training[cue].targets, (3, 1))
            if arm == "current_only"
            else np.concatenate((monitors[cue].targets, training[cue].targets))
        )
        np.testing.assert_array_equal(b.targets, expected)
        roles = [s["role"] for s in plan["wakes"][cue]["spans"]]
        assert roles == (
            ["probe_train"] * 3 if arm == "current_only" else ["train", "train", "probe_train"]
        )
        assert not any(
            np.shares_memory(b.features, original.features)
            or np.shares_memory(b.targets, original.targets)
            for original in (*training, *monitors, *evaluation)
        )
    assert pickle.dumps((monitors, spans, training, evaluation)) == before


@pytest.mark.parametrize("fault", ["arm", "role", "rows", "spans"])
def test_should_refuse_other_recipes_before_matrix_composition(monkeypatch, fault):
    monitors, spans, training, _ = prepared()
    arm = "joint_A3_A4_F"
    if fault == "arm":
        arm = "joint_ABC_F"
    elif fault == "role":
        monitors = (replace(monitors[0], role="final_test"), monitors[1])
    elif fault == "rows":
        monitors = (replace(monitors[0], features=monitors[0].features[:-1]), monitors[1])
    else:
        spans = (spans[0][:-1], spans[1])
    monkeypatch.setattr(study.np, "concatenate", lambda *a, **k: pytest.fail("matrix allocated"))
    with pytest.raises(ValueError):
        study.compose_training(training, monitors, spans, arm, retained_a4=True)


@pytest.mark.parametrize("flag", [None, 1, []])
def test_should_require_exact_recipe_flag_before_budget_or_source_work(monkeypatch, flag):
    monkeypatch.setattr(study, "ToyBudgetSession", lambda *a: pytest.fail("budget allocated"))
    with pytest.raises(ValueError, match="retained_a4"):
        study.compare_private_rehearsal((), None, (), retained_a4=flag)


@pytest.mark.parametrize("fault", ["updates", "rows", "head", "failed"])
def test_should_refuse_other_native_source_geometry_before_fork(monkeypatch, fault):
    s = state()
    if fault == "updates":
        s = replace(s, completed_updates=6)
    elif fault == "rows":
        s = replace(s, training_examples=55)
    elif fault == "head":
        s = replace(
            s, heads=(replace(s.heads[0], fit_rows=9, singular_values=(0.0,) * 9), s.heads[1])
        )
    else:
        s = replace(s, failed_work=(1, 18, 0, 0, True))
    monkeypatch.setattr(study, "_PrivateSource", lambda *a: pytest.fail("source allocated"))
    r = study.compare_private_rehearsal(sources(), s, probe_sources(), retained_a4=True)
    assert r["status"] == "failed" and r["arms"] == []


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ContextLinearLearner, "train_batch", literal_train)
        patch.setattr(ContextLinearLearner, "predict", literal_predict)
        old, probes, saved = sources(), probe_sources(), state()
        before = pickle.dumps((old, probes, saved, random.getstate(), np.random.get_state()))
        r = study.compare_private_rehearsal(old, saved, probes, retained_a4=True)
        assert r["status"] == "complete", r.get("reason")
        assert (
            pickle.dumps((old, probes, saved, random.getstate(), np.random.get_state())) == before
        )
        return r


def test_should_run_both_private_arms_through_existing_probe_on_the_same_source(integrated):
    r = integrated
    assert r["budget"]["updates_completed"] == 4 and r["budget"]["replay_examples_completed"] == 36
    assert [a["arm"] for a in r["arms"]] == list(study.RETAINED_ARMS)
    for i, arm in enumerate(r["arms"]):
        assert arm["source_native_state_unchanged"] and arm["source_fork_calls"] == 1
        assert (
            arm["native_endpoint"]["completed_updates"] == 6
            and arm["native_endpoint"]["training_examples"] == 108
        )
        assert [h["fit_rows"] for h in arm["native_endpoint"]["heads"]] == [27, 27]
        assert arm["probe"]["completed_training_examples"] == 54
        assert [p["updates"] for p in arm["probe"]["checkpoints"]] == [0, 1, 2]
        assert len(arm["old_task_readouts"][-1]["contexts"]) == 4
        assert arm["committed_old_TRAIN_reuse_rows"] == (36 if i else 0)
        assert arm["committed_extra_F_rows"] == (0 if i else 36)


def test_should_keep_original_preallocated_lifetimes_and_all_score_work(integrated):
    r = integrated
    assert r["budget"]["initial_limits"] == r["budget"]["limits"]
    for a in r["arms"]:
        assert a["budget"]["started_at"] == r["budget"]["started_at"]
        assert a["budget"]["initial_limits"] == a["budget"]["limits"]
        assert a["probe"]["score_forward_calls_completed"] == 6
        assert a["probe"]["training_readouts"]["completed_rows"] == 108
        assert a["budget"]["limits"]["max_wall_seconds"] == 90


@pytest.mark.parametrize("point", [0, 1, 2, 3])
def test_should_keep_attempted_exposure_and_known_commits_when_fit_is_lost(
    literal, monkeypatch, point
):
    calls = [0]

    def train(owner, x, y):
        n = calls[0]
        calls[0] += 1
        if n == point:
            owner._state = replace(owner.snapshot_state(), failed_work=(1, len(x), 0, 0, True))
            raise TrainingUpdateFailed("lost literal fit", "literal", dict(fit_calls_attempted=1))
        return literal_train(owner, x, y)

    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)
    r = run()
    assert r["status"] == "failed" and calls[0] == point + 1
    failed = r["arms"][-1]
    assert len(r["arms"]) == (1 if point < 2 else 2)
    assert len(failed["attempts"]) == point % 2 + 1
    key = "extra_F_exposures_charged" if point < 2 else "old_TRAIN_reuse_charged"
    assert all(a[key] == 18 for a in failed["attempts"])
    assert failed["native_endpoint"]["failed_work"] == (1, 27, 0, 0, True)
    assert r["budget"]["updates_completed"] == point
    assert r["budget"]["replay_examples_completed"] == (18 if point == 3 else 0)


@pytest.mark.parametrize("fault", ["value", "replacement", "prefix"])
def test_should_refuse_changed_first_arm_before_second_arm_fits(literal, monkeypatch, fault):
    original = study.run_fresh_task_probe
    results: list[dict[str, Any]] = []

    def call(*args, **kwargs):
        if results:
            if fault == "value":
                results[0]["status"] = "changed"
            elif fault == "replacement":
                results[0]["checkpoints"] = deepcopy(results[0]["checkpoints"])
            else:
                results[0]["training_readouts"]["checkpoints"].pop()
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(study, "run_fresh_task_probe", call)
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 2
    assert r["arms"][0]["status"] == "complete"


@pytest.mark.parametrize("fault", ["future_clock", "study_clock", "counter", "probe", "late_clock"])
def test_should_refuse_changed_original_owners_or_selected_sources(literal, monkeypatch, fault):
    original = study.run_fresh_task_probe

    def call(*args, **kwargs):
        outer = closed_value(kwargs["check_boundary"], "check_boundary")
        sessions, root = closed_value(outer, "sessions"), closed_value(outer, "study")
        if fault == "future_clock":
            sessions[1].clock = lambda: 1.0
        elif fault == "study_clock":
            root.started_at = -1.0
        elif fault == "counter":
            sessions[0].updates_completed = False
        elif fault == "probe":
            object.__setattr__(closed_value(outer, "probe_episodes")[0], "role", "final_test")
        else:
            clock = sessions[0].clock

            def corrupt():
                sessions[0].replay_examples_completed = 1
                return clock()

            # Preserve the original clock object; inject the callback through
            # its budget's final observation, after preflight metadata checks.
            final = sessions[0].before_final

            def observe():
                value = final()
                corrupt()
                return value

            sessions[0].before_final = observe
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "run_fresh_task_probe", call)
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 0


def test_should_keep_probe_test_labels_out_of_both_fixed_fit_matrices():
    monitors, spans, training, evaluation = prepared()
    for arm in study.RETAINED_ARMS:
        batches, _ = study.compose_training(training, monitors, spans, arm, retained_a4=True)
        before = pickle.dumps(batches)
        tests = tuple(replace(e, targets=np.full_like(e.targets, 6)) for e in evaluation)
        study._validate_batches(batches, tests, monitors)
        assert pickle.dumps(batches) == before
