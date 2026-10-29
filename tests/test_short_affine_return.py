"""Short F/old-TRAIN return through the real loop, literal model and game IO."""

from copy import deepcopy
from dataclasses import replace
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_context_linear_learner as native
from src.app import fresh_task_probe as core
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.reference_controller import ReferencePolicy
from tests import test_a4_fresh_probe_workflow as whole
from tests import test_minigrid_affine_probe as old
from tests import test_minigrid_controller_bank as bank
from tests.test_a4_adaptation import sources
from tests.test_retained_context_linear_games import state
from tests.test_probe_return_phase import return_templates

WORK = dict(probe_collections=0, records=0, failed_fits=0, failed_rows=0)
NATIVE_OWNER = native.ContextLinearLearner


def train_bridge(owner, x, y):
    child = bank.BankLiteral()
    child.restore_state(owner.snapshot_state())
    result = child.train_batch(x, y)
    owner._state = child.snapshot_state()
    return result


def predict_bridge(owner, x):
    child = bank.BankLiteral()
    child.restore_state(owner.snapshot_state())
    return child.predict(x)


@pytest.fixture
def scripted(monkeypatch):
    old.scripted_collector(monkeypatch)
    collect = binding.reserved._collect_probe_sources

    def counted(*args, **kwargs):
        WORK["probe_collections"] += 1
        result = collect(*args, **kwargs)
        WORK["records"] += 4
        return result

    monkeypatch.setattr(binding.reserved, "_collect_probe_sources", counted)
    monkeypatch.setattr(NATIVE_OWNER, "train_batch", train_bridge)
    monkeypatch.setattr(NATIVE_OWNER, "predict", predict_bridge)


def run(**kwargs):
    return binding.run_affine_fresh_probe(
        sources(), state(), retained_a4=True, return_old=True, **kwargs
    )


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = whole.Harness(patch)
        patch.setattr(NATIVE_OWNER, "train_batch", train_bridge)
        patch.setattr(NATIVE_OWNER, "predict", predict_bridge)
        original = app.run_fresh_adaptation
        patch.setattr(app, "run_fresh_adaptation", lambda **kw: original(return_old=True, **kw))
        before = pickle.dumps((random.getstate(), np.random.get_state()))
        r = h.run()
        assert r["status"] == "complete", (r.get("reason"), r.get("capture_error"))
        assert pickle.dumps((random.getstate(), np.random.get_state())) == before
        return r


def test_should_relearn_old_train_on_the_same_private_fork_after_paid_f_work(integrated):
    r = integrated
    p = r["probe"]
    assert p["source_native_state_unchanged"] and p["source_fork_calls"] == 1
    assert r["training"]["stream"]["completed_updates"] == 4
    assert r["training"]["stream"]["training_examples"] == 54
    assert r["training"]["stream"]["pending_unapplied"] == 2
    assert [g["updates"] for g in r["games"]] == [2, 4, 4]
    assert p["probe_endpoint"]["completed_updates"] == 6
    assert p["probe_endpoint"]["training_examples"] == 72
    assert [h["fit_rows"] for h in p["probe_endpoint"]["heads"]] == [9, 9]
    assert p["disposable_state"]["completed_updates"] == 8
    assert p["disposable_state"]["training_examples"] == 108
    assert [h["fit_rows"] for h in p["disposable_state"]["heads"]] == [18, 18]
    assert all(c["correct"] == 0 for c in p["old_task_readouts"][-1]["contexts"].values())
    assert all(
        c["correct"] == c["rows"] == 9
        for c in p["return_old_task_readouts"][-1]["contexts"].values()
    )
    assert r["summary"]["learning_recovery"] == [None, None]


def test_should_account_for_original_probe_and_return_and_clock_owners_separately(integrated):
    r, p = integrated, integrated["probe"]
    f, a = p["probe"], p["probe"]["return_phase"]
    assert f["completed_updates"] == p["budget"]["updates_completed"] == 2
    assert f["completed_training_examples"] == 18
    assert (
        a["status"] == "complete"
        and a["completed_updates"] == p["return_budget"]["updates_completed"] == 2
    )
    assert a["completed_training_examples"] == 36
    assert a["checkpoint_updates"] == [0, 1, 2] and a["schedule"] == [0, 1]
    assert a["checkpoints"][0]["scores"] == f["checkpoints"][-1]["scores"]
    assert (
        a["training_readouts"]["checkpoints"][0]["scores"]
        == f["training_readouts"]["checkpoints"][-1]["scores"]
    )
    assert f["score_forward_calls_completed"] == a["score_forward_calls_completed"] == 6
    assert (
        f["training_readouts"]["completed_rows"] == a["training_readouts"]["completed_rows"] == 108
    )
    assert r["readout_budget"]["initial_limits"] == r["readout_budget"]["limits"]
    assert r["readout_budget"]["started_at"] == r["readout_budget"]["observed_started_at"]
    assert p["return_budget"]["initial_limits"] == p["return_budget"]["limits"]
    assert p["return_budget"]["started_at"] >= p["budget"]["started_at"]
    assert p["return_budget"]["replay_examples_completed"] == 0
    assert len(p["collection"]) == 4 and len(p["return_old_task_readouts"]) == 3


@pytest.mark.parametrize("flag,retained", [(None, True), (1, True), ([], True), (True, False)])
def test_should_refuse_wrong_return_recipe_before_source_work(monkeypatch, flag, retained):
    monkeypatch.setattr(binding, "_AffineProbeSource", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError, match="return_old"):
        binding.run_affine_fresh_probe((), None, retained_a4=retained, return_old=flag)


@pytest.mark.parametrize("flag,probe", [(None, True), (1, True), (True, False)])
def test_should_refuse_fresh_return_without_exact_probe_before_work(monkeypatch, flag, probe):
    monkeypatch.setattr(app, "_run_adaptation", lambda *a, **k: pytest.fail("workflow reached"))
    with pytest.raises(ValueError, match="return_old"):
        app.run_fresh_adaptation(
            reference_policy=ReferencePolicy("drift"), retained_probe=probe, return_old=flag
        )


def cohorts():
    (training, evaluation), _ = binding._prepare_probe(*old.packets())
    _, monitors, _ = binding._prepare_monitors(sources(), retained_a4=True)
    return training, evaluation, monitors, return_templates(monitors)


def owner():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=2, max_replay_examples=0, max_wall_seconds=90),
        lambda: 0.0,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "flag",
        "disabled",
        "rows",
        "width",
        "classes",
        "role",
        "alias",
        "values",
        "checkpoints",
        "budget",
        "extra",
        "rehearsal",
    ],
)
def test_should_admit_the_entire_short_contract_before_fork_or_copies(monkeypatch, fault):
    training, evaluation, monitors, templates = cohorts()
    first, second = owner(), owner()
    options: dict[str, Any] = dict(short_affine_return=True)
    points: tuple[int, ...] = (0, 1, 2)
    if fault == "flag":
        options["short_affine_return"] = 1
    elif fault == "disabled":
        options = {}
    elif fault in ("rows", "width", "classes", "role"):
        b = training[0]
        changes = {
            "rows": dict(
                features=b.features[:-1].copy(),
                targets=b.targets[:-1].copy(),
                row_ids=b.row_ids[:-1],
            ),
            "width": dict(features=b.features[:, :-1].copy()),
            "classes": dict(num_classes=8),
            "role": dict(role="final"),
        }
        training = (replace(b, **changes[fault]), training[1])
    elif fault == "alias":
        templates = (replace(templates[0], features=monitors[0].features), templates[1])
    elif fault == "values":
        templates[0].targets[0, 0] = 6
    elif fault == "checkpoints":
        points = (0, 2)
    elif fault == "budget":
        second = first
    elif fault == "extra":
        options["extra_f_exposure"] = True
    else:
        options["rehearsal_factory"] = lambda *a: None
    monkeypatch.setattr(core, "_copy_batch", lambda *a: pytest.fail("copies reached"))
    with pytest.raises(ValueError):
        missing_source: Any = None
        core.run_fresh_task_probe(
            missing_source,
            None,
            training,
            evaluation,
            budget=first,
            checkpoint_updates=points,
            training_readouts=monitors,
            return_training=templates,
            return_budget=second,
            **options,
        )
    assert first.updates_completed == second.updates_completed == 0


@pytest.mark.parametrize("point", [6, 7])
def test_should_keep_paid_f_and_partial_return_when_a_return_fit_is_lost(
    scripted, monkeypatch, point
):
    original = native.ContextLinearLearner.train_batch

    def fail(child, x, y):
        if child.completed_updates == point:
            WORK["failed_fits"] += 1
            WORK["failed_rows"] += len(x)
            child._state = replace(child.snapshot_state(), failed_work=(1, len(x), 0, 0, True))
            raise RuntimeError("lost return fit")
        return original(child, x, y)

    monkeypatch.setattr(native.ContextLinearLearner, "train_batch", fail)
    r = run()
    assert r["status"] == "failed" and r["source_native_state_unchanged"]
    assert r["probe"]["completed_updates"] == r["budget"]["updates_completed"] == 2
    assert r["probe_endpoint"]["completed_updates"] == 6
    assert r["return_budget"]["updates_completed"] == point - 6
    assert r["probe"]["return_phase"]["completed_updates"] == point - 6
    assert r["disposable_state"]["completed_updates"] == point


@pytest.mark.parametrize("damage", ["value", "replacement"])
def test_should_refuse_modified_paid_f_endpoint_before_return_fit(scripted, monkeypatch, damage):
    original = binding.paid_report_guard

    def seal(value):
        check, detached = original(value)
        if "heads" in value and value.get("completed_updates") == 6:
            if damage == "value":
                value["training_examples"] = 999
            else:
                value["heads"] = deepcopy(value["heads"])
        return check, detached

    monkeypatch.setattr(binding, "paid_report_guard", seal)
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 2
    assert r["return_budget"]["updates_completed"] == 0
    assert (
        r["probe_endpoint"]["completed_updates"] == 6
        and r["probe_endpoint"]["training_examples"] == 72
    )


@pytest.mark.parametrize("damage", ["clock", "budget", "source"])
def test_should_refuse_changed_owner_or_source_during_return(scripted, monkeypatch, damage):
    original = core._run_return_phase

    def corrupt(learner, templates, evaluation, monitors, paid, second, *args, **kwargs):
        if damage == "clock":
            second.clock = lambda: 0.0
        elif damage == "budget":
            object.__setattr__(second.budget, "max_training_updates", 3)
        else:
            # Parent protects the caller's source witness at every return boundary.
            templates[0].role = "final"
        return original(learner, templates, evaluation, monitors, paid, second, *args, **kwargs)

    if damage == "source":
        # Frozen dataclass cannot be reassigned; make the same forbidden edit explicit.
        def corrupt_source(learner, templates, evaluation, monitors, paid, second, *args, **kwargs):
            object.__setattr__(templates[0], "role", "final")
            return original(learner, templates, evaluation, monitors, paid, second, *args, **kwargs)

        monkeypatch.setattr(core, "_run_return_phase", corrupt_source)
    else:
        monkeypatch.setattr(core, "_run_return_phase", corrupt)
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 2
    assert r["return_budget"]["updates_completed"] == 0


def test_should_preserve_original_no_return_schema(scripted):
    r = binding.run_affine_fresh_probe(sources(), state(), retained_a4=True)
    assert r["status"] == "complete" and "return_budget" not in r and "probe_endpoint" not in r
    assert "return_phase" not in r["probe"]


@pytest.mark.parametrize("damage", ["role", "value", "replacement"])
def test_should_refuse_input_changes_in_boundary_callback_before_any_return_read(
    scripted, monkeypatch, damage
):
    original = core._run_return_phase

    def wrap(learner, templates, evaluation, monitors, paid, second, *args, **kwargs):
        changed = False
        other = list(args)
        parent = other[3]

        def corrupt():
            nonlocal changed
            parent()
            if not changed:
                changed = True
                if damage == "role":
                    object.__setattr__(templates[0], "role", "final")
                elif damage == "value":
                    templates[0].targets[0, 0] = 6
                else:
                    object.__setattr__(templates[0], "features", templates[0].features.copy())

        other[3] = corrupt
        return original(learner, templates, evaluation, monitors, paid, second, *other, **kwargs)

    monkeypatch.setattr(core, "_run_return_phase", wrap)
    r = run()
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 2
    assert r["return_budget"]["updates_completed"] == 0
    assert r["probe"]["return_phase"]["score_forward_calls_attempted"] == 0
    assert r["probe_endpoint"]["completed_updates"] == 6
