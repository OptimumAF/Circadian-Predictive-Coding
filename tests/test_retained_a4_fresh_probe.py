"""Same-run isolated F diagnostic and readonly admission, literal learning/IO."""

from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.learner_ports import TrainingUpdateFailed
from tests import test_minigrid_affine_probe as old
from tests import test_minigrid_controller_learning as literal
from tests.test_a4_adaptation import sources
from tests.test_catalogue_capacity import sources as abc_sources
from tests.test_context_linear_causal_games import PhaseEnvironment
from tests.test_retained_context_linear_games import state, budget
from tests.test_after_c_linear_games import state as abc_state


def literal_train(self, x, y):
    owner = object.__new__(literal.LiteralLearner)
    owner.state = self.snapshot_state()
    result = literal.LiteralLearner.train_batch(owner, x, y)
    self._state = owner.state
    return result


def literal_predict(self, x):
    owner = object.__new__(literal.LiteralLearner)
    owner.state = self.snapshot_state()
    return literal.LiteralLearner.predict(owner, x)


@pytest.fixture
def scripted(monkeypatch):
    captured = old.scripted_collector(monkeypatch)
    monkeypatch.setattr(native.ContextLinearLearner, "train_batch", literal_train)
    monkeypatch.setattr(native.ContextLinearLearner, "predict", literal_predict)
    return captured


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        old.scripted_collector(patch)
        patch.setattr(native.ContextLinearLearner, "train_batch", literal_train)
        patch.setattr(native.ContextLinearLearner, "predict", literal_predict)
        episodes, saved = sources(), state()
        original = pickle.dumps((episodes, saved, np.random.get_state()))
        report = binding.run_affine_fresh_probe(episodes, saved, retained_a4=True)
        assert report["status"] == "complete", report["reason"]
        assert pickle.dumps((episodes, saved, np.random.get_state())) == original
        return report, saved, episodes


def test_should_measure_f_learning_and_old_interference_on_one_isolated_fork(integrated):
    r, saved, episodes = integrated
    assert r["source_native_state_unchanged"] and r["source_fork_calls"] == 1
    assert r["budget"]["updates_completed"] == 2 and r["budget"]["replay_examples_completed"] == 0
    assert r["budget"]["limits"] == r["initial_probe_limits"]
    assert r["probe"]["completed_training_examples"] == 18
    assert r["disposable_state"]["completed_updates"] == 6
    assert r["disposable_state"]["training_examples"] == 72
    assert [h["fit_rows"] for h in r["disposable_state"]["heads"]] == [9, 9]
    assert [p["updates"] for p in r["old_task_readouts"]] == [0, 1, 2]
    assert all(
        set(p["contexts"]) == {e.episode_id for e in episodes} for p in r["old_task_readouts"]
    )
    assert all(d["rows"] == 9 for p in r["old_task_readouts"] for d in p["contexts"].values())
    assert r["protocol"]["selected_native_updates"] == saved.completed_updates == 4
    assert r["protocol"]["paid_source_fits"] is r["protocol"]["paid_source_rows"] is None
    assert r["preparation_seconds"] >= 0
    assert r["whole_diagnostic_seconds"] >= r["budget"]["elapsed_seconds"]
    assert all(s["accuracy"] == 1.0 for s in r["probe"]["checkpoints"][-1]["scores"].values())
    assert all(d["correct"] == 0 for d in r["old_task_readouts"][-1]["contexts"].values())


def test_should_keep_actual_probe_and_monitor_work_separate(integrated):
    r, _, _ = integrated
    p = r["probe"]
    assert (
        p["score_forward_calls_completed"]
        == p["training_readouts"]["score_forward_calls_completed"]
        == 6
    )
    assert p["training_readouts"]["completed_rows"] == 108
    assert p["training_readouts"]["reserved_data_bytes"] == 36 * 297 * 8
    assert p["reserved_data_bytes"] == 36 * 297 * 8
    assert r["protocol"]["score_rows"] == 162
    assert len(r["collection"]) == len(r["probe_data"]) == 4
    assert len(r["monitor_source_spans"]) == 2
    assert all(len(group) == 2 for group in r["monitor_source_spans"])


@pytest.mark.parametrize("phase", ["before_probe", "after_probe"])
def test_should_read_this_run_f_endpoints_through_original_loop(
    scripted, integrated, monkeypatch, phase
):
    r, saved, _ = integrated
    selected = (
        saved
        if phase == "before_probe"
        else cli._context_linear_training_snapshot(r["disposable_state"])
    )
    monkeypatch.setattr(PhaseEnvironment, "work", {})
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", PhaseEnvironment)
    owner = native.ContextLinearLearner()
    initial = owner.snapshot_state()
    report = evaluation.evaluate_context_linear_probe_checkpoint(
        owner, selected, phase=phase, budget=budget(), check_boundary=lambda: None, retained_a4=True
    )
    assert report["status"] == "complete" and report["state_unchanged"]
    assert report["source_recipe"] == "retained_a4"
    assert report["source_fit_calls"] == selected.completed_updates
    assert report["source_fit_rows"] == selected.training_examples
    assert report["new_fit_calls"] == report["new_optimizer_updates"] == 0
    assert [e["context"] for e in report["episodes"]] == ["F-blue", "F-red"]
    assert all(e["environment"]["closed"] for e in report["episodes"])
    assert all(e.role == "probe_test" for e in PhaseEnvironment.work["environments"])
    assert owner.snapshot_state() == initial


def test_should_preserve_default_abc_recipe_and_offsets(scripted):
    r = binding.run_affine_fresh_probe(abc_sources(), abc_state())
    assert r["status"] == "complete", r["reason"]
    assert r["definition"] == "minigrid_after_c_affine_fresh_F_probe_v1"
    assert r["protocol"]["paid_source_fits"] == 10
    assert r["protocol"]["score_rows"] == 216
    assert r["disposable_state"]["completed_updates"] == 8
    assert r["disposable_state"]["training_examples"] == 126
    assert r["probe"]["training_readouts"]["completed_rows"] == 162


@pytest.mark.parametrize("flag", [1, None, "yes", [], {}])
def test_should_refuse_nonbool_recipe_before_work(monkeypatch, flag):
    monkeypatch.setattr(binding, "_AffineProbeSource", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError, match="exact bool"):
        binding.describe_protocol(retained_a4=flag)
    with pytest.raises(ValueError, match="exact bool"):
        binding.run_affine_fresh_probe((), None, retained_a4=flag)
    with pytest.raises(ValueError, match="exact bool"):
        evaluation.evaluate_context_linear_probe_checkpoint(
            None,
            None,
            phase="before_probe",
            budget=None,
            check_boundary=lambda: None,
            retained_a4=flag,
        )


@pytest.mark.parametrize("fault", ["updates", "rows", "head", "failed", "role", "order", "context"])
def test_should_refuse_wrong_retained_source_before_collection(monkeypatch, fault):
    saved, episodes = state(), sources()
    if fault == "updates":
        saved = replace(saved, completed_updates=5)
    elif fault == "rows":
        saved = replace(saved, training_examples=55)
    elif fault == "head":
        saved = replace(
            saved,
            heads=(replace(saved.heads[0], fit_rows=9, singular_values=(0.0,) * 9), saved.heads[1]),
        )
    elif fault == "failed":
        saved = replace(saved, failed_work=(1, 9, 0, 0, True))
    elif fault == "role":
        episodes = (replace(episodes[0], role="probe_train"), *episodes[1:])
    elif fault == "order":
        episodes = episodes[::-1]
    else:
        episodes = (replace(episodes[0], context="B"), *episodes[1:])
    monkeypatch.setattr(binding, "_AffineProbeSource", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError):
        binding.run_affine_fresh_probe(episodes, saved, retained_a4=True)


@pytest.mark.parametrize("fault", ["wrong_flag", "rows", "head", "phase", "failed"])
def test_should_refuse_unpaid_f_readout_geometry_before_loop(monkeypatch, fault):
    saved, phase, flag = state(), "before_probe", True
    if fault == "wrong_flag":
        flag = False
    elif fault == "rows":
        saved = replace(saved, training_examples=55)
    elif fault == "head":
        saved = replace(saved, heads=abc_state().heads)
    elif fault == "phase":
        phase = "after_probe"
    else:
        saved = replace(saved, failed_work=(1, 9, 0, 0, True))
    monkeypatch.setattr(evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("loop reached"))
    with pytest.raises(ValueError):
        evaluation.evaluate_context_linear_probe_checkpoint(
            native.ContextLinearLearner(),
            saved,
            phase=phase,
            budget=budget(),
            check_boundary=lambda: None,
            retained_a4=flag,
        )


def test_should_keep_failed_second_fit_paid_and_no_later_success(scripted, monkeypatch):
    def train(owner, x, y):
        if owner.completed_updates == 5:
            literal.WORK["updates"] += 1
            literal.WORK["rows"] += len(x)
            owner._state = replace(owner.snapshot_state(), failed_work=(1, 9, 0, 0, True))
            raise TrainingUpdateFailed(
                "lost second fit", native.DIAGNOSTIC, native._failure(owner._state.failed_work)
            )
        return literal_train(owner, x, y)

    monkeypatch.setattr(native.ContextLinearLearner, "train_batch", train)
    r = binding.run_affine_fresh_probe(sources(), state(), retained_a4=True)
    assert r["status"] == "failed" and r["source_native_state_unchanged"]
    assert r["budget"]["updates_completed"] == r["probe"]["completed_updates"] == 1
    assert r["disposable_state"]["completed_updates"] == 5
    assert r["disposable_state"]["failed_work"] is not None
    assert [p["updates"] for p in r["old_task_readouts"]] == [0, 1]


@pytest.mark.parametrize(
    "fault", ["collection_replacement", "collection_value", "charge", "clock", "source"]
)
def test_should_stop_damaged_prior_sources_and_owners_before_probe(scripted, monkeypatch, fault):
    def probe(source, saved, *args, budget, check_boundary, **kwargs):
        if fault == "collection_replacement":
            scripted["reports"][0] = deepcopy(scripted["reports"][0])
        elif fault == "collection_value":
            scripted["reports"][0]["closed"] = False
        elif fault == "charge":
            budget.updates_completed = 1
        elif fault == "clock":
            budget.clock = lambda: 0.0
        else:
            source.restore_state(replace(saved, completed_updates=5, training_examples=63))
        check_boundary()
        pytest.fail("damaged evidence admitted")

    monkeypatch.setattr(binding, "run_fresh_task_probe", probe)
    r = binding.run_affine_fresh_probe(sources(), state(), retained_a4=True)
    assert r["status"] == "failed" and r["source_fork_calls"] == 0
    assert len(r["known_collection_prefix"]) == 4


@pytest.mark.parametrize("damage", ["value", "replacement"])
def test_should_refuse_changed_completed_probe_before_further_capture(
    scripted, monkeypatch, damage
):
    original = binding.run_fresh_task_probe
    completed = []

    def probe(*args, **kwargs):
        r = original(*args, **kwargs)
        completed.append(r)
        return r

    def boundary():
        if completed:
            if damage == "value":
                completed[0]["completed_updates"] = 0
            else:
                completed[0]["training_readouts"] = deepcopy(completed[0]["training_readouts"])

    monkeypatch.setattr(binding, "run_fresh_task_probe", probe)
    r = binding.run_affine_fresh_probe(
        sources(), state(), retained_a4=True, check_boundary=boundary
    )
    assert r["status"] == "failed" and r["budget"]["updates_completed"] == 2
    assert r["probe"]["completed_updates"] == 2
    assert r["disposable_state"]["completed_updates"] == 6
