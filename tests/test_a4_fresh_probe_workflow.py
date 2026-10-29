"""First-party source-to-isolated-probe composition, literal learning/game IO."""

from copy import deepcopy
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as probe
from src.adapters import minigrid_context_linear_learner as native
from src.core.reference_controller import ReferencePolicy
from tests import test_a4_reference_workflow as old
from tests import test_minigrid_affine_probe as fakes
from tests import test_minigrid_controller_bank as bank

WORK = dict(
    workflows=0,
    source_reads=0,
    source_collections=0,
    source_teacher_steps=0,
    probe_collections=0,
    probe_records=0,
    failed_fit_attempts=0,
    failed_fit_rows=0,
)


class Harness:
    def __init__(self, patch):
        old.HOOKS.clear()
        self.box = old.bind_collection(patch)
        patch.setattr(native, "ContextLinearLearner", bank.BankLiteral)
        patch.setattr(app, "ContextLinearLearner", bank.BankLiteral)
        original_read = cli.read_training_sources

        def read(path):
            WORK["source_reads"] += 1
            return original_read(path)

        patch.setattr(cli, "read_training_sources", read)
        patch.setattr(app, "evaluate_context_linear_stream_checkpoint", self.game)
        fakes.scripted_collector(patch)
        original_collect = probe.reserved._collect_probe_sources

        def collect(*args, **kwargs):
            WORK["probe_collections"] += 1
            result = original_collect(*args, **kwargs)
            WORK["probe_records"] += len(args[1])
            return result

        patch.setattr(probe.reserved, "_collect_probe_sources", collect)
        self.on_game = lambda: None

    def game(self, owner, state, **kwargs):
        self.on_game()
        return old.scripted_readout(owner, old.literal_state(state.completed_updates), **kwargs)

    def run(self, *, enabled=True):
        WORK["workflows"] += 1
        before_envs, before_steps = len(self.box["environments"]), self.box["teacher_calls"]
        result = app.run_fresh_adaptation(
            reference_policy=ReferencePolicy("drift"), retained_probe=enabled
        )
        WORK["source_collections"] += len(self.box["environments"]) - before_envs
        WORK["source_teacher_steps"] += self.box["teacher_calls"] - before_steps
        assert all(e.closed for e in self.box["environments"])
        return result


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        before = dict(WORK)
        rng = pickle.dumps((random.getstate(), np.random.get_state()))
        original_probe = probe.run_affine_fresh_probe

        def after_source(episodes, state, **kwargs):
            assert len(old.HOOKS["observed"]) == 2
            assert (state.completed_updates, state.training_examples) == (4, 54)
            assert len(episodes) == 4 and all(e.role == "train" for e in episodes)
            return original_probe(episodes, state, **kwargs)

        patch.setattr(probe, "run_affine_fresh_probe", after_source)
        r = h.run()
        assert r["status"] == "complete", (r.get("reason"), r.get("capture_error"))
        assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
        return r, {k: WORK[k] - before[k] for k in WORK}


def test_should_run_existing_source_then_one_isolated_f_probe_without_extra_source_read(integrated):
    r, work = integrated
    assert work == dict(
        workflows=1,
        source_reads=1,
        source_collections=2,
        source_teacher_steps=18,
        probe_collections=1,
        probe_records=4,
        failed_fit_attempts=0,
        failed_fit_rows=0,
    )
    assert r["training"]["stream"]["completed_updates"] == 4
    assert r["training"]["stream"]["training_examples"] == 54
    assert r["training"]["stream"]["pending_unapplied"] == 2
    assert [p["updates"] for p in r["games"]] == [2, 4, 4]
    assert r["games"][-1]["readout_reused_from_tick"] == 4
    assert r["probe_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    p = r["probe"]
    assert p["status"] == "complete" and p["source_native_state_unchanged"]
    assert p["source_fork_calls"] == 1
    assert p["disposable_state"]["completed_updates"] == 6
    assert p["disposable_state"]["training_examples"] == 72
    assert p["probe"]["completed_updates"] == p["budget"]["updates_completed"] == 2
    assert p["probe"]["completed_training_examples"] == 18


def test_should_keep_original_readout_owner_and_separate_probe_fit_costs_and_monitors(integrated):
    r, _ = integrated
    readout, p = r["readout_budget"], r["probe"]
    assert readout["initial_limits"] == readout["limits"]
    assert readout["started_at"] == readout["observed_started_at"]
    assert readout["updates"] == readout["replay"] == 0
    assert (
        readout["started_at"] <= r["training"]["budget"]["started_at"] <= p["budget"]["started_at"]
    )
    assert p["initial_probe_limits"] == p["budget"]["limits"]
    assert p["budget"]["replay_examples_completed"] == 0
    assert p["probe"]["score_forward_calls_completed"] == 6
    assert p["probe"]["training_readouts"]["completed_rows"] == 108
    assert len(p["old_task_readouts"][-1]["contexts"]) == 4
    assert r["summary"]["complete"] and r["summary"]["learning_recovery"] == [None, None]
    assert "F_games" not in r and "retained_probe" in r["protocol"]


def test_should_preserve_default_schema_and_skip_probe(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(
        probe, "run_affine_fresh_probe", lambda *a, **k: pytest.fail("probe reached")
    )
    r = h.run(enabled=False)
    assert r["status"] == "complete" and "probe" not in r and "probe_work" not in r
    assert "retained_probe" not in r["protocol"]
    assert r["protocol"]["definition"] == "portable_affine_a4_reference_adaptation_v1"


def test_should_refuse_wrong_actual_source_geometry_before_f_collection(monkeypatch):
    h = Harness(monkeypatch)
    before = WORK["probe_collections"]

    def wrong(owner, features):
        from tests import test_minigrid_controller_learning as counters

        counters.WORK["predicts"] += 1
        counters.WORK["predict_rows"] += len(features)
        values = np.zeros((len(features), 7))
        values[:, 0] = 1
        return values

    monkeypatch.setattr(bank.BankLiteral, "predict", wrong)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert r["training"]["stream"]["completed_updates"] == 6
    assert len(r["games"]) == 3 and r["probe"] is None
    assert WORK["probe_collections"] == before
    assert r["probe_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


@pytest.mark.parametrize(
    "enabled,policy,controller",
    [
        (None, ReferencePolicy("drift"), None),
        (1, ReferencePolicy("drift"), None),
        (True, None, None),
        (True, ReferencePolicy("periodic"), None),
        (True, ReferencePolicy("drift", drift_threshold=0.2), None),
        (True, None, "retain_arrived"),
    ],
)
def test_should_refuse_invalid_probe_recipe_before_data_or_models(
    monkeypatch, enabled, policy, controller
):
    monkeypatch.setattr(app, "_run_adaptation", lambda *a, **k: pytest.fail("workflow reached"))
    with pytest.raises(ValueError, match="retained_probe"):
        app.run_fresh_adaptation(
            reference_policy=policy, controller_mode=controller, retained_probe=enabled
        )


def test_should_describe_probe_without_reading_or_collecting(monkeypatch):
    monkeypatch.setattr(
        probe.reserved, "_collect_probe_sources", lambda *a, **k: pytest.fail("collector reached")
    )
    p = app.describe_fresh_adaptation(
        reference_policy=ReferencePolicy("drift"), retained_probe=True
    )
    assert p["historical_source_reads"] == 0
    assert p["retained_probe"]["selected_native_updates"] == 4
    assert p["retained_probe"]["probe_updates"] == 2


def test_should_keep_source_prefix_when_probe_call_loses_its_return(monkeypatch):
    h = Harness(monkeypatch)

    def lost(*args, **kwargs):
        kwargs["check_boundary"]()
        raise RuntimeError("lost probe return")

    monkeypatch.setattr(probe, "run_affine_fresh_probe", lost)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert len(r["games"]) == 3 and r["training"]["stream"]["completed_updates"] == 4
    assert r["probe"] is None
    assert r["probe_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


def test_should_retain_partial_failed_probe_costs_without_retry(monkeypatch):
    h = Harness(monkeypatch)
    original = bank.BankLiteral.train_batch

    def fail(owner, x, y):
        if owner.snapshot_state().completed_updates == 5:
            WORK["failed_fit_attempts"] += 1
            WORK["failed_fit_rows"] += len(x)
            raise RuntimeError("lost second probe update")
        return original(owner, x, y)

    monkeypatch.setattr(bank.BankLiteral, "train_batch", fail)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert len(r["games"]) == 3 and r["probe"]["status"] == "failed"
    assert r["probe"]["probe"]["completed_updates"] == 1
    assert r["probe"]["budget"]["updates_completed"] == 1
    assert r["probe_work"]["attempts"] == r["probe_work"]["returned"] == 1


@pytest.mark.parametrize("damage", ["value", "replacement"])
def test_should_refuse_changed_probe_call_costs_and_keep_known_counts(monkeypatch, damage):
    h = Harness(monkeypatch)

    def corrupt(*args, check_boundary, **kwargs):
        cells = dict(zip(check_boundary.__code__.co_freevars, check_boundary.__closure__))
        if damage == "value":
            cells["probe_work"].cell_contents["attempts"] = 99
        else:
            report = cells["report"].cell_contents
            report["probe_work"] = deepcopy(report["probe_work"])
        check_boundary()
        pytest.fail("damaged probe-call evidence admitted")

    monkeypatch.setattr(probe, "run_affine_fresh_probe", corrupt)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert len(r["games"]) == 3 and r["probe"] is None
    assert r["probe_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)


@pytest.mark.parametrize("damage", ["game_value", "game_replacement", "budget_clock"])
def test_should_refuse_prior_source_evidence_changed_by_probe(monkeypatch, damage):
    h = Harness(monkeypatch)
    original = app.paid_report_guard
    games: list[Any] = []

    def seal(value):
        if type(value) is dict and "updates" in value and "games" in value:
            games.append(value)
        return original(value)

    monkeypatch.setattr(app, "paid_report_guard", seal)

    def probe_call(*args, check_boundary, **kwargs):
        assert len(games) == 3
        if damage == "game_value":
            games[0]["games"]["episodes"][0]["success"] = False
        elif damage == "game_replacement":
            games[0]["games"] = deepcopy(games[0]["games"])
        else:
            # The boundary's actual original owner is observed in the read port.
            old.HOOKS["owner"].clock = lambda: 0.0
        check_boundary()
        pytest.fail("damaged source admitted")

    def remember(budget, check):
        old.HOOKS["owner"] = budget

    old.HOOKS["readout"] = remember
    monkeypatch.setattr(probe, "run_affine_fresh_probe", probe_call)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert r["probe_work"]["unreturned_work_unknown"]
    assert len(r["games"]) == 3 and r["games"][0]["games"]["episodes"][0]["success"]


@pytest.mark.parametrize("damage", ["value", "replacement"])
def test_should_refuse_changed_completed_probe_and_keep_paid_copy(monkeypatch, damage):
    h = Harness(monkeypatch)
    original_probe = probe.run_affine_fresh_probe
    original_guard = app.paid_report_guard
    captured = []

    def run(*args, **kwargs):
        p = original_probe(*args, **kwargs)
        captured.append(p)
        return p

    def seal(value):
        check, saved = original_guard(value)
        if captured and value is captured[0]:
            if damage == "value":
                value["source_fork_calls"] = 99
            else:
                value["probe"] = deepcopy(value["probe"])
        return check, saved

    monkeypatch.setattr(probe, "run_affine_fresh_probe", run)
    monkeypatch.setattr(app, "paid_report_guard", seal)
    r = h.run()
    assert r["status"] == "failed" and not r["summary"]["complete"]
    assert r["probe"]["status"] == "complete" and r["probe"]["source_fork_calls"] == 1
    assert r["probe"]["probe"]["completed_updates"] == 2
    assert r["probe_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
