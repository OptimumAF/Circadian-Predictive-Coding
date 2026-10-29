"""Bounded validation sharing through real guards, literal fits and game IO."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from scripts import minigrid_controller_learning as learning
from scripts import minigrid_controller_validation as validation
from tests import test_minigrid_correct_ack as ack
from tests import test_minigrid_controller_learning as old


class Harness(ack.Harness):
    cold_source: Any = None

    def run_shared(self, *, share=True, acknowledge=True, boundary=None):
        original = learning.run_controller_learning
        with pytest.MonkeyPatch.context() as patch:

            def run(*args, **kwargs):
                return original(
                    *args, **kwargs, share_readouts=share, acknowledge_correct=acknowledge
                )

            patch.setattr(learning, "run_controller_learning", run)
            return self.run_validation(boundary=boundary)


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        report = h.run_shared()
        assert report["status"] == "complete", report.get("error")
        return report, h


def test_should_share_only_equal_complete_native_states_after_all_training(integrated):
    r, h = integrated
    v = r["validation"]
    assert [e["reused_from"] for e in v["endpoints"]] == [
        "shared_base",
        None,
        "periodic",
        "shared_base",
        "periodic",
    ]
    assert v["game_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    assert h.endpoint_train_counts == [h.train_calls]
    assert r["controller"]["attempts"] == 4
    for i in (1, 2, 4):
        assert v["endpoints"][i]["after"] == dict(old_A3=2, new_A4=2)
        row = v["candidates"][i]
        assert row["window"]["applied_packet_indices"] == [0, 2]
        assert row["budget"]["updates"] == 2 and row["window"]["consumption_complete"]


def test_should_report_physical_cost_and_suppress_utility_for_all_reused_results(integrated):
    r, _ = integrated
    ends = r["validation"]["endpoints"]
    assert sum(e["readout_work"]["logical_episode_slots"] for e in ends) == 20
    assert sum(e["readout_work"]["reused_episode_slots"] for e in ends) == 16
    for e in ends:
        physical = e["reused_from"] is None
        assert e["readout_work"]["physical_game_calls"] == int(physical)
        assert e["readout_work"]["utility_available"] is physical
        assert (e["utility"] is not None) is physical
        assert (e["measures"] is not None) is physical
        assert e["readout_seconds"] >= 0 and e["recovery"] is None
    assert r["validation"]["protocol"]["readout_sharing"]["max_entries"] == 6


def test_should_keep_original_default_schema_and_three_physical_readouts(monkeypatch):
    h = Harness(monkeypatch)
    r = h.run_shared(share=False)
    assert r["status"] == "complete"
    v = r["validation"]
    assert "readout_sharing" not in v["protocol"]
    assert v["game_work"]["attempts"] == v["game_work"]["returned"] == 3
    assert all(e["utility"] is not None and "readout_work" not in e for e in v["endpoints"])


@pytest.mark.parametrize("flag", [None, 1, "yes", [], {}])
def test_should_refuse_nonbool_before_model_or_clock_work(flag):
    before = dict(old.WORK)
    with pytest.raises(ValueError, match="readout sharing"):
        learning.run_controller_learning(
            (), readout_budget=None, validation_recipe=validation.RECIPE, share_readouts=flag
        )
    with pytest.raises(ValueError, match="readout sharing"):
        validation.describe_validation(validation.RECIPE, share_readouts=flag)
    assert old.WORK == before


def test_should_require_explicit_validation_before_model_work():
    before = dict(old.WORK)
    with pytest.raises(ValueError, match="explicit validation"):
        learning.run_controller_learning((), readout_budget=None, share_readouts=True)
    assert old.WORK == before


@pytest.mark.parametrize("change", ["coefficients", "residual", "rank", "failed_work"])
def test_should_miss_for_full_snapshot_difference_despite_equal_counters(monkeypatch, change):
    h = Harness(monkeypatch)
    original = learning._Study.window

    def window(study, *args, **kwargs):
        states = original(study, *args, **kwargs)
        if args[2] == "validation-drift":
            state = states[2]
            if change == "failed_work":
                state = replace(state, failed_work=(1, 9, 0, 0, True))
            else:
                head = state.heads[0]
                if change == "coefficients":
                    coefficients = list(head.coefficients)
                    coefficients[0] = (0.25,) + coefficients[0][1:]
                    head = replace(head, coefficients=tuple(coefficients))
                elif change == "residual":
                    head = replace(head, squared_error=0.25)
                else:
                    head = replace(head, rank=1)
                state = replace(state, heads=(head, state.heads[1]))
            states[2] = state
        return states

    monkeypatch.setattr(learning._Study, "window", window)
    r = h.run_shared()
    v = r["validation"]
    assert v["game_work"]["attempts"] == 2
    if change == "failed_work":
        assert r["status"] == "failed"
        assert v["endpoints"][-1]["mode"] == "drift"
        assert v["endpoints"][-1]["status"] == "running"
    else:
        assert r["status"] == "complete", r.get("error")
        assert v["endpoints"][2]["reused_from"] is None
        assert v["endpoints"][4]["reused_from"] == "periodic"


@pytest.mark.parametrize(
    "damage", ["value", "replacement", "budget", "fit", "source", "port", "clock"]
)
def test_should_refuse_damaged_prior_success_before_cache_hit(monkeypatch, damage):
    h = Harness(monkeypatch)
    changed: list[bool] = []

    def boundary():
        v = h.cohort
        if v is None or changed or len(v.get("endpoints", [])) != 2:
            return
        e = v["endpoints"][1]
        if e["status"] != "complete":
            return
        changed.append(True)
        if damage == "value":
            e["games"]["episodes"][0]["success"] = False
        elif damage == "replacement":
            e["games"] = deepcopy(e["games"])
        elif damage == "budget":
            h.budget.updates_completed = 1
        elif damage == "fit":
            object.__setattr__(h.fit_owner._fitted, "fitted_at", 99)
        elif damage == "source":
            object.__setattr__(h.cold_source, "completed_updates", 99)
        elif damage == "clock":
            h.time += 200
        else:
            monkeypatch.setattr(learning._Study, "game", lambda *a, **k: e["games"])

    # Cold source replacement is observable through the actual study owner.
    original = validation.run_validation

    def run(study, *args, **kwargs):
        h.cold_source = study.cold.state
        return original(study, *args, **kwargs)

    monkeypatch.setattr(validation, "run_validation", run)
    r = h.run_shared(boundary=boundary)
    assert changed and r["status"] == r["validation"]["status"] == "failed"
    assert r["validation"]["game_work"]["attempts"] == 1
    assert r["validation"]["endpoints"][0]["status"] == "complete"


def test_should_preserve_lost_game_return_without_cache_admission(monkeypatch):
    h = Harness(monkeypatch)
    original = learning._Study.game

    def lose(study, state, **kwargs):
        if study.report is h.cohort:
            study.game_attempts += 1
            raise RuntimeError("lost sharing readout return")
        return original(study, state, **kwargs)

    monkeypatch.setattr(learning._Study, "game", lose)
    r = h.run_shared()
    v = r["validation"]
    assert r["status"] == v["status"] == "failed"
    assert v["game_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)
    assert len(v["endpoints"]) == 2 and v["endpoints"][1]["games"] is None


def test_should_not_share_between_invocations(monkeypatch):
    h = Harness(monkeypatch)
    first = h.run_shared()
    assert first["status"] == "complete"
    with pytest.MonkeyPatch.context() as patch:
        other = Harness(patch)
        second = other.run_shared()
    assert second["status"] == "complete"
    assert first["validation"]["game_work"]["attempts"] == 1
    assert second["validation"]["game_work"]["attempts"] == 1


@pytest.mark.parametrize(
    "options", [{"applied_cues": (1, 0)}, {"cue_prefix": (0, 1)}, {"applied_cues": (0,)}]
)
def test_should_miss_same_state_with_different_readout_arguments(integrated, options):
    r, h = integrated
    state = next(s for s in h.states if s.completed_updates == 4)
    games = r["validation"]["endpoints"][1]["games"]
    cache = [(deepcopy(state), {"applied_cues": (0, 1)}, games, "periodic")]
    assert validation._find_shared_readout(cache, state, options) is None
    found = validation._find_shared_readout(cache, deepcopy(state), {"applied_cues": (0, 1)})
    assert found[0] is games and found[1] == "periodic"


def test_should_share_without_acknowledgement_and_preserve_original_pending_work(monkeypatch):
    h = Harness(monkeypatch)
    r = h.run_shared(acknowledge=False)
    v = r["validation"]
    assert r["status"] == "complete", r.get("error")
    assert "acknowledge_correct" not in v["protocol"]
    assert v["game_work"]["attempts"] == v["game_work"]["returned"] == 2
    assert v["endpoints"][4]["reused_from"] == "drift"
    assert [c["window"]["pending_unapplied"] for c in v["candidates"]] == [4, 2, 3, 4, 3]


def test_should_refuse_candidate_changed_during_readout_before_cache_admission(monkeypatch):
    h = Harness(monkeypatch)
    original = learning._Study.game

    def change(study, state, **kwargs):
        result = original(study, state, **kwargs)
        if study.report is h.cohort:
            object.__setattr__(state, "training_examples", state.training_examples + 18)
        return result

    monkeypatch.setattr(learning._Study, "game", change)
    r = h.run_shared()
    v = r["validation"]
    assert r["status"] == v["status"] == "failed"
    assert "complete candidate snapshot" in v["error"]
    assert v["game_work"] == dict(attempts=1, returned=1, unreturned_work_unknown=False)
    assert len(v["endpoints"]) == 2 and v["endpoints"][1]["games"] is None
