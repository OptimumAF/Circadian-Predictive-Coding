"""Frozen schedule composition with literal learners and scripted native games."""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

import pytest

from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from scripts import minigrid_controller_learning as learning
from scripts import minigrid_controller_validation as validation
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSampler
from tests import test_minigrid_controller_bank as bank
from tests import test_minigrid_controller_learning as old


class Harness(bank.BankHarness):
    def __init__(self, patch):
        super().__init__(patch)
        self.cohort: Any = None
        self.fit_owner: Any = None
        self.endpoint_train_counts: list[int] = []
        original = validation.run_validation
        patch.setattr(validation, "monotonic", lambda: self.time)

        def run(study, controller, recipe, **kwargs):
            self.cohort = study.report
            self.fit_owner = controller
            return original(study, controller, recipe, **kwargs)

        patch.setattr(validation, "run_validation", run)
        self.on_game = self.require_release

    def game(
        self,
        cold,
        state,
        *,
        budget,
        check_boundary,
        reference_window,
        reference_catalogue=False,
        reference_catalogue_cues=None,
    ):
        if reference_catalogue_cues is None:
            return super().game(
                cold,
                state,
                budget=budget,
                check_boundary=check_boundary,
                reference_window=reference_window,
                reference_catalogue=reference_catalogue,
            )
        # Exercise the real cue-aware admission; replace only native episode IO.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native, "ContextLinearLearner", bank.BankLiteral)
            patch.setattr(evaluation, "_run_checkpoint", self.literal_checkpoint)
            return evaluation.evaluate_context_linear_stream_checkpoint(
                cold,
                state,
                budget=budget,
                check_boundary=check_boundary,
                reference_window=reference_window,
                reference_catalogue=reference_catalogue,
                reference_catalogue_cues=reference_catalogue_cues,
            )

    def literal_checkpoint(self, source, state, **kwargs):
        old.WORK["games"] += 1
        kwargs["extra_check"]()
        kwargs["check_boundary"]()
        self.time += 0.1
        self.states.append(deepcopy(state))
        rows = []
        for i, (branch, cue) in enumerate(((3, "blue"), (3, "red"), (4, "blue"), (4, "red"))):
            action = 2 if branch == 3 else 5
            success = bool(state.heads[i % 2].coefficients[-1][action])
            rows.append(
                dict(
                    status="complete",
                    source_role="train",
                    cue=cue,
                    success=success,
                    predictions=[dict(action=action)],
                    return_value=float(success),
                    environment=dict(
                        episode_id=f"validation-{len(self.states)}-{i}",
                        role="train",
                        cue=cue,
                        layout=asdict(MiniGridKeyDoorLayout("A", branch_row=branch)),
                        seed=17,
                        max_steps=16,
                        closed=True,
                        environment_steps=1,
                        failure_work_may_be_unobserved=False,
                        transitions=[dict(action=action, applied=True)],
                    ),
                )
            )
            rows[-1]["return"] = rows[-1].pop("return_value")
        result = dict(
            status="complete", state_unchanged=True, episodes=rows, **kwargs["return_metadata"]
        )
        self.games.append(result)
        self.on_game(result)
        kwargs["check_boundary"]()
        return result

    def require_release(self, games):
        if self.cohort is not None:
            assert self.cohort["endpoint_reads_released"] is True
            assert len(self.cohort["candidates"]) == 5
            assert all(r["status"] == "training_complete" for r in self.cohort["candidates"])
            self.endpoint_train_counts.append(self.train_calls)

    def run_validation(self, *, recipe=validation.RECIPE, episodes=None, boundary=None):
        with ProcessRssSampler(interval_seconds=0.01) as memory:
            self.budget = ToyBudgetSession(
                ToyExecutionBudget(
                    max_training_updates=0,
                    max_replay_examples=0,
                    max_wall_seconds=150,
                    max_process_rss_bytes=512 * 1024**2,
                ),
                lambda: self.time,
            )
            self.budget.attach_memory(memory)
            return learning.run_controller_learning(
                old.sources() if episodes is None else episodes,
                readout_budget=self.budget,
                check_boundary=boundary,
                retain_arrived=True,
                validation_recipe=recipe,
            )


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        report = h.run_validation()
        assert report["status"] == "complete", report.get("error")
        return report, h


def test_should_fit_once_then_compare_same_source_with_five_frozen_policies(integrated):
    r, h = integrated
    v = r["validation"]
    assert r["application"] is None and len(r["trials"]) == 4
    assert r["controller"]["attempts"] == r["controller"]["completed"] == 4
    assert v["status"] == "complete" and v["frozen_fit_unchanged"]
    assert [c["mode"] for c in v["candidates"]] == list(validation.MODES)
    assert len(v["budgets"]) == 5 and len(r["budgets"]) == 5
    for row in v["candidates"]:
        assert row["checkpoints"][0] == dict(relative_updates=0, native_updates=2, native_rows=18)
        assert row["source_unchanged"] and row["actor_unchanged"]
        assert row["budget"]["limits"]["max_training_updates"] == 4
        assert row["budget"]["limits"]["max_wall_seconds"] == 90
    fitted = v["candidates"][-1]["window"]["controller_application"]
    assert fitted["fit"]["attempts"] == fitted["fit"]["completed"] == 4
    assert fitted["fit"]["discarded_trials"] == 4
    assert h.fit_owner is not None and v["protocol"]["references"][-1]["reserved_updates"] == 4


def test_should_defer_all_endpoints_until_all_training_is_terminal(integrated):
    r, h = integrated
    v = r["validation"]
    assert v["endpoint_reads_released"] and len(v["endpoints"]) == 5
    assert h.endpoint_train_counts and len(set(h.endpoint_train_counts)) == 1
    assert h.endpoint_train_counts[0] == h.train_calls
    assert all(e["status"] == "complete" for e in v["endpoints"])


def test_should_share_causal_delay2_opportunities_and_explicit_prior_epochs(integrated):
    r, _ = integrated
    for row in r["validation"]["candidates"]:
        window = row["window"]
        assert [d["observation"]["tick"] for d in window["decisions"]] == list(range(6))
        assert window["decisions"][0]["completed_updates"] == 0
        assert window["decisions"][1]["completed_updates"] == 0
        assert row["source_episode_ids"] == [old.sources()[i].episode_id for i in (2, 0, 3, 1)]
        plan = row["catalogue"]["planned"]
        assert plan["window_label_delay"] == 2
        assert [w["arrived_at"] for w in plan["window_wakes"]] == [2, 3, 4, 5]
        for wake in plan["window_wakes"]:
            assert wake["arrived_at"] == wake["observed_at"] + 2
            for prior in wake["retained"]:
                if prior["epoch"] == "window":
                    assert prior["arrived_at"] < wake["arrived_at"]
                else:
                    assert prior["epoch"] == "warmup" and prior["arrived_at"] in (1, 2)
        for receipt in window["applied_updates"]:
            assert receipt["applied_at"] >= receipt["arrived_at"]


def test_should_report_pending_actual_exposures_and_endpoint_utility_without_recovery(integrated):
    r, _ = integrated
    v = r["validation"]
    assert v["candidates"][0]["budget"]["updates"] == 0
    assert v["candidates"][3]["budget"]["updates"] == 0
    assert all(c["window"]["window_complete"] for c in v["candidates"])
    assert not v["candidates"][0]["window"]["training_complete"]
    for row, endpoint in zip(v["candidates"], v["endpoints"], strict=True):
        n = row["budget"]["updates"]
        assert row["catalogue"]["committed"]["native_rows"] == 18 * n
        assert row["catalogue"]["committed"]["replay_rows"] == 9 * n
        assert row["budget"]["replay"] == 0
        assert (endpoint["native_updates"], endpoint["native_rows"]) == (2 + n, 18 + 18 * n)
        assert endpoint["recovery"] is None and endpoint["measures"]["work_complete"]
        assert endpoint["measures"]["execution_seconds"] >= row["training_seconds"]


@pytest.mark.parametrize("recipe", ["other", True, 1, (), {"recipe_id": validation.RECIPE}])
def test_should_refuse_undeclared_recipe_before_model_work(monkeypatch, recipe):
    h = Harness(monkeypatch)
    before = old.WORK["owners"]
    with pytest.raises(ValueError, match="declared"):
        h.run_validation(recipe=recipe)
    assert old.WORK["owners"] == before and h.train_calls == 0 and not h.games


@pytest.mark.parametrize("role", ["final_test", "probe_train", "probe_test"])
def test_should_refuse_nontrain_sources_before_model_work(monkeypatch, role):
    h = Harness(monkeypatch)
    packets = list(old.sources())
    packets[2] = replace(packets[2], role=role)
    before = old.WORK["owners"]
    r = h.run_validation(episodes=tuple(packets))
    assert r["status"] == "failed" and old.WORK["owners"] == before
    assert h.train_calls == 0 and not h.games


def test_should_stop_after_failed_candidate_without_any_validation_readouts(monkeypatch):
    h = Harness(monkeypatch)

    def lose(owner):
        if h.cohort is not None:
            raise RuntimeError("lost validation update")

    h.on_train = lose
    r = h.run_validation()
    v = r["validation"]
    assert r["status"] == v["status"] == "failed"
    assert not v["endpoint_reads_released"] and "endpoints" not in v
    assert len(v["candidates"]) == 2 and v["candidates"][0]["status"] == "training_complete"
    assert not h.endpoint_train_counts and v["candidates"][1]["budget"]["updates"] == 0
    assert v["candidates"][1]["window"]["attempted_wakes"] == 1
    assert v["candidates"][1]["window"]["unreturned_work_unknown"]


@pytest.mark.parametrize("damage", ["fit", "budget", "prior", "replacement"])
def test_should_refuse_changed_evidence_before_more_validation_work(monkeypatch, damage):
    h = Harness(monkeypatch)
    changed: list[bool] = []

    def boundary():
        v = h.cohort
        if v is None or changed or not v["candidates"]:
            return
        row = v["candidates"][0]
        if row.get("status") != "training_complete":
            return
        changed.append(True)
        if damage == "fit":
            object.__setattr__(h.fit_owner._fitted, "fitted_at", 99)
        elif damage == "budget":
            h.budget.updates_completed = 1
        elif damage == "prior":
            row["budget"]["updates"] = 99
        else:
            v["candidates"][0] = deepcopy(row)

    r = h.run_validation(boundary=boundary)
    assert changed and r["status"] == "failed" and not h.endpoint_train_counts
    assert r["validation"]["status"] == "failed"
    assert not r["validation"]["endpoint_reads_released"]


def test_should_preserve_failed_game_prefix_and_unknown_readout_charge(monkeypatch):
    h = Harness(monkeypatch)
    original = learning._Study.game

    def lose(study, state, **kwargs):
        if h.cohort is study.report:
            study.game_attempts += 1
            raise RuntimeError("lost validation game return")
        return original(study, state, **kwargs)

    monkeypatch.setattr(learning._Study, "game", lose)
    r = h.run_validation()
    v = r["validation"]
    assert r["status"] == v["status"] == "failed" and v["endpoint_reads_released"]
    assert len(v["candidates"]) == 5 and len(v["endpoints"]) == 2
    assert v["endpoints"][0]["status"] == "complete"
    assert v["game_work"] == dict(attempts=1, returned=0, unreturned_work_unknown=True)
    assert len(v["budgets"]) == 5


@pytest.mark.parametrize("cues", [(0, 0, 1, 1), (0, 1, 0, 1), (True, 0, 1, 1)])
def test_should_check_declared_cue_prefix_in_real_readout_admission(integrated, monkeypatch, cues):
    _, h = integrated
    state = next(
        s for s in h.states if s.completed_updates == 4 and [x.fit_rows for x in s.heads] == [18, 9]
    )
    with ProcessRssSampler(interval_seconds=0.01) as memory:
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=0,
                max_replay_examples=0,
                max_wall_seconds=150,
                max_process_rss_bytes=512 * 1024**2,
            ),
            lambda: h.time,
        )
        budget.attach_memory(memory)
        cold = bank.BankLiteral()
        if cues == (0, 0, 1, 1) and type(cues[0]) is int:
            result = h.game(
                cold,
                state,
                budget=budget,
                check_boundary=lambda: None,
                reference_window=True,
                reference_catalogue=True,
                reference_catalogue_cues=cues,
            )
            assert result["reference_catalogue_cues"] == list(cues)
        else:
            with pytest.raises(ValueError):
                h.game(
                    cold,
                    state,
                    budget=budget,
                    check_boundary=lambda: None,
                    reference_window=True,
                    reference_catalogue=True,
                    reference_catalogue_cues=cues,
                )


def test_should_record_unknown_validation_setup_without_claiming_no_work(monkeypatch):
    h = Harness(monkeypatch)
    original = learning._Study.__init__
    calls = []

    def init(owner, *args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("lost validation setup return")
        original(owner, *args, **kwargs)

    monkeypatch.setattr(learning._Study, "__init__", init)
    r = h.run_validation()
    assert r["status"] == r["validation"]["status"] == "failed"
    assert r["validation"]["setup_work"] == dict(
        attempts=1, returned=0, unreturned_work_unknown=True
    )
    assert len(r["trials"]) == 4 and not h.endpoint_train_counts
