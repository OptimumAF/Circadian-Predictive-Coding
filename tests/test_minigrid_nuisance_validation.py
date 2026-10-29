"""Separate validation sources and exact readout domains; all work is scripted."""

from copy import deepcopy
from dataclasses import asdict
import pickle
import random
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_controller_learning as learning
from scripts import minigrid_controller_validation as validation
from src.adapters import minigrid_context_linear_learner as native
from src.adapters import minigrid_policy_evaluation as evaluation
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from src.core.minigrid_key_door import MiniGridKeyDoorLayout
from tests import test_minigrid_controller_learning as old
from tests import test_minigrid_controller_validation as existing
from tests.test_arrived_cue_catalogue import episode

WORK = dict(collections=0, steps=0, closes=0)


class Harness(existing.Harness):
    def __init__(self, patch):
        super().__init__(patch)
        self.collections: list[Any] = []
        self.nuisance_states = []
        self.on_collect = lambda world: None
        self.bad_readout = False
        self.observed_fit = None
        self.collection_report: Any = None
        self.collection_controller: Any = None
        self.validation_study: Any = None
        self.check_rng = True
        original_prepare = validation.prepare_nuisance_sources

        def prepare(study, controller, report):
            self.collection_report = report
            self.collection_controller = controller
            return original_prepare(study, controller, report)

        patch.setattr(validation, "prepare_nuisance_sources", prepare)
        original_validation = validation.run_validation

        def run(study, controller, recipe, **options):
            self.validation_study = study
            return original_validation(study, controller, recipe, **options)

        patch.setattr(validation, "run_validation", run)
        from src.infra import minigrid_key_door as world_port
        from src.core import minigrid_key_door as teacher_port

        harness = self

        class Teacher:
            def choose_action(self, x):
                return 2

        class World:
            def __init__(self, layout, cue, *, seed, max_steps, role, episode_id):
                WORK["collections"] += 1
                assert layout == MiniGridKeyDoorLayout("A", wall_color="red")
                assert seed == 17 and max_steps == 16 and role == "train"
                # Collection must occur after the actual four-trial freeze.
                assert harness.fit_owner is None  # run_validation has not begun yet
                assert harness.train_calls == 10  # warm2 + two forced wake trials4each
                self.layout, self.cue = layout, cue
                self.identity, self.closed = episode_id, False
                self.rows: list[Any] = []
                self.observation = (
                    episode("temporary", 2 if cue == "blue" else 0).transitions[0].observation[:148]
                )
                self.terminal = False
                harness.collections.append(self)
                harness.on_collect(self)

            def step(self, action):
                WORK["steps"] += 1
                self.terminal = len(self.rows) == 8
                self.rows.append(
                    dict(
                        observation=self.observation,
                        next_observation=self.observation,
                        action=action,
                        applied=True,
                        reward=0.5 if self.terminal else 0.0,
                        terminated=self.terminal,
                    )
                )

            def to_demonstration(self):
                return DemonstrationEpisode(
                    self.identity,
                    "A",
                    "train",
                    tuple(
                        DemonstrationTransition(
                            i,
                            r["observation"],
                            r["action"],
                            r["reward"],
                            r["next_observation"],
                            r["terminated"],
                            7,
                        )
                        for i, r in enumerate(self.rows)
                    ),
                    7,
                )

            def close(self):
                WORK["closes"] += 1
                self.closed = True

            @property
            def report(self):
                return dict(
                    episode_id=self.identity,
                    context="A",
                    role="train",
                    cue=self.cue,
                    layout=asdict(self.layout),
                    status="complete",
                    closed=self.closed,
                    terminated=self.terminal,
                    truncated=False,
                    seed=17,
                    max_steps=16,
                    failure_work_may_be_unobserved=False,
                    environment_steps=len(self.rows),
                    environment_steps_attempted=len(self.rows),
                    transitions=deepcopy(self.rows),
                    total_return=0.5,
                )

        patch.setattr(world_port, "MiniGridKeyDoor", World)
        patch.setattr(teacher_port, "CueKeyDoorTeacher", Teacher)

    def game(self, cold, state, **kwargs):
        if not kwargs.get("reference_nuisance"):
            return super().game(cold, state, **kwargs)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(native, "ContextLinearLearner", existing.bank.BankLiteral)
            patch.setattr(evaluation, "_run_checkpoint", self.nuisance_checkpoint)
            return evaluation.evaluate_context_linear_stream_checkpoint(cold, state, **kwargs)

    def nuisance_checkpoint(self, source, state, **kwargs):
        old.WORK["games"] += 1
        kwargs["extra_check"]()
        kwargs["check_boundary"]()
        self.require_release(None)
        plan = kwargs["live_plan"]
        assert tuple((p.branch_row, p.wall_color, cue, role) for p, cue, role in plan) == (
            (3, "grey", "blue", "train"),
            (3, "grey", "red", "train"),
            (3, "red", "blue", "train"),
            (3, "red", "red", "train"),
        )
        self.nuisance_states.append(deepcopy(state))
        self.time += 0.1
        rows = []
        for layout, cue, role in plan:
            rows.append(
                dict(
                    status="complete",
                    source_role=role,
                    cue=cue,
                    success=True,
                    return_value=1.0,
                    predictions=[dict(action=2)],
                    environment=dict(
                        episode_id=evaluation._live_episode_id(
                            kwargs["method"], kwargs["updates"], layout, cue, kwargs["role"]
                        ),
                        role=role,
                        cue=cue,
                        layout=asdict(layout),
                        seed=17,
                        max_steps=16,
                        closed=True,
                        failure_work_may_be_unobserved=False,
                        environment_steps=1,
                        transitions=[dict(action=2, applied=True)],
                    ),
                )
            )
            rows[-1]["return"] = rows[-1].pop("return_value")
        if self.bad_readout:
            rows[-1]["environment"]["layout"]["branch_row"] = 4
        result = dict(
            status="complete", state_unchanged=True, episodes=rows, **kwargs["return_metadata"]
        )
        kwargs["check_boundary"]()
        return result

    def run_nuisance(self, **options):
        from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
        from src.shared.process_memory import ProcessRssSampler

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
            sources = old.sources()
            before = pickle.dumps((sources, random.getstate(), np.random.get_state()))
            result = learning.run_controller_learning(
                sources,
                readout_budget=self.budget,
                retain_arrived=True,
                validation_recipe=validation.NUISANCE_RECIPE,
                **options,
            )
            if self.check_rng:
                assert pickle.dumps((sources, random.getstate(), np.random.get_state())) == before
            assert self.budget.updates_completed == self.budget.replay_examples_completed == 0
            return result


@pytest.fixture(scope="module")
def integrated():
    with pytest.MonkeyPatch.context() as patch:
        h = Harness(patch)
        result = h.run_nuisance(share_readouts=True)
        assert result["status"] == "complete", result.get("error")
        return result, h


def test_should_preserve_fitting_and_collect_once_after_freeze(integrated):
    r, h = integrated
    assert len(r["trials"]) == r["controller"]["completed"] == 4
    assert len(h.collections) == 2 and all(w.closed for w in h.collections)
    v = r["validation"]
    assert v["collection_work"] == dict(
        attempts=1, returned=1, admitted=True, unreturned_work_unknown=False
    )
    assert len(v["collection"]) == 2 and v["frozen_fit_unchanged"]
    assert r["base"]["state"]["completed_updates"] == 2
    assert [c["mode"] for c in v["candidates"]] == list(validation.MODES)


def test_should_bind_separate_sources_and_match_all_readout_layouts(integrated):
    r, h = integrated
    v = r["validation"]
    for row in v["candidates"]:
        assert row["source_episode_ids"] == [
            "minigrid-source-A3-redwall-train-blue",
            "minigrid-source-A-blue",
            "minigrid-source-A3-redwall-train-red",
            "minigrid-source-A-red",
        ]
        assert row["source_unchanged"] and row["actor_unchanged"]
    assert h.nuisance_states and h.endpoint_train_counts
    assert set(h.endpoint_train_counts) == {h.train_calls}
    assert v["baseline"]["games"] is not r["baseline"]
    assert all(e["before"] == dict(old_A3=2, shifted_A3=2) for e in v["endpoints"])
    assert all("new_A4" not in e["after"] for e in v["endpoints"])


def test_should_keep_reused_endpoint_costs_explicit(integrated):
    r, _ = integrated
    for endpoint in r["validation"]["endpoints"]:
        if endpoint["reused_from"] is not None:
            assert endpoint["utility"] is endpoint["measures"] is None
            assert endpoint["readout_work"]["physical_game_calls"] == 0
    assert len(r["validation"]["endpoints"]) == 5


def test_should_keep_nuisance_baseline_reuse_honest_without_state_cache(monkeypatch):
    h = Harness(monkeypatch)
    r = h.run_nuisance()
    assert r["status"] == "complete", r.get("error")
    v = r["validation"]
    assert v["baseline"]["readout_seconds"] >= 0
    for endpoint in v["endpoints"]:
        if endpoint["reused_from"] is not None:
            assert endpoint["utility"] is endpoint["measures"] is None


def test_should_refuse_late_collection_corruption_and_restore_paid_prefix(monkeypatch):
    h = Harness(monkeypatch)
    changed: list[bool] = []

    def damage(owner):
        if h.cohort is not None and not changed:
            changed.append(True)
            h.collection_report["collection"][0]["environment_steps"] = 0

    h.on_train = damage
    r = h.run_nuisance()
    assert r["status"] == "failed" and changed
    assert r["validation"]["collection"][0]["environment_steps"] == 9
    assert not h.nuisance_states
    assert r["validation"]["endpoint_reads_released"] is False


def test_should_keep_known_recipe_description_unchanged():
    d = validation.describe_validation(validation.RECIPE)
    assert d["recipe_id"] == "interleaved_delay2" and "wall_colors" not in d
    assert "same known TRAIN" in d["endpoint"]
    n = validation.describe_validation(validation.NUISANCE_RECIPE)
    assert n["wall_colors"] == ["grey", "red"] and n["layout"]["branch_row"] == 3


def test_should_preserve_original_a4_collection_wrapper(monkeypatch):
    from scripts import minigrid_affine_workflow as workflow
    from tests.test_a4_fresh_workflow import bind_collection

    box = bind_collection(monkeypatch)
    records: list[dict[str, Any]] = []
    packets = workflow._collect_recurring_train_sources(
        lambda: None, records, layout=MiniGridKeyDoorLayout("A", branch_row=4)
    )
    assert len(packets) == len(records) == 2
    assert packets[0].episode_id == "minigrid-source-A4-train-blue"
    assert all(e.closed for e in box["environments"])


def test_should_keep_correct_acknowledgement_separate_from_optimizer_work(monkeypatch):
    h = Harness(monkeypatch)
    r = h.run_nuisance(acknowledge_correct=True, share_readouts=True)
    assert r["status"] == "complete", r.get("error")
    for row in r["validation"]["candidates"]:
        window = row["window"]
        assert row["budget"]["updates"] == len(window["applied_packet_indices"])
        assert window["processed_packets"] == (
            row["budget"]["updates"] + len(window["acknowledged_experiences"])
        )
    periodic = r["validation"]["candidates"][1]
    assert periodic["budget"]["updates"] == 2
    assert len(periodic["window"]["acknowledged_experiences"]) == 2
    assert r["validation"]["protocol"]["acknowledge_correct"] is True


def test_should_refuse_wrong_layout_in_returned_nuisance_game(monkeypatch):
    h = Harness(monkeypatch)
    h.bad_readout = True
    r = h.run_nuisance()
    assert r["status"] == "failed" and "phase evidence" in r["error"]
    assert r["validation"]["status"] == "failed"
    assert r["validation"]["endpoint_reads_released"]
    assert r["validation"]["baseline"]["status"] == "failed"
    assert r["validation"]["endpoints"] == []


def test_should_close_and_record_a_collection_failure_without_validation_models(monkeypatch):
    h = Harness(monkeypatch)

    def fail(world):
        if world.cue == "red":
            raise RuntimeError("second nuisance construction lost")

    h.on_collect = fail
    r = h.run_nuisance()
    assert r["status"] == "failed" and "construction lost" in r["error"]
    v = r["validation"]
    assert v["collection_work"]["returned"] == 0 and v["collection_work"]["unreturned_work_unknown"]
    assert v["collection"][0]["closed"] is True
    assert v["collection"][1]["status"] == "construction_failed"
    assert v["candidates"] == [] and h.cohort is None


@pytest.mark.parametrize("replacement", [False, True])
def test_should_refuse_corrupted_paid_collection_before_more_teacher_work(monkeypatch, replacement):
    h = Harness(monkeypatch)

    def damage(world):
        if world.cue == "red":
            records = h.collection_report["collection"]
            if replacement:
                records[0] = deepcopy(records[0])
            else:
                records[0]["environment_steps"] = 0

    h.on_collect = damage
    r = h.run_nuisance()
    assert r["status"] == "failed"
    assert h.train_calls == 10 and h.nuisance_states == []
    assert len(h.collections[1].rows) == 0 and h.collections[1].closed
    assert r["validation"]["collection_work"]["returned"] == 0
    assert r["validation"]["collection"][0]["environment_steps"] == 9


def test_should_refuse_a_changed_frozen_chooser_during_source_collection(monkeypatch):
    h = Harness(monkeypatch)

    def damage(world):
        owner = h.collection_controller
        original = owner.application_snapshot
        monkeypatch.setattr(owner, "application_snapshot", lambda: dict(original(), altered=True))

    h.on_collect = damage
    r = h.run_nuisance()
    assert r["status"] == "failed" and "frozen chooser changed" in r["error"]
    assert h.train_calls == 10 and not h.nuisance_states
    assert h.collections[0].closed and not h.collections[0].rows


def test_should_refuse_global_rng_changes_before_teacher_work(monkeypatch):
    h = Harness(monkeypatch)
    h.check_rng = False
    original = np.random.get_state()
    h.on_collect = lambda world: np.random.seed(99)
    try:
        r = h.run_nuisance()
        assert r["status"] == "failed" and "global RNG changed" in r["error"]
        assert h.train_calls == 10 and not h.nuisance_states
        assert h.collections[0].closed and not h.collections[0].rows
    finally:
        np.random.set_state(original)


@pytest.mark.parametrize(
    "recipe,flag", [(validation.NUISANCE_RECIPE, False), (validation.RECIPE, True)]
)
def test_should_refuse_recipe_source_domain_mismatch_before_any_work(recipe, flag):
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="domain differ"):
        validation.run_validation(SimpleNamespace(nuisance=flag), None, recipe, baseline=None)


def test_should_refuse_fitting_domain_games_as_nuisance_baseline(integrated):
    result, _ = integrated
    with pytest.raises(ValueError, match="phase evidence"):
        validation._counts(result["baseline"], nuisance=True)


def test_should_bind_domain_flag_to_original_source_identity(monkeypatch):
    h = Harness(monkeypatch)
    changed: list[bool] = []

    def damage(owner):
        if h.cohort is not None and not changed:
            changed.append(True)
            h.validation_study.nuisance = h.validation_study.nuisance_pin = False

    h.on_train = damage
    result = h.run_nuisance()
    assert result["status"] == "failed" and changed
    assert h.train_calls == 11 and not h.nuisance_states


@pytest.mark.parametrize("flag", [True, 1, None, "red"])
def test_should_refuse_unbound_nuisance_readout_before_episode_io(monkeypatch, flag):
    from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

    monkeypatch.setattr(native, "ContextLinearLearner", existing.bank.BankLiteral)
    cold = existing.bank.BankLiteral()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=0), lambda: 0.0)
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **kw: pytest.fail("episode IO reached")
    )
    with pytest.raises(ValueError, match="nuisance"):
        evaluation.evaluate_context_linear_stream_checkpoint(
            cold,
            cold.snapshot_state(),
            budget=budget,
            check_boundary=lambda: None,
            reference_nuisance=flag,
        )


@pytest.mark.parametrize("bad", [True, "nuisance_bluewall_delay2"])
def test_should_refuse_undeclared_recipe_before_models(monkeypatch, bad):
    monkeypatch.setattr(learning, "ContextLinearLearner", lambda: pytest.fail("model reached"))
    with pytest.raises(ValueError, match="declared"):
        learning.run_controller_learning(
            old.sources(), readout_budget=None, retain_arrived=True, validation_recipe=bad
        )


@pytest.mark.parametrize("updates", range(2, 7))
def test_should_distinguish_native_nuisance_episode_identities(updates):
    identities = [
        evaluation._live_episode_id(
            "arrived_public_cue_affine_fit",
            updates,
            MiniGridKeyDoorLayout("A", branch_row=3, wall_color=color),
            cue,
            "grey_red_A3_nuisance_TRAIN",
        )
        for color in ("grey", "red")
        for cue in ("blue", "red")
    ]
    assert len(set(identities)) == 4
    assert all(identity.isascii() and len(identity.encode()) <= 64 for identity in identities)
    assert identities == [
        f"live-arrived_public_cue_affine_fit-{updates}-A-row3-{color}-{cue}"
        for color in ("grey", "red")
        for cue in ("blue", "red")
    ]


def test_should_still_refuse_duplicate_nuisance_episode_identities(integrated):
    result, _ = integrated
    games = deepcopy(result["validation"]["baseline"]["games"])
    games["episodes"][2]["environment"]["episode_id"] = games["episodes"][0]["environment"][
        "episode_id"
    ]
    with pytest.raises(ValueError, match="identities must be distinct"):
        validation._counts(games, nuisance=True)
