"""Typed probe game admission; scripted native ports and fake physics only."""

from copy import deepcopy
from dataclasses import replace
import pickle
from typing import Any
from types import SimpleNamespace

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_probe_rollout as live
from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.app.coupled_attribution import _same_value
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.actor_shadow import StableActor
from src.core.demonstration_episode import episode_arrays
from test_public_cue_recurrent_learner import configure_scripted_native, ScriptedNative, data
from test_minigrid_learning_pilot import Environment


def saved(owner, phase="after_probe"):
    n = 9 if phase == "after_probe" else 17
    cold = owner.snapshot_state()
    return replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=n, training_examples=9 * n),
            replace(cold.heads[1], completed_updates=n, training_examples=9 * n),
        ),
    )


def limits():
    return ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0, max_wall_seconds=40.0),
        lambda: 1.0,
    )


class CueEnvironment(Environment):
    def __init__(self, layout, cue, **options):
        super().__init__(layout, cue, **options)
        self.observation = tuple(float(v) for v in data(cue)[0][0])

    @property
    def report(self):
        return dict(
            super().report,
            closed=self.closed,
            role=self.role,
            seed=self.seed,
            max_steps=self.max_steps,
        )


@pytest.fixture
def scripted(monkeypatch):
    configure_scripted_native(monkeypatch)
    work: dict[str, Any] = {}
    monkeypatch.setattr(Environment, "work", work)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", CueEnvironment)
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)
    return work


@pytest.mark.parametrize(
    "phase,contexts,role,count",
    [
        ("after_probe", ["F-blue", "F-red"], "probe_test", 18),
        ("after_return", ["A", "A"], "train", 34),
    ],
)
def test_should_bind_only_declared_paid_state_and_target_free_plan(
    scripted, monkeypatch, phase, contexts, role, count
):
    owner = port.PublicCueRecurrentLearner()
    state = saved(owner, phase)
    owner.restore_state(state)
    seen = []

    def kernel(*a, **k):
        seen.append((a, k))
        return k

    monkeypatch.setattr(evaluation, "_run_checkpoint", kernel)
    result = evaluation.evaluate_public_cue_probe_checkpoint(
        owner, state, source_updates=2, phase=phase, budget=limits(), check_boundary=lambda: None
    )
    assert result["sources"] == () and result["public_history"] is False
    assert result["updates"] == result["return_metadata"]["native_training_updates"] == count
    assert result["return_metadata"]["native_head_updates"] == [count // 2] * 2
    assert [(l.context, q, r) for l, q, r in result["live_plan"]] == [
        (c, q, role) for c, q in zip(contexts, ("blue", "red"), strict=True)
    ]
    assert result["memory_copier"](owner.snapshot_state()).completed_updates == count
    changed = deepcopy(state)
    changed.heads[0].parameters["marker"][0] += 1
    with pytest.raises(ValueError, match="differs"):
        result["memory_copier"](changed)
    assert ScriptedNative.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "origin",
        "bool_origin",
        "phase",
        "counter",
        "head",
        "rows",
        "policy",
        "failed",
        "budget",
        "guard",
    ],
)
def test_should_refuse_unpaid_or_wrong_endpoint_before_kernel(scripted, monkeypatch, fault):
    owner = port.PublicCueRecurrentLearner()
    state = saved(owner)
    origin, phase, budget = 2, "after_probe", limits()
    guard: Any = lambda: None
    if fault == "origin":
        origin = 0
    if fault == "bool_origin":
        origin = True
    if fault == "phase":
        phase = "final"
    if fault == "counter":
        state = saved(owner, "after_return")
    if fault == "head":
        state = replace(
            state,
            heads=(
                replace(state.heads[0], completed_updates=8, training_examples=72),
                replace(state.heads[1], completed_updates=10, training_examples=90),
            ),
        )
    if fault == "rows":
        state = replace(
            state, heads=(replace(state.heads[0], training_examples=82), state.heads[1])
        )
    if fault == "policy":
        state = replace(
            state,
            policy=replace(
                state.policy, native_policy=replace(state.policy.native_policy, hidden_dim=2)
            ),
        )
    if fault == "failed":
        state = replace(state, failed_work=port.FailureWork("train", 0))
    if fault == "budget":
        budget = ToyBudgetSession(
            ToyExecutionBudget(
                max_training_updates=1, max_replay_examples=0, max_wall_seconds=40.0
            ),
            lambda: 1.0,
        )
    if fault == "guard":
        guard = None
    monkeypatch.setattr(
        evaluation, "_run_checkpoint", lambda *a, **k: pytest.fail("no native kernel")
    )
    with pytest.raises(ValueError):
        evaluation.evaluate_public_cue_probe_checkpoint(
            owner, state, source_updates=origin, phase=phase, budget=budget, check_boundary=guard
        )
    assert ScriptedNative.calls == []


@pytest.mark.parametrize(
    "values",
    [
        dict(source_updates=2),
        dict(source_updates=0, public_cue=True),
        dict(source_updates=128, public_cue=True),
        dict(source_updates=2, public_cue=1),
        dict(source_updates=2, public_cue=True, wide_method="bp"),
        dict(source_updates=2, public_cue=True, extended_wide_probe_budget=True),
    ],
)
def test_should_require_separate_exact_public_cue_mode(values):
    with pytest.raises(ValueError):
        live.ProbeRollouts(**values)


def test_should_play_both_phase_plans_with_fresh_sessions_and_original_zero_work_budgets(scripted):
    owner = port.PublicCueRecurrentLearner()
    observer = live.ProbeRollouts(2, public_cue=True)
    for phase in live.PHASES:
        owner.restore_state(saved(owner, phase))
        before = owner.snapshot_state()
        observer.observe(owner, phase, lambda: None)
        assert _same_value(owner.snapshot_state(), before)
    observer.check()
    assert [r["status"] for r in observer.records] == ["complete", "complete"]
    assert all(r["charged_updates"] == r["replay_examples"] == 0 for r in observer.records)
    assert [(e.layout.context, e.cue, e.role) for e in scripted["environments"]] == [
        ("F-blue", "blue", "probe_test"),
        ("F-red", "red", "probe_test"),
        ("A", "blue", "train"),
        ("A", "red", "train"),
    ]
    assert all(e.closed and e.actions == [0, 0] for e in scripted["environments"])
    assert len(ScriptedNative.calls) == 0  # Native scripted sessions use their own predict port.
    for row in observer.records:
        assert row["report"]["work"]["live_sessions_started"] == 2
        assert row["report"]["work"]["raw_score_forwards_completed"] == 0


@pytest.mark.parametrize("fault", ["replace", "value", "config"])
def test_should_refuse_prior_paid_corruption_before_admitting_return(scripted, fault):
    owner = port.PublicCueRecurrentLearner()
    owner.restore_state(saved(owner))
    observer = live.ProbeRollouts(2, public_cue=True)
    observer.observe(owner, "after_probe", lambda: None)
    original = pickle.dumps(observer.records)
    count = len(scripted["environments"])
    if fault == "replace":
        observer.records[0]["report"] = deepcopy(observer.records[0]["report"])
    if fault == "value":
        observer.records[0]["report"]["episodes"][0]["return"] += 1
    if fault == "config":
        observer.public_cue = False
    with pytest.raises(ValueError):
        observer.observe(owner, "after_return", lambda: None)
    assert len(scripted["environments"]) == count
    observer.restore_paid()
    assert pickle.dumps(observer.records) == original


def test_should_keep_partial_failed_game_and_refuse_any_next_phase(scripted):
    scripted["step_error"] = True
    owner = port.PublicCueRecurrentLearner()
    owner.restore_state(saved(owner))
    observer = live.ProbeRollouts(2, public_cue=True)
    with pytest.raises(RuntimeError):
        observer.observe(owner, "after_probe", lambda: None)
    assert observer.records[0]["status"] == "failed"
    assert observer.records[0]["report"]["work"]["failure_work_may_be_unobserved"] is True
    assert observer.records[0]["charged_updates"] == observer.records[0]["replay_examples"] == 0
    count = len(scripted["environments"])
    with pytest.raises(ValueError, match="preceding successful"):
        observer.observe(owner, "after_return", lambda: None)
    assert len(scripted["environments"]) == count and scripted["environments"][0].closed


def test_should_refuse_live_source_not_equal_to_its_paid_snapshot_before_any_game(scripted):
    owner = port.PublicCueRecurrentLearner()
    state = saved(owner)
    with pytest.raises(evaluation.MiniGridEvaluationFailed):
        evaluation.evaluate_public_cue_probe_checkpoint(
            owner,
            state,
            source_updates=2,
            phase="after_probe",
            budget=limits(),
            check_boundary=lambda: None,
        )
    assert scripted.get("environments", []) == [] and ScriptedNative.calls == []


def test_should_compose_real_generic_probe_observer_and_return_without_deployed_updates(scripted):
    from test_public_cue_fresh_task_probe import cohorts

    def nine_rows(batch):
        assert batch.episode is not None
        original = batch.episode
        episode = replace(
            original,
            transitions=tuple(
                replace(original.transitions[0 if i == 0 else 1], step=i, terminal=i == 8)
                for i in range(9)
            ),
        )
        x, y = episode_arrays(episode)
        return replace(
            batch,
            episode=episode,
            row_ids=tuple(f"{episode.episode_id}:step:{i}" for i in range(9)),
            features=x,
            targets=y,
        )

    training, evaluation_batches, monitors, templates = tuple(
        tuple(nine_rows(b) for b in group) for group in cohorts(16)
    )
    owner = port.PublicCueRecurrentLearner()
    cold = owner.snapshot_state()
    warm = replace(
        cold,
        heads=(
            replace(cold.heads[0], completed_updates=1, training_examples=9),
            replace(cold.heads[1], completed_updates=1, training_examples=9),
        ),
    )
    owner.restore_state(warm)
    actor = StableActor(owner, version="deployed")
    actor_before = actor.snapshot_state()

    def guard():
        assert _same_value(owner.snapshot_state(), warm)
        assert _same_value(actor.snapshot_state(), actor_before)

    probe_budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_replay_examples=0, max_wall_seconds=30),
        lambda: 1.0,
    )
    return_budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=16, max_replay_examples=0, max_wall_seconds=30),
        lambda: 1.0,
    )
    observer = live.ProbeRollouts(2, public_cue=True)
    result = run_fresh_task_probe(
        owner,
        warm,
        training,
        evaluation_batches,
        budget=probe_budget,
        checkpoint_updates=(0, 4, 8, 16),
        training_readouts=monitors,
        return_training=templates,
        return_budget=return_budget,
        check_boundary=guard,
        endpoint_observer=observer.observe_qualified,
        timer=lambda: 1.0,
    )
    assert result["status"] == result["return_phase"]["status"] == "complete", result.get("reason")
    assert probe_budget.updates_completed == return_budget.updates_completed == 16
    assert [r["report"]["native_training_updates"] for r in observer.records] == [18, 34]
    assert [r["report"]["native_head_updates"] for r in observer.records] == [[9, 9], [17, 17]]
    assert len(scripted["environments"]) == 4 and all(e.closed for e in scripted["environments"])
    assert sum(c[0] == "train" for c in ScriptedNative.calls) == 32
    observer.check()
    guard()


def test_should_preserve_old_wide_dispatch_and_both_parent_callback_forms(monkeypatch):
    from test_minigrid_probe_rollout import report

    seen = []
    monkeypatch.setattr(live, "monotonic", lambda: 1.0)
    monkeypatch.setattr(
        live, "_copy_wide_neural_probe_snapshot", lambda value, **kw: deepcopy(value)
    )
    monkeypatch.setattr(live, "_same_state", lambda a, b: a == b)

    def evaluate(owner, state, **options):
        seen.append(options)
        options["qualified_boundary"](lambda: None)
        return report()

    monkeypatch.setattr(live, "evaluate_wide_neural_probe_checkpoint", evaluate)
    owner = SimpleNamespace(
        snapshot_state=lambda: SimpleNamespace(completed_updates=16, training_examples=144)
    )

    def boundary(read_ports=None, *, after_clocks=None):
        if read_ports is not None:
            read_ports()
        if after_clocks is not None:
            after_clocks()

    for qualified in (False, True):
        observer = live.ProbeRollouts(0, wide_method="bp")
        if qualified:
            observer.observe_qualified(owner, "after_probe", boundary)
        else:
            observer.observe(owner, "after_probe", lambda: None)
        observer.check()
        assert observer.records[0]["status"] == "complete"
    assert len(seen) == 2 and all(o["budget"].budget.max_wall_seconds == 40 for o in seen)
