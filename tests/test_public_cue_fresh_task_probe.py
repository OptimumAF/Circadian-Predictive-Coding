"""Direct generic-probe compatibility; scripted native ports, not learned tasks."""

from copy import deepcopy
from dataclasses import asdict, replace
import pickle

import numpy as np
import pytest

from src.adapters import minigrid_public_cue_recurrent_learner as port
from src.app.actor_shadow import ActorShadowRuntime
from src.app.coupled_attribution import AttributionBatch, _same_value
from src.app.fresh_task_probe import run_fresh_task_probe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import episode_arrays
from src.core.experience import LogicalClock
from test_public_cue_recurrent_learner import ScriptedNative, configure_scripted_native, episode


def packet(identity, role, cue, *, old=False):
    original = episode(cue, role)
    transitions = (
        original.transitions if old else tuple(replace(t, action=6) for t in original.transitions)
    )
    owned = replace(
        original, episode_id=identity, context="A" if old else "F-" + cue, transitions=transitions
    )
    x, y = episode_arrays(owned)
    return AttributionBatch(
        role, owned.context, tuple(f"{identity}:step:{i}" for i in range(2)), x, y, owned, 7
    )


def cohorts(count=4):
    training = tuple(
        packet(f"F-train-{i}", "probe_train", ("blue", "red")[i % 2]) for i in range(count)
    )
    evaluation = tuple(packet(f"F-test-{cue}", "probe_test", cue) for cue in ("blue", "red"))
    monitors = tuple(packet(f"A-monitor-{cue}", "train", cue, old=True) for cue in ("blue", "red"))
    templates = tuple(packet(f"A-return-{cue}", "train", cue, old=True) for cue in ("blue", "red"))
    return training, evaluation, monitors, templates


def budget(updates):
    return ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=updates, max_replay_examples=0, max_wall_seconds=30
        ),
        lambda: 0.0,
    )


@pytest.fixture
def environment(monkeypatch):
    configure_scripted_native(monkeypatch)
    records = []
    original_train, original_predict = ScriptedNative.train_batch, ScriptedNative.predict

    def train(self, x, y):
        records.append(("train", self, x.copy(), y.copy(), self.snapshot_state()))
        diagnostic = original_train(self, x, y)
        # Exercise ownership of all optimizer arrays, not just replaced parameters.
        self.state.gradients["gradient"] += 1
        self.state.optimizer_state["momentum"] += 2
        return diagnostic

    def predict(self, x):
        records.append(("predict", self, x.copy(), None, self.snapshot_state()))
        return original_predict(self, x)

    monkeypatch.setattr(ScriptedNative, "train_batch", train)
    monkeypatch.setattr(ScriptedNative, "predict", predict)
    root = port.PublicCueRecurrentLearner()
    cold = root.snapshot_state()
    heads = tuple(
        replace(
            h,
            completed_updates=i + 2,
            training_examples=2 * (i + 2),
            parameters={"marker": np.array([11.0 + i])},
            gradients={"gradient": np.array([21.0 + i])},
            optimizer_state={"momentum": np.array([31.0 + i])},
        )
        for i, h in enumerate(cold.heads)
    )
    root.restore_state(replace(cold, heads=(heads[0], heads[1])))
    benchmark_budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=8, max_wall_seconds=30), lambda: 0.0
    )
    runtime = ActorShadowRuntime(
        root,
        actor_version="deployed",
        candidate_version="benchmark",
        clock=LogicalClock(),
        budget=benchmark_budget,
    )
    source = runtime._candidate
    assert isinstance(source, port.PublicCueRecurrentLearner)
    state = source.snapshot_state()
    owners = (root._state, source._state, runtime.actor, runtime._candidate)
    expected = deepcopy(
        (root.snapshot_state(), source.snapshot_state(), runtime.actor.snapshot_state(), state)
    )
    limits = benchmark_budget.budget
    original_budget = (
        deepcopy(asdict(limits)),
        benchmark_budget.clock,
        benchmark_budget.started_at,
        benchmark_budget.progress,
        benchmark_budget.process_rss_sampler,
    )

    def guard():
        if (
            root._state is not owners[0]
            or source._state is not owners[1]
            or runtime.actor is not owners[2]
            or runtime._candidate is not owners[3]
            or not _same_value(
                (
                    root.snapshot_state(),
                    source.snapshot_state(),
                    runtime.actor.snapshot_state(),
                    state,
                ),
                expected,
            )
            or benchmark_budget.budget is not limits
            or asdict(limits) != original_budget[0]
            or benchmark_budget.clock is not original_budget[1]
            or benchmark_budget.started_at != original_budget[2]
            or benchmark_budget.progress is not original_budget[3]
            or benchmark_budget.process_rss_sampler is not original_budget[4]
            or benchmark_budget.updates_completed != 0
            or benchmark_budget.replay_examples_completed != 0
        ):
            raise ValueError(
                "original public-cue source/checkpoint/actor/benchmark authority changed"
            )

    children = []
    original_fork = port.PublicCueRecurrentLearner.fork

    def fork(self):
        child = original_fork(self)
        children.append(child)
        return child

    monkeypatch.setattr(port.PublicCueRecurrentLearner, "fork", fork)
    return dict(
        root=root,
        runtime=runtime,
        source=source,
        state=state,
        guard=guard,
        children=children,
        records=records,
        benchmark_budget=benchmark_budget,
    )


def run(env, *, data=None, return_to_a=False, check=True, observer=None):
    training, evaluation, monitors, templates = (
        cohorts(16 if return_to_a else 4) if data is None else data
    )
    fresh_budget = budget(len(training))
    return_budget = budget(16) if return_to_a else None
    before = pickle.dumps((training, evaluation, monitors, templates))
    rng = pickle.dumps(np.random.get_state())
    report = run_fresh_task_probe(
        env["source"],
        env["state"],
        training,
        evaluation,
        budget=fresh_budget,
        checkpoint_updates=(0, 4, 8, 16) if return_to_a else (0, 2, 4),
        timer=lambda: 0.0,
        decision_diagnostics=True,
        training_readouts=monitors,
        check_boundary=env["guard"] if check else None,
        **(dict(return_training=templates, return_budget=return_budget) if return_to_a else {}),
        **(dict(endpoint_observer=observer) if observer is not None else {}),
    )
    assert pickle.dumps((training, evaluation, monitors, templates)) == before
    assert pickle.dumps(np.random.get_state()) == rng
    return report, fresh_budget, return_budget


def test_should_use_existing_probe_with_owned_full_state_and_both_public_cue_heads(environment):
    env = environment
    data = cohorts()
    result, fresh, _ = run(env, data=data)
    assert result["status"] == "complete", result["reason"]
    assert len(env["children"]) == 1
    child = env["children"][0]
    assert (
        child is not env["source"]
        and child.completed_updates == 9
        and child.training_examples == 18
    )
    assert [h.completed_updates for h in child.snapshot_state().heads] == [4, 5]
    assert result["completed_updates"] == fresh.updates_completed == 4
    assert (
        result["completed_training_examples"] == 8 and result["score_forward_calls_completed"] == 6
    )
    assert result["training_readouts"]["score_forward_calls_completed"] == 6
    assert result["episode_scope"]["training_transitions"] == 8
    trained = [r for r in env["records"] if r[0] == "train"]
    assert [r[4].completed_updates for r in trained] == [2, 3, 3, 4]
    for record, batch in zip(trained, data[0], strict=True):
        assert _same_value((record[2], record[3]), (batch.features, batch.targets))
    assert len({id(r[1]) for r in env["records"]}) == len(env["records"])
    assert [p["updates"] for p in result["checkpoints"]] == [0, 2, 4]
    assert env["source"].completed_updates == 5 and fresh.replay_examples_completed == 0
    env["guard"]()


def test_should_keep_existing_fixed_f16_return16_on_one_disposable_owner(environment):
    env = environment
    data = cohorts(16)
    endpoints = []

    def observe(owner, phase, boundary):
        expected = owner.snapshot_state()

        def read():
            assert _same_value(expected, owner.snapshot_state())

        boundary(read, after_clocks=read)
        endpoints.append((phase, owner, expected))

    report, fresh, returned = run(env, data=data, return_to_a=True, observer=observe)
    assert report["status"] == "complete", report["reason"]
    phase = report["return_phase"]
    assert phase["status"] == "complete", phase["reason"]
    assert fresh.updates_completed == returned.updates_completed == 16
    assert report["completed_training_examples"] == phase["completed_training_examples"] == 32
    assert phase["checkpoints"][0]["scores"] == report["checkpoints"][-1]["scores"]
    assert (
        phase["training_readouts"]["checkpoints"][0]["scores"]
        == report["training_readouts"]["checkpoints"][-1]["scores"]
    )
    assert len(env["children"]) == 1
    final = env["children"][0].snapshot_state()
    assert final.completed_updates == 37 and final.training_examples == 74
    assert [h.completed_updates for h in final.heads] == [18, 19]
    assert [(phase, state.completed_updates) for phase, _, state in endpoints] == [
        ("after_probe", 21),
        ("after_return", 37),
    ]
    assert all(owner is env["children"][0] for _, owner, _ in endpoints)
    calls = [r for r in env["records"] if r[0] == "train"]
    assert len(calls) == 32
    for i, record in enumerate(calls):
        packet = data[0][i] if i < 16 else data[3][i % 2]
        assert _same_value((record[2], record[3]), (packet.features, packet.targets))
    assert all(np.all(r[3] == 6) for r in calls[:16])
    assert all(not np.all(r[3] == 6) for r in calls[16:])
    env["guard"]()


@pytest.mark.parametrize(
    "fault",
    [
        "training_final",
        "evaluation_final",
        "monitor_probe",
        "alias",
        "mixed_episode",
        "classes",
        "return_probe",
    ],
)
def test_should_refuse_bad_roles_or_data_before_fork_and_native_calls(environment, fault):
    env = environment
    training, evaluation, monitors, templates = cohorts(16 if fault == "return_probe" else 4)
    if fault == "training_final":
        training = (replace(training[0], role="final_test"), *training[1:])
    if fault == "evaluation_final":
        evaluation = (replace(evaluation[0], role="final_test"), evaluation[1])
    if fault == "monitor_probe":
        monitors = (replace(monitors[0], role="probe_test"), monitors[1])
    if fault == "alias":
        evaluation = (replace(evaluation[0], features=training[0].features), evaluation[1])
    if fault == "mixed_episode":
        evaluation = (replace(evaluation[0], episode=None), evaluation[1])
    if fault == "classes":
        training = (replace(training[0], num_classes=3), *training[1:])
    if fault == "return_probe":
        templates = (replace(templates[0], role="probe_train"), templates[1])
    with pytest.raises(ValueError):
        run(
            env,
            data=(training, evaluation, monitors, templates),
            return_to_a=fault == "return_probe",
        )
    assert env["children"] == env["records"] == []
    env["guard"]()


@pytest.mark.parametrize(
    "fault",
    ["source_gradient", "actor_momentum", "checkpoint_gradient", "budget", "source_replacement"],
)
def test_should_refuse_external_mutation_after_paid_native_success(environment, monkeypatch, fault):
    env = environment
    original = ScriptedNative.train_batch

    def mutate(self, x, y):
        result = original(self, x, y)
        if fault == "source_gradient":
            env["source"]._state.heads[0].gradients["gradient"] += 1
        if fault == "actor_momentum":
            env["runtime"].actor._learner._state.heads[0].optimizer_state["momentum"] += 1
        if fault == "checkpoint_gradient":
            env["state"].heads[0].gradients["gradient"] += 1
        if fault == "budget":
            env["benchmark_budget"].budget = replace(env["benchmark_budget"].budget)
        if fault == "source_replacement":
            env["source"]._state = deepcopy(env["source"]._state)
        return result

    monkeypatch.setattr(ScriptedNative, "train_batch", mutate)
    report, fresh, _ = run(env)
    assert report["status"] == "failed" and "original public-cue" in report["reason"]
    assert report["completed_updates"] == fresh.updates_completed == 1
    assert report["completed_training_examples"] == 2
    assert sum(r[0] == "train" for r in env["records"]) == 1
    assert len(report["checkpoints"]) == 1
    with pytest.raises(ValueError):
        env["guard"]()  # Rejected external changes are never silently restored.


def test_should_keep_preceding_probe_success_and_unknown_failed_native_work(
    environment, monkeypatch
):
    env = environment
    original = ScriptedNative.train_batch

    def fail(self, x, y):
        diagnostic = original(self, x, y)
        if sum(r[0] == "train" for r in env["records"]) == 2:
            raise RuntimeError("scripted native return lost")
        return diagnostic

    monkeypatch.setattr(ScriptedNative, "train_batch", fail)
    result, fresh, _ = run(env)
    assert result["status"] == "failed" and result["attempted_updates"] == 2
    assert result["completed_updates"] == fresh.updates_completed == 1
    assert result["completed_training_examples"] == 2 and len(result["diagnostics"]) == 1
    assert result["failed_native_work"]["work"]["committed_updates"] == 6
    assert result["failed_native_work"]["work"]["failure_work_may_be_unobserved"] is True
    assert env["children"][0].snapshot_state().failed_work is not None
    env["guard"]()


def test_should_leave_paid_f_report_intact_when_return_native_fails(environment, monkeypatch):
    env = environment
    original = ScriptedNative.train_batch

    def fail(self, x, y):
        diagnostic = original(self, x, y)
        if sum(r[0] == "train" for r in env["records"]) == 17:
            raise RuntimeError("scripted first return update lost")
        return diagnostic

    monkeypatch.setattr(ScriptedNative, "train_batch", fail)
    report, fresh, returned = run(env, return_to_a=True)
    assert report["status"] == "complete" and fresh.updates_completed == 16
    assert [p["updates"] for p in report["checkpoints"]] == [0, 4, 8, 16]
    phase = report["return_phase"]
    assert phase["status"] == "failed" and phase["attempted_updates"] == 1
    assert phase["completed_updates"] == returned.updates_completed == 0
    assert phase["failed_native_work"]["work"]["committed_updates"] == 21
    assert phase["failed_native_work"]["work"]["failure_work_may_be_unobserved"] is True
    env["guard"]()
