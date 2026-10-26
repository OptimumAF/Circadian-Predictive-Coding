"""Eight sampled TRAIN packets through the real actor/inbox reward runtime."""

from copy import deepcopy
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_reward_smoke as cli
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters.minigrid_reward_collection import MiniGridRewardCollectionFailed
from src.app.actor_shadow import ActorShadowRuntime
import src.app.toy_execution_budget as budgets
from src.core.demonstration_episode import DemonstrationTransition
from src.core.policy_gradient_episode import PolicyGradientEpisode
import src.shared.torch_runtime as torch_runtime
from test_key_door_reward_pilot import Learner as BaseLearner
from test_minigrid_reward_collection import frame
from test_recurrent_key_door_pilot import State
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


class Learner(BaseLearner):
    after_native = None

    def __init__(self, policy):
        self.policy = policy
        shapes = dict(
            weight_input=(148, 8),
            weight_recurrent=(8, 8),
            bias_hidden=(1, 8),
            weight_output=(8, 7),
            bias_output=(1, 7),
        )
        self.state = State(
            {key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()},
            {},
            {key: None for key in shapes},
        )
        BaseLearner.sources.append(self)

    def train_batch(self, update, identifier):
        result = super().train_batch(update, identifier)
        self.state.parameters["bias_output"] += 0.01
        self.state.optimizer_state["trace"] = self.state.completed_updates
        if Learner.after_native is not None:
            Learner.after_native(self)
        return result

    def predict(self, values):
        pytest.fail("reward-only route attempted an evaluation forward")

    def start_episode(self):
        pytest.fail("patched collection must own its own sampling session")


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(
        collections=[], polls=[], labels=[], experiences=[], runtimes=[], reports=[]
    )
    monkeypatch.setattr(pilot, "TorchRewardUpdateLearner", Learner)
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(pilot, "monotonic", lambda: 10.0)
    monkeypatch.setattr(
        pilot, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )
    monkeypatch.setattr(Learner, "after_native", None)
    for key in ("sessions", "updates", "identifiers", "classifications", "sources"):
        monkeypatch.setattr(BaseLearner, key, [])
    monkeypatch.setattr(BaseLearner, "fail_update", False)
    monkeypatch.setattr(BaseLearner, "fail_collection", False)
    factory = pilot._new_report

    def report_factory(*args, **kwargs):
        report = factory(*args, **kwargs)
        work["reports"].append(report)
        return report

    def runtime(source, **kwargs):
        actual = ActorShadowRuntime(source, **kwargs)
        work["runtimes"].append(actual)
        work["budget"], work["clock"] = kwargs["budget"], kwargs["clock"]
        poll, label, experience = actual.train_ready, actual.record_label, actual.record_experience

        def train_ready(**options):
            before = kwargs["budget"].updates_completed
            receipts = poll(**options)
            work["polls"].append((before, kwargs["budget"].updates_completed, receipts))
            if work.get("bad_receipt") and receipts:
                return (replace(receipts[0], update_number=99),)
            return receipts

        def record_label(value):
            work["labels"].append(value)
            if work.get("bad_target"):
                value = replace(value, targets="unrelated-release")
            return label(value)

        def record_experience(value):
            work["experiences"].append(value)
            return experience(value)

        monkeypatch.setattr(actual, "train_ready", train_ready)
        monkeypatch.setattr(actual, "record_label", record_label)
        monkeypatch.setattr(actual, "record_experience", record_experience)
        return actual

    def collect(source, layout, cue, *, episode_id, sampler, check_boundary, **options):
        index = len(work["collections"])
        call = dict(
            source=source,
            layout=layout,
            cue=cue,
            episode_id=episode_id,
            sampler=sampler,
            state=source.snapshot_state(),
        )
        work["collections"].append(call)
        callback = work.get("on_collection")
        if callback:
            callback(index, call)
        check_boundary()
        assert call["state"].completed_updates == index
        assert call["state"].training_examples == 2 * index
        assert source is not work["runtimes"][0]._candidate and source is not BaseLearner.sources[0]
        assert options.get("role", "train") == "train"
        failed = work.get("fail_collection") == index
        count = 1 if failed else 2
        before = deepcopy(sampler.bit_generator.state)
        probabilities = (0.0,) * 6 + (1.0,)
        predictions = [
            dict(
                observation=frame(step),
                probabilities=probabilities,
                uniform=float(sampler.random()),
                action=6,
            )
            for step in range(count)
        ]
        transitions = tuple(
            DemonstrationTransition(step, frame(step), 6, 0.0, frame(step + 1), False, 7)
            for step in range(count)
        )
        episode = PolicyGradientEpisode(
            episode_id,
            layout.context,
            "train",
            transitions,
            tuple(probabilities for _ in transitions),
            "truncated",
            7,
        )
        report = dict(
            status="failed" if failed else "complete",
            reason="third collection failed after one paid draw" if failed else None,
            episode=None if failed else episode,
            behavior_state=source.snapshot_state(),
            sampler_before=before,
            sampler_after=deepcopy(sampler.bit_generator.state),
            sampler_binding_preserved=True,
            state_unchanged=True,
            elapsed_seconds=0.0,
            predictions=predictions,
            environment=dict(
                status="failed" if failed else "complete",
                closed=True,
                context=layout.context,
                cue=cue,
                role="train",
                environment_steps_attempted=count,
                environment_steps=count,
                total_return=0.0,
                terminated=False,
                truncated=not failed,
                transitions=[
                    dict(
                        observation=row.observation,
                        next_observation=row.next_observation,
                        action=6,
                        reward=0.0,
                        terminated=False,
                        truncated=not failed and i == count - 1,
                        attempted=True,
                        applied=True,
                    )
                    for i, row in enumerate(transitions)
                ],
                failure_work_may_be_unobserved=False,
            ),
            work=dict(
                snapshots_attempted=2,
                snapshots_returned=2,
                sessions_attempted=1,
                sessions_started=1,
                environment_constructions_attempted=1,
                environment_constructions_returned=1,
                prediction_calls_attempted=count,
                prediction_calls_returned=count,
                sampling_draws=count,
                environment_steps_attempted=count,
                environment_steps_returned=count,
                close_calls_attempted=1,
                close_calls_returned=1,
                failure_work_may_be_unobserved=failed,
            ),
            close_error=None,
            readback_error=None,
        )
        call["report"] = report
        check_boundary()
        if failed:
            raise MiniGridRewardCollectionFailed(report)
        return report

    monkeypatch.setattr(pilot, "_new_report", report_factory)
    monkeypatch.setattr(pilot, "ActorShadowRuntime", runtime)
    monkeypatch.setattr(pilot, "collect_minigrid_reward_episode", collect)
    return SimpleNamespace(work=work)


def test_should_describe_fixed_reward_recipe_without_constructing_native_resources(monkeypatch):
    for name in (
        "require_torch",
        "TorchRewardUpdateLearner",
        "ProcessRssSampler",
        "collect_minigrid_reward_episode",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    protocol = pilot.describe_protocol()
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol
    assert protocol["protocol_id"] == "minigrid_cue_key_door_recurrent_reward_smoke_v1"
    assert protocol["methods"] == ["recurrent_rl"] and protocol["source_role"] == "train"
    assert (
        protocol["feature_dim"],
        protocol["hidden_dim"],
        protocol["num_classes"],
        protocol["seed"],
    ) == (148, 8, 7, 17)
    assert (
        protocol["learning_rate"],
        protocol["momentum"],
        protocol["discount"],
        protocol["sampling_seed"],
    ) == (0.03, 0.9, 1.0, 29)
    assert protocol["planned_work"]["native_train_calls"] == 8
    assert protocol["limits"]["body_seconds"] == 45
    assert protocol["limits"]["process_rss_bytes"] == 1024**3
    assert all(
        protocol["planned_work"][key] == 0
        for key in (
            "score_forwards",
            "teacher_episodes",
            "final_episodes",
            "probe_updates",
            "replay_updates",
            "promotions",
        )
    )


def test_should_release_exact_ids_only_after_empty_early_poll_and_preserve_full_state(case):
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    slots = report["training"]
    assert len(slots) == len(BaseLearner.updates) == len(case.work["collections"]) == 8
    assert [(slot["context"], slot["cue"]) for slot in slots] == [
        (context, cue) for context in ("A", "B", "C_train", "A") for cue in ("blue", "red")
    ]
    assert report["wake_updates"] == case.work["budget"].updates_completed == 8
    assert report["training_examples"] == 16
    assert report["source_unchanged"] is report["actor_unchanged"] is True
    assert report["scientific_metrics"] is None
    expected = np.random.Generator(np.random.PCG64(29))
    assert len(case.work["polls"]) == 16
    for index, slot in enumerate(slots):
        assert slot["status"] == "complete" and slot["early_release_receipts"] == []
        assert (
            slot["update_attempted"] is slot["update_committed"] is slot["commit_validated"] is True
        )
        assert slot["receipt"]["update_number"] == index + 1
        assert slot["charged_updates_before"] == slot["native_updates_before"] == index
        assert slot["charged_updates_after"] == slot["native_updates_after"] == index + 1
        assert slot["native_training_examples_before"] == 2 * index
        assert slot["native_training_examples_after"] == 2 * (index + 1)
        assert slot["sampler_before"] == expected.bit_generator.state
        expected.random(2)
        assert slot["sampler_after"] == expected.bit_generator.state
        update, identifier = BaseLearner.updates[index], BaseLearner.identifiers[index]
        assert update.episode.episode_id == identifier == slot["episode_id"]
        assert update.episode.role == "train" and update.episode.num_classes == 7
        assert len(update.episode.transitions[0].observation) == 148
        assert pickle.dumps(update.behavior_state) == pickle.dumps(
            case.work["collections"][index]["state"]
        )
        if index:
            assert update.behavior_state.optimizer_state["trace"] == index
        np.testing.assert_allclose(
            update.behavior_state.parameters["bias_output"], 0.01 * index, rtol=1e-6, atol=1e-8
        )
        early, committed = case.work["polls"][2 * index : 2 * index + 2]
        assert early == (index, index, ())
        assert committed[0:2] == (index, index + 1) and len(committed[2]) == 1
        assert case.work["labels"][index].targets == identifier
        assert case.work["experiences"][index].sample_id == identifier
        assert "behavior_state" not in slot["collection"]
    runtime = case.work["runtimes"][0]
    assert runtime.candidate_snapshot().state.completed_updates == 8
    assert (
        runtime.actor.snapshot_state().state.completed_updates
        == BaseLearner.sources[0].completed_updates
        == 0
    )
    before = deepcopy(report["training"][0]["collection"])
    case.work["collections"][0]["report"]["predictions"][0]["action"] = 123
    assert report["training"][0]["collection"] == before
    assert BaseLearner.classifications == [] and BaseLearner.sessions == []
    assert report["work"]["native_updates_committed"] == 8
    assert report["work"]["prediction_calls_returned"] == report["work"]["sampling_draws"] == 16
    assert report["work"]["environment_steps_returned"] == 16
    assert report["work"]["observed_recurrent_unroll_steps"] == 32
    assert all(
        report["work"][key] == 0
        for key in (
            "score_forwards",
            "teacher_episodes",
            "final_episodes",
            "probe_updates",
            "replay_updates",
            "promotions",
        )
    )


@pytest.mark.parametrize("phase", ["collection", "native"])
def test_should_stop_at_third_failure_preserving_two_commits_and_spent_sampling(
    case, monkeypatch, phase
):
    if phase == "collection":
        case.work["fail_collection"] = 2
    else:
        monkeypatch.setattr(BaseLearner, "fail_update", True)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["wake_updates"] == 2
    assert case.work["budget"].updates_completed == 2
    assert len(case.work["runtimes"][0].applied_updates) == 2
    slot = report["training"][2]
    assert slot["status"] == "failed" and slot["update_committed"] is False
    assert slot["charged_updates_after"] == 2
    assert all(slot["status"] == "not_run" for slot in report["training"][3:])
    expected = np.random.Generator(np.random.PCG64(29))
    expected.random(5 if phase == "collection" else 6)
    assert slot["sampler_after"] == expected.bit_generator.state
    assert slot["update_attempted"] is (phase == "native")
    if phase == "collection":
        assert slot["collection"]["work"]["sampling_draws"] == 1


@pytest.mark.parametrize(
    "target", ["source", "actor", "candidate", "sampler", "budget", "protocol"]
)
def test_should_keep_first_committed_receipt_and_charge_when_postcommit_boundary_refuses(
    case, monkeypatch, target
):
    def corrupt(owner):
        if owner.completed_updates != 1:
            return
        runtime = case.work["runtimes"][0]
        if target in ("source", "actor", "candidate"):
            selected = (
                BaseLearner.sources[0]
                if target == "source"
                else runtime.actor._learner
                if target == "actor"
                else owner
            )
            selected.state.parameters["bias_output"] += 1
            if target == "candidate":
                selected.state.completed_updates += 1
        elif target == "sampler":
            case.work["collections"][-1]["sampler"].random()
        elif target == "budget":
            budget = case.work["budget"]
            budget.budget = replace(budget.budget, max_training_updates=9)
        else:
            case.work["reports"][0]["protocol"]["seed"] = 18

    monkeypatch.setattr(Learner, "after_native", corrupt)
    report = pilot.run_pilot()
    first = report["training"][0]
    assert report["status"] == "incomplete" and case.work["budget"].updates_completed == 1
    assert first["update_committed"] is True and first["receipt"]["update_number"] == 1
    assert first["commit_validated"] is False
    assert len(BaseLearner.updates) == len(case.work["collections"]) == 1
    assert all(slot["status"] == "not_run" for slot in report["training"][1:])


@pytest.mark.parametrize("fault", ["receipt", "target"])
def test_should_refuse_replaced_receipt_or_nonmatching_release_without_more_training(case, fault):
    case.work["bad_receipt" if fault == "receipt" else "bad_target"] = True
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and len(case.work["collections"]) == 1
    if fault == "receipt":
        assert case.work["budget"].updates_completed == 1
        assert report["training"][0]["receipt"]["update_number"] == 1
    else:
        assert case.work["budget"].updates_completed == 0
    assert all(slot["status"] == "not_run" for slot in report["training"][1:])


def test_should_refuse_mutated_preceding_success_before_second_collection_can_spend_rng(case):
    def mutate(index, call):
        if index == 1:
            case.work["reports"][0]["training"][0]["receipt"]["update_number"] = 99

    case.work["on_collection"] = mutate
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and case.work["budget"].updates_completed == 1
    assert len(BaseLearner.updates) == 1
    assert case.work["runtimes"][0].applied_updates[0].update_number == 1
    assert "report" not in case.work["collections"][1]


@pytest.mark.parametrize("target", ["paid_record", "candidate_counter"])
def test_should_refuse_final_readback_mutation_and_return_original_paid_records(
    case, monkeypatch, target
):
    snapshot = Learner.snapshot_state
    close = Sampler.__exit__
    final_phase: list[bool] = []
    changed: list[bool] = []

    def exited(owner, *arguments):
        result = close(owner, *arguments)
        final_phase.append(True)
        return result

    def read(owner):
        value = snapshot(owner)
        if final_phase and not changed:
            changed.append(True)
            if target == "paid_record":
                case.work["reports"][0]["training"][0]["collection"]["work"]["sampling_draws"] = 999
            else:
                case.work["runtimes"][0]._candidate.state.completed_updates += 1
        return value

    monkeypatch.setattr(Learner, "snapshot_state", read)
    monkeypatch.setattr(Sampler, "__exit__", exited)
    report = pilot.run_pilot()
    assert changed == [True] and report["status"] == "incomplete"
    assert report["wake_updates"] == case.work["budget"].updates_completed == 8
    assert len(BaseLearner.updates) == len(case.work["collections"]) == 8
    assert [slot["receipt"]["update_number"] for slot in report["training"]] == list(range(1, 9))
    assert report["training"][0]["collection"]["work"]["sampling_draws"] == 2
    assert report["work"]["sampling_draws"] == 16


@pytest.mark.parametrize("target", ["candidate", "sampler"])
def test_should_recheck_full_candidate_and_rng_after_last_budget_callback(
    case, monkeypatch, target
):
    complete, before = (
        budgets.ToyBudgetSession.complete_memory,
        budgets.ToyBudgetSession.before_final,
    )
    memory_done: list[bool] = []
    changed: list[bool] = []

    def finish_memory(owner):
        result = complete(owner)
        memory_done.append(True)
        return result

    def final(owner):
        result = before(owner)
        if memory_done and not changed:
            changed.append(True)
            if target == "candidate":
                case.work["runtimes"][0]._candidate.state.parameters["bias_output"] += 1
            else:
                case.work["collections"][-1]["sampler"].random()
        return result

    monkeypatch.setattr(budgets.ToyBudgetSession, "complete_memory", finish_memory)
    monkeypatch.setattr(budgets.ToyBudgetSession, "before_final", final)
    report = pilot.run_pilot()
    assert changed == [True] and report["status"] == "incomplete"
    assert report["wake_updates"] == case.work["budget"].updates_completed == 8
    assert report["work"]["native_updates_committed"] == 8
    assert report["work"]["sampling_draws"] == 16
    assert len(BaseLearner.updates) == len(case.work["collections"]) == 8


def test_should_describe_cli_without_loading_native_runtime(monkeypatch, capsys):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    monkeypatch.setattr(pilot, "describe_protocol", lambda: {"protocol_id": "fixed"})
    assert cli.main(["--describe"]) == 0
    assert json.loads(capsys.readouterr().out) == {"protocol_id": "fixed"}


@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_should_publish_exact_cli_result_exclusively_and_return_operational_status(
    monkeypatch, tmp_path, capsys, status
):
    torch = SimpleNamespace(
        set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    report = {"status": status, "training": [{"charged_updates_after": 1}]}
    monkeypatch.setattr(pilot, "run_pilot", lambda: report)
    path = tmp_path / "reward.json"
    assert cli.main(["--output", str(path)]) == (0 if status == "complete" else 1)
    assert json.loads(path.read_text(encoding="utf8")) == report
    assert json.loads(capsys.readouterr().out)["bytes"] == path.stat().st_size


@pytest.mark.parametrize("fault", ["existing", "parent", "describe_output"])
def test_should_refuse_unsafe_output_request_before_optional_runtime(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    path = tmp_path / "reward.json"
    arguments = ["--output", str(path)]
    if fault == "existing":
        path.write_text("original", encoding="utf8")
    elif fault == "parent":
        arguments = ["--output", str(tmp_path / "missing" / "reward.json")]
    else:
        arguments.append("--describe")
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2
    if fault == "existing":
        assert path.read_text(encoding="utf8") == "original"


@pytest.mark.parametrize("fault", ["nonfinite", "oversized"])
def test_should_refuse_invalid_cli_payload_before_publication(monkeypatch, tmp_path, fault):
    torch = SimpleNamespace(
        set_num_threads=lambda value: None, set_num_interop_threads=lambda value: None
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)
    report = {"status": "complete", "value": float("nan") if fault == "nonfinite" else "x" * 100}
    monkeypatch.setattr(pilot, "run_pilot", lambda: report)
    if fault == "oversized":
        monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 32)
    path = tmp_path / "unpublished.json"
    with pytest.raises(ValueError):
        cli.main(["--output", str(path)])
    assert not path.exists()
