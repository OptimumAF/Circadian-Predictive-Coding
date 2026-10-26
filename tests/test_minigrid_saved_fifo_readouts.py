"""Saved FIFO selection and orchestration with scripted ports, never learning.

Literal snapshots and fake games exercise exact admission, paid work and guards.
These checks establish no native task success or scientific advantage.
"""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import random

import pytest

from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_saved_fifo_readouts as pilot
from src.adapters.invariant_action_memory_learner import InvariantActionMemoryLearner
from src.adapters.minigrid_saved_fifo_inputs import decode_saved_fifo_inputs
import src.app.toy_execution_budget as budget_module
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.shared.process_memory import ProcessRssSegment
from test_minigrid_recurring_fifo_evaluator import (
    fifo_case as _fifo_case_fixture,
    refusal_case as _refusal_case_fixture,
    snapshot,
)

refusal_case = _refusal_case_fixture
fifo_case = _fifo_case_fixture
PAIRS = tuple((context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red"))


@pytest.fixture
def readonly_case(fifo_case):
    case = fifo_case
    case.budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=0, max_replay_examples=0), lambda: 10.0
    )

    def boundary():
        case.work["guard_calls"] = case.work.get("guard_calls", 0) + 1
        hook = case.work.get("on_guard")
        if hook is not None:
            hook(case)

    case.guard = boundary
    return case


def evaluate(case, *, score_pairs=(), game_pairs=(), **changes):
    options = dict(
        updates=128,
        layouts=case.layouts,
        role="train",
        sources=case.sources,
        budget=case.budget,
        check_boundary=case.guard,
        score_pairs=score_pairs,
        game_pairs=game_pairs,
    )
    options.update(changes)
    return evaluation.evaluate_invariant_fifo_checkpoint(case.source, case.state, **options)


@pytest.mark.parametrize(
    "scores,games",
    [((), ()), (PAIRS, ()), ((), PAIRS[2:]), (PAIRS[1:3], (PAIRS[-1],))],
)
def test_should_run_only_fixed_pending_cells_and_charge_only_observed_work(
    readonly_case, scores, games
):
    case = readonly_case
    before = pickle.dumps((case.source.value, case.state, case.sources))
    report = evaluate(case, score_pairs=scores, game_pairs=games)
    assert report["status"] == "complete" and report["state_unchanged"] is True
    assert [(row["context"], row["cue"]) for row in report["scores"]] == list(scores)
    assert [(row["context"], row["cue"]) for row in report["episodes"]] == list(games)
    assert all(row["source_role"] == "train" for row in report["episodes"])
    work = report["work"]
    assert work["raw_score_forwards_completed"] == len(scores)
    assert work["raw_score_rows_completed"] == 9 * len(scores)
    assert work["live_prediction_calls_returned"] == 2 * len(games)
    assert work["environment_steps_returned"] == 2 * len(games)
    assert work["internal_projection_rows_returned"] == 9 * len(scores) + 2 * len(games)
    assert case.budget.updates_completed == case.budget.replay_examples_completed == 0
    assert pickle.dumps((case.source.value, case.state, case.sources)) == before


@pytest.mark.parametrize("field", ["score_pairs", "game_pairs"])
@pytest.mark.parametrize(
    "pairs",
    [
        [],
        [["A", "blue"]],
        (("A", "blue", "train"),),
        (("A", "green"),),
        (("D_probe", "blue"),),
        (("A", "blue"), ("A", "blue")),
        (("B", "blue"), ("A", "red")),
        ((True, "blue"),),
    ],
)
def test_should_refuse_ambiguous_unordered_or_unavailable_cells_before_callbacks(
    readonly_case, field, pairs
):
    case = readonly_case
    with pytest.raises(ValueError):
        evaluate(case, **{field: pairs})
    assert case.work == {}


@pytest.mark.parametrize("training,replay", [(1, 0), (0, 1), (128, 0), (0, None)])
def test_should_require_new_readonly_budget_for_explicit_subsets(readonly_case, training, replay):
    case = readonly_case
    case.budget = ToyBudgetSession(
        ToyExecutionBudget(max_training_updates=training, max_replay_examples=replay),
        lambda: 10.0,
    )
    with pytest.raises(ValueError):
        evaluate(case, game_pairs=(PAIRS[-1],))
    assert case.work == {}


@pytest.mark.parametrize("target", ["unselected_packet", "unselected_episode", "budget"])
def test_should_guard_unselected_source_evidence_after_returned_prediction(readonly_case, target):
    case = readonly_case

    def mutate(scorer, features):
        if target == "unselected_packet":
            case.sources[-1].features[0, 0] = 0.0
        elif target == "unselected_episode":
            object.__setattr__(case.sources[-1].episode, "episode_id", "replaced-source")
        else:
            case.budget.updates_completed = 1

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case, score_pairs=(PAIRS[0],), game_pairs=(PAIRS[0],))
    report = caught.value.report
    assert report["status"] == "failed"
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["internal_projection_rows_returned"] == 9
    assert not case.work.get("environments")


def test_should_guard_all_sources_when_no_raw_score_was_selected(readonly_case):
    case = readonly_case

    def mutate(scorer, features):
        case.sources[-1].features[0, 0] = 0.0

    case.work["on_predict"] = mutate
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case, game_pairs=(PAIRS[0],))
    work = caught.value.report["work"]
    assert work["raw_score_forwards_completed"] == 0
    assert work["live_prediction_calls_returned"] == 1
    assert work["failure_work_may_be_unobserved"] is True


def test_should_keep_all_six_original_sources_required_for_small_subset(readonly_case):
    case = readonly_case
    with pytest.raises(ValueError):
        evaluate(case, sources=case.sources[:1], score_pairs=(PAIRS[0],))
    assert case.work == {}


def test_should_keep_failed_returned_subset_prediction_cost(readonly_case):
    case = readonly_case
    case.work["bad_probability"] = True
    with pytest.raises(evaluation.MiniGridEvaluationFailed) as caught:
        evaluate(case, score_pairs=(PAIRS[0],))
    report = caught.value.report
    assert report["scores"] == []
    assert report["work"]["raw_score_forwards_attempted"] == 1
    assert report["work"]["raw_score_forwards_completed"] == 1
    assert report["work"]["internal_projection_rows_returned"] == 9


def test_should_refuse_changed_snapshot_clock_even_for_empty_selection(readonly_case):
    case = readonly_case
    case.state = deepcopy(snapshot(96))
    with pytest.raises(ValueError):
        evaluate(case)
    assert case.work == {}


class Sampler:
    start_bytes = 100

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 100

    def snapshot(self):
        return ProcessRssSegment(1, 100, 100, 1, 0.01)


@pytest.fixture
def parent_case(monkeypatch):
    source = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/minigrid-recurring-fifo-20261017/native.json"
    )
    inputs = decode_saved_fifo_inputs(json.loads(source.read_text(encoding="utf-8")))
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "monotonic", lambda: 10.0)

    def forbidden(*args, **kwargs):
        pytest.fail("saved-readout parent reached native training or prediction")

    monkeypatch.setattr(InvariantActionMemoryLearner, "train_batch", forbidden)
    monkeypatch.setattr(InvariantActionMemoryLearner, "predict", forbidden)
    return inputs


def complete_report(options):
    scores, games = options["score_pairs"], options["game_pairs"]
    return dict(
        updates=options["updates"],
        role="train",
        method="public_invariant_observation_memory_fifo",
        status="complete",
        state_unchanged=True,
        scores=[dict(context=context, cue=cue) for context, cue in scores],
        episodes=[
            dict(context=context, cue=cue, status="complete", state_unchanged=True)
            for context, cue in games
        ],
        work=dict(
            raw_score_forwards_attempted=len(scores),
            raw_score_forwards_completed=len(scores),
            raw_score_rows_attempted=9 * len(scores),
            raw_score_rows_completed=9 * len(scores),
            live_prediction_calls_attempted=2 * len(games),
            live_prediction_calls_returned=2 * len(games),
            environment_resets_attempted=len(games),
            environment_resets_completed=len(games),
            environment_steps_attempted=2 * len(games),
            environment_steps_returned=2 * len(games),
            failure_work_may_be_unobserved=False,
        ),
    )


def test_should_compose_only_three_pending_groups_without_training_or_renewing_origin(
    parent_case, monkeypatch
):
    inputs = parent_case
    before = pickle.dumps(inputs)
    calls = []

    def evaluate_pending(source, state, **options):
        options["check_boundary"]()
        budget = options["budget"]
        assert budget.budget.max_training_updates == budget.budget.max_replay_examples == 0
        assert budget.budget.max_wall_seconds == 10.0
        assert source.snapshot_state().completed_updates == 0
        assert state.completed_updates == options["updates"]
        assert len(options["sources"]) == 6 and options["role"] == "train"
        calls.append(options)
        return complete_report(options)

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", evaluate_pending)
    report = pilot.run_pilot(inputs, max_seconds=10.0)
    assert report["status"] == "complete", report["reason"]
    assert [(o["updates"], o["score_pairs"], o["game_pairs"]) for o in calls] == [
        (64, (), PAIRS[2:]),
        (96, PAIRS, PAIRS),
        (128, PAIRS, PAIRS),
    ]
    assert all(o["budget"] is calls[0]["budget"] for o in calls)
    assert len(report["readouts"]) == 3 and len(report["completed_games"]) == 14
    assert report["censored_games"] == [[64, "B", "blue"]]
    assert report["origin"]["status"] == "incomplete"
    assert report["origin"]["training_updates"] == 128
    assert report["origin"]["training_rows"] == 1152
    assert report["origin"]["elapsed_seconds"] == inputs.original_elapsed_seconds
    assert report["new_training_updates"] == report["new_training_rows"] == 0
    assert report["work"]["raw_score_forwards_completed"] == 12
    assert report["work"]["raw_score_rows_completed"] == 108
    assert report["work"]["environment_resets_completed"] == 16
    assert report["scientific_advantage_verified"] is False
    assert pickle.dumps(inputs) == before


def test_should_keep_failed_paid_prefix_and_stop_before_later_groups(parent_case, monkeypatch):
    calls = []

    def fail(source, state, **options):
        calls.append(options["updates"])
        report = complete_report(options)
        report.update(status="failed", reason="scripted returned invalid prediction")
        report["work"]["failure_work_may_be_unobserved"] = True
        raise evaluation.MiniGridEvaluationFailed(report)

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", fail)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete" and calls == [64]
    assert report["readouts"] == []
    assert report["failed_or_unpublished_readout"]["status"] == "failed"
    assert report["work"]["environment_resets_completed"] == 4
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["censored_games"] == [[64, "B", "blue"]]
    assert report["origin"]["status"] == "incomplete"


@pytest.mark.parametrize("fault", ["status", "point", "pairs", "state"])
def test_should_validate_preceding_result_before_admitting_next_group(
    parent_case, monkeypatch, fault
):
    calls = []

    def wrong(source, state, **options):
        calls.append(options["updates"])
        report = complete_report(options)
        if fault == "status":
            report["status"] = "failed"
        elif fault == "point":
            report["updates"] = 96
        elif fault == "pairs":
            report["episodes"].reverse()
        else:
            report["episodes"][0]["state_unchanged"] = False
        return report

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", wrong)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete" and calls == [64]
    assert report["readouts"] == []
    assert "failed_or_unpublished_readout" in report


def test_should_refuse_corrupted_preceding_record_and_preserve_private_paid_copy(
    parent_case, monkeypatch
):
    returned = []

    def mutate_old(source, state, **options):
        if returned:
            returned[0]["episodes"][0]["cue"] = "replaced"
            options["check_boundary"]()
            pytest.fail("corrupt preceding evidence admitted another readout")
        report = complete_report(options)
        returned.append(report)
        return report

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", mutate_old)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete"
    assert len(report["readouts"]) == 1
    assert report["readouts"][0]["episodes"][0]["cue"] == "blue"
    assert report["work"]["environment_resets_completed"] == 4


def test_should_refuse_replaced_original_input_evidence_before_more_work(parent_case, monkeypatch):
    calls = []

    def change_input(source, state, **options):
        calls.append(options["updates"])
        object.__setattr__(parent_case, "origin_reason", "replaced")
        return complete_report(options)

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", change_input)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete" and calls == [64]
    assert "input evidence changed" in report["reason"]
    assert report["work"]["environment_resets_completed"] == 4


def test_should_refuse_original_raw_source_mutation_before_more_work(parent_case, monkeypatch):
    calls = []

    def change_input(source, state, **options):
        calls.append(options["updates"])
        options["sources"][-1].features[0, 0] = 0.5
        return complete_report(options)

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", change_input)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete" and calls == [64]
    assert "source packets changed" in report["reason"]
    assert report["work"]["environment_resets_completed"] == 4


@pytest.mark.parametrize("rng", ["python", "numpy"])
def test_should_refuse_rng_mutation_before_more_work_and_preserve_paid_result(
    parent_case, monkeypatch, rng
):
    python_state = random.getstate()
    numpy_state = pilot.np.random.get_state()
    calls = []

    def change_rng(source, state, **options):
        calls.append(options["updates"])
        if rng == "python":
            random.random()
        else:
            pilot.np.random.random()
        return complete_report(options)

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", change_rng)
    try:
        report = pilot.run_pilot(parent_case)
    finally:
        random.setstate(python_state)
        pilot.np.random.set_state(numpy_state)
    assert report["status"] == "incomplete" and calls == [64]
    assert "RNG state" in report["reason"]
    assert report["work"]["environment_resets_completed"] == 4


@pytest.mark.parametrize("seconds", [True, 0, -1, 121, float("inf"), float("nan")])
def test_should_refuse_invalid_wall_budget_before_runtime(parent_case, monkeypatch, seconds):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid allowance reached resource acquisition")

    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(parent_case, max_seconds=seconds)


@pytest.mark.parametrize(
    "fault",
    [
        "exit_input",
        "exit_record",
        "exit_error",
        "snapshot_input",
        "snapshot_record",
        "snapshot_error",
        "snapshot_rss",
        "closing_clock_input",
        "closing_clock_error",
        "closing_clock_deadline",
    ],
)
def test_should_guard_resource_closing_and_preserve_the_private_paid_prefix(
    parent_case, monkeypatch, fault
):
    returned = []
    phase = {"exited": False}

    def corrupt_input():
        object.__setattr__(parent_case, "origin_reason", "closing callback replacement")

    def corrupt_record():
        returned[-1]["episodes"][0]["cue"] = "closing replacement"

    class ClosingSampler(Sampler):
        def __exit__(self, *args):
            phase["exited"] = True
            if fault == "exit_input":
                corrupt_input()
            elif fault == "exit_record":
                corrupt_record()
            elif fault == "exit_error":
                raise RuntimeError("scripted sampler exit failed")
            return False

        def snapshot(self):
            if phase["exited"]:
                if fault == "snapshot_input":
                    corrupt_input()
                elif fault == "snapshot_record":
                    corrupt_record()
                elif fault == "snapshot_error":
                    raise RuntimeError("scripted closing snapshot failed")
                elif fault == "snapshot_rss":
                    return ProcessRssSegment(1, 100, pilot.RSS_LIMIT + 1, 2, 0.01)
            return super().snapshot()

    def clock():
        if phase["exited"]:
            if fault == "closing_clock_input":
                corrupt_input()
            elif fault == "closing_clock_error":
                raise RuntimeError("scripted closing clock failed")
            elif fault == "closing_clock_deadline":
                return 1000.0
        return 10.0

    def complete(source, state, **options):
        options["check_boundary"]()
        result = complete_report(options)
        returned.append(result)
        return result

    monkeypatch.setattr(pilot, "ProcessRssSampler", ClosingSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", ClosingSampler)
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", complete)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete", fault
    assert len(returned) == len(report["readouts"]) == 3
    assert report["readouts"][-1]["episodes"][0]["cue"] == "blue"
    assert report["origin"]["status"] == "incomplete"
    assert report["origin"]["reason"] != "closing callback replacement"
    assert report["censored_games"] == [[64, "B", "blue"]]
    assert report["work"]["raw_score_forwards_completed"] == 12
    assert report["work"]["raw_score_rows_completed"] == 108
    assert report["work"]["environment_resets_completed"] == 16
    assert report["new_training_updates"] == report["new_training_rows"] == 0
    if fault == "snapshot_rss":
        assert report["observed_process_rss"]["peak_bytes"] == pilot.RSS_LIMIT + 1


def test_should_mark_work_unknown_when_opaque_evaluator_raises_without_a_report(
    parent_case, monkeypatch
):
    calls = []

    def fail_without_receipt(source, state, **options):
        calls.append(options["updates"])
        raise RuntimeError("opaque prediction failed before returning its receipt")

    monkeypatch.setattr(pilot, "evaluate_invariant_fifo_checkpoint", fail_without_receipt)
    report = pilot.run_pilot(parent_case)
    assert report["status"] == "incomplete" and calls == [64]
    assert report["readouts"] == []
    assert report["work"]["failure_work_may_be_unobserved"] is True
    assert report["work"]["raw_score_forwards_completed"] == 0
    assert report["work"]["environment_resets_completed"] == 0
    assert report["origin"]["status"] == "incomplete"
    assert report["censored_games"] == [[64, "B", "blue"]]
