"""Post-training readouts preserve completed reward charges and original clocks.

The runtime is real; native learners, collection and scoring are owned fakes.
No teacher, environment, training sweep or final evaluation runs here.
"""

from copy import deepcopy
import inspect
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_minigrid_reward_smoke as cli
from src.adapters import minigrid_reward_pilot as pilot
from src.adapters import minigrid_warm_readouts as diagnostics
from src.adapters.minigrid_policy_evaluation import MiniGridEvaluationFailed
import src.app.toy_execution_budget as budgets
import src.shared.torch_runtime as torch_runtime
from test_minigrid_learning_pilot import source_batch
from test_minigrid_warm_reward import ports as ports, prepare_fixture, readout
from test_torch_feature_rehearsal_pilot import forbidden


def scored(updates, rows):
    record = readout(128)
    record.update(
        method="recurrent_warm_reward",
        updates=updates,
        native_training_examples=rows,
        state_unchanged=True,
        live_encoding="full148_normalized_before_after; redundant_raw_next_images_omitted",
    )
    record["scores"], record["episodes"] = [], []
    for context in ("A", "B", "C_train"):
        for old in readout(128)["scores"]:
            record["scores"].append(dict(old, context=context))
        for old in readout(128)["episodes"]:
            slot = deepcopy(old)
            slot["context"] = slot["environment"]["context"] = context
            slot.update(
                reason=None,
                state_unchanged=True,
                prediction_calls_attempted=2,
                prediction_calls_returned=2,
            )
            slot["environment"]["failure_work_may_be_unobserved"] = False
            record["episodes"].append(slot)
    record["work"].update(
        raw_score_forwards_attempted=6,
        raw_score_forwards_completed=6,
        raw_score_rows_attempted=54,
        raw_score_rows_completed=54,
        live_prediction_calls_attempted=12,
        live_prediction_calls_returned=12,
        live_sessions_started=6,
        environment_resets_attempted=6,
        environment_resets_completed=6,
        environment_steps_attempted=12,
        environment_steps_returned=12,
    )
    return record


@pytest.fixture
def readout_case(monkeypatch, ports):
    prepare_fixture(monkeypatch, ports)
    prepare = pilot.prepare_warm_start
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    clock = [10.0]
    work = ports.work
    work.update(readouts=[], readout_sources=sources, readout_clock=clock)
    monkeypatch.setattr(pilot, "monotonic", lambda: clock[0])
    monkeypatch.setattr(diagnostics, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        diagnostics, "_same_state", lambda left, right: pickle.dumps(left) == pickle.dumps(right)
    )

    def prepared(*args):
        result = prepare(*args)
        result["sources"] = sources
        return result

    monkeypatch.setattr(pilot, "prepare_warm_start", prepared)
    return SimpleNamespace(work=work, clock=clock, sources=sources)


def install_scoring(monkeypatch, case, module):
    work = case.work

    def evaluate(
        source,
        state,
        *,
        method,
        updates,
        native_training_examples,
        layouts,
        role,
        sources,
        budget,
        check_boundary,
        **options,
    ):
        assert method == "recurrent_warm_reward" and role == "train" and not options
        assert [layout.context for layout in layouts] == ["A", "B", "C_train"]
        assert sources is case.sources and all(
            packet.features.shape == (9, 148) for packet in sources
        )
        assert len(work["reward_calls"]) == len(work["collections"]) == 8
        assert work["budget"].updates_completed == 8
        assert all(slot["commit_validated"] for slot in work["reports"][0]["training"])
        assert (
            state.completed_updates == updates
            and state.training_examples == native_training_examples
        )
        assert budget is not work["budget"] and budget.budget.max_training_updates == 0
        frozen_reward = pickle.dumps(
            (
                work["budget"].budget,
                work["budget"].started_at,
                work["budget"].last_clock,
                work["budget"].updates_completed,
            )
        )
        call = dict(
            source=source,
            state=deepcopy(state),
            budget=budget,
            updates=updates,
            rows=native_training_examples,
            reward_binding=frozen_reward,
        )
        work["readouts"].append(call)
        check_boundary()
        budget.before_final()
        callback = work.get("during_readout")
        if callback:
            callback(len(work["readouts"]) - 1, call)
        record = scored(updates, native_training_examples)
        work["returned_readout"] = record
        if work.get("fail_second") and len(work["readouts"]) == 2:
            record.update(status="failed", reason="second readout retained native prefix")
            record["scores"] = record["scores"][:1]
            record["episodes"] = []
            record["work"].update(
                raw_score_forwards_attempted=2,
                raw_score_forwards_completed=1,
                raw_score_rows_attempted=18,
                raw_score_rows_completed=9,
                live_prediction_calls_attempted=0,
                live_prediction_calls_returned=0,
                live_sessions_started=0,
                environment_resets_attempted=0,
                environment_resets_completed=0,
                environment_steps_attempted=0,
                environment_steps_returned=0,
            )
            raise MiniGridEvaluationFailed(record)
        budget.before_final()
        check_boundary()
        assert (
            pickle.dumps(
                (
                    work["budget"].budget,
                    work["budget"].started_at,
                    work["budget"].last_clock,
                    work["budget"].updates_completed,
                )
            )
            == frozen_reward
        )
        return record

    monkeypatch.setattr(module, "evaluate_checkpoint", evaluate)


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_readout_flag_before_native_work(monkeypatch, invalid):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(warm_start=True, train_readouts=invalid)
    with pytest.raises(ValueError):
        pilot.run_pilot(warm_start=True, train_readouts=invalid)


def test_should_require_warm_start_for_readouts_before_resource_construction(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        pilot.describe_protocol(train_readouts=True)
    with pytest.raises(ValueError):
        pilot.run_pilot(train_readouts=True)


def test_should_preserve_default_descriptors_and_describe_new_mode_without_native_work(monkeypatch):
    monkeypatch.setattr(pilot, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "prepare_warm_start", forbidden)
    assert pilot.describe_protocol() == pilot.describe_protocol(train_readouts=False)
    assert pilot.describe_protocol(warm_start=True) == pilot.describe_protocol(
        warm_start=True, train_readouts=False
    )
    protocol = pilot.describe_protocol(warm_start=True, train_readouts=True)
    assert protocol["planned_work"]["native_train_calls"] == 136
    assert protocol["planned_work"]["teacher_episodes"] == 0
    assert json.loads(json.dumps(protocol, allow_nan=False)) == protocol


def test_should_finish_eight_reward_commits_before_both_guarded_train_readouts(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    sources_before = pickle.dumps(case.sources)
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "complete", report["reason"]
    result = report["train_diagnostics"]
    assert result["status"] == "complete" and [
        record["updates"] for record in result["readouts"]
    ] == [128, 136]
    assert [call["rows"] for call in case.work["readouts"]] == [1152, 1168]
    assert case.work["readouts"][0]["budget"] is case.work["readouts"][1]["budget"]
    assert case.work["readouts"][0]["budget"].budget.max_wall_seconds == 60
    phase = result["training_phase"]
    assert phase["updates_completed"] == 8 and phase["reward_training_rows"] == 16
    assert phase["native_updates"] == 136 and phase["native_training_examples"] == 1168
    assert phase["started_at"] == phase["last_clock"] == 10.0
    assert case.work["budget"].last_clock == case.work["budget"].started_at == 10.0
    assert result["work"]["raw_score_forwards_completed"] == 12
    assert result["work"]["raw_score_rows_completed"] == 108
    assert result["work"]["live_prediction_calls_returned"] == 24
    assert result["work"]["observed_recurrent_unroll_steps"] == 132
    assert all(
        not values["initial_joint_acquisition"] and not values["final_joint_acquisition"]
        for values in result["endpoint_conditions"].values()
    )
    assert all(
        all(value == 0 for value in values.values())
        for values in result["endpoint_changes"].values()
    )
    assert report["wake_updates"] == 8 and report["total_new_native_updates"] == 136
    assert len(case.work["reward_calls"]) == 8 and pickle.dumps(case.sources) == sources_before
    assert report["scientific_metrics"] is None and report["scientific_advantage_verified"] is False


def test_should_allow_readout_time_beyond_frozen_reward_phase_without_advancing_its_clock(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    case.work["during_readout"] = lambda index, call: case.clock.__setitem__(0, 60.0)
    original_before = budgets.ToyBudgetSession.before_final

    def before_final(owner):
        if case.work["readouts"] and owner is case.work["budget"]:
            pytest.fail("completed reward phase must never receive another clock observation")
        return original_before(owner)

    monkeypatch.setattr(budgets.ToyBudgetSession, "before_final", before_final)
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "complete", report["reason"]
    assert case.work["budget"].last_clock == 10.0
    assert case.work["readouts"][0]["budget"].last_clock == 60.0
    assert report["train_diagnostics"]["training_phase"]["elapsed_seconds"] == 0.0
    assert len(case.work["readouts"]) == 2 and report["total_new_native_updates"] == 136


def test_should_bound_readouts_by_remaining_original_whole_deadline(monkeypatch, readout_case):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    prepare = pilot.prepare_warm_start

    def finish_prefix(*args):
        result = prepare(*args)
        case.clock[0] = 170.0
        return result

    monkeypatch.setattr(pilot, "prepare_warm_start", finish_prefix)
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "complete", report["reason"]
    assert case.work["budget"].budget.max_wall_seconds == 20.0
    assert case.work["readouts"][0]["budget"].budget.max_wall_seconds == 20.0
    assert report["train_diagnostics"]["readout_budget"]["started_at"] == 170.0
    assert report["total_new_native_updates"] == 136


@pytest.mark.parametrize("late_clock", [71.0, 191.0])
def test_should_preserve_training_and_first_readout_when_original_time_limit_expires(
    monkeypatch, readout_case, late_clock
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)

    def expire(index, call):
        if index == 1:
            case.clock[0] = late_clock

    case.work["during_readout"] = expire
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 8
    assert report["total_new_native_updates"] == 136
    assert report["train_diagnostics"]["status"] == "incomplete"
    assert report["train_diagnostics"]["readouts"][0]["status"] == "complete"
    assert report["train_diagnostics"]["readouts"][1]["status"] != "complete"
    assert case.work["budget"].last_clock == 10.0 and case.work["budget"].updates_completed == 8
    assert len(case.work["reward_calls"]) == 8


@pytest.mark.parametrize(
    "target",
    [
        "receipt",
        "source",
        "actor",
        "candidate",
        "warm",
        "source_packet",
        "protocol",
        "phase_start",
        "phase_clock",
        "phase_clock_back",
        "phase_charges",
        "sampler",
    ],
)
def test_should_refuse_changed_training_authority_before_second_readout(
    monkeypatch, readout_case, target
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)

    def corrupt(index, call):
        if index:
            return
        runtime = case.work["runtimes"][0]
        if target == "receipt":
            object.__setattr__(runtime.applied_updates[0], "update_number", 99)
        elif target in ("source", "actor", "candidate"):
            learner = (
                case.work["warm_source"]
                if target == "source"
                else runtime.actor._learner
                if target == "actor"
                else runtime._candidate
            )
            learner._learner.state.parameters["bias_output"] += 1
        elif target == "warm":
            case.work["warm_report"]["parity_checks"][0] = False
        elif target == "source_packet":
            case.sources[0].features[0, 0] += 0.125
        elif target == "protocol":
            case.work["reports"][0]["protocol"]["seed"] = 18
        elif target == "phase_start":
            case.work["budget"].started_at += 1
        elif target == "phase_clock":
            case.work["budget"].last_clock += 1
        elif target == "phase_clock_back":
            case.work["budget"].last_clock -= 1
        elif target == "phase_charges":
            case.work["budget"].updates_completed += 1
        else:
            case.work["collections"][-1]["sampler"].random()

    case.work["during_readout"] = corrupt
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 8
    assert report["total_new_native_updates"] == 136 and len(case.work["readouts"]) == 1
    assert len(case.work["reward_calls"]) == len(case.work["collections"]) == 8
    assert all(
        slot["receipt"]["update_number"] == index + 1
        for index, slot in enumerate(report["training"])
    )
    assert report["warm_start"]["parity_checks"] == [True] * 5


def test_should_reject_first_completed_readout_mutation_and_restore_paid_original(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)

    def corrupt(index, call):
        if index == 1:
            case.work["reports"][0]["train_diagnostics"]["readouts"][0]["scores"][0]["accuracy"] = (
                1.0
            )

    case.work["during_readout"] = corrupt
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    assert report["status"] == "incomplete" and report["wake_updates"] == 8
    first = report["train_diagnostics"]["readouts"][0]
    assert first["status"] == "complete" and first["scores"][0]["accuracy"] == 0.0
    assert report["train_diagnostics"]["readouts"][1]["status"] != "complete"


def test_should_retain_failed_second_readout_prefix_and_all_training_charges(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    case.work["fail_second"] = True
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    result = report["train_diagnostics"]
    assert report["status"] == "incomplete" and report["wake_updates"] == 8
    assert report["total_new_native_updates"] == 136 and report["total_new_training_rows"] == 1168
    assert result["readouts"][0]["status"] == "complete"
    assert result["readouts"][1]["status"] == "failed" and len(result["readouts"][1]["scores"]) == 1
    assert "second readout retained" in result["readouts"][1]["reason"]
    assert result["work"]["raw_score_forwards_completed"] == 7
    assert result["work"]["raw_score_rows_completed"] == 63
    assert result["work"]["live_prediction_calls_returned"] == 12
    assert result["endpoint_changes"] is result["endpoint_conditions"] is None


def test_should_reject_cli_readouts_without_warm_start_before_native_setup(monkeypatch):
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--train-readouts"])
    assert error.value.code == 2


def test_should_describe_explicit_cli_readouts_without_optional_native_resources(
    monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(torch_runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)

    def describe(**options):
        calls.append(options)
        return {"declared": True}

    monkeypatch.setattr(pilot, "describe_protocol", describe)
    assert cli.main(["--warm-start", "--train-readouts", "--describe"]) == 0
    assert calls == [{"warm_start": True, "train_readouts": True}]
    assert json.loads(capsys.readouterr().out) == {"declared": True}


@pytest.mark.parametrize(
    "flags,expected,status,code",
    [
        (
            ["--warm-start", "--train-readouts"],
            {"warm_start": True, "train_readouts": True},
            "complete",
            0,
        ),
        (
            ["--warm-start", "--train-readouts"],
            {"warm_start": True, "train_readouts": True},
            "incomplete",
            1,
        ),
        (["--warm-start"], {"warm_start": True}, "complete", 0),
        ([], {}, "complete", 0),
    ],
)
def test_should_forward_only_explicit_readout_options_and_keep_cli_exit_contract(
    monkeypatch, capsys, flags, expected, status, code
):
    calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda count: None, set_num_interop_threads=lambda count: None
    )
    monkeypatch.setattr(torch_runtime, "require_torch", lambda: torch)

    def run(**options):
        calls.append(options)
        return {"status": status}

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert cli.main(flags) == code and calls == [expected]
    assert json.loads(capsys.readouterr().out)["status"] == status


def test_should_refuse_prior_record_corruption_from_last_source_readback_callback(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    run = pilot.run_train_diagnostics
    record_memory = budgets.ToyBudgetSession.record_memory_snapshot
    final_phase: list[bool] = []
    final_reads: list[bool] = []
    changed: list[bool] = []

    def memory_readback(owner):
        result = record_memory(owner)
        published = case.work["reports"][0]["train_diagnostics"]["readouts"]
        if (
            case.work["readouts"]
            and owner is case.work["readouts"][0]["budget"]
            and all(record["status"] == "complete" for record in published)
            and (frame := inspect.currentframe()) is not None
            and frame.f_back is not None
            and frame.f_back.f_code is run.__code__
        ):
            final_phase.append(True)
        return result

    class SourceReadback:
        def __init__(self, source, report):
            self.source, self.report = source, report

        def snapshot_state(self):
            state = self.source.snapshot_state()
            if final_phase and not changed:
                final_reads.append(True)
                # The guard's second source proof is its final external port
                # read. A pure evidence check must follow this callback too.
                if len(final_reads) == 2:
                    self.report["readouts"][0]["scores"][0]["accuracy"] = 1.0
                    changed.append(True)
            return state

    def guarded_readouts(report, source, *args, **kwargs):
        return run(report, SourceReadback(source, report), *args, **kwargs)

    monkeypatch.setattr(budgets.ToyBudgetSession, "record_memory_snapshot", memory_readback)
    monkeypatch.setattr(pilot, "run_train_diagnostics", guarded_readouts)
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    result = report["train_diagnostics"]
    assert changed == [True] and report["status"] == result["status"] == "incomplete"
    assert report["wake_updates"] == 8 and report["total_new_native_updates"] == 136
    assert result["readouts"][0]["scores"][0]["accuracy"] == 0.0
    assert all(record["status"] == "complete" for record in result["readouts"])
    assert result["work"]["raw_score_forwards_completed"] == 12
    assert result["work"]["raw_score_rows_completed"] == 108
    assert result["endpoint_changes"] is result["endpoint_conditions"] is None
    assert len(case.work["reward_calls"]) == 8


def test_should_preserve_both_paid_readouts_when_complete_record_cannot_form_endpoints(
    monkeypatch, readout_case
):
    case = readout_case
    install_scoring(monkeypatch, case, diagnostics)
    evaluate = diagnostics.evaluate_checkpoint

    def missing_summary_input(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        if kwargs["updates"] == 136:
            result["scores"] = result["scores"][:-1]
        return result

    monkeypatch.setattr(diagnostics, "evaluate_checkpoint", missing_summary_input)
    report = pilot.run_pilot(warm_start=True, train_readouts=True)
    result = report["train_diagnostics"]
    assert report["status"] == result["status"] == "incomplete"
    assert "both cues" in result["reason"]
    assert report["wake_updates"] == 8 and report["total_new_native_updates"] == 136
    assert [len(record["scores"]) for record in result["readouts"]] == [6, 5]
    assert all(record["status"] == "complete" for record in result["readouts"])
    assert result["work"]["raw_score_forwards_completed"] == 12
    assert result["work"]["raw_score_rows_completed"] == 108
    assert result["work"]["live_prediction_calls_returned"] == 24
    assert result["work"]["observed_recurrent_unroll_steps"] == 132
    assert result["endpoint_changes"] is result["endpoint_conditions"] is None
    assert report["total_new_work"]["raw_score_forwards"] == 22
    assert len(case.work["reward_calls"]) == 8
