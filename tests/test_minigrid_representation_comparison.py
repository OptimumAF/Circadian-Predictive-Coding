"""Matched representation orchestration with paid-budget fakes, no neural updates.

Discrete TRAIN arrays and evaluator receipts are engineering fixtures. Their
outcomes are deliberately negative controls, not game or learning evidence.
"""

from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_minigrid_history_evaluation_smoke as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.app import toy_execution_budget as budgets
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.demonstration_episode import DemonstrationEpisode, DemonstrationTransition
from test_minigrid_invariant_encoding import frame
from test_torch_feature_rehearsal_pilot import Sampler, TorchStub, forbidden


RAW = ("bp", "pc", "neutral_cpc", "gated_cpc")
METHODS = RAW + tuple("public_invariant_" + method for method in RAW)


def original_sources():
    result = []
    for cue, color in (("blue", 2), ("red", 0)):
        frames = tuple(
            frame(direction=i % 4, cue_color=color, key_color=color, door_state=2 if i < 5 else 0)
            for i in range(10)
        )
        result.append(
            DemonstrationEpisode(
                f"minigrid-source-A-{cue}",
                "A",
                "train",
                tuple(
                    DemonstrationTransition(
                        i, frames[i], i % 7, 0.5 if i == 8 else 0.0, frames[i + 1], i == 8, 7
                    )
                    for i in range(9)
                ),
                7,
            )
        )
    return tuple(result)


@dataclass
class NativeState:
    head_state: dict[str, Any]
    completed_updates: int = 0
    training_examples: int = 0
    optimizer_state: Any = None
    gradients: Any = None


class Source:
    def __init__(self, method):
        native = NativeState({"weight": np.zeros((1, 1))})
        self.state = (
            SimpleNamespace(native_state=native)
            if method.startswith("public_invariant_")
            else native
        )

    def snapshot_state(self):
        return deepcopy(self.state)


def native(state):
    return getattr(state, "native_state", state)


def readout_record(method, point, *, success=False):
    return dict(
        method=method,
        updates=point,
        status="complete",
        reason=None,
        scores=[],
        episodes=[
            dict(
                context=context,
                cue=cue,
                source_role=role,
                status="complete",
                success=success,
                environment={"environment_steps": 2},
                **{"return": 0.0},
            )
            for context, cue, role in (
                ("A", "blue", "train"),
                ("A", "red", "train"),
                ("E-blue", "blue", "probe_test"),
                ("E-red", "red", "probe_test"),
            )
        ],
        work=dict(
            raw_score_forwards_completed=0,
            raw_score_rows_completed=0,
            live_prediction_calls_returned=8,
            live_environment_steps_returned=8,
        ),
    )


@pytest.fixture
def case(monkeypatch):
    work: dict[str, Any] = dict(
        trains=[],
        readouts=[],
        records=[],
        arms=[],
        now=1.0,
        before_train=None,
        before_readout=None,
        after_readout=None,
        readout_error=None,
        exit_callback=None,
    )
    supplied = original_sources()

    class ObservedSampler(Sampler):
        def __exit__(self, *args):
            work["sampler_exited"] = True
            callback = work["exit_callback"]
            if callback is not None:
                callback()
            return False

    clock = lambda: work["now"]
    monkeypatch.setattr(cli, "monotonic", clock)
    monkeypatch.setattr(cli, "require_torch", lambda: TorchStub)
    monkeypatch.setattr(cli, "ProcessRssSampler", ObservedSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", ObservedSampler)
    monkeypatch.setattr(cli.shared, "_same_state", lambda a, b: pickle.dumps(a) == pickle.dumps(b))
    monkeypatch.setattr(cli.shared, "_collect_sources", forbidden)
    monkeypatch.setattr(cli, "_history_learner", forbidden)

    def train(row, packets, sampler, deadline, before, *, learner_factory, checkpoint_updates):
        before()
        callback = work["before_train"]
        if callback is not None:
            callback(row)
        before()
        assert checkpoint_updates == (0, 32)
        assert len(packets) == 32
        assert all(
            p.role == "train"
            and p.context == "A"
            and p.features.shape == (9, 296)
            and p.num_classes == 7
            for p in packets
        )
        source = Source(row["method"])
        if row["method"] == work.get("initial_mismatch"):
            native(source.state).head_state["weight"][:] = 2
        initial = source.snapshot_state()
        final = deepcopy(initial)
        native(final).completed_updates = 32
        native(final).training_examples = 288
        native(final).head_state["weight"][:] = 0.01
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=32, max_wall_seconds=deadline - clock()), clock
        )
        for _ in range(32):
            budget.before_update()
            budget.record_update()
        row.update(
            status="trained",
            checkpoint_updates=[0, 32],
            charged_updates=32,
            native_training_examples=288,
            source_unchanged=True,
            actor_unchanged=True,
            wake_calls_attempted=32,
            wake_calls_returned=32,
        )
        row_before = pickle.dumps(row)
        source_before = pickle.dumps(source.snapshot_state())
        state_before = pickle.dumps((initial, final))
        budget_before = (
            budget.budget,
            budget.clock,
            budget.started_at,
            budget.last_clock,
            budget.updates_completed,
        )

        def guard():
            if (
                pickle.dumps(row) != row_before
                or pickle.dumps(source.snapshot_state()) != source_before
                or pickle.dumps((initial, final)) != state_before
                or (
                    budget.budget,
                    budget.clock,
                    budget.started_at,
                    budget.last_clock,
                    budget.updates_completed,
                )
                != budget_before
            ):
                raise ValueError("fake completed TRAIN evidence changed")

        arm = dict(
            row=row, source=source, states=((0, initial), (32, final)), budget=budget, guard=guard
        )
        work["trains"].append(dict(method=row["method"], packets=packets, deadline=deadline))
        work["arms"].append(arm)
        return arm

    def evaluate(source, state, *, updates, budget, check_boundary):
        assert len(work["trains"]) == 8, "a readout started before all TRAIN arms finished"
        check_boundary()
        callback = work["before_readout"]
        if callback is not None:
            callback(source, state, budget)
        check_boundary()
        method = next(arm["row"]["method"] for arm in work["arms"] if arm["source"] is source)
        work["readouts"].append(
            dict(method=method, updates=updates, source=source, state=state, budget=budget)
        )
        budget.before_final()
        error = work["readout_error"]
        if error is not None:
            error(method, updates)
        result = readout_record(method, updates)
        work["records"].append(result)
        callback = work["after_readout"]
        if callback is not None:
            callback(result)
        check_boundary()
        return result

    monkeypatch.setattr(cli.shared, "_train_arm", train)
    monkeypatch.setattr(evaluation, "evaluate_raw_neural_representation_checkpoint", evaluate)
    monkeypatch.setattr(evaluation, "evaluate_neural_representation_checkpoint", evaluate)
    work.update(
        sources=supplied,
        run=lambda: cli.run_smoke(canonical_neural_comparison=True, sources=supplied),
    )
    return work


def test_should_describe_the_frozen_matched_dose_without_resources_or_source_io(monkeypatch):
    for name in ("require_torch", "ProcessRssSampler", "run_smoke", "load_comparison_sources"):
        monkeypatch.setattr(cli, name, forbidden)
    protocol = cli.describe_protocol(canonical_neural_comparison=True)
    assert protocol["protocol_id"] == "minigrid_matched_raw_canonical_A32_v1"
    assert protocol["methods"] == list(METHODS)
    assert protocol["checkpoints"] == [0, 32] and protocol["label_delay"] == 1
    assert protocol["seed"] == 17 and protocol["hidden_dim"] == 8 and protocol["num_classes"] == 7
    assert protocol["planned_work"]["native_training_updates"] == 256
    assert protocol["planned_work"]["training_rows"] == 2304
    assert protocol["planned_work"]["live_episodes"] == 64
    assert protocol["heldout"] == [] and protocol["training_only"] is True
    assert protocol["planned_work"]["raw_score_forwards"] == 0
    assert protocol["slots"] == [
        ["A", "blue", "train"],
        ["A", "red", "train"],
        ["E-blue", "blue", "probe_test"],
        ["E-red", "red", "probe_test"],
    ]
    assert protocol["limits"]["body_seconds"] == 300
    assert protocol["limits"]["parent_seconds"] == 330
    json.dumps(protocol, allow_nan=False)


def test_should_train_all_eight_arms_then_preserve_initial_endpoint_pairs_and_negative_results(
    case,
):
    before = pickle.dumps(case["sources"])
    report = case["run"]()
    assert report["status"] == "complete", report["reason"]
    assert [row["method"] for row in report["methods"]] == list(METHODS)
    assert [row["charged_updates"] for row in report["methods"]] == [32] * 8
    assert sum(row["native_training_examples"] for row in report["methods"]) == 2304
    assert [(r["method"], r["updates"]) for r in report["evaluations"]] == [
        (method, point) for method in METHODS for point in (0, 32)
    ]
    assert len(report["planned_evaluations"]) == 16
    assert all(row["status"] == "complete" for row in report["planned_evaluations"])
    assert report["initial_pair_parity"] == [
        dict(method=method, state_equal=True) for method in RAW
    ]
    assert len({a["deadline"] for a in case["trains"]}) == 1
    assert case["trains"][0]["deadline"] == 301.0
    readout_budget = case["readouts"][0]["budget"]
    assert all(r["budget"] is readout_budget for r in case["readouts"])
    assert readout_budget.budget.max_training_updates == readout_budget.updates_completed == 0
    assert all(readout_budget is not arm["budget"] for arm in case["arms"])
    assert all(arm["budget"].updates_completed == 32 for arm in case["arms"])
    assert [source["role"] for source in report["training_sources"]] == ["train", "train"]
    assert pickle.dumps(case["sources"]) == before
    assert all(
        not episode["success"] for record in report["evaluations"] for episode in record["episodes"]
    )
    for row in report["representation_comparison"]["methods"]:
        for context in ("A", "E"):
            assert row[context]["initial"]["successes"] == row[context]["end"]["successes"] == 0
            assert row[context]["gain"]["successes"] == 0
    assert case["sampler_exited"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "prior_result",
        "prior_row",
        "source",
        "state",
        "training_budget",
        "readout_budget",
        "sources",
    ],
)
def test_should_refuse_mutated_preceding_evidence_and_keep_original_paid_captures(case, fault):
    def corrupt(source, state, budget):
        if not case["records"]:
            return
        if fault == "prior_result":
            case["records"][0]["episodes"][0]["success"] = True
        elif fault == "prior_row":
            case["arms"][0]["row"]["charged_updates"] = 0
        elif fault == "source":
            native(case["arms"][0]["source"].state).head_state["weight"][:] = 1
        elif fault == "state":
            native(case["arms"][0]["states"][0][1]).head_state["weight"][:] = 1
        elif fault == "training_budget":
            case["arms"][0]["budget"].updates_completed = 0
        elif fault == "readout_budget":
            budget.started_at = 2.0
        else:
            object.__setattr__(case["sources"][0], "context", "final_D")

    case["before_readout"] = corrupt
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert len(case["readouts"]) == len(report["evaluations"]) == 1
    assert report["evaluations"][0]["episodes"][0]["success"] is False
    assert report["methods"][0]["charged_updates"] == 32
    assert all(slot["status"] == "not_run" for slot in report["planned_evaluations"][2:])


def test_should_refuse_unequal_initial_native_states_before_any_readout(case):
    case["initial_mismatch"] = METHODS[4]
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert case["readouts"] == []
    assert all(slot["status"] == "not_run" for slot in report["planned_evaluations"])


def test_should_stop_on_original_deadline_without_renewing_or_discarding_paid_training(case):
    def exhaust(source, state, budget):
        case["now"] = 302.0

    case["before_readout"] = exhaust
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert len(case["trains"]) == 8 and case["readouts"] == []
    assert sum(row["charged_updates"] for row in report["methods"]) == 256


@pytest.mark.parametrize("fault", ["source", "prior_result"])
def test_should_check_preceding_evidence_after_sampler_exit_and_retain_paid_readouts(case, fault):
    def late_mutation():
        if fault == "source":
            native(case["arms"][0]["source"].state).head_state["weight"][:] = 7
        else:
            case["records"][0]["episodes"][0]["success"] = True

    case["exit_callback"] = late_mutation
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert len(report["evaluations"]) == len(case["readouts"]) == 16
    assert report["evaluations"][0]["episodes"][0]["success"] is False
    assert sum(row["charged_updates"] for row in report["methods"]) == 256


@pytest.mark.parametrize("partial", [False, True])
def test_should_retain_failed_delegate_receipt_or_explicitly_unknown_lost_work(case, partial):
    def fail(method, updates):
        if partial:
            record = readout_record(method, updates)
            record.update(status="incomplete", reason="fake second prediction failed")
            record["episodes"] = record["episodes"][:1]
            record["work"]["live_prediction_calls_returned"] = 2
            record["work"]["live_environment_steps_returned"] = 2
            raise cli.MiniGridEvaluationFailed(record)
        raise RuntimeError("delegate failed without a receipt")

    case["readout_error"] = fail
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert len(case["readouts"]) == 1
    assert report["planned_evaluations"][0]["status"] != "complete"
    assert all(slot["status"] == "not_run" for slot in report["planned_evaluations"][1:])
    if partial:
        assert len(report["evaluations"]) == 1
        assert len(report["evaluations"][0]["episodes"]) == 1
        assert report["evaluations"][0]["work"]["live_prediction_calls_returned"] == 2
    else:
        assert report["evaluations"] == []
        slot = report["planned_evaluations"][0]
        assert (
            slot["delegated_report_unavailable"] is slot["failure_work_may_be_unobserved"] is True
        )
        assert slot["work"]["live_prediction_calls_attempted"] is None
        assert slot["work"]["environment_steps_attempted"] is None


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_should_refuse_nonbool_mode_before_any_resources(monkeypatch, value):
    monkeypatch.setattr(cli, "require_torch", forbidden)
    monkeypatch.setattr(cli, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        cli.describe_protocol(canonical_neural_comparison=value)
    with pytest.raises(ValueError):
        cli.run_smoke(canonical_neural_comparison=value, sources=original_sources())


@pytest.mark.parametrize(
    "fault", ["missing", "list", "reversed", "probe_role", "context", "duplicate"]
)
def test_should_validate_original_train_pair_before_resources(monkeypatch, fault):
    values: Any = original_sources()
    if fault == "missing":
        values = None
    elif fault == "list":
        values = list(values)
    elif fault == "reversed":
        values = tuple(reversed(values))
    elif fault == "probe_role":
        object.__setattr__(values[0], "role", "probe_train")
    elif fault == "context":
        object.__setattr__(values[0], "context", "E-blue")
    else:
        values = (values[0], values[0])
    monkeypatch.setattr(cli, "require_torch", forbidden)
    monkeypatch.setattr(cli, "ProcessRssSampler", forbidden)
    with pytest.raises(ValueError):
        cli.run_smoke(canonical_neural_comparison=True, sources=values)


def test_should_preserve_default_and_fixed_a_descriptors_when_new_mode_is_false():
    assert cli.describe_protocol() == cli.describe_protocol(canonical_neural_comparison=False)
    assert cli.describe_protocol(fixed_a_acquisition=True) == cli.describe_protocol(
        fixed_a_acquisition=True, canonical_neural_comparison=False
    )
    assert cli.describe_protocol()["methods"] == [*RAW, "recurrent_bc"]


def test_should_route_comparison_preview_without_source_loading(monkeypatch, capsys):
    calls = []
    descriptor = dict(protocol_id="matched", methods=list(METHODS))
    monkeypatch.setattr(sys, "argv", ["history", "--canonical-neural-comparison", "--describe"])
    monkeypatch.setattr(cli, "load_comparison_sources", forbidden)
    monkeypatch.setattr(cli, "run_smoke", forbidden)

    def describe(**options):
        calls.append(options)
        return descriptor

    monkeypatch.setattr(cli, "describe_protocol", describe)
    cli.main()
    assert calls == [dict(canonical_neural_comparison=True)]
    assert json.loads(capsys.readouterr().out) == descriptor


def test_should_reuse_the_original_single_read_source_admission(monkeypatch, tmp_path):
    from scripts import run_minigrid_invariant_pilot as loader

    path = tmp_path / "accepted-source.json"
    path.write_text('{"preserve":"original bytes"}')
    before = path.read_bytes()
    sources = original_sources()
    calls = []

    def load(original_path):
        calls.append(original_path)
        return (object(), object()), sources

    monkeypatch.setattr(loader, "load_inputs", load)
    assert cli.load_comparison_sources(path) == sources
    assert calls == [path] and path.read_bytes() == before


@pytest.mark.parametrize("status", ["complete", "incomplete"])
def test_should_load_original_pair_once_and_dispatch_only_frozen_comparison_arguments(
    monkeypatch, capsys, tmp_path, status
):
    source_path = tmp_path / "prior.json"
    source_path.write_text('{"unmodified":true}')
    original = source_path.read_bytes()
    sources = original_sources()
    loads, calls = [], []

    def load(path):
        loads.append(path)
        return sources

    def run(**options):
        calls.append(options)
        return dict(
            status=status,
            reason=None if status == "complete" else "retained negative operational receipt",
        )

    monkeypatch.setattr(cli, "load_comparison_sources", load)
    monkeypatch.setattr(cli, "run_smoke", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["history", "--canonical-neural-comparison", "--source-artifact", str(source_path)],
    )
    if status == "complete":
        cli.main()
    else:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
    assert loads == [source_path]
    assert len(calls) == 1 and calls[0] == dict(canonical_neural_comparison=True, sources=sources)
    assert calls[0]["sources"] is sources
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert source_path.read_bytes() == original


@pytest.mark.parametrize(
    "arguments",
    [
        ["--canonical-neural-comparison"],
        ["--canonical-neural-comparison", "--fixed-a-acquisition", "--describe"],
        ["--source-artifact", "source.json"],
        ["--fixed-a-acquisition", "--source-artifact", "source.json"],
        ["--canonical-neural-comparison", "--describe", "--source-artifact", "source.json"],
        ["--canonical-neural-comparison", "--seed", "42"],
    ],
)
def test_should_refuse_unfrozen_cli_routes_before_loading_or_execution(monkeypatch, arguments):
    for name in ("load_comparison_sources", "run_smoke", "describe_protocol"):
        monkeypatch.setattr(cli, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["history", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_refuse_nonfinite_comparison_output_before_printing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "load_comparison_sources", lambda path: original_sources())
    monkeypatch.setattr(
        cli, "run_smoke", lambda **options: dict(status="complete", invalid=float("nan"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "history",
            "--canonical-neural-comparison",
            "--source-artifact",
            str(tmp_path / "source.json"),
        ],
    )
    with pytest.raises(ValueError):
        cli.main()
    assert capsys.readouterr().out == ""


def test_should_enforce_the_comparison_output_allowance_before_printing(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setattr(cli, "load_comparison_sources", lambda path: original_sources())
    monkeypatch.setattr(
        cli, "run_smoke", lambda **options: dict(status="complete", oversized="x" * (33 * 1024**2))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "history",
            "--canonical-neural-comparison",
            "--source-artifact",
            str(tmp_path / "source.json"),
        ],
    )
    with pytest.raises((ValueError, RuntimeError)):
        cli.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("fault", ["missing_return", "success_integer", "nan_return", "bad_steps"])
def test_should_stop_after_malformed_paid_metrics_and_publish_unavailable_contrasts(case, fault):
    def corrupt(record):
        slot = record["episodes"][0]
        if fault == "missing_return":
            del slot["return"]
        elif fault == "success_integer":
            slot["success"] = 1
        elif fault == "nan_return":
            slot["return"] = float("nan")
        else:
            slot["environment"]["environment_steps"] = True

    case["after_readout"] = corrupt
    report = case["run"]()
    assert report["status"] == "incomplete" and report["reason"]
    assert len(case["readouts"]) == len(report["evaluations"]) == 1
    assert report["planned_evaluations"][0]["status"] == "failed"
    assert all(slot["status"] == "not_run" for slot in report["planned_evaluations"][1:])
    assert all(row["charged_updates"] == 32 for row in report["methods"])
    assert all(
        row["status"] == "unavailable" for row in report["representation_comparison"]["methods"]
    )
    assert all(
        row["A"] is None and row["E"] is None
        for row in report["representation_comparison"]["methods"]
    )
    json.dumps(report, allow_nan=False)
    if fault == "nan_return":
        assert report["evaluations"][0]["episodes"][0]["return"] == {"invalid_native_float": "nan"}
        assert (
            case["records"][0]["episodes"][0]["return"]
            != case["records"][0]["episodes"][0]["return"]
        )
