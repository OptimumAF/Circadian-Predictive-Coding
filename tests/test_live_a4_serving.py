"""Live runtime/publication/collector/game composition on scripted native ports.

Literal heads, predictions and environments exercise ownership and causality;
their success is not evidence of learned task utility.
"""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_affine_serving as published
from scripts import minigrid_live_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.app.actor_shadow import ActorShadowRuntime
from src.app.resource_sharing import ResourceSharedRuntime
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.learner_ports import TrainingDiagnostic
from src.core.experience import LogicalClock
from src.core.minigrid_context_encoding import context_public_history
from test_a4_fresh_workflow import bind_collection
from test_arrived_cue_catalogue import episode
from test_context_linear_causal_games import PhaseEnvironment
from test_affine_reference_readout import literal_state
from test_minigrid_learning_pilot import Environment

WORK = dict(
    owners=0,
    snapshots=0,
    trains=0,
    train_rows=0,
    predicts=0,
    predict_rows=0,
    games=0,
    steps=0,
    teacher_steps=0,
    collections=0,
    forbidden=0,
)
FLAG = "--example-affine-live-a4"


def bind_case(monkeypatch, *, collection_limit=12):
    box: dict[str, Any] = dict(time=0.0, trace=[], fits=[], runtime=None, reports=[], paid=[])
    collection = bind_collection(monkeypatch)
    packets = tuple(
        episode(f"minigrid-source-{context}-{cue}", color, context=context, action=2)
        for context in ("A", "B", "C_train")
        for cue, color in (("blue", 2), ("red", 0))
    )
    box["packets"] = packets
    original_init = ContextLinearLearner.__init__
    original_snapshot = ContextLinearLearner.snapshot_state
    original_runtime = ActorShadowRuntime.__init__
    original_shared = ResourceSharedRuntime.__init__
    original_run = app._run_live
    original_budget = ToyBudgetSession.__init__
    original_paid = app.paid_report_guard
    original_collection = app._collect_fresh_a4

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine numeric/physics/final/probe work forbidden")

    def empty(owner):
        WORK["owners"] += 1
        assert WORK["owners"] <= 500
        box["trace"].append("model")
        original_init(owner)

    def snapshot(owner):
        WORK["snapshots"] += 1
        assert WORK["snapshots"] <= 20000
        return original_snapshot(owner)

    def train(owner, x, y):
        WORK["trains"] += 1
        WORK["train_rows"] += len(x)
        assert WORK["trains"] <= 140 and WORK["train_rows"] <= 2100
        assert x.shape[1] == 296 and y.shape == (len(x), 1) and len(x) in (9, 18)
        box["fits"].append((len(x), box["runtime"]._inbox._clock.now()))
        if box.get("on_train"):
            box["on_train"]()
        if box.get("fit_fail") and len(box["fits"]) == 3:
            raise RuntimeError("lost third fit return")
        old = owner._state
        cue = int(context_public_history(tuple(float(v) for v in x[0]))[297])
        head = literal_state(1).heads[0]
        head = replace(head, fit_rows=len(x), singular_values=(0.0,) * len(x), rank=0)
        heads = list(old.heads)
        heads[cue] = head
        owner._state = replace(
            old,
            heads=tuple(heads),
            completed_updates=old.completed_updates + 1,
            training_examples=old.training_examples + len(x),
        )
        return TrainingDiagnostic("literal_no_solver", 0.0)

    def predict(owner, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        assert WORK["predicts"] <= 500 and WORK["predict_rows"] <= 2500
        callback = box.get("on_predict")
        if callback:
            callback(owner)
        p = np.zeros((len(x), 7))
        p[:, 2 if owner._state.completed_updates else 0] = 1
        return p

    class LiteralGame(PhaseEnvironment):
        def __init__(self, *args, **kwargs):
            WORK["games"] += 1
            assert WORK["games"] <= 100
            super().__init__(*args, **kwargs)

        def step(self, action):
            WORK["steps"] += 1
            assert WORK["steps"] <= 256
            return super().step(action)

        @property
        def report(self):
            return dict(
                super().report,
                layout=asdict(self.layout),
                role=self.role,
                cue=self.cue,
                seed=self.seed,
                max_steps=self.max_steps,
                episode_id=self.episode_id,
            )

    def runtime_init(owner, *args, **kwargs):
        original_runtime(owner, *args, **kwargs)
        box["runtime"] = owner

    def shared_init(owner, *args, **kwargs):
        original_shared(owner, *args, **kwargs)
        box["shared"] = owner

    def run(report):
        box["report"] = report
        return original_run(report)

    def budget_init(owner, *args, **kwargs):
        box["trace"].append("budget")
        original_budget(owner, *args, **kwargs)
        box.setdefault("budgets", []).append(owner)

    def read(path, *, expected_sha256):
        box["trace"].append("source")
        assert expected_sha256 == published.TRAIN_SHA
        return packets

    def collect(boundary, records):
        WORK["collections"] += 2
        assert WORK["collections"] <= collection_limit
        assert box["runtime"].actor.version == "c0"
        assert box["runtime"]._budget.updates_completed == 2
        result = original_collection(boundary, records)
        WORK["teacher_steps"] += 18
        assert WORK["teacher_steps"] <= collection_limit * 9
        return result

    def paid(value):
        box["paid"].append(value)
        check, detached = original_paid(value)
        callback = box.get("on_paid")
        if callback:
            callback(value)
        return check, detached

    monkeypatch.setattr(ContextLinearLearner, "__init__", empty)
    monkeypatch.setattr(ContextLinearLearner, "snapshot_state", snapshot)
    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)
    monkeypatch.setattr(ContextLinearLearner, "predict", predict)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(ActorShadowRuntime, "__init__", runtime_init)
    monkeypatch.setattr(ResourceSharedRuntime, "__init__", shared_init)
    monkeypatch.setattr(app, "_run_live", run)
    monkeypatch.setattr(ToyBudgetSession, "__init__", budget_init)
    monkeypatch.setattr(Environment, "work", box)
    monkeypatch.setattr(evaluation, "MiniGridKeyDoor", LiteralGame)
    monkeypatch.setattr(published, "example_affine_training", lambda: nullcontext(Path("literal")))
    monkeypatch.setattr(published, "_require_minigrid", lambda: box["trace"].append("preflight"))
    monkeypatch.setattr(cli, "read_training_sources", read)
    monkeypatch.setattr(app, "monotonic", lambda: box["time"])
    monkeypatch.setattr(app, "_collect_fresh_a4", collect)
    monkeypatch.setattr(app, "paid_report_guard", paid)
    box["collection"] = collection
    return box


@pytest.fixture
def case(monkeypatch):
    return bind_case(monkeypatch)


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        # Obtain the ordinary function fixture without a hidden numerical setup.
        fixture = bind_case(patch)
        target = tmp_path_factory.mktemp("live-a4") / "result.json"
        patch.setattr(sys, "argv", ["affine", FLAG, "--output", str(target)])
        cli.main()
        import json

        return json.loads(target.read_text()), fixture


def test_should_finish_actual_cli_collector_live_runtime_and_all_literal_games(complete):
    result, box = complete
    assert result["status"] == "complete", result.get("reason")
    assert result["preservation"] is True and result["summary"]["complete"] is True
    assert box["trace"][:4] == ["preflight", "budget", "budget", "source"]
    assert box["fits"] == [(9, 1), (9, 2), (18, 3), (18, 4), (18, 5), (18, 6)]


def test_should_preserve_published_actor_while_private_candidate_learns(complete):
    result, box = complete
    assert result["private"]["native_updates"] == 6 and result["private"]["native_rows"] == 90
    assert result["private"]["base_version"] == "a0"
    assert result["publication"]["generation"] == 1
    assert [row["updates"] for row in result["actor_games"]] == [4, 4, 6, 6]
    for row in result["actor_games"]:
        serving = row["game"]["serving"]
        assert serving["actor_version"] == "c0" and serving["generation"] == 1
        assert serving["sharing"]["latest"]["active_serving_requests"] == 0
    assert box["runtime"].actor.snapshot_state().state.completed_updates == 2


def test_should_report_fixed_private_phases_and_separate_reexposure_cost(complete):
    result, _ = complete
    assert [r["updates"] for r in result["games"]] == [2, 4, 6]
    assert all(len(r["games"]["episodes"]) == 4 for r in result["games"])
    catalogue = result["catalogue"]
    assert (catalogue["arrival_rows"], catalogue["replay_rows"], catalogue["native_rows"]) == (
        54,
        36,
        90,
    )
    assert catalogue["peak_bank_rows"] == 18
    assert result["source"]["origin"] == "current_run_A4_TRAIN"
    assert result["source"]["teacher_steps"] == 18
    assert len(result["collection"]) == 2 and all(r["closed"] for r in result["collection"])


def test_should_keep_original_lifetime_charges_and_causal_receipts(complete):
    result, _ = complete
    assert result["sharing"]["admitted_updates"] == 6
    assert result["budgets"]["native"]["updates"] == 6
    assert result["budgets"]["readout"]["updates"] == result["budgets"]["readout"]["replay"] == 0
    assert all(result["budgets"][name]["original_controls"] for name in ("native", "readout"))
    assert [r["arrived_at"] for r in result["private"]["receipts"]] == [1, 2, 3, 4, 5, 6]
    assert [r["observed_at"] for r in result["private"]["receipts"]] == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "fault",
    [
        "replace",
        "corrupt",
        "receipt",
        "budget",
        "actor",
        "gate",
        "source",
        "list_append",
        "logical_clock",
        "candidate_version",
    ],
)
def test_should_refuse_mutation_before_more_collection_or_training(case, fault):
    def mutate(record):
        if "generation" not in record:
            return
        runtime = case["runtime"]
        if fault == "replace":
            case["report"]["publication"] = deepcopy(record)
        elif fault == "corrupt":
            case["paid"][1]["updates"][0]["sample_id"] = "replaced"
        elif fault == "receipt":
            object.__setattr__(
                next(iter(runtime._inbox._applied.values())), "sample_id", "replaced"
            )
        elif fault == "budget":
            runtime._budget.started_at = 5.0
        elif fault == "actor":
            object.__setattr__(runtime.actor._slot, "generation", 99)
        elif fault == "source":
            object.__setattr__(case["packets"][0], "role", "final_test")
        elif fault == "list_append":
            case["report"]["games"].append({"unobserved": True})
        elif fault == "logical_clock":
            runtime._inbox._clock = LogicalClock(2)
        elif fault == "candidate_version":
            runtime._candidate_version = "c9"
        else:
            object.__setattr__(case["shared"]._sharing._limits, "max_admitted_updates", 7)

    case["on_paid"] = mutate
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and result["summary"] is None
    assert len(case["fits"]) == 2 and not case["collection"]["environments"]


def test_should_preserve_collection_failure_and_two_prior_paid_updates(case):
    case["collection"]["fault"] = "step"
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and result["summary"] is None
    assert len(case["fits"]) == 2 and result["private"]["native_updates"] == 2
    assert result["collection"][0]["closed"] is True
    assert "primary_error" in result["collection"][0]
    assert len(result["games"]) == 1


def test_should_keep_original_charge_when_third_fit_loses_return(case):
    case["fit_fail"] = True
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and "lost third fit" in result["reason"]
    assert result["work"]["train_work_may_be_unobserved"] is True
    assert len(case["fits"]) == 3 and len(result["private"]["receipts"]) == 2
    assert result["sharing"]["admitted_updates"] == 3
    assert result["budgets"]["native"]["original_controls"] is True


def test_should_refuse_prior_receipt_mutation_inside_next_native_fit(case):
    def mutate():
        if len(case["fits"]) == 3:
            record = next(iter(case["runtime"]._inbox._applied.values()))
            object.__setattr__(record, "sample_id", "mutated-inside-fit")

    case["on_train"] = mutate
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and "receipt prefix" in result["reason"]
    assert len(case["fits"]) == 3 and result["summary"] is None
    assert result["private"]["last_valid_receipts"][0]["sample_id"] == "batch-0"
    assert result["budgets"]["native"]["updates"] == 3


def test_should_preserve_actor_failure_and_previous_private_phase(case, monkeypatch):
    original = ResourceSharedRuntime.predict

    def fail(owner, x):
        original(owner, x)
        raise RuntimeError("lost public actor return")

    monkeypatch.setattr(ResourceSharedRuntime, "predict", fail)
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and result["summary"] is None
    assert [r["updates"] for r in result["games"]] == [2, 4]
    assert len(case["fits"]) == 4 and not result["actor_games"]
    assert (
        result["failed_actor_game"]["report"]["serving"]["sharing"]["latest"][
            "active_serving_requests"
        ]
        == 0
    )


def test_should_describe_without_models_sources_budgets_or_native_runtime(
    case, monkeypatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, "--describe"])
    cli.main()
    assert "portable_live_published_affine_a4_v1" in capsys.readouterr().out
    assert case["trace"] == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--example-affine-published"],
        ["--source", "foreign.json"],
        ["--fit-updates", "6"],
        ["--cue-relative"],
    ],
)
def test_should_refuse_unfrozen_cli_combinations_without_work(case, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["affine", FLAG, *extra])
    with pytest.raises(SystemExit):
        cli.main()
    assert case["trace"] == []


def test_should_refuse_missing_native_runtime_before_source_and_budget(case, monkeypatch):
    def missing():
        raise RuntimeError("missing native pins")

    monkeypatch.setattr(published, "_require_minigrid", missing)
    result = app.run_live_adaptation()
    assert result["status"] == "failed" and result["phase"] == "preflight"
    assert case["trace"] == []
