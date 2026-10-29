"""Fresh TRAIN collection and actual stream with literal states/scripted readouts.

No solver, parameter-dependent prediction or native environment is executed.
Native geometry/game-loop evidence belongs to #199/#200, not these fake goals.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import json
import pickle
import random
import sys
from typing import Any

import numpy as np
import pytest

from scripts import minigrid_a4_adaptation as app
from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_context_linear_learner as port
from src.core.learner_ports import TrainingDiagnostic
from src.core.reference_controller import ReferencePolicy
from tests.test_a4_adaptation import sources
from tests.test_a4_fresh_workflow import bind_collection
from tests.test_affine_reference_catalogue_readout import literal_state

WORK = dict(
    owners=0,
    snapshots=0,
    fits=0,
    fit_rows=0,
    predicts=0,
    predict_rows=0,
    readouts=0,
    collection_envs=0,
    teacher_steps=0,
    forbidden=0,
)
HOOKS: dict[str, Any] = {}
TRACE: list[Any] = []


class LiteralAffine:
    def __init__(self, state=None):
        WORK["owners"] += 1
        self.state = literal_state(0) if state is None else deepcopy(state)

    def fork(self):
        return LiteralAffine(self.state)

    def snapshot_state(self):
        WORK["snapshots"] += 1
        return deepcopy(self.state)

    def restore_state(self, state):
        self.state = deepcopy(state)

    def train_batch(self, x, y):
        WORK["fits"] += 1
        WORK["fit_rows"] += len(x)
        TRACE.append((x.copy(), y.copy()))
        if "train" in HOOKS:
            HOOKS["train"](self)
        self.state = literal_state(self.state.completed_updates + 1)
        assert self.state.training_examples == sum(len(a) for a, b in TRACE)
        return TrainingDiagnostic("literal_a4_prefix", float(len(x)))

    def predict(self, x):
        WORK["predicts"] += 1
        WORK["predict_rows"] += len(x)
        result = np.zeros((len(x), 7))
        result[:, HOOKS.get("class", 0)] = 1
        return result


def scripted_readout(owner, state, *, budget, check_boundary, **options):
    WORK["readouts"] += 1
    assert options == dict(reference_window=True, reference_catalogue=True) or options == dict(
        a4_training=True
    )
    assert state == literal_state(state.completed_updates)
    assert budget.updates_completed == budget.replay_examples_completed == 0
    check_boundary()
    if "readout" in HOOKS:
        HOOKS["readout"](budget, check_boundary)
    rows = []
    for i, (branch, cue) in enumerate(((3, "blue"), (3, "red"), (4, "blue"), (4, "red"))):
        rows.append(
            dict(
                status="complete",
                state_unchanged=True,
                source_role="train",
                cue=cue,
                success=state.completed_updates > i,
                **{"return": float(state.completed_updates > i)},
                predictions=[dict(action=0)],
                environment=dict(
                    episode_id=f"literal-{state.completed_updates}-{branch}-{cue}",
                    role="train",
                    cue=cue,
                    seed=17,
                    max_steps=16,
                    closed=True,
                    failure_work_may_be_unobserved=False,
                    environment_steps=1,
                    layout=dict(
                        context="A",
                        branch_row=branch,
                        rotation=0,
                        swap_keys=False,
                        wall_color="grey",
                    ),
                ),
            )
        )
    result = dict(status="complete", state_unchanged=True, episodes=rows)
    HOOKS.setdefault("observed", []).append(result)
    return result


@pytest.fixture(autouse=True)
def scripted_only(monkeypatch):
    HOOKS.clear()
    TRACE.clear()
    box = bind_collection(monkeypatch)
    monkeypatch.setattr(port, "ContextLinearLearner", LiteralAffine)
    monkeypatch.setattr(app, "ContextLinearLearner", LiteralAffine)
    monkeypatch.setattr(app, "evaluate_context_linear_stream_checkpoint", scripted_readout)

    def forbidden(*args, **kwargs):
        WORK["forbidden"] += 1
        raise AssertionError("genuine numerical/native work forbidden")

    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    monkeypatch.setattr(port, "fit_categorical_linear", forbidden)
    monkeypatch.setattr(port, "predict_categorical_linear", forbidden)
    yield box
    WORK["collection_envs"] += len(box["environments"])
    WORK["teacher_steps"] += box["teacher_calls"]
    assert all(e.closed for e in box["environments"])


def run(mode="periodic"):
    return app.run_fresh_adaptation(reference_policy=ReferencePolicy(mode))


@pytest.mark.parametrize("count", range(7))
def test_should_decode_actual_unfitted_odd_and_even_heads_without_fitting(count):
    state = literal_state(count)
    decoded = cli._context_linear_training_snapshot(asdict(state))
    assert decoded == state and decoded is not state
    assert [None if h is None else h.fit_rows for h in decoded.heads] == [
        None if h is None else h.fit_rows for h in state.heads
    ]
    assert not TRACE


@pytest.mark.parametrize(
    "mode,counts,rows,prior,readouts",
    [
        ("no_learning", [0, 0, 0], 0, 0, 1),
        ("periodic", [1, 2, 3], 36, 9, 3),
        ("budget_aware", [0, 0, 2], 18, 0, 2),
        ("drift", [2, 4, 6], 90, 36, 3),
    ],
)
def test_should_use_actual_partial_fifo_states_after_complete_window_and_keep_pending(
    mode, counts, rows, prior, readouts, scripted_only
):
    before = dict(WORK)
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    result = run(mode)
    assert result["status"] == "complete", result.get("reason")
    training = result["training"]
    assert training["stream"]["window_complete"] is True
    assert training["stream"]["training_complete"] is (counts[-1] == 6)
    assert training["stream"]["pending_unapplied"] == 6 - counts[-1]
    assert training["catalogue"]["committed_rows"] == dict(
        arrivals=counts[-1] * 9, replay=prior, native=rows
    )
    assert training["score_calls_completed"] == 0 and training["endpoints"] == []
    assert [p["updates"] for p in training["unscored_checkpoints"]] == list(range(counts[-1] + 1))
    assert [p["tick"] for p in result["games"]] == [2, 4, 6]
    assert [p["updates"] for p in result["games"]] == counts
    assert WORK["readouts"] - before["readouts"] == readouts
    assert result["summary"]["complete"] is True
    assert result["summary"]["learning_recovery"] == [None, None]
    for phase in result["games"]:
        if phase["readout_reused_from_tick"] is not None:
            original = next(
                p for p in result["games"] if p["tick"] == phase["readout_reused_from_tick"]
            )
            assert phase["games"] == original["games"]
    assert len(scripted_only["environments"]) == 2 and scripted_only["teacher_calls"] == 18
    assert result["readout_budget"]["initial_limits"] == result["readout_budget"]["limits"]
    assert result["readout_budget"]["started_at"] <= training["budget"]["started_at"]
    assert result["readout_budget"]["updates"] == result["readout_budget"]["replay"] == 0
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng


def test_should_preserve_original_source_values_labels_delays_and_fifo_joint_exposures():
    packets = sources()
    frozen = pickle.dumps(packets)
    report = cli.run_context_linear_stream(
        packets,
        a4_training=True,
        retain_arrived=True,
        reference_policy=ReferencePolicy("periodic"),
        ready_feedback=True,
    )
    assert report["status"] == "complete", report.get("reason")
    assert [len(x) for x, y in TRACE] == [9, 9, 18]
    assert [set(y[:, 0]) for x, y in TRACE] == [{2}, {2}, {2, 5}]
    assert [
        (r["observed_at"], r["arrived_at"], r["applied_at"]) for r in report["committed_receipts"]
    ] == [(0, 1, 2), (1, 2, 4), (2, 3, 6)]
    assert (
        pickle.dumps(packets) == frozen and report["source_unchanged"] and report["actor_unchanged"]
    )


def test_should_not_skip_correct_oldest_to_force_new_task_training():
    HOOKS["class"] = 2
    result = run("drift")
    assert result["status"] == "complete"
    assert result["training"]["stream"]["pending_unapplied"] == 6
    assert [e["batch_index"] for e in result["training"]["stream"]["ready_feedback"]["events"]] == [
        0
    ] * 6
    assert not TRACE and all(p["updates"] == 0 for p in result["games"])


@pytest.mark.parametrize(
    "fault", ["status", "rows", "receipt", "decision", "missing", "state", "unknown", "bool_count"]
)
def test_should_refuse_incomplete_or_changed_training_before_any_readout(monkeypatch, fault):
    original = cli.run_context_linear_stream

    def corrupt(*args, **options):
        r = original(*args, **options)
        if fault == "status":
            r["status"] = "failed"
        elif fault == "rows":
            r["catalogue"]["committed_rows"]["replay"] += 1
        elif fault == "receipt":
            r["committed_receipts"][0]["arrived_at"] = 0
        elif fault == "decision":
            r["stream"]["decisions"][2]["completed_updates"] = 0
        elif fault == "missing":
            r["unscored_checkpoints"].pop()
        elif fault == "unknown":
            r["stream"]["unreturned_work_unknown"] = True
        elif fault == "bool_count":
            r["unscored_checkpoints"][0]["updates"] = False
        else:
            r["unscored_checkpoints"][1]["state"]["training_examples"] += 1
        return r

    monkeypatch.setattr(cli, "run_context_linear_stream", corrupt)
    before = WORK["readouts"]
    r = run()
    assert r["status"] == "failed" and r["games"] == [] and WORK["readouts"] == before


@pytest.mark.parametrize("fault", ["mutation", "replacement", "budget_bool", "budget_limits"])
def test_should_preserve_prior_success_and_stop_before_admitting_more_work(fault):
    before = WORK["readouts"]

    def corrupt(budget, boundary):
        if WORK["readouts"] - before == 2:
            if fault == "mutation":
                HOOKS["observed"][0]["episodes"][0]["success"] = False
            elif fault == "replacement":
                row = HOOKS["observed"][0]["episodes"][0]
                row["environment"] = deepcopy(row["environment"])
            elif fault == "budget_bool":
                budget.updates_completed = False
            else:
                budget.budget = replace(budget.budget, max_wall_seconds=151)
            boundary()

    HOOKS["readout"] = corrupt
    result = run()
    assert result["status"] == "failed" and len(result["games"]) == 1
    assert result["games"][0]["games"]["episodes"][0]["success"] is True
    assert WORK["readouts"] - before == 2
    if fault == "budget_limits":
        assert result["readout_budget"]["initial_limits"]["max_wall_seconds"] == 150
        assert result["readout_budget"]["limits"]["max_wall_seconds"] == 151


@pytest.mark.parametrize("role", ["probe_train", "probe_test", "final_test"])
def test_should_refuse_other_roles_before_model_work(role):
    packets = sources()
    before = dict(WORK)
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(
            (replace(packets[0], role=role), *packets[1:]),
            a4_training=True,
            retain_arrived=True,
            reference_policy=ReferencePolicy("periodic"),
        )
    assert WORK == before


@pytest.mark.parametrize(
    "options",
    [
        dict(retain_arrived=False),
        dict(a4_current_only=True),
        dict(endpoint_observer=lambda s: None),
        dict(a4_training=1),
    ],
)
def test_should_refuse_conflicting_reference_recipes_before_source(options):
    before = dict(WORK)
    keywords: dict[str, Any] = dict(
        a4_training=True, retain_arrived=True, reference_policy=ReferencePolicy("periodic")
    )
    keywords.update(options)
    with pytest.raises(ValueError):
        cli.run_context_linear_stream(object(), **keywords)
    assert WORK == before


def invoke(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["affine", *args])
    cli.main()


@pytest.mark.parametrize("mode", ["no_learning", "periodic", "budget_aware", "drift"])
def test_should_describe_before_source_collection_or_learner(monkeypatch, capsys, mode):
    before = dict(WORK)
    monkeypatch.setattr(app, "_collect_fresh_a4", lambda *a: pytest.fail("collection"))
    invoke(
        monkeypatch,
        [
            "--example-affine-a4",
            "--context-linear-reference",
            mode,
            "--ready-feedback",
            "--describe",
        ],
    )
    protocol = json.loads(capsys.readouterr().out)
    assert protocol["historical_source_reads"] == 0
    assert protocol["training"]["reference_policy"]["mode"] == mode
    assert protocol["training"]["fit_rows_upper_bound"] == 90 and WORK == before


def test_should_execute_actual_cli_and_keep_output_exclusive(monkeypatch, tmp_path, capsys):
    output = tmp_path / "reference.json"
    args = [
        "--example-affine-a4",
        "--context-linear-reference",
        "budget_aware",
        "--ready-feedback",
        "--output",
        str(output),
    ]
    invoke(monkeypatch, args)
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["status"] == "complete" and report["training"]["stream"]["pending_unapplied"] == 4
    before = dict(WORK)
    with pytest.raises(FileExistsError):
        invoke(monkeypatch, args)
    assert WORK == before


@pytest.mark.parametrize(
    "extra",
    [
        ["--source", "literal"],
        ["--fit-updates", "6"],
        ["--prequential-feedback"],
        ["--reference-retain-arrived"],
    ],
)
def test_should_refuse_irrelevant_options_before_collection(monkeypatch, extra):
    before = dict(WORK)
    with pytest.raises(SystemExit) as error:
        invoke(
            monkeypatch,
            [
                "--example-affine-a4",
                "--context-linear-reference",
                "periodic",
                "--ready-feedback",
                "--describe",
                *extra,
            ],
        )
    assert error.value.code == 2 and WORK == before


def test_should_preserve_original_fixed_six_fresh_default():
    r = app.run_fresh_adaptation()
    assert r["status"] == "complete", r.get("reason")
    assert [p["updates"] for p in r["games"]] == [2, 4, 6]
    assert all("tick" not in p for p in r["games"])
    assert r["training"]["catalogue"]["committed_rows"] == dict(arrivals=54, replay=36, native=90)
    assert r["summary"]["learning_recovery"] == [False, False]


@pytest.mark.parametrize(
    "extra",
    [
        "--example-affine-published",
        "--context-linear-a4-adaptation",
        "--example-affine-recurring",
        "--bounds-only",
    ],
)
def test_should_keep_other_modes_exclusive_before_any_delegation(monkeypatch, extra):
    before = dict(WORK)
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, ["--example-affine-a4", extra, "--describe"])
    assert error.value.code == 2 and WORK == before


def test_should_require_explicit_ready_feedback_for_fresh_reference(monkeypatch):
    before = dict(WORK)
    with pytest.raises(SystemExit) as error:
        invoke(
            monkeypatch,
            ["--example-affine-a4", "--context-linear-reference", "periodic", "--describe"],
        )
    assert error.value.code == 2 and WORK == before


def test_should_freeze_original_policy_before_fresh_source_collection(monkeypatch):
    policy = ReferencePolicy("periodic")
    original = app._collect_fresh_a4

    def change(boundary, records):
        object.__setattr__(policy, "period", 3)
        return original(boundary, records)

    monkeypatch.setattr(app, "_collect_fresh_a4", change)
    before = WORK["fits"]
    result = app.run_fresh_adaptation(reference_policy=policy)
    assert result["status"] == "failed" and "policy changed" in (
        result.get("reason") or result.get("capture_error")
    )
    assert WORK["fits"] == before


@pytest.mark.parametrize("fault", ["step", "close"])
def test_should_keep_failed_collection_in_exclusive_cli_output_without_training(
    monkeypatch, tmp_path, capsys, scripted_only, fault
):
    scripted_only["fault"] = fault
    before = WORK["fits"]
    output = tmp_path / "failed.json"
    with pytest.raises(SystemExit) as error:
        invoke(
            monkeypatch,
            [
                "--example-affine-a4",
                "--context-linear-reference",
                "periodic",
                "--ready-feedback",
                "--output",
                str(output),
            ],
        )
    assert error.value.code == 2
    report = json.loads(output.read_text())
    assert report == json.loads(capsys.readouterr().out)
    assert report["status"] == "failed" and report["games"] == [] and WORK["fits"] == before
    assert report["collection"] and report["collection"][0]["closed"] is True
