"""New task/head probe on literal sources; original legacy route preserved."""

from copy import deepcopy
from dataclasses import asdict, replace
from hashlib import sha256
import json

import numpy as np
import pytest

from scripts import context_head_probe as head
from scripts import probe_ticket_policy_learning as recipe
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.context_perceptron import ContextPerceptronPolicy, ContextPerceptronSnapshot
from test_context_catalog_control import initial


@pytest.fixture
def spec():
    return head.DecisionFeatureSpec(head.CONTEXT_FEATURES, head.CATALOG)


@pytest.fixture
def literal_source(monkeypatch):
    """Handwritten fixture, never decoded from a qualified scientific record."""
    points = []
    for n in (0, 4, 8, 12):
        if n == 12:
            state = initial()
        else:
            coeff = tuple((0.0,) * 6 for _ in range(8))
            state = ContextPerceptronSnapshot(
                ContextPerceptronPolicy(9, 6, (4, 5)),
                (
                    PerceptronSnapshot(PerceptronPolicy(7, 6), coeff, n, 6 * n, n),
                    PerceptronSnapshot(PerceptronPolicy(7, 6), coeff, 0, 0, 0),
                ),
                n,
                6 * n,
                n,
            )
        points.append(dict(source_updates=n, source_rows=6 * n, state=asdict(state)))
    raw = json.dumps(
        dict(
            complete=True, definition="separate_context_head_ticket_workflow_v1", checkpoints=points
        )
    ).encode()
    monkeypatch.setattr(head, "HEAD_SOURCE_SHA", sha256(raw).hexdigest())
    monkeypatch.setattr(head, "_source_bytes", lambda path: raw)
    return raw


def test_should_prepare_new_pair_with_owned_roles_and_accurate_source_overlap(spec):
    cohorts = head.build_head_cohorts(spec, head_task_transfer=True)
    training, evaluation, monitors, returning = cohorts
    assert [b.context for b in training] == ["parcel-v1", "password-v2"] * 8
    assert len(training) == 16 and all(b.role == "probe_train" for b in training)
    assert len(evaluation) == len(monitors) == len(returning) == 2
    assert all(b.role == "probe_test" for b in evaluation)
    assert all(b.role == "train" and len(b.features) == 6 for b in (*monitors, *returning))
    batches = (*training, *evaluation, *monitors, *returning)
    assert len({i for b in batches for i in b.row_ids}) == 78
    arrays = [a for b in batches for a in (b.features, b.targets)]
    assert sum(a.nbytes for a in arrays) == 6240
    assert all(not np.shares_memory(a, b) for i, a in enumerate(arrays) for b in arrays[i + 1 :])
    for i, b in enumerate(training):
        assert np.array_equal(b.features, evaluation[i % 2].features)
        assert (b.features[:, (4, 5)] == ((1, 0) if i % 2 == 0 else (0, 1))).all()
    overlap = head._transfer_manifest(cohorts)
    assert [(r["policy_version"], r["concept"]) for r in overlap["fresh_tasks"]] == list(
        head.TRANSFER_TASKS
    )
    assert all(
        not r["source_optimizer_task_seen"]
        and r["concept_seen_in_another_source_head"]
        and r["shared_input_keys"] == 2
        and r["shared_label_conflicts"] == 0
        for r in overlap["fresh_tasks"]
    )


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_should_refuse_nonboolean_mode_before_teacher_or_source(value, spec, monkeypatch):
    monkeypatch.setattr(head, "training_cases", lambda **kw: pytest.fail("teacher entered"))
    monkeypatch.setattr(head, "_source_bytes", lambda path: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        head.build_head_cohorts(spec, head_task_transfer=value)
    with pytest.raises(ValueError):
        recipe.run_case("unused", separate_context_heads=True, head_task_transfer=value)


def test_should_require_head_backend_before_io_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(head, "_source_bytes", lambda path: pytest.fail("source opened"))
    with pytest.raises(ValueError):
        recipe.run_case("unused", head_task_transfer=True)
    output = tmp_path / "invalid.json"
    with pytest.raises(SystemExit):
        recipe.main(["--input", "unused", "--output", str(output), "--head-task-transfer"])
    assert not output.exists()


@pytest.mark.parametrize("role", ["final_test", "probe_train", "inner_guard", "outer_selection"])
def test_should_never_relabel_foreign_teacher_roles_as_new_probe(spec, role, monkeypatch):
    original = head.training_cases
    monkeypatch.setattr(
        head, "training_cases", lambda **kw: tuple(replace(c, role=role) for c in original(**kw))
    )
    with pytest.raises(ValueError):
        head.build_head_cohorts(spec, head_task_transfer=True)


@pytest.mark.parametrize("selected", [False, True])
def test_should_run_ordinary_private_probe_and_keep_original_default_contract(
    literal_source, selected
):
    result = recipe.run_case(
        "unused", separate_context_heads=True, **(dict(head_task_transfer=True) if selected else {})
    )
    assert result["complete"], result.get("error")
    assert result["training_work"] == dict(
        attempted_updates=64, attempted_rows=288, returned_updates=64, unreturned_work_unknown=False
    )
    assert (
        result["source_owners_unchanged"]
        and result["source_bytes_unchanged"]
        and result["numpy_rng_unchanged"]
    )
    assert result["final_examples"] == result["replay_examples"] == result["promotions"] == 0
    tasks = recipe.TRANSFER_HEAD_TASKS if selected else recipe.HEAD_TASKS
    phases = (
        ("before_probe", "after_probe", "after_return")
        if selected
        else ("after_probe", "after_return")
    )
    assert ("source_task_overlap" in result) is selected
    for arm in result["cases"]:
        recipe._require_arm(
            arm, separate_context_heads=True, **(dict(head_tasks=tasks) if selected else {})
        )
        assert [(r["phase"], r["policy_version"], r["concept"]) for r in arm["episodes"]] == [
            (p, v, c) for p in phases for v, c in tasks
        ]
        assert [s["updates"] for s in arm["sessions"]] == [16, 16]
        assert [r["updates"] for r in arm["probe"]["checkpoints"]] == [0, 4, 8, 16]
        if selected:
            first = arm["readout_work"][0]
            assert (
                first["phase"] == "before_probe"
                and first["native_updates"] == arm["prior_native_updates"]
            )
            assert first["native_rows"] == arm["prior_native_rows"]
            assert "input_diagnostics" in arm["probe"]


def test_should_stop_on_failed_initial_readout_before_probe_updates(literal_source, monkeypatch):
    monkeypatch.setattr(
        recipe,
        "ContextPerceptronDecisionPredictor",
        lambda *a: (_ for _ in ()).throw(RuntimeError("lost baseline read")),
    )
    result = recipe.run_case("unused", separate_context_heads=True, head_task_transfer=True)
    assert not result["complete"] and "lost baseline read" in result["error"]
    assert len(result["cases"]) == 1 and result["training_work"]["attempted_updates"] == 0
    assert [s["updates"] for s in result["cases"][0]["sessions"]] == [0, 0]


def test_should_keep_failed_native_cost_and_refuse_fresh_arm(literal_source, monkeypatch):
    original = head.ContextPerceptronLearner.train_batch

    def failed(owner, features, targets):
        original(owner, features, targets)
        raise RuntimeError("lost transfer native return")

    monkeypatch.setattr(head.ContextPerceptronLearner, "train_batch", failed)
    result = recipe.run_case("unused", separate_context_heads=True, head_task_transfer=True)
    assert not result["complete"] and len(result["cases"]) == 1
    assert result["training_work"] == dict(
        attempted_updates=1, attempted_rows=3, returned_updates=0, unreturned_work_unknown=True
    )
    assert result["cases"][0]["probe"]["status"] != "complete"


def test_should_reject_budget_counter_change_during_baseline_before_fitting(monkeypatch, spec):
    state = initial()
    seed = head.ContextPerceptronLearner(state.policy)
    seed.restore_state(state)
    from src.app.toy_execution_budget import ToyBudgetSession

    sessions = [ToyBudgetSession(recipe.HEAD_LIMITS, lambda: 0.0) for _ in range(2)]
    original = recipe.ContextPerceptronDecisionPredictor

    def changed(*args):
        sessions[0].updates_completed = 1
        return original(*args)

    monkeypatch.setattr(recipe, "ContextPerceptronDecisionPredictor", changed)
    row = dict(
        prior_native_updates=12, prior_native_rows=72, episodes=[], readout_work=[], mode="warm"
    )
    with pytest.raises(ValueError):
        recipe._observe_initial_tasks(
            seed, sessions, row, spec, lambda: None, recipe.TRANSFER_HEAD_TASKS
        )
    assert seed.snapshot_state() == state and not row["episodes"]


def test_should_dispatch_new_cli_flag_without_model_work(tmp_path, monkeypatch):
    seen = []

    def run(path, **kw):
        seen.append(kw)
        return dict(
            complete=False, error="scripted failure", training_work=dict(attempted_updates=0)
        )

    monkeypatch.setattr(recipe, "run_case", run)
    out = tmp_path / "cli.json"
    assert (
        recipe.main(
            [
                "--input",
                "unused",
                "--output",
                str(out),
                "--separate-context-heads",
                "--head-task-transfer",
            ]
        )
        == 1
    )
    assert seen == [dict(separate_context_heads=True, head_task_transfer=True)]
    assert not json.loads(out.read_bytes())["complete"]


def test_should_validate_actual_warm_arm_and_reject_changed_endpoint_metadata(spec):
    from src.shared.process_memory import ProcessRssSampler

    state = initial()
    seed = head.ContextPerceptronLearner(state.policy)
    seed.restore_state(state)
    cohorts = head.build_head_cohorts(spec, head_task_transfer=True)

    def guard():
        assert seed.snapshot_state() == state

    with ProcessRssSampler(interval_seconds=0.02) as memory:
        row = recipe.run_arm(
            seed,
            state,
            cohorts,
            spec,
            guard,
            memory,
            "warm",
            separate_context_heads=True,
            head_tasks=recipe.TRANSFER_HEAD_TASKS,
            input_diagnostics=True,
        )
    recipe._require_arm(row, separate_context_heads=True, head_tasks=recipe.TRANSFER_HEAD_TASKS)
    for fault in ("origin", "baseline", "type", "work", "phase", "task"):
        changed = deepcopy(row)
        if fault == "origin":
            changed["prior_native_rows"] = 73
        elif fault == "baseline":
            changed["episodes"] = changed["episodes"][6:]
        elif fault == "type":
            changed["mode"] = "fresh"
            changed["prior_native_updates"] = changed["prior_native_rows"] = 0
            for reading, delta, rows in zip(
                changed["readout_work"], (0, 16, 32), (0, 48, 144), strict=True
            ):
                reading["native_updates"], reading["native_rows"] = delta, rows
            changed["readout_work"][0]["native_updates"] = False
        elif fault == "work":
            changed["readout_work"][0]["work"]["validated_calls"] -= 1
        elif fault == "phase":
            changed["readout_work"][0]["phase"] = "after_probe"
        else:
            changed["episodes"][0]["concept"] = "legacy"
        with pytest.raises(ValueError):
            recipe._require_arm(
                changed, separate_context_heads=True, head_tasks=recipe.TRANSFER_HEAD_TASKS
            )
    assert seed.snapshot_state() == state


def test_should_restore_altered_overlap_declaration_and_stop_before_fitting(
    literal_source, monkeypatch
):
    original_manifest, original_arm = head._transfer_manifest, recipe.run_arm
    holder = {}

    def manifest(cohorts):
        holder["value"] = original_manifest(cohorts)
        return holder["value"]

    def arm(*args, **kw):
        holder["value"]["fresh_tasks"][0]["source_optimizer_task_seen"] = True
        return original_arm(*args, **kw)

    monkeypatch.setattr(head, "_transfer_manifest", manifest)
    monkeypatch.setattr(recipe, "run_arm", arm)
    result = recipe.run_case("unused", separate_context_heads=True, head_task_transfer=True)
    assert not result["complete"] and "overlap declaration changed" in result["error"]
    assert result["training_work"]["attempted_updates"] == 0 and len(result["cases"]) == 1
    assert not result["source_task_overlap"]["fresh_tasks"][0]["source_optimizer_task_seen"]
