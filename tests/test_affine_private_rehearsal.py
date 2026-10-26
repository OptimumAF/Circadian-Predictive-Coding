"""Fixed synthetic correctness fixtures; their scores aren't task evidence."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_rehearsal as study
from src.adapters.minigrid_context_linear_learner import ContextLinearLearner
from src.core.learner_ports import TrainingUpdateFailed
from test_after_c_linear_games import state
from test_catalogue_capacity import sources
from test_minigrid_affine_probe import raw_episode


def probe_sources():
    return tuple(
        raw_episode(f"selected-{role}-{name}", role, cue, f"F-{name}")
        for role in ("probe_train", "probe_test")
        for name, cue in (("blue", 2), ("red", 0))
    )


@pytest.fixture(scope="module")
def prepared():
    old, probes = sources(), probe_sources()
    witness, monitors, spans = study.original._prepare_monitors(old)
    training, evaluation = study.prepare_selected_probe(probes)
    return old, probes, witness, monitors, spans, training, evaluation


@pytest.mark.parametrize("fault", ["count", "role", "context", "width", "classes", "cue"])
def test_should_refuse_wrong_selected_role_or_native_frame_contract(fault):
    data = probe_sources()
    if fault == "count":
        data = data[:2]
    elif fault == "role":
        data = (replace(data[0], role="train"), *data[1:])
    elif fault == "context":
        data = (replace(data[0], context="E-blue"), *data[1:])
    elif fault == "width":
        e = data[0]
        data = (
            replace(
                e,
                transitions=tuple(
                    replace(
                        t, observation=t.observation * 2, next_observation=t.next_observation * 2
                    )
                    for t in e.transitions
                ),
            ),
            *data[1:],
        )
    elif fault == "classes":
        data = (object(), *data[1:])
    else:
        data = (replace(data[1], context="F-blue", episode_id="wrong-public-cue"), *data[1:])
    with pytest.raises(ValueError):
        study.prepare_selected_probe(data)


@pytest.mark.parametrize("arm", study.ARMS)
def test_should_compose_equal_rows_with_explicit_roles_disjoint_storage_and_costs(prepared, arm):
    old, probes, witness, monitors, spans, training, evaluation = prepared
    before = pickle.dumps((old, probes, training, evaluation, monitors))
    batches, plan = study.compose_training(training, monitors, spans, arm)
    study._validate_batches(batches, evaluation, monitors)
    assert [b.features.shape for b in batches] == [(36, 296)] * 2
    assert plan["input_array_bytes"] == 72 * 297 * 8
    assert (
        sum(b.features.size + b.targets.size for b in (*batches, *evaluation, *monitors)) == 42768
    )
    for i, batch in enumerate(batches):
        assert batch.role == "probe_train" and batch.episode is None
        assert len(set(batch.row_ids)) == 36
        roles = [s["role"] for s in plan["wakes"][i]["spans"]]
        assert roles == (
            ["probe_train"] * 4 if arm == "current_only" else ["train"] * 3 + ["probe_train"]
        )
        assert "probe_test" not in roles
        expected = (
            np.tile(training[i].targets, (4, 1))
            if arm == "current_only"
            else np.concatenate((monitors[i].targets, training[i].targets))
        )
        np.testing.assert_array_equal(batch.targets, expected)
        assert not any(
            np.shares_memory(batch.features, b.features)
            or np.shares_memory(batch.targets, b.targets)
            for b in (*training, *evaluation, *monitors)
        )
    assert pickle.dumps((old, probes, training, evaluation, monitors)) == before
    witness.check(old)


def test_should_keep_test_labels_out_of_both_fit_recipes(prepared):
    _, _, _, monitors, spans, training, evaluation = prepared
    tests = tuple(replace(b, targets=np.full_like(b.targets, 6)) for b in evaluation)
    for arm in study.ARMS:
        batches, _ = study.compose_training(training, monitors, spans, arm)
        study._validate_batches(batches, tests, monitors)
        assert all(6 not in b.targets for b in batches)


@pytest.mark.parametrize("target", ["training", "monitor"])
def test_should_refuse_test_roles_before_composing_any_training_matrix(prepared, target):
    _, _, _, monitors, spans, training, _ = prepared
    if target == "training":
        training = (replace(training[0], role="probe_test"), training[1])
    else:
        monitors = (replace(monitors[0], role="final_test"), monitors[1])
    with pytest.raises(ValueError, match="probe_train"):
        study.compose_training(training, monitors, spans, "joint_ABC_F")


def test_should_execute_both_real_private_two_fit_recipes_without_source_prefix_refits():
    old, probes, saved = sources(), probe_sources(), state()
    before = pickle.dumps((old, probes, saved, np.random.get_state()))
    result = study.compare_private_rehearsal(old, saved, probes)
    assert result["status"] == "complete", result
    assert [a["arm"] for a in result["arms"]] == list(study.ARMS)
    assert result["budget"]["updates_completed"] == 4
    assert result["budget"]["replay_examples_completed"] == 54
    for i, arm in enumerate(result["arms"]):
        assert arm["source_native_state_unchanged"] is True and arm["source_fork_calls"] == 1
        endpoint = arm["native_endpoint"]
        assert (endpoint["completed_updates"], endpoint["training_examples"]) == (8, 180)
        assert [h["fit_rows"] for h in endpoint["heads"]] == [36, 36]
        assert arm["budget"]["started_at"] == result["budget"]["started_at"]
        assert [p["updates"] for p in arm["probe"]["checkpoints"]] == [0, 1, 2]
        assert len(arm["old_task_readouts"][-1]["contexts"]) == 6
        assert sum(a["old_TRAIN_reuse_charged"] for a in arm["attempts"]) == (54 if i else 0)
        assert sum(a["extra_F_exposures_charged"] for a in arm["attempts"]) == (0 if i else 54)
    assert pickle.dumps((old, probes, saved, np.random.get_state())) == before


@pytest.mark.parametrize("fault", ["updates", "rows", "head", "failed"])
def test_should_refuse_wrong_warm_state_before_forks(monkeypatch, fault):
    saved = state()
    if fault == "updates":
        saved = replace(saved, completed_updates=5)
    elif fault == "rows":
        saved = replace(saved, training_examples=107)
    elif fault == "head":
        saved = replace(
            saved,
            heads=tuple(replace(h, fit_rows=18, singular_values=(0.0,) * 18) for h in saved.heads),
        )
    else:
        saved = replace(saved, failed_work=(1, 36, 0, 0, True))
    monkeypatch.setattr(study, "_PrivateSource", lambda *a: pytest.fail("source reached"))
    result = study.compare_private_rehearsal(sources(), saved, probe_sources())
    assert result["status"] == "failed" and result["arms"] == []


def closed_value(callback, name):
    return dict(
        zip(callback.__code__.co_freevars, (c.cell_contents for c in callback.__closure__))
    )[name]


def setup_fake(monkeypatch, prepared, fault=None):
    old, probes, witness, monitors, spans, training, evaluation = prepared
    monkeypatch.setattr(study.original, "_prepare_monitors", lambda *a: (witness, monitors, spans))
    monkeypatch.setattr(study, "prepare_selected_probe", lambda *a: (training, evaluation))
    monkeypatch.setattr(study.original, "_old_task_rows", lambda *a: [])
    calls: list[Any] = []

    def train(self, features, targets):
        # Literal state transitions test accounting only, not solver behavior.
        before = self.snapshot_state()
        cue = int(study.context_public_history(tuple(map(float, features[0])))[297])
        heads = list(before.heads)
        if fault == "fit_failure" and len(calls) == 2:
            self._state = replace(before, failed_work=(1, 36, 0, 0, True))
            raise TrainingUpdateFailed(
                "synthetic lost solve", "synthetic", dict(fit_calls_attempted=1)
            )
        heads[cue] = replace(heads[cue], fit_rows=36, singular_values=(0.0,) * 36)
        self._state = replace(
            before,
            heads=tuple(heads),
            completed_updates=before.completed_updates + 1,
            training_examples=before.training_examples + 36,
        )

    monkeypatch.setattr(ContextLinearLearner, "train_batch", train)

    def helper(source, saved, batches, tests, *, budget, check_boundary, **options):
        calls.append(source)
        outer = closed_value(check_boundary, "check_boundary")
        root = closed_value(outer, "report")
        if len(calls) == 2 and fault in ("prior_value", "prior_equal", "prior_prefix"):
            if fault == "prior_value":
                root["arms"][0]["budget"]["updates_completed"] = 0
            elif fault == "prior_equal":
                root["arms"][0]["attempts"] = deepcopy(root["arms"][0]["attempts"])
            else:
                root["arms"] = deepcopy(root["arms"])
        if fault == "future_clock":
            closed_value(outer, "sessions")[1].clock = lambda: 1.0
        if fault == "study_clock":
            closed_value(outer, "study").started_at = -1.0
        check_boundary()
        child = source.fork()
        child.restore_state(saved)
        if fault == "unobserved_parameters":
            s = child.snapshot_state()
            h = s.heads[0]
            weights = tuple(tuple(v + 0.1 for v in row) for row in h.coefficients)
            child._state = replace(s, heads=(replace(h, coefficients=weights), s.heads[1]))
        if fault == "counter_type":
            budget.updates_completed = False
        if fault == "arm_clock":
            budget.started_at = -1.0
        check_boundary()
        for b in batches:
            x, y = b.features.copy(), b.targets.copy()
            if fault == "fit_input":
                y[0, 0] = 6.0
            child.train_batch(x, y)
            budget.record_update()
            check_boundary()
            if fault == "joint_replay_regression" and len(calls) == 2:
                budget.replay_examples_completed = 0
                check_boundary()
        points = [dict(updates=n, scores={"F-blue": {}, "F-red": {}}) for n in (0, 1, 2)]
        p: dict[str, Any] = dict(
            status="complete",
            completed_updates=2,
            completed_training_examples=72,
            checkpoints=points,
            training_readouts=dict(
                checkpoints=[
                    dict(updates=n, status="complete", scores={"0": {}, "1": {}}) for n in (0, 1, 2)
                ]
            ),
        )
        if fault == "incomplete":
            p["status"] = "failed"
        if fault == "missing_scores":
            p["checkpoints"].pop()
        if fault == "after_fit_budget":
            budget.updates_completed = 0
        return p

    monkeypatch.setattr(study, "run_fresh_task_probe", helper)
    return calls, old, probes


def test_should_preserve_two_finite_preallocated_arm_lifetimes_and_native_charges(
    monkeypatch, prepared
):
    calls, old, probes = setup_fake(monkeypatch, prepared)
    r = study.compare_private_rehearsal(old, state(), probes)
    assert r["status"] == "complete" and len(calls) == 2
    assert r["budget"]["updates_completed"] == 4
    assert r["budget"]["replay_examples_completed"] == 54
    assert [a["budget"]["replay_examples_completed"] for a in r["arms"]] == [0, 54]
    from src.app.fresh_task_probe import _validate_controls

    session = study.ToyBudgetSession(
        study.ToyExecutionBudget(
            max_training_updates=2, max_replay_examples=54, max_wall_seconds=90
        ),
        lambda: 0.0,
    )
    _validate_controls(session, (0, 1, 2), 2, lambda: 0.0)
    assert all(a["budget"]["started_at"] == r["budget"]["started_at"] for a in r["arms"])
    assert all(a["elapsed_seconds"] > 0 for a in r["arms"])


@pytest.mark.parametrize(
    "fault",
    [
        "prior_value",
        "prior_equal",
        "prior_prefix",
        "future_clock",
        "study_clock",
        "unobserved_parameters",
        "counter_type",
        "arm_clock",
        "fit_input",
        "incomplete",
        "missing_scores",
        "after_fit_budget",
    ],
)
def test_should_stop_on_corrupted_prior_source_clock_inputs_or_incomplete_diagnostics(
    monkeypatch, prepared, fault
):
    calls, old, probes = setup_fake(monkeypatch, prepared, fault)
    r = study.compare_private_rehearsal(old, state(), probes)
    assert r["status"] == "failed"
    if fault.startswith("prior_"):
        assert len(calls) == 2 and len(r["arms"]) == 2
        assert r["arms"][0]["status"] == "complete"
        assert r["arms"][0]["budget"]["updates_completed"] == 2
    else:
        assert len(calls) == 1 and len(r["arms"]) == 1
    assert r["budget"]["started_at"] >= 0


def test_should_keep_attempted_old_reuse_charge_and_failed_native_work_without_another_fit(
    monkeypatch, prepared
):
    calls, old, probes = setup_fake(monkeypatch, prepared, "fit_failure")
    r = study.compare_private_rehearsal(old, state(), probes)
    assert r["status"] == "failed" and len(calls) == 2
    failed = r["arms"][1]
    assert len(failed["attempts"]) == 1
    assert failed["attempts"][0]["old_TRAIN_reuse_charged"] == 27
    assert failed["committed_old_TRAIN_reuse_rows"] == 0
    assert failed["native_endpoint"]["failed_work"] == (1, 36, 0, 0, True)
    assert failed["budget"]["updates_completed"] == 0
    assert r["budget"]["updates_completed"] == 2


def test_should_refuse_replay_counter_erasure_after_a_paid_joint_update(monkeypatch, prepared):
    _, old, probes = setup_fake(monkeypatch, prepared, "joint_replay_regression")
    r = study.compare_private_rehearsal(old, state(), probes)
    assert r["status"] == "failed" and len(r["arms"]) == 2
    assert r["budget"]["updates_completed"] == 3
    assert r["budget"]["replay_examples_completed"] == 27
    last = r["arms"][1]
    assert (
        last["budget"]["replay_examples_completed"] == 0
    )  # Observed corruption, not trusted cost.
    assert last["committed_old_TRAIN_reuse_rows"] == 27
    assert last["attempts"][0]["old_TRAIN_reuse_charged"] == 27


@pytest.mark.parametrize("mode", ["describe", "conflict", "fit", "retained"])
def test_should_preserve_cli_no_io_description_and_fixed_mode_errors(monkeypatch, capsys, mode):
    args = {
        "describe": ["--describe"],
        "conflict": ["--context-linear-probe"],
        "fit": ["--fit-updates", "1"],
        "retained": ["--source", "probe"],
    }[mode]
    monkeypatch.setattr("sys.argv", ["control", "--context-linear-rehearsal", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN read"))
    if mode == "describe":
        cli.main()
        r = json.loads(capsys.readouterr().out)
        assert r["total_new_fit_rows"] == 144 and r["joint_old_TRAIN_reuse_exposures"] == 54
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_route_rehearsal_mode_and_save_failed_output_exclusively(monkeypatch, tmp_path):
    path = tmp_path / "failed.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-rehearsal",
            "--source",
            "probe",
            "--retained-source",
            "before",
            "--output",
            str(path),
        ],
    )
    monkeypatch.setattr(
        cli, "read_training_sources", lambda *a, **k: pytest.fail("generic TRAIN read")
    )
    monkeypatch.setattr(
        cli, "run_saved_context_linear_rehearsal", lambda *a: dict(status="failed", arms=[])
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2 and path.is_file()
    with pytest.raises(FileExistsError):
        cli.main()


@pytest.mark.parametrize("kind", ["wrong", "oversized"])
def test_should_refuse_changed_source_before_probe_or_owner_work(tmp_path, monkeypatch, kind):
    path = tmp_path / "source.json"
    path.write_bytes(b"{}" if kind == "wrong" else b"x" * (1024**2 + 1))
    monkeypatch.setattr(
        study, "compare_private_rehearsal", lambda *a, **k: pytest.fail("study reached")
    )
    r = cli.run_saved_context_linear_rehearsal(path, path)
    assert r["status"] == "failed" and r["arms"] == []


def test_should_decode_actual_saved_four_sources_and_exact_bundled_train_without_fits(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    before = root / "artifacts/runs/catalogue-capacity-new-c-20261024/actual.json"
    probe = root / "artifacts/runs/affine-fresh-task-probe-20261024/actual.json"
    if not before.is_file() or not probe.is_file():
        pytest.skip("accepted local111/113 sources unavailable")

    def run(old, saved, selected, *, check_boundary):
        assert len(old) == 6 and saved.completed_updates == 6 and len(selected) == 4
        assert [e.role for e in selected] == ["probe_train"] * 2 + ["probe_test"] * 2
        assert all(len(t.observation) == 148 for e in selected for t in e.transitions)
        check_boundary()
        return dict(status="complete", arms=[])

    monkeypatch.setattr(study, "compare_private_rehearsal", run)
    r = cli.run_saved_context_linear_rehearsal(probe, before)
    assert r["status"] == "complete" and r["original_probe_source_fork_calls"] == 1
    assert r["original_probe_budget"]["updates_completed"] == 2
