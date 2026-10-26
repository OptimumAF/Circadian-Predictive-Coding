"""Existing helper integration with literal warm heads and scripted collection."""

from dataclasses import replace
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_context_linear_learner as port
from test_catalogue_capacity import sources
from test_after_c_linear_games import state
from test_arrived_cue_catalogue import episode


def raw_episode(identity, role, cue, context):
    source = episode(identity, cue, mark=7, action=3 if cue == 2 else 4, context=context, role=role)
    return replace(
        source,
        transitions=tuple(
            replace(t, observation=t.observation[:148], next_observation=t.next_observation[:148])
            for t in source.transitions
        ),
    )


def packets():
    originals = tuple(
        binding.reserved._packet(raw_episode(f"probe-{role}-{name}", role, cue, f"F-{name}"))
        for role in ("probe_train", "probe_test")
        for name, cue in (("blue", 2), ("red", 0))
    )
    training = tuple(
        binding.reserved._packet(
            replace(binding.shared._episode(originals[i % 2]), episode_id=f"training-copy-{i}")
        )
        for i in range(16)
    )
    return training, originals[2:]


def scripted_collector(monkeypatch):
    captured = {}

    def collect(guard, reports, *, distinct_probe=False):
        assert distinct_probe is True
        for role in ("probe_train", "probe_test"):
            for cue in ("blue", "red"):
                guard()
                reports.append(
                    dict(
                        status="complete",
                        closed=True,
                        role=role,
                        cue=cue,
                        context=f"F-{cue}",
                        environment_steps=9,
                    )
                )
        captured["reports"] = reports
        return packets()

    monkeypatch.setattr(binding.reserved, "_collect_probe_sources", collect)
    return captured


def test_should_keep_two_bounded_readonly_cue_monitors_with_all_abc_source_spans():
    episodes = sources()
    before = pickle.dumps(episodes)
    witness, monitors, spans = binding._prepare_monitors(episodes)
    assert [len(b.features) for b in monitors] == [27, 27]
    assert all(b.role == "train" and b.episode is None for b in monitors)
    assert sum(b.features.size + b.targets.size for b in monitors) == 16038
    for cue, group in enumerate(spans):
        assert [s.episode_id for s in group] == [e.episode_id for e in episodes[cue::2]]
        assert [(s.start, s.stop) for s in group] == [(0, 9), (9, 18), (18, 27)]
    witness.check(episodes)
    assert pickle.dumps(episodes) == before


def test_should_admit_original_roles_before_feedforward_transport_and_preserve_four_sources():
    train, evaluation = packets()
    (selected, tests), evidence = binding._prepare_probe(train, evaluation)
    assert len(selected) == len(tests) == 2 and len(evidence) == 4
    assert all(b.episode is None and b.features.shape == (9, 296) for b in (*selected, *tests))
    assert [b.role for b in selected] == ["probe_train"] * 2
    assert [b.role for b in tests] == ["probe_test"] * 2
    assert [e["role"] for e in evidence] == ["probe_train"] * 2 + ["probe_test"] * 2
    assert all(
        len(e["transitions"]) == 9 and len(e["transitions"][0]["observation"]) == 148
        for e in evidence
    )
    assert not set(selected[0].row_ids).intersection(tests[0].row_ids)
    binding._validate_batches(selected, tests, binding._prepare_monitors(sources())[1])


@pytest.mark.parametrize("fault", ["role", "context", "count"])
def test_should_refuse_bad_raw_probe_cohort_before_history_encoding(monkeypatch, fault):
    train, tests = packets()
    if fault == "role":
        train = (replace(train[0], role="train"), *train[1:])
    elif fault == "context":
        train = (replace(train[0], context="E-blue"), *train[1:])
    else:
        train = train[:2]
    monkeypatch.setattr(
        binding.reserved, "_history_packets", lambda *a: pytest.fail("history reached")
    )
    with pytest.raises(ValueError):
        binding._prepare_probe(train, tests)


@pytest.mark.parametrize("fault", ["updates", "rows", "head", "failure"])
def test_should_refuse_non_after_c_snapshot_before_source_or_collection(monkeypatch, fault):
    saved = state()
    if fault == "updates":
        saved = replace(saved, completed_updates=4)
    elif fault == "rows":
        saved = replace(saved, training_examples=109)
    elif fault == "head":
        saved = replace(saved, heads=(None, saved.heads[1]))
    else:
        saved = replace(saved, failed_work=(1, 9, 0, 0, True))
    monkeypatch.setattr(binding, "_AffineProbeSource", lambda *a: pytest.fail("source reached"))
    with pytest.raises(ValueError):
        binding.run_affine_fresh_probe(sources(), saved)


def test_should_use_original_probe_helper_and_fit_only_disposable_f_rows(monkeypatch):
    scripted_collector(monkeypatch)
    saved, episodes = state(), sources()
    before = pickle.dumps(saved), pickle.dumps(episodes), pickle.dumps(np.random.get_state())
    fit = port.ContextLinearLearner.train_batch
    calls = []

    def train(self, x, y):
        calls.append((self.completed_updates, len(x), y[:, 0].tolist()))
        assert len(x) == 9 and set(y[:, 0]) in ({3.0}, {4.0})
        return fit(self, x, y)

    monkeypatch.setattr(port.ContextLinearLearner, "train_batch", train)
    result = binding.run_affine_fresh_probe(episodes, saved)
    assert result["status"] == "complete", result["reason"]
    assert result["source_native_state_unchanged"] is True and result["source_fork_calls"] == 1
    assert calls == [(6, 9, [3.0] * 9), (7, 9, [4.0] * 9)]
    assert (
        result["budget"]["updates_completed"] == 2
        and result["budget"]["replay_examples_completed"] == 0
    )
    assert result["budget"]["limits"] == result["initial_probe_limits"]
    assert (
        result["disposable_state"]["completed_updates"] == 8
        and result["disposable_state"]["training_examples"] == 126
    )
    assert [h["fit_rows"] for h in result["disposable_state"]["heads"]] == [9, 9]
    probe = result["probe"]
    assert probe["completed_updates"] == 2 and probe["completed_training_examples"] == 18
    assert probe["score_forward_calls_completed"] == 6
    assert probe["training_readouts"]["score_forward_calls_completed"] == 6
    assert probe["training_readouts"]["completed_rows"] == 162
    assert (
        probe["reserved_data_bytes"] == 85536
        and probe["training_readouts"]["reserved_data_bytes"] == 128304
    )
    assert [p["updates"] for p in result["old_task_readouts"]] == [0, 1, 2]
    assert all(
        set(p["contexts"]) == {e.episode_id for e in episodes} for p in result["old_task_readouts"]
    )
    assert all(
        row["rows"] == 9 for p in result["old_task_readouts"] for row in p["contexts"].values()
    )
    assert (
        pickle.dumps(saved),
        pickle.dumps(episodes),
        pickle.dumps(np.random.get_state()),
    ) == before


def test_should_keep_one_private_child_and_refuse_second_fork_without_parameter_aliasing():
    saved = state()
    source = binding._AffineProbeSource(saved)
    child = source.fork()
    child.restore_state(replace(saved, completed_updates=7, training_examples=117))
    assert source.snapshot_state() == saved and child.snapshot_state() != saved
    with pytest.raises(ValueError, match="exactly one"):
        source.fork()


def test_should_preserve_paid_probe_prefix_and_failed_native_attempt(monkeypatch):
    scripted_collector(monkeypatch)
    fit = port.fit_categorical_linear
    calls = []

    def failing(x, y, classes):
        calls.append(len(x))
        if len(calls) == 2:
            raise RuntimeError("deliberate second F fit failure")
        return fit(x, y, classes)

    monkeypatch.setattr(port, "fit_categorical_linear", failing)
    result = binding.run_affine_fresh_probe(sources(), state())
    assert result["status"] == "failed" and result["source_native_state_unchanged"] is True
    assert calls == [9, 9] and result["budget"]["updates_completed"] == 1
    assert result["probe"]["completed_updates"] == 1
    assert result["probe"]["failed_native_work"]["work"]["fit_rows_attempted"] == 9
    assert result["disposable_state"]["completed_updates"] == 7
    assert result["disposable_state"]["failed_work"] is not None
    assert [p["updates"] for p in result["old_task_readouts"]] == [0, 1]
    assert len(result["known_collection_prefix"]) == 4


@pytest.mark.parametrize("fault", ["charge", "clock", "source"])
def test_should_refuse_unobserved_budget_clock_or_source_changes_before_probe_work(
    monkeypatch, fault
):
    scripted_collector(monkeypatch)

    def probe(source, saved, *a, budget, check_boundary, **k):
        if fault == "charge":
            budget.updates_completed = 1
        elif fault == "clock":
            budget.last_clock = budget.started_at - 1
        else:
            source.restore_state(replace(saved, completed_updates=7, training_examples=117))
        check_boundary()
        pytest.fail("changed authority accepted")

    monkeypatch.setattr(binding, "run_fresh_task_probe", probe)
    result = binding.run_affine_fresh_probe(sources(), state())
    assert result["status"] == "failed" and result["disposable_state"] is None
    assert result["source_fork_calls"] == 0


def test_should_keep_validated_collection_prefix_when_prior_source_report_changes(monkeypatch):
    captured = scripted_collector(monkeypatch)

    def probe(*a, check_boundary, **k):
        captured["reports"][0]["closed"] = False
        check_boundary()
        pytest.fail("changed preceding source accepted")

    monkeypatch.setattr(binding, "run_fresh_task_probe", probe)
    result = binding.run_affine_fresh_probe(sources(), state())
    assert result["status"] == "failed" and "collection evidence changed" in result["reason"]
    assert len(result["known_collection_prefix"]) == 4
    assert all(p["closed"] for p in result["known_collection_prefix"])
    assert result["source_fork_calls"] == 0


def test_should_capture_paid_collection_prefix_if_later_source_construction_fails(monkeypatch):
    def collect(guard, reports, **k):
        reports.append(dict(status="complete", closed=True, role="probe_train"))
        guard()
        raise RuntimeError("deliberate next source construction failure")

    monkeypatch.setattr(binding.reserved, "_collect_probe_sources", collect)
    result = binding.run_affine_fresh_probe(sources(), state())
    assert result["status"] == "failed" and len(result["known_collection_prefix"]) == 1
    assert result["budget"]["updates_completed"] == result["source_fork_calls"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["--context-linear-probe", "--describe"],
        ["--context-linear-probe", "--fit-updates", "1"],
        ["--context-linear-probe", "--context-linear-new-c"],
        ["--context-linear-probe", "--source", "train"],
    ],
)
def test_should_describe_without_data_io_and_refuse_invalid_probe_options(
    monkeypatch, capsys, args
):
    monkeypatch.setattr("sys.argv", ["control", *args])
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN opened"))
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: pytest.fail("model opened"))
    if "--describe" in args:
        cli.main()
        d = json.loads(capsys.readouterr().out)
        assert (
            d["probe_updates"] == 2
            and d["score_rows"] == 216
            and d["checkpoint_updates"] == [0, 1, 2]
        )
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 2


def test_should_save_partial_probe_cli_output_exclusively_without_duplicate_generic_read(
    monkeypatch, tmp_path
):
    path = tmp_path / "actual.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "control",
            "--context-linear-probe",
            "--source",
            "train",
            "--retained-source",
            "saved",
            "--output",
            str(path),
        ],
    )
    monkeypatch.setattr(
        cli, "read_training_sources", lambda *a: pytest.fail("generic decode reached")
    )

    def run(train, saved):
        assert str(train) == "train" and str(saved) == "saved"
        return dict(status="failed", probe=dict(completed_updates=1))

    monkeypatch.setattr(cli, "run_saved_context_linear_probe", run)
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert (
        caught.value.code == 2
        and json.loads(path.read_text(encoding="utf8"))["probe"]["completed_updates"] == 1
    )
    with pytest.raises(FileExistsError):
        cli.main()


def test_should_bind_real_after_c_and_readonly_train_sources_without_collection(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    model = root / "artifacts/runs/catalogue-capacity-new-c-20261024/actual.json"
    train = root / "artifacts/runs/joint-catalogue-study-20261018/bp.txt"
    if not model.is_file() or not train.is_file():
        pytest.skip("accepted local source unavailable")

    def run(episodes, saved, *, check_boundary):
        assert (
            len(episodes) == 6 and saved.completed_updates == 6 and saved.training_examples == 108
        )
        check_boundary()
        return dict(status="scripted_source_binding")

    monkeypatch.setattr(binding, "run_affine_fresh_probe", run)
    result = cli.run_saved_context_linear_probe(train, model)
    assert result["source_bindings"]["after_c_bytes"] == 377836


def test_should_refuse_replaced_after_c_bytes_at_probe_boundary(monkeypatch):
    raw = [b"{}"]
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: raw[0])
    monkeypatch.setattr(cli, "_after_c_affine_state", lambda *a: state())
    monkeypatch.setattr(cli, "_qualified_capacity_train_bytes", lambda *a: b"train")
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: sources())

    def run(*a, check_boundary):
        raw[0] = b"changed"
        check_boundary()
        pytest.fail("replacement admitted")

    monkeypatch.setattr(binding, "run_affine_fresh_probe", run)
    with pytest.raises(ValueError, match="after-C evidence changed"):
        cli.run_saved_context_linear_probe("train", "saved")
