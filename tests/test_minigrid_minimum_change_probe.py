"""Actual isolated adapter/runner with SCRIPTED fits/predictions/collection."""

from dataclasses import replace
import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

from scripts import run_minigrid_linear_fit_control as cli
from src.adapters import minigrid_affine_probe as binding
from src.adapters import minigrid_context_linear_learner as port
from src.core.categorical_linear_fit import CategoricalLinearFit
from src.core.learner_ports import TrainingUpdateFailed
import test_minigrid_affine_probe as legacy

CHARGE = lambda values: None
RETURN = lambda values: None


@pytest.fixture
def scripted(monkeypatch):
    holder: dict[str, Any] = dict(fits=[], operations=[], effect=None)

    def fit(kind, x, y):
        values = dict(script_fits=1, script_fit_rows=len(x))
        CHARGE(values)
        holder["fits"].append(kind)
        model = CategoricalLinearFit(296, 7, len(x), ((0.0,) * 7,) * 297, 0, (0.0,) * len(x), 0.0)
        if holder["effect"] is not None:
            holder["effect"]()
        RETURN(values)
        return model

    monkeypatch.setattr(port, "fit_categorical_linear", lambda x, y, k: fit("cold", x, y))
    monkeypatch.setattr(port, "update_categorical_linear", lambda prior, x, y: fit("warm", x, y))

    def predict(model, x):
        values = dict(script_predict=1, script_predict_rows=len(x))
        CHARGE(values)
        value = np.full((len(x), 7), 1 / 7, dtype=np.float64)
        RETURN(values)
        return value

    monkeypatch.setattr(port, "predict_categorical_linear", predict)

    def operation(original, name):
        def run(self, x, y):
            holder["operations"].append(name)
            return original(self, x, y)

        return run

    for name in ("train_batch", "train_batch_minimum_change"):
        monkeypatch.setattr(
            port.ContextLinearLearner,
            name,
            operation(getattr(port.ContextLinearLearner, name), name),
        )
    legacy.scripted_collector(monkeypatch)
    raw_collect = binding.reserved._collect_probe_sources

    def collect(*args, **kwargs):
        values = dict(script_collections=1, script_collection_records=4)
        CHARGE(values)
        result = raw_collect(*args, **kwargs)
        RETURN(values)
        return result

    monkeypatch.setattr(binding.reserved, "_collect_probe_sources", collect)
    actual_probe = binding.run_fresh_task_probe

    def probe(source, saved, *args, **kwargs):
        holder.update(source=source, budget=kwargs["budget"], saved=saved, training=args[0])
        guard = kwargs["check_boundary"]
        holder["report"] = dict(
            zip(guard.__code__.co_freevars, (c.cell_contents for c in guard.__closure__))
        )["report"]
        hook = holder.get("before_probe")
        if hook:
            hook(source, guard)
        return actual_probe(source, saved, *args, **kwargs)

    monkeypatch.setattr(binding, "run_fresh_task_probe", probe)
    return holder


def test_describes_selected_objective_without_changing_default():
    original = binding.describe_protocol()
    selected = binding.describe_protocol(probe_update="minimum_change")
    assert "probe_update" not in original
    assert selected["probe_update"] == "minimum_change"
    assert "squared error" in selected.pop("update_objective")
    selected.pop("probe_update")
    assert selected == original


class StringLike(str):
    pass


@pytest.mark.parametrize("value", [True, False, 0, 1, "cold", "", [], StringLike("minimum_change")])
def test_refuses_invalid_intent_before_source_or_collection(monkeypatch, value):
    monkeypatch.setattr(
        binding, "copy_context_linear_training_snapshot", lambda *a: pytest.fail("state read")
    )
    with pytest.raises(ValueError, match="probe_update"):
        binding.run_affine_fresh_probe((), None, probe_update=value)


@pytest.mark.parametrize(
    "options", [dict(retained_a4=True), dict(return_old=True), dict(native_games=True)]
)
def test_refuses_unsupported_selected_scope_before_source(monkeypatch, options):
    monkeypatch.setattr(
        binding, "copy_context_linear_training_snapshot", lambda *a: pytest.fail("state read")
    )
    with pytest.raises(ValueError, match="after-C scope"):
        binding.run_affine_fresh_probe((), None, probe_update="minimum_change", **options)


@pytest.mark.parametrize("selected", [False, True])
def test_runs_fixed_probe_on_one_owned_child_preserving_source_and_costs(scripted, selected):
    episodes, saved = legacy.sources(), legacy.state()
    before = pickle.dumps((episodes, saved, np.random.get_state()))
    result = binding.run_affine_fresh_probe(
        episodes, saved, **(dict(probe_update="minimum_change") if selected else {})
    )
    assert result["status"] == "complete", result["reason"]
    assert scripted["fits"] == (["warm"] * 2 if selected else ["cold"] * 2)
    assert scripted["operations"] == (
        ["train_batch_minimum_change"] * 2 if selected else ["train_batch"] * 2
    )
    assert result["source_native_state_unchanged"] is True and result["source_fork_calls"] == 1
    assert (
        result["budget"]["updates_completed"] == 2
        and result["budget"]["replay_examples_completed"] == 0
    )
    assert result["budget"]["limits"] == result["initial_probe_limits"]
    assert (
        result["disposable_state"]["completed_updates"],
        result["disposable_state"]["training_examples"],
    ) == (8, 126)
    assert result["probe"]["completed_training_examples"] == 18
    assert result["probe"]["score_forward_calls_completed"] == 6
    assert result["probe"]["training_readouts"]["completed_rows"] == 162
    assert [p["updates"] for p in result["old_task_readouts"]] == [0, 1, 2]
    assert pickle.dumps((episodes, saved, np.random.get_state())) == before
    if selected:
        assert result["probe_update"] == result["protocol"]["probe_update"] == "minimum_change"
    else:
        assert "probe_update" not in result
    with pytest.raises(ValueError, match="exactly one"):
        scripted["source"].fork()


def test_owned_child_forks_preserve_selection_but_restored_plain_owner_stays_cold(scripted):
    def arrange(source, guard):
        actual = source.fork

        def fork():
            child = actual()
            before = child.snapshot_state()
            grandchild = child.fork()
            assert type(grandchild) is type(child) and grandchild is not child
            packet = scripted["training"][0]
            grandchild.train_batch(packet.features, packet.targets)
            assert child.snapshot_state() == before
            plain = port.ContextLinearLearner()
            plain.restore_state(grandchild.snapshot_state())
            plain.train_batch(packet.features, packet.targets)
            return child

        source.fork = fork

    scripted["before_probe"] = arrange
    result = binding.run_affine_fresh_probe(
        legacy.sources(), legacy.state(), probe_update="minimum_change"
    )
    assert result["status"] == "complete", result["reason"]
    assert scripted["fits"] == ["warm", "cold", "warm", "warm"]


@pytest.mark.parametrize(
    "fault", ["root", "protocol", "replace_protocol", "replace_child", "factory"]
)
def test_intent_changes_after_known_fit_stop_and_retain_paid_native_work(scripted, fault):
    def effect():
        report, source = scripted["report"], scripted["source"]
        if fault == "root":
            report["probe_update"] = False
        elif fault == "protocol":
            report["protocol"]["probe_update"] = "cold"
        elif fault == "replace_protocol":
            report["protocol"] = dict(report["protocol"])
        elif fault == "replace_child":
            source.child = source.child.fork()
        else:
            source._child_factory = lambda: None

    scripted["effect"] = effect
    result = binding.run_affine_fresh_probe(
        legacy.sources(), legacy.state(), probe_update="minimum_change"
    )
    assert result["status"] == "failed"
    assert len(scripted["fits"]) == 1
    assert result["source_native_state_unchanged"] is True
    assert result["disposable_state"]["completed_updates"] == 7
    assert result["probe"]["failed_native_work"]["work"]["fit_calls_completed"] == 1
    assert result["probe"]["failed_native_work"]["work"]["fit_rows_completed"] == 9
    assert result["budget"]["updates_completed"] == 0  # Outer return was refused.
    packet = scripted["training"][0]
    with pytest.raises(ValueError):
        scripted["source"].child.train_batch(packet.features, packet.targets)
    assert len(scripted["fits"]) == 1


@pytest.mark.parametrize("fault", ["budget", "clock", "source", "caller", "limits"])
def test_refuses_foreign_authority_or_source_changes_before_fit(scripted, fault):
    caller = legacy.state()

    def change(source, guard):
        budget = scripted["budget"]
        if fault == "budget":
            budget.updates_completed = 1
        elif fault == "clock":
            budget.last_clock = budget.started_at - 1
        elif fault == "source":
            source.restore_state(
                replace(scripted["saved"], completed_updates=7, training_examples=117)
            )
        elif fault == "caller":
            object.__setattr__(caller, "completed_updates", 7)
        else:
            object.__setattr__(budget.budget, "max_training_updates", 3)
        guard()

    scripted["before_probe"] = change
    result = binding.run_affine_fresh_probe(legacy.sources(), caller, probe_update="minimum_change")
    assert result["status"] == "failed" and not scripted["fits"]
    assert result["source_fork_calls"] == 0
    if fault == "budget":
        assert result["budget"]["updates_completed"] == 1
    if fault == "limits":
        assert result["budget"]["limits"]["max_training_updates"] == 3


def test_raw_role_contradiction_is_refused_before_any_selected_fit(scripted, monkeypatch):
    collect = binding.reserved._collect_probe_sources

    def bad(*args, **kwargs):
        training, evaluation = collect(*args, **kwargs)
        return (replace(training[0], role="train"), *training[1:]), evaluation

    monkeypatch.setattr(binding.reserved, "_collect_probe_sources", bad)
    result = binding.run_affine_fresh_probe(
        legacy.sources(), legacy.state(), probe_update="minimum_change"
    )
    assert result["status"] == "failed" and not scripted["fits"]
    assert result["source_fork_calls"] == 0


def test_lost_second_scripted_fit_keeps_prefix_and_permanent_failure(scripted):
    def fail():
        if len(scripted["fits"]) == 2:
            raise RuntimeError("lost second scripted fit return")

    scripted["effect"] = fail
    result = binding.run_affine_fresh_probe(
        legacy.sources(), legacy.state(), probe_update="minimum_change"
    )
    assert result["status"] == "failed" and result["budget"]["updates_completed"] == 1
    assert result["probe"]["completed_updates"] == 1
    assert result["disposable_state"]["failed_work"] == (1, 9, 0, 0, True)
    assert [p["updates"] for p in result["old_task_readouts"]] == [0, 1]
    packet = scripted["training"][0]
    with pytest.raises(TrainingUpdateFailed):
        scripted["source"].child.train_batch(packet.features, packet.targets)
    assert len(scripted["fits"]) == 2


@pytest.mark.parametrize("fault", ["collection", "probe_data", "probe"])
def test_refuses_equal_replacement_of_preceding_paid_evidence(scripted, fault):
    if fault == "probe":

        def external():
            report = scripted.get("report")
            if report and report["probe"] is not None:
                report["probe"] = dict(report["probe"])

        options = dict(check_boundary=external)
    else:

        def replace_evidence(source, guard):
            report = scripted["report"]
            if fault == "collection":
                report["collection"][0] = dict(report["collection"][0])
            else:
                report["probe_data"] = list(report["probe_data"])
            guard()

        scripted["before_probe"] = replace_evidence
        options = {}
    result = binding.run_affine_fresh_probe(
        legacy.sources(), legacy.state(), probe_update="minimum_change", **options
    )
    assert result["status"] == "failed" and "replaced" in result["reason"]
    assert result["source_native_state_unchanged"] is True
    assert len(scripted["fits"]) == (2 if fault == "probe" else 0)
    if fault == "probe":
        assert result["probe"]["status"] == "complete"
        assert result["probe"]["completed_updates"] == result["budget"]["updates_completed"] == 2


@pytest.mark.parametrize("selected", [False, True])
def test_cli_describe_before_source_reads(monkeypatch, capsys, selected):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN IO"))
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: pytest.fail("model IO"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["control", "--context-linear-probe", "--describe"]
        + (["--probe-update", "minimum_change"] if selected else []),
    )
    cli.main()
    d = json.loads(capsys.readouterr().out)
    assert d["probe_updates"] == 2
    assert (d.get("probe_update") == "minimum_change") is selected


@pytest.mark.parametrize(
    "flag",
    ["--example-affine-recurring", "--context-linear-stream", "--context-linear-return-game"],
)
def test_cli_refuses_other_profiles_before_io(monkeypatch, flag):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("IO"))
    monkeypatch.setattr(
        sys, "argv", ["control", flag, "--probe-update", "minimum_change", "--describe"]
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


def test_cli_refuses_mixed_a4_profile_before_ignoring_either_update_intent(monkeypatch):
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: pytest.fail("TRAIN IO"))
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: pytest.fail("model IO"))
    for selector, update in (
        ("--context-linear-probe", "--probe-update"),
        ("--context-linear-stream", "--train-update"),
    ):
        for describe in ([], ["--describe"]):
            monkeypatch.setattr(
                sys,
                "argv",
                ["control", "--example-affine-a4", selector, update, "minimum_change", *describe],
            )
            with pytest.raises(SystemExit) as caught:
                cli.main()
            assert caught.value.code == 2


def test_cli_forwards_selection_and_keeps_exclusive_partial_capture(monkeypatch, tmp_path):
    output = tmp_path / "partial.json"

    def run(train, saved, **kwargs):
        assert str(train) == "train" and str(saved) == "saved"
        assert kwargs == dict(probe_update="minimum_change")
        return dict(status="failed", probe=dict(completed_updates=1))

    monkeypatch.setattr(cli, "run_saved_context_linear_probe", run)
    monkeypatch.setattr(cli, "read_training_sources", lambda *a: pytest.fail("generic IO"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "control",
            "--context-linear-probe",
            "--probe-update",
            "minimum_change",
            "--source",
            "train",
            "--retained-source",
            "saved",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert json.loads(output.read_text())["probe"]["completed_updates"] == 1
    with pytest.raises(FileExistsError):
        cli.main()


def test_saved_caller_validates_before_io_and_forwards_original_boundaries(monkeypatch):
    with pytest.raises(ValueError, match="probe_update"):
        cli.run_saved_context_linear_probe("missing", "missing", probe_update=True)
    raw = [b"{}"]
    monkeypatch.setattr(cli, "_qualified_after_c_bytes", lambda *a: raw[0])
    monkeypatch.setattr(cli, "_after_c_affine_state", lambda *a: legacy.state())
    monkeypatch.setattr(cli, "_qualified_capacity_train_bytes", lambda *a: b"train")
    monkeypatch.setattr(cli, "read_training_sources", lambda *a, **k: legacy.sources())

    def run(episodes, saved, *, probe_update, check_boundary):
        assert probe_update == "minimum_change" and saved.completed_updates == 6
        check_boundary()
        raw[0] = b"changed"
        with pytest.raises(ValueError, match="after-C evidence changed"):
            check_boundary()
        return dict(status="scripted_source_binding")

    monkeypatch.setattr(binding, "run_affine_fresh_probe", run)
    result = cli.run_saved_context_linear_probe("train", "saved", probe_update="minimum_change")
    assert result["source_bindings"]["after_c_bytes"] == 2
