"""Fixed paired recipe sequencing and TRAIN-only live route use bounded fakes."""

import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

import scripts.run_minigrid_paired_bc_smoke as cli
from src.adapters import minigrid_paired_bc_pilot as pilot
from src.adapters import minigrid_policy_evaluation as evaluation
from src.adapters import minigrid_learning_pilot as shared
import src.shared.torch_runtime as runtime
import src.app.toy_execution_budget as budgets
from src.core.minigrid_key_door import predefined_layouts
from test_minigrid_learning_pilot import source_batch
from test_paired_episode_stream import case as case, packets
from test_torch_feature_rehearsal_pilot import Sampler, forbidden


def record(point, *, final=False):
    return dict(
        method="recurrent_bc",
        updates=point,
        role="train",
        status="complete",
        reason=None,
        scores=[
            dict(context="A", cue=cue, ce=1.0 - point / 256, exact_sequence_accuracy=float(final))
            for cue in ("blue", "red")
        ],
        episodes=[
            dict(context="A", cue=cue, status="complete", success=final, **{"return": float(final)})
            for cue in ("blue", "red")
        ],
        work=dict(
            raw_score_forwards_attempted=2,
            raw_score_forwards_completed=2,
            raw_score_rows_attempted=18,
            raw_score_rows_completed=18,
            live_prediction_calls_attempted=2,
            live_prediction_calls_returned=2,
            live_sessions_started=2,
            environment_resets_completed=2,
            environment_steps_returned=2,
        ),
    )


@pytest.fixture
def route(monkeypatch, case):
    work = case.work
    work.update(pilots=[], live=[], parity=[], clock=[10.0])
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "monotonic", lambda: work["clock"][0])
    monkeypatch.setattr(shared, "monotonic", lambda: work["clock"][0])
    sources = tuple(
        source_batch(context, cue) for context in ("A", "B", "C_train") for cue in ("blue", "red")
    )
    reference = dict(
        provenance={"accepted_reference": True},
        sources=sources,
        evaluations=[record(point) for point in (0, 32, 64, 96, 128)],
    )
    monkeypatch.setattr(pilot, "_load_reference", lambda: {})
    monkeypatch.setattr(pilot, "_admit_reference", lambda value: reference)
    builtin_dict = dict

    def capture(*args, **kwargs):
        value = builtin_dict(*args, **kwargs)
        if "protocol" in kwargs:
            work["pilots"].append(value)
            monkeypatch.delattr(pilot, "dict")
        return value

    monkeypatch.setattr(pilot, "dict", capture, raising=False)

    def evaluate(source, state, **options):
        point = options["updates"]
        options["check_boundary"]()
        assert options["method"] == pilot.METHOD and options["sources"] == sources[:2]
        assert options["layouts"] == predefined_layouts()[:1] and options["role"] == "train"
        assert len(work["trains"]) == 64
        work["live"].append(point)
        result = record(point, final=point == 64)
        result["method"] = pilot.METHOD
        callback = work.get("after_live")
        if callback:
            callback(result)
        options["check_boundary"]()
        return result

    def parity(before, after):
        work["parity"].append((before["updates"], after["updates"]))
        if work.get("bad_parity"):
            raise ValueError("original initial probability law differs")

    monkeypatch.setattr(pilot, "evaluate_checkpoint", evaluate)
    monkeypatch.setattr(pilot, "_check_initial_condition", parity)
    return SimpleNamespace(work=work, sources=sources)


def test_should_run_full_declared_stream_then_all_readouts_without_final_or_probe_work(route):
    report = pilot.run_pilot()
    assert report["status"] == "complete", report["reason"]
    assert len(route.work["trains"]) == 64 and route.work["live"] == list(pilot.POINTS)
    assert route.work["parity"] == [(0, 0)] and report["initial_parity"] is True
    assert report["training"]["native_training_episodes"] == 128
    assert report["training"]["charged_updates"] == 64
    assert report["training"]["native_training_examples"] == 1152
    assert report["training"]["source_unchanged"] is report["training"]["actor_unchanged"] is True
    assert len(report["raw_readouts"]) == len(report["evaluations"]) == 5
    assert report["work"]["raw_scores"]["score_rows_completed"] == 270
    assert report["work"]["live_evaluation"]["raw_score_rows_completed"] == 90
    assert report["work"]["observed_forward_rows"] == 1522
    assert report["work"]["observed_backward_rows"] == 1152
    assert report["outcomes"]["paired"]["joint_acquisition"] is True  # fake outcome only
    assert report["readout_budget"]["updates_completed"] == 0


@pytest.mark.parametrize("fault", ["parity", "raw", "training", "deadline"])
def test_should_stop_later_readouts_and_preserve_prior_paid_evidence(route, fault):
    if fault == "parity":
        route.work["bad_parity"] = True
    else:

        def corrupt(result):
            if len(route.work["live"]) == 1:
                report = route.work["pilots"][0]
                if fault == "raw":
                    report["raw_readouts"][0]["scores"][0]["ce"] = 999.0
                elif fault == "training":
                    report["training"]["charged_updates"] = 999
                else:
                    route.work["clock"][0] = 80.0

        route.work["after_live"] = corrupt
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert report["training"]["charged_updates"] == 64
    assert report["work"]["charged_updates"] == 64
    assert len(report["raw_readouts"]) == 1
    assert report["raw_readouts"][0]["scores"][0]["ce"] != 999.0
    assert route.work["live"] == [0]


def test_should_keep_transport_binding_and_preserve_legacy_episode_requirement(case):
    original = packets(case.episodes)
    row = shared._row("recurrent_bc")
    with pytest.raises(ValueError, match="legacy.*episode"):
        shared._train_arm(
            row,
            original * 8,
            Sampler(),
            100.0,
            lambda: None,
            learner_factory=forbidden,
            checkpoint_updates=(0, 32),
        )
    good = SimpleNamespace(completed_updates=2, training_examples=18)
    shared._counter(good, 2)
    good.training_examples = 36
    with pytest.raises(ValueError):
        shared._counter(good, 2)


@pytest.mark.parametrize("field", ["raw_readouts", "evaluations"])
def test_should_refuse_final_resource_callback_changes_and_keep_paid_records(
    route, monkeypatch, field
):
    class LateSampler(Sampler):
        exited = False

        def __exit__(self, *args):
            self.exited = True
            return False

        def sample(self):
            if self.exited:
                route.work["pilots"][0][field][0]["scores"][0]["ce"] = 999.0
            return 100

    monkeypatch.setattr(pilot, "ProcessRssSampler", LateSampler)
    monkeypatch.setattr(budgets, "ProcessRssSampler", LateSampler)
    report = pilot.run_pilot()
    assert report["status"] == "incomplete" and report["outcomes"] is None
    assert len(report[field]) == 5, report["reason"]
    assert report[field][0]["scores"][0]["ce"] != 999.0
    assert report["work"]["charged_updates"] == 64
    assert len(report["raw_readouts"]) == len(report["evaluations"]) == 5


@pytest.mark.parametrize("fault", ["final", "layouts", "point", "history_flag", "rows"])
def test_should_refuse_unsupported_paired_live_routes_before_native_state_access(
    monkeypatch, fault
):
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    options: dict[str, Any] = dict(
        method=pilot.METHOD,
        updates=0,
        layouts=predefined_layouts()[:1],
        role="train",
        sources=tuple(source_batch("A", cue) for cue in ("blue", "red")),
        budget=SimpleNamespace(),
        check_boundary=forbidden,
    )
    if fault == "final":
        options.update(role="final_test", layouts=predefined_layouts()[3:], sources=())
    elif fault == "layouts":
        options["layouts"] = predefined_layouts()[:3]
    elif fault == "point":
        options["updates"] = 4
    elif fault == "history_flag":
        options["public_history"] = 1
    else:
        options["native_training_examples"] = 0
    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(None, None, **options)


def test_should_describe_actual_cli_without_model_or_environment_work(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    monkeypatch.setattr(pilot, "run_pilot", forbidden)
    assert cli.main(["--describe"]) == 0
    protocol = json.loads(capsys.readouterr().out)
    assert protocol == pilot.describe_protocol()
    assert protocol["checkpoints"] == [0, 16, 32, 48, 64]
    assert protocol["planned_work"]["training_rows"] == 1152


def test_should_refuse_foreign_paired_live_snapshot_before_deepcopy(case):
    class Foreign:
        def __deepcopy__(self, memo):
            raise AssertionError("foreign state must be refused before copy")

    with pytest.raises(ValueError):
        evaluation.evaluate_checkpoint(
            None,
            Foreign(),
            method=pilot.METHOD,
            updates=0,
            layouts=predefined_layouts()[:1],
            role="train",
            sources=tuple(source_batch("A", cue) for cue in ("blue", "red")),
            budget=cast(Any, SimpleNamespace(before_final=lambda: None)),
            check_boundary=lambda: None,
        )


@pytest.mark.parametrize("fault", ["existing", "missing_parent", "describe_output"])
def test_should_preflight_cli_output_before_native_setup(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(runtime, "require_torch", forbidden)
    path = tmp_path / "result.json"
    if fault == "existing":
        path.write_text("original", encoding="utf8")
    elif fault == "missing_parent":
        path = tmp_path / "absent" / "result.json"
    args = ["--output", str(path)] + (["--describe"] if fault == "describe_output" else [])
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("status,code", [("complete", 0), ("incomplete", 1)])
def test_should_publish_exact_cli_result_and_honest_exit(monkeypatch, tmp_path, status, code):
    torch = SimpleNamespace(set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None)
    monkeypatch.setattr(runtime, "require_torch", lambda: torch)
    result = dict(status=status, reason=None if code == 0 else "retained failure")
    monkeypatch.setattr(pilot, "run_pilot", lambda: result)
    path = tmp_path / "result.json"
    assert cli.main(["--output", str(path)]) == code
    assert json.loads(path.read_text(encoding="utf8")) == result
