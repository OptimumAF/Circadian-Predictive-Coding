"""The original timed comparator with fake native models and synthetic clocks.

The stage composition, six original budgets, real timed runner and complete-seed
summarizer execute here. Fake calls establish no empirical timing or benefit.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.adapters.coupled_inference_comparison as retained
import src.app.toy_execution_budget as budget_module
from src.app.coupled_inference_summary import CASE_ORDER
from src.app.timed_coupled_inference import run_timed_inference_case
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_inference import CoupledInferenceDiagnostics, CoupledInferencePolicy
from src.shared.process_memory import ProcessRssSegment
from test_timed_coupled_inference import fake_model


STAGE = (
    Path(__file__).resolve().parents[1]
    / "artifacts/runs/r15-engineering-preset-comparison-20261017/comparison_case.py"
)


def forbidden(*args, **kwargs):
    pytest.fail("native work or unselected operation reached")


@pytest.fixture
def comparison():
    specification = importlib.util.spec_from_file_location("engineering_preset_case", STAGE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FakeSampler:
    start_bytes = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


@pytest.fixture
def fake_ports(comparison, monkeypatch):
    model_type, _, clock, trace = fake_model(monkeypatch)
    sessions, cases, constructors, timeline = [], [], [], []
    original_session = ToyBudgetSession.__init__
    original_constructor = model_type.__init__
    original_train = model_type.train_epoch
    original_batches = comparison.build_batches

    def initialize(self, *args, **kwargs):
        original_session(self, *args, **kwargs)
        sessions.append(self)
        timeline.append("budget")

    def construct(self, *args, **kwargs):
        constructors.append((args, kwargs))
        timeline.append("model")
        original_constructor(self, *args, **kwargs)

    def batches():
        timeline.append("data")
        return original_batches()

    def train(self, *args, **kwargs):
        result = original_train(self, *args, **kwargs)
        if kwargs.get("inference_policy") is not None:
            # A valid synthetic port for the minimum-two engineering preset.
            result = replace(
                result,
                inference=CoupledInferenceDiagnostics(
                    "gradient_tolerance", 2, 2, 3, 3, 2, 10.0, 9.0, 0.0, False
                ),
            )
        return result

    def case(initial, training, heldout, **options):
        # The case timer is synthetic; global original budget clocks remain at
        # their shared setup origin. No native duration is being measured.
        clock.value = 0.0
        cases.append((initial, deepcopy(training), deepcopy(heldout), dict(options)))
        return run_timed_inference_case(initial, training, heldout, **dict(options, clock=clock))

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    monkeypatch.setattr(model_type, "__init__", construct)
    monkeypatch.setattr(model_type, "train_epoch", train)
    monkeypatch.setattr(comparison, "CoupledPredictiveCodingNetwork", model_type)
    monkeypatch.setattr(comparison, "run_timed_inference_case", case)
    monkeypatch.setattr(comparison, "build_batches", batches)
    monkeypatch.setattr(comparison, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(comparison, "perf_counter", lambda: 0.0)
    return SimpleNamespace(
        model_type=model_type,
        trace=trace,
        sessions=sessions,
        cases=cases,
        constructors=constructors,
        timeline=timeline,
    )


def test_should_describe_new_preset_without_changing_original_protocols_or_running_models(
    comparison, monkeypatch
):
    old = deepcopy(retained.describe_protocol())
    granularity = deepcopy(retained.describe_protocol(granularity_aware=True))
    for name in (
        "build_batches",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
        "run_timed_inference_case",
    ):
        monkeypatch.setattr(comparison, name, forbidden)
    description = comparison.describe_protocol()
    assert description["seeds"] == [23, 41, 59]
    assert description["case_order"] == [list(key) for key in CASE_ORDER]
    assert description["inference_steps"] == 6
    assert description["inference_learning_rate"] == 0.1
    assert description["learning_rate"] == 0.03
    assert description["training_seconds"] == 0.01
    assert description["maximum_window_seconds"] == 1.1 * 0.01
    assert description["max_updates_per_case"] == 80
    assert description["adaptive_policy"] == dict(
        min_steps=2, gradient_tolerance=1.0, relative_energy_tolerance=None
    )
    assert description["resources"]["wall_seconds"] == 30
    assert description["resources"]["training_updates"] == 480
    assert retained.describe_protocol() == old
    assert retained.describe_protocol(granularity_aware=True) == granularity


def test_should_run_all_original_cases_once_with_only_the_declared_adaptive_preset(
    comparison, fake_ports
):
    before = deepcopy(retained.describe_protocol())
    report = comparison.run_comparison()
    assert report["status"] == "complete", report["global_errors"]
    assert [(row["seed"], row["method"]) for row in report["runs"]] == list(CASE_ORDER)
    assert len(fake_ports.cases) == 6
    for row, (source, training, heldout, controls) in zip(
        report["runs"], fake_ports.cases, strict=True
    ):
        assert row["status"] == "complete", row["reason"]
        assert row["completed_updates"] == controls["budget"].updates_completed == 2
        assert row["source_learner_unchanged"] is True
        assert source.saved.parameters.output_bias[0, 0] == 0.0
        assert training.role == "train" and training.features.shape == (16, 2)
        assert heldout.role == "final_test" and heldout.features.shape == (128, 2)
        expected_training, expected_heldout = retained.build_batches()
        np.testing.assert_array_equal(training.features, expected_training.features)
        np.testing.assert_array_equal(training.targets, expected_training.targets)
        np.testing.assert_array_equal(heldout.features, expected_heldout.features)
        np.testing.assert_array_equal(heldout.targets, expected_heldout.targets)
        assert controls["training_seconds"] == 0.01 and controls["max_updates"] == 80
        assert controls["learning_rate"] == 0.03
        assert controls["inference_steps"] == 6 and controls["inference_learning_rate"] == 0.1
        assert controls.get("max_iteration_seconds") is None
        if row["method"] == "fixed":
            assert controls["policy"] is None
        else:
            assert type(controls["policy"]) is CoupledInferencePolicy
            assert asdict(controls["policy"]) == dict(
                min_steps=2, gradient_tolerance=1.0, relative_energy_tolerance=None
            )
    assert [event[0] for event in fake_ports.trace] == ["train", "train", "predict", "predict"] * 6
    assert len({id(case[0]) for case in fake_ports.cases}) == 3
    constructed_seeds = [
        kwargs.get("seed", args[2] if len(args) > 2 else 0)
        for args, kwargs in fake_ports.constructors
    ]
    assert [seed for seed in constructed_seeds if seed != 0] == [23, 41, 59]
    assert set(report["initial_parameters"]) == {"23", "41", "59"}
    for seed in (23, 41, 59):
        paired_sources = [
            case[0]
            for row, case in zip(report["runs"], fake_ports.cases, strict=True)
            if row["seed"] == seed
        ]
        assert paired_sources[0] is paired_sources[1]
    assert retained.describe_protocol() == before


def test_should_refuse_a_changed_source_instead_of_accepting_the_case(
    comparison, fake_ports, monkeypatch
):
    original = comparison.run_timed_inference_case

    def changed(initial, *args, **kwargs):
        result = original(initial, *args, **kwargs)
        initial.saved.parameters.output_bias.fill(999)
        return result

    monkeypatch.setattr(comparison, "run_timed_inference_case", changed)
    report = comparison.run_comparison()
    assert report["status"] == "incomplete"
    assert report["runs"][0]["status"] == "failed"
    assert report["runs"][0]["source_learner_unchanged"] is False
    assert len(fake_ports.cases) == 1
    assert all(row["status"] == "not_run" for row in report["runs"][1:])
    assert all(session.updates_completed == 0 for session in fake_ports.sessions[1:])
    assert all(item["summary"]["mean"] is None for item in report["summaries"])


def test_should_stop_admission_when_original_source_cannot_be_verified(
    comparison, fake_ports, monkeypatch
):
    original = comparison.run_timed_inference_case

    def broken_snapshot():
        raise RuntimeError("original source snapshot unavailable")

    def changed(initial, *args, **kwargs):
        result = original(initial, *args, **kwargs)
        initial.snapshot_state = broken_snapshot
        return result

    monkeypatch.setattr(comparison, "run_timed_inference_case", changed)
    report = comparison.run_comparison()
    assert report["status"] == "incomplete" and report["global_errors"]
    first = report["runs"][0]
    assert first["status"] == "failed" and "snapshot unavailable" in first["reason"]
    assert first["source_learner_unchanged"] is None
    assert first["result"]["initial_scores"] is first["result"]["final_scores"] is None
    assert len(fake_ports.cases) == 1
    assert all(row["status"] == "not_run" for row in report["runs"][1:])
    assert all(session.updates_completed == 0 for session in fake_ports.sessions[1:])


def test_should_prepare_six_original_shared_deadline_budgets_before_data_or_models(
    comparison, fake_ports
):
    report = comparison.run_comparison()
    assert report["status"] == "complete", report["global_errors"]
    sessions = fake_ports.sessions
    assert len(sessions) == len({id(session) for session in sessions}) == 6
    assert all(s.budget.max_training_updates == 80 and s.updates_completed == 2 for s in sessions)
    assert all(s.budget.max_wall_seconds == 30 and s.started_at == 0 for s in sessions)
    assert all(s.budget.max_process_rss_bytes == 256 * 1024**2 for s in sessions)
    assert all(s.clock is sessions[0].clock for s in sessions)
    assert all(s.process_rss_sampler is sessions[0].process_rss_sampler for s in sessions)
    assert fake_ports.timeline[:6] == ["budget"] * 6
    assert fake_ports.timeline[6] in {"model", "data"}


def test_should_preserve_failed_native_attempt_and_null_its_entire_seed_vector(
    comparison, fake_ports
):
    fake_ports.model_type.fail_train = 2
    report = comparison.run_comparison()
    assert report["status"] == "incomplete" and len(report["runs"]) == 6
    row = report["runs"][0]
    assert (row["seed"], row["method"]) == (23, "fixed")
    assert row["status"] == "failed" and "deliberate native failure" in row["reason"]
    result = row["result"]
    assert row["completed_updates"] == fake_ports.sessions[0].updates_completed == 1
    assert result["attempted_updates"] == 2 and result["committed_updates"] == 1
    assert len(result["records"]) == 1
    assert result["initial_scores"] is result["final_scores"] is None
    assert result["work"]["unknown_failed_call_work"] is True
    assert all(case[0].saved.parameters.output_bias[0, 0] == 0 for case in fake_ports.cases)
    fixed = next(
        item["summary"]
        for item in report["summaries"]
        if item["metric"] == "final_bce" and item["comparison"] == "fixed"
    )
    assert fixed["observed_seed_count"] == 2
    assert fixed["mean"] is fixed["marginal_interval"] is None
    assert fixed["observations"][0]["seed"] == 23 and fixed["observations"][0]["value"] is None
    paired = next(
        item["summary"]
        for item in report["summaries"]
        if item["metric"] == "final_bce" and item["comparison"] == "adaptive_minus_fixed"
    )
    assert paired["mean"] is paired["marginal_interval"] is None


def test_should_keep_the_original_eighty_update_ceiling_without_scoring_or_extension(
    comparison, fake_ports
):
    fake_ports.model_type.duration = 0.0
    report = comparison.run_comparison()
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 6
    for row, session in zip(report["runs"], fake_ports.sessions, strict=True):
        assert row["status"] == "failed" and "max_updates" in row["reason"]
        assert row["completed_updates"] == session.updates_completed == 80
        assert row["result"]["committed_updates"] == row["result"]["attempted_updates"] == 80
        assert row["result"]["initial_scores"] is row["result"]["final_scores"] is None
    assert not any(event[0] == "predict" for event in fake_ports.trace)
    assert fake_ports.model_type.total_train_calls == 480


@pytest.mark.parametrize("phase", ["constructor", "enter"])
def test_should_keep_all_unreached_rows_when_resource_setup_fails(comparison, monkeypatch, phase):
    class FailedSampler:
        def __init__(self):
            if phase == "constructor":
                raise RuntimeError("declared setup failure")

        def __enter__(self):
            raise RuntimeError("declared setup failure")

        def __exit__(self, *args):
            return False

        def snapshot(self):
            raise RuntimeError("no memory observation")

    monkeypatch.setattr(comparison, "ProcessRssSampler", FailedSampler)
    monkeypatch.setattr(comparison, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(comparison, "run_timed_inference_case", forbidden)
    report = comparison.run_comparison()
    assert report["status"] == "incomplete" and report["global_errors"]
    assert [(row["seed"], row["method"]) for row in report["runs"]] == list(CASE_ORDER)
    assert all(
        row["status"] == "not_run" and row["completed_updates"] == 0 for row in report["runs"]
    )
    assert all(item["summary"]["mean"] is None for item in report["summaries"])
