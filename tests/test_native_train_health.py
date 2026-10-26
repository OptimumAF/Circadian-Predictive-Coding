"""Fixed A6/first-B diagnosis with scalar paid-work and health records.

No Torch, model, tensor, optimizer or RNG is imported or executed. Recorded
health values are analytic fixtures, not measurements of neural adaptation.
"""

from copy import deepcopy
import json
from math import sqrt
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scripts import run_open_jev_train_health as runner
from src.core.learner_ports import TrainingUpdateFailed
from src.core.native_choice_sequence import sequence_examples
from tests.test_native_optimizer_reset_control import ResetLearner


DEFINITION = "open_jev_train_group_update_health_v1"


def scalar_groups():
    return {
        "head": dict(
            parameter_tensors=2,
            parameter_elements=3,
            gradient_tensors=2,
            missing_gradient_tensors=0,
            gradient_l2_before_clip=6.0,
            parameter_step_l2=3.0,
        ),
        "lora_A": dict(
            parameter_tensors=1,
            parameter_elements=2,
            gradient_tensors=1,
            missing_gradient_tensors=0,
            gradient_l2_before_clip=8.0,
            parameter_step_l2=2.0,
        ),
        "lora_B": dict(
            parameter_tensors=1,
            parameter_elements=2,
            gradient_tensors=1,
            missing_gradient_tensors=0,
            gradient_l2_before_clip=0.0,
            parameter_step_l2=0.0,
        ),
    }


class HealthLearner(ResetLearner):
    """Recording native port with explicit scalar health costs, no numerical model."""

    def __init__(self, *, fail_at=None):
        super().__init__([], fail_at=fail_at)
        self._health_work = dict(
            update_health_attempts=0,
            parameter_copy_elements_attempted=0,
            gradient_reduction_elements_attempted=0,
            step_reduction_elements_attempted=0,
            update_health_reports_completed=0,
        )

    @property
    def update_health_work(self):
        return dict(self._health_work)

    def train_batch(self, features, targets):
        self._health_work["update_health_attempts"] += 1
        for key in (
            "parameter_copy_elements_attempted",
            "gradient_reduction_elements_attempted",
            "step_reduction_elements_attempted",
        ):
            self._health_work[key] += 7
        try:
            diagnostic = super().train_batch(features, targets)
        except TrainingUpdateFailed as error:
            raise TrainingUpdateFailed(
                str(error), error.definition, {**self.work, **self.update_health_work}
            ) from error
        self._health_work["update_health_reports_completed"] += 1
        self.last_update = dict(
            loss=0.5,
            gradient_norm_before_clip=10.0,
            update_health=dict(
                definition=DEFINITION,
                groups=scalar_groups(),
                work=self.update_health_work,
                scope="arrived_TRAIN_update_only",
            ),
        )
        return diagnostic

    def predict(self, request):
        raise AssertionError("TRAIN-only diagnosis reached a prediction")

    def reset_optimizer_state(self):
        raise AssertionError("TRAIN-only diagnosis reset optimizer")


@pytest.fixture(scope="module")
def completed():
    owner = HealthLearner()
    report: dict[str, Any] = {}
    events: list[Any] = []
    runner.run_train_health(owner, report, events.append)
    return owner, report, events


def test_should_deliver_only_original_a_six_then_first_b_once_without_readout_or_reset(completed):
    owner, report, events = completed
    assert [(request.query, target) for request, target in owner.training] == [
        ("Case marker: zorbex", "shipping"),
        ("Intake code zorbex. Route this request.", "shipping"),
    ] * 3 + [("Case marker: neldax", "password")]
    assert events.count("before_B_update") == 1
    assert report["diagnostic_complete"] is True
    assert len(report["update_diagnostics"]) == 7
    assert report["app_committed_updates"] == owner.work["committed_updates"] == 7
    assert report["native_work"] == owner.work
    assert report["optimizer_reset_work"] == {
        "optimizer_reset_attempts": 0,
        "optimizer_resets_completed": 0,
    }
    assert report["update_health_work"] == owner.update_health_work
    assert owner.update_health_work["update_health_reports_completed"] == 7
    assert report["inbox_stopped"] is False


def test_should_preserve_seven_unique_delayed_receipts_and_each_original_health_report(completed):
    _, report, _ = completed
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 7
    assert [row["observed_at"] for row in rows] == list(range(0, 14, 2))
    assert all(row["prelabel_updates"] == 0 for row in rows)
    assert [row["source_example_id"] for row in rows] == ["train-A-0", "train-A-1"] * 3 + [
        "train-B-0"
    ]
    assert rows[-1]["phase"] == "first_B" and rows[-1]["exposure_id"] == "first_B-train-B-0"
    assert all(
        "exposure_kind" not in row and "replay_source_exposure_id" not in row for row in rows
    )
    receipts = [row["receipt"] for row in rows]
    assert [row["update_number"] for row in receipts] == list(range(1, 8))
    assert [row["arrived_at"] for row in receipts] == list(range(1, 14, 2))
    assert all(row["arrived_at"] == row["applied_at"] for row in receipts)
    for number, row in enumerate(rows, 1):
        assert row["training"]["loss"] == 0.5
        assert row["training"]["update_health"]["groups"] == scalar_groups()
        assert row["training"]["update_health"]["work"]["update_health_reports_completed"] == number
    json.dumps(report, allow_nan=False)


def test_should_derive_rms_and_declared_clip_scale_from_all_raw_group_coordinates(completed):
    _, report, _ = completed
    rows = deepcopy(report["delivery"])
    original = deepcopy(rows)
    summary = runner.summarize_update_health(rows)
    assert rows == original
    assert len(summary) == 7
    assert [row["update_number"] for row in summary] == list(range(1, 8))
    assert [row["source_example_id"] for row in summary] == [
        row["source_example_id"] for row in rows
    ]
    for row in summary:
        assert row["loss"] == 0.5 and row["gradient_norm_before_clip"] == 10.0
        assert row["clipping_scale_from_reported_norm"] == pytest.approx(1 / (10.0 + 1e-6))
        groups = row["groups"]
        assert groups["head"]["gradient_rms_before_clip"] == pytest.approx(6 / sqrt(3))
        assert groups["head"]["parameter_step_rms"] == pytest.approx(3 / sqrt(3))
        assert groups["lora_A"]["gradient_rms_before_clip"] == pytest.approx(8 / sqrt(2))
        assert groups["lora_A"]["parameter_step_rms"] == pytest.approx(2 / sqrt(2))
        assert (
            groups["lora_B"]["gradient_rms_before_clip"]
            == groups["lora_B"]["parameter_step_rms"]
            == 0.0
        )
        for name, raw in scalar_groups().items():
            assert all(groups[name][key] == value for key, value in raw.items())
    summary[0]["groups"]["head"]["gradient_l2_before_clip"] = 999.0
    assert rows == original


def test_should_cap_small_or_zero_global_norm_and_keep_missing_gradient_coordinates_in_rms_denominator(
    completed,
):
    rows = deepcopy(completed[1]["delivery"])
    rows[0]["training"]["gradient_norm_before_clip"] = 0.0
    for group in rows[0]["training"]["update_health"]["groups"].values():
        group["gradient_l2_before_clip"] = 0.0
    head = rows[1]["training"]["update_health"]["groups"]["head"]
    head.update(gradient_tensors=1, missing_gradient_tensors=1)
    rows[1]["training"]["gradient_norm_before_clip"] = 0.5
    head["gradient_l2_before_clip"] = 0.3
    rows[1]["training"]["update_health"]["groups"]["lora_A"]["gradient_l2_before_clip"] = 0.4
    result = runner.summarize_update_health(rows)
    assert (
        result[0]["clipping_scale_from_reported_norm"]
        == result[1]["clipping_scale_from_reported_norm"]
        == 1.0
    )
    assert result[1]["groups"]["head"]["gradient_rms_before_clip"] == pytest.approx(0.3 / sqrt(3))
    assert result[1]["groups"]["head"]["missing_gradient_tensors"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "number",
        "number_bool",
        "missing",
        "definition",
        "nan",
        "negative",
        "bool",
        "counts",
    ],
)
def test_should_refuse_incomplete_or_malformed_raw_health_without_inventing_a_summary(
    completed, fault
):
    rows = deepcopy(completed[1]["delivery"])
    first = rows[0]
    if fault == "partial":
        rows.pop()
    elif fault == "number":
        rows[-1]["receipt"]["update_number"] = 6
    elif fault == "number_bool":
        first["receipt"]["update_number"] = True
    elif fault == "missing":
        first["training"].pop("update_health")
    elif fault == "definition":
        first["training"]["update_health"]["definition"] = "foreign_diagnostic"
    elif fault == "nan":
        first["training"]["gradient_norm_before_clip"] = float("nan")
    elif fault == "negative":
        first["training"]["update_health"]["groups"]["head"]["parameter_step_l2"] = -1.0
    elif fault == "bool":
        first["training"]["loss"] = True
    else:
        first["training"]["update_health"]["groups"]["head"]["missing_gradient_tensors"] = 3
    with pytest.raises((ValueError, KeyError) if fault == "missing" else ValueError):
        runner.summarize_update_health(rows)


@pytest.mark.parametrize("fault", ["after_A_binding", "seventh_train", "postcommit_resource"])
def test_should_keep_paid_prefix_and_attempted_health_work_without_retry_or_later_b_update(fault):
    owner = HealthLearner(fail_at=7 if fault == "seventh_train" else None)
    report: dict[str, Any] = {}

    def progress(event):
        if fault == "after_A_binding" and event == "before_B_update":
            raise ValueError("controlled after-A binding refusal")
        if fault == "postcommit_resource" and event == "training_finished":
            raise RuntimeError("controlled paid resource refusal")

    with pytest.raises((ValueError, RuntimeError)):
        runner.run_train_health(owner, report, progress)
    attempts, committed = {
        "after_A_binding": (6, 6),
        "seventh_train": (7, 6),
        "postcommit_resource": (1, 1),
    }[fault]
    assert len(owner.training) == attempts and owner.work["committed_updates"] == committed
    assert report["app_committed_updates"] == committed
    assert (
        report["native_work"] == owner.work
        and report["update_health_work"] == owner.update_health_work
    )
    assert report["update_health_work"]["update_health_attempts"] == attempts
    assert report["update_health_work"]["parameter_copy_elements_attempted"] == 7 * attempts
    assert report["update_health_work"]["update_health_reports_completed"] == committed
    assert sum("receipt" in row for row in report["delivery"]) == committed
    assert report["diagnostic_complete"] is False
    assert "update_diagnostics" not in report
    if fault == "postcommit_resource":
        assert report["delivery"][0]["receipt"]["update_number"] == 1
        assert report["delivery"][0]["training"] == owner.last_update
    elif fault == "seventh_train":
        assert report["delivery"][-1]["exposure_id"] == "first_B-train-B-0"
        assert "receipt" not in report["delivery"][-1]
    json.dumps(report, allow_nan=False)


def test_should_refuse_changed_corpus_before_runtime_or_learner_work(monkeypatch):
    examples = deepcopy(sequence_examples())
    object.__setattr__(examples[0], "role", "final_test")
    monkeypatch.setattr(runner, "sequence_examples", lambda: examples)
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid corpus reached TRAIN diagnosis work")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = HealthLearner()
    with pytest.raises(ValueError):
        runner.run_train_health(owner, {}, lambda event: None)
    assert calls == [] and owner.training == []


def completed_work(completed):
    report = deepcopy(completed[1])
    # Synthetic shape only; actual CLI state binding is independently checked.
    report["initial_binding"] = {"weights": {"tiny_trainables": {"elements": 7}}}
    counts = dict(
        decision_forward_calls=7,
        scorer_calls=0,
        native_prediction_calls_attempted=0,
        native_prediction_calls_returned=0,
        initializer_cpu_elements=1,
        initializer_meta_elements=511147008,
        candidate_sequence_lengths=[256] * 21,
    )
    return report, counts


def test_should_admit_exact_native_health_and_loader_counts_at_declared_limits(completed):
    report, counts = completed_work(completed)
    original_report, original_counts = deepcopy(report), deepcopy(counts)
    runner.validate_completed_work(report, counts)
    assert report == original_report and counts == original_counts


@pytest.mark.parametrize("fault", ["forward", "health", "scorer", "initializer", "length"])
def test_should_refuse_overbudget_or_uncredited_completed_work(completed, fault):
    report, counts = completed_work(completed)
    if fault == "forward":
        counts["decision_forward_calls"] = 8
    elif fault == "health":
        report["update_health_work"]["update_health_reports_completed"] = 6
    elif fault == "scorer":
        counts["scorer_calls"] = 1
    elif fault == "initializer":
        counts["initializer_cpu_elements"] = 2
    else:
        counts["candidate_sequence_lengths"][0] = 257
    with pytest.raises(ValueError):
        runner.validate_completed_work(report, counts)


def cli_args(tmp_path, output):
    return [
        "--reference",
        str(tmp_path / "reference.json"),
        "--source",
        str(tmp_path / "source"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--output",
        str(output),
    ]


def test_should_refuse_existing_cli_output_before_reference_or_loader_work(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserve diagnosis")
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("existing output reached source or loader")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output))
    assert error.value.code == 2 and calls == []
    assert output.read_bytes() == b"preserve diagnosis"


def test_should_discard_prediction_callable_and_retain_loader_work_emitter_after_failed_load(
    tmp_path, monkeypatch
):
    from tests.test_native_optimizer_reset_control import reset_reference

    reference = reset_reference()
    reads: list[Any] = []
    options_seen: list[dict[str, Any]] = []
    emissions: list[None] = []
    resource = ModuleType("resource")
    setattr(resource, "RUSAGE_SELF", 0)
    setattr(resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=1))
    monkeypatch.setitem(sys.modules, "resource", resource)

    def read(path, *, include_after_A=False):
        reads.append((path, include_after_A))
        return deepcopy(reference)

    def load(source, checkpoint, progress, **options):
        options_seen.append(options)

        def emit():
            emissions.append(None)
            progress(
                dict(
                    event="work_observed",
                    observed_work={
                        "decision_forward_calls": 0,
                        "scorer_calls": 0,
                        "native_prediction_calls_attempted": 0,
                    },
                )
            )

        options["work_reporter"](emit)
        raise RuntimeError("controlled scalar loader failure")

    monkeypatch.setattr(runner, "load_reference", read)
    monkeypatch.setattr(runner, "_load", load)
    output = tmp_path / "failed-load.json"
    assert runner.main(cli_args(tmp_path, output)) == 1
    assert reads == [(tmp_path / "reference.json", True)]
    assert len(options_seen) == 1
    assert options_seen[0]["prediction_limit"] == 1
    assert options_seen[0]["forward_limit"] == 7
    assert options_seen[0]["token_limit"] == 5376
    assert emissions == [None]
    report = json.loads(output.read_text())
    assert report["complete"] is False and report["model_load_attempts"] == 1
    assert (
        report["prediction_calls"]
        == report["replay_examples"]
        == report["probe_updates"]
        == report["final_examples"]
        == 0
    )
    assert "controlled scalar loader failure" in report["error"]
    assert report["events"][-1]["event"] == "work_observed"
    assert report["events"][-1]["observed_work"]["native_prediction_calls_attempted"] == 0
