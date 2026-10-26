"""A6/capture/B6/terminal consolidation orchestration using scalar fakes."""

from copy import deepcopy
import json
from math import log
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import run_open_jev_b_control as loader
from scripts import run_open_jev_optimizer_control as runner
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_native_choice_sequence import native_reply
from tests.test_native_cold_b_control import reference_payload, write_reference
from tests.test_native_optimizer_reset_control import (
    ResetLearner,
    cli_args,
    reset_reference,
    summary_rows,
)


class ConsolidationLearner(ResetLearner):
    """Records interventions only; no numerical consolidation is simulated."""

    def __init__(self, timeline, *, fail_at=None, fail_capture=False, fail_consolidation=False):
        super().__init__(timeline, fail_at=fail_at)
        self.fail_capture, self.fail_consolidation = fail_capture, fail_consolidation
        self._consolidation_work = dict(
            reference_attempts=0,
            reference_copy_elements_attempted=0,
            reference_rng_read_elements_attempted=0,
            references_completed=0,
            consolidation_attempts=0,
            consolidations_completed=0,
        )
        self._projection_work = dict(projection_attempts=0, projections_completed=0)
        self._adapter_reference = None
        self.last_adapter_reference = None
        self.last_adapter_projection = None

    @property
    def adapter_consolidation_work(self):
        return dict(self._consolidation_work)

    @property
    def adapter_projection_work(self):
        return dict(self._projection_work)

    def capture_adapter_reference(self):
        self.timeline.append(("capture", self.work["committed_updates"]))
        self._consolidation_work["reference_attempts"] += 1
        if self.fail_capture:
            raise TrainingUpdateFailed(
                "controlled capture failure",
                "fixture_capture",
                {**self.work, **self.adapter_consolidation_work},
            )
        self._consolidation_work["references_completed"] += 1
        self._adapter_reference = object()
        receipt = {
            "definition": "open_jev_owned_adapter_reference_v1",
            "source_native_work": self.work,
            "adapter_consolidation_work": self.adapter_consolidation_work,
        }
        self.last_adapter_reference = deepcopy(receipt)
        return receipt

    def consolidate_adapters(self, rank):
        self.timeline.append(("consolidate", self.work["committed_updates"], rank))
        self._consolidation_work["consolidation_attempts"] += 1
        self._projection_work["projection_attempts"] += 1
        if self.fail_consolidation:
            raise TrainingUpdateFailed(
                "controlled consolidation failure",
                "fixture_consolidation",
                {**self.work, **self.adapter_consolidation_work, **self.adapter_projection_work},
            )
        self._consolidation_work["consolidations_completed"] += 1
        self._projection_work["projections_completed"] += 1
        self._reset_work["optimizer_reset_attempts"] += 1
        self._reset_work["optimizer_resets_completed"] += 1
        self._adapter_reference = None
        receipt = {
            "definition": "open_jev_owned_adapter_consolidation_v1",
            "effective_rank": rank,
            "coefficients": {"reference": 0.5, "current": 0.5},
            "head_policy": "preserve_current",
            "adapter_consolidation_work": self.adapter_consolidation_work,
        }
        self.last_adapter_projection = deepcopy(receipt)
        return receipt


@pytest.fixture(scope="module")
def reference():
    result = reset_reference()
    result["after_B_weights"] = deepcopy(result["after_A_weights"])
    for row in result["after_B_weights"].values():
        row["after_sha256"] = "1" * 64
    return result


@pytest.fixture(scope="module")
def completed(reference):
    timeline: list[Any] = []
    owner = ConsolidationLearner(timeline)
    wires: list[Any] = []
    progress: list[str] = []

    def predict(wire):
        timeline.append(("predict", wire["state"]))
        wires.append(deepcopy(wire))
        return native_reply(wire)

    report: dict[str, Any] = {}
    runner.run_optimizer_control(
        owner, predict, reference, report, progress.append, consolidation_rank=8
    )
    return report, owner, timeline, wires, progress


def test_should_capture_after_a_without_reset_and_consolidate_only_after_all_b_updates(completed):
    report, owner, timeline, wires, progress = completed
    assert [event[0] for event in timeline] == ["train"] * 6 + ["capture"] + ["train"] * 6 + [
        "consolidate"
    ] + ["predict"] * 6
    assert timeline[6] == ("capture", 6) and timeline[13] == ("consolidate", 12, 8)
    assert [target for _, target in owner.training] == ["shipping"] * 6 + ["password"] * 6
    assert len(wires) == 6 and report["control_complete"] is True
    assert report["app_committed_updates"] == 12
    assert report["reference_receipt"]["source_native_work"]["committed_updates"] == 6
    assert report["consolidation_receipt"] == owner.last_adapter_projection
    assert report["reset_receipt"] == {
        "optimizer_reset_attempts": 1,
        "optimizer_resets_completed": 1,
    }
    assert [event for event in progress if "adapter" in event] == [
        "before_adapter_reference",
        "adapter_reference_finished",
        "before_adapter_consolidation",
        "adapter_consolidation_finished",
    ]
    assert "before_optimizer_reset" not in progress and "optimizer_reset_finished" not in progress


def test_should_keep_unique_causal_deliveries_and_unlabeled_terminal_readouts(completed):
    report, owner, _, wires, _ = completed
    rows = report["delivery"]
    assert len(rows) == len({row["exposure_id"] for row in rows}) == 12
    assert [row["phase"] for row in rows] == ["after_A"] * 6 + ["after_B_consolidation"] * 6
    assert [row["observed_at"] for row in rows] == list(range(0, 24, 2))
    assert [row["receipt"]["arrived_at"] for row in rows] == list(range(1, 24, 2))
    assert [row["receipt"]["update_number"] for row in rows] == list(range(1, 13))
    assert all(
        row["prelabel_updates"] == 0
        and row["receipt"]["arrived_at"] == row["receipt"]["applied_at"]
        for row in rows
    )
    assert all(
        row["phase"] == "after_adapter_consolidation_B" and row["role"] == "outer_selection"
        for row in report["readouts"]
    )
    assert all(set(wire) == {"model", "state", "questions"} for wire in wires)
    assert report["adapter_consolidation_work"] == owner.adapter_consolidation_work
    assert report["native_work"] == owner.work and report["inbox_stopped"] is False
    json.dumps(report, allow_nan=False)


def consolidation_rows():
    rows = summary_rows()
    for row in rows:
        row["phase"] = "after_adapter_consolidation_B"
    return rows


def test_should_keep_signed_warm_endpoint_and_original_baseline_comparisons(reference):
    original = deepcopy(reference)
    result = runner.summarize_consolidation_control(consolidation_rows(), reference)
    assert result["contrasts"]["consolidation_minus_warm_after_B"]["B"]["accuracy"] == 0.0
    assert result["contrasts"]["consolidation_minus_baseline"]["B"]["accuracy"] == 1.0
    assert result["contrasts"]["B_acquisition"]["nll_mean"] == pytest.approx(log(0.1) - log(0.9))
    assert result["criteria"] == {
        "B_after_phase_success": True,
        "A_retained": False,
        "anchors_preserved": True,
    }
    assert result["recovery"] == "not_tested" and reference == original


def test_should_keep_infinite_nll_unavailable_differences_and_original_anchor_guard(reference):
    rows = consolidation_rows()
    rows[2].update(nll=None, nll_status="infinite")
    for row in rows:
        if row["context"] == "anchor":
            row["nll"] = reference["baseline_summary"]["anchor"]["nll_mean"] + 0.11
    result = runner.summarize_consolidation_control(rows, reference)
    assert result["by_context"]["B"]["nll_mean"] is None
    assert result["contrasts"]["consolidation_minus_warm_after_B"]["B"]["nll_mean"] is None
    assert (
        result["contrasts"]["consolidation_minus_warm_after_B"]["B"]["nll_status"]
        == "nonfinite_comparison_unavailable"
    )
    assert result["criteria"]["anchors_preserved"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "capture",
        "after_capture",
        "b_binding",
        "consolidation",
        "after_consolidation",
        "readout",
        "postcommit",
    ],
)
def test_should_keep_paid_capture_consolidation_and_training_prefix_before_callback_failures(
    reference, fault
):
    owner = ConsolidationLearner(
        [], fail_capture=fault == "capture", fail_consolidation=fault == "consolidation"
    )
    predictions: list[None] = []

    def predict(wire):
        predictions.append(None)
        if fault == "readout" and len(predictions) == 2:
            raise RuntimeError("controlled second consolidation readout failure")
        return native_reply(wire)

    def progress(event):
        stops = {
            "after_capture": "adapter_reference_finished",
            "b_binding": "before_adapter_consolidation",
            "after_consolidation": "adapter_consolidation_finished",
            "postcommit": "training_finished",
        }
        if event == stops.get(fault):
            raise RuntimeError("controlled consolidation boundary stop")

    report: dict[str, Any] = {}
    with pytest.raises((RuntimeError, ValueError)):
        runner.run_optimizer_control(
            owner, predict, reference, report, progress, consolidation_rank=8
        )
    trained, captured, consolidated, prediction_calls = {
        "capture": (6, 0, 0, 0),
        "after_capture": (6, 1, 0, 0),
        "b_binding": (12, 1, 0, 0),
        "consolidation": (12, 1, 0, 0),
        "after_consolidation": (12, 1, 1, 0),
        "readout": (12, 1, 1, 2),
        "postcommit": (1, 0, 0, 0),
    }[fault]
    assert len(owner.training) == report["app_committed_updates"] == trained
    assert sum("receipt" in row for row in report["delivery"]) == trained
    assert owner.adapter_consolidation_work["references_completed"] == captured
    assert owner.adapter_consolidation_work["consolidations_completed"] == consolidated
    assert report["adapter_consolidation_work"] == owner.adapter_consolidation_work
    assert report["optimizer_reset_work"] == {
        "optimizer_reset_attempts": consolidated,
        "optimizer_resets_completed": consolidated,
    }
    if captured:
        assert report["reference_receipt"] == owner.last_adapter_reference
    if consolidated:
        assert report["consolidation_receipt"] == owner.last_adapter_projection
        assert report["reset_receipt"] == owner.optimizer_reset_work
    assert len(predictions) == prediction_calls and report["control_complete"] is False
    assert "summary" not in report
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    ["bool", "negative", "too_large", "no_reset", "rehearsal", "projection", "reset_reference"],
)
def test_should_refuse_invalid_consolidation_options_before_runtime(reference, monkeypatch, fault):
    options: dict[str, Any] = {"consolidation_rank": 8}
    if fault in ("bool", "negative", "too_large"):
        options["consolidation_rank"] = {"bool": True, "negative": -1, "too_large": 9}[fault]
    elif fault == "no_reset":
        options["reset_optimizer"] = False
    elif fault == "rehearsal":
        options["rehearsal"] = True
    elif fault == "projection":
        options.update(
            projection_rank=4,
            reset_reference={"source_sha256": "x", "by_context": {"A": {}, "B": {}, "anchor": {}}},
        )
    else:
        options["reset_reference"] = {"by_context": {"A": {}, "B": {}, "anchor": {}}}

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid consolidation option reached runtime")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    owner = ConsolidationLearner([])
    with pytest.raises(ValueError):
        runner.run_optimizer_control(owner, forbidden, reference, {}, lambda event: None, **options)
    assert owner.training == []


@pytest.mark.parametrize("fault", ["spent_capture", "pending_reference", "projection_receipt"])
def test_should_refuse_inherited_consolidation_or_reference_history_before_runtime(
    reference, monkeypatch, fault
):
    owner = ConsolidationLearner([])
    if fault == "spent_capture":
        owner._consolidation_work["reference_attempts"] = 1
    elif fault == "pending_reference":
        owner._adapter_reference = object()
    else:
        owner.last_adapter_projection = {"definition": "old receipt"}

    def forbidden(*args, **kwargs):
        raise AssertionError("inherited consolidation reached runtime")

    monkeypatch.setattr(runner, "ExperienceInbox", forbidden)
    with pytest.raises(ValueError):
        runner.run_optimizer_control(
            owner, forbidden, reference, {}, lambda event: None, consolidation_rank=8
        )
    assert owner.training == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--consolidation-rank", "8", "--optimizer-reset", "0"],
        ["--consolidation-rank", "8", "--rehearsal"],
        ["--consolidation-rank", "8", "--update-health"],
        ["--consolidation-rank", "8", "--projection-rank", "4", "--reset-reference", "reset.json"],
        ["--consolidation-rank", "8", "--reset-reference", "reset.json"],
        ["--consolidation-rank", "9"],
    ],
)
def test_should_refuse_invalid_cli_mode_before_output_or_loading(tmp_path, monkeypatch, extra):
    calls: list[None] = []

    def forbidden(*args, **kwargs):
        calls.append(None)
        raise AssertionError("invalid consolidation CLI reached loader")

    monkeypatch.setattr(runner, "load_reference", forbidden)
    monkeypatch.setattr(runner, "_load", forbidden)
    output = tmp_path / "not-created.json"
    with pytest.raises(SystemExit) as error:
        runner.main(cli_args(tmp_path, output) + extra)
    assert error.value.code == 2 and calls == [] and not output.exists()


def after_b_payload(reference):
    payload = reference_payload()
    payload["summary"]["by_phase"]["after_A"] = deepcopy(reference["after_A_summary"])
    payload["weight_checkpoints"].extend(
        [
            {
                "after_updates": 6,
                "model_training": False,
                "weights": deepcopy(reference["after_A_weights"]),
            },
            {
                "after_updates": 12,
                "model_training": False,
                "weights": deepcopy(reference["after_B_weights"]),
            },
        ]
    )
    return payload


def test_should_opt_in_after_b_binding_without_changing_default_or_after_a_loader_schema(
    reference, tmp_path, monkeypatch
):
    path, body, _ = write_reference(tmp_path, monkeypatch, after_b_payload(reference))
    default = loader.load_reference(path)
    after_a = loader.load_reference(path, include_after_A=True)
    selected = loader.load_reference(path, include_after_A=True, include_after_B=True)
    assert "after_B_weights" not in default and "after_B_weights" not in after_a
    assert set(selected) == set(after_a) | {"after_B_weights"}
    assert selected["after_B_weights"] == reference["after_B_weights"]
    runner.validate_after_B_weights(deepcopy(reference["after_B_weights"]), selected)
    selected["after_B_weights"]["head.weight"]["after_sha256"] = "0" * 64
    assert (
        loader.load_reference(path, include_after_B=True)["after_B_weights"]
        == reference["after_B_weights"]
    )
    assert path.read_bytes() == body


@pytest.mark.parametrize("fault", ["training_mode", "duplicate"])
def test_should_require_one_after_b_checkpoint_in_evaluation_mode(
    reference, tmp_path, monkeypatch, fault
):
    payload = after_b_payload(reference)
    if fault == "training_mode":
        payload["weight_checkpoints"][-1]["model_training"] = True
    else:
        payload["weight_checkpoints"].append(deepcopy(payload["weight_checkpoints"][-1]))
    path, _, _ = write_reference(tmp_path, monkeypatch, payload)
    with pytest.raises(ValueError):
        loader.load_reference(path, include_after_B=True)


def test_should_refuse_missing_foreign_or_changed_after_b_weights(reference):
    original = deepcopy(reference)
    for fault in ("missing", "foreign", "changed"):
        observed = deepcopy(reference["after_B_weights"])
        if fault == "missing":
            observed.pop("head.weight")
        elif fault == "foreign":
            observed["foreign.weight"] = deepcopy(observed["head.weight"])
        else:
            observed["head.weight"]["after_sha256"] = "9" * 64
        with pytest.raises(ValueError):
            runner.validate_after_B_weights(observed, reference)
    assert reference == original


@pytest.mark.parametrize("fault", [None, "protected_state", "head", "stale_gradient"])
def test_should_allow_only_lora_weight_changes_and_cleared_lora_gradients_at_terminal_boundary(
    monkeypatch, fault
):
    owner = SimpleNamespace(
        _trainable={
            "head.weight": SimpleNamespace(grad=object()),
            "backbone.lora_A.default.weight": SimpleNamespace(grad=None),
            "backbone.lora_B.default.weight": SimpleNamespace(grad=None),
        }
    )
    boundary = {
        "gradient_sha256": "all-old-gradients",
        "private_rng_sha256": "private",
        "caller_rng_sha256": "caller",
        "model_training": False,
        "native_work": {"committed_updates": 12},
    }
    state = deepcopy(boundary)
    state["gradient_sha256"] = "lora-gradients-cleared"
    bound = {
        "head.weight": {"after_sha256": "head"},
        "backbone.lora_A.default.weight": {"after_sha256": "old-a"},
        "backbone.lora_B.default.weight": {"after_sha256": "old-b"},
    }
    observed = deepcopy(bound)
    observed["backbone.lora_A.default.weight"]["after_sha256"] = "consolidated-a"
    observed["backbone.lora_B.default.weight"]["after_sha256"] = "consolidated-b"
    monkeypatch.setattr(runner, "_head_gradient_hash", lambda learner: "same-head-gradient")
    if fault == "protected_state":
        state["private_rng_sha256"] = "changed"
    elif fault == "head":
        observed["head.weight"]["after_sha256"] = "changed"
    elif fault == "stale_gradient":
        owner._trainable["backbone.lora_A.default.weight"].grad = object()
    if fault is None:
        runner._validate_projected_state(
            owner, observed, state, boundary, bound, "same-head-gradient"
        )
    else:
        with pytest.raises(ValueError):
            runner._validate_projected_state(
                owner, observed, state, boundary, bound, "same-head-gradient"
            )
