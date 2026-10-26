"""Synthetic saved controls and recording native callbacks; no model loading.

The linear fixture is an explicit valid zero-coefficient snapshot with declared
native counters, not a claim that this state was learned by a training history.
One shared lexical report costs twelve ordinary lexical callback observations.
"""

from copy import deepcopy
from dataclasses import asdict
import builtins
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_native_decision_comparison as comparison
from scripts.run_decision_reference import example_cases
from src.app.decision_reference_evaluation import evaluate_decision_reference


@pytest.fixture(scope="module")
def controls():
    lexical = json.loads(json.dumps(evaluate_decision_reference(example_cases())))
    catalog = example_cases()[0].request.allowed_candidates
    linear = {
        "definition": "typed_linear_routing_v1",
        "complete": True,
        "feature_spec": {
            "vocabulary": ["credentials", "invoice", "parcel", "password", "refund", "tracking"],
            "catalog": [asdict(candidate) for candidate in catalog],
        },
        "snapshots": [
            {
                "update": 6,
                "state": {
                    "policy": {"feature_dim": 6, "num_classes": 3},
                    "coefficients": [[0.0] * 3 for _ in range(7)],
                    "completed_updates": 6,
                    "training_examples": 18,
                    "mistake_updates": 6,
                },
            }
        ],
    }
    return lexical, linear


def native_reply(wire: dict[str, Any]) -> dict[str, Any]:
    options = wire["questions"]["decision"]["criteria"]
    query = wire["state"]
    wanted = (
        "password"
        if "password" in query or "credentials" in query
        else "shipping"
        if "delivery" in query
        else "billing"
    )
    selected = wanted if wanted in options else next(iter(options))
    probabilities = {
        name: 0.8 if name == selected else 0.2 / (len(options) - 1) for name in options
    }
    return {
        "model": "Qwen/Qwen3.5-2B",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": selected,
                "probabilities": probabilities,
                "confidence": (0.8 - 1 / len(options)) / (1 - 1 / len(options)),
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 0},
        "metadata": {
            "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "method": "lora_decision_head",
            "temperature": 1.518796342858676,
            "max_length": 256,
            "backbone_parameter_dtypes": ["torch.float32"],
            "candidate_sequences": len(options),
            "prefix_cache": {"enabled": False, "mode": "independent_candidates"},
            "inference_seconds": 0.01,
        },
    }


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("invalid saved controls must refuse before prediction or loading")


@pytest.fixture(scope="module")
def successful(controls):
    lexical, linear = deepcopy(controls)
    original = deepcopy((lexical, linear))
    calls: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        return native_reply(wire)

    report = comparison.compare_decisions(predict, lexical, linear, progress=events.append)
    return report, calls, (lexical, linear), original, events


def test_should_compare_all_six_original_and_reverse_requests_without_forwarding_labels(successful):
    report, calls, _, _, events = successful
    assert report["complete"] is True, report.get("error")
    assert len(calls) == 12
    assert sum(len(wire["questions"]["decision"]["criteria"]) for wire in calls) == 32
    for index, record in enumerate(example_cases()):
        original, reverse = calls[2 * index : 2 * index + 2]
        assert original["state"] == reverse["state"] == record.request.query
        assert list(original["questions"]["decision"]["criteria"].items()) == [
            (candidate.candidate_id, candidate.description)
            for candidate in record.request.allowed_candidates
        ]
        assert list(reverse["questions"]["decision"]["criteria"]) == list(
            reversed(list(original["questions"]["decision"]["criteria"]))
        )
        for wire in (original, reverse):
            assert set(wire) == {"model", "state", "questions"}
            assert wire["model"] == "open-jev"
            assert not {"target_id", "role", "source_group", "labels"}.intersection(wire)
    assert set(report["results"]) == {"lexical", "linear", "native"}
    assert len(report["results"]["native"]["cases"]) == 6
    assert len(report["native_returns"]) == 12
    assert len(events) == 13
    assert events[0]["event"] == "comparison_linear_complete"
    assert events[0]["selected_update"] == 6
    assert events[0]["linear_report"] == report["results"]["linear"]
    assert all(event["event"] == "comparison_native_reply" for event in events[1:])
    assert all(set(event) == {"event", "request", "response_json"} for event in events[1:])
    assert "unequal pretraining/capacity/compute" in report["qualification"]
    json.dumps(report, allow_nan=False)


def test_should_keep_saved_controls_unchanged_and_count_only_new_linear_readouts(successful):
    report, _, current, before, _ = successful
    assert current == before
    assert report["results"]["lexical"] == before[0]
    linear = report["results"]["linear"]
    assert len(linear["cases"]) == 6
    assert linear["work"]["prediction_calls_validated"] == 12
    assert linear["learning_active"] is None and linear["training_updates"] is None
    assert report["results"]["native"]["work"]["prediction_calls_validated"] == 12


def test_should_retain_missing_candidates_and_signed_metric_differences(successful):
    report, _, _, _, _ = successful
    for name in ("lexical", "linear", "native"):
        result = report["results"][name]
        assert result["metrics"]["candidate_recall"] == pytest.approx(4 / 6)
        assert result["metrics"]["nll_mean"] is None
        assert result["metrics"]["zero_target_probability_count"] == 2
        assert [row["candidate_recalled"] for row in result["cases"]][-2:] == [False, False]
    for name in ("lexical", "linear"):
        contrast = report["contrasts"]["native_minus_" + name]
        for metric in ("accuracy", "candidate_recall", "brier_mean", "ece_5bin"):
            assert contrast[metric] == pytest.approx(
                report["results"]["native"]["metrics"][metric]
                - report["results"][name]["metrics"][metric]
            )
        assert contrast["nll_mean"] is None
        assert contrast["nll_status"] == "nonfinite_comparison_unavailable"
    assert report["directions"] == {
        "accuracy": "higher",
        "candidate_recall": "higher",
        "brier_mean": "lower",
        "nll_mean": "lower",
        "ece_5bin": "lower",
    }


@pytest.mark.parametrize(
    "fault",
    [
        "query",
        "target",
        "role",
        "group",
        "lexical-selected",
        "linear-incomplete",
        "duplicate-final",
        "counter",
        "vocabulary",
        "description",
    ],
)
def test_should_refuse_malformed_or_unmatched_controls_before_native_prediction(controls, fault):
    lexical, linear = deepcopy(controls)
    if fault == "query":
        lexical["cases"][0]["request"]["query"] = "changed query"
    elif fault == "target":
        lexical["cases"][0]["target_id"] = "shipping"
    elif fault == "role":
        lexical["cases"][0]["role"] = "final"
    elif fault == "group":
        lexical["cases"][0]["source_group"] = "replacement"
    elif fault == "lexical-selected":
        lexical["cases"][0]["result"]["selected_id"] = "not-allowed"
    elif fault == "linear-incomplete":
        linear["complete"] = False
    elif fault == "duplicate-final":
        linear["snapshots"].append(deepcopy(linear["snapshots"][0]))
    elif fault == "counter":
        linear["snapshots"][0]["state"]["training_examples"] = 17
    elif fault == "vocabulary":
        linear["feature_spec"]["vocabulary"][0] = "changed"
    else:
        linear["feature_spec"]["catalog"][0]["description"] = "replacement"
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        raise AssertionError("invalid controls reached native prediction")

    report = comparison.compare_decisions(predict, lexical, linear)
    assert report["complete"] is False and report["error"]
    assert calls == [] and report["native_returns"] == []


def test_should_keep_all_valid_native_order_disagreements_without_selecting_or_retrying(controls):
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        reply = native_reply(wire)
        allowed = wire["questions"]["decision"]["criteria"]
        reply["answers"]["decision"].update(
            choice=next(iter(allowed)),
            probabilities={name: 1 / len(allowed) for name in allowed},
            confidence=0.0,
        )
        return reply

    report = comparison.compare_decisions(predict, *deepcopy(controls))
    assert report["complete"] is True, report.get("error")
    assert len(calls) == 12 and len(report["native_returns"]) == 12
    native = report["results"]["native"]
    assert native["candidate_order_control"]["all_equal"] is False
    assert all(row["order_control"]["probabilities_agree"] is True for row in native["cases"])
    assert all(row["order_control"]["choice_agrees"] is False for row in native["cases"])


def test_should_retain_a_complete_case_before_third_native_call_fails(controls):
    calls: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    def predict(wire):
        assert events[0]["event"] == "comparison_linear_complete"
        calls.append(deepcopy(wire))
        if len(calls) == 3:
            raise RuntimeError("controlled third native call failure")
        return native_reply(wire)

    lexical, linear = deepcopy(controls)
    report = comparison.compare_decisions(predict, lexical, linear, progress=events.append)
    assert report["complete"] is False and len(calls) == 3
    assert "controlled third native call failure" in report["error"]
    native = report["results"]["native"]
    assert len(native["cases"]) == 1
    assert native["work"]["prediction_calls_attempted"] == 3
    assert native["work"]["prediction_calls_returned"] == 2
    assert native["work"]["prediction_calls_validated"] == 2
    assert native["work"]["unreturned_prediction_work_unknown"] is True
    assert len(report["native_returns"]) == 3
    assert "response_json" in report["native_returns"][0]
    assert "response_json" in report["native_returns"][1]
    assert "response_json" not in report["native_returns"][2]
    assert "contrasts" not in report
    assert events[0]["selected_update"] == 6
    assert events[0]["linear_report"]["work"]["prediction_calls_validated"] == 12
    assert [event["event"] for event in events] == [
        "comparison_linear_complete",
        "comparison_native_reply",
        "comparison_native_reply",
    ]
    json.dumps(report, allow_nan=False)


def test_should_retain_raw_return_and_frozen_wire_before_refusing_callback_mutation(controls):
    calls: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        wire["state"] = "callback changed the query"
        return native_reply(wire)

    report = comparison.compare_decisions(predict, *deepcopy(controls))
    assert report["complete"] is False and len(calls) == 1
    row = report["native_returns"][0]
    assert row["request"] == calls[0]
    assert json.loads(row["response_json"])["answers"]["decision"]["type"] == "choice"
    assert report["results"]["native"]["work"]["prediction_calls_attempted"] == 1
    json.dumps(report, allow_nan=False)


def test_should_keep_nonfinite_native_return_as_raw_text_before_strict_validation_failure(controls):
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        reply = native_reply(wire)
        reply["metadata"]["invalid"] = float("nan")
        return reply

    report = comparison.compare_decisions(predict, *deepcopy(controls))
    assert report["complete"] is False and calls == [None]
    assert "NaN" in report["native_returns"][0]["response_json"]
    native = report["results"]["native"]
    assert native["work"]["prediction_calls_attempted"] == 1
    assert native["work"]["prediction_calls_validated"] == 0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "name,value",
    [
        ("prediction_limit", True),
        ("prediction_limit", 0),
        ("prediction_limit", 129),
        ("forward_limit", 1025),
        ("token_limit", 262145),
    ],
)
def test_should_refuse_invalid_load_limits_before_importing_inference_dependencies(
    monkeypatch, tmp_path, name, value
):
    from scripts import run_open_jev_native_smoke as smoke

    signature = inspect.signature(smoke._load)
    assert [
        signature.parameters[key].default
        for key in ("prediction_limit", "forward_limit", "token_limit")
    ] == [2, 4, 1024]
    original_import = builtins.__import__
    imports: list[str] = []

    def guarded_import(module, *args, **kwargs):
        if module.split(".")[0] in {"resource", "torch", "jev", "transformers", "peft"}:
            imports.append(module)
            raise AssertionError("invalid limits reached an inference import")
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError):
        smoke._load(tmp_path / "source", tmp_path / "checkpoint", forbidden, **{name: value})
    assert imports == []


def cli_args(tmp_path: Path, lexical: Path, linear: Path, output: Path) -> list[str]:
    return [
        "--lexical-report",
        str(lexical),
        "--linear-report",
        str(linear),
        "--source",
        str(tmp_path / "source"),
        "--checkpoint",
        str(tmp_path / "checkpoint"),
        "--output",
        str(output),
    ]


def test_should_refuse_existing_cli_output_before_controls_or_loader(monkeypatch, tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("original")
    monkeypatch.setattr(comparison, "_read", forbidden)
    monkeypatch.setattr(comparison, "_load", forbidden)
    with pytest.raises(SystemExit) as caught:
        comparison.main(
            cli_args(tmp_path, tmp_path / "lexical.json", tmp_path / "linear.json", output)
        )
    assert caught.value.code == 2 and output.read_text() == "original"


def test_should_refuse_invalid_cli_controls_without_even_attempting_a_model_load(
    controls, monkeypatch, tmp_path
):
    lexical, linear = deepcopy(controls)
    lexical["complete"] = False
    lexical_path, linear_path = tmp_path / "lexical.json", tmp_path / "linear.json"
    lexical_path.write_text(json.dumps(lexical))
    linear_path.write_text(json.dumps(linear))
    loads: list[None] = []

    def load(*args, **kwargs):
        loads.append(None)
        raise AssertionError("inadmissible controls reached model loader")

    monkeypatch.setattr(comparison, "_load", load)
    output = tmp_path / "refused.json"
    assert comparison.main(cli_args(tmp_path, lexical_path, linear_path, output)) == 1
    report = json.loads(output.read_text())
    assert loads == [] and report["complete"] is False
    assert report["model_load_attempts"] == 0
    assert (
        report["current_work"]["training_updates"] == report["current_work"]["training_rows"] == 0
    )


def test_should_pass_separate_native_work_limits_to_one_lazy_cli_load(
    controls, monkeypatch, tmp_path
):
    lexical_path, linear_path = tmp_path / "lexical.json", tmp_path / "linear.json"
    lexical_path.write_text(json.dumps(controls[0]))
    linear_path.write_text(json.dumps(controls[1]))
    calls: list[str] = []
    caps: list[dict[str, Any]] = []

    def load(source, checkpoint, progress, **limits):
        calls.append("load")
        caps.append(limits)

        def predict(wire):
            calls.append("predict")
            return {"declared_fixture": True}

        return predict

    def compare(predict, lexical, linear, progress):
        calls.append("compare")
        assert lexical == controls[0] and linear == controls[1]
        assert calls == ["compare"]
        assert predict({"metadata_fixture_only": True}) == {"declared_fixture": True}
        return {"complete": True, "results": {}, "native_returns": []}

    monkeypatch.setattr(comparison, "_load", load)
    monkeypatch.setattr(comparison, "compare_decisions", compare)
    output = tmp_path / "routed.json"
    assert comparison.main(cli_args(tmp_path, lexical_path, linear_path, output)) == 0
    report = json.loads(output.read_text())
    assert calls == ["compare", "load", "predict"]
    assert caps == [{"prediction_limit": 12, "forward_limit": 32, "token_limit": 8192}]
    assert report["model_load_attempts"] == 1
    assert report["current_work"]["compute_matched"] is False
