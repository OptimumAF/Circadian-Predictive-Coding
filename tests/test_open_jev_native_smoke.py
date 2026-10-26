"""Recording-fake qualification of the two-call native smoke orchestration.

No Torch, upstream module, model, tokenizer, network or inference is used here.
Native probabilities and provenance are declared fixture observations only.
"""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_open_jev_native_smoke as smoke


BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TEMPERATURE = 1.518796342858676


def native_reply() -> dict[str, Any]:
    return {
        "model": "Qwen/Qwen3.5-2B",
        "answers": {
            "decision": {
                "type": "choice",
                "choice": "password",
                "probabilities": {"password": 0.8, "billing": 0.2},
                "confidence": 0.6,
            }
        },
        "usage": {"input_tokens": 96, "output_tokens": 0},
        "metadata": {
            "base_revision": BASE_REVISION,
            "method": "lora_decision_head",
            "temperature": TEMPERATURE,
            "max_length": 256,
            "backbone_parameter_dtypes": ["torch.float32"],
            "candidate_sequences": 2,
            "inference_seconds": 0.01,
            "prefix_cache": {"enabled": False, "mode": "independent_candidates"},
        },
    }


def record_replies(replies: list[dict[str, Any]]):
    calls: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        return deepcopy(replies[len(calls) - 1])

    return predict, calls


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("must refuse before model loading or another prediction")


def test_should_call_original_and_reverse_once_with_only_frozen_native_inputs():
    predict, calls = record_replies([native_reply(), native_reply()])
    report = smoke.run_case(predict)
    assert report["complete"] is True, report.get("error")
    assert len(calls) == 2
    assert calls[0] == {
        "model": "open-jev",
        "state": "I forgot my password and cannot sign in.",
        "questions": {
            "decision": {
                "type": "choice",
                "instructions": "Choose the customer's support intent.",
                "criteria": {
                    "password": "Password resets and login problems",
                    "billing": "Charges, invoices and refunds",
                },
            }
        },
    }
    assert list(calls[1]["questions"]["decision"]["criteria"]) == ["billing", "password"]
    assert calls[1]["state"] == calls[0]["state"]
    for wire in calls:
        assert not {"target", "target_id", "labels", "role", "source_group"}.intersection(wire)
    assert report["work"]["prediction_calls_attempted"] == 2
    assert report["work"]["prediction_calls_returned"] == 2
    assert report["work"]["prediction_calls_validated"] == 2
    assert report["work"]["unreturned_call_cost_unknown"] is False
    assert report["work"]["returned_unvalidated_cost"] is False
    assert len(report["requests"]) == 2
    assert report["requests"][0]["response"] == native_reply()
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "fault",
    [
        "base_revision",
        "method",
        "temperature",
        "max_length",
        "dtype",
        "candidate_sequences",
        "negative_tokens",
        "bool_tokens",
        "output_tokens",
    ],
)
def test_should_retain_observed_reply_before_refusing_unqualified_native_metadata(fault):
    bad = native_reply()
    if fault == "base_revision":
        bad["metadata"]["base_revision"] = "0" * 40
    elif fault == "method":
        bad["metadata"]["method"] = "pretrained_yes_minus_no_no_training"
    elif fault == "temperature":
        bad["metadata"]["temperature"] = 1.0
    elif fault == "max_length":
        bad["metadata"]["max_length"] = 255
    elif fault == "dtype":
        bad["metadata"]["backbone_parameter_dtypes"] = ["torch.bfloat16"]
    elif fault == "candidate_sequences":
        bad["metadata"]["candidate_sequences"] = 1
    elif fault == "negative_tokens":
        bad["usage"]["input_tokens"] = -1
    elif fault == "bool_tokens":
        bad["usage"]["input_tokens"] = True
    else:
        bad["usage"]["output_tokens"] = 1
    predict, calls = record_replies([bad])
    report = smoke.run_case(predict)
    assert report["complete"] is False
    assert report["error"]
    assert len(calls) == 1
    assert report["work"]["prediction_calls_attempted"] == 1
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["prediction_calls_validated"] == 0
    assert report["work"]["unreturned_call_cost_unknown"] is False
    assert report["work"]["returned_unvalidated_cost"] is True
    assert report["requests"][0]["response"] == bad


def test_should_keep_first_validated_reply_when_second_prediction_raises():
    calls: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        if len(calls) == 2:
            raise RuntimeError("controlled second call failure")
        return native_reply()

    report = smoke.run_case(predict)
    assert report["complete"] is False
    assert "controlled second call failure" in report["error"]
    assert len(calls) == 2
    assert report["requests"][0]["response"] == native_reply()
    assert report["work"]["prediction_calls_attempted"] == 2
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["prediction_calls_validated"] == 1
    assert report["work"]["unreturned_call_cost_unknown"] is True
    assert report["work"]["returned_unvalidated_cost"] is False


def test_should_stop_after_first_unreturned_prediction_without_retry():
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        raise RuntimeError("controlled first call failure")

    report = smoke.run_case(predict)
    assert report["complete"] is False and calls == [None]
    assert report["work"]["prediction_calls_attempted"] == 1
    assert report["work"]["prediction_calls_returned"] == 0
    assert report["work"]["prediction_calls_validated"] == 0
    assert report["work"]["unreturned_call_cost_unknown"] is True
    assert report["work"]["returned_unvalidated_cost"] is False


def test_should_keep_second_return_and_first_validation_when_native_answer_is_forged():
    bad = native_reply()
    bad["answers"]["decision"]["confidence"] = 0.8
    predict, calls = record_replies([native_reply(), bad])
    report = smoke.run_case(predict)
    assert report["complete"] is False and len(calls) == 2
    assert report["work"]["prediction_calls_attempted"] == 2
    assert report["work"]["prediction_calls_returned"] == 2
    assert report["work"]["prediction_calls_validated"] == 1
    assert report["work"]["unreturned_call_cost_unknown"] is False
    assert report["work"]["returned_unvalidated_cost"] is True
    assert report["requests"][0]["response"] == native_reply()
    assert report["requests"][1]["response"] == bad


def test_should_preserve_an_ordinary_wrong_but_order_consistent_native_choice():
    negative = native_reply()
    negative["answers"]["decision"].update(
        choice="billing", probabilities={"password": 0.2, "billing": 0.8}
    )
    predict, calls = record_replies([negative, negative])
    report = smoke.run_case(predict)
    assert report["complete"] is True, report.get("error")
    assert len(calls) == 2 and report["work"]["prediction_calls_validated"] == 2
    assert all(
        row["response"]["answers"]["decision"]["choice"] == "billing" for row in report["requests"]
    )


def test_should_report_valid_order_disagreement_without_retry_or_changing_the_result():
    reversed_reply = native_reply()
    reversed_reply["answers"]["decision"].update(
        choice="billing", probabilities={"password": 0.2, "billing": 0.8}
    )
    predict, calls = record_replies([native_reply(), reversed_reply])
    report = smoke.run_case(predict)
    assert report["complete"] is True and len(calls) == 2
    order = report["order_control"]
    assert order["probabilities_agree"] is False
    assert order["choice_agrees"] is False and order["agreement"] is False
    assert order["max_absolute_probability_difference"] == pytest.approx(0.6)
    assert report["requests"][1]["response"] == reversed_reply


@pytest.mark.parametrize("delta,agrees", [(0.0000005, True), (0.000002, False)])
def test_should_compare_native_probabilities_by_id_with_fixed_absolute_tolerance(delta, agrees):
    other = native_reply()
    other["answers"]["decision"].update(
        probabilities={"billing": 0.2 - delta, "password": 0.8 + delta},
        confidence=0.6 + 2 * delta,
    )
    predict, calls = record_replies([native_reply(), other])
    report = smoke.run_case(predict)
    assert report["complete"] is True and len(calls) == 2
    assert report["order_control"]["choice_agrees"] is True
    assert report["order_control"]["probabilities_agree"] is agrees
    assert report["order_control"]["agreement"] is agrees
    assert report["order_control"]["max_absolute_probability_difference"] == pytest.approx(delta)


def test_should_own_first_return_before_predictor_reuses_its_reply_object():
    first = native_reply()
    expected = deepcopy(first)
    calls: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        if len(calls) == 1:
            return first
        first["metadata"]["base_revision"] = "later mutation of predictor-owned object"
        return native_reply()

    report = smoke.run_case(predict)
    assert report["complete"] is True and len(calls) == 2
    assert report["requests"][0]["response"] == expected


def test_should_retain_frozen_wire_and_paid_response_when_callback_mutates_its_input():
    calls: list[dict[str, Any]] = []

    def predict(wire):
        calls.append(deepcopy(wire))
        wire["state"] = "callback changed the frozen query"
        return native_reply()

    report = smoke.run_case(predict)
    assert report["complete"] is False and len(calls) == 1
    assert report["requests"][0]["request"] == calls[0]
    assert report["requests"][0]["request"]["state"] == "I forgot my password and cannot sign in."
    assert report["requests"][0]["response"] == native_reply()
    assert report["work"]["prediction_calls_attempted"] == 1
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["prediction_calls_validated"] == 0
    assert report["work"]["returned_unvalidated_cost"] is True
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", ["nan", "unsupported"])
def test_should_keep_invalid_return_evidence_in_a_strict_json_serializable_report(invalid):
    bad = native_reply()
    bad["metadata"]["invalid"] = float("nan") if invalid == "nan" else object()
    calls: list[None] = []

    def predict(wire):
        calls.append(None)
        return bad

    report = smoke.run_case(predict)
    assert report["complete"] is False and calls == [None]
    row = report["requests"][0]
    assert "response" not in row
    if invalid == "nan":
        assert "NaN" in row["response_json"]
        assert isinstance(row["response_json"], str)
    else:
        assert isinstance(row["response_unavailable"], str)
    assert report["work"]["prediction_calls_attempted"] == 1
    assert report["work"]["prediction_calls_returned"] == 1
    assert report["work"]["prediction_calls_validated"] == 0
    assert report["work"]["unreturned_call_cost_unknown"] is False
    assert report["work"]["returned_unvalidated_cost"] is True
    assert report["work"]["counter_boundary"] == "run_case_callable_return"
    json.dumps(report, allow_nan=False)


def cli_args(source: Path, checkpoint: Path, output: Path) -> list[str]:
    return ["--source", str(source), "--checkpoint", str(checkpoint), "--output", str(output)]


def cli_paths(tmp_path: Path) -> tuple[Path, Path]:
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    (source / "jev").mkdir(parents=True)
    checkpoint.mkdir()
    (source / "jev" / "serving.py").write_text("# never imported by these fake-loader tests\n")
    (checkpoint / "head.pt").write_text("not a tensor; fake loader never opens this file\n")
    return source, checkpoint


def test_should_refuse_existing_output_before_loading_upstream_or_constructing_model(
    monkeypatch, tmp_path
):
    output = tmp_path / "existing.json"
    output.write_text("original bytes")
    monkeypatch.setattr(smoke, "_load", forbidden)
    monkeypatch.setattr(smoke, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        smoke.main(cli_args(tmp_path / "source", tmp_path / "checkpoint", output))
    assert error.value.code == 2
    assert output.read_text() == "original bytes"


def test_should_route_one_loaded_predictor_to_one_case_and_write_fresh_json(monkeypatch, tmp_path):
    calls: list[str] = []
    source, checkpoint = cli_paths(tmp_path)
    report = {
        "definition": "open-jev-native-choice-smoke-v1",
        "complete": True,
        "requests": [],
        "work": {
            "prediction_calls_attempted": 0,
            "prediction_calls_returned": 0,
            "prediction_calls_validated": 0,
        },
    }

    def load(source, checkpoint, progress):
        calls.append("load")
        return forbidden

    def run(predict):
        calls.append("case")
        assert predict is forbidden
        return deepcopy(report)

    monkeypatch.setattr(smoke, "_load", load)
    monkeypatch.setattr(smoke, "run_case", run)
    output = tmp_path / "fresh.json"
    assert smoke.main(cli_args(source, checkpoint, output)) == 0
    saved = json.loads(output.read_text())
    assert saved["complete"] is True
    assert saved["requests"] == [] and saved["work"] == report["work"]
    assert calls == ["load", "case"]


def test_should_preserve_load_failure_progress_and_never_start_a_prediction_case(
    monkeypatch, tmp_path
):
    source, checkpoint = cli_paths(tmp_path)
    loads: list[None] = []
    event = {"event": "load_attempt", "observed_work": {"initializer_cpu_elements": 0}}

    def load(source, checkpoint, progress):
        loads.append(None)
        progress(deepcopy(event))
        raise RuntimeError("controlled fake loader failure")

    monkeypatch.setattr(smoke, "_load", load)
    monkeypatch.setattr(smoke, "run_case", forbidden)
    output = tmp_path / "load-failure.json"
    assert smoke.main(cli_args(source, checkpoint, output)) == 1
    saved = json.loads(output.read_text())
    assert saved["complete"] is False
    assert "controlled fake loader failure" in saved["error"]
    assert saved["model_load_attempts"] == 1 and saved["events"] == [event]
    assert loads == [None]
