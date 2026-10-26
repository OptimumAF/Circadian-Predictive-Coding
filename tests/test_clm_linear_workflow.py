"""Literal fit-to-typed-effects path; no released models or trained assets."""

from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import scripts.run_clm_ticket_workflow as study
from src.adapters.clm_frozen_encoder import FrozenCLMEncoder, text_recipe
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_linear_decision import CLMLinearDecisionPredictor
from src.adapters.clm_linear_fit_control import fit_clm_linear_control
from src.core.clm_action_features import FrozenActionFeatures
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import FrozenEncoderIdentity
from src.core.ticket_workflow import CATALOG
from src.core.typed_decision import DecisionRequest, ProbabilityDecisionResult

literal_cli_capture: dict[str, Any] | None = None


class WorkflowLiteralTokenizer:
    pad_token_id = 0
    eos_token_id = 7

    def __call__(self, text, *, add_special_tokens):
        words = text.split()
        if words[0] == "sealed":
            token = 1
        elif words[0] == "staged":
            token = 6
        elif words[0] == "ready":
            token = 2 + ("invoice", "legacy", "parcel", "password").index(words[1])
        else:
            token = 7
        return {"input_ids": [token]}


class WorkflowLiteralBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=6)
        self.fault = None

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        values = torch.nn.functional.one_hot((input_ids - 1) % 6, num_classes=6).float()
        if self.fault is not None:
            self.fault()
        return SimpleNamespace(last_hidden_state=values)


def loaded_case():
    def head():
        value = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(value)
        value.in_features = value.out_features = 6
        value.register_parameter("weight", nn.Parameter(torch.eye(6), requires_grad=False))
        value.register_parameter("bias", None)
        return value.eval()

    return SimpleNamespace(
        model=WorkflowLiteralBackbone().eval(),
        tokenizer=WorkflowLiteralTokenizer(),
        state_head=head(),
        action_head=head(),
        logit_scale=torch.tensor(1.2),
        identity=FrozenEncoderIdentity(
            "literal", "workflow", "fixture", "last", text_recipe(2048), 6
        ),
    )


def prepared(*, probe=False, maximum=18):
    loaded = loaded_case()
    encoder = FrozenCLMEncoder(
        loaded.model,
        loaded.tokenizer,
        identity=loaded.identity,
        max_sequences=16,
        max_forward_calls=19,
    )
    batch, targets, metadata = study.training_inputs(encoder, public_observation_train=True)
    if probe:
        batch = FrozenProbeCatalogBatch(
            batch.identity, batch.states, batch.actions, batch.candidate_ids
        )
    report: dict[str, Any] = {}
    control = fit_clm_linear_control(batch, targets, report)
    actions = FrozenActionFeatures(
        batch.identity,
        tuple(c for c in CATALOG if c.candidate_id in batch.candidate_ids),
        batch.actions,
    )
    provider = (
        None
        if probe
        else CLMLinearDecisionPredictor(
            encoder, control, actions, instructions=study.INSTRUCTIONS, max_requests=maximum
        )
    )
    return loaded, encoder, control, actions, provider


def request(actions):
    return DecisionRequest(
        "literal-request", "ready invoice feedback applied please", actions.candidates
    )


@pytest.fixture(scope="module")
def completed():
    loaded = loaded_case()
    before = deepcopy(
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    study.run_loaded(loaded, report, linear_fit_control=True, public_observation_train=True)
    return loaded, before, rng, report


def test_should_fit_then_complete_original_ordinary_recovery_and_permission_denial(completed):
    report = completed[3]
    assert report["complete"] is True and report["pending_episode"] is None
    assert report["linear_fit"]["attempts"] == report["linear_fit"]["returns"] == 1
    assert report["linear_fit"]["rows_attempted"] == 12
    assert "training" not in report  # No invented eight SGD updates or receipts.
    assert report["episode_controls"]["observed"] == ["linear_fit"]
    episodes = [r["episode"] for r in report["episodes"]]
    assert len(episodes) == 3 and all(e["complete"] for e in episodes)
    assert [e["success"] for e in episodes] == [True, True, False]
    assert episodes[1]["error_recovery_demonstrated"] is True
    assert episodes[2]["stopped_reason"] == "permission_denied"
    assert all(e["external_actions"] == e["training_updates"] == 0 for e in episodes)
    assert all(row["correct"] for row in report["train_fit"]["rows"])
    assert report["summary"]["linear_fit"] == dict(
        episodes=3, successes=2, recoveries=1, permission_denials=1
    )


def test_should_preserve_source_and_charge_query_only_encodings_and_linear_reads(completed):
    loaded, before, rng, report = completed
    assert _equal(
        before,
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale),
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None and not p.requires_grad for p in loaded.model.parameters())
    work = report["predictor_work"]
    assert work["request_attempts"] == work["request_returns"] == work["linear_rows"] == 11
    assert work["head_forward_attempts"] == work["optimizer_steps"] == 0
    assert (
        report["encoder_work"]["forward_attempts"]
        == report["encoder_work"]["forward_returns"]
        == 12
    )
    assert report["encoder_work"]["sequences_attempted"] == 22


def test_should_return_valid_probabilities_margin_confidence_and_bounded_owner():
    _, encoder, control, actions, provider = prepared(maximum=1)
    expected = deepcopy(asdict(control))
    before = encoder.work
    value = provider(request(actions))
    assert type(value) is ProbabilityDecisionResult and value.reported_input_tokens == 1
    assert value.reported_output_tokens is None
    probabilities = [p for _, p in value.probabilities]
    assert sum(probabilities) == pytest.approx(1.0)
    peak = max(probabilities)
    assert value.reported_confidence == pytest.approx(peak - (sum(probabilities) - peak) / 4)
    assert encoder.work["forward_attempts"] == before["forward_attempts"] + 1
    assert asdict(control) == expected
    last = provider.last_readout
    last["linear_prediction"]["complete"] = False
    assert provider.last_readout["linear_prediction"]["complete"] is True
    with pytest.raises(ValueError, match="allowance"):
        provider(request(actions))
    assert provider.work["request_attempts"] == 1


@pytest.mark.parametrize(
    "fault", ["description", "order", "model", "actions", "counter", "encoder_counter", "limit"]
)
def test_should_refuse_changed_binding_before_more_query_encoding(fault):
    _, encoder, control, actions, provider = prepared()
    supplied = request(actions)
    if fault == "description":
        supplied = replace(
            supplied,
            allowed_candidates=(
                replace(actions.candidates[0], description="changed"),
                *actions.candidates[1:],
            ),
        )
    elif fault == "order":
        supplied = replace(supplied, allowed_candidates=tuple(reversed(actions.candidates)))
    elif fault == "model":
        object.__setattr__(control.model, "squared_error", 1.0)
    elif fault == "actions":
        actions.features.add_(0.1)
    elif fault == "counter":
        provider._work["request_attempts"] += 1
    elif fault == "encoder_counter":
        encoder._work["forward_attempts"] += 1
    elif fault == "limit":
        provider._max_requests += 1
    before = encoder.work["embedding_attempts"]
    with pytest.raises(ValueError):
        provider(supplied)
    assert encoder.work["embedding_attempts"] == before


def test_should_refuse_deploying_a_probe_fit():
    _, encoder, control, actions, _ = prepared(probe=True)
    before = encoder.work
    with pytest.raises(ValueError, match="probe"):
        CLMLinearDecisionPredictor(encoder, control, actions, instructions=study.INSTRUCTIONS)
    assert encoder.work == before


@pytest.mark.parametrize("fault", ["lost_return", "encoder_counter", "request"])
def test_should_keep_uncertain_work_and_stop_provider_without_retry(monkeypatch, fault):
    loaded, encoder, _, actions, provider = prepared()
    supplied = request(actions)
    if fault == "lost_return":
        real = encoder.embed_texts

        def lost(*args):
            real(*args)
            raise RuntimeError("controlled lost encoding")

        monkeypatch.setattr(encoder, "embed_texts", lost)
    elif fault == "encoder_counter":
        loaded.model.fault = lambda: encoder._work.__setitem__(
            "forward_attempts", encoder._work["forward_attempts"] + 1
        )
    else:
        loaded.model.fault = lambda: object.__setattr__(supplied, "request_id", "changed")
    with pytest.raises((ValueError, RuntimeError)):
        provider(supplied)
    assert (
        provider.work["request_attempts"] == 1 and provider.work["unreturned_work_unknown"] is True
    )
    before = encoder.work["embedding_attempts"]
    with pytest.raises(ValueError, match="stopped"):
        provider(request(actions))
    assert encoder.work["embedding_attempts"] == before


@pytest.mark.parametrize(
    "fault",
    [
        "fit",
        "fit_owner",
        "episode",
        "episode_owner",
        "source",
        "features",
        "reported_work",
        "provider_work",
        "encoder_work",
        "request_clock",
        "save",
        "final_save",
        "final_summary",
        "final_pending",
    ],
)
def test_should_preserve_trusted_paid_prefix_and_refuse_corrupted_callbacks(monkeypatch, fault):
    loaded = loaded_case()
    report: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    real_inputs, real_init = study.training_inputs, CLMLinearDecisionPredictor.__init__

    def inputs(*args, **kwargs):
        result = real_inputs(*args, **kwargs)
        holder["batch"] = result[0]
        return result

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        holder["provider"] = self

    monkeypatch.setattr(study, "training_inputs", inputs)
    monkeypatch.setattr(CLMLinearDecisionPredictor, "__init__", init)
    changed = False

    def save():
        nonlocal changed
        if changed or not report.get("linear_fit", {}).get("complete"):
            return
        episode_fault = fault in ("episode", "episode_owner")
        provider_fault = fault in (
            "provider_work",
            "encoder_work",
            "reported_work",
            "request_clock",
        )
        if episode_fault and not report["episodes"]:
            return
        if provider_fault and "provider" not in holder:
            return
        if fault in ("final_save", "final_summary", "final_pending") and not report.get("complete"):
            return
        changed = True
        if fault in ("fit", "final_save"):
            report["linear_fit"]["rows_attempted"] = 99
        elif fault == "final_summary":
            report["summary"]["linear_fit"]["successes"] = 99
        elif fault == "final_pending":
            report["pending_episode"] = dict(effects_unknown=True)
        elif fault == "fit_owner":
            report["linear_fit"] = deepcopy(report["linear_fit"])
        elif fault == "episode":
            report["episodes"][0]["episode"]["success"] = False
        elif fault == "episode_owner":
            report["episodes"][0] = deepcopy(report["episodes"][0])
        elif fault == "source":
            next(loaded.state_head.parameters()).add_(0.1)
        elif fault == "features":
            holder["batch"].states.add_(0.1)
        elif fault == "provider_work":
            holder["provider"]._work["request_attempts"] += 1
        elif fault == "encoder_work":
            holder["provider"]._encoder._work["forward_attempts"] += 1
        elif fault == "reported_work":
            report["predictor_work"]["request_attempts"] += 1
        elif fault == "request_clock":
            report["logical_prediction_attempts"] += 1
        elif fault == "save":
            raise OSError("controlled save failure")

    with pytest.raises((ValueError, OSError)):
        study.run_loaded(
            loaded, report, linear_fit_control=True, public_observation_train=True, save=save
        )
    assert changed and report["complete"] is False
    assert report["preserved_training_prefix"]["rows_attempted"] == 12
    if fault in ("episode", "episode_owner", "final_save", "final_summary", "final_pending"):
        assert report["preserved_episode_prefix"][0]["episode"]["success"] is True
    if fault in ("final_summary", "final_pending"):
        assert report["preserved_summary"]["linear_fit"]["successes"] == 2
        assert report["preserved_pending_episode"] is None
    before = report["encoder_work"]["embedding_attempts"]
    with pytest.raises(ValueError):
        study.run_loaded(loaded, report, linear_fit_control=True)
    assert report["encoder_work"]["embedding_attempts"] == before


@pytest.mark.parametrize("option", ["reuse_action_features", "record_train_fit", "learned_only"])
def test_should_refuse_incompatible_options_before_loading(option, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "load_local_clm", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="cannot combine"):
        study.run_case(None, None, {}, linear_fit_control=True, **{option: True})
    output = tmp_path / "refused.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "workflow",
            "--model-dir",
            "unused",
            "--head",
            "unused",
            "--output",
            str(output),
            "--linear-fit-control",
            "--" + option.replace("_", "-"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not output.exists()


def test_should_route_public_cli_explicit_flag_without_overwriting_output(tmp_path, monkeypatch):
    global literal_cli_capture
    output = tmp_path / "linear.json"
    calls = []
    threads = []
    real_threads = torch.set_num_threads

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        kwargs["before_step"]()
        kwargs["progress"](dict(event="literal_loader_fixture"))
        return loaded_case()

    def one_thread(requested):
        threads.append(requested)
        real_threads(1)

    monkeypatch.setattr(study, "load_local_clm", load)
    monkeypatch.setattr(torch, "set_num_threads", one_thread)
    monkeypatch.setattr(
        "sys.argv",
        [
            "workflow",
            "--model-dir",
            "model",
            "--head",
            "head",
            "--output",
            str(output),
            "--linear-fit-control",
            "--public-observation-train",
        ],
    )
    study.main()
    assert len(calls) == 1 and threads == [2]
    saved = json.loads(output.read_text())
    assert saved["complete"] is True and saved["linear_fit"]["rows_attempted"] == 12
    assert saved["summary"]["linear_fit"] == dict(
        episodes=3, successes=2, recoveries=1, permission_denials=1
    )
    assert saved["training_data"]["observation_recipe"] == "public_workflow_request_v1"
    literal_cli_capture = dict(
        definition="literal_cli_fixture_not_native_v1",
        requested_threads=threads,
        actual_fixture_threads=1,
        report=saved,
    )
    with pytest.raises(FileExistsError):
        study.main()
    assert len(calls) == 1


def test_should_preserve_saved_before_default_report_except_observed_timing():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-linear-workflow-20261025/workflow_before.py"
    )
    spec = importlib.util.spec_from_file_location("before_linear_workflow", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def clean(value):
        if type(value) is dict:
            return {k: clean(v) for k, v in value.items() if k != "elapsed_seconds"}
        if type(value) is list:
            return [clean(v) for v in value]
        return value

    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    module.run_loaded(loaded_case(), first)
    study.run_loaded(loaded_case(), second)
    assert clean(first) == clean(second)


@pytest.mark.parametrize("option", ["reuse_action_features", "record_train_fit", "learned_only"])
def test_should_refuse_false_like_nonboolean_options_before_source_access(option, monkeypatch):
    monkeypatch.setattr(study, "load_local_clm", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="cannot combine"):
        study.run_case(None, None, {}, linear_fit_control=True, **{option: 0})
    with pytest.raises(ValueError, match="cannot combine"):
        study.run_loaded(None, {}, linear_fit_control=True, **{option: 0})


@pytest.mark.parametrize("target", ["head", "scale"])
def test_should_refuse_changed_original_source_gradients_before_encoding(target):
    loaded = loaded_case()
    report: dict[str, Any] = {}

    def corrupt():
        parameter = next(loaded.state_head.parameters()) if target == "head" else loaded.logit_scale
        parameter.grad = torch.zeros_like(parameter)

    with pytest.raises(ValueError, match="gradients"):
        study.run_loaded(loaded, report, linear_fit_control=True, before_step=corrupt)
    assert report["complete"] is False
    assert report["encoder_work"]["embedding_attempts"] == 0
    assert "linear_fit" not in report
