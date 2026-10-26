"""Actual tiny CLM training/typed predictor to existing local workflow effects."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_ticket_workflow as study
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.adapters.clm_decision import CLMDecisionPredictor
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_frozen_encoder import text_recipe
from src.app.clm_workflow_training import ACTIVE_IDS, fit_catalog_workflow, expected_targets
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import FrozenEncoderIdentity
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone


class PublicTextTokens(dict):
    def get(self, text, default=None):
        return [1 + sum(text.encode("utf8")) % 29]


def execute(*, fault=None, reuse_action_features=False):
    import torch

    state, action, scale = parts(torch)
    model, tokenizer = backbone(torch), FakeTokenizer(PublicTextTokens(dummy=1))
    loaded = SimpleNamespace(
        state_head=state,
        action_head=action,
        logit_scale=scale,
        model=model,
        tokenizer=tokenizer,
        identity=FrozenEncoderIdentity(
            "literal", "workflow", "fixture", "last", text_recipe(2048), 2
        ),
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), deepcopy(scale)
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    observations = []
    real_init, real_snapshot = CLMCatalogLearner.__init__, CLMCatalogLearner.snapshot_state
    real_predict = CLMDecisionPredictor.__call__

    def created(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        holder["learner"] = self

    def snapshot(self):
        holder["source_pin"] = self._capture()
        return real_snapshot(self)

    def predicted(self, request):
        if fault == "head" and not holder.get("injected"):
            holder["injected"] = True

            def fail(*args):
                raise RuntimeError("controlled workflow head return lost")

            self._state_head.register_forward_hook(fail)
        return real_predict(self, request)

    def save():
        training = report.get("training", {})
        work = report.get("predictor_work", {}).get("unchanged", {})
        observations.append((training.get("committed_updates", 0), work.get("request_attempts", 0)))
        if fault == "source" and work.get("request_returns") == 1 and not holder.get("injected"):
            holder["injected"] = True
            with torch.no_grad():
                next(holder["learner"]._bundle[0].parameters()).add_(0.1)
        if fault == "prefix" and report.get("episodes") and not holder.get("injected"):
            holder["injected"] = True
            holder["episode"] = deepcopy(report["episodes"][0])
            report["episodes"][0]["episode"]["success"] = not report["episodes"][0]["episode"][
                "success"
            ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CLMCatalogLearner, "__init__", created)
        patch.setattr(CLMCatalogLearner, "snapshot_state", snapshot)
        patch.setattr(CLMDecisionPredictor, "__call__", predicted)
        options = {"reuse_action_features": True} if reuse_action_features else {}
        if fault:
            with pytest.raises(Exception):
                study.run_loaded(loaded, report, save=save, **options)
        else:
            study.run_loaded(loaded, report, save=save, **options)
    return dict(
        report=report,
        holder=holder,
        loaded=loaded,
        original=original,
        rng=rng,
        observations=observations,
    )


@pytest.fixture(scope="module")
def completed():
    return execute()


def test_should_train_only_then_complete_all_six_existing_workflow_episodes(completed):
    r = completed["report"]
    assert r["complete"] is True and r["pending_episode"] is None
    t = r["training"]
    assert t["complete"] and t["committed_updates"] == t["learner_work"]["commits"] == 8
    assert t["learner_work"]["training_rows_attempted"] == 96
    assert t["learner_work"]["catalog_logit_values_attempted"] == 480
    assert t["permissions"] == dict(training=True, replay=False, evaluation=False)
    assert [v["observed_at"] for v in t["receipts"]] == list(range(0, 16, 2))
    assert [v["applied_at"] for v in t["receipts"]] == list(range(1, 16, 2))
    assert r["training_data"]["encoded_sequences"] == 11
    assert len(r["episodes"]) == 6 and all(v["episode"]["complete"] for v in r["episodes"])
    assert all(
        v["episode"]["external_actions"] == v["episode"]["training_updates"] == 0
        for v in r["episodes"]
    )
    calls = sum(v["episode"]["work"]["prediction_calls_attempted"] for v in r["episodes"])
    assert calls == r["logical_prediction_attempts"] <= 36
    assert (
        r["encoder_work"]["forward_attempts"] == r["encoder_work"]["forward_returns"] == calls + 1
    )
    assert r["encoder_work"]["sequences_attempted"] == 11 + 6 * calls
    assert all(updates == 8 for updates, predictions in completed["observations"] if predictions)
    assert r["summary"] == study._summary(r["episodes"])


def test_should_preserve_source_loaded_heads_and_rng_during_frozen_inference(completed):
    import torch

    r, holder = completed["report"], completed["holder"]
    source = holder["learner"]
    assert _equal(vars(source._capture()), vars(holder["source_pin"]))
    assert source.work["snapshot_attempts"] == source.work["snapshot_returns"] == 1
    assert r["training"]["learner_work"]["snapshot_attempts"] == 0
    assert r["binding_work"]["binding_returns"] == 1
    assert r["source_guard_work"]["attempts"] == r["source_guard_work"]["returns"] > 1
    assert r["source_guard_work"]["unreturned_capture_work_unknown"] is False
    loaded = completed["loaded"]
    assert _equal(
        completed["original"],
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale),
    )
    assert torch.equal(torch.get_rng_state(), completed["rng"])
    assert all(parameter.grad is None for parameter in loaded.model.parameters())


def test_should_retain_paid_head_failure_without_more_episodes_or_training():
    r = execute(fault="head")["report"]
    assert r["complete"] is False and r["training"]["committed_updates"] == 8
    assert len(r["episodes"]) == 1 and r["episodes"][0]["episode"]["complete"] is False
    assert r["logical_prediction_attempts"] == 1
    assert r["predictor_work"]["unchanged"]["head_forward_attempts"] == 1
    assert r["predictor_work"]["unchanged"]["head_forward_returns"] == 0
    assert r["predictor_work"]["unchanged"]["unreturned_work_unknown"] is True


def test_should_refuse_source_corruption_before_more_predictions():
    r = execute(fault="source")["report"]
    assert r["complete"] is False and "source state" in r["error"]
    assert r["logical_prediction_attempts"] == 1 and r["training"]["committed_updates"] == 8
    assert r["preserved_training_prefix"]["committed_updates"] == 8


def test_should_preserve_trusted_prior_episode_when_callback_changes_prefix():
    case = execute(fault="prefix")
    r = case["report"]
    assert r["complete"] is False and "episode prefix" in r["error"]
    assert study._json(r["preserved_episode_prefix"]) == study._json([case["holder"]["episode"]])
    assert len(r["episodes"]) == 1


@pytest.mark.parametrize("fault", ["role", "target", "candidate_order"])
def test_should_refuse_changed_train_cases_before_encoder(fault, monkeypatch):
    cases = study.training_cases()
    if fault == "role":
        item = replace(cases[0], role="outer_selection")
    elif fault == "target":
        item = replace(cases[0], target_id="commit")
    else:
        item = replace(
            cases[0],
            request=replace(
                cases[0].request,
                allowed_candidates=tuple(reversed(cases[0].request.allowed_candidates)),
            ),
        )
    monkeypatch.setattr(study, "training_cases", lambda: (item, *cases[1:]))

    def forbidden(*args):
        raise AssertionError("unadmitted TRAIN reached encoder")

    with pytest.raises(ValueError, match="TRAIN"):
        study.training_inputs(SimpleNamespace(embed_texts=forbidden))


def test_should_refuse_spent_training_owner_before_more_updates(completed):
    import torch

    source = completed["holder"]["learner"]
    batch = FrozenCatalogTrainingBatch(
        source.identity, torch.zeros(12, 2), torch.zeros(5, 2), ACTIVE_IDS
    )
    before = source.work
    with pytest.raises(ValueError, match="fresh exact"):
        fit_catalog_workflow(source, batch, CLMCatalogTargets(ACTIVE_IDS, expected_targets()), {})
    assert source.work == before


def test_should_refuse_occupied_cli_output_before_loading(tmp_path, monkeypatch):
    import sys

    output = tmp_path / "original.json"
    output.write_text("original")
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "--model-dir", "unused", "--head", "unused", "--output", str(output)],
    )
    monkeypatch.setattr(
        study, "run_case", lambda *a, **k: pytest.fail("occupied output reached loader")
    )
    with pytest.raises(FileExistsError):
        study.main()
    assert output.read_text() == "original"


def test_should_dispatch_cli_to_fixed_prepared_asset_case(tmp_path, monkeypatch):
    import sys

    output = tmp_path / "new.json"
    calls = []

    def execute(model, head, report, *, save):
        calls.append((model, head))
        report["complete"] = True
        save()

    monkeypatch.setattr(
        sys, "argv", ["runner", "--model-dir", "model", "--head", "head", "--output", str(output)]
    )
    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    assert calls == [(Path("model"), Path("head"))]
