"""Small source32/F6 composition checks; no pretrained model or scientific claim."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from src.adapters.clm_frozen_encoder import text_recipe
from src.core.clm_head import FrozenEncoderIdentity
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_ticket_workflow import PublicTextTokens


def module():
    path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-workflow-state-only-native-20261025/fresh_case.py"
    )
    spec = importlib.util.spec_from_file_location("fresh_workflow_case", path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def loaded():
    state, action, scale = parts(torch)
    state.eval().requires_grad_(False)
    action.eval().requires_grad_(False)
    return SimpleNamespace(
        state_head=state,
        action_head=action,
        logit_scale=scale.detach(),
        model=backbone(torch),
        tokenizer=FakeTokenizer(PublicTextTokens(dummy=1)),
        identity=FrozenEncoderIdentity(
            "literal", "fresh-workflow", "fixture", "last", text_recipe(2048), 2
        ),
    )


def test_should_finish_source_and_isolated_probe_before_using_cached_metrics():
    report: dict[str, Any] = {}
    rng = torch.get_rng_state().clone()
    module().run_loaded(loaded(), report)
    assert report["complete"] and report["training"]["committed_updates"] == 32
    assert (
        report["source_work"]["commits"] == 32
        and not report["source_work"]["unreturned_work_unknown"]
    )
    assert report["loaded_heads_and_caller_rng_preserved"]
    assert torch.equal(torch.get_rng_state(), rng)
    assert report["encoder_work"]["forward_returns"] == 2
    assert report["encoder_work"]["sequences_returned"] == 15
    probe = report["probe"]
    assert probe["complete"] and probe["stream"]["completed_updates"] == 6
    assert probe["parameter_update_scope"]["name"] == "state_only"
    assert probe["parameter_update_scope"]["fixed_parameters_unchanged"] is True
    assert len(probe["stream"]["receipts"]) == 6
    assert probe["probe_work"]["training_rows_attempted"] == 12
    assert probe["probe_work"]["catalog_candidate_rows_attempted"] == 18
    assert probe["probe_work"]["prediction_rows_attempted"] == 32
    assert probe["probe_work"]["choice_candidate_rows_attempted"] == 22
    for group, count in (("probe_train", 2), ("probe_test", 2), ("old_task", 12)):
        scores = report["probe_scores"]["groups"][group]
        assert len(scores["before"]["cases"]) == len(scores["after"]["cases"]) == count
    assert len(report["probe_scores"]["groups"]["old_task"]["after"]["training_fit"]["rows"]) == 12
    assert "episodes" not in report and "binding_work" not in report
    json.dumps(report, allow_nan=False)


def test_should_stop_on_incomplete_source_before_more_encoding_or_probe_work(monkeypatch):
    study = module()
    report: dict[str, Any] = {}

    def incomplete(*args, **kwargs):
        report["training"] = dict(complete=False, committed_updates=0)

    monkeypatch.setattr(study, "fit_catalog_workflow", incomplete)
    with pytest.raises(ValueError, match="preceding source"):
        study.run_loaded(loaded(), report)
    assert report["complete"] is False and "probe" not in report
    assert report["source_work"]["commits"] == 0
    assert report["encoder_work"]["forward_returns"] == 1
    assert report["encoder_work"]["sequences_returned"] == 11
    assert report["loaded_heads_and_caller_rng_preserved"]


def test_should_refuse_cached_scoring_before_the_probe_is_complete():
    with pytest.raises(ValueError, match="before cached scoring"):
        module().cached_scores(dict(complete=False), {})
