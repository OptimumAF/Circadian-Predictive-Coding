"""Phase fit is TRAIN-only readonly work, never another optimization decision."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
import sys

import pytest

pytest.importorskip("torch")
import torch

import scripts.run_clm_ticket_workflow as study
from src.adapters.clm_catalog_learner import CLMCatalogLearner
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_frozen_encoder import text_recipe
from src.app.clm_workflow_training import ACTIVE_IDS, expected_targets, summarize_training_fit
from src.core.clm_head import FrozenEncoderIdentity
from src.core.clm_head import CLMHeadPolicy
from src.core.clm_catalog import FrozenCatalogTrainingBatch
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_ticket_workflow import PublicTextTokens


def run(*, enabled=False, fault=None):
    state, action, scale = parts(torch)
    loaded = SimpleNamespace(
        state_head=state,
        action_head=action,
        logit_scale=scale,
        model=backbone(torch),
        tokenizer=FakeTokenizer(PublicTextTokens(dummy=1)),
        identity=FrozenEncoderIdentity(
            "literal", "workflow", "fixture", "last", text_recipe(2048), 2
        ),
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), deepcopy(scale)
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    holder: dict[str, Any] = {}
    real_init, real_read = CLMCatalogLearner.__init__, CLMCatalogLearner.predict_choices
    real_inputs = study.training_inputs

    def created(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        holder["learner"] = self

    def inputs(encoder):
        result = real_inputs(encoder)
        holder["batch"] = result[0]
        return result

    def read(self, batch):
        holder.setdefault("roles", []).append(batch.role)
        assert batch.states is holder["batch"].states and batch.actions is holder["batch"].actions
        result = real_read(self, batch)
        if fault == "native_lost" or (fault == "after_lost" and self.completed_updates == 8):
            raise RuntimeError("controlled readout return lost")
        if fault == "nan":
            result[0, 0] = float("nan")
        elif fault == "width":
            result = result[:, :4]
        elif fault == "state":
            with torch.no_grad():
                next(self._bundle[0].parameters()).add_(0.1)
        elif fault == "feature":
            # Deliberately bypass version tracking: equality must still catch it.
            batch.states.data[0, 0] += 0.1
        elif fault == "counter":
            self._work["prediction_rows_attempted"] += 1
        elif fault == "generation":
            self._generation += 1
        elif fault == "owners":
            self._work = dict(self._work)
        return result

    def save():
        if holder.get("tampered"):
            return
        fit = report.get("train_fit", {}).get("readouts", [])
        if fault == "replace_evidence" and fit and fit[0]["complete"]:
            fit[0] = deepcopy(fit[0])
            holder["tampered"] = True
        if fault == "training_evidence" and len(fit) == 2:
            report["training"]["committed_updates"] = 7
            holder["tampered"] = True

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(CLMCatalogLearner, "__init__", created)
        patch.setattr(CLMCatalogLearner, "predict_choices", read)
        patch.setattr(study, "training_inputs", inputs)
        options = {"record_train_fit": True} if enabled else {}
        if fault:
            with pytest.raises(Exception):
                study.run_loaded(loaded, report, save=save, reuse_action_features=True, **options)
        else:
            study.run_loaded(loaded, report, save=save, reuse_action_features=True, **options)
    return report, holder, loaded, original, rng


@pytest.fixture(scope="module")
def pair():
    return run(), run(enabled=True)


def test_should_observe_two_paid_train_points_without_changing_training_or_episodes(pair):
    off, on = pair
    before, after = off[0], on[0]
    assert before["complete"] and after["complete"] and "train_fit" not in before
    assert before["episodes"] == after["episodes"] and before["summary"] == after["summary"]
    assert before["encoder_work"] == after["encoder_work"]
    assert before["predictor_work"] == after["predictor_work"]
    assert before["training"]["receipts"] == after["training"]["receipts"]
    assert after["source_work"]["train_attempts"] == after["source_work"]["commits"] == 8
    assert after["source_work"]["training_rows_attempted"] == 96
    assert (
        after["source_work"]["prediction_attempts"]
        == after["source_work"]["prediction_returns"]
        == 2
    )
    assert after["source_work"]["prediction_rows_attempted"] == 24
    assert after["source_work"]["choice_candidate_rows_attempted"] == 10
    assert after["source_work"]["snapshot_returns"] == 1
    assert _equal(vars(off[1]["learner"]._capture()), vars(on[1]["learner"]._capture()))
    assert on[1]["roles"] == ["train", "train"]
    for case in pair:
        assert _equal(
            case[3],
            (
                case[2].state_head.state_dict(),
                case[2].action_head.state_dict(),
                case[2].logit_scale,
            ),
        )
        assert torch.equal(case[4], torch.get_rng_state())
        assert all(p.grad is None for p in case[2].model.parameters())
    fit = after["train_fit"]
    assert fit["role"] == "train" and fit["candidate_ids"] == list(ACTIVE_IDS)
    assert [r["point"] for r in fit["readouts"]] == ["before", "after"]
    assert [r["completed_updates"] for r in fit["readouts"]] == [0, 8]
    for result in fit["readouts"]:
        assert result["complete"] and result["returned"] and result["validated"]
        assert result["source_preserved"] is True and result["elapsed_seconds"] >= 0
        assert result["rows"] == summarize_training_fit(result["probabilities"])["rows"]
        assert list(result["by_phase"]) == ["sealed", "ready", "staged"]
        assert all(r["rows"] == 4 for r in result["by_phase"].values())
    assert "preserved_train_fit" not in after


def test_should_define_margins_against_best_other_candidate_and_use_id_tie_break():
    values = [[0.2] * 5 for _ in range(12)]
    values[0] = [0.1, 0.1, 0.1, 0.6, 0.1]
    values[1] = [0.1, 0.1, 0.5, 0.2, 0.1]
    result = summarize_training_fit(values)
    assert result["rows"][0]["correct"] and result["rows"][0]["probability_margin"] == 0.5
    assert result["rows"][1]["target_id"] == "inspect"
    assert result["rows"][1]["probability_margin"] == 0.3
    assert result["rows"][2]["selected_id"] == "commit"
    assert result["rows"][2]["probability_margin"] == 0
    assert tuple(ACTIVE_IDS.index(row["target_id"]) for row in result["rows"]) == expected_targets()


@pytest.mark.parametrize(
    "fault",
    ["count", "width", "nan", "inf", "negative", "large", "integer", "bool", "sum", "tuple"],
)
def test_should_refuse_malformed_probability_rows(fault):
    values: list[Any] = [[0.2] * 5 for _ in range(12)]
    if fault == "count":
        values.pop()
    elif fault == "width":
        values[0].pop()
    elif fault == "tuple":
        values[0] = tuple(values[0])
    else:
        values[0][0] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative": -0.1,
            "large": 1.1,
            "integer": 0,
            "bool": True,
            "sum": 0.3,
        }[fault]
    with pytest.raises(ValueError):
        summarize_training_fit(values)


@pytest.mark.parametrize(
    "fault", ["native_lost", "nan", "width", "state", "feature", "counter", "generation", "owners"]
)
def test_should_stop_and_preserve_paid_bad_read_before_any_training_or_effect(fault):
    report, holder, *_ = run(enabled=True, fault=fault)
    assert not report["complete"] and report["episodes"] == []
    assert holder["learner"].completed_updates == 0
    assert report["source_work"]["prediction_attempts"] == 1
    assert report["source_work"]["prediction_returns"] == 1
    record = report["preserved_train_fit"]["readouts"][0]
    assert record["attempted"] and not record["complete"] and not record["validated"]
    assert record["returned"] is (fault != "native_lost")
    assert record["work_after"]["prediction_attempts"] == 1


def test_should_refuse_equal_replaced_success_record_before_fitting():
    report, holder, *_ = run(enabled=True, fault="replace_evidence")
    assert not report["complete"] and "fit evidence" in report["error"]
    assert holder["learner"].completed_updates == 0 and report["episodes"] == []
    assert report["preserved_train_fit"]["readouts"][0]["complete"]


def test_should_preserve_completed_training_when_after_read_return_is_lost():
    report, holder, *_ = run(enabled=True, fault="after_lost")
    assert not report["complete"] and report["episodes"] == []
    assert holder["learner"].completed_updates == 8
    assert report["preserved_training_prefix"]["committed_updates"] == 8
    records = report["preserved_train_fit"]["readouts"]
    assert len(records) == 2 and records[0]["complete"] and not records[1]["complete"]
    assert records[1]["attempted"] and not records[1]["returned"]
    assert records[1]["work_after"]["prediction_attempts"] == 2


def test_should_refuse_changed_completed_training_before_after_read():
    report, holder, *_ = run(enabled=True, fault="training_evidence")
    assert not report["complete"] and "TRAIN evidence" in report["error"]
    assert holder["learner"].completed_updates == 8
    assert report["preserved_training_prefix"]["committed_updates"] == 8
    assert report["source_work"]["prediction_attempts"] == 1 and report["episodes"] == []


@pytest.mark.parametrize("bad", [None, 0, 1, "true"])
def test_should_require_explicit_boolean_before_model_access(bad):
    with pytest.raises(ValueError, match="explicit boolean"):
        study.run_loaded(SimpleNamespace(), {}, record_train_fit=bad)
    with pytest.raises(ValueError, match="explicit boolean"):
        study.run_case("unused", "unused", {}, record_train_fit=bad)


def test_should_forward_explicit_cli_opt_in_without_loading(tmp_path, monkeypatch):
    output = tmp_path / "out.json"
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model-dir",
            "unused",
            "--head",
            "unused",
            "--output",
            str(output),
            "--record-train-fit",
        ],
    )

    def stub(*args, save, **kwargs):
        calls.append(kwargs)
        save()

    monkeypatch.setattr(study, "run_case", stub)
    study.main()
    assert calls == [dict(record_train_fit=True)]


@pytest.mark.parametrize("enabled", [False, True])
def test_should_reserve_diagnostic_input_key_only_when_enabled(enabled):
    report = {"train_fit": {"custom": "caller metadata"}}
    error = ValueError if enabled else AttributeError
    with pytest.raises(error):
        study.run_loaded(SimpleNamespace(), report, record_train_fit=enabled)
    assert report == {"train_fit": {"custom": "caller metadata"}}


def test_should_measure_readout_admission_captures_and_summary_without_training(monkeypatch):
    identity = FrozenEncoderIdentity("literal", "fit", "fixture", "last", "fit-test", 2)
    learner = CLMCatalogLearner(
        *parts(torch),
        identity=identity,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=8, max_rows=12, max_restores=0),
    )
    batch = FrozenCatalogTrainingBatch(identity, torch.ones(12, 2), torch.ones(5, 2), ACTIVE_IDS)
    outcome: dict[str, Any] = {}
    events = []
    original = learner._capture

    def capture():
        events.append("capture")
        return original()

    def clock():
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(learner, "_capture", capture)
    monkeypatch.setattr(study, "monotonic", clock)
    study._training_fit_readout(learner, batch, outcome)
    assert events[0] == events[-1] == "clock" and events.count("capture") >= 2
    assert outcome["elapsed_seconds"] == len(events) - 1
    assert outcome["validated"] and outcome["source_preserved"]
    assert learner.completed_updates == learner.work["train_attempts"] == 0
