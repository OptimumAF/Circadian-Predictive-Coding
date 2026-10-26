"""Literal public context oracle: software behavior, no pretrained claim."""

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
from torch import nn
import scripts.run_clm_ticket_workflow as study
import scripts.clm_linear_context_workflow as context
import src.adapters.clm_linear_catalog_learner as linear
from src.adapters.clm_head_learner import _equal
from src.core.clm_head import FrozenEncoderIdentity
from src.adapters.clm_frozen_encoder import text_recipe
from src.core.clm_linear_catalog import linear_catalog_work
from src.core.ticket_workflow import ACTION_IDS

literal_capture: dict[str, Any] = {}


class Tokenizer:
    pad_token_id, eos_token_id = 0, 13

    def __call__(self, text, *, add_special_tokens):
        words = text.split()
        phase = words[0]
        token = 0
        if phase == "ready":
            token = 1 + ("invoice", "legacy", "parcel", "password").index(words[1])
        elif phase == "staged":
            token = 5
        if phase in ("sealed", "ready", "staged"):
            token += 6 * int("policy_v2" in words)
        # Public state/context only; private ticket IDs and teacher targets
        # never enter this representation. Actions are checked metadata.
        return {"input_ids": [token + 1]}


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.config = SimpleNamespace(hidden_size=12)

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        return SimpleNamespace(
            last_hidden_state=torch.nn.functional.one_hot(input_ids - 1, num_classes=12).float()
        )


def loaded_case():
    def head():
        value = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(value)
        value.in_features = value.out_features = 12
        value.register_parameter("weight", nn.Parameter(torch.eye(12), requires_grad=False))
        value.register_parameter("bias", None)
        return value.eval()

    return SimpleNamespace(
        model=Backbone().eval(),
        tokenizer=Tokenizer(),
        state_head=head(),
        action_head=head(),
        logit_scale=torch.tensor(1.2),
        identity=FrozenEncoderIdentity(
            "literal", "context", "fixture", "last", text_recipe(2048), 12
        ),
    )


def run(loaded, report, **kwargs):
    return study.run_loaded(
        loaded, report, linear_context_updates=True, public_observation_train=True, **kwargs
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    output = tmp_path_factory.mktemp("context") / "literal.json"
    loaded = loaded_case()
    snapshot = deepcopy(
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    rng = torch.get_rng_state().clone()
    calls = []
    with pytest.MonkeyPatch.context() as patch:

        def load(*args, **kwargs):
            calls.append(args)
            kwargs["before_step"]()
            kwargs["progress"](dict(event="literal_loader_fixture"))
            return loaded

        patch.setattr(study, "load_local_clm", load)
        patch.setattr(
            "sys.argv",
            [
                "workflow",
                "--model-dir",
                "unused-model",
                "--head",
                "unused-head",
                "--output",
                str(output),
                "--linear-context-updates",
                "--public-observation-train",
            ],
        )
        study.main()
        with pytest.raises(FileExistsError):
            study.main()
    report = json.loads(output.read_text())
    assert len(calls) == 1 and torch.get_num_threads() == 1
    literal_capture.update(definition="literal_context_cli_not_native_v1", report=report)
    return loaded, snapshot, rng, report


def test_should_execute_actual_cli_and_original_three_exposures(completed):
    report = completed[3]
    assert report["complete"] is True and len(report["training"]) == 3
    assert report["pending_update"] is report["pending_episode"] is None
    assert report["learner_work"] == linear_catalog_work(3, 36)
    assert report["exposure"] == dict(
        original_teacher_rows=24,
        teacher_dispatch_transitions=24,
        repeated_return_rows=12,
        total_train_rows=36,
        preparation_sequences=18,
        replay_consent_from_birth=True,
        return_is_novel_data=False,
    )
    for i, row in enumerate(report["training"]):
        receipt = row["receipt"]
        assert receipt["observed_at"] == 2 * i
        assert receipt["arrived_at"] == receipt["applied_at"] == 2 * i + 1
        assert receipt["diagnostic"]["definition"] == context.OBJECTIVE
        assert row["repeated_rows"] == (12 if i == 2 else 0)
    assert report["budget"] == dict(updates_completed=3, max_training_updates=3, logical_tick=5)


def test_should_measure_retention_before_return_and_separate_recovery_claims(completed):
    report = completed[3]
    rows = report["episodes"]
    assert len(rows) == 14 and all(r["episode"]["complete"] for r in rows)
    # Untrained v2 is an observed negative, not a requirement to generalize
    # before exposure. Keep its two complete max-step outcomes in the report.
    assert [r["episode"]["success"] for r in rows[:4]] == [True, True, False, False]
    assert all(r["episode"]["stopped_reason"] == "max_steps" for r in rows[2:4])
    assert all(r["episode"]["success"] for r in rows[4:12])
    assert [r["scenario"] for r in rows[-2:]] == ["stale_once", "revoke_commit"]
    assert rows[-2]["episode"]["error_recovery_demonstrated"] is True
    assert rows[-1]["episode"]["stopped_reason"] == "permission_denied"
    assert report["summary"]["old_context_after_B"] == [True, True]
    assert report["summary"]["learning_recovery_demonstrated"] is False
    assert (
        report["summary"]["stale_episode_recoveries"]
        == report["summary"]["permission_denials"]
        == 1
    )
    assert all(
        r["role"] == "outer_selection" and r["episode"]["training_updates"] == 0 for r in rows
    )
    assert [r["policy_version"] for r in rows[:4]] == ["v1", "v1", "v2", "v2"]


def test_should_preserve_source_rng_and_account_cumulative_work(completed):
    loaded, snapshot, rng, report = completed
    assert _equal(
        snapshot,
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale),
    )
    assert torch.equal(rng, torch.get_rng_state())
    work = report["encoder_work"]
    requests = report["logical_prediction_attempts"]
    assert requests == sum(r["predictor_work"]["request_returns"] for r in report["episodes"])
    assert work["forward_attempts"] == work["forward_returns"] == requests + 2
    assert work["sequences_attempted"] == work["sequences_returned"] == requests + 18
    assert all(
        r["predictor_work"]["source_catalog_columns"] == 6
        and r["predictor_work"]["active_catalog_columns"] == 5
        for r in report["episodes"]
    )
    assert all(
        r["predictor_work"]["head_forward_attempts"] == r["predictor_work"]["optimizer_steps"] == 0
        for r in report["episodes"]
    )


@pytest.mark.parametrize("fault", ["receipt_owner", "episode", "work"])
def test_should_stop_corrupted_prior_evidence_before_next_update(monkeypatch, fault):
    loaded, changed = loaded_case(), False
    report: dict[str, Any] = {}

    def save():
        nonlocal changed
        if changed or not report.get("episodes"):
            return
        changed = True
        if fault == "receipt_owner":
            report["training"][0]["receipt"] = deepcopy(report["training"][0]["receipt"])
        elif fault == "episode":
            report["episodes"][0]["episode"]["success"] = False
        else:
            report["encoder_work"]["forward_returns"] += 1

    with pytest.raises((ValueError, RuntimeError)):
        run(loaded, report, save=save)
    assert changed and report["complete"] is False
    assert report["learner_work"]["commits"] == 1
    assert report["preserved_training"][0]["receipt"]["applied_at"] == 1
    assert report["preserved_episodes"][0]["episode"]["success"] is True


def test_should_preserve_lost_warm_return_and_stop_without_more_episodes(monkeypatch):
    original = linear.update_clm_linear_control

    def lost(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("controlled lost warm return")

    monkeypatch.setattr(linear, "update_clm_linear_control", lost)
    report: dict[str, Any] = {}
    with pytest.raises(Exception, match="controlled lost warm return"):
        run(loaded_case(), report)
    assert report["complete"] is False and len(report["episodes"]) == 4
    assert report["learner_work"]["solve_attempts"] == 2
    assert report["learner_work"]["commits"] == 1
    assert report["learner_work"]["unreturned_work_unknown"] is True
    assert report["preserved_pending_update"]["phase"] == "after_B"


def test_should_retain_partial_episode_and_observed_prediction_work(monkeypatch):
    original = context.CLMLinearDecisionPredictor.__call__

    def lost(self, request):
        original(self, request)
        raise OSError("controlled lost prediction return")

    monkeypatch.setattr(context.CLMLinearDecisionPredictor, "__call__", lost)
    report: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="uncertain"):
        run(loaded_case(), report)
    assert report["complete"] is False and report["episodes"] == []
    partial = report["pending_episode"]["episode"]
    assert partial["complete"] is False and partial["unreturned_prediction_work_unknown"]
    assert report["predictor_work"]["request_returns"] == 1
    assert report["encoder_work"]["forward_returns"] == 3
    assert report["learner_work"]["commits"] == 1


def test_should_fail_final_save_and_preserve_completed_work():
    report: dict[str, Any] = {}

    def save():
        if report.get("complete"):
            report["summary"]["permission_denials"] = 99

    with pytest.raises(ValueError, match="prefix"):
        run(loaded_case(), report, save=save)
    assert report["complete"] is False and report["learner_work"]["commits"] == 3
    assert len(report["episodes"]) == 14
    assert report["preserved_summary"]["permission_denials"] == 1
    assert len(report["preserved_episodes"]) == 14


def test_should_refuse_changed_teacher_role_before_encoder(monkeypatch):
    original = context.training_cases

    def bad(**kwargs):
        rows = original(**kwargs)
        return (replace(rows[0], role="outer_selection"), *rows[1:])

    monkeypatch.setattr(context, "training_cases", bad)
    report: dict[str, Any] = {}
    with pytest.raises(ValueError, match="TRAIN"):
        run(loaded_case(), report)
    assert report["encoder_work"]["forward_attempts"] == 0


def test_should_refuse_control_before_fit_and_changed_catalog_without_solve():
    loaded = loaded_case()
    actions = torch.eye(12)[:6].clone()
    learner = linear.CLMLinearCatalogLearner(loaded.identity, actions, (), candidate_ids=ACTION_IDS)
    with pytest.raises(ValueError, match="committed"):
        _ = learner.current_control
    learner._ids = tuple(reversed(ACTION_IDS))
    with pytest.raises(ValueError, match="owner"):
        _ = learner.current_control
    assert learner.work == linear_catalog_work()


def test_should_not_claim_learning_recovery_when_old_task_was_never_lost():
    rows: list[dict[str, Any]] = [
        dict(
            phase=p,
            scenario="ordinary",
            episode=dict(success=ok, error_recovery_demonstrated=False, stopped_reason="success"),
        )
        for p in ("after_A", "after_B", "after_return")
        for ok in (True, True, True, True)
    ]
    assert context._summary(rows)["learning_recovery_demonstrated"] is False
    rows[4]["episode"]["success"] = False
    assert context._summary(rows)["learning_recovery_demonstrated"] is True
