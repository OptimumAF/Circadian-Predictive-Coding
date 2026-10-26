"""The explicit TRAIN recipe shares real public observations, not outcome labels."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import pytest

import scripts.run_clm_ticket_workflow as study
from scripts.run_ticket_workflow import training_cases
from src.app.ticket_workflow import build_public_workflow_request, run_workflow_episode
from src.core.temporal_decision_sequence import VOCABULARY, policy_for_phase
from src.core.ticket_workflow import ACTION_IDS, CATALOG, WorkflowCapabilities, WorkflowState
from src.core.typed_decision import DecisionRequest, ProbabilityDecisionResult


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("context", [False, True])
def test_should_preserve_legacy_teacher_text_targets_and_full_candidate_order(version, context):
    cases = training_cases(policy_version=version, include_policy_context=context)
    policy = policy_for_phase("after_A" if version == "v1" else "after_return")
    prefix = "train-v1" if version == "v1" else "train"
    group = (
        "train-workflow-traces"
        + ("-v1" if version == "v1" else "")
        + ("-context" if context else "")
    )
    active = (*policy["allowed_ids"], "open", "commit")
    candidates = tuple(c for c in CATALOG if c.candidate_id in active)
    for index, case in enumerate(cases):
        phase = ("sealed", "ready", "staged")[index % 3]
        concept = VOCABULARY[index // 3]
        query = phase + (" " + concept if phase == "ready" else "")
        query += (" policy_" + version) if context else ""
        query += " feedback " + ("initial" if phase == "sealed" else "applied") + " train"
        assert case.request.query == query and case.request.allowed_candidates == candidates
        assert case.request.request_id == f"{prefix}-{index // 3}-{index % 3}"
        assert case.target_id == ("open", policy["rules"][concept], "commit")[index % 3]
        assert case.role == "train" and case.source_group == group


@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("context", [False, True])
def test_should_match_actual_public_episode_inputs_for_each_opt_in_teacher_step(version, context):
    cases = training_cases(
        policy_version=version, include_policy_context=context, public_observation_train=True
    )
    legacy = training_cases(policy_version=version, include_policy_context=context)
    assert len(cases) == 12
    for index, concept in enumerate(VOCABULARY):
        observed: list[DecisionRequest] = []

        def choose(request):
            expected = cases[index * 3 + len(observed)]
            assert request.query == expected.request.query
            assert request.allowed_candidates == expected.request.allowed_candidates
            assert "private-ticket" not in request.query and " train" not in request.query
            assert request.query.endswith(" please")
            observed.append(request)
            return ProbabilityDecisionResult(
                request.request_id,
                expected.target_id,
                tuple(
                    (c.candidate_id, float(c.candidate_id == expected.target_id))
                    for c in request.allowed_candidates
                ),
            )

        report = run_workflow_episode(
            WorkflowState("private-ticket", concept, policy_version=version),
            choose,
            capabilities=WorkflowCapabilities(ACTION_IDS),
            include_policy_context=context,
        )
        assert report["complete"] and report["success"] and len(observed) == 3
        assert report["external_actions"] == report["training_updates"] == 0
    for actual, prior in zip(cases, legacy, strict=True):
        assert actual.request.allowed_candidates == prior.request.allowed_candidates
        assert actual.target_id == prior.target_id and actual.role == prior.role == "train"
        assert actual.source_group == prior.source_group + "-public-observation-v1"
        assert actual.request.query == prior.request.query.removesuffix(" train") + " please"


def test_should_preserve_feedback_capabilities_and_hide_ticket_identity_in_shared_builder():
    state = WorkflowState("private-ticket", "invoice", phase="ready", revision=1)
    caps = WorkflowCapabilities(("commit", "open"))
    request = build_public_workflow_request(state, caps, "public", "stale_revision")
    assert request.query == "ready invoice feedback stale_revision please"
    assert tuple(c.candidate_id for c in request.allowed_candidates) == ("commit", "open")
    assert "private-ticket" not in request.query


@pytest.mark.parametrize("enabled", [False, True])
def test_should_encode_only_six_teacher_queries_and_five_actions_for_selected_recipe(enabled):
    torch = pytest.importorskip("torch")
    from src.core.clm_head import FrozenEncoderIdentity

    calls = []
    features = torch.arange(22, dtype=torch.float32).reshape(11, 2)

    def embed(texts, positions):
        calls.append((texts, positions))
        return features.clone()

    encoder = SimpleNamespace(
        identity=FrozenEncoderIdentity("literal", "fixture", "fixture", "last", "plain", 2),
        embed_texts=embed,
    )
    batch, targets, metadata = study.training_inputs(encoder, public_observation_train=enabled)
    cases = training_cases(public_observation_train=enabled)
    expected_states = tuple(
        dict.fromkeys(c.request.query + "\n\n" + study.INSTRUCTIONS for c in cases)
    )
    assert len(calls) == 1 and len(expected_states) == 6 and len(calls[0][0]) == 11
    assert calls[0][0][:6] == expected_states
    assert calls[0][1] == ("tail",) * 6 + ("head",) * 5
    assert targets.candidate_indices == study.expected_targets()
    assert batch.role == "train" and tuple(batch.actions.shape) == (5, 2)
    assert metadata["encoded_sequences"] == 11 and metadata["demonstration_transitions"] == 12
    if enabled:
        assert metadata["observation_recipe"] == "public_workflow_request_v1"
        assert all(c["source_group"].endswith("-public-observation-v1") for c in metadata["cases"])
    else:
        assert "observation_recipe" not in metadata


@pytest.mark.parametrize(
    "fault", ["legacy_group", "public_without_flag", "role", "target", "candidate_order"]
)
def test_should_refuse_incompatible_teacher_before_encoding(fault, monkeypatch):
    cases = training_cases(public_observation_train=True)
    enabled = fault != "public_without_flag"
    if fault == "legacy_group":
        item = replace(cases[0], source_group="train-workflow-traces")
    elif fault == "role":
        item = replace(cases[0], role="outer_selection")
    elif fault == "target":
        item = replace(cases[0], target_id="commit")
    elif fault == "candidate_order":
        item = replace(
            cases[0],
            request=replace(
                cases[0].request,
                allowed_candidates=tuple(reversed(cases[0].request.allowed_candidates)),
            ),
        )
    else:
        item = cases[0]
    monkeypatch.setattr(study, "training_cases", lambda **kwargs: (item, *cases[1:]))

    def forbidden(*args):
        pytest.fail("unadmitted recipe reached encoder")

    with pytest.raises(ValueError, match="TRAIN"):
        study.training_inputs(
            SimpleNamespace(embed_texts=forbidden), public_observation_train=enabled
        )


def _check_aligned_case(monkeypatch, *, learned_only=False):
    torch = pytest.importorskip("torch")
    from src.adapters.clm_catalog_learner import CLMCatalogLearner
    from src.adapters.clm_head_learner import _equal
    from src.adapters.clm_frozen_encoder import text_recipe
    from src.core.clm_head import FrozenEncoderIdentity
    from test_clm_catalog_learning import parts
    from test_clm_frozen_encoder import FakeTokenizer, backbone
    from test_clm_ticket_workflow import PublicTextTokens

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
    holder = {}
    real_init = CLMCatalogLearner.__init__

    def created(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        holder["learner"] = self

    monkeypatch.setattr(CLMCatalogLearner, "__init__", created)
    report: dict[str, Any] = {}
    study.run_loaded(
        loaded,
        report,
        public_observation_train=True,
        record_train_fit=True,
        reuse_action_features=True,
        learned_only=learned_only,
    )
    assert report["complete"] and report["pending_episode"] is None
    assert len(report["episodes"]) == (3 if learned_only else 6) and all(
        row["episode"]["complete"] for row in report["episodes"]
    )
    assert report["training_data"]["observation_recipe"] == "public_workflow_request_v1"
    assert report["training"]["committed_updates"] == report["source_work"]["commits"] == 8
    assert report["source_work"]["training_rows_attempted"] == 96
    assert report["source_work"]["catalog_candidate_rows_attempted"] == 40
    assert report["source_work"]["catalog_logit_values_attempted"] == 480
    assert report["training"]["permissions"] == dict(training=True, replay=False, evaluation=False)
    assert [r["applied_at"] for r in report["training"]["receipts"]] == list(range(1, 16, 2))
    assert [r["completed_updates"] for r in report["train_fit"]["readouts"]] == [0, 8]
    assert all(r["validated"] and r["source_preserved"] for r in report["train_fit"]["readouts"])
    assert (
        report["source_work"]["prediction_attempts"]
        == report["source_work"]["prediction_returns"]
        == 2
    )
    assert report["source_work"]["snapshot_returns"] == 1
    assert report["source_work"]["restore_attempts"] == 0
    assert holder["learner"].generation == 8
    assert _equal(original, (state.state_dict(), action.state_dict(), scale))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in loaded.model.parameters())
    assert report["encoder_work"]["forward_attempts"] == 1 + report["logical_prediction_attempts"]
    if learned_only:
        assert report["episode_controls"] == dict(observed=["learned"], unobserved=["unchanged"])
        assert tuple(report["summary"]) == ("learned",)
        assert all(r["control"] == "learned" for r in report["episodes"])
        assert [r["episode"]["scenario"] for r in report["episodes"]] == [
            "ordinary",
            "stale_once",
            "revoke_commit",
        ]
        assert report["logical_prediction_attempts"] <= 18
        assert report["predictor_work"]["unchanged"]["request_attempts"] == 0
    else:
        assert "episode_controls" not in report


def test_should_run_aligned_tiny_training_diagnostics_binding_and_all_six_episodes(monkeypatch):
    _check_aligned_case(monkeypatch)


def test_should_observe_only_learned_episodes_without_repeating_unchanged_predictions(monkeypatch):
    _check_aligned_case(monkeypatch, learned_only=True)


@pytest.mark.parametrize("bad", [None, 0, 1, "on"])
def test_should_require_boolean_before_any_encoder_or_model_access(bad):
    for call in (
        lambda: training_cases(public_observation_train=bad),
        lambda: study.training_inputs(SimpleNamespace(), public_observation_train=bad),
        lambda: study.run_loaded(SimpleNamespace(), {}, public_observation_train=bad),
        lambda: study.run_case("unused", "unused", {}, public_observation_train=bad),
    ):
        with pytest.raises(ValueError, match="exact bool"):
            call()


def test_should_forward_explicit_recipe_and_existing_flags_through_cli(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
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
            "--public-observation-train",
            "--record-train-fit",
            "--reuse-action-features",
            "--learned-only",
        ],
    )

    def execute(model, head, report, *, save, **options):
        calls.append((model, head, options))
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    assert calls == [
        (
            Path("unused"),
            Path("unused"),
            dict(
                public_observation_train=True,
                record_train_fit=True,
                reuse_action_features=True,
                learned_only=True,
            ),
        )
    ]


@pytest.mark.parametrize("bad", [None, 0, 1, "on"])
def test_should_refuse_nonboolean_observation_selection_before_model_access(bad):
    with pytest.raises(ValueError, match="exact bool"):
        study.run_loaded(SimpleNamespace(), {}, learned_only=bad)
    with pytest.raises(ValueError, match="exact bool"):
        study.run_case("unused", "unused", {}, learned_only=bad)


@pytest.mark.parametrize("enabled", [False, True])
def test_should_reserve_observation_metadata_only_when_mode_is_enabled(enabled):
    report = {"episode_controls": {"custom": "caller metadata"}}
    with pytest.raises(ValueError if enabled else AttributeError):
        study.run_loaded(SimpleNamespace(), report, learned_only=enabled)
    assert report == {"episode_controls": {"custom": "caller metadata"}}
