"""Bounded raw catalog reuse; actual tiny encoder, projections, TRAIN and effects."""

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import scripts.run_clm_ticket_workflow as study
from src.adapters.clm_catalog_decision import CLMCatalogPredictorBinding
from src.adapters.clm_decision import CLMDecisionPredictor
from src.adapters.clm_head_learner import _equal
from src.core.clm_action_features import FrozenActionFeatures
from test_clm_catalog_decision import request as catalog_request, trained_source
from test_clm_catalog_learning import parts
from test_clm_frozen_encoder import encoder, request
from test_clm_ticket_workflow import execute


MAPPING = {"query\n\nChoose.": [1], "first": [2], "second": [3]}


def raw_case():
    frozen, model, tokenizer = encoder(torch, mapping=MAPPING, forwards=8)
    rows = frozen.embed_texts(("first", "second"), ("head", "head"))
    payload = FrozenActionFeatures(frozen.identity, request().allowed_candidates, rows)
    heads = parts(torch)
    return frozen, model, tokenizer, payload, heads


def predictor(case, *, cached=True):
    frozen, _, _, payload, heads = case
    options = {"action_features": payload} if cached else {}
    return CLMDecisionPredictor(
        frozen, *heads, instructions="Choose.", head_revision="raw-reuse", **options
    )


def test_should_reuse_raw_rows_but_project_both_heads_on_every_query():
    case = raw_case()
    frozen, _, tokenizer, payload, _ = case
    rng = torch.get_rng_state().clone()
    cached, reference = predictor(case), predictor(case, cached=False)
    a, b, c = cached(request()), reference(request()), cached(request())
    assert a == b == c
    assert tuple(k for k, _ in a.probabilities) == ("z", "a")
    assert frozen.work["sequences_attempted"] == 2 + 1 + 3 + 1
    assert len(tokenizer.calls) == 7
    work = cached.work
    assert work["head_forward_attempts"] == work["head_forward_returns"] == 4
    assert work["projected_rows"] == 6 and work["action_feature_rows_used"] == 4
    assert work["action_feature_copy_attempts"] == work["action_feature_copy_returns"] == 1
    assert work["action_feature_elements_copied"] == 4
    assert work["action_feature_read_attempts"] == work["action_feature_read_returns"] == 2
    assert work["action_feature_guard_attempts"] == work["action_feature_guard_returns"] > 2
    assert work["action_feature_guard_values_checked"] == 4 * work["action_feature_guard_returns"]
    assert not any(key.startswith("action_feature") for key in reference.work)
    assert cached._action_values.data_ptr() != payload.features.data_ptr()
    assert torch.equal(torch.get_rng_state(), rng)


def test_should_reproject_the_same_raw_catalog_through_each_new_head_owner():
    case = raw_case()
    first = predictor(case)
    # Literal changed head, not another fit or selected random initialization.
    with torch.no_grad():
        case[4][1].weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    second, reference = predictor(case), predictor(case, cached=False)
    a, b, c = first(request()), second(request()), reference(request())
    assert b == c and a.probabilities != b.probabilities
    assert first.work["head_forward_returns"] == second.work["head_forward_returns"] == 2


@pytest.mark.parametrize("change", ["order", "id", "description"])
def test_should_refuse_a_different_catalog_before_encoder_or_projection(change):
    case = raw_case()
    actor = predictor(case)
    original = request()
    candidates = original.allowed_candidates
    if change == "order":
        candidates = candidates[::-1]
    elif change == "id":
        candidates = (replace(candidates[0], candidate_id="other"), candidates[1])
    else:
        candidates = (replace(candidates[0], description="other"), candidates[1])
    before = case[0].work
    with pytest.raises(ValueError, match="candidate IDs, descriptions or order"):
        actor(replace(original, allowed_candidates=candidates))
    assert case[0].work == before
    assert actor.work["request_attempts"] == actor.work["head_forward_attempts"] == 0
    assert actor(request()).request_id == original.request_id


@pytest.mark.parametrize(
    "fault", ["identity", "shape", "dtype", "nan", "grad", "duplicate", "backing"]
)
def test_should_refuse_invalid_features_before_predictor_copy(fault):
    case = raw_case()
    frozen, _, _, payload, heads = case
    identity, candidates, values = payload.identity, payload.candidates, payload.features
    if fault == "identity":
        identity = replace(identity, revision="different")
    elif fault == "shape":
        values = values[:1]
    elif fault == "dtype":
        values = values.double()
    elif fault == "nan":
        values = values.clone()
        values[0, 0] = float("nan")
    elif fault == "grad":
        values = values.clone().requires_grad_()
    elif fault == "backing":
        values = torch.zeros(100, 2)[:2]
    else:
        candidates = (candidates[0], candidates[0])
    before = frozen.work
    with pytest.raises(ValueError):
        CLMDecisionPredictor(
            frozen,
            *heads,
            instructions="Choose.",
            head_revision="invalid",
            action_features=FrozenActionFeatures(identity, candidates, values),
        )
    assert frozen.work == before


@pytest.mark.parametrize(
    "fault",
    ["input", "input_data", "replace_input", "owned", "owned_data", "replace_owned", "metadata"],
)
def test_should_permanently_stop_after_raw_feature_corruption(fault):
    case = raw_case()
    actor = predictor(case)
    payload = case[3]
    if fault == "input":
        payload.features.add_(0.1)
    elif fault == "input_data":
        payload.features.data.add_(0.1)
    elif fault == "replace_input":
        object.__setattr__(payload, "features", payload.features.clone())
    elif fault == "owned":
        actor._action_values.add_(0.1)
    elif fault == "owned_data":
        actor._action_values.data.add_(0.1)
    elif fault == "replace_owned":
        actor._action_values = actor._action_values.clone()
    else:
        object.__setattr__(payload.candidates[0], "description", "changed")
    before = case[0].work
    with pytest.raises(ValueError):
        actor(request())
    assert case[0].work == before and actor.work["request_attempts"] == 0
    assert actor._failed is True
    with pytest.raises(ValueError, match="stopped"):
        actor(request())


def test_should_refuse_a_changed_frozen_encoder_before_query_encoding():
    case = raw_case()
    actor = predictor(case)
    case[1].anchor.add_(0.1)
    before = case[0].work
    with pytest.raises(ValueError, match="backbone"):
        actor(request())
    assert case[0].work == before and actor.work["request_attempts"] == 0
    assert actor._failed


def test_should_not_refresh_after_encoder_mode_is_repaired():
    case = raw_case()
    actor = predictor(case)
    case[1].train()
    with pytest.raises(ValueError, match="eval mode"):
        actor(request())
    case[1].eval()
    with pytest.raises(ValueError, match="stopped"):
        actor(request())
    assert actor.work["request_attempts"] == 0


def test_should_keep_raw_read_and_partial_head_charges_after_failed_prediction():
    actor = predictor(raw_case())

    def fail(*args):
        raise RuntimeError("lost head return")

    actor._action_head.register_forward_hook(fail)
    with pytest.raises(RuntimeError, match="lost head"):
        actor(request())
    work = actor.work
    assert work["encoder_returns"] == work["action_feature_read_returns"] == 1
    assert work["action_feature_rows_used"] == 2
    assert work["head_forward_attempts"] == 2 and work["head_forward_returns"] == 1
    assert work["request_returns"] == 0 and work["unreturned_work_unknown"] is True
    with pytest.raises(ValueError, match="stopped"):
        actor(request())
    assert actor.work == work


def test_should_bind_trained_heads_with_only_one_raw_copy_and_one_public_snapshot():
    case = trained_source(torch)
    source, frozen = case["source"], case["encoder"]
    payload = FrozenActionFeatures(
        frozen.identity, catalog_request().allowed_candidates, case["training"].actions
    )
    before, counts = source._capture(), source.work
    factory = CLMCatalogPredictorBinding(
        source, frozen, instructions="Choose.", head_revision="cached", action_features=payload
    )
    actor = factory.bind()
    assert _equal(vars(source._capture()), vars(before))
    expected = dict(counts)
    expected["snapshot_attempts"] += 1
    expected["snapshot_returns"] += 1
    assert source.work == expected
    assert factory.work["action_feature_copy_returns"] == 1
    assert factory.work["action_feature_elements_copied"] == 6
    assert factory.receipt["work"] == factory.work
    assert not hasattr(factory, "_action_features")
    assert actor(catalog_request()).request_id == "learned-choice"
    assert actor.work["action_feature_rows_used"] == 3


@pytest.fixture(scope="module")
def workflow_pair():
    return execute(), execute(reuse_action_features=True)


def test_should_preserve_complete_workflow_outcomes_and_train_dose_with_less_encoding(
    workflow_pair,
):
    baseline, cached = workflow_pair
    a, b = baseline["report"], cached["report"]
    assert a["complete"] is b["complete"] is True
    assert a["training"] == b["training"]
    assert a["episodes"] == b["episodes"] and a["summary"] == b["summary"]
    assert a["logical_prediction_attempts"] == b["logical_prediction_attempts"]
    calls = b["logical_prediction_attempts"]
    assert 0 < calls <= 36 and len(b["episodes"]) == 6
    assert b["encoder_work"]["sequences_attempted"] == 11 + calls
    assert a["encoder_work"]["sequences_attempted"] == 11 + 6 * calls
    assert b["encoder_work"]["forward_returns"] == a["encoder_work"]["forward_returns"]
    assert b["action_feature_reuse"]["source"] == "already_paid_train_candidate_rows"
    assert b["action_feature_reuse"]["rows"] == 5
    assert b["action_feature_reuse"]["raw_values_per_owner"] == 10
    assert "action_feature_reuse" not in a
    for control in ("unchanged", "learned"):
        aw, bw = a["predictor_work"][control], b["predictor_work"][control]
        assert all(bw[key] == value for key, value in aw.items())
        assert bw["action_feature_copy_returns"] == 1
        assert bw["action_feature_elements_copied"] == 10
        assert bw["action_feature_rows_used"] == 5 * bw["request_returns"]
        assert bw["head_forward_returns"] == 2 * bw["request_returns"]
    assert b["binding_work"]["action_feature_elements_copied"] == 10


def test_should_keep_workflow_source_rng_and_loaded_heads_isolated(workflow_pair):
    case = workflow_pair[1]
    source = case["holder"]["learner"]
    assert _equal(vars(source._capture()), vars(case["holder"]["source_pin"]))
    assert source.completed_updates == 8 and source.training_examples == 96
    assert source.work["snapshot_returns"] == 1
    loaded = case["loaded"]
    assert _equal(
        case["original"],
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale),
    )
    assert torch.equal(torch.get_rng_state(), case["rng"])


def test_should_forward_cli_opt_in_without_changing_the_default(tmp_path, monkeypatch):
    import sys

    output = tmp_path / "cached.json"
    calls = []

    def run(model, head, report, *, save, reuse_action_features):
        calls.append((model, head, reuse_action_features))
        report["complete"] = True
        save()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--model-dir",
            "model",
            "--head",
            "head",
            "--output",
            str(output),
            "--reuse-action-features",
        ],
    )
    monkeypatch.setattr(study, "run_case", run)
    study.main()
    assert calls == [(Path("model"), Path("head"), True)]


def test_should_refuse_non_boolean_workflow_opt_in_before_model_work():
    with pytest.raises(ValueError, match="explicit boolean"):
        study.run_loaded(None, {}, reuse_action_features=1)
