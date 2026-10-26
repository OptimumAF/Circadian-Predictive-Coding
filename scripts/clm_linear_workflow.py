"""Direct-fit opt-in composition for the existing CLM workflow script.

Calls its original TRAIN builder and local dispatcher; no new stream, loader,
optimizer, data release or experiment allowance. The caller supplies the existing
three development cases and owns any prospective native execution budget.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any

import torch

from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
from src.adapters.clm_head_learner import _equal
from src.adapters.clm_linear_decision import (
    CLMLinearDecisionPredictor,
    _same_work,
    _encoder_return_work,
)
from src.adapters.clm_linear_fit_control import (
    fit_clm_linear_control,
    predict_clm_linear_control,
    _capture,
    _require_original,
)
from src.app.clm_workflow_training import ACTIVE_IDS, expected_targets, summarize_training_fit
from src.app.ticket_workflow import run_workflow_episode
from src.core.clm_action_features import FrozenActionFeatures
from src.core.clm_catalog import FrozenCatalogTrainingBatch, CLMCatalogTargets
from src.core.clm_head import FrozenChoiceBatch, _exact
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState
from src.core.typed_decision import DecisionCandidate


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _head_traits(loaded):
    return tuple(
        (
            tuple((id(m), m.training) for m in head.modules()),
            tuple(
                (
                    id(p),
                    p.requires_grad,
                    p.untyped_storage().data_ptr(),
                    p._version,
                    id(p.grad),
                    None if p.grad is None else p.grad._version,
                )
                for p in head.parameters()
            ),
        )
        for head in (loaded.state_head, loaded.action_head)
    ) + (
        (
            loaded.logit_scale.untyped_storage().data_ptr(),
            loaded.logit_scale._version,
            loaded.logit_scale.requires_grad,
            id(loaded.logit_scale.grad),
            None if loaded.logit_scale.grad is None else loaded.logit_scale.grad._version,
        ),
    )


def run_linear_loaded(
    loaded,
    report,
    *,
    training_inputs,
    development,
    instructions,
    before_step=lambda: None,
    save=lambda: None,
    public_observation_train=False,
):
    if (
        type(public_observation_train) is not bool
        or not callable(before_step)
        or not callable(save)
    ):
        raise ValueError("linear workflow requires exact data option and callable boundaries")
    if (
        type(development) is not tuple
        or len(development) != 3
        or tuple(row[0] for row in development) != ("ordinary", "stale_once", "revoke_commit")
    ):
        raise ValueError("linear workflow requires the original three development scenarios")
    if type(report) is not dict or any(
        k in report for k in ("identity", "linear_fit", "train_fit", "episodes")
    ):
        raise ValueError("linear workflow needs unused source/fit/episode keys")
    owners = (
        loaded.model,
        loaded.tokenizer,
        loaded.state_head,
        loaded.action_head,
        loaded.logit_scale,
        loaded.identity,
    )
    identity = deepcopy(loaded.identity)
    heads = deepcopy(
        (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale)
    )
    head_traits = _head_traits(loaded)
    rng = torch.get_rng_state().clone()
    encoder = FrozenCLMEncoder(
        loaded.model,
        loaded.tokenizer,
        identity=loaded.identity,
        max_sequences=16,
        max_forward_calls=19,
    )
    provider = None
    features = None
    proofs = {}
    episodes: list[dict[str, Any]] = []
    episode_owners: list[dict[str, Any]] = []
    attempts = 0
    succeeded = False
    report.update(
        complete=False,
        objective="clm_frozen_feature_linear_fit_control_v1",
        identity=asdict(loaded.identity),
        episodes=episodes,
        pending_episode=None,
        episode_controls=dict(
            observed=["linear_fit"], unobserved=["unchanged", "contrastive_fine_tuned"]
        ),
        logical_prediction_attempts=0,
        scope="one detached TRAIN fit then original local development episodes; no final/probe/CPC claim",
    )
    live_pin: tuple[dict[str, Any], dict[str, Any]] = encoder.work, {}
    work_owners = None
    work_proof = ""

    def protect(key):
        proofs[key] = report[key], _json(report[key])

    def live_work():
        return encoder.work, {} if provider is None else provider.work

    def refresh():
        nonlocal work_proof, work_owners
        report["encoder_work"], report["predictor_work"] = live_work()
        work_owners = report["encoder_work"], report["predictor_work"]
        work_proof = _json(work_owners)

    def guard(expected=None):
        desired = live_pin if expected is None else expected
        if any(not _same_work(a, b) for a, b in zip(live_work(), desired, strict=True)):
            raise ValueError("linear workflow paid work differs from its observed operation")
        if work_owners is not None and (
            report.get("encoder_work") is not work_owners[0]
            or report.get("predictor_work") is not work_owners[1]
            or _json((report["encoder_work"], report["predictor_work"])) != work_proof
        ):
            raise ValueError("original linear workflow reported work changed")
        for key, (owner, text) in proofs.items():
            if report.get(key) is not owner or _json(report[key]) != text:
                raise ValueError("original linear workflow paid prefix changed")
        if (
            type(report["logical_prediction_attempts"]) is not int
            or report["logical_prediction_attempts"] != attempts
            or report["episodes"] is not episodes
            or len(episodes) != len(episode_owners)
            or any(a is not b for a, b in zip(episodes, episode_owners, strict=True))
        ):
            raise ValueError("linear workflow episode owners or request clock changed")
        if any(
            a is not b
            for a, b in zip(
                (
                    loaded.model,
                    loaded.tokenizer,
                    loaded.state_head,
                    loaded.action_head,
                    loaded.logit_scale,
                    loaded.identity,
                ),
                owners,
                strict=True,
            )
        ) or not _equal(
            heads,
            (loaded.state_head.state_dict(), loaded.action_head.state_dict(), loaded.logit_scale),
        ):
            raise ValueError("original loaded workflow heads changed")
        if not _equal(asdict(loaded.identity), asdict(identity)):
            raise ValueError("original loaded workflow identity changed")
        if _head_traits(loaded) != head_traits:
            raise ValueError("original loaded workflow head modes, gradients or storage changed")
        if not torch.equal(torch.get_rng_state(), rng):
            raise ValueError("linear workflow changed caller RNG")
        encoder._check()
        if provider is not None:
            provider._check()
        if features is not None:
            _require_original(features[0], features[1], features[2])

    def admit(expected):
        nonlocal live_pin
        guard(expected)
        live_pin = expected
        refresh()

    def fail(error):
        names = dict(
            linear_fit="training_prefix",
            episodes="episode_prefix",
            summary="summary",
            pending_episode="pending_episode",
        )
        for key, name in names.items():
            if key in proofs:
                report["preserved_" + name] = json.loads(proofs[key][1])
        report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])

    for key in ("identity", "objective", "scope", "episode_controls", "episodes"):
        protect(key)
    refresh()
    try:
        before_step()
        guard()
        before = encoder.work
        batch, targets, metadata = training_inputs(
            encoder, public_observation_train=public_observation_train
        )
        observed, _ = _encoder_return_work(encoder, before, ("tail",) * 6 + ("head",) * 5)
        admit((observed, {}))
        _exact(batch, FrozenCatalogTrainingBatch)
        _exact(targets, CLMCatalogTargets)
        if (
            batch.candidate_ids != ACTIVE_IDS
            or targets.candidate_ids != ACTIVE_IDS
            or targets.candidate_indices != expected_targets()
            or len(batch.states) != 12
        ):
            raise ValueError("linear workflow requires original twelve current-policy TRAIN rows")
        features = batch, targets, _capture(batch, targets)
        report["training_data"] = metadata
        protect("training_data")
        save()
        guard()
        before_step()
        guard()
        fitted = fit_clm_linear_control(batch, targets, report)
        guard()
        protect("linear_fit")
        save()
        guard()
        diagnostic = report["train_fit"] = dict(role="train", complete=False, readout={})
        p = predict_clm_linear_control(
            fitted,
            FrozenChoiceBatch(batch.identity, batch.states, batch.actions, "train"),
            ACTIVE_IDS,
            diagnostic["readout"],
        )
        diagnostic.update(
            complete=True, probabilities=p.tolist(), **summarize_training_fit(p.tolist())
        )
        guard()
        protect("train_fit")
        actions = FrozenActionFeatures(
            batch.identity,
            tuple(
                # The original teacher metadata supplies descriptions, not targets from episodes.
                DecisionCandidate(**c)
                for c in metadata["cases"][0]["request"]["allowed_candidates"]
            ),
            batch.actions,
        )
        provider = CLMLinearDecisionPredictor(encoder, fitted, actions, instructions=instructions)
        admit((live_pin[0], provider.work))
        save()
        guard()
        for index, (scenario, concept) in enumerate(development):
            guard()
            report["pending_episode"] = dict(
                control="linear_fit", scenario=scenario, returned=False, effects_unknown=True
            )
            save()
            guard()

            def predict(request):
                nonlocal attempts
                before_step()
                guard()
                attempts += 1
                report["logical_prediction_attempts"] = attempts
                before_encoder, before_provider = live_work()
                try:
                    result = provider(request)
                    expected_encoder, _ = _encoder_return_work(encoder, before_encoder, ("tail",))
                    expected_provider = dict(before_provider)
                    for key in (
                        "request_attempts",
                        "request_returns",
                        "encoder_attempts",
                        "encoder_returns",
                        "linear_readout_attempts",
                        "linear_readout_returns",
                        "linear_rows",
                    ):
                        expected_provider[key] += 1
                    expected_provider["unreturned_work_unknown"] = False
                    admit((expected_encoder, expected_provider))
                    return result
                finally:
                    refresh()
                    save()
                    guard()

            episode = run_workflow_episode(
                WorkflowState(f"development-ticket-{index}", concept),
                predict,
                capabilities=WorkflowCapabilities(ACTION_IDS),
                max_steps=6,
                scenario=scenario,
            )
            report["pending_episode"].update(
                returned=True, episode=episode, effects_unknown=not episode["complete"]
            )
            guard()
            row = dict(
                control="linear_fit",
                role="outer_selection",
                source_group="development-workflow-traces",
                fit_calls=1,
                optimizer_steps=0,
                episode=episode,
            )
            episodes.append(row)
            episode_owners.append(row)
            protect("episodes")
            report["pending_episode"] = None
            save()
            guard()
            if not episode["complete"]:
                raise RuntimeError("linear workflow episode contains uncertain paid work")
        report["summary"] = dict(
            linear_fit=dict(
                episodes=3,
                successes=sum(r["episode"]["success"] for r in episodes),
                recoveries=sum(r["episode"]["error_recovery_demonstrated"] for r in episodes),
                permission_denials=sum(
                    r["episode"]["stopped_reason"] == "permission_denied" for r in episodes
                ),
            )
        )
        protect("summary")
        protect("pending_episode")
        report["complete"] = True
        succeeded = True
    except Exception as error:
        fail(error)
        raise
    finally:
        refresh()
        try:
            save()
            if succeeded:
                guard()
                if report.get("complete") is not True:
                    raise ValueError("linear workflow completion changed during final save")
            else:
                report["complete"] = False
        except Exception as error:
            fail(error)
            raise
    return report
