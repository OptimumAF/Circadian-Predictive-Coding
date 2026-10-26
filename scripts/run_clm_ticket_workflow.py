"""Fixed catalog CLM TRAIN-to-local-effects slice; prepared offline assets only."""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.run_ticket_workflow import training_cases
from src.app.clm_workflow_training import ACTIVE_IDS, PASSES, fit_catalog_workflow, expected_targets
from src.app.clm_workflow_training import summarize_training_fit
from src.app.ticket_workflow import run_workflow_episode
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import CLMHeadPolicy
from src.core.ticket_workflow import ACTION_IDS, WorkflowCapabilities, WorkflowState
from src.infra.clm_local_loader import load_local_clm
from src.shared.process_memory import ProcessRssSampler

INSTRUCTIONS = "Choose the ticket workflow tool."
DEVELOPMENT = (("ordinary", "invoice"), ("stale_once", "parcel"), ("revoke_commit", "invoice"))


def _admit_context_options(enabled, public, others):
    if type(enabled) is not bool:
        raise ValueError("linear_context_updates must be an exact bool")
    if enabled and (public is not True or any(type(v) is not bool or v for v in others)):
        raise ValueError(
            "linear-context-updates requires public-observation-train and no other workflow modes"
        )


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def training_inputs(encoder, *, public_observation_train=False):
    """Bind twelve TRAIN cases from the explicitly selected observation recipe."""
    if type(public_observation_train) is not bool:
        raise ValueError("public_observation_train must be an exact bool")
    cases = (
        training_cases(public_observation_train=True)
        if public_observation_train
        else training_cases()
    )
    group = "train-workflow-traces" + ("-public-observation-v1" if public_observation_train else "")
    if (
        type(cases) is not tuple
        or len(cases) != 12
        or any(c.role != "train" or c.source_group != group for c in cases)
        or any(
            tuple(c.candidate_id for c in row.request.allowed_candidates) != ACTIVE_IDS
            for row in cases
        )
        or tuple(ACTIVE_IDS.index(c.target_id) for c in cases) != expected_targets()
    ):
        raise ValueError("only the fixed twelve current-policy workflow TRAIN cases admitted")
    states = tuple(dict.fromkeys(c.request.query.strip() + "\n\n" + INSTRUCTIONS for c in cases))
    actions = tuple(c.description for c in cases[0].request.allowed_candidates)
    if len(states) != 6 or len(actions) != 5:
        raise ValueError("fixed workflow raw text geometry differs")
    features = encoder.embed_texts(states + actions, ("tail",) * 6 + ("head",) * 5)
    batch = FrozenCatalogTrainingBatch(
        encoder.identity,
        features[
            [states.index(c.request.query.strip() + "\n\n" + INSTRUCTIONS) for c in cases]
        ].clone(),
        features[6:].clone(),
        ACTIVE_IDS,
    )
    targets = CLMCatalogTargets(ACTIVE_IDS, expected_targets())
    metadata: dict[str, Any] = dict(
        cases=[asdict(c) for c in cases],
        states=list(states),
        actions=list(actions),
        encoded_sequences=11,
        demonstration_transitions=12,
        candidate_constructions=12,
        catalog_entries_considered=72,
        candidates_returned=60,
    )
    if public_observation_train:
        metadata["observation_recipe"] = "public_workflow_request_v1"
    return (
        batch,
        targets,
        metadata,
    )


def _summary(episodes):
    return {
        control: dict(
            episodes=len(rows),
            successes=sum(r["success"] for r in rows),
            recoveries=sum(r["error_recovery_demonstrated"] for r in rows),
            permission_denials=sum(r["stopped_reason"] == "permission_denied" for r in rows),
            wrong_completions=sum(r["stopped_reason"] == "wrong_workflow" for r in rows),
        )
        for control in ("unchanged", "learned")
        for rows in ([r["episode"] for r in episodes if r["control"] == control],)
    }


def _training_fit_readout(learner, batch, outcome):
    """One paid native read on original TRAIN features, with full state guards."""
    started = monotonic()
    from src.adapters.clm_head_learner import _equal
    from src.app.clm_temporal_sequence import _feature_pin, _require_features, _tensor_traits
    from src.app.clm_reserved_probe import _require_work
    from src.core.clm_head import FrozenChoiceBatch, _exact

    _exact(batch, FrozenCatalogTrainingBatch)
    if batch.candidate_ids != ACTIVE_IDS or batch.identity != learner.identity:
        raise ValueError("TRAIN fit requires the original catalog and identity")
    _tensor_traits(batch.states, (12, learner.identity.embedding_dim))
    _tensor_traits(batch.actions, (5, learner.identity.embedding_dim))
    pin = _feature_pin(batch)
    values = batch.states.clone(), batch.actions.clone()
    before = learner._capture()
    generation = learner.generation
    owners = learner._bundle, learner._rng, learner._work
    work = learner.work
    outcome.update(
        attempted=True,
        returned=False,
        validated=False,
        source_preserved=None,
        work_before=work,
        probabilities=None,
    )
    try:
        result = learner.predict_choices(
            FrozenChoiceBatch(batch.identity, batch.states, batch.actions, role="train")
        )
        outcome["returned"] = True
        expected = dict(work)
        for key, delta in dict(
            prediction_attempts=1,
            prediction_returns=1,
            prediction_rows_attempted=12,
            choice_prediction_attempts=1,
            choice_prediction_returns=1,
            choice_candidate_rows_attempted=5,
        ).items():
            expected[key] += delta
        _require_work(learner.work, expected)
        _require_features(batch, pin)
        if (
            any(
                a is not b
                for a, b in zip((learner._bundle, learner._rng, learner._work), owners, strict=True)
            )
            or learner.generation != generation
            or not _equal(vars(learner._capture()), vars(before))
            or not batch.states.equal(values[0])
            or not batch.actions.equal(values[1])
        ):
            outcome["source_preserved"] = False
            raise ValueError("TRAIN fit changed source state, owners or features")
        learner._check()
        outcome["source_preserved"] = True
        _tensor_traits(result, (12, 5))
        probabilities = result.detach().tolist()
        summary = summarize_training_fit(probabilities)
        outcome.update(probabilities=probabilities, **summary, validated=True)
    except Exception as error:
        outcome["error"] = f"{type(error).__name__}: {error}"[:512]
        raise
    finally:
        outcome.update(work_after=learner.work, elapsed_seconds=monotonic() - started)


def run_loaded(
    loaded,
    report,
    *,
    before_step=lambda: None,
    save=lambda: None,
    reuse_action_features=False,
    record_train_fit=False,
    public_observation_train=False,
    learned_only=False,
    linear_fit_control=False,
    linear_context_updates=False,
):
    """Compose existing encoder/learner/binding/dispatcher with bounded ownership."""
    _admit_context_options(
        linear_context_updates,
        public_observation_train,
        (linear_fit_control, reuse_action_features, record_train_fit, learned_only),
    )
    if linear_context_updates:
        from scripts.clm_linear_context_workflow import run_context_loaded

        return run_context_loaded(
            loaded, report, instructions=INSTRUCTIONS, before_step=before_step, save=save
        )
    if type(linear_fit_control) is not bool:
        raise ValueError("linear_fit_control must be an exact bool")
    if linear_fit_control:
        if any(
            type(value) is not bool or value
            for value in (reuse_action_features, record_train_fit, learned_only)
        ):
            raise ValueError(
                "linear-fit-control cannot combine with contrastive-only workflow options"
            )
        from scripts.clm_linear_workflow import run_linear_loaded

        return run_linear_loaded(
            loaded,
            report,
            training_inputs=training_inputs,
            development=DEVELOPMENT,
            instructions=INSTRUCTIONS,
            before_step=before_step,
            save=save,
            public_observation_train=public_observation_train,
        )
    import torch
    from src.adapters.clm_catalog_decision import CLMCatalogPredictorBinding, _tensor_elements
    from src.adapters.clm_catalog_learner import CLMCatalogLearner
    from src.adapters.clm_decision import CLMDecisionPredictor, _geometry
    from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
    from src.adapters.clm_head_learner import _equal
    from src.core.clm_action_features import FrozenActionFeatures

    if type(reuse_action_features) is not bool:
        raise ValueError("raw action reuse must be an explicit boolean")
    if type(record_train_fit) is not bool:
        raise ValueError("TRAIN fit recording must be an explicit boolean")
    if type(public_observation_train) is not bool:
        raise ValueError("public_observation_train must be an exact bool")
    if type(learned_only) is not bool:
        raise ValueError("learned_only must be an exact bool")
    keys: tuple[str, ...] = (
        ("training", "episodes", "identity", "train_fit")
        if record_train_fit
        else ("training", "episodes", "identity")
    )
    if learned_only:
        keys += ("episode_controls",)
    if type(report) is not dict or any(k in report for k in keys):
        raise ValueError("workflow needs fresh source/training/episode keys")
    caller_rng = torch.get_rng_state().clone()
    original = (
        deepcopy(loaded.state_head.state_dict()),
        deepcopy(loaded.action_head.state_dict()),
        deepcopy(loaded.logit_scale),
    )
    original_owners = loaded.state_head, loaded.action_head, loaded.logit_scale
    projection = _geometry(loaded.state_head, loaded.identity.embedding_dim)
    if _geometry(loaded.action_head, loaded.identity.embedding_dim) != projection:
        raise ValueError("workflow requires matching native head geometry")
    encoder = FrozenCLMEncoder(
        loaded.model,
        loaded.tokenizer,
        identity=loaded.identity,
        max_sequences=16,
        max_forward_calls=37,
    )
    unchanged = None
    if not reuse_action_features:
        unchanged = CLMDecisionPredictor(
            encoder,
            loaded.state_head,
            loaded.action_head,
            loaded.logit_scale,
            instructions=INSTRUCTIONS,
            head_revision="unchanged-workflow",
        )
    state, action = deepcopy(loaded.state_head), deepcopy(loaded.action_head)
    state.requires_grad_(True)
    action.requires_grad_(True)
    learner = CLMCatalogLearner(
        state,
        action,
        loaded.logit_scale.detach().clone(),
        identity=loaded.identity,
        policy=CLMHeadPolicy(
            projection_dim=projection, max_updates=PASSES, max_rows=12, max_restores=0
        ),
    )
    del state, action
    predictors = {} if unchanged is None else {"unchanged": unchanged}
    report.update(
        complete=False,
        objective=CATALOG_OBJECTIVE,
        identity=asdict(loaded.identity),
        torch_version=str(torch.__version__),
        episodes=[],
        pending_episode=None,
        scope="catalog TRAIN then frozen local workflow; no final/probe/CPC claim",
        logical_prediction_attempts=0,
        source_guard_work=dict(attempts=0, returns=0, tensor_elements=0),
    )
    if learned_only:
        report["episode_controls"] = dict(observed=["learned"], unobserved=["unchanged"])
    binding = None
    attempts = 0
    guard_work = dict(
        attempts=0, returns=0, tensor_elements=0, unreturned_capture_work_unknown=False
    )
    report["source_guard_work"] = dict(guard_work)
    live_pin = None
    work_proof = ""
    source_state = source_owner = source_work = None
    source_rng_owner = source_work_owner = None
    training_proof = episode_proof = ""
    fit_records: list[dict[str, Any]] = []
    fit_report: dict[str, Any] = {}
    fit_proof = ""
    if record_train_fit:
        fit_report = dict(
            definition="workflow_train_probability_margin_v1",
            role="train",
            candidate_ids=list(ACTIVE_IDS),
            readouts=fit_records,
            scope="fixed TRAIN fit only; no checkpoint selection or task success inference",
        )
        report["train_fit"] = fit_report
        fit_proof = _json(fit_report)
    fit_owners: list[dict[str, Any]] = []
    fit_features: Any = None

    def capture():
        guard_work["attempts"] += 1
        guard_work["unreturned_capture_work_unknown"] = True
        try:
            snapshot = learner._capture()
            guard_work["returns"] += 1
            guard_work["tensor_elements"] += _tensor_elements(vars(snapshot))
            guard_work["unreturned_capture_work_unknown"] = False
        finally:
            report["source_guard_work"] = dict(guard_work)
        return snapshot

    def live_work():
        return encoder.work, {key: p.work for key, p in predictors.items()}

    def reported_work():
        return {
            key: report.get(key)
            for key in (
                "source_work",
                "encoder_work",
                "predictor_work",
                "binding_work",
                "binding_receipt",
            )
        }

    def require_source():
        if learned_only and report.get("episode_controls") != dict(
            observed=["learned"], unobserved=["unchanged"]
        ):
            raise ValueError("declared workflow observation controls changed")
        if record_train_fit and (
            report.get("train_fit") is not fit_report
            or fit_report["readouts"] is not fit_records
            or len(fit_records) != len(fit_owners)
            or any(a is not b for a, b in zip(fit_records, fit_owners, strict=True))
            or _json(fit_report) != fit_proof
        ):
            raise ValueError("original TRAIN fit evidence changed")
        if fit_features is not None:
            original_batch, pin, values, original_targets = fit_features
            from src.app.clm_temporal_sequence import _require_features

            original_batch.__post_init__()
            original_targets.__post_init__()
            _require_features(original_batch, pin)
            if (
                original_batch.identity != learner.identity
                or original_batch.candidate_ids != ACTIVE_IDS
                or original_targets.candidate_ids != ACTIVE_IDS
                or original_targets.candidate_indices != expected_targets()
                or not original_batch.states.equal(values[0])
                or not original_batch.actions.equal(values[1])
            ):
                raise ValueError("original TRAIN fit feature values or metadata changed")
        if training_proof and _json(report["training"]) != training_proof:
            raise ValueError("completed workflow TRAIN evidence changed")
        if (
            not _equal(report["logical_prediction_attempts"], attempts)
            or not _equal(report["source_guard_work"], guard_work)
            or (work_proof and _json(reported_work()) != work_proof)
            or (live_pin is not None and not _equal(live_work(), live_pin))
        ):
            raise ValueError("original workflow serving work or counters changed")
        encoder._check()
        if (
            any(
                now is not before
                for now, before in zip(
                    (loaded.state_head, loaded.action_head, loaded.logit_scale),
                    original_owners,
                    strict=True,
                )
            )
            or not _equal(
                original,
                (
                    loaded.state_head.state_dict(),
                    loaded.action_head.state_dict(),
                    loaded.logit_scale,
                ),
            )
            or not torch.equal(torch.get_rng_state(), caller_rng)
        ):
            raise ValueError("original loaded heads or caller RNG changed")
        if source_state is not None:
            if (
                learner._bundle is not source_owner
                or learner._rng is not source_rng_owner
                or learner._work is not source_work_owner
                or not _equal(learner.work, source_work)
                or not _equal(vars(capture()), vars(source_state))
            ):
                raise ValueError("frozen workflow TRAIN source state or paid work changed")
        if episode_proof and _json(report["episodes"]) != episode_proof:
            raise ValueError("completed workflow episode prefix changed")

    def refresh():
        nonlocal live_pin, work_proof
        report["source_work"] = learner.work
        report["encoder_work"] = encoder.work
        report["predictor_work"] = {key: p.work for key, p in predictors.items()}
        if binding is not None:
            report["binding_work"], report["binding_receipt"] = binding.work, binding.receipt
        live_pin = live_work()
        work_proof = _json(reported_work())

    def observe_fit(point, batch):
        nonlocal fit_proof
        before_step()
        require_source()
        record = dict(
            point=point,
            completed_updates=learner.completed_updates,
            complete=False,
            attempted=False,
            returned=False,
            validated=False,
        )
        fit_records.append(record)
        fit_owners.append(record)
        fit_proof = _json(fit_report)
        refresh()
        save()
        require_source()
        # Keep native output private until the preceding public prefix is checked.
        outcome: dict[str, Any] = {}
        try:
            _training_fit_readout(learner, batch, outcome)
        finally:
            try:
                require_source()
            finally:
                trusted = json.loads(fit_proof)
                trusted["readouts"][-1].update(outcome)
                report["preserved_train_fit"] = trusted
                refresh()
        record.update(outcome, complete=True)
        fit_proof = _json(fit_report)
        report.pop("preserved_train_fit", None)
        save()
        require_source()

    try:
        before_step()
        require_source()
        input_options = {"public_observation_train": True} if public_observation_train else {}
        batch, targets, metadata = training_inputs(encoder, **input_options)
        report["training_data"] = metadata
        if record_train_fit:
            from src.app.clm_temporal_sequence import _feature_pin

            # Why: callback boundaries must not admit a changed TRAIN batch later.
            fit_features = (
                batch,
                _feature_pin(batch),
                (batch.states.clone(), batch.actions.clone()),
                targets,
            )
        binding_options: dict[str, Any] = {}
        if reuse_action_features:
            action_features = FrozenActionFeatures(
                encoder.identity,
                (
                    training_cases(public_observation_train=True)
                    if public_observation_train
                    else training_cases()
                )[0].request.allowed_candidates,
                batch.actions,
            )
            predictors["unchanged"] = CLMDecisionPredictor(
                encoder,
                loaded.state_head,
                loaded.action_head,
                loaded.logit_scale,
                instructions=INSTRUCTIONS,
                head_revision="unchanged-workflow",
                action_features=action_features,
            )
            binding_options["action_features"] = action_features
            report["action_feature_reuse"] = dict(
                source="already_paid_train_candidate_rows",
                rows=5,
                raw_values_per_owner=5 * encoder.identity.embedding_dim,
                logical_bytes_per_owner=20 * encoder.identity.embedding_dim,
                lifetime="fixed_frozen_encoder_and_exact_catalog; no_refresh",
            )
        refresh()

        if record_train_fit:
            observe_fit("before", batch)

        def training_save():
            require_source()
            save()
            require_source()

        fit_catalog_workflow(
            learner, batch, targets, report, before_step=before_step, save=training_save
        )
        training_proof = _json(report["training"])
        if record_train_fit:
            # Training updated paid counters; admit its already-guarded final work.
            refresh()
            observe_fit("after", batch)
        source_state, source_owner = capture(), learner._bundle
        source_rng_owner, source_work_owner = learner._rng, learner._work
        before_binding = learner.work
        binding = CLMCatalogPredictorBinding(
            learner,
            encoder,
            instructions=INSTRUCTIONS,
            head_revision="workflow-after8",
            **binding_options,
        )
        predictors["learned"] = binding.bind()
        # Exactly one observed public snapshot belongs to this binding, not TRAIN.
        expected = dict(before_binding)
        expected["snapshot_attempts"] += 1
        expected["snapshot_returns"] += 1
        if not _equal(learner.work, expected):
            raise ValueError("binding work differs from its single observed snapshot")
        source_work = expected
        episode_proof = _json(report["episodes"])
        refresh()
        require_source()
        save()
        require_source()
        for control, predictor in predictors.items():
            if learned_only and control != "learned":
                continue
            for index, (scenario, concept) in enumerate(DEVELOPMENT):
                require_source()
                report["pending_episode"] = dict(
                    control=control, scenario=scenario, returned=False, effects_unknown=True
                )
                refresh()
                save()
                require_source()

                def predict(request):
                    nonlocal attempts
                    attempts += 1
                    report["logical_prediction_attempts"] = attempts
                    if attempts > 36:
                        raise ValueError("workflow typed request allowance exhausted")
                    before_step()
                    require_source()
                    try:
                        return predictor(request)
                    finally:
                        refresh()
                        save()
                        require_source()

                episode = run_workflow_episode(
                    WorkflowState(f"development-ticket-{index}", concept),
                    predict,
                    capabilities=WorkflowCapabilities(ACTION_IDS),
                    max_steps=6,
                    scenario=scenario,
                )
                require_source()
                report["episodes"].append(
                    dict(
                        control=control,
                        role="outer_selection",
                        source_group="development-workflow-traces",
                        source_updates=0 if control == "unchanged" else PASSES,
                        episode=episode,
                    )
                )
                episode_proof = _json(report["episodes"])
                report["pending_episode"] = None
                refresh()
                save()
                require_source()
                if not episode["complete"]:
                    raise RuntimeError("workflow episode contains uncertain paid work")
        report["summary"] = _summary(report["episodes"])
        if learned_only:
            report["summary"].pop("unchanged")
        report["complete"] = True
    except Exception as error:
        if fit_proof and "preserved_train_fit" not in report:
            report["preserved_train_fit"] = json.loads(fit_proof)
        if episode_proof:
            report["preserved_episode_prefix"] = json.loads(episode_proof)
        if training_proof:
            report["preserved_training_prefix"] = json.loads(training_proof)
        report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        raise
    finally:
        refresh()
        save()
    return report


def run_case(
    model_dir,
    head,
    report,
    *,
    save=lambda: None,
    reuse_action_features=False,
    record_train_fit=False,
    public_observation_train=False,
    learned_only=False,
    linear_fit_control=False,
    linear_context_updates=False,
):
    _admit_context_options(
        linear_context_updates,
        public_observation_train,
        (linear_fit_control, reuse_action_features, record_train_fit, learned_only),
    )
    if type(linear_fit_control) is not bool:
        raise ValueError("linear_fit_control must be an exact bool")
    if linear_fit_control and any(
        type(value) is not bool or value
        for value in (reuse_action_features, record_train_fit, learned_only)
    ):
        raise ValueError("linear-fit-control cannot combine with contrastive-only workflow options")
    if type(record_train_fit) is not bool:
        raise ValueError("TRAIN fit recording must be an explicit boolean")
    if type(public_observation_train) is not bool:
        raise ValueError("public_observation_train must be an exact bool")
    if type(learned_only) is not bool:
        raise ValueError("learned_only must be an exact bool")
    import torch
    from src.adapters.clm_decision import make_clm_head

    torch.set_num_threads(1 if linear_context_updates else 2)
    started = monotonic()
    report.update(complete=False, events=[])
    with ProcessRssSampler(interval_seconds=0.02) as sampler:

        def boundary():
            if monotonic() - started > 540:
                raise ValueError("workflow inner wall allowance exhausted")
            rss = sampler.sample()
            if rss is None or rss > 22 * 1024**3:
                raise ValueError("workflow RSS unavailable or allowance exceeded")

        def progress(event):
            report["events"].append(dict(elapsed_seconds=monotonic() - started, **event))
            save()

        try:
            loaded = load_local_clm(
                model_dir, head, head_factory=make_clm_head, before_step=boundary, progress=progress
            )
            extra = {} if not reuse_action_features else {"reuse_action_features": True}
            if record_train_fit:
                extra["record_train_fit"] = True
            if public_observation_train:
                extra["public_observation_train"] = True
            if learned_only:
                extra["learned_only"] = True
            if linear_fit_control:
                extra["linear_fit_control"] = True
            if linear_context_updates:
                extra["linear_context_updates"] = True
            run_loaded(loaded, report, before_step=boundary, save=save, **extra)
        finally:
            report.update(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-action-features", action="store_true")
    parser.add_argument("--record-train-fit", action="store_true")
    parser.add_argument("--public-observation-train", action="store_true")
    parser.add_argument("--learned-only", action="store_true")
    parser.add_argument("--linear-fit-control", action="store_true")
    parser.add_argument("--linear-context-updates", action="store_true")
    args = parser.parse_args()
    try:
        _admit_context_options(
            args.linear_context_updates,
            args.public_observation_train,
            (
                args.linear_fit_control,
                args.reuse_action_features,
                args.record_train_fit,
                args.learned_only,
            ),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.linear_fit_control and (
        args.reuse_action_features or args.record_train_fit or args.learned_only
    ):
        parser.error("--linear-fit-control cannot combine with contrastive-only workflow options")
    with args.output.open("x", encoding="utf8") as target:
        target.write("{}\n")
    report: dict[str, Any] = {}

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    try:
        extra = {} if not args.reuse_action_features else {"reuse_action_features": True}
        if args.record_train_fit:
            extra["record_train_fit"] = True
        if args.public_observation_train:
            extra["public_observation_train"] = True
        if args.learned_only:
            extra["learned_only"] = True
        if args.linear_fit_control:
            extra["linear_fit_control"] = True
        if args.linear_context_updates:
            extra["linear_context_updates"] = True
        run_case(args.model_dir, args.head, report, save=save, **extra)
    except Exception as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}"[:512])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
