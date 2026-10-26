"""Released frozen CLM features on the existing changing-tool/rule task.

Only the head learns. Fixed TRAIN/development records, receipts, native objective
and post-TRAIN metrics reuse existing ports. Optional disposable reserved probes
have separate consent and budgets; no downloads or final evaluation inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.run_clm_head_adaptation import _control_readout
from scripts.run_temporal_decision_reference import (
    _accuracy,
    _annotate,
    _cases,
    _contrasts,
    _metric_difference,
)
from src.app.decision_reference_evaluation import evaluate_probability_reference
from src.adapters.clm_checkpoint_probes import (
    _reserved_probe_metrics,
    build_reserved_probe_batches,
    make_reserved_probe_hook,
    score_reserved_probes as _score_reserved_probes,
)
from src.core.temporal_clm_probe import PROBE_QUERIES
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_linear_catalog import (
    OBJECTIVE as LINEAR_OBJECTIVE,
    CONTROL_KIND,
    linear_catalog_work,
    uniform_linear_controls,
)
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
    _exact,
)
from src.core.temporal_decision_sequence import (
    CATALOG,
    CHECKPOINTS,
    PHASES,
    VOCABULARY,
    policy_for_phase,
    temporal_examples,
)
from src.core.temporal_catalog_support import SUPPORT_QUERY, catalog_training_examples
from src.core.typed_decision import ProbabilityDecisionResult, validate_probability_decision_result
from src.infra.clm_local_loader import load_local_clm
from src.shared.process_memory import ProcessRssSampler

INSTRUCTIONS = "Choose the ticket workflow tool."


def feature_texts(*, include_catalog_support=False, include_reserved_probe=False):
    if type(include_reserved_probe) is not bool:
        raise ValueError("reserved probe selection must be an exact boolean")
    if include_reserved_probe and include_catalog_support:
        raise ValueError("reserved probes and catalog support exceed the encoder sequence bound")
    queries = (*VOCABULARY, *(concept + " please" for concept in VOCABULARY))
    if include_catalog_support:
        queries += (SUPPORT_QUERY,)
    if include_reserved_probe:
        queries += PROBE_QUERIES
    return tuple(query + "\n\n" + INSTRUCTIONS for query in queries), tuple(
        candidate.description for candidate in CATALOG
    )


def build_feature_batches(identity, features, *, catalog_support="none"):
    """Bind exact text-order features to TRAIN pairs and development choices."""
    import torch

    _exact(identity, FrozenEncoderIdentity)
    catalog_training_examples("after_A", catalog_support)
    count = 12 if catalog_support == "none" else 13
    action_start = count - 4
    if (
        type(features) is not torch.Tensor
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.layout != torch.strided
        or tuple(features.shape) != (count, identity.embedding_dim)
        or not torch.isfinite(features).all().item()
    ):
        raise ValueError("exact finite CPU float32 recipe-bound text features required")
    features = features.detach().clone()
    ids = tuple(candidate.candidate_id for candidate in CATALOG)
    train = []
    for phase in PHASES:
        for exposure in range(4):
            examples = catalog_training_examples(phase, catalog_support, exposure=exposure)
            codes = tuple(ids.index(example.target_id) for example in examples)
            states = features[
                [
                    (
                        8
                        if example.request.query == SUPPORT_QUERY
                        else VOCABULARY.index(example.request.query)
                    )
                    for example in examples
                ]
            ]
            actions = features[[action_start + code for code in codes]]
            train.append((FrozenEmbeddingBatch(identity, states, actions), CLMTargets(codes)))
    choices = {}
    for _, phase in CHECKPOINTS:
        examples = temporal_examples(phase, "outer_selection")
        candidates = examples[0].request.allowed_candidates
        choices[phase] = FrozenChoiceBatch(
            identity,
            features[4:8].clone(),
            features[
                [action_start + ids.index(candidate.candidate_id) for candidate in candidates]
            ],
        )
    return tuple(train), choices


def build_train_choices(train_batches, choice_batches):
    """Rank actual phase TRAIN states against all currently active candidates."""
    return {
        phase: FrozenChoiceBatch(
            train_batches[4 * index][0].identity,
            train_batches[4 * index][0].states.detach().clone(),
            choice_batches[phase].actions.detach().clone(),
            role="train",
        )
        for index, phase in enumerate(PHASES)
    }


def build_catalog_training_batches(identity, features, *, catalog_support="none"):
    """Bind the declared TRAIN recipe to its complete ordered action bank."""
    paired, choices = build_feature_batches(identity, features, catalog_support=catalog_support)
    batches = []
    for index, (batch, _) in enumerate(paired):
        phase, exposure = PHASES[index // 4], index % 4
        examples = catalog_training_examples(phase, catalog_support, exposure=exposure)
        ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
        targets = CLMCatalogTargets(ids, tuple(ids.index(row.target_id) for row in examples))
        states = batch.states.detach().clone()
        actions = choices[phase].actions.detach().clone()
        batches.append((FrozenCatalogTrainingBatch(identity, states, actions, ids), targets))
    return tuple(batches), choices


def _train_probability_rows(value, *, rows=3):
    if type(value) is not list or len(value) != rows:
        raise ValueError("recipe-bound cached TRAIN query probability rows required")
    for row in value:
        if (
            type(row) is not list
            or len(row) != 3
            or any(
                type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in row
            )
            or not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise ValueError("complete finite normalized three-candidate TRAIN rows required")
    return value


def _train_margin_row(example, values):
    candidates = tuple(item.candidate_id for item in example.request.allowed_candidates)
    target = candidates.index(example.target_id)
    selected = max(range(3), key=values.__getitem__)
    alternative = max(value for index, value in enumerate(values) if index != target)
    obsolete = example.obsolete_target_id
    available = obsolete in candidates
    obsolete_probability = values[candidates.index(obsolete)] if available else None
    return dict(
        concept=example.concept,
        query=example.request.query,
        candidate_ids=list(candidates),
        probabilities=list(values),
        target_id=example.target_id,
        selected_id=candidates[selected],
        correct=selected == target,
        target_probability=values[target],
        best_alternative_probability=alternative,
        target_minus_best_alternative=values[target] - alternative,
        obsolete_target_id=obsolete,
        obsolete_target_available=available,
        obsolete_target_probability=obsolete_probability,
        target_minus_obsolete=(
            None if obsolete_probability is None else values[target] - obsolete_probability
        ),
    )


def score_train_choices(report):
    """Score cached exact TRAIN templates after the fixed dose; no model calls."""
    if _catalog_report(report):
        _require_catalog_completed(report)
    mode = report.get("catalog_support", "none")
    examples_by_phase = {phase: catalog_training_examples(phase, mode) for phase in PHASES}
    training_pairs = 4 * sum(len(examples) for examples in examples_by_phase.values())
    if (
        report.get("sequence_complete") is not True
        or report["stream"]["completed_updates"] != 12
        or report["stream"]["training_pairs"] != training_pairs
        or len(report["rows"]) != 12
        or report["learner_work"]["commits"] != 12
        or report["learner_work"]["unreturned_work_unknown"]
        or type(report.get("train_raw_readouts")) is not list
        or len(report["train_raw_readouts"]) != 3
    ):
        raise ValueError("complete fixed TRAIN stream and three diagnostic readouts required")
    expected = dict(
        prediction_attempts=8,
        prediction_returns=8,
        choice_prediction_attempts=8,
        choice_prediction_returns=8,
        prediction_rows_attempted=20
        + sum(len(examples) for examples in examples_by_phase.values()),
        choice_candidate_rows_attempted=24,
    )
    if any(report["learner_work"][key] != value for key, value in expected.items()):
        raise ValueError("TRAIN diagnostic paid readout counters differ from the fixed schedule")
    points = []
    for raw, phase, update in zip(report["train_raw_readouts"], PHASES, (4, 8, 12), strict=True):
        if (
            type(raw) is not dict
            or set(raw) != {"phase", "completed_updates", "probabilities"}
            or raw["phase"] != phase
            or type(raw["completed_updates"]) is not int
            or raw["completed_updates"] != update
        ):
            raise ValueError("TRAIN diagnostic phase or paid update boundary changed")
        examples = examples_by_phase[phase]
        rows = _train_probability_rows(raw["probabilities"], rows=len(examples))
        points.append(
            dict(
                phase=phase,
                completed_updates=update,
                rows=[
                    _train_margin_row(example, values)
                    for example, values in zip(examples, rows, strict=True)
                ],
            )
        )
    return points


def _score_cached(probabilities, examples):
    if type(probabilities) is not list or len(probabilities) != 4:
        raise ValueError("four cached query probability rows required")
    originals = {example.request.request_id: example.request for example in examples}
    rows = {
        example.request.request_id: row
        for example, row in zip(examples, probabilities, strict=True)
    }
    calls = 0

    def predict(request):
        nonlocal calls
        original = originals[request.request_id]
        if request.query != original.query or request.allowed_candidates not in (
            original.allowed_candidates,
            tuple(reversed(original.allowed_candidates)),
        ):
            raise ValueError("cached readout request binding changed")
        values = rows[request.request_id]
        if type(values) is not list or len(values) != len(original.allowed_candidates):
            raise ValueError("cached probability columns differ from active candidates")
        by_id = dict(
            zip(
                (candidate.candidate_id for candidate in original.allowed_candidates),
                values,
                strict=True,
            )
        )
        ordered = tuple(
            (candidate.candidate_id, by_id[candidate.candidate_id])
            for candidate in request.allowed_candidates
        )
        winner = max(ordered, key=lambda pair: pair[1])
        confidence = winner[1] - sum(value for key, value in ordered if key != winner[0]) / (
            len(ordered) - 1
        )
        calls += 1
        return ProbabilityDecisionResult(
            request.request_id, winner[0], ordered, reported_confidence=confidence
        )

    evaluation = evaluate_probability_reference(_cases(examples), predict=predict)
    evaluation.update(
        method="cached_clm_temporal_probabilities",
        predictor_behavior="scalar_decode_of_previously_returned_head_output",
        qualification="development suffix split; no final evaluation or native serving parity",
    )
    return _annotate(evaluation, examples), calls


def _catalog_report(report):
    work = report.get("learner_work")
    return "training_objective" in report or (
        type(work) is dict and any(key.startswith("catalog_") for key in work)
    )


def _require_catalog_completed(report):
    from src.app.clm_reserved_probe import _expected_work, _require_work

    linear = report.get("training_objective") == LINEAR_OBJECTIVE
    objective = LINEAR_OBJECTIVE if linear else CATALOG_OBJECTIVE
    if report.get("training_objective") != objective:
        raise ValueError("explicit completed catalog objective required")
    if linear and (
        report.get("control_kind") != CONTROL_KIND
        or report.get("control_probabilities") != uniform_linear_controls()
    ):
        raise ValueError("linear completed report requires original untrained uniform control")
    if linear and (
        report.get("catalog_support", "none") != "none"
        or any(
            key in report
            for key in (
                "typed_development",
                "train_parameter_update_scope",
                "reserved_probes",
                "training_replay_consent",
            )
        )
    ):
        raise ValueError("linear completed report contains unsupported native extensions")
    if "train_parameter_update_scope" in report:
        scope = report["train_parameter_update_scope"]
        keys = (
            "optimized_parameter_elements",
            "fixed_parameter_elements",
            "total_parameter_elements",
        )
        if (
            type(scope) is not dict
            or set(scope) != {"name", "fixed_parameters_unchanged", *keys}
            or type(scope["name"]) is not str
            or scope["name"] != "state_only"
            or scope["fixed_parameters_unchanged"] is not True
            or any(type(scope[key]) is not int or scope[key] <= 0 for key in keys)
            or scope[keys[0]] + scope[keys[1]] != scope[keys[2]]
            or scope[keys[2]] > 32000000
        ):
            raise ValueError("completed state-only TRAIN scope or parameter counts changed")
    mode = report.get("catalog_support", "none")
    phase_rows = tuple(len(catalog_training_examples(phase, mode)) for phase in PHASES)
    training_rows = 4 * sum(phase_rows)
    diagnostic = "train_raw_readouts" in report
    expected = _expected_work(
        12, 8 if diagnostic else 5, 20 + sum(phase_rows) if diagnostic else 20
    )
    expected.update(
        training_rows_attempted=training_rows,
        catalog_candidate_rows_attempted=36,
        catalog_logit_values_attempted=3 * training_rows,
    )
    if "typed_development" in report:
        _require_typed_return(report)
        expected.update(snapshot_attempts=1, snapshot_returns=1)
    if linear:
        expected = linear_catalog_work(
            12, training_rows, 8 if diagnostic else 5, 20 + sum(phase_rows) if diagnostic else 20
        )
    _require_work(report.get("learner_work"), expected)
    stream = report.get("stream")
    if (
        type(stream) is not dict
        or type(stream.get("completed_updates")) is not int
        or stream.get("completed_updates") != 12
        or type(stream.get("training_pairs")) is not int
        or stream.get("training_pairs") != training_rows
        or type(stream.get("receipts")) is not list
        or len(stream["receipts"]) != 12
        or type(report.get("rows")) is not list
        or len(report["rows"]) != 12
    ):
        raise ValueError("twelve completed catalog TRAIN receipts and rows required")
    for index, (receipt, row) in enumerate(zip(stream["receipts"], report["rows"], strict=True)):
        phase, exposure = PHASES[index // 4], index % 4
        sample = f"clm-{phase}-{exposure}"
        metadata = dict(
            sample_id=sample,
            episode_id="clm-temporal",
            event_id="label-" + sample,
            actor_version=phase,
            learner_version="clm-temporal",
            observed_at=2 * index,
            arrived_at=2 * index + 1,
            applied_at=2 * index + 1,
            update_number=index + 1,
        )
        if type(receipt) is not dict or any(
            type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in metadata.items()
        ):
            raise ValueError("catalog receipt identity or clocks changed")
        loss = receipt.get("diagnostic")
        if (
            type(loss) is not dict
            or loss.get("definition") != objective
            or type(loss.get("value")) is not float
            or not math.isfinite(loss["value"])
            or loss["value"] < 0
        ):
            raise ValueError("catalog receipt objective or diagnostic changed")
        examples = catalog_training_examples(phase, mode, exposure=exposure)
        ids = [candidate.candidate_id for candidate in examples[0].request.allowed_candidates]
        if (
            type(row) is not dict
            or row.get("phase") != phase
            or type(row.get("update")) is not int
            or row.get("update") != index + 1
            or type(row.get("exposure")) is not int
            or row.get("exposure") != exposure
            or row.get("candidate_ids") != ids
            or type(row.get("candidate_indices")) is not list
            or any(type(value) is not int for value in row["candidate_indices"])
            or row.get("candidate_indices") != [ids.index(item.target_id) for item in examples]
            or row.get("loss") != loss["value"]
        ):
            raise ValueError("catalog row differs from fixed observed labels or receipt")


def _typed_requests():
    return tuple(row.request for row in temporal_examples("after_return", "outer_selection"))


def _json_record(record):
    return json.loads(json.dumps(asdict(record), allow_nan=False))


def _typed_difference(report, predictions):
    last = report["raw_readouts"][-1]
    if last.get("phase") != "after_return" or last.get("completed_updates") != 12:
        raise ValueError("original completed return cache required")
    cached = last["probabilities"]
    if type(cached) is not list or len(cached) != 4:
        raise ValueError("complete return cache required")
    differences: list[float] = []
    for values, result in zip(cached, predictions, strict=True):
        if (
            type(values) is not list
            or len(values) != 3
            or any(
                type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
                for value in values
            )
        ):
            raise ValueError("bounded cached return probabilities required")
        differences.extend(
            abs(a - pair[1]) for a, pair in zip(values, result["probabilities"], strict=True)
        )
    return max(differences)


def _require_typed_return(report):
    """Admit the extra source snapshot only with the complete typed work prefix."""
    from src.app.clm_reserved_probe import _require_work

    record = report["typed_development"]
    memory_mode = record.get("mode") == "evidence_fallback" if type(record) is dict else False
    if memory_mode:
        from src.app.clm_memory_serving import memory_serving_requests, require_memory_serving

        requests = memory_serving_requests()
    else:
        requests = _typed_requests()
    if (
        type(record) is not dict
        or record.get("complete") is not True
        or record.get("requests") != [_json_record(request) for request in requests]
        or type(record.get("predictions")) is not list
        or len(record["predictions"]) != len(requests)
    ):
        raise ValueError("complete canonical typed development readout required")
    for request, raw in zip(requests, record["predictions"], strict=True):
        if type(raw) is not dict:
            raise ValueError("typed result dictionary required")
        result = ProbabilityDecisionResult(
            **{**raw, "probabilities": tuple(tuple(pair) for pair in raw["probabilities"])}
        )
        validate_probability_decision_result(request, result)
        if [pair[0] for pair in result.probabilities] != [
            candidate.candidate_id for candidate in request.allowed_candidates
        ]:
            raise ValueError("typed candidate order changed")
    if memory_mode:
        require_memory_serving(report, record)
    calls = 1 if memory_mode else 4
    _require_work(
        record.get("predictor_work"),
        dict(
            request_attempts=calls,
            request_returns=calls,
            encoder_attempts=calls,
            encoder_returns=calls,
            head_forward_attempts=2 * calls,
            head_forward_returns=2 * calls,
            projected_rows=4 * calls,
            readout_attempts=calls,
            readout_returns=calls,
            unreturned_work_unknown=False,
        ),
    )
    work, receipt = record.get("binding_work"), record.get("binding_receipt")
    if type(work) is not dict or type(receipt) is not dict:
        raise ValueError("completed binding work and receipt required")
    for key in ("head_parameter_elements_attempted", "snapshot_tensor_elements_copied"):
        if type(work.get(key)) is not int or not 1 <= work[key] <= 2**53 - 1:
            raise ValueError("positive measured binding copy work required")
    _require_work(
        work,
        dict(
            binding_attempts=1,
            binding_returns=1,
            snapshot_attempts=1,
            snapshot_returns=1,
            guard_capture_attempts=3,
            guard_capture_returns=3,
            snapshot_tensor_elements_copied=work["snapshot_tensor_elements_copied"],
            head_copy_attempts=2,
            head_copy_returns=2,
            head_parameter_elements_attempted=work["head_parameter_elements_attempted"],
            head_parameter_elements_returned=work["head_parameter_elements_attempted"],
            scale_copy_elements_attempted=2,
            scale_copy_elements_returned=2,
            unreturned_work_unknown=False,
        ),
    )
    _require_work(receipt.get("work"), work)
    try:
        policy = CLMHeadPolicy(**receipt["policy"])
    except (TypeError, KeyError, ValueError) as error:
        raise ValueError("exact native binding policy required") from error
    if (
        asdict(policy) != receipt["policy"]
        or receipt["policy"] != record.get("source_policy")
        or policy != CLMHeadPolicy(projection_dim=policy.projection_dim, max_updates=12, max_rows=4)
        or (report["identity"]["embedding_dim"] == 4096 and policy.projection_dim != 512)
    ):
        raise ValueError("binding policy differs from original temporal source")
    guard_work = record.get("source_guard_work")
    if type(guard_work) is not dict:
        raise ValueError("paid source guards required")
    guards = len(requests) + 2
    _require_work(
        guard_work,
        dict(
            capture_attempts=guards,
            capture_returns=guards,
            tensor_elements_copied=guards * work["snapshot_tensor_elements_copied"] // 4,
        ),
    )
    if 4 * guard_work["tensor_elements_copied"] != guards * work["snapshot_tensor_elements_copied"]:
        raise ValueError("source guard copy charge differs from binding state")
    rows = 4 * sum(
        len(catalog_training_examples(phase, report.get("catalog_support", "none")))
        for phase in PHASES
    )
    if (
        receipt.get("objective") != CATALOG_OBJECTIVE
        or type(receipt.get("source_completed_updates")) is not int
        or receipt["source_completed_updates"] != 12
        or type(receipt.get("source_generation")) is not int
        or receipt["source_generation"] != 12
        or type(receipt.get("source_training_examples")) is not int
        or receipt["source_training_examples"] != rows
        or receipt.get("identity") != report.get("identity")
        or receipt.get("instructions") != INSTRUCTIONS
        or receipt.get("head_revision") != "temporal-catalog-after-return-12"
    ):
        raise ValueError("binding receipt differs from completed catalog source")
    before, after = record.get("encoder_work_before"), record.get("encoder_work_after")
    keys = set(
        (
            "embedding_attempts",
            "embedding_returns",
            "tokenization_attempts",
            "tokenization_returns",
            "token_ids_visited",
            "retained_token_ids",
            "forward_attempts",
            "forward_returns",
            "sequences_attempted",
            "sequences_returned",
            "padded_tokens_attempted",
            "pooled_values",
            "unreturned_work_unknown",
        )
    )
    if (
        type(before) is not dict
        or type(after) is not dict
        or set(before) != keys
        or set(after) != keys
    ):
        raise ValueError("original and completed encoder work required")
    for key, old in before.items():
        if key == "unreturned_work_unknown":
            if old is not False or after[key] is not False:
                raise ValueError("uncertain encoder work cannot complete typed readout")
        elif (
            type(old) is not int
            or type(after[key]) is not int
            or not 0 <= old <= after[key] <= 2**53 - 1
        ):
            raise ValueError("encoder work must be permanent bounded integer counts")
    for key, delta in dict(
        embedding_attempts=calls,
        embedding_returns=calls,
        forward_attempts=calls,
        forward_returns=calls,
        sequences_attempted=4 * calls,
        sequences_returned=4 * calls,
        pooled_values=4 * calls * report["identity"]["embedding_dim"],
    ).items():
        if after.get(key, -1) - before.get(key, -1) != delta:
            raise ValueError("typed encoder work differs from four paid request batches")
    if before.get("forward_attempts") != 1 or before.get("forward_returns") != 1:
        raise ValueError("typed readout requires the original single feature forward")
    difference = record.get("max_abs_cached_difference")
    if type(difference) is not float or difference != _typed_difference(
        report, record["predictions"][:4]
    ):
        raise ValueError("typed/cache difference changed")


def read_typed_return(learner, encoder, report, *, before_step, save, memory=None):
    """Execute four development requests after TRAIN; never select further work."""
    from src.adapters.clm_catalog_decision import CLMCatalogPredictorBinding, _tensor_elements
    from src.adapters.clm_head_learner import _equal
    from src.app.clm_reserved_probe import _require_work

    before_step()
    if "typed_development" in report or report.get("sequence_complete") is not True:
        raise ValueError("fresh typed readout requires a completed original sequence")
    _require_catalog_completed(report)
    _require_work(learner.work, report["learner_work"])
    if (
        report.get("identity") != asdict(learner.identity)
        or learner.completed_updates != 12
        or learner.generation != 12
        or learner.training_examples != report["stream"]["training_pairs"]
    ):
        raise ValueError("typed source identity differs from completed sequence")
    requests = _typed_requests()
    if memory is not None:
        from src.app.clm_memory_serving import memory_serving_requests
        from src.app.clm_train_evidence_memory import CLMTrainEvidenceMemory

        if type(memory) is not CLMTrainEvidenceMemory or memory._report is not report:
            raise ValueError("typed memory requires the original completed memory owner")
        memory.require_serving_source()
        requests = memory_serving_requests()
    record = dict(
        complete=False,
        requests=[_json_record(request) for request in requests],
        predictions=[],
        encoder_work_before=encoder.work,
        source_policy=asdict(learner.policy),
        source_guard_work=dict(capture_attempts=0, capture_returns=0, tensor_elements_copied=0),
    )
    report["typed_development"] = record
    if memory is not None:
        record.update(
            mode="evidence_fallback",
            memory_source=json.loads(json.dumps(report["train_evidence_memory"], allow_nan=False)),
            unscored_request_ids=["typed-after-return-unseen"],
        )
    binding, predictor, handle, memory_predictor = None, None, None, None
    prediction_proof = "[]"

    def capture():
        work = record["source_guard_work"]
        work["capture_attempts"] += 1
        result = learner._capture()
        work["capture_returns"] += 1
        work["tensor_elements_copied"] += _tensor_elements(vars(result))
        return result

    try:
        try:
            original, bundle = capture(), learner._bundle
            binding = CLMCatalogPredictorBinding(
                learner,
                encoder,
                instructions=INSTRUCTIONS,
                head_revision="temporal-catalog-after-return-12",
            )
            predictor = binding.bind()
            if memory is not None:
                memory_predictor = memory.bind_fallback(predictor)
            bound_work = learner.work

            def require_source():
                if memory is not None:
                    memory.require_serving_source()
                    if (
                        report.get("typed_development") is not record
                        or record["requests"] != [_json_record(request) for request in requests]
                        or json.dumps(record["predictions"], sort_keys=True, allow_nan=False)
                        != prediction_proof
                        or record["memory_source"]
                        != json.loads(json.dumps(report["train_evidence_memory"], allow_nan=False))
                    ):
                        raise ValueError("preceding typed memory return evidence changed")
                learner._check()
                _require_work(learner.work, bound_work)
                if (
                    learner._bundle is not bundle
                    or learner.generation != 12
                    or not _equal(vars(capture()), vars(original))
                ):
                    raise ValueError("live source changed during typed inference")

            def guard(_model, args, kwargs):
                ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
                if (
                    ids.shape != mask.shape
                    or ids.ndim != 2
                    or ids.shape[0] != 4
                    or ids.shape[1] > 96
                    or kwargs.get("use_cache") is not False
                ):
                    raise ValueError("typed request encoder allowance exceeded")
                before_step()

            handle = encoder._model.register_forward_pre_hook(guard, with_kwargs=True)
            for request in requests:
                before_step()
                require_source()
                result = (
                    predictor(request) if memory_predictor is None else memory_predictor(request)
                )
                validate_probability_decision_result(request, result)
                record["predictions"].append(_json_record(result))
                prediction_proof = json.dumps(
                    record["predictions"], sort_keys=True, allow_nan=False
                )
                save()
            record["max_abs_cached_difference"] = _typed_difference(
                report, record["predictions"][:4]
            )
        finally:
            if handle is not None:
                handle.remove()
            record.update(
                binding_work=None if binding is None else binding.work,
                binding_receipt=None if binding is None else binding.receipt,
                predictor_work=None if predictor is None else predictor.work,
                encoder_work_after=encoder.work,
            )
            if memory is not None:
                record.update(
                    memory_binding_work=memory.serving_work,
                    memory_view_report=None
                    if memory_predictor is None
                    else json.loads(json.dumps(memory_predictor.report, allow_nan=False)),
                )
            report["learner_work"] = learner.work
            save()
        require_source()
        record["complete"] = True
        _require_typed_return(report)
    except Exception as error:
        record["complete"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        raise


def score_sequence(report):
    """Reuse fixed temporal metrics on cached observations after all TRAIN work."""
    if _catalog_report(report):
        _require_catalog_completed(report)
    mode = report.get("catalog_support", "none")
    training_pairs = 4 * sum(len(catalog_training_examples(phase, mode)) for phase in PHASES)
    if (
        report.get("sequence_complete") is not True
        or report["stream"]["completed_updates"] != 12
        or report["stream"]["training_pairs"] != training_pairs
        or len(report["rows"]) != 12
        or report["learner_work"]["commits"] != 12
        or report["learner_work"]["unreturned_work_unknown"]
        or len(report["raw_readouts"]) != 4
    ):
        raise ValueError("complete fixed TRAIN stream required before diagnostic scoring")
    checkpoints = []
    calls = 0
    for raw, (update, phase) in zip(report["raw_readouts"], CHECKPOINTS, strict=True):
        if raw["phase"] != phase or raw["completed_updates"] != update:
            raise ValueError("cached phase or paid update boundary changed")
        examples = temporal_examples(phase, "outer_selection")
        learned, count = _score_cached(raw["probabilities"], examples)
        calls += count
        control, count = _score_cached(
            report["control_probabilities"][policy_for_phase(phase)["version"]], examples
        )
        calls += count
        point = dict(phase=phase, update=update, learned=learned, unchanged=control)
        if update:
            frozen = (
                report["raw_readouts"][1]["probabilities"]
                if update == 4
                else report["frozen_A_probabilities"]
            )
            point["frozen_A"], count = _score_cached(frozen, examples)
            calls += count
        checkpoints.append(point)
    return dict(
        checkpoints=checkpoints, contrasts=_contrasts(checkpoints), cached_typed_decodes=calls
    )


def _require_comparison_arm(mode, arm):
    if type(arm) is dict and "typed_development" in arm:
        raise ValueError("typed development readout is unavailable to recipe comparison")
    if (
        type(arm) is not dict
        or arm.get("catalog_support") != mode
        or arm.get("sequence_complete") is not True
        or arm["stream"]["completed_updates"] != 12
        or arm["stream"]["training_pairs"] != 44
        or len(arm["rows"]) != 12
        or arm["learner_work"]["unreturned_work_unknown"]
    ):
        raise ValueError("both fixed TRAIN trajectories must complete before comparison scoring")
    if _catalog_report(arm):
        _require_catalog_completed(arm)
    expected = dict.fromkeys(
        (
            "train_attempts",
            "loss_forward_attempts",
            "loss_forward_returns",
            "backward_attempts",
            "backward_returns",
            "optimizer_attempts",
            "optimizer_returns",
            "scheduler_attempts",
            "scheduler_returns",
            "commits",
        ),
        12,
    )
    expected.update(
        dict.fromkeys(
            (
                "prediction_attempts",
                "prediction_returns",
                "choice_prediction_attempts",
                "choice_prediction_returns",
            ),
            8,
        )
    )
    expected.update(
        training_rows_attempted=44, prediction_rows_attempted=31, choice_candidate_rows_attempted=24
    )
    if any(arm["learner_work"].get(key) != value for key, value in expected.items()):
        raise ValueError("both recipe paid work ledgers must match before any scoring")
    if len(arm["raw_readouts"]) != 4 or len(arm.get("train_raw_readouts", ())) != 3:
        raise ValueError("both recipe readout caches must be complete before any scoring")
    for raw, (update, phase) in zip(arm["raw_readouts"], CHECKPOINTS, strict=True):
        if raw["phase"] != phase or raw["completed_updates"] != update:
            raise ValueError("recipe development cache phase differs")
        _train_probability_rows(raw["probabilities"], rows=4)
    for raw, phase, update in zip(arm["train_raw_readouts"], PHASES, (4, 8, 12), strict=True):
        if raw["phase"] != phase or raw["completed_updates"] != update:
            raise ValueError("recipe TRAIN cache phase differs")
        _train_probability_rows(
            raw["probabilities"], rows=len(catalog_training_examples(phase, mode))
        )
    _train_probability_rows(arm["frozen_A_probabilities"], rows=4)
    for version in ("v1", "v2"):
        _train_probability_rows(arm["control_probabilities"][version], rows=4)


def score_catalog_support_comparison(report):
    """Score both completed recipes together; never expose a result to later TRAIN."""
    arms = report.get("arms")
    if type(arms) is not dict or tuple(arms) != ("repeat", "route"):
        raise ValueError("ordered repeat/route comparison arms required")
    for mode, arm in arms.items():
        _require_comparison_arm(mode, arm)
    repeat, route = arms["repeat"], arms["route"]
    if _catalog_report(repeat) != _catalog_report(route):
        raise ValueError("comparison arms must use the same training objective")
    parity = dict(
        baseline_and_A=repeat["raw_readouts"][:2] == route["raw_readouts"][:2],
        frozen_A=repeat["frozen_A_probabilities"] == route["frozen_A_probabilities"],
        A_losses=repeat["rows"][:4] == route["rows"][:4],
        A_receipts=repeat["stream"]["receipts"][:4] == route["stream"]["receipts"][:4],
        controls=repeat["control_probabilities"] == route["control_probabilities"],
    )
    if not all(parity.values()):
        raise ValueError("identical baseline/A prefix or shared control differs between recipes")
    scores = {
        mode: dict(**score_sequence(arm), train_catalog_diagnostics=score_train_choices(arm))
        for mode, arm in arms.items()
    }
    contrasts = []
    for left, right in zip(
        scores["route"]["checkpoints"], scores["repeat"]["checkpoints"], strict=True
    ):
        a, b = left["learned"], right["learned"]
        concepts = {
            x["concept"]: dict(
                correct=int(x["correct"]) - int(y["correct"]),
                target_probability=x["target_probability"] - y["target_probability"],
            )
            for x, y in zip(a["cases"], b["cases"], strict=True)
        }
        contrasts.append(
            dict(
                phase=left["phase"],
                update=left["update"],
                route_minus_repeat=dict(
                    **_metric_difference(a["metrics"], b["metrics"]),
                    current_old_domain_accuracy=_accuracy(a, ("invoice", "legacy", "password"))
                    - _accuracy(b, ("invoice", "legacy", "password")),
                    new_domain_accuracy=_accuracy(a, ("parcel",)) - _accuracy(b, ("parcel",)),
                    obsolete_rule_error_count=a["temporal"]["obsolete_rule_errors"]
                    - b["temporal"]["obsolete_rule_errors"],
                    by_concept=concepts,
                ),
            )
        )
    return dict(arm_scores=scores, shared_A_equal=parity, recipe_contrasts=contrasts)


def score_reserved_probes(report):
    """Score only fully completed cached probes; this never invokes a learner."""
    return _score_reserved_probes(
        report, score_group=_reserved_probe_metrics, metric_difference=_metric_difference
    )


def _run_head_stream(
    loaded,
    train,
    choices,
    report,
    *,
    mode,
    diagnostics,
    before_step,
    save,
    reserved_probe_data: tuple[Any, Any] | None = None,
    catalog_training=False,
    typed_encoder=None,
    probe_rehearsal=False,
    probe_marker_control=False,
    probe_train_readouts=False,
    probe_state_only=False,
    train_state_only=False,
    train_evidence_memory=False,
    paired_replay_candidate=False,
    probe_linear_fit_control=False,
    train_linear_update=False,
    linear_actions=None,
):
    from src.adapters.clm_decision import _geometry
    from src.adapters.clm_head_learner import CLMHeadLearner
    from src.app.clm_temporal_sequence import run_temporal_stream

    _admit_linear_training(
        train_linear_update,
        catalog_training=catalog_training,
        catalog_support=mode,
        reserved_task_probes=reserved_probe_data is not None,
        typed_development_readout=typed_encoder is not None,
        probe_rehearsal=probe_rehearsal,
        probe_marker_control=probe_marker_control,
        probe_train_readouts=probe_train_readouts,
        probe_state_only=probe_state_only,
        train_state_only=train_state_only,
        train_evidence_memory=train_evidence_memory,
        paired_replay_candidate=paired_replay_candidate,
        probe_linear_fit_control=probe_linear_fit_control,
    )
    if train_linear_update:
        from src.adapters.clm_linear_catalog_learner import run_linear_catalog_stream

        return run_linear_catalog_stream(
            loaded.identity,
            linear_actions,
            train,
            choices,
            report,
            train_choices=build_train_choices(train, choices) if diagnostics else None,
            before_step=before_step,
            save=save,
        )

    _admit_linear_probes(
        probe_linear_fit_control,
        catalog_training=catalog_training,
        reserved_task_probes=reserved_probe_data is not None,
        catalog_support=mode,
        typed_development_readout=typed_encoder is not None,
        probe_rehearsal=probe_rehearsal,
        probe_marker_control=probe_marker_control,
        probe_train_readouts=probe_train_readouts,
        probe_state_only=probe_state_only,
        train_state_only=train_state_only,
        train_evidence_memory=train_evidence_memory,
        paired_replay_candidate=paired_replay_candidate,
    )

    _admit_paired_replay(
        paired_replay_candidate,
        catalog_training=catalog_training,
        catalog_support=mode,
        reserved_task_probes=reserved_probe_data is not None,
        typed_development_readout=typed_encoder is not None,
        train_catalog_diagnostics=diagnostics,
    )

    if type(train_state_only) is not bool or (train_state_only and not catalog_training):
        raise ValueError("state-only TRAIN requires catalog training")
    if train_state_only and typed_encoder is not None:
        raise ValueError("typed binding does not yet admit state-only TRAIN snapshots")
    if type(train_evidence_memory) is not bool or (
        train_evidence_memory
        and (not catalog_training or mode != "none" or reserved_probe_data is not None)
    ):
        raise ValueError(
            "TRAIN evidence memory requires one none-recipe catalog stream without probes"
        )
    state, action = deepcopy(loaded.state_head), deepcopy(loaded.action_head)
    state.requires_grad_(True)
    action.requires_grad_(True)
    projection = _geometry(state, loaded.identity.embedding_dim)
    if _geometry(action, loaded.identity.embedding_dim) != projection or (
        loaded.identity.embedding_dim == 4096 and projection != 512
    ):
        raise ValueError("matching native projection geometry required before training")
    learner_kind = CLMHeadLearner
    if catalog_training:
        from src.adapters.clm_catalog_learner import CLMCatalogLearner

        learner_kind = CLMCatalogLearner
        if train_state_only:
            from src.adapters.clm_state_only_catalog_learner import CLMStateOnlyCatalogLearner

            learner_kind = CLMStateOnlyCatalogLearner
    learner = learner_kind(
        state,
        action,
        loaded.logit_scale.detach().clone(),
        identity=loaded.identity,
        policy=CLMHeadPolicy(projection_dim=projection, max_updates=12, max_rows=4),
    )
    del state, action
    try:
        scope = None
        if train_state_only:
            from src.app.clm_reserved_probe import _parameter_update_metadata

            # Same explicit scope vocabulary; this reads the TRAIN owner's values.
            scope = _parameter_update_metadata(learner)
            report["train_parameter_update_scope"] = scope.copy()
        extra: dict[str, Any] = {}
        memory = None
        if catalog_training:
            extra["catalog_training"] = True
        replay_history = []
        if paired_replay_candidate:
            from src.app.clm_replay_candidate import CLMReplayRecord

            def remember(stage, read):
                if stage == "completed":
                    origin = read()
                    if origin.completed_updates in (5, 9):
                        replay_history.append(
                            CLMReplayRecord(origin.source, origin.label, origin.receipt, "v2")
                        )

            extra["allow_train_replay"] = True
            extra["native_observer"] = remember
        if train_evidence_memory:
            from src.app.clm_train_evidence_memory import CLMTrainEvidenceMemory
            from src.core.experience import ExperiencePermissions

            # The raw mirrors and projected stream each receive original consent
            # at birth; copying their features cannot create reuse permission.
            memory = CLMTrainEvidenceMemory(
                report, permissions=ExperiencePermissions(training=True, replay=True)
            )
            extra["allow_train_replay"] = True
            extra["on_checkpoint"] = memory
        if probe_rehearsal:
            extra["allow_train_replay"] = True
        if reserved_probe_data is not None:
            probe_options: dict[str, Any] = {}
            if catalog_training:
                probe_options["catalog_training"] = True
            if probe_linear_fit_control:
                probe_options["linear_fit_control"] = True
            if probe_state_only:
                probe_options["parameter_scope"] = "state_only"
            if probe_marker_control:
                probe_options["marker_control"] = True
            if probe_rehearsal or probe_train_readouts:
                arrived = {phase: train[4 * index] for index, phase in enumerate(PHASES)}
                if probe_rehearsal:
                    probe_options["rehearsal_training"] = arrived
                if probe_train_readouts:
                    probe_options["readout_training"] = arrived
            extra["on_checkpoint"] = make_reserved_probe_hook(
                *reserved_probe_data, report, before_step=before_step, save=save, **probe_options
            )
        run_temporal_stream(
            learner,
            train,
            choices,
            report,
            catalog_support=mode,
            train_choice_batches=build_train_choices(train, choices) if diagnostics else None,
            before_step=before_step,
            save=save,
            **extra,
        )
        if report["sequence_complete"] is not True or learner.completed_updates != 12:
            raise ValueError("preceding recipe did not complete its original allowance")
        if paired_replay_candidate:
            from src.adapters.clm_replay_candidate import CLMReplayCandidate
            from src.app.clm_replay_candidate import run_clm_replay_candidate
            from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget

            if len(replay_history) != 2:
                raise ValueError("both original current-policy TRAIN receipts required")
            if "paired_replay_candidate" in report:
                raise ValueError("refusing to replace original paired replay report")
            child: dict[str, Any] = {}
            report["paired_replay_candidate"] = child
            run_clm_replay_candidate(
                learner,
                tuple(replay_history),
                (0, 1, 0, 1, 0, 1),
                choices["after_return"],
                child,
                policy=replace(learner.policy, max_updates=6, max_restores=0),
                source_version="clm-temporal",
                current_policy="v2",
                at=23,
                candidate_factory=CLMReplayCandidate,
                budget=ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0),
                replay_budget=ToyBudgetSession(
                    ToyExecutionBudget(max_replay_examples=18), lambda: 0.0
                ),
                before_step=before_step,
                save=save,
            )
        if memory is not None:
            memory.finish()
        if train_state_only and (
            report.get("train_parameter_update_scope") != scope
            or _parameter_update_metadata(learner) != scope
        ):
            raise ValueError("state-only TRAIN scope or fixed values changed during stream")
        if typed_encoder is not None:
            if memory is None:
                read_typed_return(
                    learner, typed_encoder, report, before_step=before_step, save=save
                )
            else:
                read_typed_return(
                    learner,
                    typed_encoder,
                    report,
                    before_step=before_step,
                    save=save,
                    memory=memory,
                )
    finally:
        report["learner_work"] = learner.work
        save()


def _control_matches(torch, loaded, original):
    current = (
        loaded.state_head.state_dict(),
        loaded.action_head.state_dict(),
        loaded.logit_scale.detach(),
    )
    return all(
        torch.equal(original[i][key], current[i][key]) for i in (0, 1) for key in original[i]
    ) and torch.equal(original[2], current[2])


def _admit_paired_replay(enabled, **options):
    if type(enabled) is not bool or (
        enabled
        and any(
            value != "none" if name == "catalog_support" else bool(value)
            for name, value in options.items()
        )
    ):
        raise ValueError("paired replay requires one original paired stream without other options")


def _admit_linear_training(
    enabled,
    *,
    catalog_training,
    catalog_support="none",
    train_catalog_diagnostics=False,
    **conflicts,
):
    if type(enabled) is not bool or (
        enabled
        and (catalog_training is not True or catalog_support != "none" or any(conflicts.values()))
    ):
        raise ValueError(
            "linear TRAIN requires one none-recipe catalog stream without native extensions"
        )


def _admit_linear_probes(
    enabled, *, catalog_training, reserved_task_probes, catalog_support="none", **conflicts
):
    if type(enabled) is not bool or (
        enabled
        and (
            catalog_training is not True
            or reserved_task_probes is not True
            or catalog_support != "none"
            or any(type(value) is not bool or value for value in conflicts.values())
        )
    ):
        raise ValueError(
            "linear probes require catalog reserved probes with the original none recipe and no native probe options"
        )


def run_case(
    model_dir: Path,
    head: Path,
    report: dict[str, Any],
    *,
    train_catalog_diagnostics=False,
    catalog_support="none",
    compare_catalog_support=False,
    reserved_task_probes=False,
    catalog_training=False,
    typed_development_readout=False,
    probe_rehearsal=False,
    probe_marker_control=False,
    probe_train_readouts=False,
    probe_state_only=False,
    train_state_only=False,
    train_evidence_memory=False,
    paired_replay_candidate=False,
    probe_linear_fit_control=False,
    save=lambda: None,
    train_linear_update=False,
):
    _admit_linear_training(
        train_linear_update,
        catalog_training=catalog_training,
        catalog_support=catalog_support,
        compare_catalog_support=compare_catalog_support,
        reserved_task_probes=reserved_task_probes,
        typed_development_readout=typed_development_readout,
        probe_rehearsal=probe_rehearsal,
        probe_marker_control=probe_marker_control,
        probe_train_readouts=probe_train_readouts,
        probe_state_only=probe_state_only,
        train_state_only=train_state_only,
        train_evidence_memory=train_evidence_memory,
        paired_replay_candidate=paired_replay_candidate,
        probe_linear_fit_control=probe_linear_fit_control,
    )
    _admit_linear_probes(
        probe_linear_fit_control,
        catalog_training=catalog_training,
        reserved_task_probes=reserved_task_probes,
        catalog_support=catalog_support,
        compare_catalog_support=compare_catalog_support,
        typed_development_readout=typed_development_readout,
        probe_rehearsal=probe_rehearsal,
        probe_marker_control=probe_marker_control,
        probe_train_readouts=probe_train_readouts,
        probe_state_only=probe_state_only,
        train_state_only=train_state_only,
        train_evidence_memory=train_evidence_memory,
        paired_replay_candidate=paired_replay_candidate,
    )
    _admit_paired_replay(
        paired_replay_candidate,
        catalog_training=catalog_training,
        catalog_support=catalog_support,
        compare_catalog_support=compare_catalog_support,
        reserved_task_probes=reserved_task_probes,
        typed_development_readout=typed_development_readout,
        probe_rehearsal=probe_rehearsal,
        probe_marker_control=probe_marker_control,
        probe_train_readouts=probe_train_readouts,
        probe_state_only=probe_state_only,
        train_state_only=train_state_only,
        train_evidence_memory=train_evidence_memory,
        train_catalog_diagnostics=train_catalog_diagnostics,
    )
    catalog_training_examples("after_A", catalog_support)
    if type(train_evidence_memory) is not bool or (
        train_evidence_memory
        and (
            not catalog_training
            or catalog_support != "none"
            or reserved_task_probes
            or compare_catalog_support
        )
    ):
        raise ValueError(
            "TRAIN evidence memory requires one none-recipe catalog stream without probes"
        )
    if type(train_state_only) is not bool or (train_state_only and not catalog_training):
        raise ValueError("state-only TRAIN requires catalog training")
    if train_state_only and typed_development_readout:
        raise ValueError("typed binding does not yet admit state-only TRAIN snapshots")
    if type(probe_state_only) is not bool or (
        probe_state_only
        and (
            not catalog_training
            or not reserved_task_probes
            or compare_catalog_support
            or catalog_support != "none"
            or typed_development_readout
        )
    ):
        raise ValueError(
            "state-only probes require catalog reserved probes with the original none recipe"
        )
    if type(probe_train_readouts) is not bool or (
        probe_train_readouts
        and (
            not catalog_training
            or not reserved_task_probes
            or compare_catalog_support
            or catalog_support != "none"
            or typed_development_readout
        )
    ):
        raise ValueError(
            "probe TRAIN readouts require catalog reserved probes with the original none recipe"
        )
    if type(probe_marker_control) is not bool or (
        probe_marker_control
        and (
            not catalog_training
            or not reserved_task_probes
            or probe_rehearsal
            or compare_catalog_support
            or catalog_support != "none"
            or typed_development_readout
        )
    ):
        raise ValueError(
            "probe marker control requires catalog reserved probes without rehearsal or support"
        )
    if type(probe_rehearsal) is not bool or (
        probe_rehearsal
        and (
            not catalog_training
            or not reserved_task_probes
            or compare_catalog_support
            or catalog_support != "none"
            or typed_development_readout
        )
    ):
        raise ValueError(
            "probe rehearsal requires catalog reserved probes with the original none recipe"
        )
    if compare_catalog_support and catalog_support != "none":
        raise ValueError("comparison and selected single recipe are mutually exclusive")
    if type(reserved_task_probes) is not bool:
        raise ValueError("reserved probe selection must be an exact boolean")
    if reserved_task_probes and (compare_catalog_support or catalog_support != "none"):
        raise ValueError("reserved probes require the original none recipe")
    selected = compare_catalog_support or catalog_support != "none"
    if type(catalog_training) is not bool:
        raise ValueError("catalog training selection must be an exact boolean")
    if type(typed_development_readout) is not bool or (
        typed_development_readout
        and (not catalog_training or compare_catalog_support or reserved_task_probes)
    ):
        raise ValueError("typed development readout requires one catalog learner without probes")
    states, actions = feature_texts(
        include_catalog_support=selected, include_reserved_probe=reserved_task_probes
    )
    sequences = len(states) + len(actions)
    import torch
    from src.adapters.clm_decision import make_clm_head
    from src.adapters.clm_frozen_encoder import FrozenCLMEncoder

    torch.set_num_threads(2)
    started = monotonic()
    caller_rng = torch.get_rng_state().clone()
    report.update(
        complete=False,
        events=[],
        scope="fixed temporal development CLM head pilot; no final/CPC/serving claim",
    )
    if train_linear_update:
        report["scope"] = (
            "fixed union-catalog linear TRAIN development stream; no final/CPC/serving claim"
        )
        report["control_kind"] = CONTROL_KIND
    save()
    encoder = None
    with ProcessRssSampler(interval_seconds=0.02) as sampler:

        def before_step():
            if monotonic() - started > (720 if reserved_task_probes else 540):
                raise ValueError("native inner wall allowance exhausted")
            rss = sampler.sample()
            if rss is None or rss > 24 * 1024**3:
                raise ValueError("native RSS unavailable or allowance exceeded")

        def progress(event):
            report["events"].append(dict(elapsed_seconds=monotonic() - started, **event))
            save()

        try:
            loaded = load_local_clm(
                model_dir,
                head,
                head_factory=make_clm_head,
                before_step=before_step,
                progress=progress,
            )
            report.update(
                identity=asdict(loaded.identity),
                torch_version=torch.__version__,
                texts=dict(states=states, actions=actions),
            )
            encoder = FrozenCLMEncoder(
                loaded.model,
                loaded.tokenizer,
                identity=loaded.identity,
                max_sequences=sequences,
                max_forward_calls=5 if typed_development_readout else 1,
            )

            def guard(_model, args, kwargs):
                ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
                if (
                    ids.shape != mask.shape
                    or ids.ndim != 2
                    or ids.shape[0] != sequences
                    or ids.shape[1] > 96
                    or ids.numel() > sequences * 96
                    or kwargs.get("use_cache") is not False
                ):
                    raise ValueError("native recipe-bound sequence token allowance exceeded")
                before_step()
                progress(
                    dict(
                        event="encoder_forward_attempt",
                        sequences=sequences,
                        padded_slots=ids.numel(),
                    )
                )

            handle = loaded.model.register_forward_pre_hook(guard, with_kwargs=True)
            try:
                features = encoder.embed_texts(
                    states + actions, ("tail",) * len(states) + ("head",) * 4
                )
            finally:
                handle.remove()
            facts = encoder.last_readout
            report["tokens"] = {
                key: facts[key] for key in ("token_ids", "retained_lengths", "padded_tokens")
            }
            progress(dict(event="features_returned", feature_elements=features.numel()))
            reserved_probe_data = None
            if reserved_task_probes:
                features, training, readouts = build_reserved_probe_batches(
                    loaded.identity,
                    features,
                    **({"catalog_training": True} if catalog_training else {}),
                )
                reserved_probe_data = (training, readouts)
            modes = ("repeat", "route") if compare_catalog_support else (catalog_support,)
            first_train, choices = (
                build_catalog_training_batches(loaded.identity, features, catalog_support=modes[0])
                if catalog_training
                else build_feature_batches(loaded.identity, features, catalog_support=modes[0])
            )
            original = deepcopy(
                (
                    loaded.state_head.state_dict(),
                    loaded.action_head.state_dict(),
                    loaded.logit_scale.detach(),
                )
            )
            controls = {}
            for version, phase in (("v1", "baseline"), ("v2", "after_B")):
                before_step()
                controls[version] = (
                    [[1 / 3] * 3 for _ in range(4)]
                    if train_linear_update
                    else _control_readout(loaded, choices[phase])
                )
            report["control_probabilities"] = controls
            if compare_catalog_support:
                report["arms"] = {}
            for index, mode in enumerate(modes):
                before_step()
                encoder._check()
                if not _control_matches(torch, loaded, original) or not torch.equal(
                    caller_rng, torch.get_rng_state()
                ):
                    raise ValueError("shared source changed before next recipe")
                arm = (
                    dict(complete=False, control_probabilities=deepcopy(controls))
                    if compare_catalog_support
                    else report
                )
                if compare_catalog_support:
                    report["arms"][mode] = arm
                train = (
                    first_train
                    if index == 0
                    else (
                        build_catalog_training_batches(
                            loaded.identity, features, catalog_support=mode
                        )[0]
                        if catalog_training
                        else build_feature_batches(loaded.identity, features, catalog_support=mode)[
                            0
                        ]
                    )
                )
                stream_extra: dict[str, Any] = {}
                if train_linear_update:
                    stream_extra.update(
                        train_linear_update=True, linear_actions=features[-4:].clone()
                    )
                if paired_replay_candidate:
                    stream_extra["paired_replay_candidate"] = True
                if catalog_training:
                    stream_extra["catalog_training"] = True
                if train_state_only:
                    stream_extra["train_state_only"] = True
                if train_evidence_memory:
                    stream_extra["train_evidence_memory"] = True
                if reserved_task_probes:
                    stream_extra["reserved_probe_data"] = reserved_probe_data
                if probe_linear_fit_control:
                    stream_extra["probe_linear_fit_control"] = True
                if probe_state_only:
                    stream_extra["probe_state_only"] = True
                if probe_train_readouts:
                    stream_extra["probe_train_readouts"] = True
                if probe_marker_control:
                    stream_extra["probe_marker_control"] = True
                if probe_rehearsal:
                    stream_extra["probe_rehearsal"] = True
                if typed_development_readout:
                    stream_extra["typed_encoder"] = encoder
                _run_head_stream(
                    loaded,
                    train,
                    choices,
                    arm,
                    mode=mode,
                    diagnostics=(train_catalog_diagnostics or compare_catalog_support),
                    before_step=before_step,
                    save=save,
                    **stream_extra,
                )
                if compare_catalog_support:
                    _require_comparison_arm(mode, arm)
                encoder._check()
                if not _control_matches(torch, loaded, original) or not torch.equal(
                    caller_rng, torch.get_rng_state()
                ):
                    raise ValueError("shared source changed during preceding recipe")
            # Why this: neither arm's labels or diagnostic scores can select the other's work.
            if compare_catalog_support:
                scored = score_catalog_support_comparison(report)
                for mode, values in scored.pop("arm_scores").items():
                    report["arms"][mode].update(values, complete=True)
                report.update(scored)
                report["cached_typed_decodes"] = sum(
                    arm["cached_typed_decodes"] for arm in report["arms"].values()
                )
            else:
                report.update(score_sequence(report))
                if train_catalog_diagnostics:
                    report["train_catalog_diagnostics"] = score_train_choices(report)
                if reserved_task_probes:
                    report["reserved_probe_scores"] = score_reserved_probes(report)
            report.update(control_unchanged=True, caller_rng_unchanged=True)
            before_step()
            report["complete"] = True
        finally:
            if encoder is not None:
                report["encoder_work"] = encoder.work
            report.update(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-catalog-diagnostics", action="store_true")
    parser.add_argument("--reserved-task-probes", action="store_true")
    parser.add_argument("--probe-linear-fit-control", action="store_true")
    parser.add_argument("--catalog-training", action="store_true")
    parser.add_argument("--train-linear-update", action="store_true")
    parser.add_argument("--typed-development-readout", action="store_true")
    parser.add_argument("--probe-rehearsal", action="store_true")
    parser.add_argument("--probe-marker-control", action="store_true")
    parser.add_argument("--probe-train-readouts", action="store_true")
    parser.add_argument("--probe-state-only", action="store_true")
    parser.add_argument("--train-state-only", action="store_true")
    parser.add_argument("--train-evidence-memory", action="store_true")
    parser.add_argument("--paired-replay-candidate", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--catalog-support", choices=("none", "repeat", "route"), default="none")
    group.add_argument("--compare-catalog-support", action="store_true")
    args = parser.parse_args()
    try:
        _admit_linear_training(
            args.train_linear_update,
            **{
                key: value
                for key, value in vars(args).items()
                if key not in ("model_dir", "head", "output", "train_linear_update")
            },
        )
        _admit_linear_probes(
            args.probe_linear_fit_control,
            **{
                key: value
                for key, value in vars(args).items()
                if key
                not in (
                    "model_dir",
                    "head",
                    "output",
                    "probe_linear_fit_control",
                    "train_catalog_diagnostics",
                )
            },
        )
    except ValueError as error:
        parser.error(str(error))
    if args.paired_replay_candidate:
        try:
            _admit_paired_replay(
                True,
                **{
                    key: value
                    for key, value in vars(args).items()
                    if key not in ("model_dir", "head", "output", "paired_replay_candidate")
                },
            )
        except ValueError as error:
            parser.error(str(error))
    if args.train_evidence_memory and (
        not args.catalog_training
        or args.catalog_support != "none"
        or args.reserved_task_probes
        or args.compare_catalog_support
    ):
        parser.error(
            "--train-evidence-memory requires one none-recipe --catalog-training run without probes"
        )
    if args.train_state_only and (not args.catalog_training or args.typed_development_readout):
        parser.error("--train-state-only requires --catalog-training without typed readout")
    if args.probe_state_only and (
        not args.catalog_training
        or not args.reserved_task_probes
        or args.catalog_support != "none"
        or args.compare_catalog_support
        or args.typed_development_readout
    ):
        parser.error(
            "--probe-state-only requires --catalog-training --reserved-task-probes with the original none recipe"
        )
    if args.probe_train_readouts and (
        not args.catalog_training
        or not args.reserved_task_probes
        or args.catalog_support != "none"
        or args.compare_catalog_support
        or args.typed_development_readout
    ):
        parser.error(
            "--probe-train-readouts requires --catalog-training --reserved-task-probes with the original none recipe"
        )
    if args.probe_marker_control and (
        not args.catalog_training
        or not args.reserved_task_probes
        or args.probe_rehearsal
        or args.catalog_support != "none"
        or args.compare_catalog_support
        or args.typed_development_readout
    ):
        parser.error(
            "--probe-marker-control requires --catalog-training --reserved-task-probes without rehearsal or support"
        )
    if args.probe_rehearsal and (
        not args.catalog_training
        or not args.reserved_task_probes
        or args.catalog_support != "none"
        or args.compare_catalog_support
        or args.typed_development_readout
    ):
        parser.error(
            "--probe-rehearsal requires --catalog-training --reserved-task-probes with the original none recipe"
        )
    if args.typed_development_readout and (
        not args.catalog_training or args.compare_catalog_support or args.reserved_task_probes
    ):
        parser.error(
            "--typed-development-readout requires one --catalog-training run without probes"
        )
    if args.reserved_task_probes and (
        args.compare_catalog_support or args.catalog_support != "none"
    ):
        parser.error("--reserved-task-probes requires the original none recipe")
    if args.output.exists():
        raise ValueError("refusing to replace existing temporal evidence")
    report: dict[str, Any] = {}

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    extra: dict[str, Any] = {}
    if args.paired_replay_candidate:
        extra["paired_replay_candidate"] = True
    if args.catalog_support != "none":
        extra["catalog_support"] = args.catalog_support
    if args.compare_catalog_support:
        extra["compare_catalog_support"] = True
    if args.reserved_task_probes:
        extra["reserved_task_probes"] = True
    if args.probe_linear_fit_control:
        extra["probe_linear_fit_control"] = True
    if args.train_linear_update:
        extra["train_linear_update"] = True
    if args.catalog_training:
        extra["catalog_training"] = True
    if args.train_state_only:
        extra["train_state_only"] = True
    if args.train_evidence_memory:
        extra["train_evidence_memory"] = True
    if args.probe_state_only:
        extra["probe_state_only"] = True
    if args.probe_train_readouts:
        extra["probe_train_readouts"] = True
    if args.probe_marker_control:
        extra["probe_marker_control"] = True
    if args.probe_rehearsal:
        extra["probe_rehearsal"] = True
    if args.typed_development_readout:
        extra["typed_development_readout"] = True
    try:
        run_case(
            args.model_dir,
            args.head,
            report,
            train_catalog_diagnostics=args.train_catalog_diagnostics,
            save=save,
            **extra,
        )
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
