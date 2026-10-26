"""Read-only TRAIN fit/alignment diagnosis of saved native BP ticket snapshots.

No learner construction, restoration, training, RNG, task rollout or final data.
Uses a bounded NumPy reference to the existing tanh head and requires agreement
with already-paid native scores before interpreting the teacher projections.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

from scripts.recurring_ticket_workflow_example import CHECKPOINTS, PHASES, episode_slots
from scripts.run_ticket_workflow import FEATURES, training_cases
from src.adapters.perceptron_decision import build_decision_training_batch
from src.core.categorical_probability_margins import probability_margins
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.temporal_decision_sequence import VOCABULARY
from src.core.ticket_workflow import CATALOG
from src.core.typed_decision import (
    CandidateScore,
    DecisionCandidate,
    DecisionRequest,
    DecisionResult,
    validate_decision_result,
)

TOLERANCE = 1e-6
SHAPES = dict(
    weight_feature_hidden=(7, 3),
    bias_hidden=(1, 3),
    weight_hidden_output=(3, 6),
    bias_output=(1, 6),
)
POLICY = dict(
    method="bp",
    feature_dim=7,
    hidden_dim=3,
    seed=17,
    learning_rate=0.03,
    inference_steps=2,
    inference_learning_rate=0.04,
    momentum=0.9,
    num_classes=6,
)
POLICY_STUDY_SHA = "e3403039c01ffef482fa4e128fcfaa6450c413d4ea15e77bbe6a9837873b905f"


def _parameters(checkpoint):
    state, point = checkpoint["state"], checkpoint["source_updates"]
    if (
        any(
            type(state[k]) is not int
            for k in ("completed_updates", "training_examples", "format_version")
        )
        or state["completed_updates"] != point
        or state["training_examples"] != point * 3
        or state["format_version"] != 1
        or checkpoint["source_rows"] != point * 3
    ):
        raise ValueError("saved BP native counters differ from fixed source checkpoint")
    return _head_parameters(state)


def _head_parameters(state):
    """Decode identical fixed float32 head geometry, without creating a learner."""
    if state["policy"] != POLICY or any(
        type(state["policy"][k]) is not type(v) for k, v in POLICY.items()
    ):
        raise ValueError("saved BP policy differs from frozen native recipe")
    head = state["head_state"]
    if any(
        head.get(k) != v
        for k, v in dict(
            format_version=1, feature_dim=7, hidden_dim=3, num_classes=6, device="cpu"
        ).items()
    ):
        raise ValueError("saved BP head geometry differs")
    arrays: dict[str, Any] = {}
    for name, shape in SHAPES.items():
        record = head[name]
        if (
            set(record) != {"dtype", "shape", "values"}
            or record["dtype"] != "torch.float32"
            or record["shape"] != list(shape)
        ):
            raise ValueError("saved parameter geometry/dtype differs")
        rows = record["values"]
        if (
            type(rows) is not list
            or len(rows) != shape[0]
            or any(type(row) is not list or len(row) != shape[1] for row in rows)
            or any(type(value) not in (int, float) for row in rows for value in row)
        ):
            raise ValueError("saved parameter values must match bounded scalar geometry")
        values = np.array(rows, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            native = values.astype(np.float32)
        if (
            values.shape != shape
            or not np.isfinite(native).all()
            or not np.array_equal(native.astype(np.float64), values)
        ):
            raise ValueError("saved parameters must be exact finite float32 values")
        arrays[name] = native
    return arrays


def _project(features, arrays, work):
    work["projection_calls"] += 1
    work["projection_rows"] += len(features)
    work["nominal_affine_terms"] += len(features) * 39
    x = features.astype(np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        hidden = np.tanh(x @ arrays["weight_feature_hidden"] + arrays["bias_hidden"])
        values = hidden @ arrays["weight_hidden_output"] + arrays["bias_output"]
    if not np.isfinite(values).all():
        raise ValueError("nonfinite reference projection")
    return values.astype(np.float64)


def _admit(source):
    if (
        type(source) is not dict
        or source.get("complete") is not True
        or source.get("method") != "bp"
        or source.get("definition") != "chronological_ticket_workflow_v1"
    ):
        raise ValueError("complete original BP recipe result required")
    if (
        type(source.get("checkpoints")) is not list
        or len(source["checkpoints"]) != 4
        or type(source.get("episodes")) is not list
        or len(source["episodes"]) != 11
    ):
        raise ValueError("original bounded checkpoints and episode slots required")
    if source.get("training_work") != dict(
        attempted_updates=12,
        attempted_rows=36,
        returned_updates=12,
        unreturned_work_unknown=False,
        budget_committed_updates=12,
    ):
        raise ValueError("original complete BP TRAIN work required")
    if any(
        type(source.get(k)) is not int or source[k] != 0
        for k in ("final_examples", "probe_updates", "replay_examples", "external_actions")
    ):
        raise ValueError("diagnosis requires original TRAIN-only no-final/probe/replay source")
    if (
        source.get("numpy_rng_unchanged") is not True
        or source.get("torch_rng_unchanged") is not True
    ):
        raise ValueError("original RNG isolation required")
    expected = [
        dict(
            role="train",
            context=phase,
            row_ids=[f"{phase}-{repeat}-{step}" for step in range(3)],
            rows=3,
        )
        for phase, _ in PHASES
        for repeat in range(4)
    ]
    if (
        source["training_data"] != expected
        or len(source["applied_updates"]) != 12
        or source["budget"]["updates_completed"] != 12
    ):
        raise ValueError("original chronological TRAIN binding differs")
    if any(type(p["source_updates"]) is not int for p in source["checkpoints"]) or [
        p["source_updates"] for p in source["checkpoints"]
    ] != list(CHECKPOINTS):
        raise ValueError("all four original checkpoints required")
    slots = [
        (point, scenario, concept)
        for point in CHECKPOINTS
        for scenario, concept in episode_slots(point)
    ]
    actual = [
        (e["source_updates"], e["episode"]["scenario"], e["concept"]) for e in source["episodes"]
    ]
    if slots != actual or any(not e["episode"]["complete"] for e in source["episodes"]):
        raise ValueError("original complete bounded episode slots required")
    if any(not 1 <= len(e["episode"]["steps"]) <= 6 for e in source["episodes"]):
        raise ValueError("saved decision trace exceeds original bounds")


def _teacher(spec, *, policy_change=False):
    cases = training_cases(policy_version="v1") if policy_change else training_cases()
    group = "train-workflow-traces-v1" if policy_change else "train-workflow-traces"
    if len(cases) != 12 or any(c.role != "train" or c.source_group != group for c in cases):
        raise ValueError("original TRAIN teacher required")
    selected = tuple(
        cases[3 * VOCABULARY.index(concept) + step]
        for concept in (("invoice", "password") if policy_change else ("invoice", "parcel"))
        for step in range(3)
    )
    batch = build_decision_training_batch(selected, spec, "offline-teacher", allow_subsets=True)
    for i, case in enumerate(selected):
        public = replace(case.request, query=case.request.query.removesuffix(" train") + " please")
        if not np.array_equal(batch.features[i : i + 1], encode_decision(public, spec)[0]):
            raise ValueError("teacher/public feature alignment differs")
    for i in range(6):
        for j in range(i):
            if (
                np.array_equal(batch.features[i], batch.features[j])
                and batch.targets[i, 0] != batch.targets[j, 0]
            ):
                raise ValueError("conflicting teacher labels on identical features")
    return selected, batch


def _native_parity(source, parameters, spec, work):
    error = 0.0
    rows = scores = ambiguous = 0
    for episode in source["episodes"]:
        arrays = parameters[episode["source_updates"]]
        for step in episode["episode"]["steps"]:
            r = step["request"]
            request = DecisionRequest(
                r["request_id"],
                r["query"],
                tuple(DecisionCandidate(**c) for c in r["allowed_candidates"]),
            )
            result = DecisionResult(
                **{
                    **step["result"],
                    "scores": tuple(CandidateScore(**c) for c in step["result"]["scores"]),
                }
            )
            validate_decision_result(request, result)
            logits = _project(encode_decision(request, spec)[0], arrays, work)[0]
            raw = {c.candidate_id: float(logits[i]) for i, c in enumerate(spec.catalog)}
            for score in result.scores:
                difference = abs(score.raw_score - raw[score.candidate_id])
                if difference > TOLERANCE:
                    raise ValueError("saved native raw-score parity exceeds frozen tolerance")
                error = max(error, difference)
                scores += 1
            ranked = sorted(
                ((-raw[c.candidate_id], c.candidate_id) for c in request.allowed_candidates)
            )
            uncertain = len(ranked) > 1 and ranked[1][0] - ranked[0][0] <= 2 * TOLERANCE
            ambiguous += uncertain
            if not uncertain and ranked[0][1] != result.selected_id:
                raise ValueError("saved native action differs from reference")
            rows += 1
    if rows != source["episode_work"]["prediction_calls_validated"]:
        raise ValueError("native parity trace work differs")
    return dict(
        rows=rows,
        scores=scores,
        max_absolute_raw_error=error,
        absolute_tolerance=TOLERANCE,
        ambiguous_decisions=ambiguous,
    )


def _summarize_logits(logits, targets):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    labels = targets[:, 0].astype(np.int64)
    ce = np.log(weights.sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    competitors = logits.copy()
    competitors[np.arange(len(labels)), labels] = -np.inf
    margins = logits[np.arange(len(labels)), labels] - competitors.max(axis=1)
    return probabilities, ce, margins, probability_margins(probabilities, targets, logits.shape[1])


def diagnose(source, *, work=None):
    if work is None:
        work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    _admit(source)
    spec = DecisionFeatureSpec(FEATURES, CATALOG)
    parameters = {p["source_updates"]: _parameters(p) for p in source["checkpoints"]}
    cases, batch = _teacher(spec)
    parity = _native_parity(source, parameters, spec, work)
    points = []
    class_ids = tuple(c.candidate_id for c in CATALOG)
    for point in CHECKPOINTS:
        logits = _project(batch.features, parameters[point], work)
        _full, ce, margins, summary = _summarize_logits(logits, batch.targets)
        rows = []
        for i, case in enumerate(cases):
            concept = ("invoice", "parcel")[i // 3]
            allowed = tuple(
                c.candidate_id
                for c in sorted(case.request.allowed_candidates, key=lambda c: c.candidate_id)
            )
            indices = [class_ids.index(c) for c in allowed]
            target = allowed.index(case.target_id)
            probs, allowed_ce, raw_margin, details = _summarize_logits(
                logits[i : i + 1, indices], np.array([[float(target)]])
            )
            exposures = (
                (4 * int(point >= 4) + 4 * int(point >= 12))
                if concept == "invoice"
                else 4 * int(point >= 8)
            )
            rows.append(
                dict(
                    concept=concept,
                    phase=("sealed", "ready", "staged")[i % 3],
                    target_id=case.target_id,
                    allowed_selected_id=allowed[int(np.argmax(probs[0]))],
                    historical_train_exposures=exposures,
                    features=batch.features[i].tolist(),
                    global_ce=float(ce[i]),
                    global_raw_margin=float(margins[i]),
                    global_probability_margin=summary["target_margins"][i],
                    allowed_ce=float(allowed_ce[0]),
                    allowed_raw_margin=float(raw_margin[0]),
                    allowed_probability_margin=details["target_margins"][0],
                    allowed_correct=bool(details["argmax_correct_rows"]),
                    numerically_ambiguous=abs(float(raw_margin[0])) <= 2 * TOLERANCE,
                )
            )
        points.append(
            dict(
                source_updates=point,
                global_mean_ce=float(np.mean(ce)),
                global_probability_margins=summary,
                rows=rows,
                exposed_correct=sum(
                    r["allowed_correct"] for r in rows if r["historical_train_exposures"]
                ),
                exposed_rows=sum(bool(r["historical_train_exposures"]) for r in rows),
            )
        )
    return dict(
        complete=True,
        definition="saved_bp_ticket_train_fit_v1",
        role="offline_train_diagnosis_only",
        source_primary_summary=deepcopy(source["summary"]),
        native_parity=parity,
        checkpoints=points,
        work=dict(work),
        training_updates=0,
        probe_updates=0,
        final_examples=0,
        model_initializations=0,
        random_draws=0,
        alignment=dict(teacher_public_equal_rows=6, conflicting_feature_targets=0),
        qualification="NumPy float32 reference checked against saved native scores; TRAIN fit is not task success, causal underfitting proof or a new comparative metric; historical pending TRAIN labels never enter fitting/controllers",
    )


def _policy_parameters(checkpoint):
    """Explicit qualified96/576 + six-row clocks; old admission remains strict."""
    state, point = checkpoint["state"], checkpoint["source_updates"]
    if (
        type(point) is not int
        or not 0 <= point <= 4
        or any(
            type(state[k]) is not int
            for k in ("completed_updates", "training_examples", "format_version")
        )
        or state["completed_updates"] != 96 + point
        or state["training_examples"] != 576 + 6 * point
        or state["format_version"] != 1
        or type(checkpoint["source_rows"]) is not int
        or checkpoint["source_rows"] != 576 + 6 * point
    ):
        raise ValueError("qualified policy native counters differ from original six-row history")
    return _head_parameters(state)


def _admit_policy_study(source):
    from scripts.ticket_controller_comparison import (
        POLICY_MODES,
        POLICY_FITTED_SHA,
        _require_policy_trial,
        _policy_outcomes,
    )
    from scripts.ticket_controller_streams import POLICY_CHANGE_PHASES
    from scripts.run_bp_new_contexts import SOURCE_SHA

    if (
        type(source) is not dict
        or source.get("complete") is not True
        or source.get("definition") != "phase_matched_policy_controllers_v1"
        or type(source.get("models_frozen_at")) is not int
        or source.get("models_frozen_at") != 1
        or type(source.get("clock")) is not int
        or source.get("clock") != 7
        or source.get("numpy_rng_unchanged") is not True
        or source.get("torch_rng_unchanged") is not True
        or any(
            type(source.get(k)) is not int or source[k] != 0
            for k in (
                "new_controller_fits",
                "promotions",
                "final_examples",
                "final_training_updates",
            )
        )
        or source["protocol"]["source_sha256"] != SOURCE_SHA
        or source["protocol"]["fitted_report_sha256"] != POLICY_FITTED_SHA
        or [c["mode"] for c in source["cases"]] != list(POLICY_MODES)
        or [r["mode"] for r in source["development_readouts"]] != list(POLICY_MODES)
    ):
        raise ValueError("complete original five-controller policy study required")
    base = source["cases"][0]["result"]["policy_checkpoints"][0]["checkpoint"]
    for case, evaluation, expected_updates in zip(
        source["cases"], source["development_readouts"], (0, 2, 4, 0, 4), strict=True
    ):
        _require_policy_trial(case["result"], case["mode"])
        if case["result"]["policy_checkpoints"][1]["applied_receipts"] != expected_updates:
            raise ValueError("original opportunity4 fixed controller doses differ")
        # JSON object keys are strings; normalize only the declared four clocks.
        bound = deepcopy(evaluation)
        mapping = evaluation["source_updates_by_opportunity"]
        expected = {
            r["opportunity"]: r["checkpoint"]["source_updates"]
            for r in case["result"]["policy_checkpoints"]
        }
        if mapping != {str(k): v for k, v in expected.items()}:
            raise ValueError("paid readout does not bind to the original captured updates")
        bound["source_updates_by_opportunity"] = expected
        _policy_outcomes(bound)
        if case["result"]["policy_checkpoints"][0]["checkpoint"] != base:
            raise ValueError("original common qualified source differs across cases")
        stream = case["result"]["training_stream"]
        if (
            stream["role"] != "train"
            or stream["label_delay"] != 1
            or stream["row_counts"] != [6] * 12
            or stream["contexts"] != [p for p, _, _ in POLICY_CHANGE_PHASES for _ in range(4)]
        ):
            raise ValueError("policy diagnosis requires the original TRAIN corpus")


def _policy_fit_rows(cases, batch, arrays, spec, updates, work):
    logits = _project(batch.features, arrays, work)
    _, global_ce, global_margin, _ = _summarize_logits(logits, batch.targets)
    ids = tuple(c.candidate_id for c in spec.catalog)
    rows = []
    for i, case in enumerate(cases):
        allowed = tuple(
            c.candidate_id
            for c in sorted(case.request.allowed_candidates, key=lambda c: c.candidate_id)
        )
        indices = [ids.index(c) for c in allowed]
        probs, ce, margin, detail = _summarize_logits(
            logits[i : i + 1, indices], np.array([[float(allowed.index(case.target_id))]])
        )
        rows.append(
            dict(
                concept=("invoice", "password")[i // 3],
                phase=("sealed", "ready", "staged")[i % 3],
                target_id=case.target_id,
                allowed_selected_id=allowed[int(np.argmax(probs[0]))],
                new_train_exposures=updates,
                prior_v2_teacher_exposures=96 if i < 3 else 0,
                prior_v2_target_id=("open", "inspect", "commit")[i] if i < 3 else None,
                features=batch.features[i].tolist(),
                global_ce=float(global_ce[i]),
                global_raw_margin=float(global_margin[i]),
                allowed_ce=float(ce[0]),
                allowed_raw_margin=float(margin[0]),
                allowed_probability_margin=detail["target_margins"][0],
                allowed_correct=bool(detail["argmax_correct_rows"]),
                numerically_ambiguous=abs(float(margin[0])) <= 2 * TOLERANCE,
                route_vs_inspect_raw_margin=float(
                    logits[i, ids.index("route")] - logits[i, ids.index("inspect")]
                )
                if i == 1
                else None,
            )
        )
    return rows


def diagnose_policy_study(source, *, work=None):
    """Read all fixed source/op4 v1 TRAIN fits only after paid-native parity.

    Consumes saved scalar states and traces; no native owner/restore/update or
    source authority. This is fit evidence, not a learnability or tuning decision.
    """
    if work is None:
        work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    _admit_policy_study(source)
    spec = DecisionFeatureSpec(FEATURES, CATALOG)
    cases, batch = _teacher(spec, policy_change=True)
    expected_features, expected_targets = batch.features.tolist(), batch.targets.tolist()
    for case in source["cases"]:
        stream = case["result"]["training_stream"]
        if (
            stream["first_features"] != expected_features
            or stream["first_targets"] != expected_targets
        ):
            raise ValueError("saved first TRAIN packet differs from actual v1 teacher")
        if any(
            type(v) is not float
            for row in stream["first_features"] + stream["first_targets"]
            for v in row
        ):
            raise ValueError("saved TRAIN packet scalar types differ")
    selected = [("source", 0, source["cases"][0], source["development_readouts"][0])]
    selected.extend(
        (c["mode"], 4, c, e)
        for c, e in zip(source["cases"], source["development_readouts"], strict=True)
    )
    prepared = []
    for name, tick, case, evaluation in selected:
        record = next(r for r in case["result"]["policy_checkpoints"] if r["opportunity"] == tick)
        arrays = _policy_parameters(record["checkpoint"])
        episodes = [r for r in evaluation["episodes"] if r["opportunity"] == tick]
        native_point = record["checkpoint"]["state"]["completed_updates"]
        wrapper = dict(
            episodes=[dict(source_updates=native_point, episode=e["episode"]) for e in episodes],
            episode_work=dict(
                prediction_calls_validated=sum(
                    e["episode"]["work"]["prediction_calls_validated"] for e in episodes
                )
            ),
        )
        prepared.append((name, tick, record, arrays, wrapper))
    parity = [
        _native_parity(trace, {r["checkpoint"]["state"]["completed_updates"]: a}, spec, work)
        for _, _, r, a, trace in prepared
    ]
    points = []
    for name, tick, record, arrays, _ in prepared:
        rows = _policy_fit_rows(cases, batch, arrays, spec, record["applied_receipts"], work)
        points.append(
            dict(
                mode=name,
                opportunity=tick,
                new_updates=record["applied_receipts"],
                native_updates=record["checkpoint"]["state"]["completed_updates"],
                native_rows=record["checkpoint"]["source_rows"],
                v1_teacher_rows=rows,
                allowed_correct=sum(r["allowed_correct"] for r in rows),
                global_mean_ce=float(np.mean([r["global_ce"] for r in rows])),
            )
        )
    return dict(
        complete=True,
        definition="saved_policy_v1_train_fit_v1",
        role="offline_train_diagnosis_only",
        prior_native_updates=96,
        prior_native_rows=576,
        points=points,
        native_parity=parity,
        work=dict(work),
        parameter_array_bytes=sum(
            a.nbytes for _, _, _, arrays, _ in prepared for a in arrays.values()
        ),
        training_updates=0,
        probe_updates=0,
        final_examples=0,
        model_initializations=0,
        random_draws=0,
        alignment=dict(teacher_public_equal_rows=6, conflicting_feature_targets=0),
        qualification="Original NumPyfloat32 reference checked against already-paid native v1 scores at source+ALLop4states; TRAIN fit/margins are not task success, capacity/plasticity-loss/probe evidence or a tuning choice.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy-comparison", action="store_true")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        raise ValueError("input must stay sealed")
    with args.output.open("xb") as stream:
        stream.write(b"{}\n")
    started = monotonic()
    work = dict(projection_calls=0, projection_rows=0, nominal_affine_terms=0)
    limit = 8 * 1024**2 if args.policy_comparison else 512 * 1024
    if args.input.stat().st_size > limit:
        raise ValueError("source exceeds declared diagnostic report bound")
    data = args.input.read_bytes()
    try:
        if args.policy_comparison and hashlib.sha256(data).hexdigest() != POLICY_STUDY_SHA:
            raise ValueError("policy comparison bytes differ from frozen original study")
        result = (
            diagnose_policy_study(json.loads(data), work=work)
            if args.policy_comparison
            else diagnose(json.loads(data), work=work)
        )
    except Exception as error:
        result = dict(complete=False, error=f"{type(error).__name__}: {error}", work=work)
    result.update(
        source_sha256=hashlib.sha256(data).hexdigest(), elapsed_seconds=monotonic() - started
    )
    if args.input.read_bytes() != data:
        result.update(complete=False, error="source bytes changed")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
