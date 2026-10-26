"""One bounded, disposable linear fit of complete public TRAIN episodes.

Existing core encoders provide the canonical inputs. No benchmark/deployed/probe
learner, stream, promotion, final data, replay, RNG or filesystem is accessed.
This offline fit question is separate from matched CE learning and adaptation.
"""

from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from typing import Any, Callable

import numpy as np

from src.core.categorical_linear_fit import (
    CategoricalLinearFit,
    RCOND,
    fit_categorical_linear,
    predict_categorical_linear,
)
from src.core.categorical_probability_scores import probability_scores, validate_class_probabilities
from src.core.categorical_probability_margins import probability_margins
from src.core.demonstration_episode import DemonstrationEpisode
from src.core.demonstration_schedule_witness import DemonstrationScheduleWitness
from src.core.minigrid_invariant_encoding import (
    INVARIANT_HISTORY_ENCODING_ID,
    canonicalize_public_history,
)
from src.core.minigrid_context_encoding import CONTEXT_HISTORY_ENCODING_ID, context_public_history
from src.adapters.minigrid_train_catalogue import _validate_sources
from src.app.probe_return_phase import paid_report_guard

DEFINITION = "minigrid_categorical_linear_train_fit_v1"


class LinearFitAdmissionFailed(ValueError):
    """Public TRAIN validation failed; retain its bounded partial work report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(report["reason"])


def describe_protocol() -> dict[str, Any]:
    return dict(
        definition=DEFINITION,
        role="train",
        encoding_id=INVARIANT_HISTORY_ENCODING_ID,
        objective="equal-row-weight one-hot least squares; not native CE",
        dtype="float64",
        intercept="append constant1",
        solver="numpy.linalg.lstsq",
        rcond=RCOND,
        classes=list(range(7)),
        regularizer=None,
        scaling=None,
        max_episodes=16,
        max_rows=128,
        prediction="stable softmax of linear scores; first argmax ties",
        interpretation="constructive observed TRAIN fit only; failure does not prove linear inseparability",
        additional_native_forwards=0,
        optimizer_steps=0,
        benchmark_updates=0,
    )


def _work() -> dict[str, Any]:
    return dict.fromkeys(
        (
            "fit_calls_attempted",
            "fit_calls_completed",
            "fit_rows_attempted",
            "fit_rows_completed",
            "prediction_calls_attempted",
            "prediction_calls_completed",
            "prediction_rows_attempted",
            "prediction_rows_completed",
            "projection_rows_attempted",
            "projection_rows_completed",
            "projected_array_equivalent_bytes",
            "feature_input_array_bytes",
            "target_input_array_bytes",
            "design_array_bytes",
            "one_hot_array_bytes",
            "coefficient_array_bytes",
            "prediction_array_bytes",
            "score_reductions_completed",
            "score_rows_completed",
            "native_forwards",
            "optimizer_steps",
            "benchmark_updates",
        ),
        0,
    ) | dict(
        fit_seconds=0.0,
        prediction_seconds=0.0,
        elapsed_seconds=0.0,
        failure_work_may_be_unobserved=False,
    )


def _project(episodes, witness, work):
    features, targets = [], []
    for episode in episodes:
        for transition in episode.transitions:
            for name in ("observation", "next_observation"):
                witness.check(episodes)
                work["projection_rows_attempted"] += 1
                view = canonicalize_public_history(getattr(transition, name))
                work["projection_rows_completed"] += 1
                work["projected_array_equivalent_bytes"] += len(view) * 8
                witness.check(episodes)
                if name == "observation":
                    features.append(view)
            targets.append((float(transition.action),))
    return np.array(features, dtype=np.float64), np.array(targets, dtype=np.float64)


def _unchanged_arrays(expected, actual) -> None:
    for before, after in zip(expected, actual, strict=True):
        if (
            type(after) is not np.ndarray
            or after.dtype != before.dtype
            or after.shape != before.shape
            or not np.array_equal(before, after)
        ):
            raise ValueError("linear control detached operation inputs changed")


def _score(probabilities, targets, work):
    metrics = probability_scores(probabilities, targets, 7)
    margins = probability_margins(probabilities, targets, 7)
    work["score_reductions_completed"] += 1
    work["score_rows_completed"] += len(targets)
    return dict(**metrics, action_margins=margins)


def _fit_model(features, targets, episodes, witness, work):
    inputs = (features.copy(), targets.copy())
    work["fit_calls_attempted"] += 1
    work["fit_rows_attempted"] += len(features)
    entered = monotonic()
    try:
        model = fit_categorical_linear(*inputs, 7)
        work["fit_calls_completed"] += 1
        work["fit_rows_completed"] += len(features)
    finally:
        work["fit_seconds"] += monotonic() - entered
    _unchanged_arrays((features, targets), inputs)
    witness.check(episodes)
    if type(model) is not CategoricalLinearFit:
        raise ValueError("linear fit returned an invalid model")
    model.__post_init__()
    if (
        model.feature_dim != features.shape[1]
        or model.num_classes != 7
        or model.fit_rows != len(features)
    ):
        raise ValueError("linear fit returned a different TRAIN model geometry")
    return model


def _predict_model(model, features, episodes, witness, work):
    expected = asdict(model)
    coefficients, singular_values = model.coefficients, model.singular_values
    inputs = features.copy()
    work["prediction_calls_attempted"] += 1
    work["prediction_rows_attempted"] += len(features)
    entered = monotonic()
    try:
        probabilities = predict_categorical_linear(model, inputs)
        validate_class_probabilities(probabilities, 7)
        if probabilities.shape != (len(features), 7):
            raise ValueError("linear prediction must match every admitted TRAIN row")
        work["prediction_calls_completed"] += 1
        work["prediction_rows_completed"] += len(features)
    finally:
        work["prediction_seconds"] += monotonic() - entered
    _unchanged_arrays((features,), (inputs,))
    witness.check(episodes)
    model.__post_init__()
    if (
        model.coefficients is not coefficients
        or model.singular_values is not singular_values
        or asdict(model) != expected
        or model.squared_error.hex() != expected["squared_error"].hex()
    ):
        raise ValueError("linear fit model changed during prediction")
    return probabilities


def _episode_scores(episodes, probabilities, targets, work, scores, witness):
    offset = 0
    for episode in episodes:
        end = offset + len(episode.transitions)
        selected, labels = probabilities[offset:end], targets[offset:end]
        actions = np.argmax(selected, axis=1)
        score = dict(
            episode_id=episode.episode_id,
            context=episode.context,
            target_actions=labels[:, 0].astype(int).tolist(),
            predicted_actions=actions.tolist(),
            probabilities=selected.tolist(),
            exact_sequence_accuracy=float(np.array_equal(actions, labels[:, 0])),
            **_score(selected, labels, work),
        )
        witness.check(episodes)
        scores.append(score)
        offset = end


def diagnose_train_linear_fit(episodes: tuple[DemonstrationEpisode, ...]) -> dict[str, Any]:
    """Fit once on every provided TRAIN row, without an existing learner owner."""
    if type(episodes) is not tuple or not 1 <= len(episodes) <= 16:
        raise ValueError("linear control requires an exact tuple of1..16 complete TRAIN episodes")
    witness = DemonstrationScheduleWitness(episodes)
    rows = sum(len(episode.transitions) for episode in episodes)
    if rows > 128 or any(
        episode.num_classes != 7 or len(episode.transitions[0].observation) != 296
        for episode in episodes
    ):
        raise ValueError("linear control requires at most128 TRAIN296/K7 rows")
    started, work = monotonic(), _work()
    report: dict[str, Any] = dict(
        definition=DEFINITION,
        role="train",
        status="running",
        reason=None,
        protocol=describe_protocol(),
        fit=None,
        scores=[],
        aggregate_scores=None,
        prediction_probabilities=None,
        target_actions=None,
        work=work,
        source_unchanged=None,
    )
    try:
        features, targets = _project(episodes, witness, work)
        work.update(
            feature_input_array_bytes=features.nbytes,
            target_input_array_bytes=targets.nbytes,
            design_array_bytes=rows * 297 * 8,
            one_hot_array_bytes=rows * 7 * 8,
        )
        model = _fit_model(features, targets, episodes, witness, work)
        report["fit"] = asdict(model)
        work["coefficient_array_bytes"] = 297 * 7 * 8
        probabilities = _predict_model(model, features, episodes, witness, work)
        report["prediction_probabilities"] = probabilities.tolist()
        report["target_actions"] = targets[:, 0].astype(int).tolist()
        work["prediction_array_bytes"] = probabilities.nbytes
        _episode_scores(episodes, probabilities, targets, work, report["scores"], witness)
        witness.check(episodes)
        report["aggregate_scores"] = _score(probabilities, targets, work)
        witness.check(episodes)
        report.update(status="complete", source_unchanged=True)
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
        work["failure_work_may_be_unobserved"] = True
        try:
            witness.check(episodes)
            report["source_unchanged"] = True
        except ValueError:
            report["source_unchanged"] = False
        if work["fit_calls_attempted"] == 0:
            work["elapsed_seconds"] = monotonic() - started
            raise LinearFitAdmissionFailed(report) from error
    work["elapsed_seconds"] = monotonic() - started
    return report


def describe_context_linear_protocol() -> dict[str, Any]:
    """Describe the separate two-head constructive control without data access."""
    return dict(
        definition="minigrid_public_cue_linear_train_fit_v1",
        role="train",
        encoding_id=CONTEXT_HISTORY_ENCODING_ID,
        routing="initial public blue/red ball cue only; canonical296 input per head",
        per_head=describe_protocol(),
        fit_calls=2,
        max_rows=128,
        max_episodes=16,
        coefficient_elements=4158,
        numeric_coefficient_bytes=33264,
        witness="every original TRAIN row and complete trace correct; no game success implied",
        limitation="different objective from online perceptron; no optimizer/history transfer, deployment, final or probe access",
    )


def _context_groups(episodes):
    witness, rows = _validate_sources(episodes, 1)
    groups: dict[str, list[DemonstrationEpisode]] = {"blue": [], "red": []}
    # Complete current/next admission above precedes either fit. Initial history
    # is constant within each episode, so routing needs only its first row.
    for episode in episodes:
        projected = context_public_history(episode.transitions[0].observation)
        cue = "blue" if projected[-2:] == (1.0, 0.0) else "red"
        groups[cue].append(episode)
    if not all(groups.values()):
        raise ValueError("context linear fit requires both initial public cue groups")
    witness.check(episodes)
    return witness, rows, groups


def _combine_context_work(work, heads):
    for name, default in _work().items():
        if name in ("elapsed_seconds", "failure_work_may_be_unobserved"):
            continue
        work[name] += sum(head["work"].get(name, default) for head in heads.values())
    work["failure_work_may_be_unobserved"] |= any(
        head["work"]["failure_work_may_be_unobserved"] for head in heads.values()
    )


def diagnose_context_train_linear_fit(episodes: tuple[DemonstrationEpisode, ...]) -> dict[str, Any]:
    """Fit each admitted public-cue head once, reusing the original fit owner.

    Inputs are complete causal TRAIN episodes. Outputs retain separate complete
    head models/work and scores in original episode order. There is no deployed
    learner, model restoration, game, final/probe source or fitting callback.
    """
    started = monotonic()
    witness, rows, groups = _context_groups(episodes)
    work = _work()
    work.update(admission_projection_rows=2 * rows, routing_projection_rows=len(episodes))
    heads: dict[str, Any] = {}
    report: dict[str, Any] = dict(
        definition=describe_context_linear_protocol()["definition"],
        role="train",
        status="running",
        reason=None,
        protocol=describe_context_linear_protocol(),
        heads=heads,
        scores=[],
        aggregate_scores=None,
        fit_witness=False,
        work=work,
        source_unchanged=None,
    )
    guards: list[tuple[str, dict[str, Any], Callable[[], None], dict[str, Any]]] = []

    def check():
        witness.check(episodes)
        if report["heads"] is not heads:
            report["heads"] = heads
            raise ValueError("paid context head container replaced")
        for cue, owner, guard, expected in guards:
            if heads.get(cue) is not owner:
                heads[cue] = owner
                raise ValueError("preceding paid context head replaced")
            try:
                guard()
            except ValueError:
                owner.clear()
                owner.update(deepcopy(expected))
                raise ValueError("preceding paid context head corrupted") from None

    try:
        for cue, sources in groups.items():
            check()
            try:
                head = diagnose_train_linear_fit(tuple(sources))
            except LinearFitAdmissionFailed as error:
                # Pre-fit projection failures already carry paid work. Retain
                # that report as well as ordinary returned fit failures.
                heads[cue] = error.report
                raise
            heads[cue] = head  # Preserve paid failures before any more work.
            check()
            if head["status"] != "complete" or head["source_unchanged"] is not True:
                raise ValueError(f"{cue} TRAIN linear fit incomplete")
            guard, expected = paid_report_guard(head)
            guards.append((cue, head, guard, expected))
        check()
        by_id = {s["episode_id"]: s for h in heads.values() for s in h["scores"]}
        scores = [by_id[e.episode_id] for e in episodes]
        probabilities = np.array([p for s in scores for p in s["probabilities"]], dtype=np.float64)
        targets = np.array([[s.action] for e in episodes for s in e.transitions], dtype=np.float64)
        aggregate = _score(probabilities, targets, work)
        check()
        report.update(
            scores=scores,
            aggregate_scores=aggregate,
            fit_witness=all(s["exact_sequence_accuracy"] == 1 for s in scores),
            status="complete",
            source_unchanged=True,
        )
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}"[:512])
        work["failure_work_may_be_unobserved"] = True
        try:
            witness.check(episodes)
            report["source_unchanged"] = True
        except ValueError:
            report["source_unchanged"] = False
    finally:
        _combine_context_work(work, heads)
        work["elapsed_seconds"] = monotonic() - started
    return report
