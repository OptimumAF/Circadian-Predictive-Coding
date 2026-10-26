"""Fixed-feature routing behavior; one shared six-update stream integration.

Literal snapshot fixtures are controlled coefficient states, not learned histories.
The shared ``linear_example`` and one paid-failure test run the existing stream;
CLI tests stub it.
"""

from copy import deepcopy
from dataclasses import replace
from itertools import permutations
import json
from math import exp
from typing import Any

import numpy as np
import pytest

from src.adapters import perceptron_decision as adapter
from src.adapters.perceptron_decision import (
    PerceptronDecisionPredictor,
    build_decision_training_batch,
)
from src.app.decision_reference_evaluation import DecisionCase, evaluate_decision_reference
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.typed_decision import DecisionCandidate, DecisionRequest, validate_decision_result


def feature_spec() -> DecisionFeatureSpec:
    return DecisionFeatureSpec(
        ("blue", "red"),
        (
            DecisionCandidate("a", "red invoice"),
            DecisionCandidate("b", "blue password"),
            DecisionCandidate("c", "green shipping"),
        ),
    )


def request(query: str = "red", *, identifier: str = "request") -> DecisionRequest:
    return DecisionRequest(identifier, query, feature_spec().catalog)


def cases() -> tuple[DecisionCase, ...]:
    return (
        DecisionCase(request("red", identifier="one"), "a", "train-A", "train"),
        DecisionCase(request("blue", identifier="two"), "b", "train-A", "train"),
        DecisionCase(request("unknown", identifier="three"), "c", "train-A", "train"),
    )


def snapshot(
    coefficients: tuple[tuple[float, ...], ...] | None = None,
    *,
    large: bool = False,
) -> PerceptronSnapshot:
    return PerceptronSnapshot(
        PerceptronPolicy(2, 3),
        coefficients or ((0.0, 0.0, 0.0),) * 3,
        8 if large else 1,
        1000 if large else 3,
        1000 if large else 3,
    )


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("refused inputs must not reach projection or learning")


def test_should_encode_binary_unicode_query_presence_without_catalog_or_identifier_words():
    spec = DecisionFeatureSpec(("café", "red"), feature_spec().catalog)
    projected, visits = encode_decision(
        DecisionRequest("red-café-id", "CAFÉ café unknown unknown", spec.catalog), spec
    )
    assert projected.dtype == np.dtype(np.float64)
    assert projected.shape == (1, 2)
    np.testing.assert_array_equal(projected, [[1.0, 0.0]])
    assert visits == 4
    projected[0, 0] = 0.0
    np.testing.assert_array_equal(
        encode_decision(request("RED red blue"), feature_spec())[0], [[1, 1]]
    )


@pytest.mark.parametrize(
    "words", [(), ("red", "blue"), ("red", "red"), ("RED",), ("two words",), ["red"]]
)
def test_should_refuse_noncanonical_vocabulary(words):
    with pytest.raises(ValueError):
        DecisionFeatureSpec(words, feature_spec().catalog)


@pytest.mark.parametrize("kind", ["binary", "reversed", "duplicate"])
def test_should_refuse_catalog_without_stable_categorical_class_order(kind):
    catalog = feature_spec().catalog
    bad = (
        catalog[:2]
        if kind == "binary"
        else tuple(reversed(catalog))
        if kind == "reversed"
        else (catalog[0], catalog[0], catalog[2])
    )
    with pytest.raises(ValueError):
        DecisionFeatureSpec(("red",), bad)


def test_should_refuse_unknown_or_replaced_candidate_description():
    for candidate in (DecisionCandidate("other", "red"), DecisionCandidate("a", "replacement")):
        with pytest.raises(ValueError):
            encode_decision(DecisionRequest("changed", "red", (candidate,)), feature_spec())


def test_should_build_canonical_class_targets_and_keep_context_out_of_features():
    original = cases()
    before = deepcopy(original)
    reversed_allowed = tuple(
        replace(
            case,
            request=replace(
                case.request, allowed_candidates=tuple(reversed(case.request.allowed_candidates))
            ),
        )
        for case in original
    )
    first = build_decision_training_batch(original, feature_spec(), "phase-A")
    second = build_decision_training_batch(reversed_allowed, feature_spec(), "phase-B")
    assert first.role == "train" and first.context == "phase-A" and first.num_classes == 3
    assert first.row_ids == ("one", "two", "three") and first.episode is None
    np.testing.assert_array_equal(first.features, [[0, 1], [1, 0], [0, 0]])
    np.testing.assert_array_equal(first.targets, [[0], [1], [2]])
    np.testing.assert_array_equal(second.features, first.features)
    np.testing.assert_array_equal(second.targets, first.targets)
    first.features[:] = -1
    first.targets[:] = 2
    assert original == before
    np.testing.assert_array_equal(second.features, [[0, 1], [1, 0], [0, 0]])


@pytest.mark.parametrize("role", ["outer_selection", "final", "probe_train", "inner_guard"])
def test_should_refuse_nontrain_before_any_feature_projection(monkeypatch, role):
    bad = deepcopy(cases()[0])
    object.__setattr__(bad, "role", role)
    monkeypatch.setattr(adapter, "encode_decision", forbidden)
    with pytest.raises(ValueError):
        build_decision_training_batch((cases()[1], bad), feature_spec(), "phase-A")


@pytest.mark.parametrize("kind", ["duplicate", "subset", "target", "empty", "list"])
def test_should_refuse_training_packets_without_unique_complete_allowed_labels(kind):
    original = cases()
    bad: Any = original
    if kind == "duplicate":
        bad = (original[0], original[0])
    elif kind == "subset":
        bad = (
            replace(
                original[0],
                request=replace(original[0].request, allowed_candidates=feature_spec().catalog[:1]),
            ),
        )
    elif kind == "target":
        bad = (replace(original[0], target_id="absent"),)
    elif kind == "empty":
        bad = ()
    else:
        bad = list(original)
    with pytest.raises(ValueError):
        build_decision_training_batch(bad, feature_spec(), "phase-A")


def test_should_project_class_columns_to_native_affine_logits():
    predictor = PerceptronDecisionPredictor(
        feature_spec(), snapshot(((0.0, 2.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    )
    result = predictor(request("red red"))
    assert result.selected_id == "a"
    assert [score.candidate_id for score in result.scores] == ["a", "b", "c"]
    assert [score.raw_score for score in result.scores] == [3.0, 0.0, -1.0]
    denominator = exp(3) + 1 + exp(-1)
    assert [score.probability for score in result.scores] == pytest.approx(
        [exp(3) / denominator, 1 / denominator, exp(-1) / denominator]
    )
    assert result.token_visits == 2 and result.candidate_comparisons == 3
    validate_decision_result(request("red red"), result)


def test_should_mask_disallowed_global_winner_before_stable_softmax():
    predictor = PerceptronDecisionPredictor(
        feature_spec(),
        snapshot(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1000.0, 0.0, -1.0)), large=True),
    )
    query = DecisionRequest("mask", "unknown", feature_spec().catalog[1:])
    result = predictor(query)
    assert result.selected_id == "b" and result.candidate_comparisons == 2
    assert [score.raw_score for score in result.scores] == [0.0, -1.0]
    assert [score.probability for score in result.scores] == pytest.approx(
        [1 / (1 + exp(-1)), exp(-1) / (1 + exp(-1))]
    )
    assert all(score.probability > 0 for score in result.scores)


def test_should_keep_ties_and_probabilities_independent_of_candidate_order():
    predictor = PerceptronDecisionPredictor(feature_spec(), snapshot())
    results = [
        predictor(DecisionRequest("tie", "red", order))
        for order in permutations(feature_spec().catalog)
    ]
    assert all(result == results[0] for result in results)
    assert results[0].selected_id == "a"
    assert [score.probability for score in results[0].scores] == pytest.approx([1 / 3] * 3)


def test_should_own_snapshot_and_spec_and_never_mutate_them_during_readout():
    spec, state = feature_spec(), snapshot()
    predictor = PerceptronDecisionPredictor(spec, state)
    before = deepcopy((predictor.spec, predictor.state))
    assert predictor.spec is not spec and predictor.state is not state
    object.__setattr__(spec.catalog[0], "description", "caller mutation")
    object.__setattr__(state, "coefficients", ((3.0, 0.0, 0.0),) * 3)
    query = DecisionRequest("owned", "red", predictor.spec.catalog)
    result = predictor(query)
    object.__setattr__(result, "selected_id", "c")
    assert predictor(query).selected_id == "a"
    assert (predictor.spec, predictor.state) == before


@pytest.mark.parametrize("kind", ["dimension", "classes", "bool_counter", "coefficient"])
def test_should_refuse_incompatible_or_corrupted_native_snapshot(kind):
    state = snapshot()
    if kind == "dimension":
        state = PerceptronSnapshot(PerceptronPolicy(1, 3), ((0.0, 0.0, 0.0),) * 2, 1, 3, 3)
    elif kind == "classes":
        state = PerceptronSnapshot(PerceptronPolicy(2, 4), ((0.0, 0.0, 0.0, 0.0),) * 3, 1, 3, 3)
    elif kind == "bool_counter":
        object.__setattr__(state, "completed_updates", True)
    else:
        object.__setattr__(state, "coefficients", ((float("nan"), 0.0, 0.0),) * 3)
    with pytest.raises(ValueError):
        PerceptronDecisionPredictor(feature_spec(), state)


def test_should_revalidate_private_snapshot_on_every_readout():
    predictor = PerceptronDecisionPredictor(feature_spec(), snapshot())
    object.__setattr__(predictor.state, "training_examples", -1)
    with pytest.raises(ValueError):
        predictor(request())


def test_should_leave_injected_predictor_behavior_unverified_in_existing_evaluator():
    predictor = PerceptronDecisionPredictor(feature_spec(), snapshot())
    evaluation = evaluate_decision_reference(
        (replace(cases()[0], role="outer_selection", source_group="dev-A"),), predict=predictor
    )
    assert evaluation["complete"] is True
    assert evaluation["learning_active"] is None
    assert evaluation["training_updates"] is None and evaluation["external_actions"] is None
    assert evaluation["predictor_behavior"] == "UNVERIFIED_CALLBACK_BEHAVIOR"
    assert evaluation["work"]["prediction_calls_validated"] == 2


@pytest.fixture(scope="module")
def linear_example():
    from scripts import run_linear_decision_reference as driver
    from src.adapters.perceptron_learner import PerceptronLearner

    owners: list[PerceptronLearner] = []
    observed: dict[str, Any] = {"training_rows": [], "readout_calls": 0}
    original_init = PerceptronLearner.__init__
    original_train = PerceptronLearner.train_batch
    original_readout = PerceptronDecisionPredictor.__call__

    def construct(owner, policy):
        original_init(owner, policy)
        owners.append(owner)

    def train(owner, features, targets):
        observed["training_rows"].append(len(features))
        return original_train(owner, features, targets)

    def readout(owner, query):
        # Development scoring must start only after the entire fixed TRAIN dose.
        assert observed["training_rows"] == [3] * 6
        observed["readout_calls"] += 1
        return original_readout(owner, query)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PerceptronLearner, "__init__", construct)
        patch.setattr(PerceptronLearner, "train_batch", train)
        patch.setattr(PerceptronDecisionPredictor, "__call__", readout)
        report = driver.run_example()
    return report, owners, observed


def test_should_execute_one_fixed_six_update_stream_with_immutable_source_and_actor(linear_example):
    report, owners, observed = linear_example
    assert report["complete"] is True, report.get("error")
    assert observed == {"training_rows": [3] * 6, "readout_calls": 48}
    assert len(owners) == 3
    assert [owner.completed_updates for owner in owners] == [0, 0, 6]
    assert [owner.training_examples for owner in owners] == [0, 0, 18]
    assert owners[0].snapshot_state() == owners[1].snapshot_state()
    assert all(value == 0.0 for row in owners[0].snapshot_state().coefficients for value in row)
    stream = report["stream"]
    assert stream["evaluation_scope"] == "training_only"
    assert stream["completed_updates"] == 6 and stream["training_examples"] == 18
    assert stream["score_forward_calls"] == 0 and stream["checkpoints"] == []
    assert len(stream["applied_updates"]) == 6
    assert report["work"]["updates_committed"] == 6
    assert report["work"]["training_rows_committed"] == 18
    assert report["work"]["replay_examples"] == 0
    assert report["work"]["coefficient_bytes_per_snapshot"] == 168
    assert report["work"]["retained_coefficient_bytes"] == 672
    assert report["work"]["committed_training_affine_multiply_terms"] == 378
    assert report["work"]["validated_readout_affine_multiply_terms"] == 1008
    json.dumps(report, allow_nan=False)


def test_should_keep_all_fixed_checkpoint_states_and_group_calibration(linear_example):
    report, owners, _ = linear_example
    assert [row["update"] for row in report["snapshots"]] == [0, 2, 4, 6]
    assert [row["state"]["completed_updates"] for row in report["snapshots"]] == [0, 2, 4, 6]
    assert [row["state"]["training_examples"] for row in report["snapshots"]] == [0, 6, 12, 18]
    assert (
        report["snapshots"][-1]["state"]["mistake_updates"]
        == owners[-1].snapshot_state().mistake_updates
    )
    assert report["work"]["latest_observed_mistake_update"] == 6
    assert (
        report["work"]["coefficient_update_additions_observed"]
        == 14 * owners[-1].snapshot_state().mistake_updates
    )
    assert [row["update"] for row in report["checkpoints"]] == [0, 2, 4, 6]
    for checkpoint in report["checkpoints"]:
        evaluation = checkpoint["evaluation"]
        assert evaluation["complete"] is True
        assert len(evaluation["cases"]) == 6
        assert evaluation["candidate_order_control"]["all_equal"] is True
        assert evaluation["learning_active"] is None
        assert evaluation["work"]["prediction_calls_validated"] == 12
        assert set(checkpoint["by_group"]) == {"dev-A", "dev-B"}
        for group, metrics in checkpoint["by_group"].items():
            rows = [row for row in evaluation["cases"] if row["source_group"] == group]
            assert len(rows) == 3
            assert metrics["accuracy"] == sum(row["correct"] for row in rows) / 3
            assert metrics["brier_mean"] == pytest.approx(sum(row["brier"] for row in rows) / 3)
            assert metrics["candidate_recall"] == 1.0
            assert sum(item["support"] for item in metrics["calibration_bins"]) == 3
            assert {"nll_mean", "ece_5bin", "nll_status"} <= metrics.keys()


def test_should_report_signed_changes_without_selecting_a_favorable_checkpoint(linear_example):
    report, _, _ = linear_example
    points = {row["update"]: row["by_group"] for row in report["checkpoints"]}
    changes = report["accuracy_contrasts"]
    assert (
        changes["new_task_gain"] == points[4]["dev-B"]["accuracy"] - points[2]["dev-B"]["accuracy"]
    )
    assert (
        changes["old_task_retention"]
        == points[4]["dev-A"]["accuracy"] - points[2]["dev-A"]["accuracy"]
    )
    assert (
        changes["return_recovery"]
        == points[6]["dev-A"]["accuracy"] - points[4]["dev-A"]["accuracy"]
    )
    assert "higher is favorable" in changes["definition"]


def test_should_keep_declared_train_words_and_development_labels_in_separate_packets():
    from scripts.run_linear_decision_reference import example_data

    spec, training, development = example_data()
    assert spec.vocabulary == ("credentials", "invoice", "parcel", "password", "refund", "tracking")
    assert [batch.context for batch in training] == ["A", "A", "B", "B", "A", "A"]
    assert all(batch.role == "train" and len(batch.features) == 3 for batch in training)
    train_ids = {identifier for batch in training for identifier in batch.row_ids}
    assert len(train_ids) == 18
    assert all(case.role == "outer_selection" for case in development)
    assert not train_ids.intersection(case.request.request_id for case in development)
    assert {case.source_group for case in development} == {"dev-A", "dev-B"}
    assert all(case.request.query.endswith(" please") for case in development)


def test_should_retain_paid_checkpoint_prefix_when_training_stops_before_finish(monkeypatch):
    from scripts import run_linear_decision_reference as driver

    original = driver.AttributionStream.run_until
    calls: list[int] = []

    def fail_after_two(stream, update):
        calls.append(update)
        original(stream, 2)
        raise RuntimeError("controlled failure after two paid updates")

    monkeypatch.setattr(driver.AttributionStream, "run_until", fail_after_two)
    report = driver.run_example()
    assert calls == [6]
    assert report["complete"] is False
    assert "controlled failure after two paid updates" in report["error"]
    assert report["checkpoints"] == [] and "stream" not in report
    assert [row["update"] for row in report["snapshots"]] == [0, 2]
    assert report["snapshots"][-1]["state"]["training_examples"] == 6
    assert report["work"]["updates_committed"] == 2
    assert report["work"]["training_rows_committed"] == 6
    assert report["work"]["retained_coefficient_bytes"] == 336
    assert report["work"]["committed_training_affine_multiply_terms"] == 126
    assert report["work"]["validated_readout_affine_multiply_terms"] == 0
    assert report["training_failure_evidence"]["receipts_and_diagnostics"].startswith(
        "unavailable:"
    )
    assert "unknown on failure" in report["work"]["attempted_uncommitted_training_work"]


@pytest.mark.parametrize("destination", ["stdout", "file"])
def test_should_publish_one_cli_report_without_repeating_training(
    monkeypatch, tmp_path, capsys, destination
):
    from scripts import run_linear_decision_reference as driver

    calls: list[None] = []
    expected = {"definition": "typed_linear_routing_v1", "complete": True, "checkpoints": []}

    def run():
        calls.append(None)
        return deepcopy(expected)

    monkeypatch.setattr(driver, "run_example", run)
    path = tmp_path / "linear.json"
    driver.main([] if destination == "stdout" else ["--output", str(path)])
    actual = json.loads(capsys.readouterr().out if destination == "stdout" else path.read_text())
    assert actual == expected and calls == [None]


def test_should_refuse_existing_cli_output_before_learning(monkeypatch, tmp_path):
    from scripts import run_linear_decision_reference as driver

    path = tmp_path / "keep.json"
    path.write_text("original")
    monkeypatch.setattr(driver, "run_example", forbidden)
    with pytest.raises(SystemExit) as error:
        driver.main(["--output", str(path)])
    assert error.value.code == 2 and path.read_text() == "original"


def test_should_publish_incomplete_observed_work_and_fail_without_retry(monkeypatch, tmp_path):
    from scripts import run_linear_decision_reference as driver

    calls: list[None] = []
    partial = {
        "complete": False,
        "error": "controlled failure",
        "snapshots": [{"update": 2}],
        "work": {"updates_committed": 2},
    }

    def fail():
        calls.append(None)
        return deepcopy(partial)

    monkeypatch.setattr(driver, "run_example", fail)
    path = tmp_path / "partial.json"
    with pytest.raises(SystemExit) as error:
        driver.main(["--output", str(path)])
    assert error.value.code == 1
    assert json.loads(path.read_text()) == partial and calls == [None]
