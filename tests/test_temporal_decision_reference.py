"""Fixed temporal policy semantics and one bounded ordinary learned stream."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from typing import Any

import numpy as np
import pytest

from src.adapters import perceptron_decision as adapter
from src.adapters.perceptron_decision import build_decision_training_batch
from src.app.decision_reference_evaluation import DecisionCase, DecisionEvaluationFailed
from src.core.decision_features import DecisionFeatureSpec, encode_decision
from src.core.categorical_perceptron import PerceptronPolicy, PerceptronSnapshot
from src.core.temporal_decision_sequence import (
    CATALOG,
    VOCABULARY,
    policy_for_phase,
    temporal_examples,
    validate_temporal_examples,
)
from src.core.typed_decision import CandidateScore, DecisionCandidate, DecisionResult


def feature_spec():
    return DecisionFeatureSpec(VOCABULARY, CATALOG)


def train_cases(phase="after_A", *, exposure=0):
    return tuple(
        DecisionCase(row.request, row.target_id, row.source_group, row.role)
        for row in temporal_examples(phase, "train", exposure=exposure)
    )


def forbidden(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("refused temporal input reached projection or learning")


@pytest.mark.parametrize(
    "phase,version,allowed",
    [
        ("baseline", "v1", ("inspect", "retire", "route")),
        ("after_A", "v1", ("inspect", "retire", "route")),
        ("after_B", "v2", ("defer", "inspect", "route")),
        ("after_return", "v2", ("defer", "inspect", "route")),
    ],
)
def test_should_bind_active_tools_and_rules_to_observation_phase(phase, version, allowed):
    policy = policy_for_phase(phase)
    assert policy["version"] == version and policy["allowed_ids"] == allowed
    rows = temporal_examples(phase, "outer_selection")
    assert [row.concept for row in rows] == list(VOCABULARY)
    assert [row.target_id for row in rows] == (
        ["route", "retire", "route", "inspect"]
        if version == "v1"
        else ["inspect", "defer", "defer", "inspect"]
    )
    assert all(
        tuple(candidate.candidate_id for candidate in row.request.allowed_candidates) == allowed
        for row in rows
    )
    assert all(
        row.policy_version == version and row.role == "outer_selection" and row.exposure == 0
        for row in rows
    )
    validate_temporal_examples(rows)


def test_should_keep_old_labels_immutable_when_policy_changes_and_old_domain_returns():
    old = temporal_examples("after_A", "train")
    original = deepcopy(old)
    changed = temporal_examples("after_B", "train")
    returned = temporal_examples("after_return", "train")
    assert old == original and old[0].target_id == "route" and old[1].target_id == "retire"
    assert changed[0].target_id == returned[0].target_id == "inspect"
    assert changed[1].target_id == returned[1].target_id == "defer"
    assert returned[0].obsolete_target_id == "route" and returned[1].obsolete_target_id == "retire"
    assert returned[2].obsolete_target_id is None
    with pytest.raises(FrozenInstanceError):
        setattr(old[0], "target_id", "inspect")
    detached = policy_for_phase("after_return")
    detached["rules"]["invoice"] = "route"
    assert policy_for_phase("after_return")["rules"]["invoice"] == "inspect"


def test_should_keep_all_twelve_train_exposures_unique_and_separate_from_development_templates():
    packets = [
        temporal_examples(phase, "train", exposure=index)
        for phase in ("after_A", "after_B", "after_return")
        for index in range(4)
    ]
    train_ids = [row.request.request_id for rows in packets for row in rows]
    development = [
        row
        for phase in ("baseline", "after_A", "after_B", "after_return")
        for row in temporal_examples(phase, "outer_selection")
    ]
    assert len(train_ids) == len(set(train_ids)) == 36
    assert not set(train_ids).intersection(row.request.request_id for row in development)
    assert all(
        row.request.query == row.concept and row.source_group == "train-token"
        for rows in packets
        for row in rows
    )
    assert all(
        row.request.query == row.concept + " please" and row.source_group == "development-please"
        for row in development
    )
    assert [[row.concept for row in rows] for rows in packets] == [
        ["invoice", "legacy", "password"]
    ] * 4 + [["invoice", "legacy", "parcel"]] * 4 + [["invoice", "legacy", "password"]] * 4


@pytest.mark.parametrize(
    "phase,role,exposure",
    [
        ("baseline", "train", 0),
        ("future", "train", 0),
        ("after_A", "final", 0),
        ("after_A", "probe_train", 0),
        ("after_B", "outer_selection", 1),
        ("after_A", "train", True),
        ("after_A", "train", 4),
    ],
)
def test_should_refuse_undeclared_phase_role_or_exposure(phase, role, exposure):
    with pytest.raises(ValueError):
        temporal_examples(phase, role, exposure=exposure)


@pytest.mark.parametrize("fault", ["list", "missing", "reversed", "mixed_phase", "obsolete_label"])
def test_should_refuse_noncanonical_or_relabelled_temporal_packets(fault):
    original = temporal_examples("after_return", "train")
    bad: Any = original
    if fault == "list":
        bad = list(original)
    elif fault == "missing":
        bad = original[:2]
    elif fault == "reversed":
        bad = tuple(reversed(original))
    elif fault == "mixed_phase":
        bad = (temporal_examples("after_A", "train")[0], *original[1:])
    else:
        bad = deepcopy(original)
        object.__setattr__(bad[0], "target_id", "route")
    with pytest.raises(ValueError):
        validate_temporal_examples(bad)


def test_should_explicitly_admit_subsets_using_fixed_union_class_indices_and_own_arrays():
    spec = feature_spec()
    inputs = tuple(train_cases(phase) for phase in ("after_A", "after_B", "after_return"))
    original = deepcopy(inputs)
    packets = [
        build_decision_training_batch(rows, spec, phase, allow_subsets=True)
        for rows, phase in zip(inputs, ("A-v1", "B-v2", "A-v2"), strict=True)
    ]
    assert all(packet.num_classes == 4 and packet.role == "train" for packet in packets)
    np.testing.assert_array_equal(packets[0].targets, [[3.0], [2.0], [1.0]])
    np.testing.assert_array_equal(packets[1].targets, [[1.0], [0.0], [0.0]])
    np.testing.assert_array_equal(packets[2].targets, [[1.0], [0.0], [1.0]])
    np.testing.assert_array_equal(packets[0].features, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(packets[1].features, [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    packets[0].features.fill(-1)
    packets[0].targets.fill(0)
    assert inputs == original
    np.testing.assert_array_equal(packets[2].targets, [[1.0], [0.0], [1.0]])


def test_should_keep_identifier_phase_version_domain_and_allowed_tools_out_of_query_features():
    spec = feature_spec()
    old = temporal_examples("after_A", "train")[0]
    current = temporal_examples("after_return", "outer_selection")[0]
    old_features, old_visits = encode_decision(old.request, spec)
    current_features, current_visits = encode_decision(current.request, spec)
    np.testing.assert_array_equal(old_features, [[1, 0, 0, 0]])
    np.testing.assert_array_equal(current_features, old_features)
    assert old_visits == 1 and current_visits == 2
    assert old.request.request_id != current.request.request_id
    assert old.target_id != current.target_id and old.policy_version != current.policy_version


@pytest.mark.parametrize("flag", [False, 1, None])
def test_should_preserve_default_full_catalog_rule_and_require_explicit_boolean_opt_in(
    monkeypatch, flag
):
    monkeypatch.setattr(adapter, "encode_decision", forbidden)
    with pytest.raises(ValueError):
        build_decision_training_batch(train_cases(), feature_spec(), "phase-A", allow_subsets=flag)
    if flag is False:
        with pytest.raises(ValueError):
            build_decision_training_batch(train_cases(), feature_spec(), "phase-A")


@pytest.mark.parametrize("fault", ["retired_target", "unknown_tool", "development", "probe"])
def test_should_refuse_illegal_subset_targets_tools_or_roles_before_projection(monkeypatch, fault):
    cases = deepcopy(train_cases("after_B"))
    first = cases[0]
    if fault == "retired_target":
        first = replace(first, target_id="retire")
    elif fault == "unknown_tool":
        first = replace(
            first,
            request=replace(
                first.request,
                allowed_candidates=(
                    *first.request.allowed_candidates,
                    DecisionCandidate("unknown", "unadmitted tool"),
                ),
            ),
        )
    else:
        object.__setattr__(
            first, "role", "outer_selection" if fault == "development" else "probe_train"
        )
    monkeypatch.setattr(adapter, "encode_decision", forbidden)
    with pytest.raises(ValueError):
        build_decision_training_batch(
            (first, *cases[1:]), feature_spec(), "phase-B", allow_subsets=True
        )


@pytest.fixture(scope="module")
def literal_evaluation():
    from scripts import run_temporal_decision_reference as driver

    # Controlled readout, not a learned checkpoint: retired global winner must
    # be masked, while the still-allowed obsolete route can remain an error.
    state = PerceptronSnapshot(
        PerceptronPolicy(4, 4), ((0.0,) * 4,) * 4 + ((0.0, 0.0, 4.0, 2.0),), 4, 12, 5
    )
    before = deepcopy(state)
    result = driver._evaluate(
        feature_spec(), state, temporal_examples("after_B", "outer_selection")
    )
    assert state == before
    return result


def test_should_count_obsolete_still_allowed_choices_as_errors_and_distinguish_retired_masking(
    literal_evaluation,
):
    result = literal_evaluation
    rows = {row["concept"]: row for row in result["cases"]}
    assert all(row["result"]["selected_id"] == "route" for row in rows.values())
    assert all(
        "retire" not in {score["candidate_id"] for score in row["result"]["scores"]}
        for row in rows.values()
    )
    assert (
        rows["invoice"]["obsolete_target_available"]
        is rows["parcel"]["obsolete_target_available"]
        is True
    )
    assert rows["invoice"]["obsolete_rule_error"] is rows["parcel"]["obsolete_rule_error"] is True
    assert (
        rows["legacy"]["obsolete_target_id"] == "retire"
        and rows["legacy"]["obsolete_target_available"] is False
    )
    assert rows["legacy"]["obsolete_rule_error"] is False and rows["legacy"]["correct"] is False
    assert (
        result["temporal"]["obsolete_rule_errors"]
        == result["temporal"]["obsolete_rule_opportunities"]
        == 2
    )
    assert result["temporal"]["scored_cases"] == 4


def test_should_keep_frozen_literal_readout_costs_and_injected_behavior_unverified(
    literal_evaluation,
):
    result = literal_evaluation
    assert result["complete"] is True
    assert (
        result["work"]["prediction_calls_attempted"]
        == result["work"]["prediction_calls_validated"]
        == 8
    )
    assert (
        result["learning_active"]
        is result["training_updates"]
        is result["external_actions"]
        is None
    )
    assert result["predictor_behavior"] == "UNVERIFIED_CALLBACK_BEHAVIOR"
    assert result["candidate_order_control"]["all_equal"] is True
    assert set(result["by_domain"]) == {"A", "B"}


@pytest.fixture(scope="module")
def temporal_example():
    from scripts import run_temporal_decision_reference as driver
    from src.adapters.perceptron_learner import PerceptronLearner

    owners: list[PerceptronLearner] = []
    observed: dict[str, Any] = {"training_rows": [], "readout_calls": 0}
    original_init = PerceptronLearner.__init__
    original_train = PerceptronLearner.train_batch
    original_readout = adapter.PerceptronDecisionPredictor.__call__
    original_stream = driver.AttributionStream.__init__

    def construct(owner, policy):
        original_init(owner, policy)
        owners.append(owner)

    def train(owner, features, targets):
        observed["training_rows"].append(len(features))
        return original_train(owner, features, targets)

    def readout(owner, query):
        assert observed["training_rows"] == [3] * 12
        observed["readout_calls"] += 1
        return original_readout(owner, query)

    def construct_stream(stream, *args, **kwargs):
        observed["stream_options"] = {
            key: kwargs[key] for key in ("checkpoint_updates", "label_delay", "training_only")
        }
        original_stream(stream, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PerceptronLearner, "__init__", construct)
        patch.setattr(PerceptronLearner, "train_batch", train)
        patch.setattr(adapter.PerceptronDecisionPredictor, "__call__", readout)
        patch.setattr(driver.AttributionStream, "__init__", construct_stream)
        report = driver.run_example()
    return report, owners, observed


def test_should_finish_exact_fixed_train_dose_before_any_development_readout_and_preserve_source_actor(
    temporal_example,
):
    report, owners, observed = temporal_example
    assert report["complete"] is True, report.get("error")
    assert observed["training_rows"] == [3] * 12 and observed["readout_calls"] == 80
    assert observed["stream_options"] == {
        "checkpoint_updates": (0, 4, 8, 12),
        "label_delay": 1,
        "training_only": True,
    }
    assert len(owners) == 3
    assert [owner.completed_updates for owner in owners] == [0, 0, 12]
    assert [owner.training_examples for owner in owners] == [0, 0, 36]
    assert owners[0].snapshot_state() == owners[1].snapshot_state()
    assert all(value == 0.0 for row in owners[0].snapshot_state().coefficients for value in row)
    assert report["stream"]["evaluation_scope"] == "training_only"
    assert (
        report["stream"]["score_forward_calls"] == 0
        and len(report["stream"]["applied_updates"]) == 12
    )
    work = report["work"]
    assert work["updates_committed"] == 12 and work["training_rows_committed"] == 36
    assert (
        work["prediction_calls_attempted"]
        == work["prediction_calls_returned"]
        == work["prediction_calls_validated"]
        == 80
    )
    assert (
        work["retained_coefficient_bytes"] == 640 and work["coefficient_bytes_per_snapshot"] == 160
    )
    assert work["committed_training_affine_multiply_terms"] == 720
    assert work["validated_readout_affine_multiply_terms"] == 1200
    assert work["external_actions"] == work["replay_examples"] == work["model_random_draws"] == 0
    json.dumps(report, allow_nan=False)


def test_should_score_every_checkpoint_control_under_its_current_policy_without_retraining_controls(
    temporal_example,
):
    report, _, _ = temporal_example
    assert [row["update"] for row in report["snapshots"]] == [0, 4, 8, 12]
    assert [row["state"]["training_examples"] for row in report["snapshots"]] == [0, 12, 24, 36]
    points = report["checkpoints"]
    assert [(row["update"], row["phase"]) for row in points] == [
        (0, "baseline"),
        (4, "after_A"),
        (8, "after_B"),
        (12, "after_return"),
    ]
    assert [row["source_updates"] for row in points] == [
        {"learned": 0, "unchanged": 0},
        {"learned": 4, "unchanged": 0},
        {"learned": 8, "unchanged": 0, "frozen_A": 4},
        {"learned": 12, "unchanged": 0, "frozen_A": 4},
    ]
    for checkpoint in points:
        expected = temporal_examples(checkpoint["phase"], "outer_selection")
        for name in checkpoint["source_updates"]:
            evaluation = checkpoint[name]
            assert (
                len(evaluation["cases"]) == 4
                and evaluation["work"]["prediction_calls_validated"] == 8
            )
            assert evaluation["candidate_order_control"]["all_equal"] is True
            assert evaluation["temporal"]["policy_version"] == checkpoint["policy"]["version"]
            for row, example in zip(evaluation["cases"], expected, strict=True):
                assert (
                    row["target_id"] == example.target_id
                    and row["obsolete_target_id"] == example.obsolete_target_id
                )
                assert row["correct"] == (row["result"]["selected_id"] == example.target_id)
                assert row["obsolete_rule_error"] == (
                    example.obsolete_target_id is not None
                    and row["result"]["selected_id"] == example.obsolete_target_id
                )


def test_should_report_signed_current_rule_contrasts_and_require_prior_loss_for_recovery(
    temporal_example,
):
    report, _, _ = temporal_example
    contrasts = report["contrasts"]
    for checkpoint in report["checkpoints"]:
        learned = checkpoint["learned"]
        for control in ("unchanged", "frozen_A"):
            if control not in checkpoint:
                continue
            observed = contrasts["by_phase"][checkpoint["phase"]]["learned_minus_" + control]
            assert (
                observed["accuracy"]
                == learned["metrics"]["accuracy"] - checkpoint[control]["metrics"]["accuracy"]
            )
            for metric in ("brier_mean", "ece_5bin"):
                assert (
                    observed[metric]
                    == learned["metrics"][metric] - checkpoint[control]["metrics"][metric]
                )
            expected_nll = (
                learned["metrics"]["nll_mean"] - checkpoint[control]["metrics"]["nll_mean"]
            )
            assert (
                observed["nll_mean"] == expected_nll
                and observed["nll_status"] == "finite_difference"
            )
            assert (
                observed["obsolete_rule_error_count"]
                == learned["temporal"]["obsolete_rule_errors"]
                - checkpoint[control]["temporal"]["obsolete_rule_errors"]
            )
    points = {row["phase"]: row["learned"]["metrics"] for row in report["checkpoints"]}
    expected_pairs = [
        ("baseline", "after_A", True),
        ("after_A", "after_B", False),
        ("after_B", "after_return", True),
    ]
    assert set(contrasts["phase_differences"]) == {
        later + "_minus_" + earlier for earlier, later, _ in expected_pairs
    }
    for earlier, later, same_policy in expected_pairs:
        difference = contrasts["phase_differences"][later + "_minus_" + earlier]
        assert difference["same_policy_truth"] is same_policy
        for metric in ("accuracy", "brier_mean", "ece_5bin", "nll_mean"):
            assert difference[metric] == points[later][metric] - points[earlier][metric]
    password = contrasts["stable_password_accuracy"]
    assert contrasts["stable_password_recovery_demonstrated"] == (
        password["after_A"] == 1 and password["after_B"] < 1 and password["after_return"] == 1
    )
    returned = report["checkpoints"][-1]["learned"]["cases"]
    old_domain = [row for row in returned if row["domain"] == "A"]
    assert contrasts["current_rule_old_domain_return_success"] == all(
        row["correct"] for row in old_domain
    )
    assert report["checkpoints"][-1]["policy"]["version"] == "v2"


@pytest.mark.parametrize("infinite", [False, True])
def test_should_keep_metric_delta_direction_and_nonfinite_nll_unavailable(infinite):
    from scripts import run_temporal_decision_reference as driver

    before = {"accuracy": 0.5, "brier_mean": 0.4, "ece_5bin": 0.2, "nll_mean": 0.9}
    after = {
        "accuracy": 0.75,
        "brier_mean": 0.2,
        "ece_5bin": 0.1,
        "nll_mean": None if infinite else 0.5,
    }
    original = deepcopy((before, after))
    result = driver._metric_difference(after, before)
    assert result["accuracy"] == 0.25 and result["brier_mean"] == pytest.approx(-0.2)
    assert result["ece_5bin"] == pytest.approx(-0.1)
    if infinite:
        assert (
            result["nll_mean"] is None
            and result["nll_status"] == "nonfinite_comparison_unavailable"
        )
    else:
        assert (
            result["nll_mean"] == pytest.approx(-0.4)
            and result["nll_status"] == "finite_difference"
        )
    assert (before, after) == original


def test_should_keep_owned_temporal_annotation_and_paid_original_return_when_reverse_prediction_fails(
    monkeypatch,
):
    from scripts import run_temporal_decision_reference as driver

    calls: list[None] = []

    def predict(request):
        calls.append(None)
        assert not hasattr(request, "target_id") and not hasattr(request, "role")
        if len(calls) == 4:
            raise RuntimeError("controlled reversed temporal readout failure")
        identities = tuple(candidate.candidate_id for candidate in request.allowed_candidates)
        return DecisionResult(
            request.request_id,
            min(identities),
            tuple(CandidateScore(identity, 0.0, 1 / len(identities)) for identity in identities),
            1,
            len(identities),
        )

    monkeypatch.setattr(driver, "PerceptronDecisionPredictor", lambda spec, state: predict)
    state = PerceptronSnapshot(PerceptronPolicy(4, 4), ((0.0,) * 4,) * 5, 0, 0, 0)
    with pytest.raises(DecisionEvaluationFailed) as captured:
        driver._evaluate(feature_spec(), state, temporal_examples("after_B", "outer_selection"))
    failure = captured.value
    assert calls == [None] * 4 and failure.report["complete"] is False
    assert len(failure.report["cases"]) == 1
    assert failure.report["cases"][0]["concept"] == "invoice"
    assert failure.report["cases"][0]["obsolete_target_id"] == "route"
    assert failure.report["cases"][0]["policy_version"] == "v2"
    assert (
        failure.report["temporal"]["policy_version"] == "v2"
        and failure.report["temporal"]["scored_cases"] == 1
    )
    assert set(failure.report["by_domain"]) == {"A"}
    assert failure.work["prediction_calls_attempted"] == 4
    assert (
        failure.work["prediction_calls_returned"] == failure.work["prediction_calls_validated"] == 3
    )
    assert failure.report["pending_case"]["original_returned"]["validated"] is True
    detached_work = failure.work
    detached_work["prediction_calls_attempted"] = 0
    assert failure.work["prediction_calls_attempted"] == 4


def test_should_retain_four_paid_updates_when_stream_fails_before_any_scoring(monkeypatch):
    from scripts import run_temporal_decision_reference as driver

    original = driver.AttributionStream.run_until
    calls: list[int] = []

    def fail_after_four(stream, update):
        calls.append(update)
        original(stream, 4)
        raise RuntimeError("controlled failure after four paid temporal updates")

    monkeypatch.setattr(driver.AttributionStream, "run_until", fail_after_four)
    report = driver.run_example()
    assert calls == [12] and report["complete"] is False
    assert "four paid temporal updates" in report["error"]
    assert report["checkpoints"] == [] and "stream" not in report
    assert [row["update"] for row in report["snapshots"]] == [0, 4]
    assert report["snapshots"][-1]["state"]["training_examples"] == 12
    assert (
        report["work"]["updates_committed"] == 4 and report["work"]["training_rows_committed"] == 12
    )
    assert report["work"]["prediction_calls_attempted"] == 0
    assert report["work"]["retained_coefficient_bytes"] == 320
    assert "unavailable" in report["training_failure_evidence"]


def test_should_publish_one_stubbed_cli_report_without_repeating_learning(
    tmp_path, monkeypatch, capsys
):
    from scripts import run_temporal_decision_reference as driver

    calls: list[None] = []
    expected = {"complete": True, "work": {"updates_committed": 12}}

    def run():
        calls.append(None)
        return deepcopy(expected)

    monkeypatch.setattr(driver, "run_example", run)
    output = tmp_path / "temporal.json"
    driver.main(["--output", str(output)])
    assert calls == [None] and json.loads(output.read_bytes()) == expected
    assert json.loads(capsys.readouterr().out)["complete"] is True


def test_should_preserve_existing_cli_output_before_learning(tmp_path, monkeypatch):
    from scripts import run_temporal_decision_reference as driver

    output = tmp_path / "keep.json"
    output.write_bytes(b"original temporal report")
    monkeypatch.setattr(driver, "run_example", forbidden)
    with pytest.raises(FileExistsError):
        driver.main(["--output", str(output)])
    assert output.read_bytes() == b"original temporal report"


def test_should_publish_incomplete_cli_prefix_and_exit_without_retry(tmp_path, monkeypatch):
    from scripts import run_temporal_decision_reference as driver

    calls: list[None] = []
    expected = {"complete": False, "error": "controlled", "work": {"updates_committed": 4}}

    def run():
        calls.append(None)
        return deepcopy(expected)

    monkeypatch.setattr(driver, "run_example", run)
    output = tmp_path / "partial.json"
    with pytest.raises(SystemExit) as error:
        driver.main(["--output", str(output)])
    assert error.value.code == 1 and calls == [None]
    assert json.loads(output.read_bytes()) == expected
