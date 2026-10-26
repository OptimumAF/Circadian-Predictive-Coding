"""Opt-in TRAIN catalog observations; fake features, literal native heads only."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
from src.app.clm_temporal_sequence import run_temporal_stream
from src.core.temporal_decision_sequence import PHASES, temporal_examples
from test_clm_temporal_sequence import inputs, owner


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


@pytest.fixture(scope="module")
def diagnostic_stream(torch):
    declared, features, (training, choices) = inputs(torch)
    learner, state, action = owner(torch, declared)
    diagnostics = study.build_train_choices(training, choices)
    source = deepcopy(state.state_dict()), deepcopy(action.state_dict())
    original_features, rng = features.clone(), torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    observations = []

    def save():
        observations.append(
            (
                learner.completed_updates,
                len(report.get("train_raw_readouts", [])),
                "train_catalog_diagnostics" in report,
            )
        )

    run_temporal_stream(
        learner, training, choices, report, train_choice_batches=diagnostics, save=save
    )
    return dict(
        learner=learner,
        training=training,
        choices=choices,
        diagnostics=diagnostics,
        report=report,
        observations=observations,
        features=features,
        original_features=original_features,
        rng=rng,
        sources=(state, action),
        source=source,
    )


def test_should_clone_exact_train_states_and_complete_active_candidate_features(torch):
    _, _, (training, choices) = inputs(torch)
    diagnostic = study.build_train_choices(training, choices)
    assert tuple(diagnostic) == PHASES
    for index, phase in enumerate(PHASES):
        batch = diagnostic[phase]
        assert batch.role == "train" and batch.identity == training[4 * index][0].identity
        assert torch.equal(batch.states, training[4 * index][0].states)
        assert torch.equal(batch.actions, choices[phase].actions)
        assert batch.states.data_ptr() != training[4 * index][0].states.data_ptr()
        assert batch.actions.data_ptr() != choices[phase].actions.data_ptr()
    assert 3 not in training[4][1].group_codes and 3 not in training[8][1].group_codes
    # Full-catalog route remains observable even when absent as an in-batch negative.
    assert torch.equal(diagnostic["after_B"].actions[2], choices["after_B"].actions[2])
    original_state, original_action = (
        training[0][0].states.clone(),
        choices["after_A"].actions.clone(),
    )
    diagnostic["after_A"].states.zero_()
    diagnostic["after_A"].actions.zero_()
    assert torch.equal(training[0][0].states, original_state)
    assert torch.equal(choices["after_A"].actions, original_action)


def test_should_add_only_three_unscored_catalog_observations_after_phase_completion(
    diagnostic_stream,
):
    case, report = diagnostic_stream, diagnostic_stream["report"]
    assert report["sequence_complete"] is True
    assert [
        (row["phase"], row["completed_updates"]) for row in report["train_raw_readouts"]
    ] == list(zip(PHASES, (4, 8, 12)))
    assert len(report["raw_readouts"]) == 4 and len(report["frozen_A_probabilities"]) == 4
    assert all(
        len(row["probabilities"]) == 3 and all(len(values) == 3 for values in row["probabilities"])
        for row in report["train_raw_readouts"]
    )
    assert all(not scored for _, _, scored in case["observations"])
    assert all(count == 0 for updates, count, _ in case["observations"] if updates < 4)
    assert all(count <= 1 for updates, count, _ in case["observations"] if updates < 8)
    assert all(count <= 2 for updates, count, _ in case["observations"] if updates < 12)
    assert "train_catalog_diagnostics" not in report


def test_should_keep_same_train_dose_receipts_and_exact_additional_readout_cost(diagnostic_stream):
    report = diagnostic_stream["report"]
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 36
    assert len(report["rows"]) == len(report["stream"]["receipts"]) == 12
    receipts = report["stream"]["receipts"]
    assert [row["observed_at"] for row in receipts] == list(range(0, 24, 2))
    assert (
        [row["arrived_at"] for row in receipts]
        == [row["applied_at"] for row in receipts]
        == list(range(1, 24, 2))
    )
    work = report["learner_work"]
    assert (
        work["train_attempts"]
        == work["commits"]
        == work["backward_attempts"]
        == work["optimizer_attempts"]
        == 12
    )
    assert work["training_rows_attempted"] == 36
    assert work["prediction_attempts"] == work["prediction_returns"] == 8
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 8
    assert work["prediction_rows_attempted"] == 29 and work["choice_candidate_rows_attempted"] == 24
    assert work["unreturned_work_unknown"] is False


def test_should_preserve_original_features_source_heads_modes_and_rng(torch, diagnostic_stream):
    case = diagnostic_stream
    assert torch.equal(case["features"], case["original_features"])
    assert torch.equal(torch.get_rng_state(), case["rng"])
    assert not case["sources"][0].training and case["sources"][1].training
    for head, before in zip(case["sources"], case["source"]):
        assert all(torch.equal(value, before[name]) for name, value in head.state_dict().items())


def test_should_score_all_bare_train_queries_from_cached_probabilities_only(diagnostic_stream):
    report = diagnostic_stream["report"]
    before = deepcopy(report)
    paid = diagnostic_stream["learner"].work
    scores = study.score_train_choices(report)
    assert report == before and diagnostic_stream["learner"].work == paid
    for point, raw, phase in zip(scores, report["train_raw_readouts"], PHASES):
        assert point["phase"] == phase and point["completed_updates"] == raw["completed_updates"]
        for row, values, example in zip(
            point["rows"], raw["probabilities"], temporal_examples(phase, "train")
        ):
            ids = [candidate.candidate_id for candidate in example.request.allowed_candidates]
            index = ids.index(example.target_id)
            winner = max(range(3), key=values.__getitem__)
            assert row["concept"] == row["query"] == example.concept
            assert row["candidate_ids"] == ids and row["probabilities"] == values
            assert row["target_id"] == example.target_id and row["selected_id"] == ids[winner]
            assert row["correct"] is (winner == index)
            assert row["target_probability"] == values[index]
            assert row["best_alternative_probability"] == max(
                value for other, value in enumerate(values) if other != index
            )
            assert row["target_minus_best_alternative"] == pytest.approx(
                values[index] - row["best_alternative_probability"]
            )
    scores[0]["rows"][0]["probabilities"].clear()
    assert report == before


def test_should_preserve_zero_tie_and_missing_obsolete_semantics(diagnostic_stream):
    report = deepcopy(diagnostic_stream["report"])
    matrices = (
        [[0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.2, 0.3, 0.5], [0.0, 0.0, 1.0], [0.6, 0.2, 0.2]],
        [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.5, 0.5, 0.0]],
    )
    for raw, matrix in zip(report["train_raw_readouts"], matrices):
        raw["probabilities"] = matrix
    points = study.score_train_choices(report)
    a, b, returned = [point["rows"] for point in points]
    assert a[0]["selected_id"] == "inspect" and a[0]["correct"] is False
    assert a[0]["target_minus_best_alternative"] == 0
    assert a[1]["best_alternative_probability"] == 0
    assert a[1]["target_minus_best_alternative"] == 1
    assert a[2]["target_probability"] == 0 and a[2]["target_minus_best_alternative"] == -1
    assert all(
        row["obsolete_target_probability"] is None and row["target_minus_obsolete"] is None
        for row in a
    )
    assert b[0]["obsolete_target_id"] == "route" and b[0]["obsolete_target_available"] is True
    assert b[0]["target_minus_obsolete"] == pytest.approx(-0.2)
    assert b[0]["target_minus_best_alternative"] == pytest.approx(-0.2)
    assert b[1]["obsolete_target_id"] == "retire" and b[1]["obsolete_target_available"] is False
    assert b[1]["obsolete_target_probability"] is None and b[1]["target_minus_obsolete"] is None
    assert b[2]["target_minus_obsolete"] == pytest.approx(0.4)
    assert (
        returned[2]["selected_id"] == "defer" and returned[2]["target_minus_best_alternative"] == 0
    )


@pytest.fixture(scope="module")
def fresh_inputs(torch):
    declared, features, _ = inputs(torch)
    learner, *_ = owner(torch, declared)
    return learner, declared, features


@pytest.mark.parametrize("fault", ["order", "role", "states", "actions", "identity", "missing"])
def test_should_refuse_misbound_train_diagnostics_before_any_work(torch, fresh_inputs, fault):
    learner, declared, features = fresh_inputs
    training, choices = study.build_feature_batches(declared, features)
    diagnostic = study.build_train_choices(training, choices)
    if fault == "order":
        diagnostic = dict(reversed(tuple(diagnostic.items())))
    elif fault == "role":
        object.__setattr__(diagnostic["after_B"], "role", "outer_selection")
    elif fault == "states":
        diagnostic["after_B"].states.add_(0.1)
    elif fault == "actions":
        object.__setattr__(
            diagnostic["after_B"], "actions", diagnostic["after_B"].actions[[2, 0, 1]]
        )
    elif fault == "identity":
        object.__setattr__(diagnostic["after_B"], "identity", replace(declared, model_id="foreign"))
    else:
        diagnostic.pop("after_return")
    before = learner.work
    with pytest.raises(ValueError):
        run_temporal_stream(learner, training, choices, {}, train_choice_batches=diagnostic)
    assert learner.work == before


def test_should_preserve_paid_train_readout_and_stop_on_future_diagnostic_mutation(torch):
    declared, _, (training, choices) = inputs(torch)
    learner, *_ = owner(torch, declared)
    diagnostic = study.build_train_choices(training, choices)
    report: dict[str, Any] = {}

    def save():
        if (
            report.get("train_raw_readouts")
            and report["train_raw_readouts"][0]["probabilities"] is not None
        ):
            diagnostic["after_B"].actions.add_(0.1)

    with pytest.raises(ValueError, match="feature"):
        run_temporal_stream(
            learner, training, choices, report, train_choice_batches=diagnostic, save=save
        )
    assert report["sequence_complete"] is False and learner.completed_updates == 4
    assert len(report["stream"]["receipts"]) == len(report["rows"]) == 4
    assert report["stream"]["training_pairs"] == 12
    assert len(report["train_raw_readouts"]) == 1
    assert report["train_raw_readouts"][0]["probabilities"] is not None
    assert len(report["raw_readouts"]) == 2 and report["frozen_A_probabilities"] is not None
    work = report["learner_work"]
    assert work["train_attempts"] == work["commits"] == 4
    assert work["training_rows_attempted"] == 12
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 4
    assert work["prediction_rows_attempted"] == 15 and work["choice_candidate_rows_attempted"] == 12
    with pytest.raises(ValueError):
        study.score_train_choices(report)


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "missing",
        "stale",
        "order",
        "work",
        "unknown",
        "shape",
        "mass",
        "bool",
        "nonfinite",
    ],
)
def test_should_refuse_stale_incomplete_or_invalid_scalar_diagnostics(diagnostic_stream, fault):
    report = deepcopy(diagnostic_stream["report"])
    if fault == "incomplete":
        report["sequence_complete"] = False
    elif fault == "missing":
        report["train_raw_readouts"].pop()
    elif fault == "stale":
        report["train_raw_readouts"][0]["completed_updates"] = 8
    elif fault == "order":
        report["train_raw_readouts"].reverse()
    elif fault == "work":
        report["learner_work"]["choice_prediction_returns"] = 7
    elif fault == "unknown":
        report["learner_work"]["unreturned_work_unknown"] = True
    elif fault == "shape":
        report["train_raw_readouts"][0]["probabilities"].pop()
    elif fault == "mass":
        report["train_raw_readouts"][0]["probabilities"][0] = [0.5, 0.5, 0.5]
    elif fault == "bool":
        report["train_raw_readouts"][0]["probabilities"][0] = [True, 0.0, 0.0]
    else:
        report["train_raw_readouts"][0]["probabilities"][0] = [float("nan"), 0.0, 1.0]
    with pytest.raises(ValueError):
        study.score_train_choices(report)


@pytest.mark.parametrize("enabled", [False, True])
def test_should_route_cli_flag_without_extra_native_work(monkeypatch, tmp_path, enabled):
    import json

    output = tmp_path / "new.json"
    args = [
        "clm-temporal",
        "--model-dir",
        str(tmp_path),
        "--head",
        str(tmp_path / "head.pt"),
        "--output",
        str(output),
    ]
    if enabled:
        args.append("--train-catalog-diagnostics")
    monkeypatch.setattr("sys.argv", args)
    calls = []

    def case(model_dir, head, report, *, train_catalog_diagnostics, save):
        calls.append(train_catalog_diagnostics)
        report.update(complete=True, fixture="metadata only", diagnostics=train_catalog_diagnostics)
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [enabled]
    assert json.loads(output.read_text(encoding="utf8"))["diagnostics"] is enabled
