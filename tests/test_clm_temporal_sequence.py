"""Bounded temporal CLM orchestration on literal features and native tiny heads."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
from src.adapters.clm_head_learner import CLMHeadLearner
from src.app.clm_temporal_sequence import run_temporal_stream
from src.core.clm_head import CLMHeadPolicy, FrozenEncoderIdentity
from src.core.temporal_decision_sequence import CATALOG, temporal_examples


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def inputs(torch):
    identity = FrozenEncoderIdentity("fake", "literal", "literal", "last", "temporal-literal", 2)
    # Twelve declared raw frozen vectors; none is a learned semantic embedding.
    features = torch.tensor(
        [[(index + 1) / 13, (-1) ** index * (index + 2) / 14] for index in range(12)]
    )
    return identity, features, study.build_feature_batches(identity, features)


def owner(torch, identity, *, maximum=12):
    def linear(values):
        layer = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(layer)
        layer.in_features = layer.out_features = 2
        layer.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        layer.register_parameter("bias", None)
        return layer

    state, action = linear([[1.0, 0.1], [0.2, 0.9]]), linear([[0.8, -0.1], [0.1, 1.1]])
    state.eval()
    learner = CLMHeadLearner(
        state,
        action,
        torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32)),
        identity=identity,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=maximum, max_rows=4),
    )
    return learner, state, action


@pytest.fixture(scope="module")
def full_stream(torch):
    declared, features, (training, choices) = inputs(torch)
    learner, state, action = owner(torch, declared)
    frozen = deepcopy(state.state_dict()), deepcopy(action.state_dict())
    original_features = features.clone()
    caller_rng = torch.get_rng_state().clone()
    paid, observations = [], []
    report: dict[str, Any] = {}
    train = learner.train_batch

    def record_train(batch, targets):
        paid.append((batch.states.clone(), batch.actions.clone(), targets.group_codes))
        return train(batch, targets)

    learner.train_batch = record_train

    def save():
        observations.append(
            (learner.completed_updates, "checkpoints" in report, "contrasts" in report)
        )

    run_temporal_stream(learner, training, choices, report, save=save)
    assert learner.completed_updates == 12
    return dict(
        learner=learner,
        report=report,
        features=features,
        original_features=original_features,
        training=training,
        choices=choices,
        paid=paid,
        observations=observations,
        frozen=frozen,
        sources=(state, action),
        rng=caller_rng,
    )


def test_should_bind_fixed_train_and_development_texts_to_all_catalog_descriptions():
    states, actions = study.feature_texts()
    concepts = ("invoice", "legacy", "parcel", "password")
    instructions = "Choose the ticket workflow tool."
    assert states == tuple(
        query + "\n\n" + instructions
        for query in (*concepts, *(name + " please" for name in concepts))
    )
    assert actions == tuple(item.description for item in CATALOG)
    assert len(set(states)) == 8 and len(set(actions)) == 4


def test_should_bind_observation_time_targets_and_active_choice_order(torch):
    declared, features, (training, choices) = inputs(torch)
    texts, _ = study.feature_texts()
    catalog = {item.candidate_id: index for index, item in enumerate(CATALOG)}
    assert len(training) == 12 and tuple(choices) == (
        "baseline",
        "after_A",
        "after_B",
        "after_return",
    )
    for index, (batch, targets) in enumerate(training):
        phase = ("after_A", "after_B", "after_return")[index // 4]
        examples = temporal_examples(phase, "train", exposure=index % 4)
        expected_states = [
            texts.index(example.request.query + "\n\nChoose the ticket workflow tool.")
            for example in examples
        ]
        expected_actions = [8 + catalog[example.target_id] for example in examples]
        assert batch.role == "train" and batch.identity == declared
        assert torch.equal(batch.states, features[expected_states])
        assert torch.equal(batch.actions, features[expected_actions])
        assert targets.group_codes == tuple(catalog[example.target_id] for example in examples)
    assert training[0][1].group_codes == (3, 2, 1)
    assert training[4][1].group_codes == (1, 0, 0)
    assert training[8][1].group_codes == (1, 0, 1)
    for phase, batch in choices.items():
        allowed = temporal_examples(phase, "outer_selection")[0].request.allowed_candidates
        assert batch.role == "outer_selection"
        assert torch.equal(batch.states, features[4:8])
        assert torch.equal(
            batch.actions, features[[8 + catalog[item.candidate_id] for item in allowed]]
        )


def test_should_detach_factory_outputs_from_caller_features(torch):
    _, features, (training, choices) = inputs(torch)
    train_state, train_action = training[0][0].states.clone(), training[0][0].actions.clone()
    choice_state = choices["baseline"].states.clone()
    features.zero_()
    assert torch.equal(training[0][0].states, train_state)
    assert torch.equal(training[0][0].actions, train_action)
    assert torch.equal(choices["baseline"].states, choice_state)


def test_should_apply_all_twelve_observation_bound_updates_before_any_scoring(full_stream):
    case, report = full_stream, full_stream["report"]
    assert report["sequence_complete"] is True
    assert report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 36
    assert [row["update"] for row in report["rows"]] == list(range(1, 13))
    assert [row["phase"] for row in report["rows"]] == ["after_A"] * 4 + ["after_B"] * 4 + [
        "after_return"
    ] * 4
    assert [row["exposure"] for row in report["rows"]] == list(range(4)) * 3
    assert [(row["phase"], row["completed_updates"]) for row in report["raw_readouts"]] == [
        ("baseline", 0),
        ("after_A", 4),
        ("after_B", 8),
        ("after_return", 12),
    ]
    assert len(report["frozen_A_probabilities"]) == 4
    assert all(not scored and not contrasts for _, scored, contrasts in case["observations"])
    assert "checkpoints" not in report and "contrasts" not in report
    for index, (states, actions, codes) in enumerate(case["paid"]):
        expected, targets = case["training"][index]
        assert states.tolist() == expected.states.tolist()
        assert actions.tolist() == expected.actions.tolist()
        assert codes == targets.group_codes == tuple(report["rows"][index]["group_codes"])


def test_should_keep_delayed_unique_receipts_and_exact_native_work(full_stream):
    report = full_stream["report"]
    receipts = report["stream"]["receipts"]
    assert len(receipts) == 12 and len({row["sample_id"] for row in receipts}) == 12
    assert [row["observed_at"] for row in receipts] == list(range(0, 24, 2))
    assert (
        [row["arrived_at"] for row in receipts]
        == [row["applied_at"] for row in receipts]
        == list(range(1, 24, 2))
    )
    assert [row["update_number"] for row in receipts] == list(range(1, 13))
    work = report["learner_work"]
    assert work["train_attempts"] == work["commits"] == 12
    assert work["training_rows_attempted"] == 36
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 5
    assert work["prediction_attempts"] == work["prediction_returns"] == 5
    assert work["prediction_rows_attempted"] == 20 and work["choice_candidate_rows_attempted"] == 15
    assert (
        work["backward_attempts"] == work["optimizer_attempts"] == work["scheduler_attempts"] == 12
    )
    assert work["unreturned_work_unknown"] is False


def test_should_leave_features_frozen_control_and_caller_rng_unchanged(torch, full_stream):
    case = full_stream
    assert torch.equal(case["features"], case["original_features"])
    assert torch.equal(torch.get_rng_state(), case["rng"])
    assert not case["sources"][0].training and case["sources"][1].training
    for head, original in zip(case["sources"], case["frozen"]):
        for key, tensor in head.state_dict().items():
            assert torch.equal(tensor, original[key])


@pytest.fixture(scope="module")
def scored_stream(full_stream):
    report = deepcopy(full_stream["report"])
    report["control_probabilities"] = {
        version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")
    }
    original = deepcopy(report)
    result = study.score_sequence(report)
    assert report == original
    return report, result


def test_should_score_cached_controls_and_frozen_A_without_more_native_calls(
    full_stream, scored_stream
):
    raw, scores = scored_stream
    points = scores["checkpoints"]
    assert [(row["phase"], row["update"]) for row in points] == [
        ("baseline", 0),
        ("after_A", 4),
        ("after_B", 8),
        ("after_return", 12),
    ]
    assert all(
        len(row["learned"]["cases"]) == len(row["unchanged"]["cases"]) == 4 for row in points
    )
    assert "frozen_A" not in points[0] and all("frozen_A" in row for row in points[1:])
    for row in points:
        order = [case["order_control"] for case in row["learned"]["cases"]]
        assert all(item["probabilities_agree"] for item in order)
        assert row["learned"]["candidate_order_control"]["all_equal"] is all(
            item["agreement"] for item in order
        )
        for case in row["learned"]["cases"]:
            assert case["request"]["query"].endswith(" please")
            assert case["source_group"] == "development-please"
    assert points[1]["frozen_A"]["metrics"] == points[1]["learned"]["metrics"]
    assert points[2]["frozen_A"]["metrics"] == points[3]["frozen_A"]["metrics"]
    assert full_stream["learner"].work == raw["learner_work"]


def test_should_keep_current_v2_truth_and_conditional_recovery_interpretation(scored_stream):
    _, scores = scored_stream
    points = {row["phase"]: row for row in scores["checkpoints"]}
    contrasts = scores["contrasts"]
    assert contrasts["phase_differences"]["after_B_minus_after_A"]["same_policy_truth"] is False
    assert contrasts["phase_differences"]["after_return_minus_after_B"]["same_policy_truth"] is True
    password = {}
    for phase in ("after_A", "after_B", "after_return"):
        rows = points[phase]["learned"]["cases"]
        password[phase] = float(
            next(row for row in rows if row["concept"] == "password")["correct"]
        )
    assert contrasts["stable_password_accuracy"] == password
    assert contrasts["stable_password_recovery_demonstrated"] is (
        password["after_A"] == 1 and password["after_B"] == 0 and password["after_return"] == 1
    )
    for row in points["after_return"]["learned"]["cases"]:
        assert row["policy_version"] == "v2"
        expected = next(
            example
            for example in temporal_examples("after_return", "outer_selection")
            if example.concept == row["concept"]
        )
        assert row["target_id"] == expected.target_id


def test_should_preserve_exact_zero_nll_and_still_allowed_obsolete_errors(scored_stream):
    report = deepcopy(scored_stream[0])
    for row in report["raw_readouts"]:
        row["probabilities"] = [[0.0, 0.0, 1.0] for _ in range(4)]
    report["frozen_A_probabilities"] = [[0.0, 0.0, 1.0] for _ in range(4)]
    result = study.score_sequence(report)
    point = next(row for row in result["checkpoints"] if row["phase"] == "after_B")
    assert point["learned"]["metrics"]["nll_mean"] is None
    assert point["learned"]["metrics"]["nll_status"] == "infinite_due_to_zero_target_probability"
    assert point["learned"]["metrics"]["accuracy"] == 0
    assert point["learned"]["temporal"]["obsolete_rule_errors"] == 2
    legacy = next(row for row in point["learned"]["cases"] if row["concept"] == "legacy")
    invoice = next(row for row in point["learned"]["cases"] if row["concept"] == "invoice")
    assert legacy["obsolete_target_available"] is False and legacy["obsolete_rule_error"] is False
    assert invoice["obsolete_target_available"] is True and invoice["obsolete_rule_error"] is True
    contrast = result["contrasts"]["by_phase"]["after_B"]["learned_minus_unchanged"]
    assert contrast["accuracy"] == -0.5
    assert (
        contrast["nll_mean"] is None
        and contrast["nll_status"] == "nonfinite_comparison_unavailable"
    )
    assert result["contrasts"]["stable_password_recovery_demonstrated"] is False
    assert result["cached_typed_decodes"] == 88


def test_should_preserve_paid_receipt_when_callback_changes_bound_features(torch):
    declared, _, (training, choices) = inputs(torch)
    learner, *_ = owner(torch, declared)
    report: dict[str, Any] = {}
    saves = []

    def save():
        saves.append(deepcopy(report))
        if learner.completed_updates == 1:
            training[1][0].states.add_(0.1)

    with pytest.raises(ValueError, match="feature"):
        run_temporal_stream(learner, training, choices, report, before_step=lambda: None, save=save)
    assert report["sequence_complete"] is False
    assert report["stream"]["completed_updates"] == learner.completed_updates == 1
    assert report["stream"]["training_pairs"] == 3
    assert len(report["stream"]["receipts"]) == len(report["rows"]) == 1
    assert len(report["raw_readouts"]) == 1 and report["frozen_A_probabilities"] is None
    assert "checkpoints" not in report and "contrasts" not in report
    assert report["learner_work"]["train_attempts"] == report["learner_work"]["commits"] == 1
    assert report["learner_work"]["choice_prediction_attempts"] == 1
    assert saves[-1]["stream"]["receipts"] == report["stream"]["receipts"]
    with pytest.raises(ValueError):
        study.score_sequence(report)


@pytest.fixture(scope="module")
def fresh_inputs(torch):
    declared, features, batches = inputs(torch)
    learner, *_ = owner(torch, declared)
    return learner, declared, features, batches


@pytest.mark.parametrize(
    "fault",
    [
        "train_role",
        "choice_role",
        "identity",
        "phase_targets",
        "train_order",
        "missing_choice",
        "shape",
        "spent",
    ],
)
def test_should_refuse_invalid_or_spent_stream_before_any_prediction(
    torch, monkeypatch, fresh_inputs, fault
):
    learner, declared, features, _ = fresh_inputs
    training, choices = study.build_feature_batches(declared, features)
    if fault == "train_role":
        object.__setattr__(training[0][0], "role", "probe_train")
    elif fault == "choice_role":
        object.__setattr__(choices["after_B"], "role", "final_test")
    elif fault == "identity":
        object.__setattr__(training[0][0], "identity", replace(declared, revision="foreign"))
    elif fault == "phase_targets":
        object.__setattr__(training[4][1], "group_codes", (3, 2, 1))
    elif fault == "train_order":
        packets = list(training)
        packets[0], packets[4] = packets[4], packets[0]
        training = tuple(packets)
    elif fault == "missing_choice":
        choices.pop("after_return")
    elif fault == "shape":
        object.__setattr__(choices["after_B"], "states", choices["after_B"].states[:3])
    with monkeypatch.context() as patch:
        if fault == "spent":
            patch.setitem(learner._work, "train_attempts", 1)
        before = learner.work
        with pytest.raises(ValueError):
            run_temporal_stream(
                learner, training, choices, {}, before_step=lambda: None, save=lambda: None
            )
        assert learner.work == before and learner.completed_updates == 0


def test_should_refuse_nonprospective_update_budget_before_baseline(torch):
    declared, _, (training, choices) = inputs(torch)
    learner, *_ = owner(torch, declared, maximum=8)
    before = learner.work
    with pytest.raises(ValueError):
        run_temporal_stream(
            learner, training, choices, {}, before_step=lambda: None, save=lambda: None
        )
    assert learner.work == before


@pytest.mark.parametrize("fault", ["rows", "width", "dtype", "nonfinite"])
def test_should_refuse_unbound_raw_features_at_factory_boundary(torch, fault):
    declared, features, _ = inputs(torch)
    if fault == "rows":
        features = features[:11]
    elif fault == "width":
        features = features[:, :1]
    elif fault == "dtype":
        features = features.double()
    else:
        features[0, 0] = float("nan")
    with pytest.raises(ValueError):
        study.build_feature_batches(declared, features)


@pytest.mark.parametrize("fault", ["partial", "counter", "unknown", "phase"])
def test_should_refuse_partial_or_unbound_scoring_before_cached_decodes(
    scored_stream, monkeypatch, fault
):
    report = deepcopy(scored_stream[0])
    if fault == "partial":
        report["sequence_complete"] = False
    elif fault == "counter":
        report["stream"]["training_pairs"] = 35
    elif fault == "unknown":
        report["learner_work"]["unreturned_work_unknown"] = True
    else:
        report["raw_readouts"][0]["phase"] = "after_A"

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound stream reached cached evaluation")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    with pytest.raises(ValueError):
        study.score_sequence(report)


def test_should_refuse_existing_cli_evidence_before_native_loader(monkeypatch, tmp_path):
    output = tmp_path / "paid.json"
    output.write_text("preserved paid result", encoding="utf8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(output),
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("existing evidence reached native execution")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(ValueError, match="refusing"):
        study.main()
    assert output.read_text(encoding="utf8") == "preserved paid result"


def test_should_retain_paid_baseline_and_refuse_wrong_control_before_first_train(torch):
    declared, _, (training, choices) = inputs(torch)
    learner, *_ = owner(torch, declared)
    report = {"control_probabilities": {"v1": [[1.0, 0.0, 0.0] for _ in range(4)]}}
    with pytest.raises(ValueError, match="unchanged released control"):
        run_temporal_stream(learner, training, choices, report)
    assert report["sequence_complete"] is False
    assert report["initial_control_max_abs_error"] > 2e-6
    assert (
        len(report["raw_readouts"]) == 1 and report["raw_readouts"][0]["probabilities"] is not None
    )
    assert report["rows"] == [] and report["stream"]["receipts"] == []
    assert (
        report["learner_work"]["choice_prediction_attempts"]
        == report["learner_work"]["choice_prediction_returns"]
        == 1
    )
    assert report["learner_work"]["train_attempts"] == report["learner_work"]["commits"] == 0
