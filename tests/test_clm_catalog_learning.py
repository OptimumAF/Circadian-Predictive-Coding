"""Explicit full-catalog objective and owned native updates on literal tiny heads."""

from copy import deepcopy
from dataclasses import replace
from math import exp, log, sqrt
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_catalog_learner as catalog
import src.adapters.clm_head_learner as native
from src.adapters.clm_head_probe import CLMHeadProbe
from src.app.clm_temporal_sequence import run_temporal_stream
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog import CLMCatalogTargets, FrozenCatalogTrainingBatch
from src.core.clm_head import (
    CLMHeadPolicy,
    CLMTargets,
    FrozenChoiceBatch,
    FrozenEmbeddingBatch,
    FrozenEncoderIdentity,
)
from src.core.clm_probe import FrozenProbeEmbeddingBatch
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.core.learner_ports import TrainingUpdateFailed
from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS, PHASES, temporal_examples

IDS = ("alpha", "beta", "gamma")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def identity():
    return FrozenEncoderIdentity("fake", "literal", "literal", "last", "catalog-learning", 2)


def parts(torch) -> tuple[Any, Any, Any]:
    def linear(values):
        layer = torch.nn.Linear.__new__(torch.nn.Linear)
        torch.nn.Module.__init__(layer)
        layer.in_features = layer.out_features = 2
        layer.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
        layer.register_parameter("bias", None)
        return layer

    state, action = linear([[1.0, 0.2], [-0.1, 0.9]]), linear([[0.8, -0.2], [0.3, 1.1]])
    state.eval()
    return state, action, torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float32))


def owner(torch, *, maximum=3, restores=1):
    source = parts(torch)
    learner = catalog.CLMCatalogLearner(
        *source,
        identity=identity(),
        policy=CLMHeadPolicy(
            projection_dim=2, max_updates=maximum, max_rows=4, max_restores=restores
        ),
    )
    return learner, source


def batch(torch):
    states = torch.tensor([[1.0, 0.3], [-0.4, 0.8]], dtype=torch.float32)
    actions = torch.tensor([[0.9, 0.1], [0.2, 1.0], [-0.6, 0.4]], dtype=torch.float32)
    return FrozenCatalogTrainingBatch(identity(), states, actions, IDS), CLMCatalogTargets(
        IDS, (0, 0)
    )


def temporal_features(torch):
    return torch.tensor(
        [[(index + 1) / 13, (-1) ** index * (index + 2) / 14] for index in range(12)],
        dtype=torch.float32,
    )


def scalar_logits(source, states, actions):
    """Independent Python matrix, normalization and exponential-scale arithmetic."""

    def project(vector, weights):
        projected = [sum(x * weight for x, weight in zip(vector, row)) for row in weights]
        norm = max(sqrt(sum(value * value for value in projected)), 1e-12)
        return [value / norm for value in projected]

    state, action, scale = source
    queries = [project(vector, state.weight.detach().tolist()) for vector in states]
    candidates = [project(vector, action.weight.detach().tolist()) for vector in actions]
    multiplier = min(exp(scale.detach().item()), 100.0)
    return [
        [multiplier * sum(x * y for x, y in zip(query, candidate)) for candidate in candidates]
        for query in queries
    ]


@pytest.fixture(scope="module")
def math_case(torch):
    source = parts(torch)
    packet, targets = batch(torch)
    states, actions = packet.states.clone(), packet.actions.clone().requires_grad_()
    indices = torch.tensor(targets.candidate_indices, dtype=torch.int64)
    loss, logits, labels = catalog.catalog_head_loss(*source, states, actions, indices)
    logits.retain_grad()
    expected_logits = scalar_logits(source, states.tolist(), actions.detach().tolist())
    probabilities = []
    for row in expected_logits:
        masses = [exp(value - max(row)) for value in row]
        probabilities.append([value / sum(masses) for value in masses])
    expected_loss = -sum(log(row[0]) for row in probabilities) / 2
    loss.backward()
    permutation = (1, 2, 0)
    permuted_indices = torch.tensor((2, 2), dtype=torch.int64)
    permuted = catalog.catalog_head_loss(
        *source, states, actions.detach()[list(permutation)], permuted_indices
    )
    return dict(
        source=source,
        packet=packet,
        targets=targets,
        states=states,
        actions=actions,
        loss=loss,
        logits=logits,
        labels=labels,
        probabilities=probabilities,
        expected_logits=expected_logits,
        expected_loss=expected_loss,
        permutation=permutation,
        permuted=permuted,
    )


def test_should_match_manual_rectangular_logits_cross_entropy_and_logit_gradient(torch, math_case):
    case = math_case
    assert case["logits"].shape == (2, 3)
    assert case["labels"].tolist() == [0, 0]
    torch.testing.assert_close(
        case["logits"], torch.tensor(case["expected_logits"]), rtol=1e-6, atol=1e-7
    )
    assert case["loss"].item() == pytest.approx(case["expected_loss"], abs=1e-7)
    expected_gradient = [
        [(p - int(column == 0)) / 2 for column, p in enumerate(row)]
        for row in case["probabilities"]
    ]
    torch.testing.assert_close(
        case["logits"].grad, torch.tensor(expected_gradient), rtol=1e-6, atol=1e-7
    )
    assert case["loss"].item() > 0  # Repeated labels do not remove competing catalog entries.


def test_should_backpropagate_into_both_competing_candidates_without_target_exposure(math_case):
    case = math_case
    assert case["targets"].candidate_indices == (0, 0)
    assert case["actions"].grad is not None
    assert all(case["actions"].grad[index].abs().sum().item() > 0 for index in (1, 2))
    assert all(
        parameter.grad is not None for head in case["source"][:2] for parameter in head.parameters()
    )
    assert case["source"][2].grad is not None


def test_should_preserve_loss_and_reorder_logits_when_candidate_ids_and_targets_permute(
    torch, math_case
):
    case = math_case
    loss, logits, labels = case["permuted"]
    assert labels.tolist() == [2, 2]
    torch.testing.assert_close(loss, case["loss"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(logits, case["logits"][:, list(case["permutation"])], rtol=0, atol=0)


def test_should_use_native_scale_cap_with_zero_scale_gradient(torch, math_case):
    case = math_case
    scale = torch.nn.Parameter(torch.tensor(log(1000), dtype=torch.float32))
    loss, logits, _ = catalog.catalog_head_loss(
        case["source"][0],
        case["source"][1],
        scale,
        case["packet"].states,
        case["packet"].actions,
        torch.tensor((0, 0), dtype=torch.int64),
    )
    loss.backward()
    assert scale.grad is not None and scale.grad.item() == 0
    assert logits.abs().max().item() <= 100.0


@pytest.mark.parametrize(
    "ids",
    (["a", "b"], ("a",), ("a", "a"), (" a", "b"), ("a\n", "b"), ("a" * 129, "b"), (True, "b")),
)
def test_should_reject_mutable_duplicate_or_unbounded_candidate_identifiers(ids):
    with pytest.raises(ValueError):
        FrozenCatalogTrainingBatch(identity(), None, None, ids)
    with pytest.raises(ValueError):
        CLMCatalogTargets(ids, (0, 0))


@pytest.mark.parametrize("indices", ([0, 0], (0,), (True, 0), (0, 3), (-1, 0), (0.0, 1)))
def test_should_reject_noncanonical_or_out_of_catalog_target_indices(indices):
    with pytest.raises(ValueError):
        CLMCatalogTargets(IDS, indices)


@pytest.mark.parametrize("role", ("probe_train", "probe_test", "outer_selection", "final_test"))
def test_should_admit_only_explicit_benchmark_train_catalog_records(role):
    with pytest.raises(ValueError):
        FrozenCatalogTrainingBatch(identity(), None, None, IDS, role)


@pytest.fixture(scope="module")
def fresh_owner(torch):
    learner, _ = owner(torch)
    return learner


@pytest.mark.parametrize(
    "fault",
    (
        "paired",
        "choice",
        "probe",
        "role",
        "identity",
        "target_type",
        "target_order",
        "target_rows",
        "batch_order",
        "extra",
        "state_rows",
        "candidate_rows",
        "width",
        "dtype",
        "nonfinite",
    ),
)
def test_should_refuse_invalid_catalog_training_before_any_native_work(torch, fresh_owner, fault):
    learner = fresh_owner
    packet, targets = batch(torch)
    if fault == "paired":
        packet = FrozenEmbeddingBatch(packet.identity, packet.states, packet.actions[:2])
    elif fault == "choice":
        packet = FrozenChoiceBatch(packet.identity, packet.states, packet.actions)
    elif fault == "probe":
        packet = FrozenProbeEmbeddingBatch(packet.identity, packet.states, packet.actions[:2])
    elif fault == "role":
        object.__setattr__(packet, "role", "probe_train")
    elif fault == "identity":
        object.__setattr__(packet, "identity", replace(packet.identity, revision="foreign"))
    elif fault == "target_type":
        targets = CLMTargets((0, 0))
    elif fault == "target_order":
        targets = CLMCatalogTargets(tuple(reversed(IDS)), (0, 0))
    elif fault == "target_rows":
        targets = CLMCatalogTargets(IDS, (0, 0, 0))
    elif fault == "batch_order":
        object.__setattr__(packet, "candidate_ids", tuple(reversed(IDS)))
    elif fault == "extra":
        object.__setattr__(packet, "target", 0)
    elif fault == "state_rows":
        object.__setattr__(packet, "states", packet.states[:1])
    elif fault == "candidate_rows":
        object.__setattr__(packet, "actions", packet.actions[:2])
    elif fault == "width":
        object.__setattr__(packet, "states", packet.states[:, :1])
    elif fault == "dtype":
        object.__setattr__(packet, "actions", packet.actions.double())
    else:
        packet.actions[0, 0] = float("nan")
    work = learner.work
    with pytest.raises(ValueError):
        learner.train_batch(packet, targets)
    assert learner.work == work and learner.completed_updates == 0


@pytest.fixture(scope="module")
def trajectory(torch):
    learner, source = owner(torch)
    packet, targets = batch(torch)
    rng = torch.get_rng_state().clone()
    source_weights = [deepcopy(head.state_dict()) for head in source[:2]]
    source_modes = tuple(head.training for head in source[:2])
    empty = learner.snapshot_state()
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=3), lambda: 0.0)
    inbox = ExperienceInbox(
        learner, clock=clock, budget=budget, learner_version="catalog", max_experiences=3
    )
    inbox.record_experience(
        Experience(
            "sample", "episode", 0, "actor", packet, "train", ExperiencePermissions(training=True)
        )
    )
    assert inbox.drain(max_updates=1) == ()
    assert learner.completed_updates == 0 and learner.work["train_attempts"] == 0
    clock.advance_to(1)
    inbox.record_label(LabelArrival("label", "sample", "episode", 1, "actor", targets, "train"))
    receipt = inbox.drain(max_updates=1)
    assert len(receipt) == 1
    second = learner.train_batch(packet, targets)
    learned = learner.snapshot_state()
    paid = learner.work
    learner.restore_state(empty)
    restored_work = learner.work
    assert learner.completed_updates == learner.training_examples == 0
    assert (
        restored_work["catalog_candidate_rows_attempted"]
        == paid["catalog_candidate_rows_attempted"]
        == 6
    )
    assert (
        restored_work["catalog_logit_values_attempted"]
        == paid["catalog_logit_values_attempted"]
        == 12
    )
    third = learner.train_batch(packet, targets)
    paired = FrozenEmbeddingBatch(identity(), packet.states, packet.actions[:2])
    choice = FrozenChoiceBatch(identity(), packet.states, packet.actions)
    before_read = learner._capture()
    learner.predict(paired)
    learner.predict_choices(choice)
    assert native._equal(vars(before_read), vars(learner._capture()))
    return dict(
        learner=learner,
        source=source,
        source_weights=source_weights,
        source_modes=source_modes,
        rng=rng,
        packet=packet,
        targets=targets,
        empty=empty,
        learned=learned,
        receipt=receipt[0],
        diagnostics=(second, third),
    )


def test_should_commit_catalog_objective_through_delayed_inbox_and_native_state(torch, trajectory):
    case, learner = trajectory, trajectory["learner"]
    assert type(case["empty"]) is type(case["learned"]) is catalog.CLMCatalogSnapshot
    assert (
        case["receipt"].observed_at == 0
        and case["receipt"].arrived_at == case["receipt"].applied_at == 1
    )
    assert (
        case["receipt"].diagnostic.definition
        == catalog.OBJECTIVE
        == "clm_full_catalog_cross_entropy_v1"
    )
    assert all(diagnostic.definition == catalog.OBJECTIVE for diagnostic in case["diagnostics"])
    assert learner.completed_updates == 1 and learner.training_examples == 2
    work = learner.work
    assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 3
    assert work["training_rows_attempted"] == 6
    assert work["catalog_candidate_rows_attempted"] == 9
    assert work["catalog_logit_values_attempted"] == 18
    assert work["snapshot_attempts"] == work["snapshot_returns"] == 2
    assert work["restore_attempts"] == work["restore_returns"] == 1
    assert work["prediction_returns"] == 2 and work["choice_prediction_returns"] == 1
    assert work["unreturned_work_unknown"] is False
    assert torch.equal(case["rng"], torch.get_rng_state())
    for head, expected, mode in zip(
        case["source"][:2], case["source_weights"], case["source_modes"]
    ):
        assert native._equal(head.state_dict(), expected) and head.training is mode
    assert not native._equal(learner._bundle[0].state_dict(), case["source"][0].state_dict())
    for owned, original in zip(learner._bundle[:2], case["source"][:2]):
        assert all(
            a is not b and a.data_ptr() != b.data_ptr()
            for a, b in zip(owned.parameters(), original.parameters())
        )


def test_should_keep_update_and_catalog_exposure_caps_permanent_after_restore(trajectory):
    case, learner = trajectory, trajectory["learner"]
    before = learner.work
    with pytest.raises(ValueError, match="allowance exhausted"):
        learner.train_batch(case["packet"], case["targets"])
    with pytest.raises(ValueError, match="restore allowance exhausted"):
        learner.restore_state(case["empty"])
    assert learner.work == before


@pytest.fixture(scope="module")
def cross_objective(torch, trajectory):
    case = trajectory
    paired = native.CLMHeadLearner(
        *case["source"], identity=identity(), policy=case["learner"].policy
    )
    return paired, paired.snapshot_state()


def test_should_refuse_cross_objective_snapshots_and_probe_source_without_native_work(
    trajectory, cross_objective
):
    learner, paired = trajectory["learner"], cross_objective[0]
    catalog_work, paired_work = learner.work, paired.work
    with pytest.raises(ValueError):
        learner.restore_state(cross_objective[1])
    with pytest.raises(ValueError):
        paired.restore_state(trajectory["learned"])
    with pytest.raises(ValueError):
        CLMHeadProbe(
            learner, policy=replace(learner.policy, max_updates=3, max_restores=0), max_readouts=1
        )
    assert learner.work == catalog_work and paired.work == paired_work


def test_should_refuse_catalog_records_in_normal_paired_and_readout_ports(
    torch, cross_objective, fresh_owner
):
    packet, targets = batch(torch)
    paired = cross_objective[0]
    for learner in (paired, fresh_owner):
        before = learner.work
        with pytest.raises(ValueError):
            learner.predict(packet)
        with pytest.raises(ValueError):
            learner.predict_choices(packet)
        if learner is paired:
            with pytest.raises(ValueError):
                learner.train_batch(packet, targets)
        assert learner.work == before


def test_should_charge_candidates_and_logits_before_failed_forward_and_poison_owner(
    torch, monkeypatch
):
    learner, _ = owner(torch)
    packet, targets = batch(torch)
    initial = learner._capture()

    def fail(*args, **kwargs):
        raise RuntimeError("catalog loss return lost")

    monkeypatch.setattr(catalog, "catalog_head_loss", fail)
    with pytest.raises(TrainingUpdateFailed) as error:
        learner.train_batch(packet, targets)
    assert error.value.work == learner.work
    assert error.value.definition == catalog.OBJECTIVE
    work = learner.work
    assert work["train_attempts"] == work["loss_forward_attempts"] == 1
    assert work["training_rows_attempted"] == 2
    assert work["catalog_candidate_rows_attempted"] == 3
    assert work["catalog_logit_values_attempted"] == 6
    assert work["loss_forward_returns"] == work["backward_attempts"] == work["commits"] == 0
    assert work["unreturned_work_unknown"] is True
    assert native._equal(initial.state_head, learner._bundle[0].state_dict())
    assert learner.completed_updates == learner.training_examples == 0
    with pytest.raises(ValueError):
        learner.train_batch(packet, targets)
    assert learner.work == work


@pytest.fixture(scope="module")
def temporal_stream(torch):
    features = temporal_features(torch)
    training, choices = study.build_catalog_training_batches(identity(), features)
    learner, source = owner(torch, maximum=12)
    report: dict[str, Any] = {}
    observations = []

    def save():
        observations.append((learner.completed_updates, "checkpoints" in report))

    run_temporal_stream(learner, training, choices, report, catalog_training=True, save=save)
    return dict(
        learner=learner,
        source=source,
        report=report,
        training=training,
        choices=choices,
        features=features,
        observations=observations,
    )


def test_should_bind_full_active_catalog_and_current_observation_targets_at_all_phases(
    torch, temporal_stream
):
    case = temporal_stream
    assert len(case["training"]) == 12
    ids = [candidate.candidate_id for candidate in CATALOG]
    for index, (packet, targets) in enumerate(case["training"]):
        phase = PHASES[index // 4]
        examples = temporal_examples(phase, "train", exposure=index % 4)
        allowed = examples[0].request.allowed_candidates
        expected_ids = tuple(candidate.candidate_id for candidate in allowed)
        assert type(packet) is FrozenCatalogTrainingBatch and type(targets) is CLMCatalogTargets
        assert (
            packet.role == "train" and packet.candidate_ids == targets.candidate_ids == expected_ids
        )
        assert targets.candidate_indices == tuple(
            expected_ids.index(example.target_id) for example in examples
        )
        assert torch.equal(
            packet.actions,
            case["features"][[8 + ids.index(candidate.candidate_id) for candidate in allowed]],
        )
        assert torch.equal(packet.actions, case["choices"][phase].actions)
        vocabulary = ("invoice", "legacy", "parcel", "password")
        assert torch.equal(
            packet.states,
            case["features"][[vocabulary.index(example.request.query) for example in examples]],
        )


def test_should_complete_original_stream_boundaries_with_catalog_receipts_and_separate_work(
    temporal_stream,
):
    report = temporal_stream["report"]
    assert report["sequence_complete"] is True
    assert report["training_objective"] == catalog.OBJECTIVE
    assert report["stream"]["completed_updates"] == 12 and report["stream"]["training_pairs"] == 36
    receipts = report["stream"]["receipts"]
    assert len(receipts) == 12 and len({row["sample_id"] for row in receipts}) == 12
    assert [row["observed_at"] for row in receipts] == list(range(0, 24, 2))
    assert (
        [row["arrived_at"] for row in receipts]
        == [row["applied_at"] for row in receipts]
        == list(range(1, 24, 2))
    )
    assert all(row["diagnostic"]["definition"] == catalog.OBJECTIVE for row in receipts)
    assert [(row["completed_updates"], row["phase"]) for row in report["raw_readouts"]] == list(
        CHECKPOINTS
    )
    work = report["learner_work"]
    assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 12
    assert work["training_rows_attempted"] == work["catalog_candidate_rows_attempted"] == 36
    assert work["catalog_logit_values_attempted"] == 108
    assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 5
    assert work["prediction_rows_attempted"] == 20 and work["choice_candidate_rows_attempted"] == 15
    assert all(not scored for _, scored in temporal_stream["observations"])


def test_should_reuse_original_cached_development_metrics_after_catalog_learning(temporal_stream):
    report = deepcopy(temporal_stream["report"])
    report["control_probabilities"] = {
        version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")
    }
    original, work = deepcopy(report), temporal_stream["learner"].work
    scores = study.score_sequence(report)
    assert report == original and temporal_stream["learner"].work == work
    assert scores["cached_typed_decodes"] == 88
    assert [(point["update"], point["phase"]) for point in scores["checkpoints"]] == list(
        CHECKPOINTS
    )
    for point in scores["checkpoints"]:
        assert len(point["learned"]["cases"]) == 4
        assert all(row["role"] == "outer_selection" for row in point["learned"]["cases"])


@pytest.fixture(scope="module")
def fresh_stream_owner(torch):
    learner, _ = owner(torch, maximum=12)
    return learner


@pytest.mark.parametrize(
    "fault", ("candidate_order", "candidate_vector", "target", "role", "disabled")
)
def test_should_refuse_misbound_catalog_stream_before_baseline(torch, fresh_stream_owner, fault):
    features = temporal_features(torch)
    training, choices = study.build_catalog_training_batches(identity(), features)
    learner = fresh_stream_owner
    if fault == "candidate_order":
        object.__setattr__(
            training[4][0], "candidate_ids", tuple(reversed(training[4][0].candidate_ids))
        )
    elif fault == "candidate_vector":
        training[4][0].actions.add_(0.1)
    elif fault == "target":
        object.__setattr__(training[4][1], "candidate_indices", (2, 2, 2))
    elif fault == "role":
        object.__setattr__(training[4][0], "role", "probe_train")
    before = learner.work
    with pytest.raises(ValueError):
        run_temporal_stream(learner, training, choices, {}, catalog_training=fault != "disabled")
    assert learner.work == before


@pytest.mark.parametrize(
    "arguments,expected", [([], {}), (["--catalog-training"], {"catalog_training": True})]
)
def test_should_keep_catalog_cli_opt_in_and_default_unchanged(
    monkeypatch, tmp_path, arguments, expected
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            *arguments,
        ],
    )
    calls = []

    def case(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True, fixture="metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [expected]


@pytest.mark.parametrize(
    "other",
    (
        ["--catalog-support", "repeat"],
        ["--catalog-support", "route"],
        ["--compare-catalog-support"],
    ),
)
def test_should_route_catalog_support_cross_products_to_existing_runner(
    monkeypatch, tmp_path, other
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm-temporal",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--catalog-training",
            *other,
        ],
    )

    calls = []

    def capture(*args, **kwargs):
        kwargs.pop("save")
        calls.append(kwargs)

    monkeypatch.setattr(study, "run_case", capture)
    study.main()
    expected: dict[str, Any] = dict(train_catalog_diagnostics=False, catalog_training=True)
    if other[0] == "--compare-catalog-support":
        expected["compare_catalog_support"] = True
    else:
        expected["catalog_support"] = other[1]
    assert calls == [expected]


@pytest.mark.parametrize(
    "fault", ("objective", "missing_objective", "candidate_work", "logit_work", "diagnostic")
)
def test_should_validate_catalog_objective_and_paid_exposures_before_any_cached_score(
    temporal_stream, monkeypatch, fault
):
    report = deepcopy(temporal_stream["report"])
    report["control_probabilities"] = {
        version: [[1 / 3] * 3 for _ in range(4)] for version in ("v1", "v2")
    }
    if fault == "objective":
        report["training_objective"] = native.OBJECTIVE
    elif fault == "missing_objective":
        report.pop("training_objective")
    elif fault == "candidate_work":
        report["learner_work"]["catalog_candidate_rows_attempted"] -= 1
    elif fault == "logit_work":
        report["learner_work"]["catalog_logit_values_attempted"] -= 1
    else:
        report["stream"]["receipts"][-1]["diagnostic"]["definition"] = native.OBJECTIVE

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound catalog objective exposed cached development scores")

    monkeypatch.setattr(study, "_score_cached", forbidden)
    with pytest.raises(ValueError):
        study.score_sequence(report)


def test_should_support_train_diagnostics_with_full_catalog_updates_and_cached_scoring(torch):
    features = temporal_features(torch)
    training, choices = study.build_catalog_training_batches(identity(), features)
    learner, _ = owner(torch, maximum=12)
    report: dict[str, Any] = {}
    run_temporal_stream(
        learner,
        training,
        choices,
        report,
        catalog_training=True,
        train_choice_batches=study.build_train_choices(training, choices),
    )
    assert report["sequence_complete"] is True
    assert learner.work["choice_prediction_returns"] == 8
    assert learner.work["prediction_rows_attempted"] == 29
    assert learner.work["choice_candidate_rows_attempted"] == 24
    assert learner.work["catalog_candidate_rows_attempted"] == 36
    assert learner.work["catalog_logit_values_attempted"] == 108
    assert [row["phase"] for row in report["train_raw_readouts"]] == list(PHASES)
    scored = study.score_train_choices(report)
    assert [len(point["rows"]) for point in scored] == [3, 3, 3]
