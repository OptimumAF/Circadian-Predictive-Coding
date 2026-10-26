"""Matched extra-marker exposure without arrived-task replay or learner mutation."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_checkpoint_probes as checkpoint
import src.adapters.clm_head_learner as native
import src.app.clm_reserved_probe as probe_app
from src.adapters.clm_frozen_encoder import text_recipe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog_probe import CatalogProbeMarkerControl
from src.core.experience import ExperiencePermissions
from src.core.learner_ports import TrainingUpdateFailed
from src.core.temporal_decision_sequence import CHECKPOINTS, PHASES
from src.infra.clm_local_loader import LoadedCLM
from test_clm_catalog_learning import identity, owner, parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_probe_rehearsal import budget, data, factory, rehearsal
from test_clm_typed_stream_readout import LiteralSampler


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def control(training, *, phase="after_A"):
    return checkpoint.build_probe_marker_control(training[phase], phase=phase)


@pytest.mark.parametrize("phase", PHASES)
def test_should_repeat_only_fixed_marker_rows_with_independent_storage(torch, phase):
    training, readouts, _, _, arrived = data(torch)
    record = control(training, phase=phase)
    marker, labels = training[phase]
    assert type(record) is CatalogProbeMarkerControl and record.phase == phase
    assert len(record.packets) == 6
    for packet, targets in record.packets:
        assert packet.role == "probe_train" and packet.identity == marker.identity
        assert packet.candidate_ids == marker.candidate_ids == targets.candidate_ids
        assert targets.candidate_indices == labels.candidate_indices * 2
        assert torch.equal(packet.states[:2], marker.states)
        assert torch.equal(packet.states[2:], marker.states)
        assert torch.equal(packet.actions, marker.actions)
        assert not torch.equal(packet.states[2:], readouts[phase]["probe_test"].states)
        assert not torch.equal(packet.states[2:], arrived[phase][0].states[:2])
    saved = [(batch.states.clone(), batch.actions.clone()) for batch, _ in record.packets]
    marker.states.add_(0.5)
    marker.actions.add_(0.25)
    for (packet, _), (states, actions) in zip(record.packets, saved, strict=True):
        assert torch.equal(packet.states, states) and torch.equal(packet.actions, actions)
    record.packets[0][0].states.add_(1)
    assert torch.equal(record.packets[1][0].states, saved[1][0])


def test_should_match_rehearsal_paid_work_without_recording_markers_as_replay():
    original = probe_app._expected_work(6, 6, 16, catalog_training=True)
    marker = probe_app._expected_work(6, 6, 16, catalog_training=True, extra_marker_rows=2)
    history = probe_app._expected_work(6, 6, 16, catalog_training=True, rehearsal_rows=2)
    assert marker == history
    assert original["training_rows_attempted"] == 12
    assert marker["training_rows_attempted"] == 24
    assert marker["catalog_candidate_rows_attempted"] == 18
    assert marker["catalog_logit_values_attempted"] == 72


@pytest.mark.parametrize(
    "options",
    (
        {"extra_marker_rows": True},
        {"extra_marker_rows": 1},
        {"extra_marker_rows": -2},
        {"extra_marker_rows": 2, "rehearsal_rows": 2},
        {"extra_marker_rows": 2, "catalog_training": False},
    ),
)
def test_should_refuse_ambiguous_extra_exposure_work(options):
    with pytest.raises(ValueError):
        probe_app._expected_work(6, 6, 16, **{"catalog_training": True, **options})


@pytest.fixture(scope="module")
def full_case(torch):
    state_texts, action_texts = study.feature_texts(include_reserved_probe=True)
    model = backbone(torch)
    tokenizer = FakeTokenizer(
        {text: [index + 1] for index, text in enumerate(state_texts + action_texts)}
    )
    state, action, scale = parts(torch)
    loaded = LoadedCLM(
        model, tokenizer, state, action, scale, replace(identity(), recipe=text_recipe(2048)), {}
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), scale.detach().clone()
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    observed = []
    real_probe = probe_app.run_reserved_probe

    def checked(source, *args, **kwargs):
        before, work = source._capture(), source.work
        owners = tuple(source._owners)
        assert report["sequence_complete"] is False
        assert "checkpoints" not in report and "reserved_probe_scores" not in report
        result = real_probe(source, *args, **kwargs)
        assert native._equal(vars(before), vars(source._capture()))
        assert source.work == work
        assert all(left is right for left, right in zip(owners, source._owners, strict=True))
        assert kwargs.get("rehearsal") is None and kwargs.get("replay_budget") is None
        observed.append((kwargs["phase"], source.completed_updates, kwargs.get("marker_control")))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "load_local_clm", lambda *args, **kwargs: loaded)
        patch.setattr(study, "ProcessRssSampler", LiteralSampler)
        patch.setattr(probe_app, "run_reserved_probe", checked)
        study.run_case(
            Path("literal-model"),
            Path("literal-head"),
            report,
            catalog_training=True,
            reserved_task_probes=True,
            probe_marker_control=True,
        )
    return dict(report=report, observed=observed, loaded=loaded, original=original, rng=rng)


def test_should_keep_main_and_baseline_fixed_and_control_only_postbaseline(torch, full_case):
    report = full_case["report"]
    assert report["complete"] is True and report["sequence_complete"] is True
    assert report["probe_marker_control"] is True
    assert "probe_rehearsal" not in report and "training_replay_consent" not in report
    assert report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 36
    assert report["learner_work"]["train_attempts"] == 12
    assert report["learner_work"]["catalog_candidate_rows_attempted"] == 36
    assert report["learner_work"]["catalog_logit_values_attempted"] == 108
    assert [(phase, update) for phase, update, _ in full_case["observed"]] == [
        (phase, update) for update, phase in CHECKPOINTS
    ]
    assert full_case["observed"][0][2] is None
    assert all(type(row[2]) is CatalogProbeMarkerControl for row in full_case["observed"][1:])
    children = report["reserved_probes"]
    assert "marker_control" not in children[0]
    assert children[0]["stream"]["training_pairs"] == 12
    assert children[0]["probe_work"]["catalog_logit_values_attempted"] == 36
    for child in children[1:]:
        assert child["complete"] is True and "rehearsal" not in child
        assert child["stream"]["completed_updates"] == 6
        assert child["stream"]["training_pairs"] == 24
        assert child["probe_work"] == probe_app._expected_work(
            6, 6, 16, catalog_training=True, extra_marker_rows=2
        )
        record = child["marker_control"]
        assert record["source_phase"] == child["phase"]
        assert record["source_role"] == "probe_train"
        assert record["source_row_indices"] == [[0, 1]] * 6
        assert record["extra_marker_rows_attempted"] == 12
        retained = record["retained_tensor_storage_bytes"]
        assert type(retained) is int and 48 <= retained <= 1024
        assert "replay_rows_attempted" not in record
        for index, receipt in enumerate(child["stream"]["receipts"]):
            assert receipt["observed_at"] == 2 * index
            assert receipt["arrived_at"] == receipt["applied_at"] == 2 * index + 1
            assert receipt["update_number"] == index + 1
    assert sum(child["stream"]["training_pairs"] for child in children) + 36 == 120
    assert (
        sum(child["probe_work"]["catalog_candidate_rows_attempted"] for child in children) + 36
        == 108
    )
    assert (
        sum(child["probe_work"]["catalog_logit_values_attempted"] for child in children) + 108
        == 360
    )
    assert study._control_matches(torch, full_case["loaded"], full_case["original"])
    assert torch.equal(torch.get_rng_state(), full_case["rng"])


def test_should_score_only_complete_cached_work_without_mutating_it(full_case):
    report = full_case["report"]
    saved = deepcopy(report)
    scores = study.score_reserved_probes(report)
    assert report == saved and scores == report["reserved_probe_scores"]
    assert scores["cached_probability_rows"] == 64
    assert len(scores["checkpoints"]) == 4
    assert all(
        tuple(row["groups"]) == ("probe_train", "probe_test", "old_task")
        for row in scores["checkpoints"]
    )


@pytest.mark.parametrize(
    "fault",
    (
        "missing_record",
        "phase",
        "role",
        "order",
        "attempted",
        "boolean_attempted",
        "storage",
        "negative_storage",
        "extra",
        "pairs",
        "logits",
        "partial",
        "mode",
        "missing_mode",
        "rehearsal_mode",
        "baseline_record",
        "receipt",
    ),
)
def test_should_refuse_corrupted_control_cache_before_any_score(full_case, monkeypatch, fault):
    report = deepcopy(full_case["report"])
    child = report["reserved_probes"][-1]
    record = child["marker_control"]
    if fault == "missing_record":
        child.pop("marker_control")
    elif fault == "phase":
        record["source_phase"] = "after_B"
    elif fault == "role":
        record["source_role"] = "train"
    elif fault == "order":
        record["source_row_indices"][0].reverse()
    elif fault == "attempted":
        record["extra_marker_rows_attempted"] = 10
    elif fault == "boolean_attempted":
        record["extra_marker_rows_attempted"] = True
    elif fault == "storage":
        record["retained_tensor_storage_bytes"] = True
    elif fault == "negative_storage":
        record["retained_tensor_storage_bytes"] = -1
    elif fault == "extra":
        record["replay_rows_attempted"] = 12
    elif fault == "pairs":
        child["stream"]["training_pairs"] = 12
    elif fault == "logits":
        child["probe_work"]["catalog_logit_values_attempted"] = 36
    elif fault == "partial":
        child["complete"] = False
    elif fault == "mode":
        report["probe_marker_control"] = False
    elif fault == "missing_mode":
        report.pop("probe_marker_control")
    elif fault == "rehearsal_mode":
        report["probe_rehearsal"] = True
        report["training_replay_consent"] = True
    elif fault == "baseline_record":
        report["reserved_probes"][0]["marker_control"] = deepcopy(record)
    else:
        child["stream"]["receipts"][-1]["arrived_at"] += 1

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt control cache exposed quality")

    monkeypatch.setattr(study, "_reserved_probe_metrics", forbidden)
    with pytest.raises(ValueError):
        study.score_reserved_probes(report)


@pytest.fixture(scope="module")
def source_after_a(torch):
    _, _, benchmark, _, _ = data(torch)
    source, _ = owner(torch, maximum=12)
    for batch, targets in benchmark[:4]:
        source.train_batch(batch, targets)
    return source


@pytest.mark.parametrize(
    "fault",
    (
        "baseline",
        "phase",
        "packet_count",
        "packet_role",
        "packet_states",
        "packet_targets",
        "packet_actions",
        "packet_identity",
        "budget",
        "spent_updates",
        "replay_consent",
        "evaluation_consent",
        "paired",
        "replay_budget",
        "both_modes",
    ),
)
def test_should_refuse_misbound_control_before_constructing_probe(torch, source_after_a, fault):
    training, readouts, _, _, arrived = data(torch)
    record = control(training)
    selected_budget = budget()
    permissions = ExperiencePermissions(training=True)
    phase, catalog_training = "after_A", True
    extra: dict[str, Any] = {}
    if fault == "baseline":
        phase = "baseline"
    elif fault == "phase":
        object.__setattr__(record, "phase", "after_B")
    elif fault == "packet_count":
        object.__setattr__(record, "packets", record.packets[:-1])
    elif fault == "packet_role":
        object.__setattr__(record.packets[0][0], "role", "probe_test")
    elif fault == "packet_states":
        record.packets[0][0].states[2:].copy_(readouts["after_A"]["probe_test"].states)
    elif fault == "packet_targets":
        object.__setattr__(record.packets[0][1], "candidate_indices", (0, 2, 2, 2))
    elif fault == "packet_actions":
        record.packets[0][0].actions.add_(0.125)
    elif fault == "packet_identity":
        object.__setattr__(
            record.packets[0][0], "identity", replace(identity(), revision="foreign")
        )
    elif fault == "budget":
        selected_budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=5), lambda: 0.0)
    elif fault == "spent_updates":
        selected_budget.updates_completed = 1
    elif fault == "replay_consent":
        permissions = ExperiencePermissions(training=True, replay=True)
    elif fault == "evaluation_consent":
        permissions = ExperiencePermissions(training=True, evaluation=True)
    elif fault == "paired":
        catalog_training = False
    elif fault == "replay_budget":
        extra["replay_budget"] = ToyBudgetSession(
            ToyExecutionBudget(max_replay_examples=12), lambda: 0.0
        )
    else:
        extra["rehearsal"] = rehearsal(training, arrived)
    before = source_after_a.work

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted control reached native probe construction")

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training[phase],
            readouts[phase],
            {},
            phase=phase,
            probe_factory=forbidden,
            permissions=permissions,
            budget=selected_budget,
            catalog_training=catalog_training,
            marker_control=record,
            **extra,
        )
    assert source_after_a.work == before and source_after_a.completed_updates == 4


def test_should_charge_extra_markers_before_failed_update_without_replay_or_refund(
    torch, source_after_a
):
    training, readouts, _, _, _ = data(torch)
    record, selected_budget = control(training), budget()
    report: dict[str, Any] = {}
    before, work = source_after_a._capture(), source_after_a.work

    def broken_factory(source):
        probe = factory(source)

        def fail(module, args):
            if probe.work["train_attempts"]:
                raise RuntimeError("extra-marker native forward failed")

        probe._learner._bundle[1].register_forward_pre_hook(fail)
        return probe

    kwargs = dict(
        phase="after_A",
        probe_factory=broken_factory,
        permissions=ExperiencePermissions(training=True),
        budget=selected_budget,
        catalog_training=True,
        marker_control=record,
    )
    with pytest.raises(TrainingUpdateFailed, match="extra-marker native forward failed"):
        probe_app.run_reserved_probe(
            source_after_a, training["after_A"], readouts["after_A"], report, **kwargs
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == selected_budget.updates_completed == 0
    assert report["stream"]["training_pairs"] == 0 and report["stream"]["receipts"] == []
    assert report["marker_control"]["extra_marker_rows_attempted"] == 2
    assert selected_budget.replay_examples_completed == 0 and "rehearsal" not in report
    assert report["probe_work"]["train_attempts"] == 1
    assert report["probe_work"]["training_rows_attempted"] == 4
    assert report["probe_work"]["commits"] == 0
    paid = deepcopy(report)
    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a, training["after_A"], readouts["after_A"], report, **kwargs
        )
    assert report == paid and source_after_a.work == work
    assert native._equal(vars(before), vars(source_after_a._capture()))


@pytest.mark.parametrize("fault", ("packets", "consent", "budget"))
def test_should_refuse_changed_original_ownership_after_one_commit(torch, source_after_a, fault):
    training, readouts, _, _, _ = data(torch)
    record, selected_budget = control(training), budget()
    report: dict[str, Any] = {}
    work = source_after_a.work
    permissions = ExperiencePermissions(training=True)

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            if fault == "packets":
                object.__setattr__(record, "packets", tuple(list(record.packets)))
            elif fault == "consent":
                object.__setattr__(permissions, "training", False)
            else:
                selected_budget.budget = replace(selected_budget.budget)

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training["after_A"],
            readouts["after_A"],
            report,
            phase="after_A",
            probe_factory=factory,
            permissions=permissions,
            budget=selected_budget,
            catalog_training=True,
            marker_control=record,
            save=save,
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == selected_budget.updates_completed == 1
    assert report["stream"]["training_pairs"] == 4
    assert len(report["stream"]["receipts"]) == 1
    assert report["marker_control"]["extra_marker_rows_attempted"] == 2
    assert selected_budget.replay_examples_completed == 0
    assert report["probe_work"]["commits"] == 1 and source_after_a.work == work


@pytest.mark.parametrize(
    "flags",
    (
        [],
        ["--catalog-training"],
        ["--reserved-task-probes"],
        ["--catalog-training", "--reserved-task-probes", "--catalog-support", "repeat"],
        ["--catalog-training", "--reserved-task-probes", "--compare-catalog-support"],
        ["--catalog-training", "--reserved-task-probes", "--typed-development-readout"],
        ["--catalog-training", "--reserved-task-probes", "--probe-rehearsal"],
    ),
)
def test_should_refuse_invalid_marker_cli_before_loading(monkeypatch, tmp_path, flags):
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--probe-marker-control",
            *flags,
        ],
    )
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid CLI loaded model")
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not (tmp_path / "new.json").exists()


def test_should_route_explicit_marker_cli_without_changing_default_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(tmp_path / "new.json"),
            "--catalog-training",
            "--reserved-task-probes",
            "--probe-marker-control",
        ],
    )

    def execute(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True)
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    assert calls == [
        {"catalog_training": True, "reserved_task_probes": True, "probe_marker_control": True}
    ]


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"catalog_training": True},
        {"reserved_task_probes": True},
        {"catalog_training": True, "reserved_task_probes": True, "catalog_support": "route"},
        {"catalog_training": True, "reserved_task_probes": True, "compare_catalog_support": True},
        {"catalog_training": True, "reserved_task_probes": True, "typed_development_readout": True},
        {"catalog_training": True, "reserved_task_probes": True, "probe_rehearsal": True},
        {"catalog_training": True, "reserved_task_probes": True, "probe_marker_control": 1},
    ),
)
def test_should_refuse_invalid_marker_api_before_loading(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid marker API reached loader")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    with pytest.raises(ValueError):
        study.run_case(
            Path("unused"), Path("unused"), {}, **{"probe_marker_control": True, **options}
        )
