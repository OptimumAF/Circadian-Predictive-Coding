"""Finite arrived-TRAIN rehearsal in the existing disposable catalog probe."""

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
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_frozen_encoder import text_recipe
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog_probe import CatalogProbeRehearsal
from src.core.experience import ExperiencePermissions
from src.core.learner_ports import TrainingUpdateFailed
from src.core.temporal_decision_sequence import CHECKPOINTS, PHASES
from src.infra.clm_local_loader import LoadedCLM
from test_clm_catalog_learning import identity, owner, parts
from test_clm_catalog_reserved_probe import features
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_typed_stream_readout import LiteralSampler

ROWS = ((0, 1), (2, 0), (1, 2)) * 2
CONSENT = ExperiencePermissions(training=True, replay=True)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def data(torch):
    base, training, readouts = study.build_reserved_probe_batches(
        identity(), features(torch), catalog_training=True
    )
    benchmark, choices = study.build_catalog_training_batches(identity(), base)
    arrived = {phase: benchmark[4 * index] for index, phase in enumerate(PHASES)}
    return training, readouts, benchmark, choices, arrived


def rehearsal(training, arrived, *, phase="after_A"):
    return checkpoint.build_probe_rehearsal(
        training[phase], arrived[phase], phase=phase, permissions=CONSENT
    )


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0)


def replay_budget() -> ToyBudgetSession:
    return ToyBudgetSession(ToyExecutionBudget(max_replay_examples=12), lambda: 0.0)


def factory(source):
    return CLMCatalogProbe(
        source, policy=replace(source.policy, max_updates=6, max_restores=0), max_readouts=6
    )


@pytest.mark.parametrize("phase", PHASES)
def test_should_bind_exact_six_packets_to_already_arrived_train_without_test_rows(torch, phase):
    training, readouts, _, _, arrived = data(torch)
    record = rehearsal(training, arrived, phase=phase)
    marker, marker_targets = training[phase]
    old, old_targets = arrived[phase]
    assert type(record) is CatalogProbeRehearsal and record.phase == phase
    assert record.training is arrived[phase] and record.permissions is CONSENT
    assert len(record.packets) == 6
    for (packet, targets), indices in zip(record.packets, ROWS, strict=True):
        assert packet.role == "probe_train" and packet.identity == identity()
        assert packet.candidate_ids == marker.candidate_ids == old.candidate_ids
        assert torch.equal(packet.states[:2], marker.states)
        assert torch.equal(packet.states[2:], old.states[list(indices)])
        assert torch.equal(packet.actions, marker.actions)
        assert targets.candidate_indices == marker_targets.candidate_indices + tuple(
            old_targets.candidate_indices[index] for index in indices
        )
        assert not torch.equal(packet.states[2:], readouts[phase]["probe_test"].states)
    assert [sum(index in indices for indices in ROWS) for index in range(3)] == [4, 4, 4]
    original = record.packets[0][0].states.clone()
    marker.states.add_(0.25)
    old.states.add_(0.5)
    assert torch.equal(record.packets[0][0].states, original)


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
        observed.append((kwargs["phase"], source.completed_updates, kwargs.get("rehearsal")))
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
            probe_rehearsal=True,
        )
    return dict(report=report, observed=observed, loaded=loaded, original=original, rng=rng)


def test_should_keep_benchmark_fixed_and_only_rehearse_after_train_has_arrived(torch, full_case):
    report = full_case["report"]
    assert report["complete"] is True and report["sequence_complete"] is True
    assert report["training_replay_consent"] is True and report["probe_rehearsal"] is True
    assert report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 36
    assert report["learner_work"]["train_attempts"] == 12
    assert report["learner_work"]["training_rows_attempted"] == 36
    assert report["learner_work"]["catalog_logit_values_attempted"] == 108
    assert [(phase, update) for phase, update, _ in full_case["observed"]] == [
        (phase, update) for update, phase in CHECKPOINTS
    ]
    assert full_case["observed"][0][2] is None
    assert all(type(row[2]) is CatalogProbeRehearsal for row in full_case["observed"][1:])
    children = report["reserved_probes"]
    assert "rehearsal" not in children[0]
    assert children[0]["stream"]["training_pairs"] == 12
    assert children[0]["probe_work"]["catalog_logit_values_attempted"] == 36
    for child in children[1:]:
        assert child["complete"] is True
        assert child["stream"]["completed_updates"] == 6
        assert child["stream"]["training_pairs"] == 24
        assert len(child["stream"]["receipts"]) == 6
        assert child["probe_work"]["training_rows_attempted"] == 24
        assert child["probe_work"]["catalog_candidate_rows_attempted"] == 18
        assert child["probe_work"]["catalog_logit_values_attempted"] == 72
        assert child["probe_work"]["choice_prediction_returns"] == 6
        assert child["probe_work"]["prediction_rows_attempted"] == 16
        assert child["rehearsal"]["source_phase"] == child["phase"]
        assert child["rehearsal"]["source_role"] == "train"
        assert child["rehearsal"]["source_row_indices"] == [list(row) for row in ROWS]
        assert child["rehearsal"]["replay_rows_attempted"] == 12
        retained = child["rehearsal"]["retained_tensor_storage_bytes"]
        assert type(retained) is int and 48 <= retained <= 1024
    loaded = full_case["loaded"]
    assert study._control_matches(torch, loaded, full_case["original"])
    assert torch.equal(torch.get_rng_state(), full_case["rng"])


def test_should_keep_existing_cached_metrics_and_score_only_after_all_work(full_case):
    report = full_case["report"]
    original = deepcopy(report)
    scores = study.score_reserved_probes(report)
    assert report == original
    assert scores == report["reserved_probe_scores"]
    assert scores["cached_probability_rows"] == 64
    assert len(scores["checkpoints"]) == 4
    for row in scores["checkpoints"]:
        assert tuple(row["groups"]) == ("probe_train", "probe_test", "old_task")
        for group in row["groups"].values():
            assert set(group["after_minus_before"]) == {
                "accuracy",
                "nll_mean",
                "nll_status",
                "brier_mean",
                "ece_5bin",
                "obsolete_rule_error_count",
            }


@pytest.mark.parametrize(
    "fault",
    (
        "missing",
        "phase",
        "role",
        "order",
        "rows",
        "storage",
        "negative_storage",
        "extra",
        "pairs",
        "logits",
        "partial",
        "mode",
        "consent",
        "missing_consent",
    ),
)
def test_should_refuse_corrupted_rehearsal_cache_before_any_metric(full_case, monkeypatch, fault):
    report = deepcopy(full_case["report"])
    child = report["reserved_probes"][-1]
    record = child["rehearsal"]
    if fault == "missing":
        child.pop("rehearsal")
    elif fault == "phase":
        record["source_phase"] = "after_B"
    elif fault == "role":
        record["source_role"] = "probe_test"
    elif fault == "order":
        record["source_row_indices"][0].reverse()
    elif fault == "rows":
        record["replay_rows_attempted"] = 10
    elif fault == "storage":
        record["retained_tensor_storage_bytes"] = True
    elif fault == "negative_storage":
        record["retained_tensor_storage_bytes"] = -1
    elif fault == "extra":
        record["unobserved_consent"] = True
    elif fault == "pairs":
        child["stream"]["training_pairs"] = 12
    elif fault == "logits":
        child["probe_work"]["catalog_logit_values_attempted"] = 36
    elif fault == "partial":
        child["complete"] = False
    elif fault == "mode":
        report["probe_rehearsal"] = False
    elif fault == "consent":
        report["training_replay_consent"] = False
    else:
        report.pop("training_replay_consent")

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupted rehearsal cache exposed a score")

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
        "source_checkpoint",
        "source_role",
        "future_targets",
        "consent",
        "evaluation",
        "packet_count",
        "packet_role",
        "packet_states",
        "packet_targets",
        "packet_actions",
        "packet_identity",
        "budget",
        "spent_replay",
        "spent_updates",
        "missing_replay_budget",
        "shared_replay_budget",
        "spent_replay_update",
        "short_replay_limit",
        "paired",
    ),
)
def test_should_refuse_unarrived_or_misbound_rehearsal_before_factory(torch, source_after_a, fault):
    training, readouts, _, _, arrived = data(torch)
    record = rehearsal(training, arrived)
    selected_budget = budget()
    selected_replay_budget: ToyBudgetSession | None = replay_budget()
    catalog_training = True
    phase = "after_A"
    if fault == "baseline":
        phase = "baseline"
    elif fault == "source_checkpoint":
        object.__setattr__(record, "phase", "after_B")
    elif fault == "source_role":
        object.__setattr__(record.training[0], "role", "probe_test")
    elif fault == "future_targets":
        object.__setattr__(
            record.training[1], "candidate_indices", arrived["after_B"][1].candidate_indices
        )
    elif fault == "consent":
        object.__setattr__(record, "permissions", ExperiencePermissions(training=True))
    elif fault == "evaluation":
        object.__setattr__(
            record,
            "permissions",
            ExperiencePermissions(training=True, replay=True, evaluation=True),
        )
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
    elif fault == "spent_replay":
        assert selected_replay_budget is not None
        selected_replay_budget.replay_examples_completed = 2
    elif fault == "spent_updates":
        selected_budget.updates_completed = 1
    elif fault == "missing_replay_budget":
        selected_replay_budget = None
    elif fault == "shared_replay_budget":
        selected_replay_budget = selected_budget
    elif fault == "spent_replay_update":
        assert selected_replay_budget is not None
        selected_replay_budget.updates_completed = 1
    elif fault == "short_replay_limit":
        selected_replay_budget = ToyBudgetSession(
            ToyExecutionBudget(max_replay_examples=10), lambda: 0.0
        )
    else:
        catalog_training = False
    before = source_after_a.work

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted rehearsal reached probe construction")

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training[phase],
            readouts[phase],
            {},
            phase=phase,
            probe_factory=forbidden,
            permissions=ExperiencePermissions(training=True),
            budget=selected_budget,
            replay_budget=selected_replay_budget,
            catalog_training=catalog_training,
            rehearsal=record,
        )
    assert source_after_a.work == before and source_after_a.completed_updates == 4


def test_should_charge_replay_before_a_failed_native_update_and_refuse_reuse(torch, source_after_a):
    training, readouts, _, _, arrived = data(torch)
    record, selected_budget = rehearsal(training, arrived), budget()
    selected_replay_budget = replay_budget()
    report: dict[str, Any] = {}
    before, work = source_after_a._capture(), source_after_a.work

    def broken_factory(source):
        probe = factory(source)

        def fail(module, args):
            if probe.work["train_attempts"]:
                raise RuntimeError("mixed native forward failed")

        probe._learner._bundle[1].register_forward_pre_hook(fail)
        return probe

    with pytest.raises(TrainingUpdateFailed, match="mixed native forward failed"):
        probe_app.run_reserved_probe(
            source_after_a,
            training["after_A"],
            readouts["after_A"],
            report,
            phase="after_A",
            probe_factory=broken_factory,
            permissions=ExperiencePermissions(training=True),
            budget=selected_budget,
            replay_budget=selected_replay_budget,
            catalog_training=True,
            rehearsal=record,
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == selected_budget.updates_completed == 0
    assert report["stream"]["training_pairs"] == 0 and report["stream"]["receipts"] == []
    assert (
        selected_replay_budget.replay_examples_completed
        == report["rehearsal"]["replay_rows_attempted"]
        == 2
    )
    assert selected_budget.replay_examples_completed == 0
    assert selected_replay_budget.updates_completed == 0
    assert report["probe_work"]["train_attempts"] == 1
    assert report["probe_work"]["training_rows_attempted"] == 4
    assert report["probe_work"]["commits"] == 0
    paid = deepcopy(report)
    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training["after_A"],
            readouts["after_A"],
            report,
            phase="after_A",
            probe_factory=broken_factory,
            permissions=ExperiencePermissions(training=True),
            budget=selected_budget,
            replay_budget=selected_replay_budget,
            catalog_training=True,
            rehearsal=record,
        )
    assert report == paid and source_after_a.work == work
    assert native._equal(vars(before), vars(source_after_a._capture()))


def test_should_refuse_replaced_train_owner_after_one_committed_update(torch, source_after_a):
    training, readouts, _, _, arrived = data(torch)
    record, selected_budget = rehearsal(training, arrived), budget()
    selected_replay_budget = replay_budget()
    report: dict[str, Any] = {}
    work = source_after_a.work

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            object.__setattr__(
                record, "training", (record.training[0], deepcopy(record.training[1]))
            )

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source_after_a,
            training["after_A"],
            readouts["after_A"],
            report,
            phase="after_A",
            probe_factory=factory,
            permissions=ExperiencePermissions(training=True),
            budget=selected_budget,
            replay_budget=selected_replay_budget,
            catalog_training=True,
            rehearsal=record,
            save=save,
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == selected_budget.updates_completed == 1
    assert report["stream"]["training_pairs"] == 4
    assert len(report["stream"]["receipts"]) == 1
    assert (
        report["rehearsal"]["replay_rows_attempted"]
        == selected_replay_budget.replay_examples_completed
        == 2
    )
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
    ),
)
def test_should_refuse_incompatible_cli_before_any_loader(monkeypatch, tmp_path, flags):
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
            "--probe-rehearsal",
            *flags,
        ],
    )
    monkeypatch.setattr(
        study,
        "load_local_clm",
        lambda *args, **kwargs: pytest.fail("CLI loaded an incompatible model"),
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
    assert not (tmp_path / "new.json").exists()


def test_should_route_explicit_rehearsal_cli_without_changing_default_options(
    monkeypatch, tmp_path
):
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
            "--probe-rehearsal",
        ],
    )

    def execute(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True)
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    assert calls == [
        {"catalog_training": True, "reserved_task_probes": True, "probe_rehearsal": True}
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
        {"catalog_training": True, "reserved_task_probes": True, "probe_rehearsal": 1},
    ),
)
def test_should_refuse_invalid_api_rehearsal_before_loading(monkeypatch, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted rehearsal API reached loader")

    monkeypatch.setattr(study, "load_local_clm", forbidden)
    kwargs = {"probe_rehearsal": True, **options}
    with pytest.raises(ValueError):
        study.run_case(Path("unused"), Path("unused"), {}, **kwargs)


def test_should_refuse_lost_original_train_replay_consent_before_first_main_update(
    torch, monkeypatch
):
    import src.app.clm_temporal_sequence as temporal

    _, _, benchmark, choices, _ = data(torch)
    source, _ = owner(torch, maximum=12)
    report: dict[str, Any] = {}
    original = temporal.Experience

    def lost_consent(*args, **kwargs):
        experience = original(*args, **kwargs)
        return replace(experience, permissions=ExperiencePermissions(training=True))

    monkeypatch.setattr(temporal, "Experience", lost_consent)
    with pytest.raises(ValueError, match="original arrived TRAIN replay consent"):
        temporal.run_temporal_stream(
            source,
            benchmark,
            choices,
            report,
            catalog_training=True,
            allow_train_replay=True,
            on_checkpoint=lambda phase, learner: None,
        )
    assert source.work["train_attempts"] == 0
    assert report["stream"]["completed_updates"] == 0
