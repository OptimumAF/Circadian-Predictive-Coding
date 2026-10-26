"""Fixed reserved-task probes through the existing temporal stream, on tiny heads."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_head_learner as native
from src.adapters.clm_head_probe import CLMHeadProbe
from src.app.clm_reserved_probe import run_reserved_probe
from src.app.clm_temporal_sequence import run_temporal_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import FrozenEncoderIdentity
from src.core.experience import ExperiencePermissions
from src.core.temporal_clm_probe import (
    PROBE_GROUP_CODES,
    PROBE_QUERIES,
    PROBE_UPDATES,
    reserved_probe_examples,
)
from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS, temporal_examples
from test_clm_temporal_sequence import owner

PHASES = tuple(phase for _, phase in CHECKPOINTS)
GROUPS = ("probe_train", "probe_test", "old_task")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def features(torch, *, catalog_support="none"):
    identity = FrozenEncoderIdentity("fake", "literal", "literal", "last", "reserved-probe", 2)
    count = 16 if catalog_support == "none" else 17
    values = torch.tensor(
        [[(index + 1) / 18, (-1) ** index * (index + 2) / 19] for index in range(count)],
        dtype=torch.float32,
    )
    return identity, values


def fresh_budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=PROBE_UPDATES), lambda: 0.0)


def probe_factory(source):
    return CLMHeadProbe(
        source,
        policy=replace(source.policy, max_updates=PROBE_UPDATES, max_restores=0),
        max_readouts=6,
    )


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("role", ("probe_train", "probe_test"))
def test_should_declare_fixed_new_task_roles_targets_and_active_catalog(phase, role):
    examples = reserved_probe_examples(phase, role)
    assert type(examples) is tuple and len(examples) == 2
    assert PROBE_UPDATES == 6 and PROBE_GROUP_CODES == (1, 3)
    assert PROBE_QUERIES == ("tavrix", "vemlot", "tavrix please", "vemlot please")
    assert [row.target_id for row in examples] == ["inspect", "route"]
    assert [row.request.query for row in examples] == list(
        PROBE_QUERIES[:2] if role == "probe_train" else PROBE_QUERIES[2:]
    )
    active = temporal_examples(phase, "outer_selection")[0].request.allowed_candidates
    assert all(row.phase == phase and row.role == role for row in examples)
    assert all(row.request.allowed_candidates == active for row in examples)
    assert len({row.request.request_id for row in examples}) == 2
    other = reserved_probe_examples(phase, "probe_test" if role == "probe_train" else "probe_train")
    assert {row.source_group for row in examples}.isdisjoint(row.source_group for row in other)
    assert {row.request.request_id for row in examples}.isdisjoint(
        row.request.request_id for row in other
    )


@pytest.mark.parametrize("fault", ("phase", "role", "target_id", "source_group", "query"))
def test_should_refuse_relabeling_or_rebinding_the_fixed_reserved_task(fault):
    example = reserved_probe_examples("baseline", "probe_train")[0]
    values: dict[str, Any] = dict(
        phase="after_B", role="outer_selection", target_id="route", source_group="train-token"
    )
    with pytest.raises(ValueError):
        if fault == "query":
            replace(example, request=replace(example.request, query="selected alternative"))
        else:
            replace(example, **{fault: values[fault]})


@pytest.mark.parametrize("role", ("train", "outer_selection", "final_test"))
def test_should_refuse_benchmark_or_final_roles_for_reserved_records(role):
    with pytest.raises(ValueError):
        reserved_probe_examples("baseline", role)


def test_should_bind_disjoint_reserved_templates_and_preserve_original_feature_bank(torch):
    mode = "none"
    identity, values = features(torch, catalog_support=mode)
    base, training, readouts = study.build_reserved_probe_batches(
        identity, values, catalog_support=mode
    )
    query_texts, actions = study.feature_texts(
        include_catalog_support=mode != "none", include_reserved_probe=True
    )
    old_texts, old_actions = study.feature_texts(include_catalog_support=mode != "none")
    offset = len(old_texts)
    assert query_texts[:offset] == old_texts and query_texts[offset:] == tuple(
        query + "\n\nChoose the ticket workflow tool." for query in PROBE_QUERIES
    )
    assert actions == old_actions == tuple(item.description for item in CATALOG)
    expected_base = torch.cat((values[:offset], values[offset + 4 :]))
    assert torch.equal(base, expected_base)
    batch, targets = training
    assert batch.role == "probe_train" and targets.group_codes == PROBE_GROUP_CODES
    assert torch.equal(batch.states, values[offset : offset + 2])
    assert torch.equal(batch.actions, values[[offset + 5, offset + 7]])
    assert tuple(readouts) == PHASES
    for phase, groups in readouts.items():
        assert tuple(groups) == GROUPS
        assert groups["probe_train"].role == "probe_train"
        assert groups["probe_test"].role == "probe_test"
        assert groups["old_task"].role == "outer_selection"
        assert torch.equal(groups["probe_train"].states, values[offset : offset + 2])
        assert torch.equal(groups["probe_test"].states, values[offset + 2 : offset + 4])
        assert torch.equal(groups["old_task"].states, values[4:8])
        active = temporal_examples(phase, "outer_selection")[0].request.allowed_candidates
        indices = [
            offset + 4 + [item.candidate_id for item in CATALOG].index(item.candidate_id)
            for item in active
        ]
        assert all(torch.equal(group.actions, values[indices]) for group in groups.values())
    original = batch.states.clone()
    values.zero_()
    assert torch.equal(batch.states, original) and torch.equal(base, expected_base)


@pytest.mark.parametrize("fault", ("rows", "dtype", "nonfinite"))
def test_should_refuse_malformed_reserved_feature_union_before_ownership(torch, fault):
    identity, values = features(torch)
    if fault == "rows":
        values = values[:15]
    elif fault == "dtype":
        values = values.double()
    else:
        values[0, 0] = float("nan")
    with pytest.raises(ValueError):
        study.build_reserved_probe_batches(identity, values)


@pytest.fixture(scope="module")
def full_probe_stream(torch):
    identity, values = features(torch)
    base, training, readouts = study.build_reserved_probe_batches(identity, values)
    benchmark, choices = study.build_feature_batches(identity, base)
    source, state, action = owner(torch, identity)
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict())
    rng = torch.get_rng_state().clone()
    raw = values.clone()
    report: dict[str, Any] = {}
    observations = []
    hook = study.make_reserved_probe_hook(training, readouts, report)

    def checked_hook(phase, learner):
        captured, work = learner._capture(), learner.work
        owners = tuple(learner._owners)
        expected = dict(CHECKPOINTS)
        assert expected[learner.completed_updates] == phase
        assert report.get("sequence_complete") is False
        assert "reserved_probe_scores" not in report and "checkpoints" not in report
        hook(phase, learner)
        assert native._equal(vars(captured), vars(learner._capture()))
        assert learner.work == work
        assert all(a is b for a, b in zip(owners, learner._owners))
        observations.append((phase, learner.completed_updates, deepcopy(work)))

    run_temporal_stream(source, benchmark, choices, report, on_checkpoint=checked_hook)
    return dict(
        report=report,
        source=source,
        observations=observations,
        rng=rng,
        values=values,
        raw=raw,
        original=original,
        heads=(state, action),
    )


def test_should_run_all_four_disposable_probes_without_spending_benchmark_allowances(
    full_probe_stream,
):
    case = full_probe_stream
    report, source = case["report"], case["source"]
    assert [(phase, update) for phase, update, _ in case["observations"]] == [
        (phase, update) for update, phase in CHECKPOINTS
    ]
    assert report["sequence_complete"] is True
    assert source.completed_updates == report["stream"]["completed_updates"] == 12
    assert source.training_examples == report["stream"]["training_pairs"] == 36
    assert report["learner_work"] == source.work
    assert (
        source.work["choice_prediction_returns"] == source.work["choice_prediction_attempts"] == 5
    )
    assert source.work["prediction_rows_attempted"] == 20
    assert source.work["choice_candidate_rows_attempted"] == 15
    probes = report["reserved_probes"]
    assert len(probes) == 4
    for child, (update, phase) in zip(probes, CHECKPOINTS, strict=True):
        assert child["phase"] == phase and child["source_update"] == update
        assert child["complete"] is True
        assert child["stream"]["completed_updates"] == 6
        assert child["stream"]["training_pairs"] == 12
        work = child["probe_work"]
        assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 6
        assert work["training_rows_attempted"] == 12
        assert work["choice_prediction_attempts"] == work["choice_prediction_returns"] == 6
        assert work["prediction_rows_attempted"] == 16
        assert work["choice_candidate_rows_attempted"] == 18
        assert work["snapshot_attempts"] == work["restore_attempts"] == 0
        assert work["unreturned_work_unknown"] is False


def test_should_keep_causal_probe_receipts_and_both_endpoints_without_labels_in_features(
    full_probe_stream,
):
    for child in full_probe_stream["report"]["reserved_probes"]:
        receipts = child["stream"]["receipts"]
        assert len(receipts) == 6 and len({row["sample_id"] for row in receipts}) == 6
        assert [row["update_number"] for row in receipts] == list(range(1, 7))
        assert [row["observed_at"] for row in receipts] == list(range(0, 12, 2))
        assert (
            [row["arrived_at"] for row in receipts]
            == [row["applied_at"] for row in receipts]
            == list(range(1, 12, 2))
        )
        assert all(
            row["diagnostic"]["definition"] == "clm_paired_group_masked_symmetric_cross_entropy_v1"
            for row in receipts
        )
        assert [(row["point"], row["group"]) for row in child["readouts"]] == [
            (point, group) for point in ("before", "after") for group in GROUPS
        ]
        assert [len(row["probabilities"]) for row in child["readouts"]] == [2, 2, 4] * 2


def test_should_leave_caller_rng_literal_features_and_released_heads_unchanged(
    torch, full_probe_stream
):
    case = full_probe_stream
    assert torch.equal(torch.get_rng_state(), case["rng"])
    assert torch.equal(case["values"], case["raw"])
    for head, expected in zip(case["heads"], case["original"]):
        assert native._equal(head.state_dict(), expected)


@pytest.fixture(scope="module")
def fresh_source(torch):
    identity, values = features(torch)
    source, *_ = owner(torch, identity)
    return source, identity, values


@pytest.mark.parametrize(
    "fault",
    (
        "role",
        "identity",
        "groups",
        "shape",
        "diagnostic",
        "consent",
        "replay",
        "evaluation",
        "budget",
        "spent_budget",
        "phase",
        "report",
    ),
)
def test_should_refuse_unbound_probe_before_constructing_any_native_owner(
    torch, fresh_source, fault
):
    source, identity, values = fresh_source
    _, training, phases = study.build_reserved_probe_batches(identity, values)
    readouts = phases["baseline"]
    batch, targets = training
    permissions = ExperiencePermissions(training=True)
    budget = fresh_budget()
    report = {}
    phase = "baseline"
    if fault == "role":
        object.__setattr__(batch, "role", "train")
    elif fault == "identity":
        object.__setattr__(batch, "identity", replace(identity, revision="foreign"))
    elif fault == "groups":
        object.__setattr__(targets, "group_codes", (3, 1))
    elif fault == "shape":
        object.__setattr__(readouts["probe_test"], "states", readouts["probe_test"].states[:1])
    elif fault == "diagnostic":
        readouts["probe_train"].states.add_(0.25)
    elif fault == "consent":
        permissions = ExperiencePermissions()
    elif fault == "replay":
        permissions = ExperiencePermissions(training=True, replay=True)
    elif fault == "evaluation":
        permissions = ExperiencePermissions(training=True, evaluation=True)
    elif fault == "budget":
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=7), lambda: 0.0)
    elif fault == "spent_budget":
        budget.updates_completed = 1
    elif fault == "phase":
        phase = "after_A"
    else:
        report["complete"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid reserved probe reached native construction")

    before = source.work
    with pytest.raises(ValueError):
        run_reserved_probe(
            source,
            training,
            readouts,
            report,
            phase=phase,
            probe_factory=forbidden,
            permissions=permissions,
            budget=budget,
        )
    assert source.work == before and source.completed_updates == 0


@pytest.mark.parametrize("fault", ("feature", "prefix"))
def test_should_keep_first_paid_probe_receipt_and_refuse_retry_after_callback_corruption(
    torch, fresh_source, fault
):
    source, identity, values = fresh_source
    _, training, phases = study.build_reserved_probe_batches(identity, values)
    report: dict[str, Any] = {}
    budget = fresh_budget()
    captured, work = source._capture(), source.work

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            if fault == "feature":
                training[0].states.add_(0.125)
            else:
                report["readouts"][0]["probabilities"][0][0] += 0.01
                report["readouts"][0]["probabilities"][0][1] -= 0.01

    with pytest.raises(ValueError):
        run_reserved_probe(
            source,
            training,
            phases["baseline"],
            report,
            phase="baseline",
            probe_factory=probe_factory,
            permissions=ExperiencePermissions(training=True),
            budget=budget,
            save=save,
        )
    assert report["complete"] is False and len(report["error"]) <= 530
    assert report["stream"]["completed_updates"] == budget.updates_completed == 1
    assert report["stream"]["training_pairs"] == 2
    assert len(report["stream"]["receipts"]) == 1
    assert report["probe_work"]["train_attempts"] == report["probe_work"]["commits"] == 1
    assert report["probe_work"]["choice_prediction_returns"] == 3
    assert source.work == work and native._equal(vars(captured), vars(source._capture()))
    paid = deepcopy(report["probe_work"])
    with pytest.raises(ValueError):
        run_reserved_probe(
            source,
            training,
            phases["baseline"],
            report,
            phase="baseline",
            probe_factory=probe_factory,
            permissions=ExperiencePermissions(training=True),
            budget=budget,
        )
    assert report["probe_work"] == paid


@pytest.mark.parametrize(
    "arguments,expected", [([], {}), (["--reserved-task-probes"], {"reserved_task_probes": True})]
)
def test_should_keep_reserved_probe_cli_off_unless_explicit(
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


def test_should_refuse_probe_comparison_mode_before_any_loader(monkeypatch, tmp_path):
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
            "--reserved-task-probes",
            "--compare-catalog-support",
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("incompatible probe modes reached native loading")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ("repeat", "route"))
def test_should_refuse_catalog_support_probe_union_before_native_work(torch, mode):
    identity, values = features(torch)
    with pytest.raises(ValueError):
        study.build_reserved_probe_batches(identity, values, catalog_support=mode)


@pytest.fixture(scope="module")
def scored_probes(full_probe_stream):
    report = deepcopy(full_probe_stream["report"])
    original = deepcopy(report)
    work = full_probe_stream["source"].work
    result = study.score_reserved_probes(report)
    assert report == original and full_probe_stream["source"].work == work
    return report, result


def test_should_score_cached_probe_roles_after_all_trajectories_and_keep_all_endpoints(
    scored_probes,
):
    _, scores = scored_probes
    assert scores["cached_probability_rows"] == 64
    assert [(row["source_update"], row["phase"]) for row in scores["checkpoints"]] == list(
        CHECKPOINTS
    )
    for point in scores["checkpoints"]:
        assert tuple(point["groups"]) == GROUPS
        for group, expected_count in zip(GROUPS, (2, 2, 4)):
            endpoints = point["groups"][group]
            before, after = endpoints["before"], endpoints["after"]
            assert len(before["cases"]) == len(after["cases"]) == expected_count
            role = "outer_selection" if group == "old_task" else group
            assert all(row["role"] == role for row in before["cases"] + after["cases"])
            difference = endpoints["after_minus_before"]
            for key in ("accuracy", "brier_mean", "ece_5bin"):
                assert difference[key] == pytest.approx(
                    after["metrics"][key] - before["metrics"][key]
                )
            if group != "old_task":
                assert [row["target_id"] for row in before["cases"]] == ["inspect", "route"]
                assert [row["request"]["query"] for row in before["cases"]] == list(
                    PROBE_QUERIES[:2] if group == "probe_train" else PROBE_QUERIES[2:]
                )


@pytest.mark.parametrize(
    "fault",
    (
        "main_partial",
        "missing",
        "order",
        "partial",
        "work",
        "receipt",
        "clock",
        "phase",
        "readout",
        "probability",
    ),
)
def test_should_validate_all_four_cached_probe_prefixes_before_scoring_any_group(
    scored_probes, monkeypatch, fault
):
    report = deepcopy(scored_probes[0])
    children = report["reserved_probes"]
    last = children[-1]
    if fault == "main_partial":
        report["sequence_complete"] = False
    elif fault == "missing":
        children.pop()
    elif fault == "order":
        children.reverse()
    elif fault == "partial":
        last["complete"] = False
    elif fault == "work":
        last["probe_work"]["choice_prediction_returns"] = 5
    elif fault == "receipt":
        last["stream"]["receipts"][0]["sample_id"] = "replaced"
    elif fault == "clock":
        last["stream"]["receipts"][0]["applied_at"] += 1
    elif fault == "phase":
        last["phase"] = "after_A"
    elif fault == "readout":
        last["readouts"][-1]["group"] = "probe_test"
    else:
        last["readouts"][-1]["probabilities"][0][0] += 0.1

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid last probe exposed earlier probe scores")

    monkeypatch.setattr(study, "_reserved_probe_metrics", forbidden)
    with pytest.raises(ValueError):
        study.score_reserved_probes(report)


@pytest.mark.parametrize("mode", ("repeat", "route"))
def test_should_refuse_probe_catalog_support_cli_before_any_loader(monkeypatch, tmp_path, mode):
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
            "--reserved-task-probes",
            "--catalog-support",
            mode,
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported feature union reached native loading")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2


def test_should_validate_initial_control_before_admitting_checkpoint_probe(torch):
    identity, values = features(torch)
    base, _, _ = study.build_reserved_probe_batches(identity, values)
    training, choices = study.build_feature_batches(identity, base)
    source, *_ = owner(torch, identity)
    report = {"control_probabilities": {"v1": [[1.0, 0.0, 0.0] for _ in range(4)]}}

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched baseline control admitted a probe")

    with pytest.raises(ValueError, match="unchanged released control"):
        run_temporal_stream(source, training, choices, report, on_checkpoint=forbidden)
    assert source.completed_updates == 0 and source.work["choice_prediction_returns"] == 1
    assert report["stream"]["receipts"] == [] and report["rows"] == []
    assert "reserved_probes" not in report


@pytest.mark.parametrize("fault", ("value", "replacement"))
def test_should_refuse_corrupted_completed_probe_before_any_benchmark_update(torch, fault):
    identity, values = features(torch)
    base, training, readouts = study.build_reserved_probe_batches(identity, values)
    benchmark, choices = study.build_feature_batches(identity, base)
    source, *_ = owner(torch, identity)
    report: dict[str, Any] = {}
    hook = study.make_reserved_probe_hook(training, readouts, report)

    def save():
        children = report["reserved_probes"]
        if children and children[-1].get("complete"):
            if fault == "replacement":
                children[-1] = deepcopy(children[-1])
            else:
                children[-1]["stream"]["receipts"][0]["sample_id"] = "corrupted"

    with pytest.raises(ValueError, match="checkpoint probe"):
        run_temporal_stream(source, benchmark, choices, report, on_checkpoint=hook, save=save)
    assert source.completed_updates == 0 and source.training_examples == 0
    assert source.work["train_attempts"] == 0
    child = report["reserved_probes"][0]
    assert child["probe_work"]["commits"] == 6
    assert child["probe_work"]["training_rows_attempted"] == 12
    assert report["stream"]["receipts"] == []


@pytest.mark.parametrize("fault", ("work", "unknown", "receipt", "clock", "rows", "diagnostic"))
def test_should_admit_main_paid_completion_before_any_cached_probe_score(
    scored_probes, monkeypatch, fault
):
    report = deepcopy(scored_probes[0])
    if fault == "work":
        report["learner_work"]["commits"] = 11
    elif fault == "unknown":
        report["learner_work"]["unreturned_work_unknown"] = True
    elif fault == "receipt":
        report["stream"]["receipts"][-1]["sample_id"] = "replaced"
    elif fault == "clock":
        report["stream"]["receipts"][-1]["applied_at"] += 1
    elif fault == "rows":
        report["rows"].pop()
    else:
        report["stream"]["receipts"][-1]["diagnostic"]["value"] = float("nan")

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt main completion exposed probe scores")

    monkeypatch.setattr(study, "_reserved_probe_metrics", forbidden)
    with pytest.raises(ValueError):
        study.score_reserved_probes(report)
