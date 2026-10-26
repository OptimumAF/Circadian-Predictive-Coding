"""Catalog-objective reserved probes through the existing temporal stream."""

from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_checkpoint_probes as checkpoint
import src.adapters.clm_head_learner as native
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.app.clm_reserved_probe import run_reserved_probe
from src.app.clm_temporal_sequence import run_temporal_stream
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog import CATALOG_OBJECTIVE, CLMCatalogTargets
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import CLMTargets
from src.core.clm_probe import FrozenProbeEmbeddingBatch
from src.core.experience import ExperiencePermissions
from src.core.temporal_clm_probe import PROBE_QUERIES, reserved_probe_examples
from src.core.temporal_decision_sequence import CATALOG, CHECKPOINTS
from test_clm_catalog_learning import identity, owner

PHASES = tuple(phase for _, phase in CHECKPOINTS)
GROUPS = ("probe_train", "probe_test", "old_task")


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def features(torch):
    return torch.tensor(
        [[(index + 1) / 18, (-1) ** index * (index + 2) / 19] for index in range(16)],
        dtype=torch.float32,
    )


def batches(torch):
    return study.build_reserved_probe_batches(identity(), features(torch), catalog_training=True)


def fresh_budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0)


def factory(source):
    return CLMCatalogProbe(
        source, policy=replace(source.policy, max_updates=6, max_restores=0), max_readouts=6
    )


def test_should_refuse_installable_probe_factory_before_any_native_update(torch, fresh_source):
    _, training, readouts = batches(torch)
    before = fresh_source.work

    class Installable:
        work = dict(before)

        def train_batch(self, *args):
            raise AssertionError("unadmitted factory performed an update")

        def predict_choices(self, *args):
            raise AssertionError("unadmitted factory performed a readout")

        def install(self):
            raise AssertionError("disposable probe was installed")

    with pytest.raises(ValueError, match="install"):
        run_reserved_probe(
            fresh_source,
            training["baseline"],
            readouts["baseline"],
            {},
            phase="baseline",
            probe_factory=lambda _: Installable(),
            permissions=ExperiencePermissions(training=True),
            budget=fresh_budget(),
            catalog_training=True,
        )
    assert fresh_source.work == before


def test_should_bind_catalog_probe_training_to_each_phases_active_ids_and_fixed_targets(torch):
    values = features(torch)
    base, training, readouts = study.build_reserved_probe_batches(
        identity(), values, catalog_training=True
    )
    assert torch.equal(base, torch.cat((values[:8], values[12:])))
    assert tuple(training) == tuple(readouts) == PHASES
    global_ids = tuple(candidate.candidate_id for candidate in CATALOG)
    for phase, pair in training.items():
        batch, targets = pair
        examples = reserved_probe_examples(phase, "probe_train")
        ids = tuple(candidate.candidate_id for candidate in examples[0].request.allowed_candidates)
        assert type(batch) is FrozenProbeCatalogBatch and type(targets) is CLMCatalogTargets
        assert batch.identity == identity() and batch.role == "probe_train"
        assert batch.candidate_ids == targets.candidate_ids == ids
        assert targets.candidate_indices == tuple(
            ids.index(example.target_id) for example in examples
        )
        assert targets.candidate_indices == ((0, 2) if phase in ("baseline", "after_A") else (1, 2))
        assert torch.equal(batch.states, values[8:10])
        assert torch.equal(
            batch.actions, values[[12 + global_ids.index(candidate) for candidate in ids]]
        )
        assert tuple(readouts[phase]) == GROUPS
        assert torch.equal(batch.actions, readouts[phase]["probe_train"].actions)
        assert torch.equal(batch.states, readouts[phase]["probe_train"].states)
    original = training["baseline"][0].states.clone()
    values.zero_()
    assert torch.equal(training["baseline"][0].states, original)


@pytest.fixture(scope="module")
def full_stream(torch):
    base, training, readouts = batches(torch)
    benchmark, choices = study.build_catalog_training_batches(identity(), base)
    source, source_parts = owner(torch, maximum=12)
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    observed = []
    hook = study.make_reserved_probe_hook(training, readouts, report, catalog_training=True)

    def checked_hook(phase, learner):
        before, work = learner._capture(), learner.work
        owners, storage = tuple(learner._owners), tuple(learner._storage)
        assert report["sequence_complete"] is False and "checkpoints" not in report
        assert all("groups" not in child for child in report["reserved_probes"])
        hook(phase, learner)
        assert native._equal(vars(before), vars(learner._capture()))
        assert learner.work == work and tuple(learner._storage) == storage
        assert all(left is right for left, right in zip(owners, learner._owners, strict=True))
        observed.append((phase, learner.completed_updates))

    run_temporal_stream(
        source, benchmark, choices, report, catalog_training=True, on_checkpoint=checked_hook
    )
    return dict(
        source=source,
        source_parts=source_parts,
        report=report,
        observed=observed,
        rng=rng,
        training=training,
        readouts=readouts,
    )


def test_should_complete_all_36_updates_with_independent_main_and_probe_work(torch, full_stream):
    case, report = full_stream, full_stream["report"]
    assert report["sequence_complete"] is True and report["training_objective"] == CATALOG_OBJECTIVE
    assert case["observed"] == [(phase, update) for update, phase in CHECKPOINTS]
    assert case["source"].completed_updates == report["stream"]["completed_updates"] == 12
    assert report["stream"]["training_pairs"] == 36
    work = report["learner_work"]
    assert work == case["source"].work
    assert work["train_attempts"] == work["commits"] == work["backward_returns"] == 12
    assert (
        work["catalog_candidate_rows_attempted"] == 36
        and work["catalog_logit_values_attempted"] == 108
    )
    assert work["choice_prediction_returns"] == 5 and work["prediction_rows_attempted"] == 20
    assert work["choice_candidate_rows_attempted"] == 15
    children = report["reserved_probes"]
    assert len(children) == 4
    assert sum(child["stream"]["completed_updates"] for child in children) == 24
    assert sum(child["stream"]["training_pairs"] for child in children) == 48
    for child, (update, phase) in zip(children, CHECKPOINTS, strict=True):
        assert (
            child["source_update"] == update
            and child["phase"] == phase
            and child["complete"] is True
        )
        assert child["training_objective"] == CATALOG_OBJECTIVE
        batch, targets = case["training"][phase]
        assert child["candidate_ids"] == list(batch.candidate_ids)
        assert child["candidate_indices"] == list(targets.candidate_indices)
        paid = child["probe_work"]
        assert paid["train_attempts"] == paid["commits"] == paid["backward_returns"] == 6
        assert paid["training_rows_attempted"] == 12
        assert paid["catalog_candidate_rows_attempted"] == 18
        assert paid["catalog_logit_values_attempted"] == 36
        assert paid["choice_prediction_returns"] == 6
        assert (
            paid["prediction_rows_attempted"] == 16
            and paid["choice_candidate_rows_attempted"] == 18
        )
        assert paid["restore_attempts"] == paid["snapshot_attempts"] == 0
        assert paid["unreturned_work_unknown"] is False
    assert torch.equal(torch.get_rng_state(), case["rng"])


def test_should_keep_each_probes_separate_causal_clock_and_exact_catalog_receipts(full_stream):
    sample_ids = set()
    for child in full_stream["report"]["reserved_probes"]:
        phase = child["phase"]
        receipts = child["stream"]["receipts"]
        assert len(receipts) == 6
        assert [row["observed_at"] for row in receipts] == list(range(0, 12, 2))
        assert (
            [row["arrived_at"] for row in receipts]
            == [row["applied_at"] for row in receipts]
            == list(range(1, 12, 2))
        )
        assert [row["update_number"] for row in receipts] == list(range(1, 7))
        for index, row in enumerate(receipts):
            assert row["sample_id"] == f"clm-probe-{phase}-{index}"
            assert row["episode_id"] == row["learner_version"] == "clm-probe-" + phase
            assert row["diagnostic"]["definition"] == CATALOG_OBJECTIVE
            sample_ids.add(row["sample_id"])
        assert [(row["point"], row["group"]) for row in child["readouts"]] == [
            (point, group) for point in ("before", "after") for group in GROUPS
        ]
    assert len(sample_ids) == 24


@pytest.fixture(scope="module")
def scored_stream(full_stream):
    report = deepcopy(full_stream["report"])
    original, work = deepcopy(report), full_stream["source"].work
    scores = study.score_reserved_probes(report)
    assert report == original and full_stream["source"].work == work
    return report, scores


def test_should_keep_all_cached_before_after_metrics_and_explicit_probe_roles(scored_stream):
    _, scores = scored_stream
    assert scores["cached_probability_rows"] == 64
    assert [(row["source_update"], row["phase"]) for row in scores["checkpoints"]] == list(
        CHECKPOINTS
    )
    for point in scores["checkpoints"]:
        assert tuple(point["groups"]) == GROUPS
        for group, count in zip(GROUPS, (2, 2, 4), strict=True):
            endpoints = point["groups"][group]
            before, after = endpoints["before"], endpoints["after"]
            assert len(before["cases"]) == len(after["cases"]) == count
            role = "outer_selection" if group == "old_task" else group
            assert all(row["role"] == role for row in before["cases"] + after["cases"])
            differences = endpoints["after_minus_before"]
            for key in ("accuracy", "brier_mean", "ece_5bin"):
                assert differences[key] == pytest.approx(
                    after["metrics"][key] - before["metrics"][key]
                )
            if before["metrics"]["nll_mean"] is None or after["metrics"]["nll_mean"] is None:
                assert differences["nll_mean"] is None
            else:
                assert differences["nll_mean"] == pytest.approx(
                    after["metrics"]["nll_mean"] - before["metrics"]["nll_mean"]
                )
            assert (
                differences["obsolete_rule_error_count"]
                == after["obsolete_rule_errors"] - before["obsolete_rule_errors"]
            )
            if group != "old_task":
                expected = PROBE_QUERIES[:2] if group == "probe_train" else PROBE_QUERIES[2:]
                assert [row["request"]["query"] for row in before["cases"]] == list(expected)
                assert [row["target_id"] for row in before["cases"]] == ["inspect", "route"]


@pytest.mark.parametrize(
    "fault",
    (
        "main_objective",
        "missing_main_objective",
        "main_candidate_work",
        "main_logit_work",
        "main_receipt",
        "child_objective",
        "missing_child_objective",
        "child_ids",
        "child_indices",
        "candidate_work",
        "logit_work",
        "receipt",
        "receipt_clock",
        "probabilities",
        "partial",
    ),
)
def test_should_admit_all_catalog_evidence_before_decoding_any_probe_scores(
    scored_stream, monkeypatch, fault
):
    report = deepcopy(scored_stream[0])
    child = report["reserved_probes"][-1]
    if fault == "main_objective":
        report["training_objective"] = native.OBJECTIVE
    elif fault == "missing_main_objective":
        report.pop("training_objective")
    elif fault == "main_candidate_work":
        report["learner_work"]["catalog_candidate_rows_attempted"] -= 1
    elif fault == "main_logit_work":
        report["learner_work"]["catalog_logit_values_attempted"] -= 1
    elif fault == "main_receipt":
        report["stream"]["receipts"][-1]["diagnostic"]["definition"] = native.OBJECTIVE
    elif fault == "child_objective":
        child["training_objective"] = native.OBJECTIVE
    elif fault == "missing_child_objective":
        child.pop("training_objective")
    elif fault == "child_ids":
        child["candidate_ids"].reverse()
    elif fault == "child_indices":
        child["candidate_indices"][0] = 0
    elif fault == "candidate_work":
        child["probe_work"]["catalog_candidate_rows_attempted"] -= 1
    elif fault == "logit_work":
        child["probe_work"]["catalog_logit_values_attempted"] -= 1
    elif fault == "receipt":
        child["stream"]["receipts"][-1]["diagnostic"]["definition"] = native.OBJECTIVE
    elif fault == "receipt_clock":
        child["stream"]["receipts"][-1]["applied_at"] += 1
    elif fault == "probabilities":
        child["readouts"][-1]["probabilities"][0][0] += 0.1
    else:
        child["complete"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("unbound final catalog probe exposed earlier scores")

    monkeypatch.setattr(study, "_reserved_probe_metrics", forbidden)
    with pytest.raises(ValueError):
        study.score_reserved_probes(report)


@pytest.fixture(scope="module")
def fresh_source(torch):
    source, _ = owner(torch, maximum=12)
    return source


@pytest.mark.parametrize(
    "fault",
    (
        "paired_type",
        "train_role",
        "identity",
        "ids",
        "indices",
        "target_type",
        "candidates",
        "nonfinite",
        "diagnostic",
        "consent",
        "replay",
        "evaluation",
        "budget",
        "spent_budget",
    ),
)
def test_should_refuse_misbound_catalog_app_inputs_before_probe_construction(
    torch, fresh_source, fault
):
    _, phases, readouts = batches(torch)
    training = phases["baseline"]
    batch, targets = training
    permissions = ExperiencePermissions(training=True)
    budget = fresh_budget()
    if fault == "paired_type":
        training = (
            FrozenProbeEmbeddingBatch(batch.identity, batch.states, batch.actions[:2]),
            CLMTargets((1, 3)),
        )
    elif fault == "train_role":
        object.__setattr__(batch, "role", "train")
    elif fault == "identity":
        object.__setattr__(batch, "identity", replace(batch.identity, revision="foreign"))
    elif fault == "ids":
        object.__setattr__(batch, "candidate_ids", tuple(reversed(batch.candidate_ids)))
    elif fault == "indices":
        object.__setattr__(targets, "candidate_indices", (2, 0))
    elif fault == "target_type":
        training = batch, CLMTargets((1, 3))
    elif fault == "candidates":
        object.__setattr__(batch, "actions", batch.actions[:2])
    elif fault == "nonfinite":
        batch.actions[0, 0] = float("nan")
    elif fault == "diagnostic":
        readouts["baseline"]["probe_train"].states.add_(0.25)
    elif fault == "consent":
        permissions = ExperiencePermissions()
    elif fault == "replay":
        permissions = ExperiencePermissions(training=True, replay=True)
    elif fault == "evaluation":
        permissions = ExperiencePermissions(training=True, evaluation=True)
    elif fault == "budget":
        budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=7), lambda: 0.0)
    else:
        budget.updates_completed = 1

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid catalog app input reached a probe constructor")

    before = fresh_source.work
    with pytest.raises(ValueError):
        run_reserved_probe(
            fresh_source,
            training,
            readouts["baseline"],
            {},
            phase="baseline",
            probe_factory=forbidden,
            permissions=permissions,
            budget=budget,
            catalog_training=True,
        )
    assert fresh_source.work == before and fresh_source.completed_updates == 0


@pytest.mark.parametrize(
    "fault", ("order", "missing", "paired", "future_indices", "future_role", "future_catalog")
)
def test_should_validate_the_entire_phase_training_plan_before_any_source_work(torch, fault):
    _, training, readouts = batches(torch)
    if fault == "order":
        training = dict(reversed(tuple(training.items())))
    elif fault == "missing":
        training.pop("after_return")
    elif fault == "paired":
        training = training["baseline"]
    elif fault == "future_indices":
        object.__setattr__(training["after_return"][1], "candidate_indices", (0, 2))
    elif fault == "future_role":
        object.__setattr__(training["after_return"][0], "role", "train")
    else:
        training["after_return"][0].actions.add_(0.125)
    with pytest.raises(ValueError):
        study.make_reserved_probe_hook(training, readouts, {}, catalog_training=True)


@pytest.fixture(scope="module")
def completed_baseline(torch):
    _, training, readouts = batches(torch)
    source, _ = owner(torch, maximum=12)
    report: dict[str, Any] = {}
    hook = study.make_reserved_probe_hook(training, readouts, report, catalog_training=True)
    hook("baseline", source)
    return dict(source=source, report=report, hook=hook, training=training)


@pytest.mark.parametrize(
    "fault",
    (
        "completed_prefix",
        "replaced_child",
        "future_indices",
        "future_pair",
        "future_ids",
        "future_features",
    ),
)
def test_should_protect_prior_success_and_future_catalog_binding_before_more_probe_work(
    torch, completed_baseline, monkeypatch, fault
):
    case = completed_baseline
    records = case["report"]["reserved_probes"]
    original = records[0]
    pair = case["training"]["after_A"]
    restore = lambda: None
    if fault == "completed_prefix":
        old = original["candidate_indices"][0]
        original["candidate_indices"][0] = 2
        restore = lambda: original["candidate_indices"].__setitem__(0, old)
        message = "completed.*evidence changed"
    elif fault == "replaced_child":
        records[0] = deepcopy(original)
        restore = lambda: records.__setitem__(0, original)
        message = "record.*replaced"
    elif fault == "future_indices":
        old = pair[1].candidate_indices
        object.__setattr__(pair[1], "candidate_indices", (2, 0))
        restore = lambda: object.__setattr__(pair[1], "candidate_indices", old)
        message = "target|catalog|metadata"
    elif fault == "future_pair":
        case["training"]["after_A"] = (pair[0], deepcopy(pair[1]))
        restore = lambda: case["training"].__setitem__("after_A", pair)
        message = "owner|replaced|training"
    elif fault == "future_ids":
        old = pair[0].candidate_ids
        object.__setattr__(pair[0], "candidate_ids", tuple(reversed(old)))
        restore = lambda: object.__setattr__(pair[0], "candidate_ids", old)
        message = "catalog|metadata"
    else:
        old = pair[0].actions
        object.__setattr__(pair[0], "actions", old.clone())
        restore = lambda: object.__setattr__(pair[0], "actions", old)
        message = "feature"

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupted prior/future catalog evidence admitted another probe")

    monkeypatch.setattr(checkpoint, "_new_catalog_probe", forbidden)
    before = case["source"].work
    try:
        with pytest.raises(ValueError, match=message):
            case["hook"]("after_A", case["source"])
    finally:
        restore()
    assert case["source"].work == before
    assert len(records) == 1 and records[0]["complete"] is True


def test_should_preserve_paid_partial_catalog_work_and_refuse_restart_after_report_corruption(
    torch, fresh_source
):
    _, phases, readouts = batches(torch)
    training = phases["baseline"]
    report: dict[str, Any] = {}
    budget = fresh_budget()
    before, work = fresh_source._capture(), fresh_source.work

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            report["candidate_indices"][0] = 2

    with pytest.raises(ValueError):
        run_reserved_probe(
            fresh_source,
            training,
            readouts["baseline"],
            report,
            phase="baseline",
            probe_factory=factory,
            permissions=ExperiencePermissions(training=True),
            budget=budget,
            save=save,
            catalog_training=True,
        )
    assert report["complete"] is False and report["stream"]["completed_updates"] == 1
    assert budget.updates_completed == 1 and len(report["stream"]["receipts"]) == 1
    assert report["stream"]["receipts"][0]["diagnostic"]["definition"] == CATALOG_OBJECTIVE
    assert report["probe_work"]["commits"] == 1
    assert report["probe_work"]["catalog_candidate_rows_attempted"] == 3
    assert report["probe_work"]["catalog_logit_values_attempted"] == 6
    assert report["probe_work"]["choice_prediction_returns"] == 3
    assert fresh_source.work == work and native._equal(vars(before), vars(fresh_source._capture()))
    paid = deepcopy(report["probe_work"])
    with pytest.raises(ValueError):
        run_reserved_probe(
            fresh_source,
            training,
            readouts["baseline"],
            report,
            phase="baseline",
            probe_factory=factory,
            permissions=ExperiencePermissions(training=True),
            budget=budget,
            catalog_training=True,
        )
    assert report["probe_work"] == paid


def test_should_admit_explicit_catalog_and_reserved_cli_combination_without_changing_defaults(
    monkeypatch, tmp_path
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
            "--reserved-task-probes",
        ],
    )
    calls = []

    def case(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True, fixture="metadata only")
        save()

    monkeypatch.setattr(study, "run_case", case)
    study.main()
    assert calls == [{"catalog_training": True, "reserved_task_probes": True}]


@pytest.mark.parametrize(
    "other",
    (
        ["--catalog-support", "repeat"],
        ["--catalog-support", "route"],
        ["--compare-catalog-support"],
    ),
)
def test_should_refuse_catalog_probe_support_cross_products_before_any_loader(
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
            "--reserved-task-probes",
            *other,
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported catalog probe combination reached loader")

    monkeypatch.setattr(study, "run_case", forbidden)
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2
