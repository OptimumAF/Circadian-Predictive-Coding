"""Workflow32 source isolation and original-TRAIN observation in existing probes."""

from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import src.adapters.clm_checkpoint_probes as adapter
import src.app.clm_reserved_probe as app
from src.adapters.clm_head_learner import _equal
from src.app.clm_temporal_sequence import _probabilities
from src.app.clm_workflow_training import ACTIVE_IDS, fit_catalog_workflow
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_catalog import CLMCatalogTargets
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_probe import FrozenProbeChoiceBatch
from src.core.experience import ExperiencePermissions
from test_clm_catalog_learning import owner
from test_clm_catalog_reserved_probe import batches
from test_clm_workflow_passes import make_case

CONSENT = ExperiencePermissions(training=True)


def budget():
    return ToyBudgetSession(ToyExecutionBudget(max_training_updates=6), lambda: 0.0)


def trained_workflow():
    source, batch, targets, _ = make_case(32)
    fit_catalog_workflow(source, batch, targets, {}, passes=32)
    return source, (batch, targets)


@pytest.fixture(scope="module")
def warm():
    return trained_workflow()


def packets(warm):
    source, (original, targets) = warm
    old = replace(original, states=original.states.clone(), actions=original.actions.clone())
    ids = ("defer", "inspect", "route")
    actions = old.actions[[ACTIVE_IDS.index(c) for c in ids]].clone()
    states = torch.tensor([[0.7, 0.1], [-0.2, 0.8]], dtype=torch.float32)
    identity = source.identity
    training = (
        FrozenProbeCatalogBatch(identity, states, actions, ids),
        CLMCatalogTargets(ids, (1, 2)),
    )
    readouts = dict(
        probe_train=FrozenProbeChoiceBatch(identity, states, actions, "probe_train"),
        probe_test=FrozenProbeChoiceBatch(
            identity, torch.tensor([[0.8, 0.2], [-0.1, 0.9]]), actions, "probe_test"
        ),
        old_task=FrozenProbeChoiceBatch(old.identity, old.states, old.actions, "train"),
    )
    return training, readouts, (old, targets)


def direct(source, training, readouts, original, report, **overrides):
    options = dict(
        phase="after_return",
        probe_factory=adapter._new_catalog_probe,
        permissions=CONSENT,
        budget=budget(),
        catalog_training=True,
        workflow_training_readout=original,
    )
    options.update(overrides)
    return app.run_reserved_probe(source, training, readouts, report, **options)


@pytest.mark.parametrize("parameter_scope", ["all", "state_only"])
def test_should_execute_six_isolated_updates_and_read_original_twelve_train_rows(
    warm, monkeypatch, parameter_scope
):
    source, _ = warm
    training, readouts, original = packets(warm)
    snapshot, work, rng = source._capture(), source.work, torch.get_rng_state().clone()
    owners = tuple(source._owners)
    calls, probes, births = [], [], []
    real_new = adapter._new_catalog_probe

    def factory(learner, **options):
        probe = real_new(learner, **options)
        probes.append(probe)
        births.append(tuple(deepcopy(head.state_dict()) for head in probe._learner._bundle[:2]))
        real_train = probe.train_batch

        def train(batch, labels):
            calls.append((batch.role, len(batch.states), len(batch.actions), batch.candidate_ids))
            assert batch.states is not original[0].states
            result = real_train(batch, labels)
            if parameter_scope == "state_only":
                from test_clm_probe_state_only import assert_fixed

                assert_fixed(torch, probe, (births[0][1], source._bundle[2].detach()))
            return result

        probe.train_batch = train
        return probe

    monkeypatch.setattr(adapter, "_new_catalog_probe", factory)
    report: dict[str, Any]
    authority, report = budget(), {}
    adapter.run_workflow_reserved_probe(
        source,
        training,
        readouts,
        original,
        report,
        permissions=CONSENT,
        budget=authority,
        parameter_scope=parameter_scope,
    )
    assert report["complete"] and report["source_update"] == 32
    assert "arrived_train_readouts" not in report and "groups" not in report
    assert calls == [("probe_train", 2, 3, ("defer", "inspect", "route"))] * 6
    assert authority.updates_completed == 6 and authority.replay_examples_completed == 0
    assert report["stream"]["training_pairs"] == 12
    assert [r["applied_at"] for r in report["stream"]["receipts"]] == list(range(1, 12, 2))
    records = report["readouts"]
    assert [(r["point"], r["group"]) for r in records] == [
        (p, g) for p in ("before", "after") for g in ("probe_train", "probe_test", "old_task")
    ]
    assert [(len(r["probabilities"]), len(r["probabilities"][0])) for r in records] == [
        (2, 3),
        (2, 3),
        (12, 5),
    ] * 2
    metadata = report["workflow_training_readout"]
    assert metadata["source_role"] == "train" and metadata["source_row_indices"] == list(range(12))
    assert metadata["candidate_ids"] == list(ACTIVE_IDS)
    assert metadata["candidate_indices"] == list(original[1].candidate_indices)
    assert report["probe_work"] == app._expected_work(
        6, 6, 32, catalog_training=True, readout_candidate_rows=22
    )
    assert source.work == work and _equal(vars(source._capture()), vars(snapshot))
    assert all(a is b for a, b in zip(owners, source._owners, strict=True))
    assert torch.equal(torch.get_rng_state(), rng)
    assert not _equal(probes[0]._learner._bundle[0].state_dict(), births[0][0])
    if parameter_scope == "state_only":
        scope = report["parameter_update_scope"]
        assert scope["name"] == "state_only" and scope["fixed_parameters_unchanged"] is True
        counts = probes[0].parameter_counts
        assert scope["optimized_parameter_elements"] == counts["optimized"]
        assert scope["fixed_parameter_elements"] == counts["fixed"]
        assert scope["total_parameter_elements"] == counts["total"]
    else:
        assert "parameter_update_scope" not in report


@pytest.mark.parametrize(
    "fault",
    [
        "updates",
        "examples",
        "generation",
        "maximum",
        "rows",
        "restores",
        "work",
        "unknown",
        "worktype",
    ],
)
def test_should_refuse_mismatched_source_before_probe_creation(warm, fault):
    source, _ = warm
    training, readouts, original = packets(warm)
    fake = SimpleNamespace(
        identity=source.identity,
        policy=source.policy,
        completed_updates=32,
        training_examples=384,
        generation=32,
        work=source.work,
    )
    if fault == "updates":
        fake.completed_updates = 12
    elif fault == "examples":
        fake.training_examples = 383
    elif fault == "generation":
        fake.generation = True
    elif fault == "maximum":
        fake.policy = replace(fake.policy, max_updates=33)
    elif fault == "rows":
        fake.policy = replace(fake.policy, max_rows=13)
    elif fault == "restores":
        fake.policy = replace(fake.policy, max_restores=1)
    elif fault == "work":
        fake.work["commits"] = 31
    elif fault == "unknown":
        fake.work["unreturned_work_unknown"] = True
    elif fault == "worktype":
        fake.work["prediction_attempts"] = False
    invoked = []
    with pytest.raises(ValueError):
        direct(fake, training, readouts, original, {}, probe_factory=lambda _: invoked.append(True))
    assert invoked == [] and source.completed_updates == 32


@pytest.mark.parametrize(
    "fault",
    ["labels", "role", "catalog", "alias", "old_role", "final_role", "marker_role", "subset"],
)
def test_should_refuse_wrong_packet_labels_roles_or_original_owners_before_probe(warm, fault):
    source, _ = warm
    training, readouts, original = packets(warm)
    old, labels = original
    if fault == "labels":
        original = old, replace(labels, candidate_indices=(0,) * 12)
    elif fault == "role":
        object.__setattr__(old, "role", "probe_test")
    elif fault == "catalog":
        object.__setattr__(old, "candidate_ids", ("commit", "defer", "inspect", "open", "zzz"))
    elif fault == "alias":
        readouts["old_task"] = replace(readouts["old_task"], states=old.states.clone())
    elif fault == "old_role":
        readouts["old_task"] = replace(readouts["old_task"], role="outer_selection")
    elif fault == "final_role":
        object.__setattr__(readouts["probe_test"], "role", "final_test")
    elif fault == "marker_role":
        object.__setattr__(training[0], "role", "train")
    elif fault == "subset":
        training[0].actions[0].add_(0.1)
    invoked = []
    with pytest.raises(ValueError):
        direct(
            source, training, readouts, original, {}, probe_factory=lambda _: invoked.append(True)
        )
    assert invoked == []


@pytest.mark.parametrize(
    "option,value",
    [
        ("phase", "after_A"),
        ("catalog_training", False),
        ("parameter_scope", "unsupported"),
        ("training_readout", object()),
        ("replay_budget", object()),
        ("rehearsal", object()),
        ("marker_control", object()),
        ("workflow_training_readout", False),
    ],
)
def test_should_refuse_unqualified_cross_products(warm, option, value):
    source, _ = warm
    training, readouts, original = packets(warm)
    invoked = []
    with pytest.raises(ValueError):
        direct(
            source,
            training,
            readouts,
            original,
            {},
            probe_factory=lambda _: invoked.append(True),
            **{option: value},
        )
    assert invoked == []


@pytest.mark.parametrize("fault", ["budget", "features", "prefix", "metadata_owner", "save"])
@pytest.mark.parametrize("parameter_scope", ["all", "state_only"])
def test_should_stop_after_first_paid_update_and_preserve_the_trusted_prefix(
    warm, fault, parameter_scope
):
    source, _ = warm
    training, readouts, original = packets(warm)
    snapshot, work = source._capture(), source.work
    report: dict[str, Any] = {}
    authority, changed = budget(), False

    def save():
        nonlocal changed
        if changed or report.get("stream", {}).get("completed_updates") != 1:
            return
        changed = True
        if fault == "budget":
            object.__setattr__(authority.budget, "max_training_updates", 7)
        elif fault == "features":
            original[0].states.add_(0.1)
        elif fault == "prefix":
            report["readouts"][0]["probabilities"][0][0] = -1.0
        elif fault == "metadata_owner":
            report["workflow_training_readout"] = deepcopy(report["workflow_training_readout"])
        elif fault == "save":
            raise OSError("controlled paid save failure")

    with pytest.raises((ValueError, OSError)):
        adapter.run_workflow_reserved_probe(
            source,
            training,
            readouts,
            original,
            report,
            permissions=CONSENT,
            budget=authority,
            parameter_scope=parameter_scope,
            save=save,
        )
    assert changed and report["complete"] is False
    assert len(report["stream"]["receipts"]) == report["probe_work"]["optimizer_returns"] == 1
    prefix = report["preserved_workflow_probe_prefix"]
    assert prefix["stream"]["completed_updates"] == len(prefix["stream"]["receipts"]) == 1
    assert prefix["readouts"][0]["probabilities"][0][0] >= 0
    assert source.work == work and _equal(vars(source._capture()), vars(snapshot))
    with pytest.raises(ValueError, match="fresh|budget"):
        adapter.run_workflow_reserved_probe(
            source, training, readouts, original, {}, permissions=CONSENT, budget=authority
        )
    assert source.work == work


@pytest.mark.parametrize("parameter_scope", ["all", "state_only"])
def test_should_keep_paid_read_cost_when_native_return_is_lost(warm, monkeypatch, parameter_scope):
    source, _ = warm
    training, readouts, original = packets(warm)
    report: dict[str, Any] = {}
    real_new = adapter._new_catalog_probe

    def factory(learner, **options):
        probe = real_new(learner, **options)
        real_read = probe.predict_choices

        def lost(batch):
            real_read(batch)
            raise RuntimeError("controlled native read return lost")

        probe.predict_choices = lost
        return probe

    monkeypatch.setattr(adapter, "_new_catalog_probe", factory)
    with pytest.raises(RuntimeError, match="return lost"):
        adapter.run_workflow_reserved_probe(
            source,
            training,
            readouts,
            original,
            report,
            permissions=CONSENT,
            budget=budget(),
            parameter_scope=parameter_scope,
        )
    assert report["complete"] is False and report["stream"]["receipts"] == []
    assert (
        report["probe_work"]["prediction_attempts"]
        == report["probe_work"]["prediction_returns"]
        == 1
    )
    assert report["probe_work"]["train_attempts"] == 0
    assert report["preserved_workflow_probe_prefix"]["readouts"][0]["probabilities"] is None


def test_should_refuse_source_weight_corruption_before_another_probe_update():
    isolated = trained_workflow()
    source, _ = isolated
    training, readouts, original = packets(isolated)
    report: dict[str, Any] = {}
    changed = False

    def save():
        nonlocal changed
        if not changed and report.get("stream", {}).get("completed_updates") == 1:
            changed = True
            with torch.no_grad():
                next(source._bundle[0].parameters()).add_(0.1)

    with pytest.raises(ValueError, match="original|changed"):
        adapter.run_workflow_reserved_probe(
            source,
            training,
            readouts,
            original,
            report,
            permissions=CONSENT,
            budget=budget(),
            save=save,
        )
    assert changed and report["complete"] is False
    assert report["probe_work"]["optimizer_returns"] == 1 and source.work["commits"] == 32


@pytest.mark.parametrize("arrived_readout", [False, True])
def test_should_keep_original_temporal_default_reports_and_source_behavior(arrived_readout):
    base, training, readouts = batches(torch)
    import scripts.run_clm_temporal_sequence as study

    benchmark, _ = study.build_catalog_training_batches(training["baseline"][0].identity, base)
    source, _ = owner(torch, maximum=12)
    for packet, targets in benchmark:
        source.train_batch(packet, targets)
    before = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-workflow-probe-20261025/app_before.py"
    )
    spec = importlib.util.spec_from_file_location("original_reserved_app", before)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    snapshot, work = source._capture(), source.work
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    options = dict(
        phase="after_return",
        probe_factory=adapter._new_catalog_probe,
        permissions=CONSENT,
        catalog_training=True,
    )
    if arrived_readout:
        options["training_readout"] = benchmark[8]
        options["probe_factory"] = lambda learner: adapter._new_catalog_probe(
            learner, max_readouts=8
        )
    old.run_reserved_probe(
        source,
        training["after_return"],
        readouts["after_return"],
        first,
        budget=budget(),
        **options,
    )
    app.run_reserved_probe(
        source,
        training["after_return"],
        readouts["after_return"],
        second,
        budget=budget(),
        **options,
    )
    assert first == second and "workflow_training_readout" not in second
    assert source.work == work and _equal(vars(source._capture()), vars(snapshot))


def test_should_require_the_declared_probability_catalog_width():
    values = torch.tensor([[0.2] * 5, [0.2] * 5], dtype=torch.float32)
    assert _probabilities(values, rows=2, candidates=5) == values.tolist()
    with pytest.raises(ValueError):
        _probabilities(values, rows=2)
    with pytest.raises(ValueError):
        _probabilities(values, rows=2, candidates=True)


def test_should_refuse_equal_valued_work_type_corruption_after_one_update(warm, monkeypatch):
    """Corrupt the observed work view without mutating/refunding native counters."""
    source, _ = warm
    original_work = source.work
    training, readouts, original = packets(warm)
    report: dict[str, Any] = {}
    changed = False

    class WorkView:
        def __getattr__(self, name):
            return getattr(source, name)

        @property
        def work(self):
            work = source.work
            if changed:
                work["prediction_attempts"] = False
            return work

    # The injected factory still creates a real separately owned catalog probe.
    real_factory = adapter._new_catalog_probe
    monkeypatch.setattr(adapter, "_new_catalog_probe", lambda _: real_factory(source))

    def save():
        nonlocal changed
        if report.get("stream", {}).get("completed_updates") == 1:
            changed = True

    with pytest.raises(ValueError, match="work|prefix"):
        adapter.run_workflow_reserved_probe(
            WorkView(),
            training,
            readouts,
            original,
            report,
            permissions=CONSENT,
            budget=budget(),
            save=save,
        )
    assert changed and report["complete"] is False
    assert report["probe_work"]["optimizer_returns"] == 1
    assert len(report["preserved_workflow_probe_prefix"]["stream"]["receipts"]) == 1
    assert source.work == original_work and type(source.work["prediction_attempts"]) is int


@pytest.mark.parametrize("scope", [None, True, 1, "ALL", "state-only"])
def test_should_refuse_invalid_workflow_scope_before_source_capture(scope):
    with pytest.raises(ValueError, match="parameter_scope"):
        adapter.run_workflow_reserved_probe(
            object(),
            None,
            None,
            None,
            {},
            permissions=CONSENT,
            budget=budget(),
            parameter_scope=scope,
        )


@pytest.mark.parametrize(
    "permissions",
    [
        ExperiencePermissions(),
        ExperiencePermissions(training=True, replay=True),
        ExperiencePermissions(training=True, evaluation=True),
    ],
)
def test_should_refuse_non_training_only_consent_for_workflow_state_only(warm, permissions):
    source, _ = warm
    training, readouts, original = packets(warm)
    authority, invoked = budget(), []
    with pytest.raises(ValueError, match="consent"):
        direct(
            source,
            training,
            readouts,
            original,
            {},
            parameter_scope="state_only",
            permissions=permissions,
            budget=authority,
            probe_factory=lambda _: invoked.append(True),
        )
    assert invoked == [] and authority.updates_completed == 0


@pytest.mark.parametrize("scope", ["all", "state_only"])
def test_should_refuse_wrong_workflow_factory_scope_before_any_update(warm, scope):
    source, _ = warm
    training, readouts, original = packets(warm)
    report: dict[str, Any] = {}
    authority = budget()
    opposite = "state_only" if scope == "all" else "all"
    with pytest.raises(ValueError, match="different probe parameter scope"):
        direct(
            source,
            training,
            readouts,
            original,
            report,
            parameter_scope=scope,
            budget=authority,
            probe_factory=lambda learner: adapter._new_catalog_probe(
                learner, parameter_scope=opposite
            ),
        )
    assert authority.updates_completed == 0 and not report["complete"]
    assert report["probe_work"]["train_attempts"] == 0
    assert report["probe_work"]["prediction_attempts"] == 0


@pytest.mark.parametrize("fault", ["scope_metadata", "scope_owner", "action", "scale"])
def test_should_stop_state_only_workflow_on_changed_fixed_evidence_after_one_update(
    warm, monkeypatch, fault
):
    source, _ = warm
    training, readouts, original = packets(warm)
    source_state, source_work, rng = source._capture(), source.work, torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    authority, probes = budget(), []
    real_new = adapter._new_catalog_probe

    def factory(learner, **options):
        probe = real_new(learner, **options)
        probes.append(probe)
        return probe

    monkeypatch.setattr(adapter, "_new_catalog_probe", factory)

    def save():
        if report.get("stream", {}).get("completed_updates") != 1:
            return
        if fault == "scope_metadata":
            report["parameter_update_scope"]["fixed_parameters_unchanged"] = False
        elif fault == "scope_owner":
            report["parameter_update_scope"] = deepcopy(report["parameter_update_scope"])
        else:
            fixed = probes[0]._learner._bundle[1:3]
            parameter = next(fixed[0].parameters()) if fault == "action" else fixed[1]
            with torch.no_grad():
                parameter.add_(0.1)

    with pytest.raises(ValueError, match="changed|fixed"):
        adapter.run_workflow_reserved_probe(
            source,
            training,
            readouts,
            original,
            report,
            permissions=CONSENT,
            budget=authority,
            parameter_scope="state_only",
            save=save,
        )
    assert not report["complete"] and authority.updates_completed == 1
    assert report["probe_work"]["optimizer_returns"] == 1
    prefix = report["preserved_workflow_probe_prefix"]
    assert prefix["parameter_update_scope"]["fixed_parameters_unchanged"] is True
    assert len(prefix["stream"]["receipts"]) == 1
    assert source.work == source_work and _equal(vars(source._capture()), vars(source_state))
    assert torch.equal(torch.get_rng_state(), rng)


def test_should_keep_preceding_workflow_all_parameter_report_exactly(warm):
    source, _ = warm
    training, readouts, original = packets(warm)
    before = (
        Path(__file__).resolve().parents[1]
        / "artifacts/runs/clm-workflow-probe-state-only-20261025/app_before.py"
    )
    spec = importlib.util.spec_from_file_location("preceding_workflow_app", before)
    assert spec is not None and spec.loader is not None
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    options = dict(
        phase="after_return",
        probe_factory=adapter._new_catalog_probe,
        permissions=CONSENT,
        catalog_training=True,
        workflow_training_readout=original,
    )
    old.run_reserved_probe(source, training, readouts, first, budget=budget(), **options)
    adapter.run_workflow_reserved_probe(
        source, training, readouts, original, second, permissions=CONSENT, budget=budget()
    )
    assert first == second and "parameter_update_scope" not in second
