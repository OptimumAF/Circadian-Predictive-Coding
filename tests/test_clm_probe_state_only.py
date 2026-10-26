"""Explicit state-only catalog adaptation preserves candidate geometry and source."""

from copy import deepcopy
from dataclasses import replace
from math import exp, sqrt
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_catalog_probe as catalog_probe
import src.adapters.clm_checkpoint_probes as checkpoint
import src.adapters.clm_head_learner as native
import src.app.clm_reserved_probe as probe_app
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.adapters.clm_frozen_encoder import text_recipe
from src.core.experience import ExperiencePermissions
from src.core.learner_ports import TrainingUpdateFailed
from src.core.temporal_decision_sequence import CHECKPOINTS
from src.infra.clm_local_loader import LoadedCLM
from test_clm_catalog_learning import identity, owner, parts
from test_clm_frozen_encoder import FakeTokenizer, backbone
from test_clm_probe_rehearsal import budget, data
from test_clm_typed_stream_readout import LiteralSampler


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def factory(source, *, scope="state_only", readouts=8):
    return CLMCatalogProbe(
        source,
        policy=replace(source.policy, max_updates=6, max_restores=0),
        max_readouts=readouts,
        parameter_scope=scope,
    )


def fixed_values(probe):
    state, action, scale = probe._learner._bundle[:3]
    return deepcopy(action.state_dict()), scale.detach().clone()


def assert_fixed(torch, probe, original):
    learner = probe._learner
    action, scale = learner._bundle[1:3]
    assert native._equal(action.state_dict(), original[0])
    assert torch.equal(scale, original[1])
    fixed = [*action.parameters(), scale]
    assert all(parameter.grad is None for parameter in fixed)
    optimized = [
        parameter for group in learner._bundle[3].param_groups for parameter in group["params"]
    ]
    state_parameters = list(learner._bundle[0].parameters())
    assert len(optimized) == len(state_parameters)
    assert all(left is right for left, right in zip(optimized, state_parameters, strict=True))
    assert all(parameter is not selected for parameter in fixed for selected in optimized)
    assert probe.fixed_parameters_unchanged is True


def scalar_state_gradient(state_weights, action_weights, scale, states, actions, labels):
    """Derivative of normalized dot-product CE, independent of Torch/autograd."""

    def projected(weights, vector):
        raw = [sum(weight * value for weight, value in zip(row, vector)) for row in weights]
        norm = sqrt(sum(value * value for value in raw))
        return [value / norm for value in raw], norm

    candidates = [projected(action_weights, vector)[0] for vector in actions]
    gradient = [[0.0 for _ in state_weights[0]] for _ in state_weights]
    temperature = min(exp(scale), 100.0)
    for vector, label in zip(states, labels, strict=True):
        query, norm = projected(state_weights, vector)
        logits = [
            temperature * sum(x * y for x, y in zip(query, candidate)) for candidate in candidates
        ]
        masses = [exp(value - max(logits)) for value in logits]
        probabilities = [value / sum(masses) for value in masses]
        direction = [
            sum(
                probability * candidate[axis]
                for probability, candidate in zip(probabilities, candidates)
            )
            - candidates[label][axis]
            for axis in range(len(query))
        ]
        radial = sum(value * derivative for value, derivative in zip(query, direction))
        derivative = [
            temperature * (value - coordinate * radial) / norm / len(states)
            for value, coordinate in zip(direction, query)
        ]
        for axis, value in enumerate(derivative):
            for column, feature in enumerate(vector):
                gradient[axis][column] += value * feature
    return gradient


@pytest.fixture(scope="module")
def direct_case(torch):
    training, _, _, _, _ = data(torch)
    batch, targets = training["baseline"]
    source, source_parts = owner(torch, maximum=12)
    original, source_work = source._capture(), source.work
    rng = torch.get_rng_state().clone()
    selected = factory(source)
    default = CLMCatalogProbe(
        source, policy=replace(source.policy, max_updates=6, max_restores=0), max_readouts=8
    )
    fixed = fixed_values(selected)
    state_before = selected._learner._bundle[0].weight.detach().clone()
    expected_gradient = scalar_state_gradient(
        selected._learner._bundle[0].weight.detach().tolist(),
        selected._learner._bundle[1].weight.detach().tolist(),
        selected._learner._bundle[2].detach().item(),
        batch.states.tolist(),
        batch.actions.tolist(),
        targets.candidate_indices,
    )
    gradients, selected_diagnostics, default_diagnostics = [], [], []
    real_clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients.append(parameters[0].grad.detach().clone())
        return real_clip(parameters, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
        for _ in range(6):
            selected_diagnostics.append(selected.train_batch(batch, targets))
            assert_fixed(torch, selected, fixed)
    for _ in range(6):
        default_diagnostics.append(default.train_batch(batch, targets))
    assert selected_diagnostics[0] == default_diagnostics[0]
    assert native._equal(vars(original), vars(source._capture()))
    assert source.work == source_work
    assert all(
        parameter.requires_grad for head in source_parts[:2] for parameter in head.parameters()
    )
    assert source_parts[2].requires_grad is True
    assert torch.equal(torch.get_rng_state(), rng)
    return dict(
        selected=selected,
        default=default,
        original_fixed=fixed,
        state_before=state_before,
        expected_gradient=expected_gradient,
        gradients=gradients,
    )


def test_should_change_state_using_correct_catalog_gradient_with_fixed_candidates(
    torch, direct_case
):
    case, selected = direct_case, direct_case["selected"]
    assert selected.parameter_scope == "state_only"
    assert selected.parameter_counts == {"optimized": 4, "fixed": 5, "total": 9}
    assert selected.completed_updates == 6 and selected.training_examples == 12
    assert len(case["gradients"]) == 6
    torch.testing.assert_close(
        case["gradients"][0], torch.tensor(case["expected_gradient"]), rtol=2e-5, atol=2e-7
    )
    assert case["gradients"][0].abs().sum().item() > 0
    assert not torch.equal(selected._learner._bundle[0].weight, case["state_before"])
    assert_fixed(torch, selected, case["original_fixed"])
    assert selected.work["backward_returns"] == selected.work["optimizer_returns"] == 6
    assert selected.work["catalog_logit_values_attempted"] == 36


def test_should_preserve_all_parameter_default_and_report_independent_counts(torch, direct_case):
    default = direct_case["default"]
    assert default.parameter_scope == "all"
    assert default.parameter_counts == {"optimized": 9, "fixed": 0, "total": 9}
    assert default.fixed_parameters_unchanged is True
    assert not native._equal(
        default._learner._bundle[1].state_dict(), direct_case["original_fixed"][0]
    )
    assert not torch.equal(default._learner._bundle[2], direct_case["original_fixed"][1])
    counts = direct_case["selected"].parameter_counts
    counts["optimized"] = -1
    assert direct_case["selected"].parameter_counts["optimized"] == 4
    with pytest.raises(AttributeError):
        direct_case["selected"].parameter_scope = "all"


@pytest.fixture(scope="module")
def full_case(torch):
    state_texts, action_texts = study.feature_texts(include_reserved_probe=True)
    state, action, scale = parts(torch)
    loaded = LoadedCLM(
        backbone(torch),
        FakeTokenizer({text: [index + 1] for index, text in enumerate(state_texts + action_texts)}),
        state,
        action,
        scale,
        replace(identity(), recipe=text_recipe(2048)),
        {},
    )
    original = deepcopy(state.state_dict()), deepcopy(action.state_dict()), scale.detach().clone()
    rng = torch.get_rng_state().clone()
    report: dict[str, Any] = {}
    observed, updates = [], []
    real_probe, real_train = probe_app.run_reserved_probe, CLMCatalogProbe.train_batch

    def checked_train(probe, batch, targets):
        fixed = fixed_values(probe)
        assert probe.parameter_scope == "state_only"
        result = real_train(probe, batch, targets)
        assert_fixed(torch, probe, fixed)
        updates.append(probe.completed_updates)
        return result

    def checked(source, *args, **kwargs):
        before, work = source._capture(), source.work
        owners = tuple(source._owners)
        assert kwargs["parameter_scope"] == "state_only"
        assert report["sequence_complete"] is False and "reserved_probe_scores" not in report
        result = real_probe(source, *args, **kwargs)
        assert native._equal(vars(before), vars(source._capture()))
        assert source.work == work
        assert all(left is right for left, right in zip(owners, source._owners, strict=True))
        observed.append((kwargs["phase"], source.completed_updates))
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "load_local_clm", lambda *args, **kwargs: loaded)
        patch.setattr(study, "ProcessRssSampler", LiteralSampler)
        patch.setattr(probe_app, "run_reserved_probe", checked)
        patch.setattr(CLMCatalogProbe, "train_batch", checked_train)
        study.run_case(
            Path("literal-model"),
            Path("literal-head"),
            report,
            catalog_training=True,
            reserved_task_probes=True,
            probe_rehearsal=True,
            probe_train_readouts=True,
            probe_state_only=True,
        )
    return dict(
        report=report, observed=observed, updates=updates, loaded=loaded, original=original, rng=rng
    )


def test_should_use_state_only_for_all_four_disposable_probes_without_changing_main(
    torch, full_case
):
    report = full_case["report"]
    assert report["complete"] is True and report["sequence_complete"] is True
    assert report["probe_parameter_scope"] == "state_only"
    assert full_case["observed"] == [(phase, update) for update, phase in CHECKPOINTS]
    assert full_case["updates"] == list(range(1, 7)) * 4
    assert report["stream"]["completed_updates"] == report["learner_work"]["train_attempts"] == 12
    assert report["stream"]["training_pairs"] == 36
    for child in report["reserved_probes"]:
        assert child["parameter_update_scope"] == {
            "name": "state_only",
            "optimized_parameter_elements": 4,
            "fixed_parameter_elements": 5,
            "total_parameter_elements": 9,
            "fixed_parameters_unchanged": True,
        }
        assert child["stream"]["completed_updates"] == 6
        baseline = child["phase"] == "baseline"
        assert child["probe_work"] == probe_app._expected_work(
            6,
            6 if baseline else 8,
            16 if baseline else 22,
            catalog_training=True,
            rehearsal_rows=0 if baseline else 2,
        )
        assert ("rehearsal" in child) is not baseline
        assert ("arrived_train_readouts" in child) is not baseline
    assert study._control_matches(torch, full_case["loaded"], full_case["original"])
    assert torch.equal(torch.get_rng_state(), full_case["rng"])
    saved = deepcopy(report)
    assert study.score_reserved_probes(report) == report["reserved_probe_scores"]
    assert report == saved


@pytest.mark.parametrize(
    "fault",
    (
        "missing_outer",
        "outer_all",
        "outer_boolean",
        "missing_child",
        "child_all",
        "extra",
        "boolean_count",
        "zero_optimized",
        "negative_fixed",
        "total",
        "fixed_false",
        "fixed_int",
        "partial",
        "baseline_missing",
    ),
)
def test_should_refuse_unbound_parameter_scope_before_any_quality(full_case, monkeypatch, fault):
    report = deepcopy(full_case["report"])
    child = report["reserved_probes"][-1]
    record = child["parameter_update_scope"]
    if fault == "missing_outer":
        report.pop("probe_parameter_scope")
    elif fault == "outer_all":
        report["probe_parameter_scope"] = "all"
    elif fault == "outer_boolean":
        report["probe_parameter_scope"] = True
    elif fault == "missing_child":
        child.pop("parameter_update_scope")
    elif fault == "child_all":
        record["name"] = "all"
    elif fault == "extra":
        record["retuned"] = False
    elif fault == "boolean_count":
        record["optimized_parameter_elements"] = True
    elif fault == "zero_optimized":
        record["optimized_parameter_elements"] = 0
    elif fault == "negative_fixed":
        record["fixed_parameter_elements"] = -1
    elif fault == "total":
        record["total_parameter_elements"] += 1
    elif fault == "fixed_false":
        record["fixed_parameters_unchanged"] = False
    elif fault == "fixed_int":
        record["fixed_parameters_unchanged"] = 1
    elif fault == "partial":
        child["complete"] = False
    else:
        report["reserved_probes"][0].pop("parameter_update_scope")

    def forbidden(*args, **kwargs):
        raise AssertionError("corrupt parameter-scope evidence exposed quality")

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


def test_should_combine_fixed_geometry_with_marker_control(torch, source_after_a):
    training, readouts, _, _, arrived = data(torch)
    report: dict[str, Any] = {}
    probe_app.run_reserved_probe(
        source_after_a,
        training["after_A"],
        readouts["after_A"],
        report,
        phase="after_A",
        probe_factory=factory,
        permissions=ExperiencePermissions(training=True),
        budget=budget(),
        catalog_training=True,
        parameter_scope="state_only",
        marker_control=checkpoint.build_probe_marker_control(training["after_A"], phase="after_A"),
        training_readout=arrived["after_A"],
    )
    assert (
        report["complete"] is True and report["marker_control"]["extra_marker_rows_attempted"] == 12
    )
    assert report["parameter_update_scope"]["fixed_parameters_unchanged"] is True
    assert report["probe_work"] == probe_app._expected_work(
        6, 8, 22, catalog_training=True, extra_marker_rows=2
    )


@pytest.mark.parametrize(
    "fault",
    (
        "action_value",
        "scale_value",
        "action_grad",
        "scale_grad",
        "optimizer",
        "optimizer_state",
        "scheduler",
        "state_disabled",
        "action_disabled",
        "scale_disabled",
        "scope",
    ),
)
def test_should_refuse_changed_fixed_geometry_and_owners_before_paid_work(torch, fault):
    source, _ = owner(torch, maximum=12)
    probe = factory(source)
    learner = probe._learner
    state, action, scale, optimizer, scheduler = learner._bundle
    if fault == "action_value":
        with torch.no_grad():
            action.weight.add_(0.25)
    elif fault == "scale_value":
        with torch.no_grad():
            scale.add_(0.25)
    elif fault == "action_grad":
        action.weight.grad = torch.zeros_like(action.weight)
    elif fault == "scale_grad":
        scale.grad = torch.zeros_like(scale)
    elif fault == "optimizer":
        optimizer.param_groups[0]["params"].append(scale)
    elif fault == "optimizer_state":
        optimizer.state[scale] = {}
    elif fault == "scheduler":
        scheduler.optimizer = source._bundle[3]
    elif fault == "state_disabled":
        state.weight.requires_grad_(False)
    elif fault == "action_disabled":
        action.weight.requires_grad_(False)
    elif fault == "scale_disabled":
        scale.requires_grad_(False)
    else:
        probe._parameter_scope = "all"
    training, _, _, _, _ = data(torch)
    before = probe.work
    with pytest.raises(ValueError):
        probe.train_batch(*training["baseline"])
    assert probe.work == before and probe.completed_updates == 0


def test_should_keep_paid_failed_state_only_backward_without_commit_or_source_mutation(
    torch, monkeypatch
):
    source, _ = owner(torch, maximum=12)
    probe = factory(source)
    training, _, _, _, _ = data(torch)
    before, work = source._capture(), source.work
    fixed = fixed_values(probe)
    real_loss = catalog_probe._state_only_catalog_loss

    def disconnected(state, action, scale, states, actions, indices):
        loss, logits, labels = real_loss(state, action, scale, states, actions, indices)
        # A successful backward that omits the optimized state must still fail.
        return loss.detach().requires_grad_(), logits, labels

    monkeypatch.setattr(catalog_probe, "_state_only_catalog_loss", disconnected)
    with pytest.raises(TrainingUpdateFailed):
        probe.train_batch(*training["baseline"])
    assert probe.work["train_attempts"] == probe.work["backward_returns"] == 1
    assert probe.work["optimizer_attempts"] == probe.completed_updates == 0
    assert probe.work["unreturned_work_unknown"] is True
    assert native._equal(probe._learner._bundle[1].state_dict(), fixed[0])
    assert torch.equal(probe._learner._bundle[2], fixed[1])
    assert native._equal(vars(before), vars(source._capture())) and source.work == work
    paid = probe.work
    with pytest.raises(ValueError):
        probe.train_batch(*training["baseline"])
    assert probe.work == paid


@pytest.mark.parametrize("scope", (None, True, 1, "", "STATE_ONLY", "state_only ", "action_only"))
def test_should_refuse_unknown_or_noncanonical_scope_before_constructing_probe(torch, scope):
    source, _ = owner(torch, maximum=12)
    before = source.work
    with pytest.raises(ValueError):
        factory(source, scope=scope)
    assert source.work == before


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
def test_should_refuse_invalid_state_only_cli_before_loading(monkeypatch, tmp_path, flags):
    output = tmp_path / "new.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "clm",
            "--model-dir",
            str(tmp_path),
            "--head",
            str(tmp_path / "head.pt"),
            "--output",
            str(output),
            "--probe-state-only",
            *flags,
        ],
    )
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid CLI loaded")
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("mode", (None, "probe-rehearsal", "probe-marker-control"))
def test_should_route_explicit_state_only_and_train_readouts_with_existing_modes(
    monkeypatch, tmp_path, mode
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
            "--probe-state-only",
            "--probe-train-readouts",
            *([] if mode is None else ["--" + mode]),
        ],
    )

    def execute(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report.update(complete=True)
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    expected = dict(
        catalog_training=True,
        reserved_task_probes=True,
        probe_state_only=True,
        probe_train_readouts=True,
    )
    if mode:
        expected[mode.replace("-", "_")] = True
    assert calls == [expected]


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"catalog_training": True},
        {"reserved_task_probes": True},
        {"catalog_training": True, "reserved_task_probes": True, "catalog_support": "route"},
        {"catalog_training": True, "reserved_task_probes": True, "compare_catalog_support": True},
        {"catalog_training": True, "reserved_task_probes": True, "typed_development_readout": True},
        {"catalog_training": True, "reserved_task_probes": True, "probe_state_only": 1},
    ),
)
def test_should_refuse_invalid_state_only_api_before_loading(monkeypatch, options):
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid API loaded")
    )
    with pytest.raises(ValueError):
        study.run_case(Path("unused"), Path("unused"), {}, **{"probe_state_only": True, **options})


def test_should_refuse_factory_scope_mismatch_before_any_native_update(torch):
    source, _ = owner(torch, maximum=12)
    training, readouts, _, _, _ = data(torch)
    report: dict[str, Any] = {}
    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source,
            training["baseline"],
            readouts["baseline"],
            report,
            phase="baseline",
            probe_factory=lambda original: factory(original, scope="all"),
            permissions=ExperiencePermissions(training=True),
            budget=budget(),
            catalog_training=True,
            parameter_scope="state_only",
        )
    assert report["complete"] is False
    assert source.work["train_attempts"] == 0


def test_should_keep_paid_progress_when_scope_evidence_changes_after_one_update(torch):
    source, _ = owner(torch, maximum=12)
    training, readouts, _, _, _ = data(torch)
    report: dict[str, Any] = {}
    session = budget()

    def save():
        if report.get("stream", {}).get("completed_updates") == 1:
            report["parameter_update_scope"]["fixed_parameters_unchanged"] = False

    with pytest.raises(ValueError):
        probe_app.run_reserved_probe(
            source,
            training["baseline"],
            readouts["baseline"],
            report,
            phase="baseline",
            probe_factory=factory,
            permissions=ExperiencePermissions(training=True),
            budget=session,
            catalog_training=True,
            parameter_scope="state_only",
            save=save,
        )
    assert report["complete"] is False
    assert report["stream"]["completed_updates"] == session.updates_completed == 1
    assert report["probe_work"]["commits"] == 1
    assert len(report["stream"]["receipts"]) == 1
    assert source.work["train_attempts"] == 0


@pytest.mark.parametrize("fixed_part", ("action", "temperature"))
def test_should_reject_staged_fixed_corruption_before_publication(torch, monkeypatch, fixed_part):
    source, _ = owner(torch, maximum=12)
    probe = factory(source)
    original = probe._learner._capture()
    source_before, source_work = source._capture(), source.work
    real_loss = catalog_probe._state_only_catalog_loss

    def corrupt(state, action, scale, states, actions, indices):
        result = real_loss(state, action, scale, states, actions, indices)
        with torch.no_grad():
            (action.weight if fixed_part == "action" else scale).add_(0.125)
        return result

    monkeypatch.setattr(catalog_probe, "_state_only_catalog_loss", corrupt)
    training, _, _, _, _ = data(torch)
    with pytest.raises(TrainingUpdateFailed, match="fixed action parameters or scale"):
        probe.train_batch(*training["baseline"])
    assert probe.completed_updates == 0
    assert probe.work["optimizer_returns"] == probe.work["scheduler_returns"] == 1
    assert probe.work["commits"] == 0 and probe.work["unreturned_work_unknown"] is True
    assert native._equal(vars(original), vars(probe._learner._capture()))
    assert native._equal(vars(source_before), vars(source._capture()))
    assert source.work == source_work
