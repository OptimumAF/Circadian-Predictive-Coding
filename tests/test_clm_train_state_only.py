"""TRAIN-role constrained catalog updates, continuation and existing stream use.

All heads and features are literal; no encoder, initialization draw or native
model is needed. State-only scope never grants access to probe/final labels.
"""

from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import scripts.run_clm_temporal_sequence as study
import src.adapters.clm_catalog_learner as catalog
import src.adapters.clm_catalog_probe as catalog_probe
import src.adapters.clm_head_learner as native
import src.adapters.clm_state_only_catalog_learner as constrained
from src.adapters.clm_catalog_probe import CLMCatalogProbe
from src.core.clm_catalog import FrozenCatalogTrainingBatch
from src.core.clm_catalog_probe import FrozenProbeCatalogBatch
from src.core.clm_head import CLMHeadPolicy, FrozenChoiceBatch
from src.core.learner_ports import TrainingUpdateFailed
from test_clm_catalog_learning import batch, identity, parts, temporal_features
from test_clm_probe_state_only import scalar_state_gradient


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def owner(torch, *, maximum=3, restores=1):
    source = parts(torch)
    learner = constrained.CLMStateOnlyCatalogLearner(
        *source,
        identity=identity(),
        policy=CLMHeadPolicy(
            projection_dim=2, max_updates=maximum, max_rows=4, max_restores=restores
        ),
    )
    return learner, source


def assert_fixed(torch, learner, action, scale):
    assert native._equal(learner._bundle[1].state_dict(), action)
    assert torch.equal(learner._bundle[2], scale)
    fixed = [*learner._bundle[1].parameters(), learner._bundle[2]]
    assert all(parameter.requires_grad and parameter.grad is None for parameter in fixed)
    optimized = [p for group in learner._bundle[3].param_groups for p in group["params"]]
    expected = list(learner._bundle[0].parameters())
    assert len(optimized) == len(expected)
    assert all(left is right for left, right in zip(optimized, expected, strict=True))
    assert all(parameter is not selected for parameter in fixed for selected in optimized)
    assert all(any(key is p for p in expected) for key in learner._bundle[3].state)
    assert learner.fixed_parameters_unchanged is True


@pytest.fixture(scope="module")
def trajectory(torch):
    learner, source = owner(torch)
    reference, _ = owner(torch)
    packet, targets = batch(torch)
    source_before = tuple(deepcopy(head.state_dict()) for head in source[:2])
    source_modes = tuple(head.training for head in source[:2])
    rng = torch.get_rng_state().clone()
    fixed = deepcopy(learner._bundle[1].state_dict()), learner._bundle[2].detach().clone()
    expected_gradient = scalar_state_gradient(
        learner._bundle[0].weight.detach().tolist(),
        learner._bundle[1].weight.detach().tolist(),
        learner._bundle[2].detach().item(),
        packet.states.tolist(),
        packet.actions.tolist(),
        targets.candidate_indices,
    )
    gradients = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def capture(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients.append(parameters[0].grad.detach().clone())
        return real_clip(parameters, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.nn.utils, "clip_grad_norm_", capture)
        first = learner.train_batch(packet, targets)
    learned = learner.snapshot_state()
    learner.train_batch(packet, targets)
    second = learner.snapshot_state()
    paid = learner.work
    learner.restore_state(learned)
    restored = learner.snapshot_state()
    assert native._equal(vars(restored), vars(learned))
    learner.train_batch(packet, targets)
    repeated = learner.snapshot_state()
    reference.train_batch(packet, targets)
    reference.train_batch(packet, targets)
    return dict(
        learner=learner,
        reference=reference,
        source=source,
        source_before=source_before,
        source_modes=source_modes,
        rng=rng,
        fixed=fixed,
        packet=packet,
        targets=targets,
        first=first,
        learned=learned,
        second=second,
        repeated=repeated,
        restored=restored,
        paid=paid,
        gradients=gradients,
        expected_gradient=expected_gradient,
    )


def test_should_update_only_state_with_independent_normalized_ce_gradient(torch, trajectory):
    learner = trajectory["learner"]
    torch.testing.assert_close(
        trajectory["gradients"][0],
        torch.tensor(trajectory["expected_gradient"]),
        rtol=1e-5,
        atol=1e-7,
    )
    assert trajectory["first"].definition == catalog.OBJECTIVE
    assert not native._equal(learner._bundle[0].state_dict(), trajectory["source_before"][0])
    assert_fixed(torch, learner, *trajectory["fixed"])
    assert learner.parameter_scope == "state_only"
    assert learner.parameter_counts == {"optimized": 4, "fixed": 5, "total": 9}
    counts = learner.parameter_counts
    counts["optimized"] = 999
    assert learner.parameter_counts["optimized"] == 4
    for name, value in (
        ("parameter_scope", "all"),
        ("parameter_counts", {}),
        ("fixed_parameters_unchanged", False),
    ):
        with pytest.raises(AttributeError):
            setattr(learner, name, value)


def test_should_resume_nonempty_adamw_and_schedule_without_renewing_physical_work(trajectory):
    case, learner = trajectory, trajectory["learner"]
    assert native._equal(vars(case["second"]), vars(case["repeated"]))
    assert native._equal(vars(case["repeated"]), vars(case["reference"].snapshot_state()))
    snapshot = case["restored"]
    assert snapshot.completed_updates == 1 and snapshot.training_examples == 2
    assert snapshot.gradients[0] is not None and snapshot.gradients[1:] == (None, None)
    assert len(snapshot.optimizer_state["state"]) == 1
    assert snapshot.scheduler_state["last_epoch"] == 1
    assert learner.completed_updates == 2 and learner.training_examples == 4
    assert learner.work["train_attempts"] == learner.work["commits"] == 3
    assert learner.work["training_rows_attempted"] == 6
    assert learner.work["catalog_candidate_rows_attempted"] == 9
    assert learner.work["catalog_logit_values_attempted"] == 18
    assert learner.work["restore_attempts"] == learner.work["restore_returns"] == 1
    paid = learner.work
    with pytest.raises(ValueError, match="allowance exhausted"):
        learner.train_batch(case["packet"], case["targets"])
    with pytest.raises(ValueError, match="restore allowance exhausted"):
        learner.restore_state(case["learned"])
    assert learner.work == paid


def test_should_own_source_and_snapshot_tensors_and_preserve_caller_rng(torch, trajectory):
    case, learner = trajectory, trajectory["learner"]
    assert torch.equal(torch.get_rng_state(), case["rng"])
    for head, values, mode in zip(
        case["source"][:2], case["source_before"], case["source_modes"], strict=True
    ):
        assert native._equal(head.state_dict(), values) and head.training is mode
    assert type(case["learned"]) is constrained.CLMStateOnlyCatalogSnapshot
    assert {field.name for field in fields(case["learned"])} == {
        field.name for field in fields(native.CLMHeadSnapshot)
    }
    incoming = learner.snapshot_state()
    before = learner._capture()
    incoming.state_head["weight"].add_(5)
    incoming.action_head["weight"].add_(5)
    incoming.logit_scale.add_(5)
    incoming.gradients[0].add_(5)
    next(iter(incoming.optimizer_state["state"].values()))["exp_avg"].add_(5)
    incoming.cpu_rng.zero_()
    assert native._equal(vars(before), vars(learner._capture()))
    for owned, original in zip(learner._bundle[:2], case["source"][:2], strict=True):
        assert all(
            left is not right and left.data_ptr() != right.data_ptr()
            for left, right in zip(owned.parameters(), original.parameters(), strict=True)
        )


def test_should_preserve_readout_state_and_fixed_owners(torch, trajectory):
    learner, packet = trajectory["learner"], trajectory["packet"]
    before = learner._capture()
    result = learner.predict_choices(FrozenChoiceBatch(identity(), packet.states, packet.actions))
    assert result.shape == (2, 3)
    assert torch.allclose(result.sum(dim=1), torch.ones(2))
    assert native._equal(vars(before), vars(learner._capture()))
    assert_fixed(torch, learner, *trajectory["fixed"])


def test_should_clear_only_owned_fixed_gradients_when_warm_inputs_have_gradients(torch):
    source = parts(torch)
    parameters = [*source[0].parameters(), *source[1].parameters(), source[2]]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    before = [parameter.grad.clone() for parameter in parameters]
    learner = constrained.CLMStateOnlyCatalogLearner(
        *source,
        identity=identity(),
        policy=CLMHeadPolicy(projection_dim=2, max_updates=3, max_rows=4),
    )
    for parameter, expected in zip(parameters, before, strict=True):
        assert torch.equal(parameter.grad, expected)
    assert torch.equal(learner._bundle[0].weight.grad, before[0])
    assert learner._bundle[1].weight.grad is None and learner._bundle[2].grad is None
    assert learner.fixed_parameters_unchanged is True


def test_should_match_accepted_private_probe_loss_without_a_second_backward(torch):
    source = parts(torch)
    packet, targets = batch(torch)
    indices = torch.tensor(targets.candidate_indices, dtype=torch.int64)
    public = constrained.state_only_catalog_head_loss(
        source[0], source[1], source[2], packet.states, packet.actions, indices
    )
    private = catalog_probe._state_only_catalog_loss(
        source[0], source[1], source[2], packet.states, packet.actions, indices
    )
    for left, right in zip(public, private, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert all(parameter.grad is None for head in source[:2] for parameter in head.parameters())
    assert source[2].grad is None


@pytest.mark.parametrize("kind", ("catalog", "paired", "extra", "subclass"))
def test_should_reject_cross_scope_objective_and_nonexact_snapshot_without_paid_restore(
    torch, kind
):
    selected, source = owner(torch)
    plain = catalog.CLMCatalogLearner(*source, identity=identity(), policy=selected.policy)
    paired = native.CLMHeadLearner(*source, identity=identity(), policy=selected.policy)
    snapshot = selected.snapshot_state()
    before = selected.work
    if kind == "catalog":
        foreign = plain.snapshot_state()
    elif kind == "paired":
        foreign = paired.snapshot_state()
    elif kind == "extra":
        foreign = deepcopy(snapshot)
        object.__setattr__(foreign, "parameter_scope", "state_only")
    else:

        class ChildSnapshot(constrained.CLMStateOnlyCatalogSnapshot):
            pass

        foreign = ChildSnapshot(**vars(snapshot))
    with pytest.raises(ValueError):
        selected.restore_state(foreign)
    assert selected.work == before
    for other in (plain, paired):
        other_before = other.work
        with pytest.raises(ValueError):
            other.restore_state(snapshot)
        assert other.work == other_before


@pytest.mark.parametrize("role", ("probe_train", "probe_test", "outer_selection", "final_test"))
def test_should_refuse_unarrived_or_probe_role_before_any_training_work(torch, role):
    learner, _ = owner(torch)
    packet, targets = batch(torch)
    object.__setattr__(packet, "role", role)
    before = learner.work
    with pytest.raises(ValueError):
        learner.train_batch(packet, targets)
    assert learner.work == before and learner.completed_updates == 0
    probe = FrozenProbeCatalogBatch(identity(), packet.states, packet.actions, packet.candidate_ids)
    with pytest.raises(ValueError):
        learner.train_batch(probe, targets)
    assert learner.work == before


@pytest.mark.parametrize(
    "fault",
    (
        "action",
        "scale",
        "action_grad",
        "scale_grad",
        "state_disabled",
        "fixed_disabled",
        "foreign_state",
    ),
)
def test_should_refuse_live_fixed_or_optimizer_corruption_before_paid_work(torch, fault):
    learner, _ = owner(torch)
    state, action, scale, optimizer, _ = learner._bundle
    with torch.no_grad():
        if fault == "action":
            action.weight.add_(0.25)
        elif fault == "scale":
            scale.add_(0.25)
        elif fault == "action_grad":
            action.weight.grad = torch.zeros_like(action.weight)
        elif fault == "scale_grad":
            scale.grad = torch.zeros_like(scale)
        elif fault == "state_disabled":
            state.weight.requires_grad_(False)
        elif fault == "fixed_disabled":
            action.weight.requires_grad_(False)
        else:
            optimizer.state[scale] = {}
    packet, targets = batch(torch)
    before = learner.work
    with pytest.raises(ValueError):
        learner.train_batch(packet, targets)
    assert learner.work == before


@pytest.mark.parametrize("fixed_part", ("action", "scale"))
def test_should_keep_paid_staged_update_corruption_unpublished(torch, monkeypatch, fixed_part):
    learner, _ = owner(torch)
    before = learner._capture()
    real_loss = constrained.state_only_catalog_head_loss

    def corrupt(state, action, scale, states, actions, indices):
        result = real_loss(state, action, scale, states, actions, indices)
        with torch.no_grad():
            (action.weight if fixed_part == "action" else scale).add_(0.125)
        return result

    monkeypatch.setattr(constrained, "state_only_catalog_head_loss", corrupt)
    packet, targets = batch(torch)
    with pytest.raises(TrainingUpdateFailed):
        learner.train_batch(packet, targets)
    assert native._equal(vars(before), vars(learner._capture()))
    assert learner.work["train_attempts"] == learner.work["optimizer_returns"] == 1
    assert learner.work["scheduler_returns"] == 1 and learner.work["commits"] == 0
    assert learner.work["unreturned_work_unknown"] is True
    paid = learner.work
    with pytest.raises(ValueError):
        learner.train_batch(packet, targets)
    assert learner.work == paid


def test_should_charge_disconnected_state_backward_without_commit(torch, monkeypatch):
    learner, _ = owner(torch)
    before = learner._capture()
    real_loss = constrained.state_only_catalog_head_loss

    def disconnected(*args):
        loss, logits, labels = real_loss(*args)
        return loss.detach().requires_grad_(), logits, labels

    monkeypatch.setattr(constrained, "state_only_catalog_head_loss", disconnected)
    packet, targets = batch(torch)
    with pytest.raises(TrainingUpdateFailed):
        learner.train_batch(packet, targets)
    assert learner.work["train_attempts"] == learner.work["backward_returns"] == 1
    assert learner.work["optimizer_attempts"] == learner.work["commits"] == 0
    assert learner.work["unreturned_work_unknown"] is True
    assert native._equal(vars(before), vars(learner._capture()))


@pytest.mark.parametrize(
    "fault", ("action", "scale", "action_grad", "scale_grad", "foreign_state", "foreign_birth")
)
def test_should_reject_paid_restore_that_changes_fixed_birth_without_publication(torch, fault):
    learner, _ = owner(torch)
    snapshot = learner.snapshot_state()
    if fault == "action":
        snapshot.action_head["weight"].add_(0.125)
    elif fault == "scale":
        snapshot.logit_scale.add_(0.125)
    elif fault in ("action_grad", "scale_grad"):
        gradients = list(snapshot.gradients)
        gradients[1 if fault == "action_grad" else 2] = torch.zeros_like(
            snapshot.action_head["weight"] if fault == "action_grad" else snapshot.logit_scale
        )
        object.__setattr__(snapshot, "gradients", tuple(gradients))
    elif fault == "foreign_state":
        snapshot.optimizer_state["state"][99] = {}
    else:
        source = parts(torch)
        with torch.no_grad():
            source[1].weight.add_(0.125)
        foreign = constrained.CLMStateOnlyCatalogLearner(
            *source, identity=identity(), policy=learner.policy
        )
        snapshot = foreign.snapshot_state()
    before = learner._capture()
    with pytest.raises(ValueError):
        learner.restore_state(snapshot)
    assert native._equal(vars(before), vars(learner._capture()))
    assert learner.work["restore_attempts"] == 1 and learner.work["restore_returns"] == 0
    assert learner.work["unreturned_work_unknown"] is True
    paid = learner.work
    with pytest.raises(ValueError):
        learner.restore_state(snapshot)
    assert learner.work == paid


@pytest.mark.parametrize("scope", ("all", "state_only"))
def test_should_start_independent_disposable_probe_from_train_source_without_sharing_updates(
    torch, scope
):
    source, _ = owner(torch)
    packet, targets = batch(torch)
    before, work = source._capture(), source.work
    probe = CLMCatalogProbe(
        source, policy=replace(source.policy, max_restores=0), max_readouts=1, parameter_scope=scope
    )
    assert probe.parameter_scope == scope
    assert probe.parameter_counts == (
        {"optimized": 9, "fixed": 0, "total": 9}
        if scope == "all"
        else {"optimized": 4, "fixed": 5, "total": 9}
    )
    probe.train_batch(
        FrozenProbeCatalogBatch(identity(), packet.states, packet.actions, packet.candidate_ids),
        targets,
    )
    assert probe.completed_updates == 1 and source.completed_updates == 0
    assert native._equal(vars(before), vars(source._capture())) and source.work == work


def test_should_refuse_train_source_subclasses_before_constructing_disposable_probe(torch):
    class ChildLearner(constrained.CLMStateOnlyCatalogLearner):
        pass

    source = ChildLearner(
        *parts(torch),
        identity=identity(),
        policy=CLMHeadPolicy(projection_dim=2, max_updates=3, max_rows=4),
    )
    before = source.work
    with pytest.raises(ValueError):
        CLMCatalogProbe(source, policy=replace(source.policy, max_restores=0), max_readouts=1)
    assert source.work == before


def test_should_run_existing_twelve_update_stream_with_explicit_train_scope(torch, monkeypatch):
    source = parts(torch)
    loaded = SimpleNamespace(
        state_head=source[0], action_head=source[1], logit_scale=source[2], identity=identity()
    )
    original = tuple(deepcopy(head.state_dict()) for head in source[:2])
    scale = source[2].detach().clone()
    training, choices = study.build_catalog_training_batches(identity(), temporal_features(torch))
    report: dict[str, Any] = {}
    owners = []

    class ObservedLearner(constrained.CLMStateOnlyCatalogLearner):
        def train_batch(self, packet, targets):
            result = super().train_batch(packet, targets)
            assert type(packet) is FrozenCatalogTrainingBatch and packet.role == "train"
            assert_fixed(torch, self, original[1], scale)
            owners.append((self.completed_updates, self.parameter_counts))
            return result

    monkeypatch.setattr(constrained, "CLMStateOnlyCatalogLearner", ObservedLearner)
    rng = torch.get_rng_state().clone()
    study._run_head_stream(
        loaded,
        training,
        choices,
        report,
        mode="none",
        diagnostics=False,
        before_step=lambda: None,
        save=lambda: None,
        catalog_training=True,
        train_state_only=True,
    )
    assert report["sequence_complete"] is True
    assert report["stream"]["completed_updates"] == report["learner_work"]["train_attempts"] == 12
    assert [step for step, _ in owners] == list(range(1, 13))
    assert all(counts == {"optimized": 4, "fixed": 5, "total": 9} for _, counts in owners)
    assert report["train_parameter_update_scope"] == {
        "name": "state_only",
        "optimized_parameter_elements": 4,
        "fixed_parameter_elements": 5,
        "total_parameter_elements": 9,
        "fixed_parameters_unchanged": True,
    }
    assert "probe_parameter_scope" not in report and "reserved_probes" not in report
    assert all(native._equal(head.state_dict(), values) for head, values in zip(source, original))
    assert torch.equal(source[2], scale) and torch.equal(torch.get_rng_state(), rng)
    study._require_catalog_completed(report)
    for fault in ("missing", "extra", "scope", "boolean", "total", "fixed", "too_large"):
        corrupted = deepcopy(report)
        record = corrupted["train_parameter_update_scope"]
        if fault == "missing":
            record.pop("fixed_parameter_elements")
        elif fault == "extra":
            record["retuned"] = False
        elif fault == "scope":
            record["name"] = "all"
        elif fault == "boolean":
            record["optimized_parameter_elements"] = True
        elif fault == "total":
            record["total_parameter_elements"] += 1
        elif fault == "fixed":
            record["fixed_parameters_unchanged"] = False
        else:
            record.update(
                optimized_parameter_elements=32_000_000, total_parameter_elements=32_000_005
            )
        with pytest.raises(ValueError):
            study._require_catalog_completed(corrupted)


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"catalog_training": True, "train_state_only": 1},
        {"catalog_training": True, "typed_development_readout": True},
    ),
)
def test_should_refuse_invalid_train_scope_before_loader(monkeypatch, options):
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid scope loaded model")
    )
    with pytest.raises(ValueError):
        study.run_case(Path("unused"), Path("unused"), {}, **{"train_state_only": True, **options})


@pytest.mark.parametrize("probe_scope", (False, True))
def test_should_route_independent_train_and_probe_scopes_via_cli(
    monkeypatch, tmp_path, probe_scope
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
            str(tmp_path / "out.json"),
            "--catalog-training",
            "--train-state-only",
            *(["--reserved-task-probes", "--probe-state-only"] if probe_scope else []),
        ],
    )

    def execute(model_dir, head, report, *, train_catalog_diagnostics, save, **kwargs):
        calls.append(kwargs)
        report["complete"] = True
        save()

    monkeypatch.setattr(study, "run_case", execute)
    study.main()
    expected = {"catalog_training": True, "train_state_only": True}
    if probe_scope:
        expected.update(reserved_task_probes=True, probe_state_only=True)
    assert calls == [expected]


@pytest.mark.parametrize("flags", ([], ["--catalog-training", "--typed-development-readout"]))
def test_should_refuse_unsupported_train_scope_cli_before_output_or_loader(
    monkeypatch, tmp_path, flags
):
    output = tmp_path / "out.json"
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
            "--train-state-only",
            *flags,
        ],
    )
    monkeypatch.setattr(
        study, "load_local_clm", lambda *args, **kwargs: pytest.fail("invalid CLI loaded model")
    )
    with pytest.raises(SystemExit) as error:
        study.main()
    assert error.value.code == 2 and not output.exists()
