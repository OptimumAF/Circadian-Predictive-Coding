"""Tiny pinned CLM head math and owned learner qualification, not encoder quality.

Only selected upstream function definitions are compiled as numerical references.
No upstream import/loader, checkpoint, tokenizer or released encoder is executed.
"""

import ast
from copy import deepcopy
from dataclasses import replace
from math import exp, log
from pathlib import Path

import pytest

from src.adapters import clm_head_learner as adapter
from src.core.clm_head import CLMHeadPolicy, CLMTargets, FrozenEmbeddingBatch, FrozenEncoderIdentity
from src.core.learner_ports import TrainingUpdateFailed


ROOT = Path(__file__).resolve().parents[1]
HEAD_SOURCE = ROOT / "artifacts/runs/native-decision-contracts-20261022/clm/src__clm__heads.py"
LOSS_SOURCE = ROOT / "artifacts/runs/clm-head-baseline-20261022/upstream-finetune.py"


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="declared tiny native head boundary requires Torch")


def compile_definition(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(definitions) == 1 and definitions[0].decorator_list == []
    body: list[ast.stmt] = list(definitions)
    selected = ast.Module(body=body, type_ignores=[])
    scope = dict(namespace)
    exec(compile(selected, str(path), "exec"), scope)
    return scope[name]


@pytest.fixture(scope="module")
def reference_functions(torch_backend):
    torch = torch_backend
    namespace = {"torch": torch, "F": torch.nn.functional, "PROJ_DIM": 2, "HIDDEN": 2}
    return compile_definition(HEAD_SOURCE, "make_head", namespace), compile_definition(
        LOSS_SOURCE, "_clm_loss", namespace
    )


@pytest.fixture(scope="module")
def seeded_parts(torch_backend, reference_functions):
    torch, (make_head, _) = torch_backend, reference_functions
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        state = make_head(width=3, depth=2, proj=2, hidden=2)
        action = make_head(width=3, depth=2, proj=2, hidden=2)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for head in (state, action) for p in head.parameters()) == 34
    state.eval()
    action.train()
    return state, action, torch.nn.Parameter(torch.tensor(log(1 / 0.07), dtype=torch.float32))


def identity():
    return FrozenEncoderIdentity(
        "fixture-only", "synthetic-v1", "tokenizer-none", "last", "handset-v1", 2
    )


def paired_batch(torch, *, role="train"):
    states = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-0.5, 0.25]], dtype=torch.float32)
    actions = torch.tensor([[0.75, -0.25], [0.0, 1.0], [-1.0, 0.5]], dtype=torch.float32)
    return FrozenEmbeddingBatch(identity(), states, actions, role=role), CLMTargets((0, 0, 1))


def make_owner(parts, *, maximum=4, restores=8):
    state, action, scale = parts
    return adapter.CLMHeadLearner(
        state,
        action,
        scale,
        identity=identity(),
        policy=CLMHeadPolicy(
            projection_dim=2, max_updates=maximum, max_parameters=4096, max_restores=restores
        ),
    )


def require_equal_tree(torch, left, right):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert type(right) is type(left) and left.keys() == right.keys()
        for key in left:
            require_equal_tree(torch, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            require_equal_tree(torch, first, second)
    else:
        assert left == right


def test_should_match_pinned_native_loss_logits_labels_and_every_head_and_scale_gradient(
    torch_backend, reference_functions, seeded_parts
):
    torch, (_, native_loss) = torch_backend, reference_functions
    actual, reference = deepcopy(seeded_parts), deepcopy(seeded_parts)
    batch, targets = paired_batch(torch)
    codes = torch.tensor(targets.group_codes, dtype=torch.int64)
    before = (batch.states.clone(), batch.actions.clone())
    loss, logits, labels = adapter.clm_head_loss(
        actual[0], actual[1], actual[2], batch.states, batch.actions, codes
    )
    expected_loss, expected_logits, expected_labels = native_loss(
        *reference, batch.states, batch.actions, codes
    )
    assert (
        torch.equal(loss, expected_loss)
        and torch.equal(logits, expected_logits)
        and torch.equal(labels, expected_labels)
    )
    loss.backward()
    expected_loss.backward()
    actual_parameters = [*actual[0].parameters(), *actual[1].parameters(), actual[2]]
    reference_parameters = [*reference[0].parameters(), *reference[1].parameters(), reference[2]]
    assert len(actual_parameters) == len(reference_parameters) == 9
    for parameter, expected in zip(actual_parameters, reference_parameters, strict=True):
        assert parameter.grad is not None and expected.grad is not None
        assert torch.equal(parameter.grad, expected.grad)
    assert torch.equal(batch.states, before[0]) and torch.equal(batch.actions, before[1])
    assert batch.states.grad is batch.actions.grad is None


@pytest.mark.parametrize(
    "codes,expected",
    [
        ((0, 0, 1), (2 * log(1 + exp(-1)) + log(1 + 2 * exp(-1))) / 3),
        ((0, 1, 2), (2 * log(2 + exp(-1)) + log(1 + 2 * exp(-1))) / 3),
        ((0, 0, 0), 0.0),
    ],
    ids=["same-group-excluded", "all-negatives", "diagonal-only"],
)
def test_should_apply_the_group_mask_to_off_diagonal_negatives_and_keep_diagonal_positives(
    torch_backend, codes, expected
):
    torch = torch_backend
    head = torch.nn.Identity()
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    scale = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    loss, logits, labels = adapter.clm_head_loss(
        head, head, scale, features, features, torch.tensor(codes, dtype=torch.int64)
    )
    assert loss.item() == pytest.approx(expected, abs=1e-7)
    assert logits.tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert labels.tolist() == [0, 1, 2]


def test_should_use_native_exponential_scale_cap_with_zero_gradient_above_it(torch_backend):
    torch = torch_backend
    head = torch.nn.Identity()
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    scale = torch.nn.Parameter(torch.tensor(log(1000), dtype=torch.float32))
    loss, logits, _ = adapter.clm_head_loss(
        head, head, scale, features, features, torch.tensor([0, 1], dtype=torch.int64)
    )
    assert logits.tolist() == [[100.0, 0.0], [0.0, 100.0]]
    loss.backward()
    assert scale.grad is not None and scale.grad.item() == 0.0


@pytest.fixture(scope="module")
def trajectory(torch_backend, reference_functions, seeded_parts):
    torch, (_, native_loss) = torch_backend, reference_functions
    source = deepcopy(seeded_parts)
    source_weights = [deepcopy(head.state_dict()) for head in source[:2]]
    source_scale = source[2].detach().clone()
    owner = make_owner(source)
    reference = deepcopy(source)
    parameters = [*reference[0].parameters(), *reference[1].parameters(), reference[2]]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=4, pct_start=0.1, anneal_strategy="cos"
    )
    batch, targets = paired_batch(torch)
    codes = torch.tensor(targets.group_codes, dtype=torch.int64)
    snapshots = []
    caller_rng = torch.get_rng_state().clone()
    for update in range(1, 5):
        expected_loss, _, _ = native_loss(*reference, batch.states, batch.actions, codes)
        optimizer.zero_grad(set_to_none=True)
        expected_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        diagnostic = owner.train_batch(batch, targets)
        snapshot = owner.snapshot_state()
        assert diagnostic.definition == adapter.OBJECTIVE
        assert diagnostic.value == expected_loss.item()
        require_equal_tree(torch, snapshot.state_head, reference[0].state_dict())
        require_equal_tree(torch, snapshot.action_head, reference[1].state_dict())
        require_equal_tree(torch, snapshot.logit_scale, reference[2].detach())
        require_equal_tree(
            torch, snapshot.gradients, tuple(parameter.grad for parameter in parameters)
        )
        require_equal_tree(torch, snapshot.optimizer_state, optimizer.state_dict())
        require_equal_tree(torch, snapshot.scheduler_state, scheduler.state_dict())
        assert snapshot.completed_updates == update and snapshot.training_examples == 3 * update
        assert torch.equal(caller_rng, torch.get_rng_state())
        snapshots.append(snapshot)
    for original, expected in zip(source[:2], source_weights, strict=True):
        require_equal_tree(torch, original.state_dict(), expected)
        assert all(parameter.grad is None for parameter in original.parameters())
    assert torch.equal(source[2].detach(), source_scale) and source[2].grad is None
    return owner, reference, snapshots, batch, targets


def test_should_continue_the_exact_clipped_adamw_and_onecycle_state_over_four_updates(
    torch_backend, trajectory
):
    torch = torch_backend
    owner, _, snapshots, _, _ = trajectory
    assert owner.completed_updates == owner.generation == 4 and owner.training_examples == 12
    for name in (
        "train_attempts",
        "loss_forward_attempts",
        "loss_forward_returns",
        "backward_attempts",
        "backward_returns",
        "optimizer_attempts",
        "optimizer_returns",
        "scheduler_attempts",
        "scheduler_returns",
        "commits",
    ):
        assert owner.work[name] == 4
    assert (
        owner.work["training_rows_attempted"] == 12
        and owner.work["unreturned_work_unknown"] is False
    )
    assert [snapshot.scheduler_state["last_epoch"] for snapshot in snapshots] == [1, 2, 3, 4]
    assert all(snapshot.modes == snapshots[0].modes for snapshot in snapshots)
    assert snapshots[-1].modes[0][0] is False and snapshots[-1].modes[1][0] is True
    assert all(torch.isfinite(gradient).all().item() for gradient in snapshots[-1].gradients)
    detached_work = owner.work
    detached_work["commits"] = 0
    assert owner.work["commits"] == 4


def test_should_recompute_fixed_dimension_probabilities_without_owning_the_callers_output(
    torch_backend, reference_functions, trajectory
):
    torch, (_, native_loss) = torch_backend, reference_functions
    owner, reference, _, batch, targets = trajectory
    before = owner.snapshot_state()
    features = (batch.states.clone(), batch.actions.clone())
    caller_rng = torch.get_rng_state().clone()
    counts = {"heads": 0}

    def count_head(_module, _arguments, _output):
        counts["heads"] += 1

    handles = [head.register_forward_hook(count_head) for head in owner._bundle[:2]]
    try:
        first = owner.predict(batch)
        expected = native_loss(
            *reference,
            batch.states,
            batch.actions,
            torch.tensor(targets.group_codes, dtype=torch.int64),
        )[1].softmax(dim=-1)
        assert first.shape == (3, 3) and first.dtype == torch.float32 and first.device.type == "cpu"
        assert torch.allclose(first, expected, atol=1e-7, rtol=0)
        first[0, 0] = -100.0
        second = owner.predict(batch)
        assert torch.allclose(second, expected, atol=1e-7, rtol=0)
        assert counts["heads"] == 4
    finally:
        for handle in handles:
            handle.remove()
    after = owner.snapshot_state()
    require_equal_tree(torch, vars(before), vars(after))
    assert torch.equal(features[0], batch.states) and torch.equal(features[1], batch.actions)
    assert torch.equal(caller_rng, torch.get_rng_state())
    assert owner.completed_updates == 4 and owner.training_examples == 12
    assert owner.work["prediction_attempts"] == owner.work["prediction_returns"] == 2


def test_should_restore_nonempty_optimizer_gradients_rng_and_modes_without_renewing_allowances(
    torch_backend, seeded_parts
):
    torch = torch_backend
    owner = make_owner(seeded_parts, maximum=2, restores=1)
    batch, targets = paired_batch(torch)
    owner.train_batch(batch, targets)
    snapshot = owner.snapshot_state()
    assert snapshot.optimizer_state["state"] and all(
        gradient is not None for gradient in snapshot.gradients
    )
    owner.train_batch(batch, targets)
    assert owner.completed_updates == 2 and owner.generation == 2
    caller_rng = torch.get_rng_state().clone()
    # A private generator seed changes state without drawing random values.
    owner._rng = torch.Generator(device="cpu").manual_seed(99).get_state()
    owner.restore_state(snapshot)
    restored = owner.snapshot_state()
    require_equal_tree(torch, vars(restored), vars(snapshot))
    assert owner.completed_updates == 1 and owner.training_examples == 3 and owner.generation == 3
    assert owner.work["train_attempts"] == owner.work["commits"] == 2
    assert owner.work["restore_attempts"] == owner.work["restore_returns"] == 1
    assert torch.equal(caller_rng, torch.get_rng_state())
    object_state = next(iter(snapshot.state_head.values()))
    with torch.no_grad():
        object_state.add_(1.0)
        snapshot.gradients[0].add_(1.0)
        next(iter(snapshot.optimizer_state["state"].values()))["exp_avg"].add_(1.0)
        snapshot.cpu_rng.zero_()
    require_equal_tree(torch, vars(restored), vars(owner.snapshot_state()))
    with pytest.raises(ValueError):
        owner.train_batch(batch, targets)
    with pytest.raises(ValueError):
        owner.restore_state(restored)
    assert owner.work["train_attempts"] == 2 and owner.work["restore_attempts"] == 1


@pytest.mark.parametrize(
    "fault", ["moment-shape", "scheduler", "matching-learning-rate", "beta1", "nonfinite-weight"]
)
def test_should_refuse_corrupt_continuation_snapshots_without_publishing_them(
    torch_backend, seeded_parts, fault
):
    torch = torch_backend
    owner = make_owner(seeded_parts)
    batch, targets = paired_batch(torch)
    owner.train_batch(batch, targets)
    snapshot = owner.snapshot_state()
    before = deepcopy(owner._capture())
    if fault == "moment-shape":
        moment = next(iter(snapshot.optimizer_state["state"].values()))
        moment["exp_avg"] = torch.zeros((1,), dtype=torch.float32)
    elif fault == "scheduler":
        snapshot.scheduler_state["total_steps"] += 1
    elif fault == "matching-learning-rate":
        snapshot.optimizer_state["param_groups"][0]["lr"] = 0.123
        snapshot.scheduler_state["_last_lr"] = [0.123]
    elif fault == "beta1":
        group = snapshot.optimizer_state["param_groups"][0]
        group["betas"] = (0.999, group["betas"][1])
    else:
        next(iter(snapshot.state_head.values())).view(-1)[0] = float("nan")
    with pytest.raises(ValueError):
        owner.restore_state(snapshot)
    require_equal_tree(torch, vars(before), vars(owner._capture()))
    assert owner.completed_updates == 1 and owner.work["restore_returns"] == 0
    with pytest.raises(ValueError):
        owner.train_batch(batch, targets)
    assert owner.work["train_attempts"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "probe-role",
        "identity",
        "dtype",
        "rows",
        "feature-width",
        "nonfinite",
        "target-count",
        "missing-target",
        "boolean-code",
    ],
)
def test_should_refuse_invalid_feature_identity_role_or_payload_before_paying_a_loss(
    torch_backend, seeded_parts, fault
):
    torch = torch_backend
    owner = make_owner(seeded_parts)
    batch, targets = paired_batch(torch)
    if fault == "role":
        object.__setattr__(batch, "role", "final_test")
    elif fault == "probe-role":
        object.__setattr__(batch, "role", "probe_train")
    elif fault == "identity":
        batch = replace(
            batch, identity=replace(batch.identity, tokenizer_revision="foreign-tokenizer")
        )
    elif fault == "dtype":
        batch = replace(batch, states=batch.states.to(torch.float64))
    elif fault == "rows":
        batch = replace(batch, states=batch.states[:1], actions=batch.actions[:1])
    elif fault == "feature-width":
        batch = replace(batch, states=batch.states[:, :1], actions=batch.actions[:, :1])
    elif fault == "nonfinite":
        batch.states[0, 0] = float("inf")
    elif fault == "target-count":
        targets = CLMTargets((0, 1))
    elif fault == "missing-target":
        targets = None
    else:
        object.__setattr__(targets, "group_codes", (0, True, 1))
    with pytest.raises(ValueError):
        owner.train_batch(batch, targets)
    assert (
        owner.work["train_attempts"]
        == owner.work["loss_forward_attempts"]
        == owner.completed_updates
        == 0
    )


@pytest.mark.parametrize(
    "fault", ["projection", "input-width", "shared-owner", "shared-storage", "parameter-cap"]
)
def test_should_bind_head_geometry_and_independent_parameter_storage_at_admission(
    torch_backend, seeded_parts, fault
):
    torch = torch_backend
    state, action, scale = deepcopy(seeded_parts)
    declaration, policy = identity(), CLMHeadPolicy(2, 4, max_parameters=4096)
    if fault == "projection":
        policy = replace(policy, projection_dim=3)
    elif fault == "input-width":
        declaration = replace(declaration, embedding_dim=3)
    elif fault == "shared-owner":
        action = state
    elif fault == "shared-storage":
        action.inp.weight = torch.nn.Parameter(state.inp.weight.detach())
    else:
        policy = replace(policy, max_parameters=34)
    with pytest.raises(ValueError):
        adapter.CLMHeadLearner(state, action, scale, identity=declaration, policy=policy)


@pytest.mark.parametrize("fault", ["loss-unreturned", "source-write", "optimizer-unreturned"])
def test_should_retain_paid_failure_costs_poison_the_owner_and_avoid_a_partial_update(
    torch_backend, seeded_parts, monkeypatch, fault
):
    torch = torch_backend
    owner = make_owner(seeded_parts)
    batch, targets = paired_batch(torch)
    before = deepcopy(owner._capture())
    native_loss = adapter.clm_head_loss
    optimizer_step = torch.optim.AdamW.step

    def fail_loss(*args):
        if fault == "loss-unreturned":
            raise RuntimeError("native loss did not return")
        result = native_loss(*args)
        with torch.no_grad():
            next(owner._bundle[0].parameters()).add_(0.25)
        return result

    def fail_optimizer(optimizer, *args, **kwargs):
        optimizer_step(optimizer, *args, **kwargs)
        raise RuntimeError("native optimizer paid return was lost")

    if fault == "optimizer-unreturned":
        monkeypatch.setattr(torch.optim.AdamW, "step", fail_optimizer)
    else:
        monkeypatch.setattr(adapter, "clm_head_loss", fail_loss)
    with pytest.raises(TrainingUpdateFailed) as failure:
        owner.train_batch(batch, targets)
    assert owner.work["train_attempts"] == owner.work["loss_forward_attempts"] == 1
    assert owner.work["commits"] == owner.completed_updates == 0
    assert owner.work["unreturned_work_unknown"] is True
    assert failure.value.work == owner.work
    if fault == "loss-unreturned":
        assert owner.work["loss_forward_returns"] == owner.work["backward_attempts"] == 0
    else:
        assert (
            owner.work["loss_forward_returns"]
            == owner.work["backward_returns"]
            == owner.work["optimizer_attempts"]
            == 1
        )
    if fault != "source-write":
        require_equal_tree(torch, vars(before), vars(owner._capture()))
    else:
        assert not torch.equal(
            before.state_head["inp.weight"], owner._capture().state_head["inp.weight"]
        )
    with pytest.raises(ValueError):
        owner.train_batch(batch, targets)
    assert owner.work["train_attempts"] == 1
