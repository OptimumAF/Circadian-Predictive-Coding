"""Literal two-state consolidation tests; no released model or random init."""

from copy import deepcopy
from dataclasses import replace

import pytest

from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.adapters import open_jev_consolidation as consolidation
from src.adapters import open_jev_projection as projection
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_open_jev_adapter_projection import literal_linear
from tests.test_open_jev_training import assert_tensor_tree_equal, request


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip(
        "torch", reason="consolidation backend unavailable; not a Torch pass"
    )


def consolidation_owner(torch, *, enabled=True, layers=1):
    from torch.nn import Module

    class Projection(Module):
        def __init__(self):
            super().__init__()
            self.base_layer = literal_linear(torch, [[1.0, 0.0], [0.0, 1.0]])
            self.lora_A = torch.nn.ModuleDict(
                {"default": literal_linear(torch, [[1.0, 0.0], [0.0, 1.0]])}
            )
            self.lora_B = torch.nn.ModuleDict(
                {"default": literal_linear(torch, [[3.0, 0.0], [0.0, 1.0]])}
            )
            self.r, self.lora_alpha, self.scaling = {"default": 2}, {"default": 4}, {"default": 2.0}
            self.fan_in_fan_out = False
            self.active_adapters = ["default"]
            self._disable_adapters = False
            self.merged_adapters = []
            self.use_dora, self.use_rslora, self.lora_bias = ({"default": False} for _ in range(3))
            self.lora_variant = {}
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
            self.lora_embedding_A, self.lora_embedding_B = (
                torch.nn.ParameterDict(),
                torch.nn.ParameterDict(),
            )
            self.lora_magnitude_vector = torch.nn.ModuleDict()

        def forward(self, value):
            return self.base_layer(value) + self.scaling["default"] * self.lora_B["default"](
                self.lora_A["default"](value)
            )

    class Backbone(Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Projection() for _ in range(layers)])
            self.register_buffer("running", torch.tensor([0.25], dtype=torch.float32))

        def gradient_checkpointing_enable(self):
            pass

        def enable_input_require_grads(self):
            pass

    class Model(Module):
        def __init__(self):
            super().__init__()
            self.lora_rank = 2
            self.backbone = Backbone()
            self.head = literal_linear(torch, [[0.2, -0.1]])

        def forward(self, records):
            table = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [-1.0, -1.0]}
            features = torch.tensor(
                [table[key] for key in records[0]["answer_keys"]], dtype=torch.float32
            )
            for layer in self.backbone.layers:
                features = layer(features)
            return [self.head(features).squeeze(-1)]

    model = Model()
    model.eval()

    def compile_request(state, questions):
        question = questions["decision"]
        return [
            {
                "kind": "choice",
                "state": state,
                "question": question["instructions"],
                "answer_keys": list(question["criteria"]),
                "options": list(question["criteria"].values()),
            }
        ]

    owner = OpenJevTrainingLearner(
        model,
        compile_request,
        instructions="Choose a literal class.",
        policy=OpenJevTrainingPolicy(
            max_updates=3,
            expected_lora_rank=2,
            max_optimizer_resets=1,
            max_forks=1,
            max_adapter_projections=1,
            max_adapter_consolidations=int(enabled),
        ),
    )
    return owner, model, model.backbone.layers[0]


def operator_value(module):
    return (
        module.scaling["default"]
        * module.lora_B["default"].weight.detach()
        @ module.lora_A["default"].weight.detach()
    )


def change_live_factors(torch, module):
    with torch.no_grad():
        module.lora_A["default"].weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 1.0]]))
        module.lora_B["default"].weight.copy_(torch.tensor([[0.5, 1.0], [2.0, -0.5]]))


def assert_no_tensors(torch, value):
    assert not torch.is_tensor(value)
    if isinstance(value, dict):
        for child in value.values():
            assert_no_tensors(torch, child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            assert_no_tensors(torch, child)


@pytest.fixture(scope="module")
def completed(torch_backend):
    torch = torch_backend
    owner, model, module = consolidation_owner(torch)
    old_a, old_b = (
        factor.weight.detach().clone()
        for factor in (module.lora_A["default"], module.lora_B["default"])
    )
    old_operator = operator_value(module).clone()
    before_capture = owner.snapshot_state()
    capture = owner.capture_adapter_reference()
    after_capture = owner.snapshot_state()
    captured = owner._adapter_reference
    change_live_factors(torch, module)
    owner.train_batch(request(), "c")
    before = owner.snapshot_state()
    current_a, current_b = (
        factor.weight.detach().clone()
        for factor in (module.lora_A["default"], module.lora_B["default"])
    )
    expected = 0.5 * (old_operator + operator_value(module))
    factor_average = 2.0 * (0.5 * (old_b + current_b)) @ (0.5 * (old_a + current_a))
    identities = {name: id(value) for name, value in model.named_parameters()}
    base = module.base_layer.weight.detach().clone()
    modes = [(name, value.training) for name, value in model.named_modules()]
    caller_rng = torch.get_rng_state().clone()
    receipt = owner.consolidate_adapters(2)
    after = owner.snapshot_state()
    actual = operator_value(module).clone()
    paid = owner.adapter_consolidation_work
    owner.train_batch(request(), "c")
    return locals()


def test_should_install_half_old_and_half_current_operators_without_factorwise_cross_terms(
    completed,
):
    state = completed
    torch = state["torch"]
    assert torch.allclose(state["actual"], state["expected"], atol=2e-6, rtol=2e-6)
    assert torch.linalg.vector_norm(state["expected"] - state["factor_average"]).item() > 0.5
    assert state["owner"]._adapter_reference is None
    assert state["receipt"]["effective_rank"] == 2


def test_should_preserve_current_head_base_rng_modes_and_owners_while_resetting_only_old_optimizer_coordinates(
    completed,
):
    state = completed
    torch, owner, model = (state[key] for key in ("torch", "owner", "model"))
    assert torch.equal(
        state["after"].weights["head.weight"], state["before"].weights["head.weight"]
    )
    assert torch.equal(
        state["after"].gradients["head.weight"], state["before"].gradients["head.weight"]
    )
    assert torch.equal(state["module"].base_layer.weight, state["base"])
    assert torch.equal(state["after"].rng, state["before"].rng)
    assert torch.equal(torch.get_rng_state(), state["caller_rng"])
    assert {name: id(value) for name, value in model.named_parameters()} == state["identities"]
    assert [(name, value.training) for name, value in model.named_modules()] == state["modes"]
    assert state["after"].optimizer["state"] == {}
    assert all(value["step"].item() == 1 for value in owner._optimizer.state.values())
    assert owner.work["committed_updates"] == 2
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": 1,
        "optimizer_resets_completed": 1,
    }


def test_should_keep_capture_metadata_tensor_free_detached_and_permanent(completed):
    state = completed
    owner = state["owner"]
    assert_no_tensors(state["torch"], state["capture"])
    assert_no_tensors(state["torch"], state["receipt"])
    assert state["before_capture"].optimizer["state"] == {}
    for name in ("weights", "gradients", "optimizer", "rng"):
        assert_tensor_tree_equal(
            state["torch"],
            getattr(state["before_capture"], name),
            getattr(state["after_capture"], name),
        )
    assert all(value == 0 for value in state["capture"]["source_native_work"].values())
    work = owner.adapter_consolidation_work
    assert work["reference_attempts"] == work["references_completed"] == 1
    assert work["reference_copy_elements_attempted"] == 8
    assert work["reference_rng_read_elements_attempted"] == 2 * owner._rng.numel()
    assert work["consolidation_attempts"] == work["consolidations_completed"] == 1
    assert (
        owner.adapter_projection_work["projection_attempts"]
        == owner.adapter_projection_work["projections_completed"]
        == 1
    )
    detached = owner.adapter_consolidation_work
    detached["references_completed"] = 0
    assert owner.adapter_consolidation_work == work == state["paid"]
    assert state["capture"]["definition"] == "open_jev_owned_adapter_reference_v1"
    assert state["capture"]["numeric_elements"] == 8 and state["capture"]["numeric_bytes"] == 32
    historical = owner.last_adapter_reference
    historical["modules"][0]["scaling"] = -1
    state["capture"]["modules"][0]["scaling"] = -2
    assert owner.last_adapter_reference["modules"][0]["scaling"] == 2.0
    assert state["receipt"]["definition"] == "open_jev_owned_adapter_consolidation_v1"
    assert state["receipt"]["coefficients"] == {"reference": 0.5, "current": 0.5}
    assert state["receipt"]["head_policy"] == "preserve_current"
    assert (
        state["receipt"]["composition"]
        == "complete_operators_current_already_contains_prior_history"
    )
    assert state["receipt"]["reference_native_work"]["committed_updates"] == 0
    assert state["receipt"]["native_work"]["committed_updates"] == 1
    assert owner.adapter_projection_work["input_copy_elements_attempted"] == 16
    assert owner.adapter_projection_work["workspace_elements_reserved"] == 608


def test_should_match_independent_dense_rank_one_tail_and_keep_current_dormant_a(torch_backend):
    torch = torch_backend
    owner, _, module = consolidation_owner(torch)
    old = operator_value(module).clone()
    owner.capture_adapter_reference()
    change_live_factors(torch, module)
    current_a = module.lora_A["default"].weight.detach().clone()
    expected = 0.5 * (old + operator_value(module))
    spectrum = torch.linalg.svdvals(expected.double())
    receipt = owner.consolidate_adapters(1)
    error = torch.linalg.vector_norm((expected - operator_value(module)).double())
    assert error.item() == pytest.approx(spectrum[1].item(), rel=2e-6, abs=2e-6)
    assert torch.equal(module.lora_A["default"].weight[1], current_a[1])
    assert torch.count_nonzero(module.lora_B["default"].weight[:, 1]).item() == 0
    assert receipt["modules"][0]["discarded_frobenius"] == pytest.approx(
        spectrum[1].item(), rel=1e-12
    )


@pytest.mark.parametrize(
    "options",
    [
        {"max_adapter_consolidations": True},
        {"max_adapter_consolidations": -1},
        {"max_adapter_consolidations": 2},
        {"max_adapter_consolidations": 1, "max_adapter_projections": 0, "max_optimizer_resets": 1},
        {"max_adapter_consolidations": 1, "max_adapter_projections": 1, "max_optimizer_resets": 0},
    ],
)
def test_should_refuse_invalid_or_unfunded_policy_without_a_tensor_backend(options):
    with pytest.raises(ValueError):
        OpenJevTrainingPolicy(**options)


def test_should_refuse_consolidation_without_a_captured_reference_before_work(torch_backend):
    owner, _, _ = consolidation_owner(torch_backend)
    before = owner.adapter_consolidation_work
    with pytest.raises(ValueError):
        owner.consolidate_adapters(1)
    assert owner.adapter_consolidation_work == before
    assert owner.adapter_projection_work["projection_attempts"] == 0


def test_should_refuse_infeasible_two_term_metadata_before_capturing_or_reserving_authority(
    torch_backend, monkeypatch
):
    owner, _, _ = consolidation_owner(torch_backend)
    calls = []

    def refuse(terms, rank, maximum):
        calls.append((len(terms), sum(term.a.shape[0] for term in terms), rank, maximum))
        raise ValueError("controlled two-term core allowance refusal")

    monkeypatch.setattr(projection, "_admit", refuse)
    with pytest.raises(ValueError):
        owner.capture_adapter_reference()
    assert calls == [(2, 4, 0, 20_000_000)]
    assert owner._adapter_reference is None
    assert all(value == 0 for value in owner.adapter_consolidation_work.values())
    assert all(value == 0 for value in owner.adapter_projection_work.values())
    assert owner.optimizer_reset_work == dict(
        optimizer_reset_attempts=0, optimizer_resets_completed=0
    )


@pytest.fixture(scope="module")
def captured_owner(torch_backend):
    owner, model, module = consolidation_owner(torch_backend)
    before = owner.snapshot_state()
    receipt = owner.capture_adapter_reference()
    return owner, model, module, before, receipt


@pytest.mark.parametrize("rank", [True, -1, 3, 1.0])
def test_should_refuse_invalid_rank_before_charging_or_consuming_reference(captured_owner, rank):
    owner, _, _, _, _ = captured_owner
    before, reference = owner.adapter_consolidation_work, owner._adapter_reference
    with pytest.raises(ValueError):
        owner.consolidate_adapters(rank)
    assert owner.adapter_consolidation_work == before and owner._adapter_reference is reference


@pytest.mark.parametrize(
    "field,value",
    [
        ("fan_in_fan_out", True),
        ("scaling", {"default": 3.0}),
        ("active_adapters", []),
        ("r", {"default": 1}),
    ],
)
def test_should_refuse_changed_live_metadata_before_historical_or_current_copies(
    captured_owner, field, value
):
    owner, _, module, _, _ = captured_owner
    original, before = getattr(module, field), owner.adapter_consolidation_work
    setattr(module, field, value)
    try:
        with pytest.raises(ValueError):
            owner.consolidate_adapters(1)
    finally:
        setattr(module, field, original)
    assert owner.adapter_consolidation_work == before


@pytest.mark.parametrize("fault", ["record", "tensor"])
def test_should_refuse_equal_value_reference_owner_replacement_before_paid_work(
    captured_owner, fault
):
    owner, _, _, _, _ = captured_owner
    reference = owner._adapter_reference
    original_tensor = reference.layers[0].a
    before = owner.adapter_consolidation_work
    if fault == "record":
        owner._adapter_reference = replace(reference)
    else:
        object.__setattr__(reference.layers[0], "a", original_tensor.clone())
    try:
        with pytest.raises(ValueError):
            owner.consolidate_adapters(1)
    finally:
        owner._adapter_reference = reference
        object.__setattr__(reference.layers[0], "a", original_tensor)
    assert owner.adapter_consolidation_work == before
    assert owner.adapter_projection_work["projection_attempts"] == 0


def test_should_refuse_normal_inplace_mutation_of_captured_factor_without_consuming_consolidation(
    torch_backend,
):
    torch = torch_backend
    owner, _, _ = consolidation_owner(torch)
    owner.capture_adapter_reference()
    reference = owner._adapter_reference
    with torch.no_grad():
        reference.layers[0].a.add_(0.25)
    before = owner.adapter_consolidation_work
    with pytest.raises(ValueError):
        owner.consolidate_adapters(1)
    assert owner.adapter_consolidation_work == before
    assert owner._adapter_reference is reference
    assert owner.adapter_projection_work["projection_attempts"] == 0


def test_should_refuse_live_alias_replacement_before_reference_or_current_copy(captured_owner):
    owner, _, module, _, _ = captured_owner
    original = module.lora_B["default"].weight
    before = owner.adapter_consolidation_work
    module.lora_B["default"].weight = module.lora_A["default"].weight
    try:
        with pytest.raises(ValueError):
            owner.consolidate_adapters(1)
    finally:
        module.lora_B["default"].weight = original
    assert owner.adapter_consolidation_work == before


def test_should_reserve_pending_reference_authority_against_standalone_reset_or_projection(
    captured_owner,
):
    owner, _, _, _, _ = captured_owner
    before = owner.adapter_projection_work, owner.optimizer_reset_work
    for operation in (lambda: owner.project_adapters(1), owner.reset_optimizer_state):
        with pytest.raises(ValueError):
            operation()
    assert (owner.adapter_projection_work, owner.optimizer_reset_work) == before


def test_should_keep_default_disabled_and_refuse_capture_and_consolidation_without_work(
    torch_backend,
):
    owner, _, _ = consolidation_owner(torch_backend, enabled=False)
    for method in (owner.capture_adapter_reference, lambda: owner.consolidate_adapters(1)):
        with pytest.raises(ValueError):
            method()
    assert all(value == 0 for value in owner.adapter_consolidation_work.values())


def test_should_preserve_pending_reference_on_restore_and_refuse_fork_then_inherit_spent_limits_after_consumption(
    torch_backend,
):
    owner, _, _ = consolidation_owner(torch_backend)
    birth = owner.snapshot_state()
    owner.capture_adapter_reference()
    pending = owner._adapter_reference
    paid = owner.adapter_consolidation_work
    owner.restore_state(birth)
    assert owner._adapter_reference is pending and owner.adapter_consolidation_work == paid
    before_fork = owner.fork_work
    with pytest.raises(ValueError):
        owner.fork()
    assert owner.fork_work == before_fork
    with pytest.raises(ValueError):
        owner.capture_adapter_reference()
    owner.consolidate_adapters(2)
    final_work = owner.adapter_consolidation_work
    child = owner.fork()
    assert child._adapter_reference is None and child.adapter_consolidation_work == final_work
    for candidate in (owner, child):
        with pytest.raises(ValueError):
            candidate.capture_adapter_reference()
        with pytest.raises(ValueError):
            candidate.consolidate_adapters(1)
    assert owner.adapter_consolidation_work == final_work


def test_should_charge_failed_second_reference_copy_and_poison_without_publishing_slot(
    torch_backend, monkeypatch
):
    owner, _, _ = consolidation_owner(torch_backend)
    original = consolidation._copy_factor
    calls: list[None] = []

    def fail_second(candidate, parameter):
        calls.append(None)
        if len(calls) == 2:
            candidate._adapter_consolidation_work["reference_copy_elements_attempted"] += (
                parameter.numel()
            )
            raise RuntimeError("controlled second reference clone failure")
        return original(candidate, parameter)

    monkeypatch.setattr(consolidation, "_copy_factor", fail_second)
    with pytest.raises(TrainingUpdateFailed) as captured:
        owner.capture_adapter_reference()
    work = owner.adapter_consolidation_work
    assert work["reference_attempts"] == 1 and work["references_completed"] == 0
    assert work["reference_copy_elements_attempted"] == 8
    assert work["reference_rng_read_elements_attempted"] == owner._rng.numel()
    assert work["consolidation_attempts"] == work["consolidations_completed"] == 0
    assert owner._adapter_reference is None and owner.last_adapter_reference is None
    assert captured.value.work["reference_copy_elements_attempted"] == 8
    with pytest.raises(ValueError):
        owner.capture_adapter_reference()
    assert owner.adapter_consolidation_work == work


@pytest.mark.parametrize("fault", ["core", "cast", "reset_factory", "publication"])
def test_should_preserve_paid_consolidation_failure_prefix_and_refuse_retry(
    torch_backend, monkeypatch, fault
):
    torch = torch_backend
    owner, model, _ = consolidation_owner(torch)
    owner.capture_adapter_reference()
    reference = owner._adapter_reference
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    original_core, original_publish = projection.compose_low_rank, projection._publish
    original_factory = owner._new_optimizer

    def fail_core(*args, **kwargs):
        raise ValueError("controlled consolidation core failure")

    def fail_cast(*args, **kwargs):
        result = original_core(*args, **kwargs)
        result.a.fill(1e100)
        return result

    def fail_factory():
        staged = original_factory()
        with torch.no_grad():
            reference.layers[0].a.add_(0.25)
        return staged

    def fail_publish(candidate, staged):
        original_publish(candidate, staged)
        raise RuntimeError("controlled postpublication failure")

    if fault == "core":
        monkeypatch.setattr(projection, "compose_low_rank", fail_core)
    elif fault == "cast":
        monkeypatch.setattr(projection, "compose_low_rank", fail_cast)
    elif fault == "reset_factory":
        monkeypatch.setattr(owner, "_new_optimizer", fail_factory)
    else:
        monkeypatch.setattr(projection, "_publish", fail_publish)
    with pytest.raises(TrainingUpdateFailed) as captured:
        owner.consolidate_adapters(1)
    work, numerical = owner.adapter_consolidation_work, owner.adapter_projection_work
    assert work["reference_attempts"] == work["references_completed"] == 1
    assert work["consolidation_attempts"] == 1 and work["consolidations_completed"] == 0
    assert numerical["input_copy_elements_attempted"] == 16
    assert numerical["core_calls_attempted"] == numerical["projection_attempts"] == 1
    assert numerical["workspace_elements_reserved"] == 592
    assert numerical["projections_completed"] == 0
    assert owner._adapter_reference is reference
    assert owner.last_adapter_projection is None
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": int(fault in ("reset_factory", "publication")),
        "optimizer_resets_completed": int(fault == "publication"),
    }
    assert captured.value.work["consolidation_attempts"] == 1
    assert captured.value.work["consolidations_completed"] == 0
    if fault != "publication":
        for name, value in model.named_parameters():
            assert torch.equal(value, weights[name])
    paid = deepcopy((work, numerical, owner.optimizer_reset_work))
    with pytest.raises(ValueError):
        owner.consolidate_adapters(1)
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    assert (
        owner.adapter_consolidation_work,
        owner.adapter_projection_work,
        owner.optimizer_reset_work,
    ) == paid


def test_should_stage_all_modules_before_any_live_publication_or_reset(torch_backend, monkeypatch):
    torch = torch_backend
    owner, model, _ = consolidation_owner(torch, layers=2)
    owner.capture_adapter_reference()
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    original = projection.compose_low_rank
    calls: list[None] = []

    def fail_second(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise ValueError("controlled second-layer consolidation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(projection, "compose_low_rank", fail_second)
    with pytest.raises(TrainingUpdateFailed):
        owner.consolidate_adapters(1)
    assert len(calls) == 2
    assert owner.adapter_consolidation_work["reference_copy_elements_attempted"] == 16
    assert owner.adapter_projection_work["input_copy_elements_attempted"] == 32
    assert owner.adapter_projection_work["core_calls_attempted"] == 2
    assert owner.adapter_projection_work["parameter_write_elements_attempted"] == 0
    assert owner.optimizer_reset_work["optimizer_reset_attempts"] == 0
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
