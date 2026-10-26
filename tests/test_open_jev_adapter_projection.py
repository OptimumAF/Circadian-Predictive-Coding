"""Literal PEFT-shaped projection fixtures; no released model or initializer."""

from copy import deepcopy
import pytest

from src.adapters import open_jev_projection as projection
from src.adapters.open_jev_training import OpenJevTrainingLearner, OpenJevTrainingPolicy
from src.core.learner_ports import TrainingUpdateFailed
from tests.test_open_jev_training import request


@pytest.fixture(scope="module")
def torch_backend():
    return pytest.importorskip("torch", reason="projection backend unavailable; not a Torch pass")


def literal_linear(torch, values):
    # Why this: a real Linear interface is required, but initialization must make
    # no random draws. Every stored coefficient is declared in this fixture.
    layer = torch.nn.Linear.__new__(torch.nn.Linear)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    layer.bias = None
    layer.out_features, layer.in_features = layer.weight.shape
    return layer


def projection_owner(torch, *, enabled=True, reset=1, forks=0, maximum=3):
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
            self.use_dora = {"default": False}
            self.use_rslora = {"default": False}
            self.lora_bias = {"default": False}
            self.lora_variant = {}
            self.lora_dropout = torch.nn.ModuleDict({"default": torch.nn.Identity()})
            self.lora_embedding_A = torch.nn.ParameterDict()
            self.lora_embedding_B = torch.nn.ParameterDict()
            self.lora_magnitude_vector = torch.nn.ModuleDict()

        def forward(self, value):
            return self.base_layer(value) + self.scaling["default"] * self.lora_B["default"](
                self.lora_A["default"](value)
            )

    class Backbone(Module):
        def __init__(self):
            super().__init__()
            self.projection = Projection()
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
            return [self.head(self.backbone.projection(features)).squeeze(-1)]

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
            max_updates=maximum,
            expected_lora_rank=2,
            max_optimizer_resets=reset,
            max_forks=forks,
            max_adapter_projections=int(enabled),
        ),
    )
    return owner, model, model.backbone.projection


def parameter_owners(model):
    return {name: id(value) for name, value in model.named_parameters()}


def operator_value(module):
    return (
        module.scaling["default"]
        * module.lora_B["default"].weight.detach()
        @ module.lora_A["default"].weight.detach()
    )


@pytest.fixture(scope="module")
def projected(torch_backend):
    torch = torch_backend
    owner, model, module = projection_owner(torch)
    owner.train_batch(request(), "c")
    before = owner.snapshot_state()
    values = {name: value.detach().clone() for name, value in model.named_parameters()}
    owners = parameter_owners(model)
    modes = [(name, value.training) for name, value in model.named_modules()]
    caller_rng = torch.get_rng_state().clone()
    original_operator = operator_value(module).clone()
    receipt = owner.project_adapters(1)
    after = owner.snapshot_state()
    projection_work = owner.adapter_projection_work
    reset_work = owner.optimizer_reset_work
    projected_operator = operator_value(module).clone()
    owner.train_batch(request(), "c")
    return locals()


def test_should_project_live_scaled_operator_and_preserve_base_head_rng_modes_and_parameter_owners(
    projected,
):
    state = projected
    torch, owner, model, module = (state[key] for key in ("torch", "owner", "model", "module"))
    before, after = state["before"], state["after"]
    singular = torch.linalg.svdvals(state["original_operator"].double())
    error = torch.linalg.vector_norm(
        (state["original_operator"] - state["projected_operator"]).double()
    )
    assert error.item() == pytest.approx(singular[1].item(), rel=2e-5, abs=1e-6)
    assert parameter_owners(model) == state["owners"]
    assert torch.equal(after.weights["head.weight"], before.weights["head.weight"])
    assert torch.equal(
        module.base_layer.weight, state["values"]["backbone.projection.base_layer.weight"]
    )
    assert torch.equal(model.backbone.running, torch.tensor([0.25]))
    assert torch.equal(after.gradients["head.weight"], before.gradients["head.weight"])
    assert torch.equal(after.rng, before.rng)
    assert torch.equal(torch.get_rng_state(), state["caller_rng"])
    assert [(name, value.training) for name, value in model.named_modules()] == state["modes"]
    assert module.r == {"default": 2} and module.scaling == {"default": 2.0}
    assert after.optimizer["state"] == {}
    assert state["reset_work"]["optimizer_resets_completed"] == 1
    assert state["receipt"] == owner.last_adapter_projection


def test_should_preserve_dormant_a_rows_and_allow_next_native_gradient_and_fresh_adamw_step(
    projected,
):
    state = projected
    torch, owner, module = (state[key] for key in ("torch", "owner", "module"))
    a_name = "backbone.projection.lora_A.default.weight"
    b_name = "backbone.projection.lora_B.default.weight"
    assert torch.equal(state["after"].weights[a_name][1], state["before"].weights[a_name][1])
    assert torch.count_nonzero(state["after"].weights[b_name][:, 1]).item() == 0
    assert state["after"].gradients[a_name] is state["after"].gradients[b_name] is None
    assert torch.count_nonzero(module.lora_B["default"].weight.grad[:, 1]).item() > 0
    assert all(value["step"].item() == 1 for value in owner._optimizer.state.values())
    assert owner.work["committed_updates"] == 2
    assert owner.adapter_projection_work == projected["projection_work"]


def test_should_retain_bounded_historical_receipt_and_charge_actual_projection_coordinates(
    projected,
):
    owner = projected["owner"]
    receipt = projected["receipt"]
    work = projected["projection_work"]
    assert receipt["definition"] == "open_jev_owned_adapter_projection_v1"
    assert receipt["effective_rank"] == 1
    assert receipt["inactive_rows"] == "original_A_zero_B_regrowth_allowed"
    assert len(receipt["modules"]) == 1
    module = receipt["modules"][0]
    assert module["module"] == "backbone.projection"
    assert module["allocated_rank"] == 2 and module["effective_rank"] == 1
    assert module["scaling"] == 2.0
    assert module["discarded_frobenius"] > 0
    assert len(module["singular_values"]) == 2
    assert receipt["native_work"]["committed_updates"] == 1
    assert owner.work["committed_updates"] == 2
    assert work == {
        "projection_attempts": 1,
        "input_copy_elements_attempted": 8,
        "core_calls_attempted": 1,
        "qr_calls_reserved": 2,
        "svd_calls_reserved": 1,
        "matmul_scalar_products_reserved": 16,
        "workspace_elements_reserved": 208,
        "staging_elements_attempted": 8,
        "cast_elements_attempted": 4,
        "parameter_write_elements_attempted": 8,
        "gradient_buffers_cleared": 2,
        "rng_copy_elements_attempted": 3 * owner._rng.numel(),
        "reset_rng_copy_elements_reserved": 4 * owner._rng.numel(),
        "projections_completed": 1,
    }
    copied = owner.adapter_projection_work
    copied[next(iter(copied))] = -1
    assert owner.adapter_projection_work == projected["projection_work"]


def test_should_keep_full_allocated_rank_projection_exact_without_random_reinitialization(
    torch_backend,
):
    torch = torch_backend
    owner, model, module = projection_owner(torch)
    before = operator_value(module).clone()
    caller_rng = torch.get_rng_state().clone()
    owners = parameter_owners(model)
    returned = owner.project_adapters(2)
    assert torch.allclose(operator_value(module), before, atol=1e-6, rtol=1e-6)
    assert parameter_owners(model) == owners and torch.equal(torch.get_rng_state(), caller_rng)
    assert owner.work["training_attempts"] == 0
    receipt = deepcopy(owner.last_adapter_projection)
    returned["modules"][0]["effective_rank"] = -1
    assert owner.last_adapter_projection == receipt


def test_should_support_zero_effective_rank_while_retaining_original_dormant_a(torch_backend):
    torch = torch_backend
    owner, _, module = projection_owner(torch)
    a = module.lora_A["default"].weight.detach().clone()
    owner.project_adapters(0)
    assert torch.equal(module.lora_A["default"].weight, a)
    assert torch.count_nonzero(module.lora_B["default"].weight).item() == 0
    owner.train_batch(request(), "c")
    assert torch.count_nonzero(module.lora_B["default"].weight.grad).item() > 0


@pytest.fixture(scope="module")
def metadata_owner(torch_backend):
    return projection_owner(torch_backend)


@pytest.mark.parametrize(
    "field,value",
    [
        ("fan_in_fan_out", True),
        ("active_adapters", []),
        ("_disable_adapters", True),
        ("merged_adapters", ["default"]),
        ("scaling", {"default": 3.0}),
        ("r", {"default": 1}),
        ("use_dora", {"default": True}),
        ("use_rslora", {"default": True}),
        ("lora_variant", {"default": object()}),
        ("lora_bias", {"default": True}),
    ],
)
def test_should_refuse_late_projection_metadata_changes_before_paid_work(
    metadata_owner, field, value
):
    owner, _, module = metadata_owner
    birth = getattr(module, field)
    work = owner.adapter_projection_work
    setattr(module, field, value)
    try:
        with pytest.raises(ValueError):
            owner.project_adapters(1)
    finally:
        setattr(module, field, birth)
    assert owner.adapter_projection_work == work


@pytest.mark.parametrize("rank", [True, -1, 3, 1.0])
def test_should_refuse_non_native_or_unavailable_rank_before_paid_work(metadata_owner, rank):
    owner, _, _ = metadata_owner
    before = owner.adapter_projection_work
    with pytest.raises(ValueError):
        owner.project_adapters(rank)
    assert owner.adapter_projection_work == before


def test_should_refuse_live_parameter_alias_replacement_before_any_projection_copy(metadata_owner):
    owner, _, module = metadata_owner
    original = module.lora_B["default"].weight
    before = owner.adapter_projection_work
    module.lora_B["default"].weight = module.lora_A["default"].weight
    try:
        with pytest.raises(ValueError):
            owner.project_adapters(1)
    finally:
        module.lora_B["default"].weight = original
    assert owner.adapter_projection_work == before


def test_should_leave_default_disabled_owner_compatible_and_refuse_projection_before_work(
    torch_backend,
):
    owner, _, _ = projection_owner(torch_backend, enabled=False)
    before = owner.work
    with pytest.raises(ValueError):
        owner.project_adapters(1)
    assert owner.work == before and all(
        value == 0 for value in owner.adapter_projection_work.values()
    )


def test_should_require_an_unspent_authorized_optimizer_reset_even_for_empty_optimizer(
    torch_backend,
):
    owner, _, _ = projection_owner(torch_backend, reset=0)
    with pytest.raises(ValueError):
        owner.project_adapters(1)
    assert all(value == 0 for value in owner.adapter_projection_work.values())


def test_should_refuse_already_spent_reset_without_consuming_projection(torch_backend):
    owner, _, _ = projection_owner(torch_backend)
    owner.reset_optimizer_state()
    with pytest.raises(ValueError):
        owner.project_adapters(1)
    assert all(value == 0 for value in owner.adapter_projection_work.values())


def test_should_not_renew_projection_or_reset_allowance_through_restore_or_owned_fork(
    torch_backend,
):
    owner, _, _ = projection_owner(torch_backend, forks=1)
    before = owner.snapshot_state()
    owner.project_adapters(1)
    receipt = deepcopy(owner.last_adapter_projection)
    paid = owner.adapter_projection_work
    owner.restore_state(before)
    assert owner.adapter_projection_work == paid and owner.last_adapter_projection == receipt
    child = owner.fork()
    assert child.adapter_projection_work == paid and child.last_adapter_projection == receipt
    for candidate in (owner, child):
        with pytest.raises(ValueError):
            candidate.project_adapters(1)
        with pytest.raises(ValueError):
            candidate.reset_optimizer_state()
    assert owner.adapter_projection_work == paid


@pytest.mark.parametrize("value", [True, -1, 2])
def test_should_refuse_invalid_projection_policy_without_backend(value):
    with pytest.raises(ValueError):
        OpenJevTrainingPolicy(max_adapter_projections=value)


@pytest.mark.parametrize("fault", ["core", "cast", "reset_factory", "publication"])
def test_should_keep_paid_failure_prefix_and_poison_without_refunding_projection_or_reset(
    torch_backend, monkeypatch: pytest.MonkeyPatch, fault: str
):
    torch = torch_backend
    owner, model, _ = projection_owner(torch)
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    original_core = projection.compose_low_rank
    original_publish = projection._publish

    def fail_core(*args, **kwargs):
        raise ValueError("controlled core failure")

    def invalid_cast(*args, **kwargs):
        result = original_core(*args, **kwargs)
        result.a.fill(1e100)
        return result

    def fail_factory():
        raise RuntimeError("controlled reset factory failure")

    def fail_publication(candidate, staged):
        original_publish(candidate, staged)
        raise RuntimeError("controlled publication failure")

    if fault == "core":
        monkeypatch.setattr(projection, "compose_low_rank", fail_core)
    elif fault == "cast":
        monkeypatch.setattr(projection, "compose_low_rank", invalid_cast)
    elif fault == "reset_factory":
        monkeypatch.setattr(owner, "_new_optimizer", fail_factory)
    else:
        monkeypatch.setattr(projection, "_publish", fail_publication)
    with pytest.raises(TrainingUpdateFailed) as captured:
        owner.project_adapters(1)
    work = owner.adapter_projection_work
    assert work["projection_attempts"] == work["core_calls_attempted"] == 1
    assert work["input_copy_elements_attempted"] == 8
    assert work["workspace_elements_reserved"] == 208
    assert work["qr_calls_reserved"] == 2 and work["svd_calls_reserved"] == 1
    assert work["projections_completed"] == 0
    assert owner.last_adapter_projection is None
    assert owner.work["committed_updates"] == owner.work["training_attempts"] == 0
    assert captured.value.work["projection_attempts"] == 1
    assert captured.value.work["projections_completed"] == 0
    assert owner.optimizer_reset_work == {
        "optimizer_reset_attempts": int(fault in ("reset_factory", "publication")),
        "optimizer_resets_completed": int(fault == "publication"),
    }
    if fault == "core":
        assert work["staging_elements_attempted"] == work["cast_elements_attempted"] == 0
    elif fault == "cast":
        assert work["staging_elements_attempted"] == 8 and work["cast_elements_attempted"] == 2
    else:
        assert work["staging_elements_attempted"] == 8 and work["cast_elements_attempted"] == 4
    assert work["parameter_write_elements_attempted"] == (8 if fault == "publication" else 0)
    if fault != "publication":
        for name, value in model.named_parameters():
            assert torch.equal(value, weights[name])
    paid = deepcopy(work)
    with pytest.raises(ValueError):
        owner.project_adapters(1)
    with pytest.raises(ValueError):
        owner.train_batch(request(), "c")
    assert owner.adapter_projection_work == paid
