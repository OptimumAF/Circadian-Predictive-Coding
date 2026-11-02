"""Bounded cold/warm BP update fixtures; no genuine source12/F-return case."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_probe as ce
from src.adapters import torch_feature_learner as native
from src.adapters.torch_decision import _same_values
from src.shared.torch_runtime import require_torch


def forbidden(*args, **kwargs):
    raise AssertionError("unbudgeted work")


@pytest.mark.parametrize("value", [None, 0, 1, "output"])
def test_output_option_requires_bool_before_source(value, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    with pytest.raises(ValueError, match="output-only"):
        entry.run_example(probe_output_only=value)


@pytest.mark.parametrize(
    "change",
    [
        dict(context_return=False),
        dict(reserved_probe=False),
        dict(method="pc"),
        dict(method="neutral_cpc"),
        dict(method="gated_cpc"),
        dict(method="linear_fit"),
    ],
)
def test_output_option_refuses_other_modes_before_source(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    args = dict(method="bp", context_return=True, reserved_probe=True, probe_output_only=True)
    args.update(change)
    with pytest.raises(ValueError, match="output-only"):
        entry.run_example(**args)


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc", "perceptron"])
def test_native_port_refuses_family_before_building(method):
    fake = SimpleNamespace(policy=SimpleNamespace(method=method), _train_batch=forbidden)
    with pytest.raises(ValueError, match="require BP"):
        native.TorchFeatureLearner.train_batch_output_only(
            cast(Any, fake), cast(Any, None), cast(Any, None)
        )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_cli_forwarding_and_failure_scope(enabled, initialized, tmp_path, monkeypatch):
    seen: list[Any] = []
    written: list[Any] = []
    from scripts import context_catalog_learning as writer

    def run(**kwargs):
        seen.append(kwargs)
        return dict(
            complete=False,
            reserved_probe=dict(
                parameter_update_scope="foreign",
                preserved_metadata=dict(parameter_update_scope="output_only"),
            ),
        )

    def write(*a, **kwargs):
        written.append(kwargs)
        return False

    monkeypatch.setattr(entry, "run_example", run)
    monkeypatch.setattr(writer, "write_catalog_report", write)
    args = [
        "--context-return",
        "--method",
        "bp",
        "--reserved-probe",
        "--output",
        str(tmp_path / "never-written.json"),
    ]
    if enabled:
        args.append("--probe-output-only")
    if initialized:
        args.extend(("--probe-initialization", "ce"))
    assert entry.main(args) == 1
    assert seen[0].get("probe_output_only", False) is enabled
    assert ("probe_parameter_update_scope" in written[0]["observed_work"]) is enabled
    if enabled:
        assert written[0]["observed_work"]["probe_parameter_update_scope"] == "output_only"


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--method", "pc", "--context-return", "--reserved-probe"],
        ["--method", "bp", "--reserved-probe"],
    ],
)
def test_bad_cli_refuses_before_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    with pytest.raises(SystemExit):
        entry.main([*flags, "--probe-output-only", "--output", str(tmp_path / "unused.json")])


@pytest.mark.parametrize("value", [None, 1])
def test_ce_rejects_bad_output_flag_before_fit(value, monkeypatch):
    monkeypatch.setattr(ce, "prepare_ce_initialization", forbidden)
    with pytest.raises(ValueError, match="output-only"):
        ce.run_initialized_probe(
            None,
            None,
            None,
            {},
            None,
            check_boundary=lambda: None,
            memory=None,
            row={},
            initialization_row={},
            probe_output_only=value,
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_ce_forwards_private_output_scope(enabled, monkeypatch):
    seen = []

    def prepare(*a, **kw):
        kw["row"].update(complete=True)
        return object(), lambda: None

    def private(*a, **kw):
        seen.append(kw)
        return kw["row"]

    monkeypatch.setattr(ce, "prepare_ce_initialization", prepare)
    monkeypatch.setattr(ce, "run_owned_probe", private)
    ce.run_initialized_probe(
        None,
        None,
        None,
        {},
        None,
        check_boundary=lambda: None,
        memory=None,
        row={},
        initialization_row={},
        probe_output_only=enabled,
        probe_update_diagnostics=True,
    )
    assert seen[0].get("probe_output_only", False) is enabled
    assert seen[0]["probe_update_diagnostics"] is True


@pytest.fixture(scope="module")
def warm_pair():
    torch = require_torch()
    rng = torch.get_rng_state().clone()
    policy = native.TorchFeaturePolicy("bp", 2, 2, 17, num_classes=3)
    source = native.TorchFeatureLearner(policy)
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [2.0]])
    source.train_batch(x, y)
    source_pin = source.snapshot_state()
    control, output = source.fork(), source.fork()
    for _ in range(2):
        before = output.snapshot_state()
        control.restore_state(before)
        control_loss = control.train_batch(x, y)
        output_loss = output.train_batch_output_only(x, y)
        after, full = output.snapshot_state(), control.snapshot_state()
        assert after.gradients is not None and full.gradients is not None
        assert before.optimizer_state is not None and after.optimizer_state is not None
        assert full.optimizer_state is not None
        assert control_loss == output_loss
        changed = False
        for i, name in enumerate(native._PARAMETERS):
            assert _same_values(after.gradients[name], full.gradients[name])
            if i < 2:
                assert _same_values(before.head_state[name], after.head_state[name])
                assert _same_values(
                    before.optimizer_state["state"][i], after.optimizer_state["state"][i]
                )
            else:
                assert _same_values(full.head_state[name], after.head_state[name])
                assert _same_values(
                    full.optimizer_state["state"][i], after.optimizer_state["state"][i]
                )
                changed = changed or not _same_values(
                    before.head_state[name], after.head_state[name]
                )
        assert changed
        assert after.completed_updates == before.completed_updates + 1
        assert after.training_examples == before.training_examples + 3
    assert _same_values(source_pin, source.snapshot_state())
    assert torch.equal(rng, torch.get_rng_state())
    return output, x, y


def test_warm_hidden_and_momentum_stay_bitwise_fixed(warm_pair):
    assert warm_pair[0].completed_updates == 3


@pytest.mark.parametrize("momentum", [0.9, 0.0])
def test_cold_output_update_keeps_strict_snapshot_format(momentum):
    torch = require_torch()
    rng = torch.get_rng_state().clone()
    learner = native.TorchFeatureLearner(
        native.TorchFeaturePolicy("bp", 2, 2, 17, num_classes=3, momentum=momentum)
    )
    before = learner.snapshot_state()
    learner.train_batch_output_only(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), np.array([[0.0], [1.0], [2.0]])
    )
    after = learner.snapshot_state()
    assert after.gradients is not None and after.optimizer_state is not None
    native._snapshot(after)
    for i, name in enumerate(native._PARAMETERS[:2]):
        assert _same_values(before.head_state[name], after.head_state[name])
        assert after.gradients[name] is not None
        if momentum:
            assert torch.count_nonzero(after.optimizer_state["state"][i]["momentum_buffer"]) == 0
    if not momentum:
        assert after.optimizer_state["state"] == {}
    assert torch.equal(rng, torch.get_rng_state())


def test_post_step_failure_never_publishes_staged_parameters(warm_pair, monkeypatch):
    torch = require_torch()
    learner = warm_pair[0].fork()
    before = deepcopy(learner.snapshot_state())
    owner = learner._payload
    original = torch.optim.SGD.step

    def fail_after_step(*a, **kw):
        original(*a, **kw)
        raise RuntimeError("injected after staged step")

    monkeypatch.setattr(torch.optim.SGD, "step", fail_after_step)
    with pytest.raises(RuntimeError, match="injected"):
        learner.train_batch_output_only(warm_pair[1], warm_pair[2])
    assert learner._payload is owner
    assert _same_values(before, learner.snapshot_state())
