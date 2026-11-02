"""Small BP correctness fixtures; no source12 or full F/return experiment."""

from copy import deepcopy
from dataclasses import replace
from math import sqrt
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from scripts import recurring_ticket_workflow_example as entry
from scripts import ticket_ce_probe as ce
from scripts import ticket_native_probe as probe
from src.adapters import torch_bp_step_norms as norms
from src.adapters import torch_feature_learner as native
from src.adapters.torch_decision import _same_values
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.decision_features import DecisionFeatureSpec
from src.core.typed_decision import DecisionCandidate
from src.shared.torch_runtime import require_torch


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected model or data work")


@pytest.mark.parametrize("value", [None, 0, 1, "yes"])
def test_nonboolean_option_refused_before_models(value, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    with pytest.raises(ValueError, match="update diagnostics"):
        entry.run_example(probe_update_diagnostics=value)


@pytest.mark.parametrize(
    "change",
    [
        dict(reserved_probe=False),
        dict(context_return=False),
        dict(method="pc"),
        dict(method="linear_fit"),
        dict(method="neutral_cpc"),
    ],
)
def test_incompatible_mode_refused_before_models(change, monkeypatch):
    monkeypatch.setattr(entry, "prepare_training", forbidden)
    options = dict(
        method="bp", context_return=True, reserved_probe=True, probe_update_diagnostics=True
    )
    options.update(change)
    with pytest.raises(ValueError, match="update diagnostics"):
        entry.run_example(**options)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_cli_forwards_only_opt_in_and_preserves_paid_failure(
    enabled, initialized, tmp_path, monkeypatch
):
    seen, written = [], []
    report = dict(
        complete=False,
        reserved_probe=dict(preserved_update_diagnostics=dict(work=dict(summary_attempts=1))),
    )

    def run(**kw):
        seen.append(kw)
        return report

    monkeypatch.setattr(entry, "run_example", run)
    from scripts import context_catalog_learning as writer

    def write(*a, **kw):
        written.append(kw)
        return False

    monkeypatch.setattr(writer, "write_catalog_report", write)
    flags = [
        "--context-return",
        "--method",
        "bp",
        "--reserved-probe",
        "--output",
        str(tmp_path / "out.json"),
    ]
    if enabled:
        flags.append("--probe-update-diagnostics")
    if initialized:
        flags.extend(("--probe-initialization", "ce"))
    assert entry.main(flags) == 1
    assert seen[0].get("probe_update_diagnostics", False) is enabled
    assert ("probe_update_diagnostics" in written[0]["observed_work"]) is enabled


@pytest.mark.parametrize("flags", [[], ["--method", "pc", "--context-return", "--reserved-probe"]])
def test_bad_cli_refused_before_dispatch(flags, tmp_path, monkeypatch):
    monkeypatch.setattr(entry, "run_example", forbidden)
    with pytest.raises(SystemExit):
        entry.main([*flags, "--probe-update-diagnostics", "--output", str(tmp_path / "out.json")])


@pytest.mark.parametrize("enabled", [False, True])
def test_ce_handoff_forwards_only_opt_in(enabled, monkeypatch):
    seen = []
    row: dict[str, Any] = {}

    def prepare(*a, **kw):
        kw["row"].update(complete=True)
        return object(), lambda: None

    monkeypatch.setattr(ce, "prepare_ce_initialization", prepare)

    def private(*a, **kw):
        seen.append(kw)
        return kw["row"]

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
        initialization_row=row,
        probe_update_diagnostics=enabled,
    )
    assert seen[0].get("probe_update_diagnostics", False) is enabled


@pytest.fixture(scope="module")
def paired():
    torch = require_torch()
    before_rng = torch.get_rng_state().clone()
    policy = native.TorchFeaturePolicy("bp", 2, 2, 17, num_classes=3)
    observed = native.TorchFeatureLearner(policy)
    control = native.TorchFeatureLearner(policy)
    control.restore_state(observed.snapshot_state())
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [2.0]])
    records, snapshots = [], []
    for _ in range(2):
        before = observed.snapshot_state()
        control.train_batch(x, y)
        observed.train_batch(x, y)
        after = observed.snapshot_state()
        pin = deepcopy((before, after))
        records.append(norms.summarize_bp_update(before, after, rows=3))
        assert _same_values((before, after), pin)
        assert _same_values(after, control.snapshot_state())
        snapshots.append((before, after))
    assert torch.equal(before_rng, torch.get_rng_state())
    return observed, x, y, snapshots, records


def test_measured_updates_preserve_state_rng_and_observe_momentum(paired):
    records = paired[4]
    assert records[0]["groups"]["hidden"]["momentum_before_l2"] == 0
    assert records[1]["groups"]["hidden"]["momentum_before_l2"] > 0
    for result in records:
        assert result["rows"] == 3 and result["work"]["scalar_values_read"] == 75
        for group in result["groups"].values():
            assert group["gradient_l2"] > 0
            assert group["sgd_max_abs_residual"] < 1e-7


def test_literal_snapshot_norm_oracle_without_a_new_update(paired):
    before, after = deepcopy(paired[3][1])
    for index, name in enumerate(native._PARAMETERS):
        before.head_state[name].zero_()
        before.optimizer_state["state"][index]["momentum_buffer"].fill_(2)
        after.head_state[name].fill_(-0.084)
        after.gradients[name].fill_(1)
        after.optimizer_state["state"][index]["momentum_buffer"].fill_(2.8)
    result = norms.summarize_bp_update(before, after, rows=3)
    for name, count in (("hidden", 6), ("output", 9)):
        g = result["groups"][name]
        assert g["parameter_values"] == count
        assert g["gradient_l2"] == pytest.approx(sqrt(count))
        assert g["momentum_before_l2"] == pytest.approx(2 * sqrt(count))
        assert g["momentum_after_l2"] == pytest.approx(2.8 * sqrt(count))
        assert g["parameter_delta_l2"] == pytest.approx(0.084 * sqrt(count))


@pytest.mark.parametrize(
    "fault", ["rows_bool", "rows_zero", "rows_large", "clock", "gradient", "policy"]
)
def test_bad_snapshot_or_nonconsecutive_work_refused(paired, fault):
    before, after = deepcopy(paired[3][1])
    rows = {"rows_bool": True, "rows_zero": 0, "rows_large": 129}.get(fault, 3)
    if fault == "clock":
        object.__setattr__(after, "training_examples", after.training_examples + 1)
    if fault == "gradient":
        after.gradients["bias_hidden"].fill_(float("nan"))
    if fault == "policy":
        object.__setattr__(after, "policy", replace(after.policy, learning_rate=0.02))
    with pytest.raises(ValueError):
        norms.summarize_bp_update(before, after, rows=rows)


def test_tiny_private_prefix_default_corruption_and_summary_failure(paired, monkeypatch):
    source, x, y = paired[:3]
    state = source.snapshot_state()
    spec = DecisionFeatureSpec(("a", "b"), tuple(DecisionCandidate(k, k) for k in ("a", "b", "c")))
    monkeypatch.setattr(probe, "_require_probe_source", lambda *a, **kw: None)
    monkeypatch.setattr(probe, "build_head_cohorts", lambda *a, **kw: (None,) * 4)
    monkeypatch.setattr(probe, "_read_tasks", lambda *a, **kw: None)
    monkeypatch.setattr(
        ToyBudgetSession, "attach_memory", lambda s, m: setattr(s, "process_rss_sampler", m)
    )
    monkeypatch.setattr(ToyBudgetSession, "_check_memory", lambda s: None)
    original_summary = norms.summarize_bp_update
    for mode in ("default", "corrupt", "summary_failure", "output_only"):
        budget = ToyBudgetSession(
            ToyExecutionBudget(max_training_updates=12, max_wall_seconds=30), lambda: 0.0
        )
        for _ in range(12):
            budget.record_update()
        memory = SimpleNamespace()
        budget.attach_memory(cast(Any, memory))
        report = dict(
            definition="native_context_return_ticket_workflow_v1",
            method="bp",
            reserved_probe_requested=True,
            reserved_probe_source_training=dict(completed_updates=12, applied_update_count=12),
            checkpoints=[dict(state=probe.native.snapshot_record(state))],
        )
        row: dict[str, Any] = {}

        def prefix(seed, *a, **kw):
            learner = seed.fork()
            learner.train_batch(x, y)
            if mode == "output_only":
                after = learner.snapshot_state()
                for i, name in enumerate(native._PARAMETERS[:2]):
                    assert _same_values(after.head_state[name], state.head_state[name])
                    assert _same_values(
                        after.optimizer_state["state"][i], state.optimizer_state["state"][i]
                    )
                assert row["parameter_update_scope"] == "output_only"
                row["parameter_update_scope"] = "all"
                learner.train_batch(x, y)
            if mode == "corrupt":
                row["update_diagnostics"]["records"][0]["measurement"]["groups"]["hidden"][
                    "gradient_l2"
                ] = -1
                learner.train_batch(x, y)
            raise RuntimeError("intentional fixture stop")

        monkeypatch.setattr(probe, "run_fresh_task_probe", prefix)
        monkeypatch.setattr(
            norms,
            "summarize_bp_update",
            forbidden if mode in ("default", "summary_failure") else original_summary,
        )
        result = probe.run_owned_probe(
            state,
            spec,
            report,
            budget,
            check_boundary=lambda: None,
            memory=memory,
            row=row,
            probe_update_diagnostics=mode != "default",
            probe_output_only=mode == "output_only",
        )
        assert not result["complete"]
        assert result["preserved_training_work"] == dict(
            attempted_updates=1, attempted_rows=3, returned_updates=1
        )
        assert _same_values(source.snapshot_state(), state)
        if mode == "default":
            assert "update_diagnostics" not in row
            assert "parameter_update_scope" not in row
        elif mode == "corrupt":
            preserved = row["preserved_update_diagnostics"]
            assert preserved["records"][0]["measurement"]["groups"]["hidden"]["gradient_l2"] > 0
            assert (
                row["update_diagnostics"]["records"][0]["measurement"]["groups"]["hidden"][
                    "gradient_l2"
                ]
                == -1
            )
        elif mode == "output_only":
            assert "reserved probe source or paid work changed" in result["error"]
            assert row["preserved_update_diagnostics"]["parameter_update_scope"] == "output_only"
            assert (
                row["preserved_update_diagnostics"]["records"][0]["measurement"]["groups"][
                    "hidden"
                ]["parameter_delta_l2"]
                == 0
            )
        else:
            work = row["preserved_update_diagnostics"]["work"]
            assert work["snapshot_attempts"] == work["snapshot_returns"] == 2
            assert work["summary_attempts"] == 1 and work["summary_returns"] == 0
            assert row["preserved_update_diagnostics"]["records"][0]["status"] == "pending"
