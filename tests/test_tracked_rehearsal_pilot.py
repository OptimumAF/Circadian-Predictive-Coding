"""Fixed tracked-mode routing/accounting; real native observer tested separately."""

import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.coupled_attribution_pilot as bindings
import src.adapters.stream_rehearsal_pilot as pilot
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.tracked_coupled_predictive_coding import TrackedCoupledSnapshot
from test_rehearsal_learnability import fake_environment, forbidden


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_route_the_fixed_tracked_cli_without_single_mode_kwargs(action, monkeypatch, capsys):
    calls: list[Any] = []

    def selected():
        calls.append(action)
        return {"status": "complete"}

    for name in (
        "describe_protocol",
        "run_pilot",
        "describe_paired_protocol",
        "run_paired_pilot",
        "describe_tracked_protocol",
        "run_tracked_pilot",
    ):
        expected = "describe_tracked_protocol" if action == "describe" else "run_tracked_pilot"
        monkeypatch.setattr(cli, name, selected if name == expected else forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", f"--{action}", "--scenario", "tracked-learnability"])
    assert cli.main() == 0 and capsys.readouterr().out
    assert calls == [action]


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("flag", ["--recent-only", "--diagnose-learnability"])
def test_should_reject_single_case_flags_before_any_tracked_callback(action, flag, monkeypatch):
    monkeypatch.setattr(cli, "describe_tracked_protocol", forbidden)
    monkeypatch.setattr(cli, "run_tracked_pilot", forbidden)
    monkeypatch.setattr(
        sys, "argv", ["pilot", f"--{action}", "--scenario", "tracked-learnability", flag]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_should_fix_tracking_metadata_without_changing_original_probe_data_or_budget(monkeypatch):
    for name in (
        "build_stream",
        "build_learner",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    original = pilot.describe_protocol(diagnose_learnability=True)
    tracked = pilot.describe_tracked_protocol()
    assert tracked["protocol_id"] == "coherent_tracked_rehearsal_reserved_sixteen_v1"
    assert tracked["control_for"] == original["protocol_id"]
    for key in (
        "resources",
        "methods",
        "seeds",
        "training_rng",
        "final_rng",
        "replay",
        "hidden_dims",
        "learning_rate",
        "inference_steps",
        "inference_learning_rate",
        "checkpoint_updates",
        "probe_isolation",
    ):
        assert tracked[key] == original[key]
    assert all(
        tracked["planned_work"][key] == value for key, value in original["planned_work"].items()
    )
    assert tracked["planned_work"]["native_unit_history_reads"] == 12
    assert tracked["planned_work"]["native_unit_history_transitions"] == 282
    assert tracked["native_unit_history"]["checkpoint_native_updates"] == [0, 10, 20, 30]
    assert tracked["native_unit_history"]["persistent_array_bytes_per_model"] == 192
    assert tracked["native_unit_history"]["ema_decay"] == 0.95
    assert tracked["native_unit_history"]["methods"] == ["pc", "neutral_cpc", "gated_cpc"]
    for key, value in original["learnability"].items():
        if key != "unavailable_diagnostics":
            assert tracked["learnability"][key] == value


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_bind_exact_tracked_types_and_unchanged_initial_parameters_without_training(method):
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    before = pickle.dumps(initial)
    ordinary = bindings.build_learner(method, initial)
    tracked = bindings.build_learner(method, initial, track_unit_history=True)
    a, b = ordinary.snapshot_state(), tracked.snapshot_state()
    for name in ("hidden_weights", "hidden_biases"):
        for first, second in zip(
            getattr(a.parameters, name), getattr(b.parameters, name), strict=True
        ):
            np.testing.assert_array_equal(first, second)
            assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(a.parameters.output_weight, b.parameters.output_weight)
    assert pickle.dumps(initial) == before
    if method == "bp":
        assert type(a) is type(b) and not hasattr(b, "unit_history")
    else:
        assert type(b) is TrackedCoupledSnapshot and b.unit_history is not None
        assert a.format_version == 1 and b.format_version == 2
        assert b.unit_history.updates == 0 and b.unit_history.ema_decay == 0.95
        assert (
            b.unit_history.unit_ids.nbytes
            + b.unit_history.importance_ema.nbytes
            + b.unit_history.update_age.nbytes
            == 192
        )


@pytest.mark.parametrize("value", [0, 1, None])
def test_should_refuse_nonboolean_builder_tracking_before_model_work(value, monkeypatch):
    monkeypatch.setattr(bindings, "CoupledPredictiveCodingNetwork", forbidden)
    initial: Any = None
    with pytest.raises(ValueError, match="exact boolean"):
        bindings.build_learner("pc", initial, track_unit_history=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(track_unit_history=1, diagnose_learnability=True),
        dict(track_unit_history=True),
        dict(track_unit_history=True, diagnose_learnability=True, recent_only=True),
    ],
)
def test_should_refuse_unsupported_tracking_combinations_before_setup(kwargs, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    with pytest.raises(ValueError):
        pilot.run_pilot(**kwargs)


def test_should_compose_the_no_argument_facade_with_the_existing_pilot(monkeypatch):
    calls: list[Any] = []

    def run(**kwargs):
        calls.append(kwargs)
        return {"status": "fixture"}

    monkeypatch.setattr(pilot, "run_pilot", run)
    assert pilot.run_tracked_pilot() == {"status": "fixture"}
    assert calls == [dict(diagnose_learnability=True, track_unit_history=True)]


def tracked_fake_environment(monkeypatch, **faults):
    control = fake_environment(monkeypatch, **faults)
    control["builder_options"] = []
    build = pilot.build_learner
    create = pilot.learnability.create_sixteen_probe_observer

    def tracked_build(method, initial, **kwargs):
        control["builder_options"].append((method, kwargs))
        assert kwargs == {"track_unit_history": True}
        return build(method, initial)

    def fake_observer_metadata(template, observations, deadline, sampler):
        observer = create(template, observations, deadline, sampler)

        def observe(updates, state, arrived):
            observer(updates, state, arrived)
            if template.method != "bp":
                row = observations[pilot.CHECKPOINTS.index(updates)]
                # Deliberately fake adapter metadata. Real state/gradient/isolation
                # semantics are verified by the separate native observer tests.
                row.update(
                    native_unit_history=dict(
                        successful_updates=int(state[0]), array_bytes=192, fixture=True
                    )
                )

        return observe

    monkeypatch.setattr(pilot, "build_learner", tracked_build)
    monkeypatch.setattr(pilot.learnability, "create_sixteen_probe_observer", fake_observer_metadata)
    return control


def test_should_run_one_fixed_tracked_matrix_with_independent_templates_and_explicit_accounting(
    monkeypatch,
):
    control = tracked_fake_environment(monkeypatch)
    report = pilot.run_tracked_pilot()
    assert report["status"] == "complete", [r["reason"] for r in report["runs"]]
    assert report["native_unit_history_observations_completed"] == 12
    assert report["native_unit_history_array_bytes_in_completed_observations"] == 2304
    assert report["pc_neutral_native_unit_histories_exact"] is True
    assert (
        report["total_completed_updates"] == 376 and report["total_training_row_exposures"] == 2672
    )
    assert len(control["builder_options"]) == 8
    assert [method for method, _ in control["builder_options"]] == [
        method for method in pilot.METHODS for _ in range(2)
    ]
    assert len({id(x) for x in control["sources"]}) == 8
    assert all(
        row["source_learner_unchanged"] and row["probe_template_unchanged"]
        for row in report["runs"]
    )
    assert all(
        o["importance"] is o["neuron_age"] is None for o in report["runs"][0]["observations"]
    )
    assert all("native_unit_history" not in o for o in report["runs"][0]["observations"])
    assert all(source.state[0] == 0 for source in control["sources"])
    assert len(control["probes"]) == 16


def test_should_preserve_failed_probe_and_unreached_history_slots_without_zero_filled_parity(
    monkeypatch,
):
    tracked_fake_environment(monkeypatch, fail_probe=True)
    report = pilot.run_tracked_pilot()
    assert report["status"] == "incomplete"
    assert report["native_unit_history_observations_completed"] == 3
    assert report["pc_neutral_native_unit_histories_exact"] is None
    assert all(row["status"] == "failed" for row in report["runs"])
    for row in report["runs"]:
        assert [o["status"] for o in row["observations"]] == [
            "complete",
            "failed",
            "not_run",
            "not_run",
        ]
        assert row["observations"][1]["probe"]["attempted_updates"] == 3
        assert "declared probe mutation" in row["reason"]


def test_should_keep_all_planned_probe_slots_when_tracked_global_setup_fails(monkeypatch):
    def fail():
        raise RuntimeError("declared setup failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    report = pilot.run_tracked_pilot()
    assert report["status"] == "incomplete" and len(report["runs"]) == 4
    assert all(len(row["observations"]) == 4 for row in report["runs"])
    assert report["native_unit_history_observations_completed"] == 0
    assert report["pc_neutral_native_unit_histories_exact"] is None
    assert "declared setup failure" in report["global_errors"][0]
