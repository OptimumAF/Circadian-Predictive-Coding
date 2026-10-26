"""Fixed native bindings and fake orchestration; no comparative pilot execution."""

import pickle
import sys
from typing import Any

import pytest

import scripts.run_coupled_attribution as cli
import src.adapters.unit_renewal_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.core.coupled_predictive_coding import CoupledPredictiveCodingNetwork
from src.core.renewable_coupled_predictive_coding import RenewableCoupledSnapshot
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("unselected operation reached")


def test_should_describe_frozen_protocol_without_training_loading_controls_or_sampling(monkeypatch):
    for name in ("_build_learner", "build_stream", "ProcessRssSampler", "run_attribution_stream"):
        monkeypatch.setattr(pilot, name, forbidden)
    description = pilot.describe_protocol()
    assert description["seed"] == 23
    assert description["executed_methods"] == ["pc", "neutral_cpc", "gated_cpc"]
    assert description["retained_controls"]["methods"] == ["bp", "pc", "neutral_cpc", "gated_cpc"]
    assert description["renewal"]["policy"] == dict(
        min_age_updates=6, max_importance=0.001, max_per_event=1, seed=29
    )
    assert description["planned_work"]["total_updates"] == 246
    assert description["planned_work"]["total_training_row_exposures"] == 1968
    assert description["planned_work"]["retained_control_new_native_updates"] == 0
    assert description["probe"]["probe_checkpoint_updates"] == [0, 1, 2, 4, 8, 16]
    assert description["benchmark"]["benchmark_checkpoint_updates"] == [0, 6, 12, 18]


@pytest.mark.parametrize("method", ["pc", "neutral_cpc", "gated_cpc"])
def test_should_bind_complete_owned_policy_and_exact_initial_parameters(method):
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    before = pickle.dumps(initial)
    learner = pilot._build_learner(method, initial)
    state = learner.snapshot_state()
    assert type(state) is RenewableCoupledSnapshot
    assert state.renewal_policy == pilot.renewal_policy()
    assert state.unit_history is not None and state.unit_history.updates == 0
    assert state.unit_history.unit_ids.tolist() == list(range(8))
    assert pickle.dumps(state.parameters) == pickle.dumps(initial)
    assert learner.checkpoint_policy().inference_steps == 4
    assert pickle.dumps(initial) == before
    fork = learner.fork()
    assert pickle.dumps(fork.snapshot_state()) == pickle.dumps(state)


@pytest.mark.parametrize("method", ["bp", "unknown"])
def test_should_refuse_extra_executed_method(method):
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    with pytest.raises(ValueError):
        pilot._build_learner(method, initial)


@pytest.mark.parametrize("scenario", ["original", "renewal-learnability"])
@pytest.mark.parametrize("describe", [False, True])
def test_should_route_only_selected_cli_without_changing_default(
    scenario, describe, monkeypatch, capsys
):
    selected = ("describe_" if describe else "run_") + (
        "renewal_protocol"
        if describe and scenario != "original"
        else "renewal_pilot"
        if scenario != "original"
        else "protocol"
        if describe
        else "pilot"
    )
    for name in (
        "describe_protocol",
        "run_pilot",
        "describe_renewal_protocol",
        "run_renewal_pilot",
    ):
        monkeypatch.setattr(
            cli, name, (lambda: {"status": "complete"}) if name == selected else forbidden
        )
    args = ["pilot"] + (["--describe"] if describe else [])
    if scenario != "original":
        args += ["--scenario", scenario]
    monkeypatch.setattr(sys, "argv", args)
    cli.main()
    assert '"status": "complete"' in capsys.readouterr().out


def test_should_refuse_unknown_cli_scenario_before_callbacks(monkeypatch):
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "run_renewal_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["pilot", "--scenario", "tuned"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


class Sampler:
    start_bytes = 100

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 100

    def snapshot(self):
        return ProcessRssSegment(1, 100, 100, 1, 0.01)


def fake_row(row, initial, deadline, sampler):
    row.update(
        status="complete",
        reason=None,
        benchmark_completed_updates=18,
        renewal=dict(charged_units=0, committed_units=0),
        stream_result=dict(checkpoints=[]),
        source_unchanged=True,
    )
    for observation in row["observations"]:
        observation.update(
            status="complete",
            feature_health=[{}, {}],
            native_unit_history={"same": True},
            probe=dict(
                completed_updates=16,
                completed_training_examples=128,
                attempted_updates=16,
                score_forward_calls_attempted=6,
                score_forward_calls_completed=6,
                checkpoints=[],
            ),
        )


@pytest.mark.parametrize("failure", ["none", "row", "setup", "sampler", "parity"])
def test_should_preserve_all_planned_rows_and_refuse_incomplete_work_without_tuning(
    failure, monkeypatch
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", Sampler)
    seen: list[str] = []

    def run_one(row, initial, deadline, sampler):
        seen.append(row["method"])
        fake_row(row, initial, deadline, sampler)
        if failure == "row" and row["method"] == "neutral_cpc":
            row.update(status="failed", reason="preserved negative fault", outcomes=None)
            row["renewal"] = dict(charged_units=1, committed_units=0)
        if failure == "parity" and row["method"] == "neutral_cpc":
            row["observations"][0]["native_unit_history"] = {"different": True}

    monkeypatch.setattr(pilot, "_run_one", run_one)
    if failure == "setup":
        monkeypatch.setattr(
            pilot,
            "CoupledPredictiveCodingNetwork",
            lambda *a: (_ for _ in ()).throw(RuntimeError("setup fault")),
        )
    if failure == "sampler":
        monkeypatch.setattr(
            pilot, "ProcessRssSampler", lambda: (_ for _ in ()).throw(RuntimeError("sampler fault"))
        )
    report = pilot.run_pilot()
    assert [row["method"] for row in report["runs"]] == ["pc", "neutral_cpc", "gated_cpc"]
    assert report["status"] == ("complete" if failure == "none" else "incomplete")
    assert report["scientific_advantage_verified"] is False
    if failure == "row":
        assert (
            report["charged_replacement_units"] == 1 and report["committed_replacement_units"] == 0
        )
        assert report["runs"][1]["reason"] == "preserved negative fault"
    if failure in ("setup", "sampler"):
        assert not seen and all(row["status"] == "not_run" for row in report["runs"])
    else:
        assert seen == ["pc", "neutral_cpc", "gated_cpc"]


def test_should_bind_separate_probe_template_and_keep_failed_charge_in_real_row_boundary(
    monkeypatch,
):
    monkeypatch.setattr(pilot, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", Sampler)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    initial = CoupledPredictiveCodingNetwork(2, (8, 8), 23).snapshot_parameters()
    sources: list[Any] = []

    def build(method, parameters):
        instance = type("Source", (), {"snapshot_state": lambda self: {"owned": True}})()
        sources.append(instance)
        return instance

    monkeypatch.setattr(pilot, "_build_learner", build)

    class Renewal:
        def __init__(self, learner, budget, **controls):
            assert learner is sources[0]
            assert controls == dict(every_updates=6, total_updates=18, max_replacements=3)
            self.report = dict(
                charged_units=1, attempted_units=1, returned_units=0, committed_units=0
            )

    monkeypatch.setattr(pilot, "StreamUnitRenewal", Renewal)

    def observer(template, slots, deadline, sampler):
        assert template is sources[1] and template is not sources[0]
        return forbidden

    monkeypatch.setattr(pilot.learnability, "create_sixteen_probe_observer", observer)

    def stream(learner, training, heldout, **kwargs):
        assert (
            learner is sources[0]
            and training[0].role == "train"
            and heldout[0].role == "final_test"
        )
        assert kwargs["checkpoint_updates"] == (0, 6, 12, 18) and kwargs["label_delay"] == 2
        raise RuntimeError("actual row boundary fault")

    monkeypatch.setattr(pilot, "run_attribution_stream", stream)
    row = dict(
        method="pc",
        status="not_run",
        observations=pilot.learnability.empty_learnability_observations(),
    )
    sampler: Any = Sampler()  # The production sampler class is replaced above.
    pilot._run_one(row, initial, 60.0, sampler)
    assert row["status"] == "failed" and row["outcomes"] is None
    assert row["renewal"]["charged_units"] == 1
    assert row["failed_attempt_work_not_fully_counted"] is True
    assert "actual row boundary fault" in row["reason"]


def test_should_keep_raw_outcome_directions_independent_of_winner():
    result = dict(
        checkpoints=[
            dict(updates=n, scores={"A": {"bce": a}, "B": {"bce": b}})
            for n, a, b in [(0, 1.0, 2.0), (6, 0.4, 1.5), (12, 0.8, 1.0), (18, 0.3, 1.2)]
        ]
    )
    assert pilot._outcomes(result) == dict(retention_loss=0.4, return_gain=0.5, new_B_gain=1.0)
