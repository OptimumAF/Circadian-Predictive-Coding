"""Real incident stream and fresh-probe composition with fake native ports.

The prepared budgets, reserved data and disposable probe runner are real. Only
native learning, feature health and process memory sampling are replaced here.
"""

import json
from copy import deepcopy
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.run_shift_signal_pilot as cli
import src.adapters.learnability_pilot as probes
import src.adapters.shift_signal_pilot as pilot
import src.app.arrived_shift_observer as observer_module
import src.app.toy_execution_budget as budget_module
from src.app.toy_execution_budget import ToyBudgetSession
from src.core.coupled_energy import CoupledParameters
from src.core.learner_ports import TrainingDiagnostic
from src.shared.process_memory import ProcessRssSegment


def forbidden(*args, **kwargs):
    pytest.fail("native work or an unselected route reached")


@pytest.mark.parametrize(
    "fault", [None, "pc_bytes", "neutral_bytes", "both_bytes", "boolean", "counter", "health"]
)
def test_should_validate_exact_native_storage_without_ignoring_other_probe_costs(fault):
    pc: list[dict[str, Any]] = [
        {
            "checkpoint_snapshot_array_bytes": 840,
            "probe": {"completed_updates": 16},
            "feature_health": [0.5],
        }
        for _ in range(4)
    ]
    neutral = deepcopy(pc)
    for slot in neutral:
        slot["checkpoint_snapshot_array_bytes"] = 968
    if fault in {"pc_bytes", "both_bytes"}:
        pc[2]["checkpoint_snapshot_array_bytes"] = 968
    if fault in {"neutral_bytes", "both_bytes"}:
        neutral[2]["checkpoint_snapshot_array_bytes"] = 840
    if fault == "boolean":
        neutral[2]["checkpoint_snapshot_array_bytes"] = True
    if fault == "counter":
        neutral[2]["probe"]["completed_updates"] = 15
    if fault == "health":
        neutral[2]["feature_health"] = [0.4]
    assert pilot._learnability_parity(pc, neutral) is (fault is None)


def test_should_keep_original_paid_probe_commits_when_delegate_loses_its_report(
    fake_ports, monkeypatch
):
    original = probes._measure_probe

    def fail(*args, **kwargs):
        original(*args, **kwargs)
        args[4]["probe"] = None
        raise RuntimeError("injected delegate failure after spending original session")

    monkeypatch.setattr(probes, "_measure_probe", fail)
    row = run_one()
    assert row["status"] == "failed" and row["benchmark_completed_updates"] == 0
    assert row["probe_budget_completed_updates"] == [16, 0, 0, 0]
    assert row["probe_completed_updates"] == 16
    assert row["probe_reported_completed_updates"] == 0
    assert row["probe_completed_training_row_exposures"] == 128
    assert row["probe_work_unknown"] is True
    assert row["probe_attempted_updates"] is None
    assert row["probe_training_row_exposures"] is None
    assert row["probe_score_forward_calls_completed"] is None
    assert all(not (e[0] == "predict" and len(e[3]) == 64) for e in fake_ports.trace)


class FakeSampler:
    start_bytes = 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def sample(self):
        return 1

    def snapshot(self):
        return ProcessRssSegment(1, 1, 1, 1, 0.005)


class FakeLearner:
    def __init__(self, origin, trace, controls, value=0):
        self.origin, self.trace, self.controls = origin, trace, controls
        self.value, self.restored_value, self.since_restore = value, None, 0

    def fork(self):
        return FakeLearner(self.origin, self.trace, self.controls, self.value)

    def snapshot_state(self):
        return np.array([float(self.value)])

    def restore_state(self, state):
        self.value = int(state[0])
        self.restored_value, self.since_restore = self.value, 0
        self.trace.append(("restore", self.origin, self.value))

    def train_batch(self, features, targets):
        self.since_restore += 1
        self.trace.append(("train", self.origin, self.value, features.copy(), targets.copy()))
        if (
            "fail_probe_state" in self.controls
            and self.origin in self.controls.get("probe_origins", set())
            and self.restored_value == self.controls["fail_probe_state"]
            and self.since_restore == 3
            and len(features) == 8
        ):
            raise RuntimeError("injected paid fresh-probe failure")
        self.value += 1
        return TrainingDiagnostic("fake_shift_probe_v1", 0.0)

    def predict(self, features):
        self.trace.append(("predict", self.origin, self.value, features.copy()))
        return np.full((len(features), 1), 0.25 + 0.001 * self.value)


@pytest.fixture
def fake_ports(monkeypatch):
    constructed: list[FakeLearner] = []
    trace: list[tuple] = []
    sessions: list[ToyBudgetSession] = []
    controls: dict[str, Any] = {}
    clock = lambda: 0.0
    original_session = ToyBudgetSession.__init__

    def initialize(self, *args, **kwargs):
        original_session(self, *args, **kwargs)
        sessions.append(self)
        trace.append(("budget", len(sessions), self.budget.max_training_updates))

    def build(method, initial):
        controls["method"] = method
        source = FakeLearner(len(constructed), trace, controls)
        constructed.append(source)
        trace.append(("build", source.origin, method))
        return source

    def predict(state, features):
        trace.append(("observe", int(state[0]), features.copy()))
        return np.full((len(features), 1), 0.25)

    def health(state, reference, row):
        trace.append(("health", int(state[0]), reference.copy()))
        row.update(
            feature_health=[],
            diagnostic_seconds=0.0,
            diagnostic_activation_bytes=0,
            checkpoint_snapshot_array_bytes=968
            if controls["method"] in {"neutral_cpc", "gated_cpc"}
            else 840,
            importance=None,
            neuron_age=None,
            unavailable_reason="fake native health boundary",
        )

    original_facade = probes.create_sixteen_probe_observer

    def create(template, *args, **kwargs):
        controls.setdefault("probe_origins", set()).add(template.origin)
        return original_facade(template, *args, **kwargs)

    monkeypatch.setattr(ToyBudgetSession, "__init__", initialize)
    monkeypatch.setattr(pilot, "build_learner", build)
    monkeypatch.setattr(pilot, "_predict_snapshot", predict)
    monkeypatch.setattr(pilot, "_snapshot_array_bytes", lambda state: state.nbytes)
    monkeypatch.setattr(pilot, "monotonic", clock)
    monkeypatch.setattr(probes, "monotonic", clock)
    monkeypatch.setattr(observer_module, "monotonic", clock)
    monkeypatch.setattr(probes, "_measure_features", health)
    monkeypatch.setattr(probes, "create_sixteen_probe_observer", create)
    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    return SimpleNamespace(
        constructed=constructed,
        trace=trace,
        sessions=sessions,
        controls=controls,
    )


def run_one(*, stream="recurring", method="bp", learnability=True):
    row = pilot._new_row(method, stream, calibrated=True, incidents=True)
    pilot._run_one(
        row,
        None,
        60.0,
        FakeSampler(),
        calibrated=True,
        incidents=True,
        incident_replay=True,
        learnability=learnability,
    )
    return row


def semantic(value):
    """Probe timing is observed independently and is not a parity target."""
    if isinstance(value, dict):
        return {
            key: semantic(item)
            for key, item in value.items()
            if key not in {"elapsed_seconds", "probe_seconds", "diagnostic_seconds"}
        }
    if isinstance(value, list):
        return [semantic(item) for item in value]
    return value


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_probe_only_fixed_recurring_snapshots_using_separate_owned_forks(fake_ports, method):
    row = run_one(method=method)
    assert row["status"] == "complete", row["reason"]
    assert len(fake_ports.constructed) == 2
    benchmark, template = fake_ports.constructed
    assert benchmark is not template and benchmark.value == template.value == 0
    assert row["source_learner_unchanged"] is True
    slots = row["learnability"]
    assert [slot["updates"] for slot in slots] == [0, 6, 12, 18]
    assert all(slot["status"] == "complete" for slot in slots)
    assert [e[1] for e in fake_ports.trace if e[0] == "health"] == [0, 6, 16, 22]
    restores = [e[2] for e in fake_ports.trace if e[0] == "restore" and e[1] == template.origin]
    assert restores == [0, 6, 16, 22]
    for slot in slots:
        probe = slot["probe"]
        assert probe["status"] == "complete" and slot["probe_source_and_checkpoint_unchanged"]
        assert probe["attempted_updates"] == probe["completed_updates"] == 16
        assert probe["attempted_training_examples"] == probe["completed_training_examples"] == 128
        assert probe["score_forward_calls_attempted"] == probe["score_forward_calls_completed"] == 6
        assert probe["reserved_data_bytes"] == 3840
        assert [point["updates"] for point in probe["checkpoints"]] == [0, 1, 2, 4, 8, 16]
        curve = probe["checkpoints"]
        assert slot["fresh_initial_bce"] == curve[0]["scores"]["C"]["bce"]
        assert (
            slot["fresh_gain"] == curve[0]["scores"]["C"]["bce"] - curve[-1]["scores"]["C"]["bce"]
        )
    assert row["probe_completed_updates"] == row["probe_attempted_updates"] == 64
    assert (
        row["probe_training_row_exposures"] == row["probe_completed_training_row_exposures"] == 512
    )
    trace = fake_ports.trace
    trains = [e for e in trace if e[0] == "train"]
    wake_replay = [e for e in trains if e[1] == benchmark.origin]
    fresh = [e for e in trains if e[1] == template.origin]
    assert [len(e[3]) for e in wake_replay] == [8] * 12 + [1] * 4 + [8] * 6
    assert len(fresh) == 64 and all(len(e[3]) == 8 for e in fresh)
    reserved, _, _ = probes.build_reserved_data(extended=True)
    for index, event in enumerate(fresh):
        np.testing.assert_array_equal(event[3], reserved[index % 16].features)
        np.testing.assert_array_equal(event[4], reserved[index % 16].targets)
    predicts = [e for e in trace if e[0] == "predict"]
    assert len([e for e in predicts if e[1] == template.origin]) == 24
    assert all(len(e[3]) == 32 for e in predicts if e[1] == template.origin)
    final = [e for e in predicts if len(e[3]) == 64]
    assert len(final) == 38 and all(e[1] == benchmark.origin for e in final)
    first_final = next(
        index for index, e in enumerate(trace) if e[0] == "predict" and len(e[3]) == 64
    )
    assert all(e[0] in {"restore", "predict"} for e in trace[first_final:])
    # The return probe also finishes before the first benchmark final score.
    assert sum(e[0] == "train" for e in trace[:first_final]) == 86
    assert sum(e[0] == "observe" for e in trace[:first_final]) == 18
    assert [o["prediction_replay_updates"] for o in row["observations"]["rows"]] == [0] * 12 + [
        4
    ] * 6


def test_should_preserve_benchmark_scores_receipts_signals_and_replay_with_probes(fake_ports):
    baseline = run_one(learnability=False)
    before = len(fake_ports.trace)
    selected = run_one()
    assert baseline["status"] == selected["status"] == "complete"
    for key in (
        "stream_result",
        "observations",
        "rehearsal",
        "signal_outcomes",
        "incident_outcomes",
        "outcomes",
    ):
        assert baseline[key] == selected[key]
    baseline_trace = fake_ports.trace[:before]
    selected_trace = fake_ports.trace[before:]
    baseline_work = [e for e in baseline_trace if e[0] == "train"]
    selected_work = [e for e in selected_trace if e[0] == "train" and e[1] == 1]
    assert len(baseline_work) == len(selected_work) == 22
    for left, right in zip(baseline_work, selected_work, strict=True):
        assert left[2] == right[2]
        np.testing.assert_array_equal(left[3], right[3])
        np.testing.assert_array_equal(left[4], right[4])


@pytest.mark.parametrize("method", ["bp", "pc", "neutral_cpc", "gated_cpc"])
def test_should_keep_stationary_false_trigger_controls_without_any_probes(fake_ports, method):
    row = run_one(stream="stationary", method=method)
    assert row["status"] == "complete", row["reason"]
    assert row["learnability"] is None
    assert row["probe_completed_updates"] == row["probe_attempted_updates"] == 0
    assert row["probe_training_row_exposures"] == 0
    assert len(fake_ports.constructed) == 1 and len(fake_ports.sessions) == 2
    assert row["incident_outcomes"]["feature"]["stationary_false_incidents"] == [8, 11]
    assert row["rehearsal"]["committed_updates"] == 4
    assert not any(e[0] == "health" for e in fake_ports.trace)
    assert sum(e[0] == "train" for e in fake_ports.trace) == 22


def test_should_prepare_all_four_original_probe_sessions_before_model_work(fake_ports):
    row = run_one()
    assert row["status"] == "complete", row["reason"]
    sessions = fake_ports.sessions
    assert len(sessions) == 6 and len({id(session) for session in sessions}) == 6
    wake, replay, *reserved = sessions
    assert wake.budget.max_training_updates == 18
    assert replay.budget.max_replay_examples == 4
    assert all(
        session.budget.max_training_updates == session.updates_completed == 16
        for session in reserved
    )
    assert all(session.clock is wake.clock for session in sessions)
    assert all(session.process_rss_sampler is wake.process_rss_sampler for session in sessions)
    assert all(
        session.started_at == 0.0 and session.budget.max_wall_seconds == 60.0
        for session in sessions
    )
    first_model = next(index for index, e in enumerate(fake_ports.trace) if e[0] == "build")
    assert sum(e[0] == "budget" for e in fake_ports.trace[:first_model]) == 6


def test_should_preserve_partial_probe_costs_and_stop_before_next_wake_or_finals(fake_ports):
    fake_ports.controls["fail_probe_state"] = 16
    row = run_one()
    assert row["status"] == "failed" and "paid fresh-probe failure" in row["reason"]
    assert row["benchmark_completed_updates"] == 12
    assert row["replay_budget_charged_exposures"] == 4
    assert row["rehearsal"]["committed_updates"] == 4
    assert row["observations"]["prediction_calls_completed"] == 12
    slots = row["learnability"]
    assert [slot["status"] for slot in slots] == ["complete", "complete", "failed", "not_run"]
    failed = slots[2]["probe"]
    assert failed["attempted_updates"] == 3 and failed["completed_updates"] == 2
    assert (
        failed["attempted_training_examples"] == 24 and failed["completed_training_examples"] == 16
    )
    assert [point["updates"] for point in failed["checkpoints"]] == [0, 1, 2]
    assert row["probe_attempted_updates"] == 35 and row["probe_completed_updates"] == 34
    assert row["probe_training_row_exposures"] == 280
    assert row["probe_completed_training_row_exposures"] == 272
    assert [s.updates_completed for s in fake_ports.sessions[2:]] == [16, 16, 2, 0]
    assert not any(e[0] == "predict" and len(e[3]) == 64 for e in fake_ports.trace)
    assert all(source.value == 0 for source in fake_ports.constructed)


def test_should_include_probe_costs_and_semantic_pc_neutral_parity_across_all_eight_cases(
    fake_ports, monkeypatch
):
    parameters = CoupledParameters(
        (np.zeros((2, 8)), np.zeros((8, 8))),
        (np.zeros((1, 8)), np.zeros((1, 8))),
        np.zeros((8, 1)),
        np.zeros((1, 1)),
    )
    monkeypatch.setattr(
        pilot,
        "CoupledPredictiveCodingNetwork",
        lambda *args: SimpleNamespace(snapshot_parameters=lambda: parameters),
    )
    report = pilot.run_pilot(
        calibrated=True, incidents=True, incident_replay=True, learnability=True
    )
    assert report["status"] == "complete", report["global_errors"]
    assert len(report["runs"]) == 8
    assert report["benchmark_completed_updates"] == report["completed_observations"] == 144
    assert report["native_training_updates"] == report["native_training_updates_committed"] == 432
    assert report["training_row_exposures"] == 3232
    assert report["training_costs_complete"] is True
    assert report["pc_neutral_exact"] == {"stationary": True, "recurring": True}
    assert sum(r["probe_completed_updates"] for r in report["runs"]) == 256
    assert all(r["learnability"] is None for r in report["runs"] if r["stream"] == "stationary")
    pc, neutral = [
        r
        for r in report["runs"]
        if r["stream"] == "recurring" and r["method"] in ("pc", "neutral_cpc")
    ]
    assert [s["checkpoint_snapshot_array_bytes"] for s in pc["learnability"]] == [840] * 4
    assert [s["checkpoint_snapshot_array_bytes"] for s in neutral["learnability"]] == [968] * 4
    strip_storage = lambda slots: [
        {k: v for k, v in s.items() if k != "checkpoint_snapshot_array_bytes"} for s in slots
    ]
    assert semantic(strip_storage(pc["learnability"])) == semantic(
        strip_storage(neutral["learnability"])
    )
    assert report["scientific_advantage_verified"] is report["full_R27_complete"] is False


def test_should_aggregate_paid_partial_probes_without_zero_imputing_failed_work(
    fake_ports, monkeypatch
):
    fake_ports.controls["fail_probe_state"] = 16
    parameters = CoupledParameters(
        (np.zeros((2, 8)), np.zeros((8, 8))),
        (np.zeros((1, 8)), np.zeros((1, 8))),
        np.zeros((8, 1)),
        np.zeros((1, 1)),
    )
    monkeypatch.setattr(
        pilot,
        "CoupledPredictiveCodingNetwork",
        lambda *args: SimpleNamespace(snapshot_parameters=lambda: parameters),
    )
    report = pilot.run_pilot(
        calibrated=True, incidents=True, incident_replay=True, learnability=True
    )
    assert report["status"] == "incomplete"
    assert report["benchmark_completed_updates"] == report["completed_observations"] == 120
    assert report["native_training_updates"] == 292
    assert report["native_training_updates_committed"] == 288
    assert report["training_row_exposures"] == 2112
    assert report["training_costs_complete"] is False
    assert sum(r["probe_attempted_updates"] for r in report["runs"]) == 140
    assert sum(r["probe_completed_updates"] for r in report["runs"]) == 136
    assert all(r["status"] == "complete" for r in report["runs"] if r["stream"] == "stationary")
    assert all(r["status"] == "failed" for r in report["runs"] if r["stream"] == "recurring")
    # Only the four completed stationary controls may have released final scores.
    assert sum(e[0] == "predict" and len(e[3]) == 64 for e in fake_ports.trace) == 152


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"calibrated": True},
        {"incidents": True},
        {"calibrated": True, "incidents": True},
        {"calibrated": True, "incidents": True, "incident_replay": True},
    ],
)
def test_should_preserve_all_five_old_descriptors_with_explicit_false(options):
    assert pilot.describe_protocol(**options) == pilot.describe_protocol(
        **options, learnability=False
    )


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", np.bool_(True)])
def test_should_refuse_non_boolean_learnability_before_native_work(entry, invalid, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    monkeypatch.setattr(pilot, "build_stream", forbidden)
    with pytest.raises(ValueError, match="bool"):
        entry(calibrated=True, incidents=True, incident_replay=True, learnability=invalid)


@pytest.mark.parametrize("entry", [pilot.describe_protocol, pilot.run_pilot])
@pytest.mark.parametrize("missing", ["calibrated", "incidents", "incident_replay"])
def test_should_require_all_feedback_prerequisites_before_probes(entry, missing, monkeypatch):
    monkeypatch.setattr(pilot, "ProcessRssSampler", forbidden)
    options = dict(calibrated=True, incidents=True, incident_replay=True, learnability=True)
    options[missing] = False
    with pytest.raises(ValueError):
        entry(**options)


@pytest.mark.parametrize("action", ["describe", "run"])
def test_should_forward_only_selected_cli_learnability(action, monkeypatch, capsys):
    calls = []

    def selected(**options):
        calls.append(options)
        return {"status": "complete"}

    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(cli, "describe_protocol" if action == "describe" else "run_pilot", selected)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "shift",
            f"--{action}",
            "--calibrated",
            "--incidents",
            "--incident-replay",
            "--learnability",
        ],
    )
    assert cli.main() == 0
    assert calls == [dict(calibrated=True, incidents=True, incident_replay=True, learnability=True)]
    assert json.loads(capsys.readouterr().out) == {"status": "complete"}


@pytest.mark.parametrize("missing", ["--calibrated", "--incidents", "--incident-replay"])
def test_should_refuse_incomplete_cli_probe_mode_before_dispatch(missing, monkeypatch):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    flags = ["--calibrated", "--incidents", "--incident-replay", "--learnability"]
    monkeypatch.setattr(
        sys, "argv", ["shift", "--describe", *[flag for flag in flags if flag != missing]]
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
