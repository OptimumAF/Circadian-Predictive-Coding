"""Reserved probes use disposable fake learners and real observer/probe orchestration."""

import json
import pickle
import sys
from typing import Any

import numpy as np
import pytest

import scripts.run_stream_rehearsal_pilot as cli
import src.adapters.learnability_pilot as learnability
import src.adapters.stream_rehearsal_pilot as pilot
import src.app.toy_execution_budget as budget_module
from src.app.fresh_task_probe import run_fresh_task_probe
from src.core.learner_ports import TrainingDiagnostic
from test_stream_rehearsal_pilot import FakeModel, FakeSampler


# Independent complete DATA oracle from r23-native-rehearsal-20261016/protocol.json.
RETAINED_PROTOCOL = json.loads(r"""{
"protocol_id":"coherent_stream_shared_rehearsal_v1",
"classification":"exploratory_native_rehearsal_engineering_smoke",
"methods":["bp","pc","neutral_cpc","gated_cpc"],"streams":["recurring"],"seeds":[23],
"training_rng":7301,"final_rng":9101,"hidden_dims":[8,8],"learning_rate":0.03,
"inference_steps":4,"inference_learning_rate":0.2,"precisions":[1.0,1.0],
"active_cpc_config":{"chemical_decay":0.995,"chemical_buildup_rate":0.02,"plasticity_sensitivity":0.7,"min_plasticity":0.2},
"neutral_cpc_config":{"chemical_decay":0.995,"chemical_buildup_rate":0.02,"plasticity_sensitivity":0.7,"min_plasticity":1.0},
"training_batches":18,"batch_rows":8,"final_rows_per_context":64,"label_delay":2,"checkpoint_updates":[0,6,12,18],
"replay":{"every_updates":3,"rows_per_event":2,"retention":{"max_examples":16,"max_bytes":384},"allocation":{"recent":{"max_examples":8,"max_bytes":192},"reservoir":{"max_examples":8,"max_bytes":192}},"seed":53},
"replay_selection":"arrived-only recent/reservoir rotating interleave; context-blind",
"score_release":"after all wake and replay training for each arm; no feedback",
"metrics":{"primary":"BCE clipped1e-12 lower better","secondary":"accuracy >=0.5 higher better"},
"directions":{"new_B_gain":"B BCE at0 minus12","retention_loss":"A BCE at12 minus6","return_gain":"A BCE at12 minus18"},
"charging":"separate initially planned wake/replay sessions, one global deadline; replay charged before admission; failed charges never refunded",
"native_clocks":"replay calls native train_batch and advances its native clocks/gates",
"planned_work":{"wake_updates":72,"replay_updates":48,"optimizer_updates":120,"wake_row_exposures":576,"replay_row_exposures":48,"total_row_exposures":624,"consolidation_events":24,"final_score_forwards":32,"probes":0,"svd":0},
"resources":{"wall_seconds":60.0,"training_updates":120,"environment_steps":0,"cpu_threads":1,"process_memory_bytes":268435456,"replay_bytes":384,"model_download_bytes":0,"storage_bytes":33554432,"gpu_memory_bytes":0},
"stop_policy":"one invocation, all failures/unreached rows retained; no retry",
"limitations":"one fixed seed/data, unequal compute versus retained no-replay evidence; no confirmatory or circadian advantage claim"}""")


def forbidden(*args, **kwargs):
    pytest.fail("unexpected model, feature evaluator, SVD, or operational callback")


def test_should_preserve_the_entire_retained_default_descriptor_with_explicit_false(monkeypatch):
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(learnability, "create_sixteen_probe_observer", forbidden)
    assert pilot.describe_protocol() == RETAINED_PROTOCOL
    assert pilot.describe_protocol(diagnose_learnability=False) == RETAINED_PROTOCOL


@pytest.mark.parametrize("action", ["describe_protocol", "run_pilot"])
@pytest.mark.parametrize("invalid", [0, 1, None])
def test_should_refuse_nonboolean_mode_before_data_model_sampler_or_observer_work(
    action, invalid, monkeypatch
):
    for name in (
        "build_stream",
        "build_learner",
        "CoupledPredictiveCodingNetwork",
        "ProcessRssSampler",
    ):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(learnability, "create_sixteen_probe_observer", forbidden)
    with pytest.raises(ValueError):
        getattr(pilot, action)(diagnose_learnability=invalid)


def test_should_add_only_the_fixed_sixteen_probe_controls_and_preserve_benchmark_inputs():
    described = pilot.describe_protocol(diagnose_learnability=True)
    assert described["protocol_id"] == "coherent_rehearsal_reserved_sixteen_v1"
    for key in (
        "methods",
        "seeds",
        "training_rng",
        "final_rng",
        "hidden_dims",
        "learning_rate",
        "inference_steps",
        "inference_learning_rate",
        "precisions",
        "checkpoint_updates",
        "replay",
        "native_clocks",
    ):
        assert described[key] == RETAINED_PROTOCOL[key]
    controls = described["learnability"]
    retained = learnability.describe_protocol(extended=True)
    assert controls["source_protocol_id"] == "reserved_fresh_task_learnability_sixteen_v1"
    assert all(
        value == retained[key] for key, value in controls.items() if key != "source_protocol_id"
    )
    assert controls["probe_training_rng"] == 17031 and controls["probe_evaluation_rng"] == 17033
    assert controls["diagnostic_features_rng"] == 17035
    assert controls["probe_training_batches"] == 16 and controls["probe_checkpoint_updates"] == [
        0,
        1,
        2,
        4,
        8,
        16,
    ]
    assert (
        controls["reserved_data_array_bytes"] == 3840
        and controls["diagnostic_reference_array_bytes"] == 512
    )
    assert set(controls["unavailable_diagnostics"]) == {"importance", "neuron_age"}
    assert (
        described["probe_feedback"] is False and described["resources"]["training_updates"] == 376
    )
    assert described["planned_work"] == dict(
        wake_updates=72,
        replay_updates=48,
        optimizer_updates=376,
        wake_row_exposures=576,
        replay_row_exposures=48,
        total_row_exposures=2672,
        consolidation_events=24,
        final_score_forwards=32,
        probes=16,
        svd=32,
        probe_updates=256,
        probe_row_exposures=2048,
        probe_clones=16,
        probe_score_forwards=96,
        diagnostic_feedforward_passes=16,
    )


@pytest.mark.parametrize("action", ["describe", "run"])
@pytest.mark.parametrize("enabled", [False, True])
def test_should_route_only_the_selected_cli_handler_and_preserve_no_keyword_default_calls(
    action, enabled, monkeypatch, capsys
):
    calls = []

    def selected(**kwargs):
        calls.append(kwargs)
        return {"status": "complete", "selected": action}

    monkeypatch.setattr(cli, "describe_protocol", selected if action == "describe" else forbidden)
    monkeypatch.setattr(cli, "run_pilot", selected if action == "run" else forbidden)
    monkeypatch.setattr(
        sys, "argv", ["rehearsal", f"--{action}", *(["--diagnose-learnability"] if enabled else [])]
    )
    assert cli.main() == 0
    assert calls == ([{"diagnose_learnability": True}] if enabled else [{}])
    assert json.loads(capsys.readouterr().out)["selected"] == action


@pytest.mark.parametrize(
    "arguments",
    [
        ["--diagnose-learnability"],
        ["--describe", "--run", "--diagnose-learnability"],
        ["--run", "--diagnose-learnability", "--probe-updates", "32"],
    ],
)
def test_should_reject_missing_conflicting_or_unplanned_cli_controls_before_work(
    arguments, monkeypatch
):
    monkeypatch.setattr(cli, "describe_protocol", forbidden)
    monkeypatch.setattr(cli, "run_pilot", forbidden)
    monkeypatch.setattr(sys, "argv", ["rehearsal", *arguments])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2


class Template:
    fork = staticmethod(forbidden)
    snapshot_state = staticmethod(forbidden)
    restore_state = staticmethod(forbidden)
    train_batch = staticmethod(forbidden)
    predict = staticmethod(forbidden)


def test_should_allocate_fresh_four_slot_lists_without_operating_on_a_template():
    first = learnability.empty_learnability_observations()
    second = learnability.empty_learnability_observations()
    assert (
        first
        == second
        == [
            dict(updates=value, status="not_run", reason="checkpoint not reached", probe=None)
            for value in (0, 6, 12, 18)
        ]
    )
    assert first is not second and all(left is not right for left, right in zip(first, second))
    first[0]["status"] = "forged"
    assert second[0]["status"] == "not_run"


def test_should_forward_original_slots_template_sampler_and_exact_extended_data_to_the_existing_observer(
    monkeypatch,
):
    template = Template()
    sampler: Any = object()
    observations = learnability.empty_learnability_observations()
    training, evaluation, reference, returned = object(), object(), object(), object()
    calls = []

    def build(**kwargs):
        assert kwargs == {"extended": True}
        calls.append("data")
        return training, evaluation, reference

    def observer(*args, **kwargs):
        assert args == (template, training, evaluation, reference, observations, 17.5, sampler)
        assert kwargs == {"probe_checkpoints": (0, 1, 2, 4, 8, 16)}
        calls.append("observer")
        return returned

    monkeypatch.setattr(learnability, "build_reserved_data", build)
    monkeypatch.setattr(learnability, "_observer", observer)
    assert (
        learnability.create_sixteen_probe_observer(template, observations, 17.5, sampler)
        is returned
    )
    assert calls == ["data", "observer"]


@pytest.mark.parametrize(
    "fault",
    [
        "template",
        "tuple_slots",
        "short_slots",
        "row_type",
        "boolean_index",
        "boolean_deadline",
        "nan_deadline",
        "infinite_deadline",
    ],
)
def test_should_validate_public_facade_before_reserved_data_generation_or_template_callbacks(
    fault, monkeypatch
):
    template: Any = Template()
    observations: Any = learnability.empty_learnability_observations()
    deadline: Any = 10.0
    if fault == "template":
        template = object()
    elif fault == "tuple_slots":
        observations = tuple(observations)
    elif fault == "short_slots":
        observations.pop()
    elif fault == "row_type":
        observations[0] = object()
    elif fault == "boolean_index":
        observations[0]["updates"] = False
    else:
        deadline = {
            "boolean_deadline": True,
            "nan_deadline": float("nan"),
            "infinite_deadline": float("inf"),
        }[fault]
    monkeypatch.setattr(learnability, "build_reserved_data", forbidden)
    monkeypatch.setattr(learnability, "_observer", forbidden)
    with pytest.raises(ValueError):
        learnability.create_sixteen_probe_observer(template, observations, deadline)


class FakeLearner:
    def __init__(self, control, method="single", origin="template", state=0.0, *, original=True):
        self.control, self.method, self.origin, self.original = control, method, origin, original
        self.state = np.array([float(state)])
        self.start_state, self.trained = float(state), 0
        control["models"].append(self)

    def fork(self):
        self.control["trace"].append((self.method, self.origin, "fork", float(self.state[0])))
        return FakeLearner(self.control, self.method, self.origin, self.state[0], original=False)

    def snapshot_state(self):
        return self.state.copy()

    def restore_state(self, state):
        self.state = state.copy()
        self.start_state, self.trained = float(state[0]), 0
        self.control["trace"].append((self.method, self.origin, "restore", self.start_state))
        if self.control.get("mutate_arguments"):
            state.fill(777)

    def train_batch(self, features, targets):
        self.trained += 1
        self.control["trace"].append(
            (self.method, self.origin, "train", len(features), features.copy(), targets.copy())
        )
        self.state += 1
        if (
            self.origin == "template"
            and self.start_state == 10
            and self.trained == 3
            and self.control.get("fail_probe")
        ):
            raise RuntimeError("declared probe mutation then failure")
        if self.control.get("mutate_arguments"):
            features.fill(999)
            targets.fill(0.5)
        return TrainingDiagnostic("fake_probe_count", float(self.state[0]))

    def predict(self, features):
        self.control["trace"].append(
            (self.method, self.origin, "predict", float(self.state[0]), features.copy())
        )
        result = np.full((len(features), 1), min(0.95, (self.state[0] + 1) / 100))
        if self.control.get("mutate_arguments"):
            features.fill(888)
        return result


def fake_environment(monkeypatch, **faults):
    control: dict[str, Any] = dict(
        trace=[], models=[], sources=[], probes=[], reserved=[], streams=[], features=[]
    )
    control.update(faults)
    for target in (pilot, learnability):
        monkeypatch.setattr(target, "monotonic", lambda: 0.0)
        monkeypatch.setattr(target, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(budget_module, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(learnability, "feedforward_latents", forbidden)
    monkeypatch.setattr(learnability, "diagnose_activations", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    build_data, build_stream = learnability.build_reserved_data, pilot.build_stream

    def data(**kwargs):
        assert kwargs == {"extended": True}
        result = build_data(**kwargs)
        control["reserved"].append((result, pickle.dumps(result)))
        return result

    def stream(name):
        result = build_stream(name)
        control["streams"].append((result, pickle.dumps(result)))
        return result

    def build(method, initial):
        origin = (
            "benchmark"
            if sum(model.method == method for model in control["sources"]) == 0
            else "template"
        )
        model = FakeLearner(control, method, origin)
        control["sources"].append(model)
        return model

    def measure(state, reference, row):
        control["features"].append((float(state[0]), reference.copy()))
        row.update(
            feature_health=[{"fixture_layer": 0}, {"fixture_layer": 1}],
            diagnostic_seconds=0.0,
            diagnostic_activation_bytes=0,
            checkpoint_snapshot_array_bytes=state.nbytes,
            importance=None,
            neuron_age=None,
            unavailable_reason="fake feature measurement; native importance and age unavailable",
        )

    def probe(template, state, training, evaluation, **kwargs):
        control["probes"].append(
            (template, float(state[0]), tuple(batch.row_ids for batch in training))
        )
        return run_fresh_task_probe(
            template, state, training, evaluation, timer=lambda: 0.0, **kwargs
        )

    monkeypatch.setattr(learnability, "build_reserved_data", data)
    monkeypatch.setattr(pilot, "build_stream", stream)
    monkeypatch.setattr(pilot, "build_learner", build)
    monkeypatch.setattr(learnability, "_measure_features", measure)
    monkeypatch.setattr(learnability, "run_fresh_task_probe", probe)
    return control


def test_should_use_real_probe_clones_and_detached_reserved_arrays_without_mutating_checkpoint_or_template(
    monkeypatch,
):
    control = fake_environment(monkeypatch, mutate_arguments=True)
    template = FakeLearner(control)
    state = np.array([10.0])
    observations = learnability.empty_learnability_observations()
    callback = learnability.create_sixteen_probe_observer(template, observations, 60.0)
    before = pickle.dumps(state)
    callback(6, state, None)
    assert pickle.dumps(state) == before and template.state[0] == 0.0
    probe = observations[1]["probe"]
    assert observations[1]["status"] == "complete"
    assert probe["attempted_updates"] == probe["completed_updates"] == 16
    assert [row["updates"] for row in probe["checkpoints"]] == [0, 1, 2, 4, 8, 16]
    assert probe["reserved_data_bytes"] == 3840
    assert len(control["models"]) == 2 and control["models"][1] is not template
    assert control["models"][1].state[0] == 26
    assert all(pickle.dumps(data) == before for data, before in control["reserved"])
    assert (
        observations[0]["status"]
        == observations[2]["status"]
        == observations[3]["status"]
        == "not_run"
    )


def test_should_retain_benchmark_work_and_post_rehearsal_states_while_probes_use_independent_templates(
    monkeypatch,
):
    control = fake_environment(monkeypatch, mutate_arguments=True)
    report = pilot.run_pilot(diagnose_learnability=True)
    assert report["status"] == "complete", [row.get("reason") for row in report["runs"]]
    assert report["wake_updates"] == 72 and report["replay_charged_examples"] == 48
    assert report["row_exposures_charged"] == 624
    assert report["probe_attempted_updates"] == report["probe_completed_updates"] == 256
    assert report["probe_training_row_exposures"] == 2048
    assert (
        report["probe_score_forward_calls_attempted"]
        == report["probe_score_forward_calls_completed"]
        == 96
    )
    assert (
        report["total_completed_updates"] == 376 and report["total_training_row_exposures"] == 2672
    )
    assert (
        report["diagnostic_feedforward_passes_completed"] == 16
        and report["diagnostic_svd_calls_completed"] == 32
    )
    assert report["pc_neutral_exact"] is report["pc_neutral_probe_curves_exact"] is True
    assert (
        report["full_R21_complete"]
        is report["full_R23_complete"]
        is report["scientific_advantage_verified"]
        is False
    )
    assert len(control["probes"]) == 16 and len(control["sources"]) == 8
    for method, row in zip(pilot.METHODS, report["runs"], strict=True):
        sources = [model for model in control["sources"] if model.method == method]
        assert len(sources) == 2 and sources[0] is not sources[1]
        assert all(model.state[0] == 0 for model in sources)
        assert row["source_learner_unchanged"] is row["probe_template_unchanged"] is True
        probes = [entry for entry in control["probes"] if entry[0].method == method]
        assert [state for _, state, _ in probes] == [0.0, 10.0, 20.0, 30.0]
        assert all(template is sources[1] for template, _, _ in probes)
        assert [entry["updates"] for entry in row["observations"]] == [0, 6, 12, 18]
        assert all(entry["status"] == "complete" for entry in row["observations"])
        assert all(
            entry["importance"] is entry["neuron_age"] is None for entry in row["observations"]
        )
        events = [event for event in control["trace"] if event[0] == method]
        benchmark = [event for event in events if event[1] == "benchmark" and event[2] == "train"]
        reserved = [event for event in events if event[1] == "template" and event[2] == "train"]
        assert len(benchmark) == 30 and len(reserved) == 64
        assert [event[3] for event in benchmark].count(8) == 18
        assert [event[3] for event in benchmark].count(1) == 12
        first_final = min(
            index for index, event in enumerate(events) if event[1:3] == ("benchmark", "predict")
        )
        assert first_final > max(
            index
            for index, event in enumerate(events)
            if event[1] == "template" and event[2] in ("train", "predict")
        )
        training, _ = control["streams"][pilot.METHODS.index(method)][0]
        for event in benchmark:
            if event[3] == 8:
                assert any(
                    np.array_equal(event[4], batch.features)
                    and np.array_equal(event[5], batch.targets)
                    for batch in training
                )
            else:
                assert any(
                    np.array_equal(event[4], batch.features[index : index + 1])
                    and np.array_equal(event[5], batch.targets[index : index + 1])
                    for batch in training
                    for index in range(8)
                )
        assert (
            len(row["stream_result"]["applied_updates"]) == 18
            and len(row["stream_result"]["applied_consolidations"]) == 6
        )
    assert all(
        pickle.dumps(data) == before for data, before in (*control["reserved"], *control["streams"])
    )


def test_should_keep_partial_probe_work_all_slots_and_no_benchmark_scores_when_a_probe_fails(
    monkeypatch,
):
    control = fake_environment(monkeypatch, fail_probe=True)
    report = pilot.run_pilot(diagnose_learnability=True)
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 4 and all(row["status"] == "failed" for row in report["runs"])
    assert report["wake_updates"] == 24 and report["replay_charged_examples"] == 16
    assert report["probe_attempted_updates"] == 76 and report["probe_completed_updates"] == 72
    for row in report["runs"]:
        assert [entry["status"] for entry in row["observations"]] == [
            "complete",
            "failed",
            "not_run",
            "not_run",
        ]
        failed = row["observations"][1]["probe"]
        assert (
            failed["status"] == "failed"
            and failed["attempted_updates"] == 3
            and failed["completed_updates"] == 2
        )
        assert "declared probe mutation" in failed["reason"]
        assert "stream_result" not in row
    assert not any(event[1:3] == ("benchmark", "predict") for event in control["trace"])
    assert all(source.state[0] == 0 for source in control["sources"])


def test_should_preallocate_all_sixteen_observation_slots_on_sampler_setup_failure(monkeypatch):
    def fail():
        raise RuntimeError("declared sampler setup failure")

    monkeypatch.setattr(pilot, "ProcessRssSampler", fail)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", forbidden)
    monkeypatch.setattr(pilot, "build_learner", forbidden)
    report = pilot.run_pilot(diagnose_learnability=True)
    assert report["status"] == "incomplete" and report["probe_attempted_updates"] == 0
    assert len(report["runs"]) == 4
    assert all(
        row["status"] == "not_run" and len(row["observations"]) == 4 for row in report["runs"]
    )
    assert all(
        entry["status"] == "not_run" and entry["probe"] is None
        for row in report["runs"]
        for entry in row["observations"]
    )


def test_should_preserve_default_run_dispatch_without_passing_new_keywords_to_old_private_hooks(
    monkeypatch,
):
    calls = []

    def old_one(row, initial, deadline, sampler):
        calls.append(row["method"])
        row.update(status="failed", reason="explicit old hook")

    monkeypatch.setattr(pilot, "ProcessRssSampler", FakeSampler)
    monkeypatch.setattr(pilot, "CoupledPredictiveCodingNetwork", FakeModel)
    monkeypatch.setattr(pilot, "monotonic", lambda: 0.0)
    monkeypatch.setattr(pilot, "_run_one", old_one)
    monkeypatch.setattr(learnability, "create_sixteen_probe_observer", forbidden)
    first = pilot.run_pilot()
    second = pilot.run_pilot(diagnose_learnability=False)
    assert first == second and calls == list(pilot.METHODS) * 2
    assert all("observations" not in row for row in first["runs"])
    assert "probe_completed_updates" not in first and first["protocol"] == RETAINED_PROTOCOL
