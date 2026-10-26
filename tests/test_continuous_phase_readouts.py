"""Detached, bounded phase observations on the existing delayed TRAIN runner.

Six cached D1/H64/A1 streams: six initializers/768 normal draws and A fits;
eight SGD rows, three GN solves/198 augmented rows and three active CPC updates.
No author execution; exact prospective costs live in the stage fixture ledger.
"""

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.adapters import continuous_learnability_demo as demo
from src.adapters import continuous_phase_readouts as phases
from src.adapters import numpy_continuous_pc as pc
from src.core.continuous_demonstration import (
    ContinuousActionSpec,
    ContinuousEpisode,
    ContinuousTransition,
)


def episode(name="A"):
    return ContinuousEpisode(
        "phase-source-" + name,
        "phase-context-" + name,
        "train",
        ContinuousActionSpec(1, ("normalized_gripper",), (-1.0,), (1.0,)),
        (
            ContinuousTransition(
                0, (0.0,), (0.25 if name == "A" else -0.25,), 1.0, (0.0,), True, False
            ),
        ),
    )


def assert_state(first, second):
    assert type(first) is type(second)
    assert first.policy == second.policy
    assert first.completed_updates == second.completed_updates
    assert first.training_examples == second.training_examples
    for a, b in zip(
        first.parameters + first.velocity, second.parameters + second.velocity, strict=True
    ):
        np.testing.assert_array_equal(a, b)
    if isinstance(first, pc.ContinuousPCSnapshot):
        assert isinstance(second, pc.ContinuousPCSnapshot)
        assert first.wake_state is not None and second.wake_state is not None
        for key in (
            "wake_updates",
            "chemical_decay",
            "chemical_buildup_rate",
            "plasticity_sensitivity",
            "min_plasticity",
        ):
            assert getattr(first.wake_state, key) == getattr(second.wake_state, key)
        for key in ("chemical_state", "plasticity_state"):
            np.testing.assert_array_equal(
                getattr(first.wake_state, key), getattr(second.wake_state, key)
            )


@pytest.fixture(scope="module")
def streams():
    saved = {}
    factory = demo.ActorShadowRuntime
    for mode in ("bp", "pc", "head", "default", "failed", "corrupt"):
        warm, new = episode(), episode("B")
        original = asdict(warm), asdict(new)
        captured: dict[str, Any] = dict(calls=[], readouts=[], states=[], results=[])

        def runtime(source, **options):
            made = factory(source, **options)
            captured.update(
                source=source,
                runtime=made,
                initial=source.snapshot_state(),
                budget=options["budget"],
                clock=options["clock"],
            )
            return made

        def observer(readout, metadata):
            captured["calls"].append(deepcopy(metadata))
            captured["readouts"].append(readout)
            captured["states"].append(readout.snapshot_state())
            if mode == "failed" and metadata["checkpoint_index"] == 1:
                raise RuntimeError("warm observer failed after its paid update")
            if mode == "corrupt" and metadata["checkpoint_index"] == 1:
                candidate = captured["runtime"]._candidate
                captured["mutated_owner"] = candidate
                captured["before_corruption"] = candidate.snapshot_state()
                candidate._state.parameters[3][0] += 0.125
            result = {"observed_updates": metadata["updates"], "nested": {"caller": "owned"}}
            if mode == "bp":
                result["prediction"] = readout.predict(np.zeros((1, 1))).tolist()
            captured["results"].append(result)
            return result

        options: dict[str, Any] = dict(phase_updates=1)
        if mode != "default":
            options["checkpoint_observer"] = observer
        if mode == "pc":
            options["learner_family"] = "active-cpc"
        if mode == "head":
            options["head_gauss_newton"] = True
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(demo, "ActorShadowRuntime", runtime)
            result = demo.run_continuous_learnability(warm, new, **options)
        assert (asdict(warm), asdict(new)) == original
        saved[mode] = result, captured
    return saved


@pytest.mark.parametrize("mode", ["bp", "pc", "head"])
def test_should_observe_four_actual_phases_after_their_delayed_updates(mode, streams):
    result, capture = streams[mode]
    assert result.report["complete"] is True and result.candidate is not None
    assert capture["calls"] == [
        dict(checkpoint_index=i, phase=phase, updates=i, training_examples=i)
        for i, phase in enumerate(("initial", "warm", "new", "warm_return"))
    ]
    report = result.report["phase_readouts"]
    assert report["attempts"] == report["returns"] == report["validated"] == report["forks"] == 4
    assert [row["metadata"] for row in report["records"]] == capture["calls"]
    assert [row["result"]["observed_updates"] for row in report["records"]] == [0, 1, 2, 3]
    assert len({id(readout) for readout in capture["readouts"]}) == 4
    assert capture["budget"].updates_completed == 3
    assert [state.completed_updates for state in capture["states"]] == [0, 1, 2, 3]
    assert [state.training_examples for state in capture["states"]] == [0, 1, 2, 3]
    if mode == "pc":
        assert [state.wake_state.wake_updates for state in capture["states"]] == [0, 1, 2, 3]


@pytest.mark.parametrize("mode", ["bp", "pc", "head"])
def test_should_keep_earlier_readouts_and_all_original_owners_unchanged(mode, streams):
    result, capture = streams[mode]
    for readout, pinned in zip(capture["readouts"], capture["states"], strict=True):
        assert type(readout) is type(result.candidate)
        assert_state(readout.snapshot_state(), pinned)
        assert readout is not result.candidate and readout is not capture["source"]
    assert_state(capture["source"].snapshot_state(), capture["initial"])
    assert_state(capture["runtime"].actor.snapshot_state().state, capture["initial"])
    assert_state(result.candidate.snapshot_state(), capture["states"][-1])
    assert all(
        row["state_exact"] is row["owners_exact"] is True for row in result.report["checkpoints"]
    )


def test_should_own_returned_callback_json_and_keep_prediction_cost_separate(streams):
    result, capture = streams["bp"]
    readouts = result.report["phase_readouts"]
    before = deepcopy(readouts)
    for returned in capture["results"]:
        returned["nested"]["caller"] = "changed outside the report"
    assert readouts == before
    assert len(capture["results"]) == 4
    assert all(len(row["result"]["prediction"]) == 1 for row in readouts["records"])
    assert readouts["work"]["unknown_callback_work"] is True
    assert result.report["work"]["optimizer_updates"] == 3
    assert result.report["work"]["conditioning_calls"] == 8


def test_should_keep_a_paid_phase_when_the_next_observer_raises(streams):
    result, capture = streams["failed"]
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert "warm observer failed" in report["error"]
    assert capture["budget"].updates_completed == 1
    assert len(report["applied_updates"]) == len(report["events"]) == 1
    assert report["events"][0]["success_validated"] is True
    observed = report["phase_readouts"]
    assert observed["attempts"] == observed["forks"] == 2
    assert observed["returns"] == observed["validated"] == 1
    assert len(observed["records"]) == 2
    assert observed["records"][0]["result"]["observed_updates"] == 0
    assert observed["records"][1]["error"]
    assert_state(capture["source"].snapshot_state(), capture["initial"])
    assert_state(capture["runtime"].actor.snapshot_state().state, capture["initial"])


def test_should_leave_the_default_schema_without_observer_metadata(streams):
    result, capture = streams["default"]
    assert result.report["complete"] is True and result.candidate is not None
    assert "phase_readouts" not in result.report and capture["calls"] == []
    assert result.report["work"]["optimizer_updates"] == 3
    assert capture["budget"].updates_completed == 3


def test_should_refuse_a_callback_that_changes_the_private_candidate_after_a_paid_phase(streams):
    result, capture = streams["corrupt"]
    report = result.report
    assert report["complete"] is False and result.candidate is None
    assert report["error"] and capture["budget"].updates_completed == 1
    assert len(report["events"]) == len(report["applied_updates"]) == 1
    assert report["events"][0]["success_validated"] is True
    observed = report["phase_readouts"]
    assert observed["attempts"] == observed["returns"] == 2
    assert observed["validated"] == 1
    assert observed["records"][-1]["result"]["observed_updates"] == 1
    assert observed["records"][-1]["error"]
    before = capture["before_corruption"]
    after = capture["mutated_owner"].snapshot_state()
    assert after.completed_updates == before.completed_updates == 1
    assert after.parameters[3][0] == before.parameters[3][0] + 0.125
    assert_state(capture["source"].snapshot_state(), capture["initial"])
    assert_state(capture["runtime"].actor.snapshot_state().state, capture["initial"])


@pytest.mark.parametrize(
    "target,key",
    [
        ("runtime", "_candidate_version"),
        ("runtime", "_base_actor_version"),
        ("runtime", "_training_role"),
        ("inbox", "_learner_version"),
        ("inbox", "_capacity"),
        ("port", "_reports"),
    ],
)
def test_should_refuse_changed_release_authority_before_more_work(target, key, streams):
    result, capture = streams["pc"]
    runtime = capture["runtime"]
    source = demo._phase_native(capture["source"])
    guard = demo._phase_callback_guard(
        source,
        runtime,
        result.candidate,
        capture["initial"],
        result.candidate.snapshot_state(),
        capture["budget"],
        capture["clock"],
        result.report,
    )
    guard()
    owner = {"runtime": runtime, "inbox": runtime._inbox, "port": runtime._candidate}[target]
    existed, original = key in vars(owner), getattr(owner, key)
    replacement = list(original) if key == "_reports" else 999 if key == "_capacity" else "changed"
    try:
        setattr(owner, key, replacement)
        with pytest.raises(RuntimeError, match="authority"):
            guard()
        assert capture["budget"].updates_completed == 3
    finally:
        if existed:
            setattr(owner, key, original)
        else:
            vars(owner).pop(key, None)
    guard()


@pytest.mark.parametrize(
    "fault", ["metadata", "nonfinite", "not_dict", "oversize", "readout", "guard"]
)
def test_should_refuse_an_invalid_callback_without_changing_the_original_model(fault, streams):
    _, capture = streams["bp"]
    original = capture["source"].snapshot_state()
    native = (
        capture["source"]._learner if hasattr(capture["source"], "_learner") else capture["source"]
    )
    readout = native.fork()
    returned = {"kept_return": True}

    def callback(detached, metadata):
        if fault == "metadata":
            metadata["updates"] = 999
        elif fault == "readout":
            object.__setattr__(detached._state, "completed_updates", 1)
        if fault == "nonfinite":
            return {"value": float("nan")}
        if fault == "not_dict":
            return ["not a bounded report"]
        if fault == "oversize":
            return {"large": "x" * 65536}
        return returned

    calls = []

    def guard():
        calls.append(True)
        if fault == "guard" and len(calls) >= 3:
            raise RuntimeError("benchmark owner changed")

    observed = phases.PhaseReadouts(callback)
    with pytest.raises((ValueError, RuntimeError)):
        observed.observe(
            readout,
            dict(checkpoint_index=0, phase="initial", updates=0, training_examples=0),
            guard,
        )
    report = observed.report
    assert report["attempts"] == report["returns"] == 1
    assert report["validated"] == 0 and report["records"][-1]["error"]
    if fault in ("metadata", "readout", "guard"):
        assert report["records"][-1]["result"] == returned
    assert_state(native.snapshot_state(), original)
    report["records"][0]["metadata"]["updates"] = 999
    assert observed.report["records"][0]["metadata"]["updates"] == 0
    with pytest.raises(RuntimeError, match="cannot resume"):
        observed.observe(
            readout,
            dict(checkpoint_index=0, phase="initial", updates=0, training_examples=0),
            guard,
        )


@pytest.mark.parametrize(
    "option",
    ["probe_plan", "replay_plan", "rgb_range_recipe", "stage_balanced_packets", "a_only_control"],
)
def test_should_refuse_unsupported_observer_modes_before_fit_or_construction(monkeypatch, option):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported observer mode reached numerical preparation")

    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    monkeypatch.setattr(demo, "NumpyContinuousBCLearner", forbidden)
    value = True if option in ("stage_balanced_packets", "a_only_control") else object()
    options: dict[str, Any] = {option: value}
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(
            episode(), episode("B"), checkpoint_observer=lambda *args: {}, **options
        )


@pytest.mark.parametrize("observer", [False, 1, "observer"])
def test_should_refuse_a_noncallable_observer_before_fit(monkeypatch, observer):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid observer reached fit")

    monkeypatch.setattr(demo, "fit_training_standardization", forbidden)
    with pytest.raises(ValueError):
        demo.run_continuous_learnability(episode(), episode("B"), checkpoint_observer=observer)


@pytest.mark.parametrize("selection", ["omitted", "none", "callback"])
def test_should_forward_only_a_real_observer_from_the_json_wrapper(monkeypatch, selection):
    calls = []
    returned = {"complete": True}

    def callback(*args):
        return {}

    def run(*args, **options):
        calls.append(options)
        return SimpleNamespace(report=returned)

    monkeypatch.setattr(demo, "run_continuous_learnability", run)
    options: dict[str, Any] = (
        {}
        if selection == "omitted"
        else {"checkpoint_observer": callback if selection == "callback" else None}
    )
    assert demo.run_continuous_learnability_demo(episode(), episode("B"), **options) is returned
    expected: dict[str, Any] = {"expected_standardization": None}
    if selection == "callback":
        expected["checkpoint_observer"] = callback
    assert calls == [expected]
