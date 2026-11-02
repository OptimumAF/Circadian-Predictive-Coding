"""Literal recorded traces only: no models, training data or physics execution."""

from copy import deepcopy
import json

import pytest

from src.app.recurrent_readout_inspector import DEFINITION, METHOD, LAYOUTS, POINTS
from src.app.key_door_inspector import project_key_door_report
from src.app.minigrid_inspector_projection import _episode
from src.infra.key_door_inspector_files import render_key_door_inspector, write_key_door_inspector


def at(report, pointer):
    for key in pointer.split("/")[1:]:
        report = report[int(key)] if type(report) is list else report[key]
    return report


def episode(point, context, cue, success):
    observation = [-1.0] * 148
    steps = 1 if success else 16
    transitions = [
        dict(
            observation=observation,
            action=0,
            attempted=True,
            applied=True,
            next_observation=observation,
            reward=0.5 if success else 0.0,
            terminated=success,
            truncated=not success and i == steps - 1,
        )
        for i in range(steps)
    ]
    environment = dict(
        environment_id="Circadian-MiniGrid-CueKeyDoor-v1",
        encoding_id="minigrid_cue_key_door_public_image_direction_148_v1",
        role="train",
        episode_id=f"live-{METHOD}-{point}-{context}-{cue}",
        context=context,
        cue=cue,
        observation_dim=148,
        num_classes=7,
        seed=17,
        max_steps=16,
        action_names=["left", "right", "forward", "pickup", "drop", "toggle", "done"],
        status="complete",
        closed=True,
        reset_calls=1,
        environment_steps_attempted=steps,
        environment_steps=steps,
        transitions=transitions,
        terminated=success,
        truncated=not success,
        total_return=0.5 if success else 0.0,
        layout=deepcopy(LAYOUTS[context]),
    )
    return dict(
        context=context,
        cue=cue,
        source_role="train",
        status="complete",
        reason=None,
        predictions=[
            dict(observation=observation, probabilities=[1.0] + [0.0] * 6, action=0)
            for _ in range(steps)
        ],
        prediction_calls_attempted=steps,
        prediction_calls_returned=steps,
        environment=environment,
        state_unchanged=True,
        success=success,
        **{"return": environment["total_return"]},
    )


def budget(readout=False):
    return dict(
        limits=dict(
            max_training_updates=0 if readout else 8,
            max_wall_seconds=90 if readout else 60,
            max_process_rss_bytes=512 * 1024**2,
            max_replay_examples=0 if readout else None,
        ),
        birth=10.0,
        last_clock=12.0,
        updates=0 if readout else 8,
        replay=0,
    )


def report():
    receipts = [
        dict(
            update_number=i + 1,
            observed_at=i,
            arrived_at=i + 1,
            applied_at=i + 1,
            actor_version="a0",
            learner_version="c0",
        )
        for i in range(8)
    ]
    readouts = []
    acquired, retained = [], []
    for point, current in zip(POINTS, ("A", "B", "C_train", "A"), strict=True):
        rows = [
            episode(point, c, cue, c == current)
            for c in tuple(LAYOUTS)[: min(point, 6) // 2]
            for cue in ("blue", "red")
        ]
        readouts.append(
            dict(
                updates=point,
                training_receipts=deepcopy(receipts[:point]),
                games=dict(
                    method=METHOD,
                    role="train",
                    updates=point,
                    readout_fit_updates=point,
                    readout_fit_rows=9 * point,
                    native_recurrent_updates=0,
                    native_recurrent_training_rows=0,
                    arrived_contexts=list(LAYOUTS)[: min(point, 6) // 2],
                    status="complete",
                    state_unchanged=True,
                    episodes=rows,
                    scores=[
                        dict(
                            context=e["context"],
                            cue=e["cue"],
                            ce=0.0,
                            accuracy=1.0,
                            exact_sequence_accuracy=1.0,
                        )
                        for e in rows
                    ],
                ),
            )
        )
        for row in rows:
            s = dict(updates=point, context=row["context"], cue=row["cue"], success=row["success"])
            if point < 8 and row["context"] == current:
                acquired.append(s)
            elif row["context"] != current:
                retained.append(s)
    return dict(
        protocol=dict(
            definition=DEFINITION,
            context_sequence=["A", "B", "C_train", "A"],
            fit_points=list(POINTS),
            readout_contexts=[["A"], ["A", "B"], list(LAYOUTS), list(LAYOUTS)],
            final_available=False,
            probe_available=False,
            limits=dict(
                body_seconds=120, training_seconds=60, readout_seconds=90, rss_bytes=512 * 1024**2
            ),
        ),
        status="complete",
        reason=None,
        seconds=3.0,
        peak_bytes=1024,
        preservation=dict(source=True, actor=True, torch_rng=True, numpy_rng=True, python_rng=True),
        training=dict(
            completed_updates=8,
            training_examples=72,
            applied_updates=[
                dict(
                    update_number=i + 1,
                    observed_at=i,
                    arrived_at=i + 1,
                    applied_at=i + 1,
                    actor_version="a0",
                    learner_version="c0",
                )
                for i in range(8)
            ],
        ),
        training_budget=budget(),
        readout_budget=budget(True),
        readouts=readouts,
        outcomes=dict(
            acquisition=acquired,
            retention=retained,
            returning_A=[
                dict(
                    cue=cue,
                    initially_acquired=True,
                    failed_while_absent=True,
                    failed_before_return=True,
                    return_success=True,
                    observed_recovery_by_return=True,
                    observed_recovery=True,
                )
                for cue in ("blue", "red")
            ],
        ),
    )


def test_should_play_all_slots_preserve_negative_results_and_link_original_source():
    source = report()
    saved = deepcopy(source)
    result = project_key_door_report(source, source_label="literal")
    assert result["schema_id"] == "recurrent_readout_inspector_v1"
    assert result["source"] == saved and result["source"] is not source
    entries = result["conditions"][0]["episodes"]
    assert len(entries) == 18
    assert sum(e["record"]["success"] is False for e in entries) == 10
    for e in entries:
        assert at(source, e["pointer"]) == e["record"]
        for frame in e["frames"]:
            assert at(source, frame["pointer"]) == frame["observation"]
            assert frame["features_pointer"] is None
    result["source"]["readouts"].clear()
    assert source == saved


@pytest.mark.parametrize(
    "key,value",
    [
        ("preservation", None),
        (
            "preservation",
            dict(source=False, actor=True, torch_rng=True, numpy_rng=True, python_rng=True),
        ),
        ("peak_bytes", True),
        ("seconds", float("nan")),
        ("seconds", 121.0),
        ("outcomes", None),
        ("readouts", []),
        ("training_budget", None),
    ],
)
def test_should_refuse_false_complete_claims(key, value):
    source = report()
    source[key] = value
    with pytest.raises(ValueError):
        project_key_door_report(source, source_label="literal")


@pytest.mark.parametrize(
    "fault",
    [
        "role",
        "action",
        "probability",
        "features",
        "context",
        "seed",
        "closed",
        "history",
        "cue",
        "score",
        "receipt",
        "replay",
        "clock",
        "exposures",
        "phase_order",
        "ending",
        "source_role",
    ],
)
def test_should_refuse_contradictory_native_or_summary_evidence(fault):
    s = report()
    e = s["readouts"][0]["games"]["episodes"][0]
    if fault == "role":
        e["environment"]["role"] = "final_test"
    elif fault == "action":
        e["predictions"][0]["action"] = 1
    elif fault == "probability":
        e["predictions"][0]["probabilities"][0] = 2.0
    elif fault == "features":
        e["predictions"][0]["features"] = [-1.0] * 296
    elif fault == "context":
        e["context"] = "B"
    elif fault == "seed":
        e["environment"]["seed"] = True
    elif fault == "closed":
        e["environment"]["closed"] = False
    elif fault == "history":
        s["outcomes"]["returning_A"][0]["failed_before_return"] = False
    elif fault == "cue":
        s["outcomes"]["retention"][0]["cue"] = "red"
    elif fault == "score":
        s["readouts"][0]["games"]["scores"][0]["accuracy"] = 2
    elif fault == "receipt":
        s["training"]["applied_updates"][0]["arrived_at"] = 0
    elif fault == "replay":
        s["readout_budget"]["replay"] = 1
    elif fault == "clock":
        s["readout_budget"]["last_clock"] = 9.0
    elif fault == "exposures":
        s["training"]["training_examples"] = 71
    elif fault == "phase_order":
        s["readouts"].reverse()
    elif fault == "ending":
        e["environment"]["transitions"][0]["terminated"] = False
    elif fault == "source_role":
        e["source_role"] = "probe_test"
    with pytest.raises(ValueError):
        project_key_door_report(s, source_label="literal")


def test_should_keep_missing_slots_unknown_and_false_global_proofs_visible():
    s = report()
    s.update(
        status="incomplete",
        outcomes=None,
        preservation=dict(
            source=False, actor=None, torch_rng=True, numpy_rng=True, python_rng=True
        ),
        readouts=s["readouts"][:1],
    )
    r = project_key_door_report(s, source_label="literal")
    entries = r["conditions"][0]["episodes"]
    assert len(entries) == 18 and sum(e["derived"] for e in entries) == 16
    assert all(e["record"]["success"] is None for e in entries[2:])
    assert r["source"]["preservation"]["source"] is False
    assert r["source"]["preservation"]["actor"] is None


def test_should_retain_literal_failed_partial_episode_and_known_paid_transition():
    s = report()
    s.update(status="incomplete", outcomes=None, readouts=s["readouts"][:1])
    del s["readouts"][0]["training_receipts"]
    s["readouts"][0]["status"] = "incomplete"
    g = s["readouts"][0]["games"]
    g.update(status="failed", state_unchanged=False)
    e = g["episodes"][0]
    e.update(status="failed", state_unchanged=False)
    r = project_key_door_report(s, source_label="literal")
    actual = r["conditions"][0]["episodes"][0]
    assert actual["record"]["state_unchanged"] is False
    assert len(actual["frames"]) == 2 and actual["record"]["environment"]["environment_steps"] == 1
    assert "training_receipts" not in r["source"]["readouts"][0]


def test_should_require_prefix_for_complete_phase():
    s = report()
    del s["readouts"][0]["training_receipts"]
    with pytest.raises(ValueError):
        project_key_door_report(s, source_label="literal")


def test_should_preserve_feedforward_and_original_raw_recurrent_playback():
    e = episode(2, "A", "blue", True)
    for method in ("bp", "recurrent_bc"):
        literal = deepcopy(e)
        literal["environment"]["episode_id"] = f"live-{method}-2-A-blue"
        if method == "bp":
            literal["predictions"][0]["features"] = [-1.0] * 296
        result = _episode(literal, "/episode", method, 2, "blue")
        assert len(result["frames"]) == 2
        assert bool(result["frames"][0]["features_pointer"]) is (method == "bp")


def test_should_use_existing_safe_renderer_and_exclusive_writer(tmp_path):
    s = report()
    s["reason"] = "</script><script>alert(1)</script>"
    projection = project_key_door_report(s, source_label="literal")
    rendered = render_key_door_inspector(projection)
    assert b"Learn, forget and relearn" in rendered
    assert b"</script><script>alert(1)" not in rendered
    source, destination = tmp_path / "input.json", tmp_path / "output.html"
    source.write_text(json.dumps(s), encoding="utf8")
    write_key_door_inspector(source, destination)
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        write_key_door_inspector(source, destination)
    assert destination.read_bytes() == original


def test_should_refuse_separate_resources_and_unsupported_protocol():
    with pytest.raises(ValueError):
        project_key_door_report(report(), source_label="literal", resources={})
    s = report()
    s["protocol"]["final_available"] = True
    with pytest.raises(ValueError):
        project_key_door_report(s, source_label="literal")


@pytest.mark.parametrize(
    "fault",
    [
        "method",
        "role",
        "updates",
        "native_recurrent_updates",
        "arrived_contexts",
        "missing_scores",
        "receipt_prefix",
        "receipt_type",
        "false_complete_phase",
        "work_counter",
    ],
)
def test_should_refuse_conflicting_phase_identity_and_paid_prefix(fault):
    s = report()
    phase = s["readouts"][0]
    games = phase["games"]
    if fault in ("method", "role"):
        games[fault] = "unexpected"
    elif fault in ("updates", "native_recurrent_updates"):
        games[fault] = 1
    elif fault == "arrived_contexts":
        games["arrived_contexts"] = ["A", "B"]
    elif fault == "missing_scores":
        games["scores"] = []
    elif fault == "receipt_prefix":
        phase["training_receipts"][0]["sample_id"] = "replaced"
    elif fault == "receipt_type":
        phase["training_receipts"][0]["update_number"] = 1.0
    elif fault == "false_complete_phase":
        s.update(status="incomplete", outcomes=None)
        games["episodes"][0]["status"] = "failed"
    elif fault == "work_counter":
        games["work"] = dict(environment_steps_returned=False)
    with pytest.raises(ValueError):
        project_key_door_report(s, source_label="literal")


@pytest.mark.parametrize("point", [2.0, True])
def test_should_refuse_protocol_fit_point_types(point):
    s = report()
    s["protocol"]["fit_points"][0] = point
    with pytest.raises(ValueError):
        project_key_door_report(s, source_label="literal")
