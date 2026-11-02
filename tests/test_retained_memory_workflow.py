"""NEW literal public-workflow cohort; no SDK, repository data or model RNG."""

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import run_taskspace_memory_demo as cli
from scripts import taskspace_memory_native as native
from scripts.push_cube_memory_control import _require_native
from src.adapters import taskspace_memory_demo as demo
from src.adapters.numpy_nearest_continuous_bc import NumpyNearestContinuousBCLearner
from src.adapters.numpy_retained_nearest_bc import NumpyRetainedNearestBCLearner
from src.core.replay_retention import ReplayRetentionPolicy
from test_retained_nearest_memory import _process_peak_bytes
from test_taskspace_memory_demo import bundle
from test_taskspace_memory_native import receipt

FIFO = ReplayRetentionPolicy("recent_fifo")
WORK = dict.fromkeys(
    (
        "initializers",
        "updates",
        "incoming_rows",
        "forks",
        "predictions",
        "query_rows",
        "distance_candidates",
        "feature_visits",
        "snapshots",
        "snapshot_bytes",
        "fidelity_rows",
        "fake_scenes",
        "fake_controls",
    ),
    0,
)
LIMITS = dict(
    initializers=160,
    updates=1024,
    incoming_rows=4096,
    forks=1024,
    predictions=512,
    query_rows=1024,
    distance_candidates=32768,
    feature_visits=393216,
    snapshots=65536,
    snapshot_bytes=256 * 1024**2,
    fidelity_rows=1024,
    fake_scenes=128,
    fake_controls=256,
)
PRIOR: dict[str, int] = {}


def charge(name, amount=1):
    WORK[name] += amount
    assert WORK[name] + PRIOR.get(name, 0) <= LIMITS[name], name


@pytest.fixture(scope="module", autouse=True)
def finite_work():
    PRIOR.update(json.loads(os.environ.get("RETAINED_WORKFLOW_PRIOR_WORK", "{}")))

    def wrap(method, kind):
        def metered(owner, *args, **kwargs):
            charge(kind)
            if kind == "updates":
                charge("incoming_rows", len(args[0]))
            if kind == "predictions":
                rows = len(args[0])
                charge("query_rows", rows)
                candidates = rows * len(owner._state.prototype_features)
                charge("distance_candidates", candidates)
                charge("feature_visits", candidates * 12)
            if kind == "snapshots":
                charge(
                    "snapshot_bytes",
                    owner._state.prototype_features.nbytes + owner._state.prototype_actions.nbytes,
                )
            return method(owner, *args, **kwargs)

        return metered

    fidelity = demo.continuous_command_fidelity

    def counted_fidelity(*args, **kwargs):
        charge("fidelity_rows", len(args[0]))
        return fidelity(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for cls in (NumpyNearestContinuousBCLearner, NumpyRetainedNearestBCLearner):
            for method, kind in (
                ("__init__", "initializers"),
                ("train_batch", "updates"),
                ("fork", "forks"),
                ("snapshot_state", "snapshots"),
                ("predict_with_work", "predictions"),
            ):
                if method in cls.__dict__:
                    patch.setattr(cls, method, wrap(cls.__dict__[method], kind))
        patch.setattr(demo, "continuous_command_fidelity", counted_fidelity)
        patch.setattr(np.random, "default_rng", forbidden)
        yield
    peak = _process_peak_bytes()
    target = os.environ.get("RETAINED_WORKFLOW_RECEIPT")
    if target:
        with Path(target).open("x", encoding="utf8") as handle:
            json.dump(
                dict(
                    work=WORK,
                    aggregate={k: v + PRIOR.get(k, 0) for k, v in WORK.items()},
                    actual_pytest_peak_bytes=peak,
                ),
                handle,
                indent=2,
            )
    assert peak <= 512 * 1024**2


def forbidden(*args, **kwargs):
    pytest.fail("refused options reached input/model/RNG work")


def distinct_bundle():
    value = bundle()
    for index, item in enumerate(value["training"]):
        row = item["episode"]["transitions"][0]
        row["observation"][0] = index / 8
        row["next_observation"][0] = index / 8
        if index >= 5:
            row["action"][0] = -0.125
            row["next_observation"][7] = -0.125
    return value


def test_retained_mismatch_is_scored_without_claiming_recorded_rewards():
    payload = distinct_bundle()
    original = deepcopy(payload)
    result = demo.run_demo(payload, retention=FIFO, max_prototypes=2)
    assert result["complete"], result["error"]
    assert result["mode"] == "offline_retained_training_readout"
    assert result["native_outcomes"] is None
    assert result["recorded_outcomes_scope"] == "original recordings, not this retained policy"
    assert result["work"]["distance_candidates_completed"] == 8
    assert all(
        item["selected_indices_exact"] is None
        and not item["native_actions_exact"]
        and item["command_fidelity"]["native_mismatch_rows"] == [0]
        for item in result["replays"]
    )
    assert result["stream"]["complete"] and payload == original


def test_reservoir_options_are_reproducible_without_rng():
    rule = ReplayRetentionPolicy("seeded_reservoir", 17)
    a = demo.run_demo(distinct_bundle(), retention=rule, max_prototypes=2)
    b = demo.run_demo(distinct_bundle(), retention=rule, max_prototypes=2)
    assert a["complete"] and b["complete"]
    assert a["retention"] == b["retention"] and a["replays"] == b["replays"]


def test_original_strict_mode_still_rejects_selector_mismatch():
    value = distinct_bundle()
    value["replay"][0]["selected_indices"] = [1]
    result = demo.run_demo(value)
    assert not result["complete"] and len(result["replays"]) == 1
    assert "selectors or nativefloat32" in result["replays"][0]["error"]


def test_original_strict_mode_still_rejects_command_mismatch():
    value = distinct_bundle()
    row = value["replay"][0]["episode"]["transitions"][0]
    row["action"][0] = row["next_observation"][7] = -0.125
    result = demo.run_demo(value)
    assert not result["complete"] and len(result["replays"]) == 1
    assert result["replays"][0]["selected_indices_exact"]
    assert not result["replays"][0]["native_actions_exact"]


@pytest.mark.parametrize("fault", ["readout", "preceding_prediction"])
def test_cpu_refuses_changed_frozen_or_preceding_evidence(monkeypatch, fault):
    predict = NumpyNearestContinuousBCLearner.predict_with_work
    paid: list[Any] = []

    def changed(owner, features):
        result = predict(owner, features)
        if fault == "readout":
            owner._state.prototype_actions[0, 0] += 0.25
        elif paid:
            paid[0].actions[0, 0] += 0.25
        paid.append(result)
        return result

    monkeypatch.setattr(NumpyNearestContinuousBCLearner, "predict_with_work", changed)
    result = demo.run_demo(distinct_bundle(), retention=FIFO, max_prototypes=2)
    assert not result["complete"] and len(paid) == (1 if fault == "readout" else 2)
    assert result["work"]["prediction_returns"] == len(paid)
    assert result["work"]["unknown_unvalidated_prediction_work"]
    assert len(result["replays"]) == len(paid)


@pytest.mark.parametrize(
    "options",
    [
        dict(max_prototypes=2),
        dict(max_prototypes=True),
        dict(retention=FIFO, max_prototypes=0),
        dict(retention=FIFO, max_prototypes=257),
        dict(retention=ReplayRetentionPolicy("content_hash")),
        dict(retention=FIFO, training_probe=object()),
    ],
)
def test_api_refuses_options_before_admission(monkeypatch, options):
    monkeypatch.setattr(demo, "_admit", forbidden)
    with pytest.raises(ValueError):
        demo.run_demo({}, **options)


@pytest.mark.parametrize(
    "options",
    [
        ["--max-prototypes", "2"],
        ["--retention-seed", "17"],
        ["--retention", "recent_fifo", "--retention-seed", "17"],
        ["--retention", "seeded_reservoir"],
        ["--retention", "seeded_reservoir", "--retention-seed", "-1"],
        ["--retention", "recent_fifo", "--max-prototypes", "0"],
        ["--retention", "recent_fifo", "--training-probe", "missing.json"],
        ["--retention", "recent_fifo", "--development-probe", "--native"],
    ],
)
def test_cli_refuses_before_loading(monkeypatch, options):
    monkeypatch.setattr(cli, "_load_default_bundle", forbidden)
    monkeypatch.setattr(cli, "load_bundle", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(options)
    assert error.value.code == 2


@pytest.mark.parametrize("retained", [False, True])
def test_cli_preserves_default_call_and_routes_explicit_options(monkeypatch, capsys, retained):
    payload = object()
    monkeypatch.setattr(cli, "_load_default_bundle", lambda: payload)
    calls: list[Any] = []

    def run(value, **opts):
        calls.append((value, opts))
        return dict(complete=True)

    monkeypatch.setattr(cli, "run_demo", run)
    options = ["--retention", "recent_fifo", "--max-prototypes", "2"] if retained else []
    assert cli.main(options) == 0
    assert calls == [(payload, dict(retention=FIFO, max_prototypes=2) if retained else {})]
    assert json.loads(capsys.readouterr().out)["complete"]


def test_cpu_late_io_preserves_paid_return(monkeypatch, tmp_path, capsys):
    output = tmp_path / "main.json"
    monkeypatch.setattr(cli, "_load_default_bundle", bundle)
    monkeypatch.setattr(cli, "run_demo", lambda *a, **k: dict(complete=True, paid="kept"))
    original = Path.open

    def opened(path, *args, **kwargs):
        if path == output:
            raise OSError("literal late IO")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    assert cli.main(["--retention", "recent_fifo", "--output", str(output)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["paid"] == "kept" and "literal late IO" in result["capture_error"]
    assert json.loads(output.with_suffix(".failure.json").read_text()) == result


def retained_receipt(seed, push=False):
    value = receipt(seed, push=push)
    value["nearest_prototype_rows"] = 2
    value["nearest_distance_candidate_upper_bound"] = 2
    value["nearest_distance_candidates_completed"] = 2
    value["nearest_distance_feature_upper_bound"] = 24
    value["nearest_distance_feature_values_completed"] = 24
    return value


@pytest.mark.parametrize("fault", [None, "exposure", "stored", "work", "boolean"])
def test_native_validator_keeps_exposure_and_storage_separate(fault):
    value = retained_receipt(17)
    if fault == "exposure":
        value["source_training_rows"] = 2
    elif fault == "stored":
        value["nearest_prototype_rows"] = 7
    elif fault == "work":
        value["nearest_distance_candidates_completed"] = 7
    elif fault == "boolean":
        value["nearest_prototype_rows"] = True
    if fault:
        with pytest.raises(ValueError):
            _require_native(value, push=False, updates=7, rows=7, prototype_rows=2)
    else:
        _require_native(value, push=False, updates=7, rows=7, prototype_rows=2)
        with pytest.raises(ValueError):
            _require_native(value, push=False, updates=7, rows=7)


@pytest.mark.parametrize(
    "fault", [None, "source", "prior_receipt", "capture", "lost_return", "current_capture_type"]
)
def test_native_public_mode_preserves_negative_scores_and_paid_failures(
    monkeypatch, tmp_path, fault
):
    output = tmp_path / "native.json"
    returned: list[Any] = []
    first_capture = []

    def endpoint(seed, learner, path=None):
        charge("fake_scenes")
        if len(returned) == 1 and fault == "lost_return":
            raise RuntimeError("literal unreturned endpoint")
        charge("fake_controls")
        value = retained_receipt(seed, push=path is None)
        # Exercise canonical JSON tuple/list capture comparison without an SDK.
        value["episode"]["transitions"][0]["observation"] = (0.0,) * 42
        if fault == "current_capture_type":
            value["nearest_selected_indices"] = [1]
        if path is not None:
            saved = deepcopy(value)
            if fault == "current_capture_type":
                saved["nearest_selected_indices"] = [True]
            path.write_text(json.dumps(saved), encoding="utf8")
            first_capture.append(path)
        if len(returned) == 1:
            if fault == "source":
                learner._state.prototype_actions[0, 0] += 0.25
            elif fault == "prior_receipt":
                returned[0]["source_training_rows"] = 2
            elif fault == "capture":
                first_capture[0].write_text("{}", encoding="utf8")
        returned.append(value)
        return value

    monkeypatch.setattr(
        native,
        "run_frozen_rollout",
        lambda path, *, learner, train_seed, **k: SimpleNamespace(
            report=endpoint(train_seed, learner, path), corrections=None
        ),
    )
    monkeypatch.setattr(native, "roll_push", lambda learner, *, seed, **k: endpoint(seed, learner))
    result = native.run_native_demo(distinct_bundle(), output, retention=FIFO, max_prototypes=2)
    if fault is None:
        assert result["complete"], result["error"]
        assert len(result["native"]) == len(result["outcomes"]) == 4
        assert all(not item["final_success"] for item in result["outcomes"].values())
        assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    else:
        assert not result["complete"] and len(returned) <= 2
        assert result["native"][0]["report"]["source_training_rows"] == 7
        if fault == "lost_return":
            assert result["unreturned_stage_work_unknown"]
        elif fault == "current_capture_type":
            assert len(returned) == 1 and "saved/returned" in result["error"]
        else:
            assert result.get("integrity_error")
    assert result["mode"] == "retained_native_training_rollout"


def test_native_conflict_is_refused_before_io_and_data(monkeypatch, tmp_path):
    monkeypatch.setattr(native, "_admit", forbidden)
    with pytest.raises(ValueError, match="incompatible"):
        native.run_native_demo(
            {},
            tmp_path / "missing" / "x.json",
            retention=FIFO,
            development_scenes=(native.NativeScene("push", 19),),
        )


@pytest.mark.parametrize("fault", ["metadata", "cost", "identity"])
def test_native_refuses_changed_retained_declaration_or_checkpoint(monkeypatch, tmp_path, fault):
    encoded, sequence = native._encoded, native.run_sequence
    owners = {}

    def encoding(value):
        if (
            type(value) is dict
            and value.get("mode") == "retained_native_training_rollout"
            and "stream" in value
        ):
            owners["report"] = value
        return encoded(value)

    def stream(warm, releases, observer, **kwargs):
        def observing(owner, metadata):
            owners["metadata"] = metadata
            return observer(owner, metadata)

        return sequence(warm, releases, observing, **kwargs)

    def endpoint(path, *, learner, train_seed, **kwargs):
        charge("fake_scenes")
        charge("fake_controls")
        value = retained_receipt(train_seed)
        path.write_text(json.dumps(value), encoding="utf8")
        if fault == "metadata":
            owners["metadata"]["prototype_rows"] = 7
        elif fault == "cost":
            owners["report"]["retention"]["unexpected"] = True
        else:
            owners["report"]["retention_options"]["max_prototypes"] = 7
        return SimpleNamespace(report=value, corrections=None)

    monkeypatch.setattr(native, "_encoded", encoding)
    monkeypatch.setattr(native, "run_sequence", stream)
    monkeypatch.setattr(native, "run_frozen_rollout", endpoint)
    monkeypatch.setattr(native, "roll_push", forbidden)
    result = native.run_native_demo(
        distinct_bundle(), tmp_path / "main.json", retention=FIFO, max_prototypes=2
    )
    assert not result["complete"] and result.get("integrity_error")
    assert result["retention_options"]["max_prototypes"] == 2
    assert "unexpected" not in result["retention"]
    assert result["pending_paid_return"]["report"]["source_training_rows"] == 7
