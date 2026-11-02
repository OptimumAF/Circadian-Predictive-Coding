"""#408 procedure fixtures: typed synthetic rows/diagnostic doubles, zero NN work."""

from dataclasses import dataclass, replace
import hashlib
import importlib.util
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace
import struct

import numpy as np
import pytest

from test_complete_b22_neural import groups

PATH = (
    Path(__file__).resolve().parents[1] / "artifacts/runs/train-command-diagnosis-20261101/study.py"
)
spec = importlib.util.spec_from_file_location("train_command_diagnosis_procedure", PATH)
assert spec is not None and spec.loader is not None
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)

WORK = dict(forbidden=0, fake_residual_attempts=0, fake_rows=0, fake_commands=0)


@pytest.fixture(autouse=True)
def forbid_real_work(monkeypatch):
    from src.adapters import numpy_continuous_bc as bp
    from src.core import continuous_bc as bc

    def forbidden(*args, **options):
        WORK["forbidden"] += 1
        raise AssertionError("procedure fixture attempted real model/data work")

    for owner, name in (
        (bp.NumpyContinuousBCLearner, "__init__"),
        (bp.NumpyContinuousBCLearner, "predict"),
        (bp.NumpyContinuousBCLearner, "train_batch"),
        (bc, "_forward"),
        (bc, "continuous_bc_loss_and_gradients"),
        (np.random, "default_rng"),
        (np.linalg, "svd"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    original_open = Path.open

    def guarded_open(path, *args, **options):
        if path.resolve() in (study.BUNDLE, study.EXPORT):
            forbidden()
        return original_open(path, *args, **options)

    monkeypatch.setattr(Path, "open", guarded_open)


@dataclass
class LiteralResidual:
    prediction_calls: int
    prediction_rows: int
    prediction_values: int
    prediction_bytes: int
    residuals: dict


def install(fault=None):
    calls: list[tuple[str, str]] = []
    original: list[Any] = []
    invalid_source = False

    def verify():
        if invalid_source:
            raise RuntimeError("original frozen source changed")

    def diagnose(owner, episode, stages, *, prediction_observer):
        nonlocal invalid_source
        WORK["fake_residual_attempts"] += 1
        WORK["fake_rows"] += len(episode.transitions)
        calls.append((episode.episode_id, "residual"))
        target = np.array([row.action for row in episode.transitions], dtype=np.float64)
        prediction = np.zeros_like(target)
        result = LiteralResidual(
            1, len(target), target.size, target.nbytes, dict(overall=dict(normalized_half_mse=0.5))
        )
        if fault == "bool_receipt":
            result.prediction_calls = True
        original.append(result)
        prediction_observer(result, prediction, target, stages)
        if fault == "duplicate":
            prediction_observer(result, prediction, target, stages)
        if fault == "source":
            invalid_source = True
        if fault == "lost_return":
            raise RuntimeError("lost residual return after paid capture")
        return replace(result) if fault == "replacement" else result

    def command(prediction, target, declared, stages, **options):
        WORK["fake_commands"] += 1
        calls.append((str(len(prediction)), "command"))
        if fault == "command_throw":
            raise RuntimeError("command failed after paid residual")
        rows = len(prediction)
        work = dict(
            conversion_attempts=8 * rows,
            conversion_returns=8 * rows,
            equality_comparisons=4 * rows,
            absolute_error_coordinates=4 * rows,
            translation_direction_comparisons=3 * rows,
            gripper_margin_products=rows,
            gripper_sign_classifications=rows,
            known_owned_numeric_bytes=160 * rows,
        )
        if fault == "counters":
            work["conversion_returns"] = 0
        if fault == "capture":
            prediction[0, 0] = 0.9
        elif fault == "native_record":
            original[0].residuals["overall"]["normalized_half_mse"] = 999.0
        elif fault == "prefix":
            sys._getframe(1).f_locals["report"]["records"].clear()
        elif fault == "holder":
            parent = sys._getframe(1).f_locals["report"]
            parent["records"] = list(parent["records"])
        return dict(work=work, definition="scalar-diagnostic-double")

    return calls, verify, diagnose, command


def test_complete_seven_groups_use_same_capture_and_keep_original_roles():
    calls, verify, diagnose, command = install()
    result = study.observe_groups(object(), groups(), verify, diagnose=diagnose, command=command)
    assert result["complete"], result
    assert len(calls) == len(result["records"]) == 14
    assert result["unreturned_operation_work_unknown"] is False
    assert result["prediction_capture_numeric_bytes"] == 14080
    assert [row["rows"] for row in result["records"][::2]] == [50, 50, 18, 15, 50, 15, 22]
    assert all(row["role"] == "train" for row in result["records"])


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "source",
        "lost_return",
        "replacement",
        "command_throw",
        "counters",
        "capture",
        "native_record",
        "prefix",
        "holder",
        "bool_receipt",
    ],
)
def test_late_refusal_preserves_paid_prefix_and_never_starts_next_group(fault):
    calls, verify, diagnose, command = install(fault)
    result = study.observe_groups(object(), groups(), verify, diagnose=diagnose, command=command)
    assert result["complete"] is False and result["error"]
    assert len(calls) <= 2
    assert result["records"][0]["kind"] == "residual"
    assert result["records"][0]["payload"]["residuals"]["overall"]["normalized_half_mse"] == 0.5
    assert len(result["attempted"]) <= 2


@pytest.mark.parametrize("fault", ["order", "missing", "probe_train", "probe_test", "final_test"])
def test_bad_training_groups_refuse_before_any_diagnostic(fault):
    data = groups()
    if fault == "order":
        data = (data[1], data[0], *data[2:])
    elif fault == "missing":
        data = data[:-1]
    else:
        data = ((data[0][0], replace(data[0][1], role=fault)), *data[1:])
    calls, verify, diagnose, command = install()
    with pytest.raises(ValueError):
        study.observe_groups(object(), data, verify, diagnose=diagnose, command=command)
    assert not calls


@pytest.mark.parametrize("fault", [None, "hash", "size", "growth"])
def test_bounded_new_input_read_never_accepts_changed_bytes(tmp_path, fault):
    path = tmp_path / "synthetic.json"
    raw = b'{"synthetic":true}'
    path.write_bytes(raw + (b" " if fault == "growth" else b""))
    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if fault == "hash":
        digest = "0" * 64
    elif fault == "size":
        size -= 1
    if fault is None:
        assert study.read_bound(path, size, digest) == raw
    else:
        with pytest.raises(ValueError):
            study.read_bound(path, size, digest)


@pytest.mark.parametrize(
    "fault", ["state_policy", "state_spec", "units", "lower", "upper", "parameters", "velocity"]
)
def test_equal_nested_replacement_in_first_getter_is_refused(fault):
    def declared():
        return SimpleNamespace(
            action_units=("x", "y"), lower_bounds=(-1.0, -1.0), upper_bounds=(1.0, 1.0)
        )

    state_policy = SimpleNamespace(spec=declared())
    owner = SimpleNamespace(_policy=SimpleNamespace(spec=declared()))
    owner._state = SimpleNamespace(
        policy=state_policy, parameters=(object(),), velocity=(object(),)
    )
    identities = study.native_identity(owner)

    def snapshot():
        if fault == "state_policy":
            owner._state.policy = SimpleNamespace(spec=state_policy.spec)
        elif fault == "state_spec":
            state_policy.spec = declared()
        elif fault in ("units", "lower", "upper"):
            name = {"units": "action_units", "lower": "lower_bounds", "upper": "upper_bounds"}[
                fault
            ]
            setattr(state_policy.spec, name, tuple(list(getattr(state_policy.spec, name))))
        else:
            setattr(owner._state, fault, tuple(list(getattr(owner._state, fault))))
        return owner._state

    owner.snapshot_state = snapshot
    adapter = SimpleNamespace(_snapshot=lambda state, policy: state)
    with pytest.raises(RuntimeError, match="nested owner"):
        study.require_frozen(
            owner, identities, study.encoded({"same": True}), adapter, lambda state: {"same": True}
        )


def test_late_return_guard_cannot_erase_acknowledged_copy_payload():
    counts = dict(acks=0, known_copy_bytes=0)
    value = SimpleNamespace(
        parameters=(SimpleNamespace(nbytes=16),), velocity=(SimpleNamespace(nbytes=8),)
    )

    def late_stop():
        raise RuntimeError("late permanent stop")

    with pytest.raises(RuntimeError):
        study.record_return(counts, "acks", value, late_stop, payload=True)
    assert counts == dict(acks=1, known_copy_bytes=24)


def test_shared_struct_packing_separates_command_scalars_from_admission_history():
    counts = dict(
        conversions=0,
        conversion_returns=0,
        validation_packs=0,
        validation_pack_returns=0,
        validation_values=0,
        packed_numeric_bytes=0,
    )

    def charge(key, amount=1):
        counts[key] += amount

    for fmt, values in (("<4f", (0.0, 0.0, 0.0, -1.0)), ("=f", (1.0,)), ("<f", (0.5,))):
        study.packed_value(struct.pack, counts, charge, lambda: None, fmt, *values)
    assert counts == dict(
        conversions=1,
        conversion_returns=1,
        validation_packs=2,
        validation_pack_returns=2,
        validation_values=5,
        packed_numeric_bytes=24,
    )


def test_metadata_bridge_preserves_values_and_passes_unchanged_strict_phase_gate():
    from src.adapters.continuous_phase_readouts import _finite_json
    import json

    fixed_spec = importlib.util.spec_from_file_location(
        "capture_fixed_procedure", PATH.with_name("capture_fixed.py")
    )
    assert fixed_spec is not None and fixed_spec.loader is not None
    fixed = importlib.util.module_from_spec(fixed_spec)
    fixed_spec.loader.exec_module(fixed)
    calls, verify, diagnose, command = install()
    original = study.observe_groups(object(), groups(), verify, diagnose=diagnose, command=command)
    assert len(calls) == 14
    with pytest.raises(ValueError, match="native JSON"):
        _finite_json(original)
    encoded = fixed.phase_payload(original)
    validated, size = _finite_json(encoded)
    assert validated == json.loads(study.encoded(original))
    assert size < 65536
    assert original["complete"] and isinstance(original["attempted"][0], tuple)
