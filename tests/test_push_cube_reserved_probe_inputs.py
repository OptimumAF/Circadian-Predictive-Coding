"""Synthetic fixed reserved exports; no historical payload or teacher is read.

All cases perform bounded JSON IO and typed/history validation only. They create
no model, optimizer, simulator, collection authority or changed reserved split.
"""

from dataclasses import asdict, replace
import hashlib
import json

import pytest

from scripts import push_cube_reserved_probe_inputs as inputs
from src.core.continuous_demonstration import ContinuousEpisode, ContinuousTransition
from src.core.cube_taskspace_state import cube_taskspace_spec
from src.core.pick_cube_probe_context import pick_cube_probe_binding


def reserved_episode(role):
    binding = pick_cube_probe_binding(role)
    action = (0.25 if role == "probe_train" else -0.25, 0.0, 0.0, -1.0)
    prefix = (0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.0)
    rows = []
    for index in range(18):
        previous = (0.0,) * 4 if index == 0 else action
        rows.append(
            ContinuousTransition(
                index,
                prefix + previous + (min(index, 6) / 6,),
                action,
                0.0,
                prefix + action + (min(index + 1, 6) / 6,),
                index == 17,
                False,
            )
        )
    return ContinuousEpisode(
        binding.episode_id, binding.context, role, cube_taskspace_spec(), tuple(rows)
    )


def write_pair(tmp_path, monkeypatch, *, altered=None):
    probes = {}
    payloads = {}
    for role in ("probe_train", "probe_test"):
        raw = asdict(reserved_episode(role))
        # Normalize exactly as a JSON file, so boundary faults modify JSON values.
        raw = json.loads(json.dumps(raw))
        if altered is not None and role == "probe_train":
            altered(raw)
        payload = json.dumps(raw, allow_nan=True, separators=(",", ":")).encode("utf8")
        path = tmp_path / (role + ".json")
        path.write_bytes(payload)
        probes[role] = (path, hashlib.sha256(payload).hexdigest())
        payloads[role] = payload
    monkeypatch.setattr(inputs, "PROBES", probes)
    return probes, payloads


def test_should_load_owned_fixed_split_and_preserve_exact_input_byte_receipts(
    tmp_path, monkeypatch
):
    probes, payloads = write_pair(tmp_path, monkeypatch)
    plan, receipts = inputs.load_reserved_probe_inputs()
    assert (plan.updates, plan.diagnostics, plan.zero_velocity_control) == (1, True, False)
    for role, episode, receipt in zip(
        ("probe_train", "probe_test"), (plan.training, plan.evaluation), receipts, strict=True
    ):
        original = reserved_episode(role)
        assert episode == original and episode is not original
        path, digest = probes[role]
        assert receipt == dict(
            path=str(path),
            sha256=digest,
            bytes=len(payloads[role]),
            role=role,
            context=original.context,
            episode_id=original.episode_id,
            rows=18,
            reader_receipt=len(payloads[role]),
        )
        assert episode.transitions[0].observation[-5:] == (0.0,) * 5
        assert episode.transitions[1].observation[-5:-1] == episode.transitions[0].action
        assert episode.transitions[-1].next_observation[-1] == 1.0
        assert path.read_bytes() == payloads[role]
    # Files and receipts are not mutable storage for the plan's owned episodes.
    probes["probe_train"][0].write_text("{}", encoding="utf8")
    receipts[0]["role"] = "train"
    assert plan.training == reserved_episode("probe_train")


@pytest.mark.parametrize(
    "fault", ["role", "context", "identity", "rows", "spec", "history", "grasp"]
)
def test_should_refuse_rehashed_invalid_reserved_authority_or_causal_history(
    tmp_path, monkeypatch, fault
):
    def corrupt(raw):
        if fault == "role":
            raw["role"] = "train"
        elif fault == "context":
            raw["context"] = "push-cube-train-17"
        elif fault == "identity":
            raw["episode_id"] = "pick-cube-probe-test-teacher-24"
        elif fault == "rows":
            raw["transitions"] = raw["transitions"][:-1]
        elif fault == "spec":
            raw["spec"]["action_units"][0] = "m"
        elif fault == "history":
            raw["transitions"][1]["observation"][-5] = 0.0
        else:
            raw["transitions"][0]["observation"][0] = 2.0

    write_pair(tmp_path, monkeypatch, altered=corrupt)
    with pytest.raises(ValueError):
        inputs.load_reserved_probe_inputs()


def test_should_refuse_changed_bytes_before_typed_reader(tmp_path, monkeypatch):
    probes, _ = write_pair(tmp_path, monkeypatch)
    probes["probe_train"][0].write_bytes(b"{}")

    def forbidden(*args, **kwargs):
        pytest.fail("digest failure reached typed parsing")

    monkeypatch.setattr(inputs, "read_continuous_probe_episode", forbidden)
    with pytest.raises(ValueError, match="digest"):
        inputs.load_reserved_probe_inputs()


@pytest.mark.parametrize("fault", ["duplicate", "nonfinite", "extra_field"])
def test_should_reuse_strict_json_reader_even_for_matching_hash(tmp_path, monkeypatch, fault):
    probes, payloads = write_pair(tmp_path, monkeypatch)
    payload = payloads["probe_train"]
    if fault == "duplicate":
        payload = b'{"role":"probe_train",' + payload[1:]
    elif fault == "nonfinite":
        raw = json.loads(payload)
        raw["transitions"][0]["observation"][1] = float("nan")
        payload = json.dumps(raw).encode("utf8")
    else:
        raw = json.loads(payload)
        raw["collected_authority"] = True
        payload = json.dumps(raw).encode("utf8")
    path = probes["probe_train"][0]
    path.write_bytes(payload)
    probes["probe_train"] = (path, hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError):
        inputs.load_reserved_probe_inputs()


def test_should_enforce_original_two_mebibyte_bound_before_reader(tmp_path, monkeypatch):
    probes, _ = write_pair(tmp_path, monkeypatch)
    payload = b" " * (inputs.MAX_PROBE_FILE_BYTES + 1)
    path = probes["probe_train"][0]
    path.write_bytes(payload)
    probes["probe_train"] = (path, hashlib.sha256(payload).hexdigest())

    def forbidden(*args, **kwargs):
        pytest.fail("oversized input reached parser")

    monkeypatch.setattr(inputs, "read_continuous_probe_episode", forbidden)
    with pytest.raises(ValueError, match="bound"):
        inputs.load_reserved_probe_inputs()


@pytest.mark.parametrize("fault", ["bytes", "receipt"])
def test_should_refuse_replaced_input_or_inaccurate_byte_receipt_after_reader(
    tmp_path, monkeypatch, fault
):
    write_pair(tmp_path, monkeypatch)
    reader = inputs.read_continuous_probe_episode
    calls: list[str] = []

    def replaced(path, *, role):
        episode, receipt = reader(path, role=role)
        calls.append(role)
        if fault == "bytes":
            path.write_bytes(path.read_bytes() + b" ")
        else:
            receipt += 1
        return episode, receipt

    monkeypatch.setattr(inputs, "read_continuous_probe_episode", replaced)
    with pytest.raises(ValueError, match="bytes changed"):
        inputs.load_reserved_probe_inputs()
    assert calls == ["probe_train"]


def test_should_refuse_changed_decoded_nonhistory_coordinate_with_original_file_bytes(
    tmp_path, monkeypatch
):
    probes, payloads = write_pair(tmp_path, monkeypatch)
    reader = inputs.read_continuous_probe_episode

    def changed(path, *, role):
        episode, receipt = reader(path, role=role)
        row = episode.transitions[0]
        observation = list(row.observation)
        observation[1] += 0.125
        changed_row = replace(row, observation=tuple(observation))
        return replace(episode, transitions=(changed_row, *episode.transitions[1:])), receipt

    monkeypatch.setattr(inputs, "read_continuous_probe_episode", changed)
    with pytest.raises(ValueError, match="decoded episode"):
        inputs.load_reserved_probe_inputs()
    assert all(probes[role][0].read_bytes() == payloads[role] for role in probes)
