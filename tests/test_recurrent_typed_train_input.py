"""Actual packaged TRAIN decoding and typed wiring, with zero native work."""

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_minigrid_linear_fit_control import read_training_sources
from src.adapters import minigrid_public_cue_recurrent_pilot as pilot
from src.infra.minigrid_affine_example import example_affine_training


@pytest.fixture(scope="module")
def examples():
    with example_affine_training() as path:
        source = json.loads(path.read_text(encoding="utf8"))["training_sources"]
        raw = read_training_sources(path, raw_frames=True)
        history = read_training_sources(path)
    return source, raw, history


def test_should_decode_exact_original_public_frames_and_preserve_default_history(examples):
    records, raw, history = examples
    assert len(raw) == len(history) == 6
    assert json.loads(json.dumps([asdict(e) for e in raw])) == records
    for episode, encoded in zip(raw, history, strict=True):
        assert (episode.episode_id, episode.role, episode.context, episode.num_classes) == (
            encoded.episode_id,
            encoded.role,
            encoded.context,
            encoded.num_classes,
        )
        first = episode.transitions[0].observation
        for step, h in zip(episode.transitions, encoded.transitions, strict=True):
            assert len(step.observation) == 148 and h.observation == first + step.observation
            assert h.next_observation == first + step.next_observation
            assert (step.action, step.reward, step.terminal) == (h.action, h.reward, h.terminal)


@pytest.mark.parametrize("bad", [0, 1, None, "raw", ()])
def test_should_refuse_nonboolean_raw_flag_before_file_access(bad):
    with pytest.raises(ValueError, match="exact boolean"):
        read_training_sources(Path("not-present"), raw_frames=bad)


def test_should_detach_raw_sources_and_preserve_ordered_schedule_without_history(examples):
    _, raw, _ = examples
    before = deepcopy(raw)
    sources = pilot._typed_sources(raw)
    assert raw == before
    assert [s.context for s in sources] == ["A", "A", "B", "B", "C_train", "C_train"]
    assert all(s.features.shape == (9, 148) and s.targets.shape == (9, 1) for s in sources)
    sources[0].features[0, 0] += 0.125
    assert raw == before


@pytest.mark.parametrize(
    "bad",
    [
        "list",
        "short",
        "foreign",
        "history",
        "order",
        "id",
        "cue",
        "final_test",
        "probe_train",
        "probe_test",
    ],
)
def test_should_refuse_bad_typed_inputs_before_any_native_or_archive_access(
    examples, monkeypatch, bad
):
    _, raw, history = examples
    values = raw
    if bad == "list":
        values = list(raw)
    if bad == "short":
        values = raw[:-1]
    if bad == "foreign":
        values = (object(), *raw[1:])
    if bad == "history":
        values = history
    if bad == "order":
        values = raw[::-1]
    if bad == "id":
        values = (replace(raw[0], episode_id="other-blue"), *raw[1:])
    if bad == "cue":
        first = replace(raw[0].transitions[0], observation=raw[1].transitions[0].observation)
        values = (replace(raw[0], transitions=(first, *raw[0].transitions[1:])), *raw[1:])
    if bad in ("final_test", "probe_train", "probe_test"):
        values = (replace(raw[0], role=bad), *raw[1:])
    monkeypatch.setattr(pilot, "require_torch", lambda: pytest.fail("no native work admitted"))
    monkeypatch.setattr(pilot, "_load_reference", lambda: pytest.fail("no archive access admitted"))
    with pytest.raises(ValueError):
        pilot.run_pilot(values)


def test_should_describe_typed_source_explicitly_and_preserve_archive_default():
    old, typed = pilot.describe_protocol(), pilot.describe_protocol(typed_input=True)
    assert old["protocol_id"] == "minigrid_public_cue_recurrent_a128_v1"
    assert "source_mode" not in old
    assert typed["protocol_id"] == "minigrid_public_cue_recurrent_typed_a128_v1"
    assert typed["planned_work"] == old["planned_work"] and typed["limits"] == old["limits"]
    assert typed["routing"] == old["routing"]


@pytest.mark.parametrize("flag", [0, "typed", None])
def test_should_refuse_nonboolean_source_protocol_flag(flag):
    with pytest.raises(ValueError):
        pilot.describe_protocol(typed_input=flag)


def test_should_use_typed_packets_without_reading_archive_or_admitting_a_model(
    examples, monkeypatch
):
    from scripts import run_minigrid_history_evaluation_smoke as existing

    class Rng:
        def clone(self):
            return self

    rng = Rng()
    monkeypatch.setattr(
        pilot,
        "require_torch",
        lambda: SimpleNamespace(get_rng_state=lambda: rng, equal=lambda a, b: a is b),
    )
    monkeypatch.setattr(
        pilot, "_load_reference", lambda: pytest.fail("typed source must avoid archive")
    )
    captured = []

    def stop(row, packets, *args, **kwargs):
        captured.append(packets)
        raise RuntimeError("zero native allowance; intentional pre-model test stop")

    monkeypatch.setattr(pilot.shared, "_train_arm", stop)
    result = pilot.run_pilot(examples[1])
    assert result["status"] == "incomplete" and result["outcomes"] is None
    assert result["reason"].endswith("intentional pre-model test stop")
    assert result["reference"]["path"] is None and result["numpy_rng_unchanged"] is True
    assert result["protocol"]["source_mode"].startswith("caller_owned")
    expected = existing._fixed_a_packets(pilot._typed_sources(examples[1]))
    assert [asdict(p.episode) for p in captured[0]] == [asdict(p.episode) for p in expected]


def test_should_keep_existing_schema_and_hash_refusals_for_raw_input(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"training_sources":[]}', encoding="utf8")
    with pytest.raises(ValueError, match="bytes differ"):
        read_training_sources(path, raw_frames=True, expected_sha256="wrong")
    with pytest.raises(ValueError, match="1..16"):
        read_training_sources(path, raw_frames=True)
