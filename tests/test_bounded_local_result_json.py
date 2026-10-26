"""Literal serialization fixtures only; no learner, data source or environment."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.infra.local_result_json import (
    encode_bounded_local_json,
    write_local_json_payload,
    _encode_numpy,
)


def test_should_preserve_all_nested_values_with_exact_compact_byte_bound(record_property):
    transition = dict(observation=[0.0] * 148, next_observation=[0.0] * 148)
    literal = dict(
        text="μ",
        scalar=np.int64(7),
        array=np.array([1.0, 2.0]),
        literal_nested_report=[dict(selected_episodes=[dict(transitions=[transition] * 9)] * 2)]
        * 80,
    )
    pretty = json.dumps(literal, default=_encode_numpy, indent=2, allow_nan=False).encode() + b"\n"
    assert len(pretty) > 8 * 1024**2
    compact = encode_bounded_local_json(literal, max_bytes=8 * 1024**2)
    record_property("pretty_bytes", len(pretty))
    record_property("compact_bytes", len(compact))
    assert len(compact) < len(pretty) and compact.endswith(b"\n")
    assert json.loads(compact) == json.loads(pretty)
    assert encode_bounded_local_json(literal, max_bytes=len(compact)) == compact
    with pytest.raises(ValueError, match=f"is {len(compact)} bytes"):
        encode_bounded_local_json(literal, max_bytes=len(compact) - 1)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_should_refuse_invalid_cap_before_attempting_encoding(limit):
    with pytest.raises(ValueError, match="exact positive integer"):
        encode_bounded_local_json(object(), max_bytes=limit)


def test_should_preserve_original_pretty_writer_and_exclusive_file(tmp_path):
    record = dict(a=np.array([1.0, 2.0]), b="μ")
    target = tmp_path / "original.json"
    write_local_json_payload(record, target)
    assert (
        target.read_text(encoding="utf8")
        == json.dumps(record, default=_encode_numpy, indent=2, allow_nan=False) + "\n"
    )
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_local_json_payload(dict(replace=True), target)
    assert target.read_bytes() == original


@pytest.mark.parametrize("flags", [[], ["--public-cue"]])
def test_should_use_bounded_compact_encoder_in_existing_cli_without_native_work(
    tmp_path, monkeypatch, flags
):
    from scripts import run_minigrid_paired_bc_smoke as cli
    from src.adapters import minigrid_paired_bc_pilot as paired
    from src.adapters import minigrid_public_cue_recurrent_pilot as cue
    from src.shared import torch_runtime

    report = dict(status="complete", literal_evidence=[dict(values=[0.0, 1.0, 2.0])] * 10)
    monkeypatch.setattr(paired, "run_pilot", lambda: report)
    monkeypatch.setattr(cue, "run_pilot", lambda: report)
    monkeypatch.setattr(
        torch_runtime,
        "require_torch",
        lambda: SimpleNamespace(
            set_num_threads=lambda n: None, set_num_interop_threads=lambda n: None
        ),
    )
    target = tmp_path / "result.json"
    assert cli.main([*flags, "--output", str(target)]) == 0
    expected = encode_bounded_local_json(
        report, max_bytes=8 * 1024**2 if flags else cli.MAX_OUTPUT_BYTES
    )
    assert target.read_bytes() == expected and json.loads(expected) == report
    with pytest.raises(SystemExit):
        cli.main([*flags, "--output", str(target)])
    assert target.read_bytes() == expected
    if not flags:
        monkeypatch.setattr(cli, "MAX_OUTPUT_BYTES", 1)
        refused = tmp_path / "refused.json"
        with pytest.raises(ValueError, match="output cap"):
            cli.main(["--output", str(refused)])
        assert not refused.exists()


def test_should_refuse_nonfinite_or_unsupported_values():
    for record in (dict(value=float("nan")), dict(value=np.array([float("inf")]))):
        with pytest.raises(ValueError):
            encode_bounded_local_json(record, max_bytes=1024)
    with pytest.raises(TypeError, match="unsupported"):
        encode_bounded_local_json(dict(value=object()), max_bytes=1024)
