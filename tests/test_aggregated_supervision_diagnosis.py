"""Accepted-input binding and exclusive CLI publication using synthetic cached JSON.

The grouped kernel is a counted scalar double here. No model, SDK, teacher or
scientific payload is read. Genuine grouped arithmetic lives in its own module.
"""

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import sys
from typing import Any

import pytest

from scripts import diagnose_aggregated_supervision as cli
from src.core.cube_taskspace_state import cube_taskspace_spec
from test_grouped_input_target_consistency import full_groups


def forbidden(*args, **options):
    pytest.fail("refused inputs reached loader or grouped arithmetic")


@pytest.fixture
def cached(tmp_path, monkeypatch):
    parts = list(full_groups())
    parts[0] = (
        parts[0][0],
        replace(parts[0][1], episode_id="push-cube-correction-targets-17-source120"),
    )
    parts[1] = (parts[1][0], replace(parts[1][1], episode_id="push-cube-correction-targets-17"))
    data = tuple(parts)
    original_path = tmp_path / "original.json"
    original_bytes = b"synthetic original correction".ljust(123, b" ")
    original_path.write_bytes(original_bytes)
    original_receipt = dict(
        path=str(original_path),
        bytes=len(original_bytes),
        sha256=hashlib.sha256(original_bytes).hexdigest(),
    )
    metadata: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    boundaries: list[list[Any]] = []
    offset = 0
    for index, (_, item) in enumerate(data):
        group = "training" if index == 0 else f"reference_{index - 1}"
        identity = dict(group=group, episode_id=item.episode_id, context=item.context, role="train")
        metadata.append(dict(**identity, rows=len(item.transitions)))
        bindings.extend(dict(**identity, step=row.step) for row in item.transitions)
        boundaries.append([group, offset, offset + len(item.transitions)])
        offset += len(item.transitions)
    study = dict(
        protocol="push_cube_aggregated_correction_train_v1",
        complete=True,
        error=None,
        source_unchanged=True,
        unreturned_stage_work_unknown=False,
        aggregated_continuation=dict(
            complete=True,
            source_exact=True,
            actor_exact=True,
            unreturned_training_work_unknown=False,
            rows=198,
            initial_counts=dict(updates=120, training_rows=11016),
            final_counts=dict(updates=128, training_rows=12600),
            original_inputs=metadata,
            row_bindings=bindings,
            groups=boundaries,
        ),
        correction_fit_protocol=dict(input_receipt=deepcopy(original_receipt)),
        current_correction_packet=asdict(data[0][1]),
        training=asdict(data[2][1]),
        references=[asdict(item) for _, item in data[3:]],
        aggregated_correction_protocol=dict(
            group_order=list(cli.GROUPS),
            groups=[
                dict(group=name, episode_id=item.episode_id, rows=len(item.transitions))
                for name, item in data
            ],
        ),
    )
    path = tmp_path / "cached.json"
    loader_calls: list[bool] = []

    def bind():
        raw = json.dumps(study, allow_nan=False).encode()
        path.write_bytes(raw)
        monkeypatch.setattr(cli, "SOURCE_SHA256", hashlib.sha256(raw).hexdigest())
        return raw

    def original():
        loader_calls.append(True)
        return data[1][1], original_receipt

    monkeypatch.setattr(cli, "load_push_correction_inputs", original)
    bind()
    return path, study, data, original_receipt, loader_calls, bind


def test_should_own_all_six_original_packets_metadata_and_bound_source_receipt(cached):
    path, raw, original, receipt, calls, _ = cached
    data, binding = cli.read_aggregated_inputs(path)
    assert len(calls) == 1 and tuple(name for name, _ in data) == cli.GROUPS
    assert tuple(len(item.transitions) for _, item in data) == cli.COUNTS
    assert all(item.spec == cube_taskspace_spec() for _, item in data)
    assert tuple(asdict(item) for _, item in data) == tuple(asdict(item) for _, item in original)
    assert data[0][1].episode_id != data[1][1].episode_id
    assert binding["accepted_study"] == dict(
        path=str(path), bytes=path.stat().st_size, sha256=cli.SOURCE_SHA256
    )
    assert binding["original_correction"] == receipt
    raw["current_correction_packet"]["context"] = "changed-local-json"
    assert data[0][1].context == original[0][1].context


@pytest.mark.parametrize(
    "fault",
    [
        "protocol",
        "complete",
        "unknown",
        "source",
        "continuation",
        "counter",
        "counter_bool",
        "role",
        "spec",
        "current_id",
        "recipe",
        "metadata",
        "binding",
        "boundary",
        "receipt",
    ],
)
def test_should_refuse_changed_accepted_scope_or_whole_row_binding_before_kernel(
    cached, monkeypatch, fault
):
    path, raw, _, _, _, bind = cached
    if fault == "protocol":
        raw["protocol"] = "other"
    elif fault == "complete":
        raw["complete"] = False
    elif fault == "unknown":
        raw["unreturned_stage_work_unknown"] = True
    elif fault == "source":
        raw["source_unchanged"] = False
    elif fault == "continuation":
        raw["aggregated_continuation"]["actor_exact"] = False
    elif fault == "counter":
        raw["aggregated_continuation"]["final_counts"]["updates"] = 127
    elif fault == "counter_bool":
        raw["aggregated_continuation"]["rows"] = True
    elif fault == "role":
        raw["current_correction_packet"]["role"] = "final_test"
    elif fault == "spec":
        units = raw["current_correction_packet"]["spec"]["action_units"]
        raw["current_correction_packet"]["spec"]["action_units"] = ("other",) + units[1:]
    elif fault == "current_id":
        raw["current_correction_packet"]["episode_id"] = "push-cube-correction-targets-17"
    elif fault == "recipe":
        raw["aggregated_correction_protocol"]["groups"][0]["rows"] = 49
    elif fault == "metadata":
        raw["aggregated_continuation"]["original_inputs"][0]["context"] = "wrong"
    elif fault == "binding":
        raw["aggregated_continuation"]["row_bindings"][-1]["step"] = 0
    elif fault == "boundary":
        raw["aggregated_continuation"]["groups"][-1][-1] = 197
    else:
        raw["correction_fit_protocol"]["input_receipt"]["sha256"] = "wrong"
    bind()
    monkeypatch.setattr(cli, "diagnose_grouped_input_targets", forbidden)
    with pytest.raises(ValueError):
        cli.run_diagnosis(path.with_name("output.json"), source=path)
    assert not path.with_name("output.json").exists()


@pytest.mark.parametrize("fault", ["hash", "oversize"])
def test_should_refuse_unbound_or_oversized_source_before_original_loader(
    cached, monkeypatch, fault
):
    path, _, _, _, calls, _ = cached
    path.write_bytes(path.read_bytes() + b" " if fault == "hash" else b" " * (6 * 1024**2 + 1))
    monkeypatch.setattr(cli, "load_push_correction_inputs", forbidden)
    with pytest.raises(ValueError):
        cli.read_aggregated_inputs(path)
    assert calls == []


@pytest.mark.parametrize("fault", [None, "partial", "replacement", "original_replacement"])
def test_should_publish_once_keep_paid_partial_and_detect_source_replacement_without_retry(
    cached, monkeypatch, fault
):
    path, _, data, original_receipt, _, _ = cached
    calls: list[object] = []
    diagnosis = dict(
        complete=fault != "partial",
        error="known kernel failure" if fault == "partial" else None,
        unknown_unreturned_kernel_work=fault == "partial",
        work=dict(kernel_calls_attempted=1, pair_attempts=1),
        pairs=[],
    )

    def diagnosed(groups):
        calls.append(groups)
        assert tuple(asdict(item) for _, item in groups) == tuple(asdict(item) for _, item in data)
        if fault == "replacement":
            path.write_bytes(path.read_bytes() + b" ")
        elif fault == "original_replacement":
            original_path = path.with_name("original.json")
            original_path.write_bytes(original_path.read_bytes() + b" ")
        return diagnosis

    monkeypatch.setattr(cli, "diagnose_grouped_input_targets", diagnosed)
    output = path.with_name("output.json")
    result = cli.run_diagnosis(output, source=path)
    assert len(calls) == 1 and result["diagnosis"] is diagnosis
    assert result["complete"] is (fault is None)
    assert result["source_bytes_unchanged"] is (
        fault not in ("replacement", "original_replacement")
    )
    read_work = result["work"]["accepted_study_reads"]
    assert read_work["attempts"] == read_work["returns"] == 2
    assert read_work["bytes"] == 2 * result["sources"]["accepted_study"]["bytes"] + (
        1 if fault == "replacement" else 0
    )
    assert result["work"]["source_derived_original_correction_reads"] == 2
    assert result["work"]["source_derived_original_correction_bytes"] == 246
    expected_final = (
        dict(attempts=0, returns=0, bytes=0)
        if fault == "replacement"
        else dict(
            attempts=1,
            returns=1,
            bytes=original_receipt["bytes"] + (1 if fault == "original_replacement" else 0),
        )
    )
    assert result["work"]["original_correction_final_read"] == expected_final
    assert json.loads(output.read_text()) == json.loads(json.dumps(result))
    before = output.read_bytes()
    with pytest.raises(ValueError):
        cli.run_diagnosis(output, source=path)
    assert output.read_bytes() == before and len(calls) == 1


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("explicit_source", [False, True])
def test_should_route_default_or_explicit_cli_source_and_return_honest_status(
    tmp_path, monkeypatch, capsys, complete, explicit_source
):
    output, source = tmp_path / "out.json", tmp_path / "input.json"
    calls: list[object] = []

    def run(path, *, source):
        calls.append((path, source))
        return dict(complete=complete, error=None if complete else "partial")

    monkeypatch.setattr(cli, "run_diagnosis", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["diagnose", str(output)] + (["--source", str(source)] if explicit_source else []),
    )
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == (0 if complete else 1)
    assert calls == [(output, source if explicit_source else cli.SOURCE)]
    assert json.loads(capsys.readouterr().out)["complete"] is complete
