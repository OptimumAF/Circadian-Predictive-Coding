from dataclasses import asdict, replace
from pathlib import Path
import subprocess

import pytest

from src.infra.prior_seed_evidence import (
    OWNED_EVIDENCE_DIRECTORY,
    OWNERSHIP_BYTES,
    freeze_prior_seed_files,
    freeze_seed_git_history,
    read_prior_seed_evidence,
)


def _repository(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "old.py").write_text("seed = 7\n", encoding="utf-8")
    subprocess.run(["git", "add", "old.py"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "seed fixture",
        ],
        cwd=root,
        check=True,
    )


def test_should_cover_all_files_history_and_duplicates_without_executing_source(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    (tmp_path / "current.json").write_text(
        '{"seed_map":{"7":{"model":1008}},"seed":11}', encoding="utf-8"
    )
    (tmp_path / "copy.json").write_bytes((tmp_path / "current.json").read_bytes())
    (tmp_path / "source.py").write_text(
        "raise RuntimeError('must not execute')\nrng = unknown.default_rng(seed + 17)\n",
        encoding="utf-8",
    )
    (tmp_path / "state.ckpt").write_bytes(b"opaque checkpoint\x00")
    files, history = freeze_prior_seed_files(tmp_path), freeze_seed_git_history(tmp_path)
    report = read_prior_seed_evidence(tmp_path, files, history)
    assert report["physical_file_count"] == 5
    assert report["git_object_count"] == 3
    assert report["history"]["objects"] == tuple(asdict(obj) for obj in history.objects)
    group = next(row for row in report["contents"] if "current.json" in row["aliases"])
    assert group["aliases"] == ["copy.json", "current.json"]
    assert group["json"]["mapped_streams"][1]["value"] == 1008
    assert any(row["text"] and row["text"]["python_expressions"] for row in report["contents"])
    assert any(row["issues"] for row in report["contents"])
    assert not any(
        report[k]
        for k in (
            "complete_prior_usage_acceptance",
            "fresh_roles_authorized",
            "original_p67_acceptance_complete",
        )
    )


@pytest.mark.parametrize("change", ["subset", "reverse", "added", "corrupted", "history"])
def test_should_reject_changed_or_incomplete_whole_boundary(tmp_path: Path, change: str) -> None:
    _repository(tmp_path)
    (tmp_path / "new.json").write_text('{"seed":11}', encoding="utf-8")
    files, history = freeze_prior_seed_files(tmp_path), freeze_seed_git_history(tmp_path)
    if change == "subset":
        files = files[:-1]
    elif change == "reverse":
        files = tuple(reversed(files))
    elif change == "added":
        (tmp_path / "added.txt").write_text("seed 13", encoding="utf-8")
    elif change == "corrupted":
        (tmp_path / "new.json").write_text('{"seed":13}', encoding="utf-8")
    else:
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=tmp_path,
            input=b"unreachable seed 17",
            check=True,
            capture_output=True,
        )
    with pytest.raises(ValueError, match="prior seed evidence"):
        read_prior_seed_evidence(tmp_path, files, history)


def test_should_include_previous_owner_outputs_and_exclude_only_new_owned_transaction(
    tmp_path: Path,
) -> None:
    _repository(tmp_path)
    output = tmp_path / OWNED_EVIDENCE_DIRECTORY
    output.mkdir(parents=True)
    (output.parent / "previous.json").write_text('{"seed":19}', encoding="utf-8")
    with pytest.raises(ValueError, match="unowned"):
        freeze_prior_seed_files(tmp_path)
    (output / "ownership.txt").write_bytes(OWNERSHIP_BYTES)
    (output / "new.json").write_text('{"seed":23}', encoding="utf-8")
    files = freeze_prior_seed_files(tmp_path)
    assert any(file.path.endswith("previous.json") for file in files)
    assert not any(file.path.startswith(OWNED_EVIDENCE_DIRECTORY) for file in files)


def test_should_keep_all_jsonl_lines_and_malformed_metadata_visible(tmp_path: Path) -> None:
    _repository(tmp_path)
    (tmp_path / "trace.jsonl").write_text('{"seed":7}\n\n{"seed":\n{"seed":11}', encoding="utf-8")
    (tmp_path / "invalid.json").write_text('{"seed":1,"seed":2}', encoding="utf-8")
    report = read_prior_seed_evidence(
        tmp_path, freeze_prior_seed_files(tmp_path), freeze_seed_git_history(tmp_path)
    )
    trace = next(row for row in report["contents"] if "trace.jsonl" in row["aliases"])
    assert [row["parse_status"] for row in trace["jsonl"]] == [
        "parsed",
        "blank",
        "unparsed",
        "parsed",
    ]
    bad = next(row for row in report["contents"] if "invalid.json" in row["aliases"])
    assert bad["issues"][0]["kind"] == "unparsed_json"


def test_should_reject_late_drift_after_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _repository(tmp_path)
    (tmp_path / "seed.json").write_text('{"seed":7}', encoding="utf-8")
    files, history = freeze_prior_seed_files(tmp_path), freeze_seed_git_history(tmp_path)
    from src.infra import prior_seed_evidence as boundary

    original = boundary._inspect
    changed = False

    def corrupt(raw: bytes, paths: tuple[str, ...]) -> dict:
        nonlocal changed
        report = original(raw, paths)
        if not changed and "seed.json" in paths:
            changed = True
            (tmp_path / "seed.json").write_text('{"seed":11}', encoding="utf-8")
        return report

    monkeypatch.setattr(boundary, "_inspect", corrupt)
    with pytest.raises(ValueError, match="after parsing"):
        read_prior_seed_evidence(tmp_path, files, history)


def test_should_reject_noninteger_file_identity_counts(tmp_path: Path) -> None:
    _repository(tmp_path)
    files, history = freeze_prior_seed_files(tmp_path), freeze_seed_git_history(tmp_path)
    forged = (replace(files[0], byte_count=float(files[0].byte_count)),)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exact file identities"):
        read_prior_seed_evidence(tmp_path, forged, history)
