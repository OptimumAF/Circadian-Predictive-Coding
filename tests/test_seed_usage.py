"""Complete raw declaration locations and conservative unresolved metadata."""

from copy import deepcopy
import json
from typing import Any

import pytest

from src.core.seed_usage import collect_seed_usage, strict_seed_metadata_json


def test_should_keep_every_regular_canonical_embedded_and_command_seed_location() -> None:
    raw = {
        "seed": 41,
        "development_seeds": [43, 59],
        "model": {"dict": [["seed", 1042], ["other", {"random_seed": 11}]]},
        "manifest_json": json.dumps({"confirmation_seeds": [101, 103]}),
        "argv": ["python", "run.py", "--seed=7", "--seeds", "13", "17,19", "--output", "x"],
    }
    before = deepcopy(raw)
    result = collect_seed_usage(raw)
    assert raw == before
    assert [r.seed for r in result.declarations] == [41, 43, 59, 1042, 11, 101, 103, 7, 13, 17, 19]
    assert len({(r.pointer, r.field, r.seed) for r in result.declarations}) == 11
    assert result.unresolved == ()
    assert any("embedded_json" in row.pointer for row in result.declarations)
    assert any("/dict/0/1" in row.pointer for row in result.declarations)


def test_should_preserve_null_invalid_late_nested_and_embedded_seed_ambiguity() -> None:
    result = collect_seed_usage(
        {
            "seeds": [0, True, -1, None],
            "last": {"seed": "variable"},
            "payload": '{"seed": 99, broken}',
        }
    )
    assert [r.seed for r in result.declarations] == [0]
    assert len(result.unresolved) == 5
    assert result.unresolved[-1].reason == "unparsed_embedded_seed_metadata"
    assert result.unresolved[-2].pointer == "/last/seed"


def test_should_keep_symbolic_unknowns_and_not_relabel_counts_offsets_or_metrics_as_seeds() -> None:
    result = collect_seed_usage(
        {
            "seed_count": 10,
            "model_seed_offset": 1001,
            "seed_mean": 0.5,
            "distinct_confirmation_seeds": 50,
            "distinct_pilot_source_seeds": 15,
            "seeds": [{"seed": 59, "score": 0.7}],
        }
    )
    assert [r.seed for r in result.declarations] == [59]
    assert len(result.unresolved) == 1
    assert result.unresolved[0].reason == "non_scalar_seed_list_member"


def test_should_keep_canonical_seed_sequences_and_escaped_pointer_fields() -> None:
    result = collect_seed_usage({"a/b~c": {"source_seeds": {"tuple": [0, 1]}}})
    assert [r.seed for r in result.declarations] == [0, 1]
    assert result.unresolved == ()
    assert result.declarations[0].pointer == "/a~1b~0c/source_seeds/tuple/0"


@pytest.mark.parametrize("value", [True, -1, 1.0, float("nan"), {}, None, "", "1,garbled"])
def test_should_retain_unresolved_values_without_silently_claiming_seed_absence(value: Any) -> None:
    result = collect_seed_usage({"seed": value})
    assert result.declarations == ()
    assert len(result.unresolved) == 1


@pytest.mark.parametrize("raw", [b'{"seed":1,"seed":2}', b'{"seed":NaN}', b"{", b"\xff"])
def test_should_reject_duplicate_nonfinite_truncated_or_non_utf8_json(raw: bytes) -> None:
    with pytest.raises(ValueError):
        strict_seed_metadata_json(raw)


def test_should_retain_missing_cli_seed_and_fail_explicit_depth_limit() -> None:
    assert (
        collect_seed_usage({"command": ["run.py", "--seed", "--output", "x"]}).unresolved[0].reason
        == "missing_seed_argument"
    )
    body: Any = {"seed": 1}
    for _ in range(258):
        body = [body]
    with pytest.raises(ValueError, match="nesting"):
        collect_seed_usage(body)
