from hashlib import sha256

import pytest

from src.core.seed_source_evidence import inspect_seed_csv, inspect_seed_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("feed seek succeeded range orangutan pc_glue rand om rn g p cg", []),
        ("sеed sееd rոg rｎg pсg sÉed ßeed", []),
        (
            "seed/rng,pCg RANDOM",
            [(1, "seed/rng,pCg RANDOM", [(0, "seed"), (5, "rng"), (9, "pCg"), (13, "RANDOM")])],
        ),
        (
            "ſEED ßeed _rng_ 7PCGα Café_random9",
            [
                (
                    1,
                    "ſEED ßeed _rng_ 7PCGα Café_random9",
                    [(0, "ſEED"), (10, "_rng_"), (16, "7PCGα"), (22, "Café_random9")],
                )
            ],
        ),
        ("se\u0301ed seed\u0301", [(1, "se\u0301ed seed\u0301", [(6, "seed")])]),
        (
            "٣seed ²rng ⅢPCG",
            [(1, "٣seed ²rng ⅢPCG", [(0, "٣seed"), (6, "²rng"), (11, "ⅢPCG")])],
        ),
        (
            "😀seed\r\n--RNG\npcg",
            [
                (1, "😀seed", [(1, "seed")]),
                (2, "--RNG", [(2, "RNG")]),
                (3, "pcg", [(0, "pcg")]),
            ],
        ),
        (
            "seed\r\nrng\nrandom\rpcg\u2028_SEED\u2029x",
            [
                (1, "seed", [(0, "seed")]),
                (2, "rng", [(0, "rng")]),
                (3, "random", [(0, "random")]),
                (4, "pcg", [(0, "pcg")]),
                (5, "_SEED", [(0, "_SEED")]),
            ],
        ),
        ("--seed=47", [(1, "--seed=47", [(2, "seed")])]),
        (
            "α" + "x" * 256 + "ſEED" + "9" * 256,
            [
                (
                    1,
                    "α" + "x" * 256 + "ſEED" + "9" * 256,
                    [(0, "α" + "x" * 256 + "ſEED" + "9" * 256)],
                )
            ],
        ),
    ],
    ids=[
        "empty",
        "unrelated-words",
        "lookalikes-and-sharp-s",
        "punctuation-and-case",
        "unicode-words-and-long-s",
        "combining-marks",
        "unicode-digit-categories",
        "character-columns",
        "line-separators",
        "last-line",
        "long-whole-word",
    ],
)
def test_should_preserve_whole_tokens_character_columns_and_normalized_line_hashes(
    text: str, expected: list[tuple[int, str, list[tuple[int, str]]]]
) -> None:
    report = inspect_seed_text(text)

    assert report == {
        "seed_lines": [
            {
                "line": number,
                "line_sha256": sha256(line.encode("utf-8")).hexdigest(),
                "references": [{"column": column, "token": token} for column, token in references],
            }
            for number, line, references in expected
        ],
        "python_expressions": [],
        "python_parse_status": "not_python",
        "python_parse_error": None,
        "execution_or_fresh_role_authority": False,
    }


def test_should_retain_every_source_rng_default_and_seed_expression_without_executing() -> None:
    text = "import forbidden\nseed = 7\ndef build(seed=11):\n    rng = forbidden.default_rng(seed + 17)\n    return forbidden.Model(seed=seed + 1001)\n"
    report = inspect_seed_text(text, python_source=True)
    assert report["python_parse_status"] == "parsed"
    expressions = [row["expression"] for row in report["python_expressions"]]
    assert "7" in expressions and "11" in expressions
    assert "forbidden.default_rng(seed + 17)" in expressions
    assert "forbidden.Model(seed=seed + 1001)" in expressions
    assert len(report["seed_lines"]) == 4
    assert not report["execution_or_fresh_role_authority"]


def test_should_retain_symbolic_rng_calls_including_aliases_and_state_restore() -> None:
    report = inspect_seed_text(
        "random.seed(None)\nrng.set_state(saved)\nmodel.rng.standard_normal(size=4)\n",
        python_source=True,
    )
    assert len(report["python_expressions"]) == 3
    assert report["python_expressions"][0]["expression"] == "random.seed(None)"


def test_should_parse_functions_with_no_defaults_and_keep_explicit_calls() -> None:
    report = inspect_seed_text(
        "def build(seed):\n    return unknown.default_rng(seed)\n", python_source=True
    )
    assert report["python_parse_status"] == "parsed"
    assert report["python_expressions"][0]["expression"] == "unknown.default_rng(seed)"


def test_should_keep_invalid_python_and_all_text_locations_visible() -> None:
    report = inspect_seed_text("seed =\nseed_unknown\n", python_source=True)
    assert report["python_parse_status"] == "unparsed"
    assert len(report["seed_lines"]) == 2
    assert report["python_parse_error"]


def test_should_include_last_line_and_csv_or_document_mentions_without_a_newline() -> None:
    report = inspect_seed_text("seed 7\n--seed=11\n--seeds 13,17")
    assert [row["line"] for row in report["seed_lines"]] == [1, 2, 3]
    assert report["python_parse_status"] == "not_python"


def test_should_retain_cli_defaults_and_seed_mapping_fields() -> None:
    report = inspect_seed_text(
        'parser.add_argument("--seed", default=7)\nkwargs = {"seed": seed + 17}\n',
        python_source=True,
    )
    expressions = {row["expression"] for row in report["python_expressions"]}
    assert "parser.add_argument('--seed', default=7)" in expressions
    assert "seed + 17" in expressions


def test_should_retain_alias_and_dynamic_calls_as_unresolved_without_importing_them() -> None:
    report = inspect_seed_text(
        "from forbidden.random import default_rng as sample\nr = forbidden.default_rng\nvalue = sample(7)\nother = r(11)\n",
        python_source=True,
    )
    expressions = {row["expression"] for row in report["python_expressions"]}
    assert "sample(7)" in expressions and "r(11)" in expressions
    assert sum(row["kind"] == "unresolved_call" for row in report["python_expressions"]) == 2


def test_should_extract_all_csv_seed_rows_with_multiline_fields_and_duplicate_headers() -> None:
    result = inspect_seed_csv('seed,seed,note\n7,11,"first\nsecond"\n13,17,end\n')
    assert result["row_count"] == 2
    assert result["duplicate_headers"] == ["seed"]
    assert [row["declaration"]["seed"] for row in result["metadata"]["declarations"]] == [
        7,
        11,
        13,
        17,
    ]
    assert not result["execution_or_fresh_role_authority"]


def test_should_preserve_short_extra_null_and_symbolic_csv_cells() -> None:
    result = inspect_seed_csv("seed,x\n,1\nsymbolic,2,extra\n7\n")
    assert result["row_count"] == 3 and len(result["row_issues"]) == 2
    assert len(result["metadata"]["unresolved"]) == 2
    assert result["metadata"]["declarations"][0]["declaration"]["seed"] == 7


@pytest.mark.parametrize("text", ["", 'seed\n"unfinished'])
def test_should_reject_empty_or_malformed_csv_with_useful_errors(text: str) -> None:
    with pytest.raises(ValueError, match="CSV"):
        inspect_seed_csv(text)
