"""Inspect complete source/text/CSV evidence without executing its contents.

Inputs are complete decoded text. Outputs bind every seed/RNG-bearing line,
Python default/assignment/call expression and CSV seed cell, retaining malformed
or symbolic cases. Literals and expressions do not prove past execution, source
independence or role release. This module owns no filesystem, Git, RNG or IO.
"""

from __future__ import annotations

import ast
from collections import Counter
import csv
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import re
from typing import Any

from src.core.seed_declaration_context import audit_seed_metadata


_REFERENCE = re.compile(r"\b\w*(?:seed|rng|random|pcg)\w*\b", re.IGNORECASE)


def _expression(node: ast.AST, value: ast.AST, kind: str) -> dict[str, Any]:
    return {
        "line": getattr(node, "lineno", 0),
        "column": getattr(node, "col_offset", 0),
        "kind": kind,
        "expression": ast.unparse(value),
        "integer_literals_are_not_seed_role_proof": [
            child.value
            for child in ast.walk(value)
            if isinstance(child, ast.Constant) and type(child.value) is int
        ],
    }


def _python_expressions(tree: ast.AST) -> list[dict[str, Any]]:
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            hinted = (
                bool(_REFERENCE.search(ast.unparse(node.func)))
                or any(
                    keyword.arg is not None and _REFERENCE.search(keyword.arg)
                    for keyword in node.keywords
                )
                or any(
                    isinstance(arg, ast.Constant)
                    and type(arg.value) is str
                    and _REFERENCE.search(arg.value)
                    for arg in node.args
                )
            )
            rows.append(
                _expression(node, node, "call_or_rng_state" if hinted else "unresolved_call")
            )
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and type(key.value) is str
                    and _REFERENCE.search(key.value)
                ):
                    rows.append(_expression(key, value, "mapping_seed_field"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is not None:
                hinted = any(_REFERENCE.search(ast.unparse(target)) for target in targets)
                rows.append(
                    _expression(
                        node,
                        node.value,
                        "assignment_or_default" if hinted else "unresolved_assignment",
                    )
                )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = node.args
            positional = args.posonlyargs + args.args
            start = len(positional) - len(args.defaults)
            for argument, default in zip(positional[start:], args.defaults, strict=True):
                kind = (
                    "parameter_default"
                    if _REFERENCE.search(argument.arg)
                    else "unresolved_parameter_default"
                )
                rows.append(_expression(argument, default, kind))
            for argument, keyword_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                if keyword_default is not None:
                    kind = (
                        "parameter_default"
                        if _REFERENCE.search(argument.arg)
                        else "unresolved_parameter_default"
                    )
                    rows.append(_expression(argument, keyword_default, kind))
    return sorted(
        rows, key=lambda row: (row["line"], row["column"], row["kind"], row["expression"])
    )


def inspect_seed_text(text: str, *, python_source: bool = False) -> dict[str, Any]:
    """Keep all text references, including source whose AST cannot be parsed."""
    if type(text) is not str:
        raise ValueError("seed evidence requires complete decoded text")
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        matches = list(_REFERENCE.finditer(line))
        if matches:
            lines.append(
                {
                    "line": number,
                    "line_sha256": sha256(line.encode()).hexdigest(),
                    "references": [
                        {"column": match.start(), "token": match.group()} for match in matches
                    ],
                }
            )
    expressions: list[dict[str, Any]] = []
    status, error = "not_python", None
    if python_source:
        try:
            expressions = _python_expressions(ast.parse(text))
            status = "parsed"
        except (SyntaxError, ValueError, RecursionError) as failure:
            status, error = "unparsed", str(failure)
    return {
        "seed_lines": lines,
        "python_expressions": expressions,
        "python_parse_status": status,
        "python_parse_error": error,
        "execution_or_fresh_role_authority": False,
    }


def inspect_seed_csv(text: str) -> dict[str, Any]:
    """Retain duplicate columns and every row; do not silently drop malformed cells."""
    reader = csv.reader(StringIO(text), strict=True)
    try:
        columns = next(reader)
        if not columns:
            raise ValueError("CSV seed evidence has an empty header")
        rows, issues = [], []
        for index, values in enumerate(reader):
            if len(values) != len(columns):
                issues.append(
                    {
                        "row": index,
                        "ending_line": reader.line_num,
                        "reason": "CSV row width differs",
                    }
                )
            pairs = [
                [name, values[position] if position < len(values) else None]
                for position, name in enumerate(columns)
            ]
            pairs.extend(
                [
                    [f"unlabeled_cell_{position}", item]
                    for position, item in enumerate(values[len(columns) :], len(columns))
                ]
            )
            rows.append({"dict": pairs})
    except (StopIteration, csv.Error) as error:
        raise ValueError("CSV seed evidence is empty or malformed") from error
    return {
        "row_count": len(rows),
        "columns": columns,
        "duplicate_headers": sorted(name for name, count in Counter(columns).items() if count > 1),
        "row_issues": issues,
        "metadata": asdict(audit_seed_metadata({"rows": rows})),
        "execution_or_fresh_role_authority": False,
    }
