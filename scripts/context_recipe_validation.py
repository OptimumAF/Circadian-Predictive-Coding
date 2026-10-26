"""Two frozen recipe engineering cases through the existing context live runner.

No controller fitting/comparison framework. Preserve paid cases and original
source before admitting the next case; task outcomes never select a recipe.
"""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from time import monotonic
from typing import Any

from scripts import context_head_live as live
from scripts.context_head_probe import HEAD_SOURCE_SHA, _source_bytes
from scripts.context_training_streams import CONTEXT_TRAINING_RECIPES
from src.app.probe_return_phase import paid_report_guard
from src.shared.process_memory import ProcessRssSampler


def run_validation(input_path):
    started = monotonic()
    report: dict[str, Any] = dict(
        definition="context_recipe_validation_v1",
        complete=False,
        cases=[],
        source_sha256=HEAD_SOURCE_SHA,
        prior_training=dict(updates=12, rows=72, mistakes=16),
        final_examples=0,
        probe_updates=0,
        replay_examples=0,
    )
    with ProcessRssSampler(interval_seconds=0.02) as memory:
        try:
            raw = _source_bytes(input_path)
            if sha256(raw).hexdigest() != HEAD_SOURCE_SHA:
                raise ValueError("qualified source replaced")
            cases = report["cases"]
            paid: list[Any] = []
            recipes = deepcopy(CONTEXT_TRAINING_RECIPES)

            def check():
                if monotonic() - started > 360 or memory.snapshot().peak_bytes > 512 * 1024**2:
                    raise ValueError("original two-recipe validation allowance exhausted")
                if _source_bytes(input_path) != raw or CONTEXT_TRAINING_RECIPES != recipes:
                    raise ValueError("original source/declared recipes changed")
                if report["cases"] is not cases or len(cases) != len(paid):
                    report["cases"] = cases
                    cases[:] = [r for r, _, _ in paid]
                    raise ValueError("preceding paid recipe case container replaced")
                for i, (row, expected, guard) in enumerate(paid):
                    if cases[i] is not row:
                        cases[:] = [r for r, _, _ in paid]
                        raise ValueError("preceding paid recipe case replaced")
                    try:
                        guard()
                    except ValueError:
                        row.clear()
                        row.update(deepcopy(expected))
                        raise ValueError("preceding paid recipe case corrupted") from None

            for recipe in recipes:
                check()
                result = live.run_case(input_path, stream_recipe=recipe)
                cases.append(result)
                if result.get("complete") is not True:
                    raise ValueError("paid recipe execution incomplete; no following case")
                if (
                    result.get("context_recipe") != asdict(recipe)
                    or result.get("source_sha256") != HEAD_SOURCE_SHA
                    or result.get("source_unchanged") is not True
                ):
                    raise ValueError("paid recipe differs from requested case")
                guard, expected = paid_report_guard(result)
                paid.append((result, expected, guard))
                check()
            report["complete"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"[:512]
        finally:
            report.update(elapsed_seconds=monotonic() - started, memory=asdict(memory.snapshot()))
    return report
