"""Four fixed TRAIN packets through the existing inbox and native CLM update.

This tiny frozen-feature engineering case qualifies a port, not pretrained task
quality. Only the pinned head/loss function definitions are compiled; upstream
loader and training entry points never run. No encoder or checkpoint is loaded.
"""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from src.adapters.clm_head_learner import CLMHeadLearner
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import CLMHeadPolicy, CLMTargets, FrozenEmbeddingBatch, FrozenEncoderIdentity
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.shared.process_memory import ProcessRssSampler
from src.shared.torch_runtime import require_torch

STAGE = Path("artifacts/runs/clm-head-baseline-20261022")
HEAD_SOURCE = Path("artifacts/runs/native-decision-contracts-20261022/clm/src__clm__heads.py")


def pinned_function(path: Path, name: str, namespace: dict[str, Any]):
    """Extract a single inspected function, never execute upstream module code."""
    tree = ast.parse(path.read_text(encoding="utf8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _equal(left, right, torch) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            type(left) is type(right)
            and left.keys() == right.keys()
            and all(_equal(left[k], right[k], torch) for k in left)
        )
    if type(left) in (tuple, list):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_equal(a, b, torch) for a, b in zip(left, right))
        )
    return bool(left == right)


def run_case(report: dict[str, Any] | None = None) -> dict[str, Any]:
    if report is None:
        report = {}
    report.update(complete=False, rows=[], stream={}, work={})
    torch = require_torch()
    torch.set_num_threads(2)
    started = monotonic()
    namespace = dict(torch=torch, F=torch.nn.functional, PROJ_DIM=512, HIDDEN=4096)
    make_head = pinned_function(HEAD_SOURCE, "make_head", namespace)
    loss_function = pinned_function(STAGE / "upstream-finetune.py", "_clm_loss", namespace)
    caller_rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        state_head = make_head(3, proj=2, hidden=2)
        action_head = make_head(3, proj=2, hidden=2)
    scale = torch.nn.Parameter(torch.tensor(0.0))
    source = deepcopy((state_head.state_dict(), action_head.state_dict(), scale.detach()))
    reference = deepcopy((state_head, action_head, scale))
    parameters = [*reference[0].parameters(), *reference[1].parameters(), reference[2]]
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, total_steps=4, pct_start=0.1, anneal_strategy="cos"
    )
    identity = FrozenEncoderIdentity(
        "fixture-only", "synthetic-v1", "tokenizer-none", "last", "handset-v1", 2
    )
    learner = CLMHeadLearner(
        state_head,
        action_head,
        scale,
        identity=identity,
        policy=CLMHeadPolicy(projection_dim=2, max_updates=4),
    )
    batch = FrozenEmbeddingBatch(
        identity,
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, -1.0]]),
    )
    targets = CLMTargets((0, 0, 1))
    codes = torch.tensor(targets.group_codes, dtype=torch.int64)
    clock = LogicalClock()
    budget = ToyBudgetSession(
        ToyExecutionBudget(
            max_training_updates=4, max_wall_seconds=60, max_process_rss_bytes=2 * 1024**3
        ),
        monotonic,
    )
    rows = report["rows"]
    report.update(
        seed=23,
        identity=asdict(identity),
        states=batch.states.tolist(),
        actions=batch.actions.tolist(),
        group_codes=[0, 0, 1],
    )
    with ProcessRssSampler(interval_seconds=0.01) as sampler:
        budget.attach_memory(sampler)
        inbox = ExperienceInbox(
            learner, clock=clock, budget=budget, learner_version="clm-fixture-v1", max_experiences=4
        )
        try:
            for index in range(4):
                clock.advance_to(2 * index)
                inbox.record_experience(
                    Experience(
                        f"packet-{index}",
                        "clm-fixture",
                        clock.now(),
                        "frozen-fixture-v1",
                        batch,
                        "train",
                        ExperiencePermissions(training=True),
                    )
                )
                assert not inbox.drain(), "an absent label must not update"
                clock.advance_to(2 * index + 1)
                inbox.record_label(
                    LabelArrival(
                        f"label-{index}",
                        f"packet-{index}",
                        "clm-fixture",
                        clock.now(),
                        "frozen-fixture-v1",
                        targets,
                    )
                )
                applied = inbox.drain()
                report["stream"] = dict(
                    completed_updates=budget.updates_completed,
                    training_pairs=learner.training_examples,
                    receipts=[asdict(r) for r in inbox.applied_updates],
                )
                report["work"] = learner.work
                assert len(applied) == 1
                loss, _, _ = loss_function(*reference, batch.states, batch.actions, codes)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                snapshot = learner.snapshot_state()
                comparisons = dict(
                    state_head=_equal(snapshot.state_head, reference[0].state_dict(), torch),
                    action_head=_equal(snapshot.action_head, reference[1].state_dict(), torch),
                    scale=_equal(snapshot.logit_scale, reference[2].detach(), torch),
                    gradients=_equal(snapshot.gradients, tuple(p.grad for p in parameters), torch),
                    optimizer=_equal(snapshot.optimizer_state, optimizer.state_dict(), torch),
                    scheduler=_equal(snapshot.scheduler_state, scheduler.state_dict(), torch),
                    loss=applied[0].diagnostic.value == float(loss.detach()),
                )
                rows.append(dict(update=index + 1, loss=float(loss.detach()), parity=comparisons))
                assert all(comparisons.values()), rows[-1]
                budget.before_final()
            source_unchanged = _equal(
                source, (state_head.state_dict(), action_head.state_dict(), scale.detach()), torch
            )
            rng_unchanged = torch.equal(caller_rng, torch.get_rng_state())
            assert source_unchanged and rng_unchanged
            budget.record_memory_snapshot()
            memory = asdict(sampler.snapshot())
        finally:
            report["stream"] = dict(
                completed_updates=budget.updates_completed,
                training_pairs=learner.training_examples,
                receipts=[asdict(r) for r in inbox.applied_updates],
            )
            report["work"] = learner.work
            report["sampled_process_rss"] = asdict(sampler.snapshot())
            report["elapsed_seconds"] = monotonic() - started
    report.update(
        complete=True,
        scope="tiny frozen-feature component; no pretrained task claim",
        torch_version=torch.__version__,
        seed=23,
        identity=asdict(identity),
        states=batch.states.tolist(),
        actions=batch.actions.tolist(),
        group_codes=[0, 0, 1],
        rows=rows,
        source_unchanged=source_unchanged,
        caller_rng_unchanged=rng_unchanged,
        stream=dict(
            completed_updates=budget.updates_completed,
            training_pairs=12,
            receipts=[asdict(r) for r in inbox.applied_updates],
        ),
        work=learner.work,
        native_reference_updates=4,
        head_generation=learner.generation,
        elapsed_seconds=monotonic() - started,
        sampled_process_rss=memory,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace existing experiment evidence")
    result: dict[str, Any] = {}
    try:
        run_case(result)
    except BaseException as error:
        result.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )


if __name__ == "__main__":
    main()
