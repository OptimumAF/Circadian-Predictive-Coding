"""Fixed conventional CLM acquisition/new-task/return on known TRAIN strings.

Compose existing frozen encoder, native head learner and delayed-label inbox.
No final/probe inputs, selection, encoder training, downloads or projection cache.
Known-example diagnostics measure rehearsal, not generalization or CPC benefit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic
from typing import Any

from scripts.run_clm_frozen_choice import INSTRUCTIONS, fixed_request
from src.app.experience_inbox import ExperienceInbox
from src.app.toy_execution_budget import ToyBudgetSession, ToyExecutionBudget
from src.core.clm_head import CLMHeadPolicy, CLMTargets, FrozenEmbeddingBatch, _exact
from src.core.experience import Experience, ExperiencePermissions, LabelArrival, LogicalClock
from src.infra.clm_local_loader import load_local_clm
from src.shared.process_memory import ProcessRssSampler

IDS = ("password", "parcel", "refund")
PHASES = (("A", (0, 1)), ("B", (2, 1)), ("return_A", (0, 1)))
POINTS = ("baseline0", "after_A4", "after_B8", "after_return12")


def fixed_texts():
    request = fixed_request()
    queries = (
        request.query,
        "Where is my parcel and when will it arrive?",
        "I was charged twice and need an invoice refund.",
    )
    states = tuple(query + "\n\n" + INSTRUCTIONS for query in queries)
    actions = tuple(candidate.description for candidate in request.allowed_candidates)
    return states, actions


def score_readouts(readouts):
    """Score already returned outputs; no learner or training decision is used."""
    import numpy as np
    from src.core.categorical_probability_scores import probability_scores

    if type(readouts) is not list or len(readouts) != 4:
        raise ValueError("four fixed complete unscored boundaries required")
    scores = {}
    for index, row in enumerate(readouts):
        if row["name"] != POINTS[index] or row["completed_updates"] != index * 4:
            raise ValueError("readout boundary order or update count differs")
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        if probabilities.shape != (3, 3):
            raise ValueError("each boundary requires three tasks and three candidates")
        targets = np.arange(3, dtype=np.float64).reshape(3, 1)
        scores[row["name"]] = dict(
            all=probability_scores(probabilities, targets, 3),
            tasks={
                name: probability_scores(probabilities[i : i + 1], targets[i : i + 1], 3)
                for i, name in enumerate(IDS)
            },
            selected_ids=[IDS[int(i)] for i in np.argmax(probabilities, axis=1)],
        )
    return scores


def _stream_status(learner, inbox, budget):
    return dict(
        completed_updates=budget.updates_completed,
        training_pairs=learner.training_examples,
        receipts=[asdict(receipt) for receipt in inbox.applied_updates],
    )


def run_stream(learner, batch, report, *, before_step=lambda: None, save=lambda: None):
    """Use ordinary delayed TRAIN receipts; score only after all updates finish."""
    _exact(batch, FrozenEmbeddingBatch)
    if (
        batch.identity != learner.identity
        or learner.completed_updates
        or learner.training_examples
        or learner.generation
        or any(
            learner.work[key]
            for key in ("train_attempts", "prediction_attempts", "restore_attempts", "commits")
        )
        or learner.policy.max_updates != 12
    ):
        raise ValueError("fresh twelve-update learner with matching frozen identity required")
    if (
        tuple(batch.states.shape) != (3, batch.identity.embedding_dim)
        or batch.actions.shape != batch.states.shape
    ):
        raise ValueError("stream requires three paired tasks under the frozen identity")
    # Own raw features so the producer/caller cannot be modified by packet slicing.
    batch = FrozenEmbeddingBatch(
        batch.identity, batch.states.detach().clone(), batch.actions.detach().clone()
    )
    clock = LogicalClock()
    budget = ToyBudgetSession(ToyExecutionBudget(max_training_updates=12), monotonic)
    inbox = ExperienceInbox(
        learner,
        clock=clock,
        budget=budget,
        learner_version="clm-head-stream-v1",
        max_experiences=12,
    )
    report.update(stream_complete=False, raw_readouts=[], rows=[])

    def refresh():
        report["stream"] = _stream_status(learner, inbox, budget)
        report["learner_work"] = learner.work
        save()

    def readout(name):
        before_step()
        probabilities = learner.predict(batch)
        report["raw_readouts"].append(
            dict(
                name=name,
                completed_updates=learner.completed_updates,
                probabilities=probabilities.tolist(),
            )
        )
        if name == POINTS[0] and "control_probabilities" in report:
            error = max(
                abs(a - b)
                for actual, control in zip(
                    probabilities.tolist(), report["control_probabilities"], strict=True
                )
                for a, b in zip(actual, control, strict=True)
            )
            report["initial_control_max_abs_error"] = error
            if error > 2e-6:
                raise ValueError("initial learner differs from frozen released control")
        refresh()

    try:
        readout(POINTS[0])
        for phase_index, (phase, indices) in enumerate(PHASES):
            paired = FrozenEmbeddingBatch(
                batch.identity, batch.states[list(indices)], batch.actions[list(indices)]
            )
            for _ in range(4):
                before_step()
                index = learner.completed_updates
                clock.advance_to(2 * index)
                key = f"clm-head-{index}"
                inbox.record_experience(
                    Experience(
                        key,
                        "clm-head-stream",
                        clock.now(),
                        "frozen-clm-v1",
                        paired,
                        "train",
                        ExperiencePermissions(training=True),
                    )
                )
                if inbox.drain():
                    raise ValueError("an absent TRAIN label unexpectedly admitted work")
                clock.advance_to(2 * index + 1)
                before_step()
                inbox.record_label(
                    LabelArrival(
                        f"label-{index}",
                        key,
                        "clm-head-stream",
                        clock.now(),
                        "frozen-clm-v1",
                        CLMTargets(indices),
                    )
                )
                applied = inbox.drain()
                if len(applied) != 1:
                    raise ValueError("one delayed TRAIN receipt expected")
                report["rows"].append(
                    dict(update=index + 1, phase=phase, loss=applied[0].diagnostic.value)
                )
                refresh()
            readout(POINTS[phase_index + 1])
        # Diagnostic labels are intentionally unavailable to the update loop.
        report["scores"] = score_readouts(report["raw_readouts"])
        report["stream_complete"] = True
    finally:
        refresh()


def _control_readout(loaded, batch):
    import torch
    from torch.nn import functional as F

    with torch.no_grad():
        states = F.normalize(loaded.state_head(batch.states), dim=-1)
        actions = F.normalize(loaded.action_head(batch.actions), dim=-1)
        logits = loaded.logit_scale.exp().clamp(max=100.0) * states @ actions.t()
        return torch.softmax(logits, dim=-1).tolist()


def _usefulness(scores):
    def correct(point, task):
        return scores[point]["tasks"][task]["accuracy"] == 1.0

    return dict(
        A_endpoint_pass=correct("after_A4", "password") and correct("after_A4", "parcel"),
        B_endpoint_pass=correct("after_B8", "refund") and correct("after_B8", "parcel"),
        return_A_endpoint_pass=correct("after_return12", "password")
        and correct("after_return12", "parcel"),
        newly_acquired_password=(
            not correct("baseline0", "password") and correct("after_A4", "password")
        ),
        refund_corrected_during_B=(
            not correct("after_A4", "refund") and correct("after_B8", "refund")
        ),
        new_refund_acquisition=(
            not correct("baseline0", "refund")
            and not correct("after_A4", "refund")
            and correct("after_B8", "refund")
        ),
        retained_password_after_B=(
            correct("after_A4", "password") and correct("after_B8", "password")
        ),
        recovered_password=(
            correct("after_A4", "password")
            and not correct("after_B8", "password")
            and correct("after_return12", "password")
        ),
    )


def run_case(model_dir: Path, head: Path, report: dict[str, Any], *, save=lambda: None):
    states, actions = fixed_texts()
    import torch
    from src.adapters.clm_decision import make_clm_head
    from src.adapters.clm_frozen_encoder import FrozenCLMEncoder
    from src.adapters.clm_head_learner import CLMHeadLearner

    torch.set_num_threads(2)
    started = monotonic()
    caller_rng = torch.get_rng_state().clone()
    report.update(
        complete=False,
        events=[],
        role="train",
        scope="known-TRAIN conventional CLM head learning; no generalization/CPC/serving claim",
    )
    save()
    encoder = learner = None
    with ProcessRssSampler(interval_seconds=0.02) as sampler:

        def before_step():
            if monotonic() - started > 450:
                raise ValueError("native inner wall allowance exhausted")
            rss = sampler.sample()
            if rss is None or rss > 24 * 1024**3:
                raise ValueError("native RSS unavailable or allowance exceeded")

        def progress(event):
            report["events"].append(dict(elapsed_seconds=monotonic() - started, **event))
            save()

        try:
            loaded = load_local_clm(
                model_dir,
                head,
                head_factory=make_clm_head,
                before_step=before_step,
                progress=progress,
            )
            report.update(identity=asdict(loaded.identity), torch_version=torch.__version__)
            encoder = FrozenCLMEncoder(
                loaded.model,
                loaded.tokenizer,
                identity=loaded.identity,
                max_sequences=6,
                max_forward_calls=1,
            )

            def guard(_model, args, kwargs):
                ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
                if (
                    ids.shape != mask.shape
                    or ids.ndim != 2
                    or ids.shape[0] != 6
                    or ids.shape[1] > 96
                    or ids.numel() > 576
                    or kwargs.get("use_cache") is not False
                ):
                    raise ValueError("native six-sequence token allowance exceeded")
                before_step()
                progress(
                    dict(event="encoder_forward_attempt", sequences=6, padded_slots=ids.numel())
                )

            handle = loaded.model.register_forward_pre_hook(guard, with_kwargs=True)
            try:
                batch = encoder.training_batch(states, actions)
            finally:
                handle.remove()
            token_facts = encoder.last_readout
            report["tokens"] = {
                key: token_facts[key] for key in ("token_ids", "retained_lengths", "padded_tokens")
            }
            report["texts"] = dict(states=states, actions=actions)
            progress(
                dict(
                    event="features_returned",
                    feature_elements=batch.states.numel() + batch.actions.numel(),
                )
            )
            original = deepcopy(
                (
                    loaded.state_head.state_dict(),
                    loaded.action_head.state_dict(),
                    loaded.logit_scale.detach(),
                )
            )
            before_step()
            report["control_probabilities"] = _control_readout(loaded, batch)
            save()
            # Input copies become trainable; neither the released control nor
            # the large encoder is mutated or transferred to the optimizer.
            train_state, train_action = deepcopy(loaded.state_head), deepcopy(loaded.action_head)
            train_state.requires_grad_(True)
            train_action.requires_grad_(True)
            learner = CLMHeadLearner(
                train_state,
                train_action,
                loaded.logit_scale.detach().clone(),
                identity=loaded.identity,
                policy=CLMHeadPolicy(projection_dim=512, max_updates=12, max_rows=3),
            )
            del train_state, train_action
            run_stream(learner, batch, report, before_step=before_step, save=save)
            report["control_scores"] = score_readouts(
                [
                    dict(
                        name=name,
                        completed_updates=i * 4,
                        probabilities=report["control_probabilities"],
                    )
                    for i, name in enumerate(POINTS)
                ]
            )
            report["usefulness"] = _usefulness(report["scores"])
            current = (
                loaded.state_head.state_dict(),
                loaded.action_head.state_dict(),
                loaded.logit_scale.detach(),
            )
            unchanged = all(
                torch.equal(original[i][key], current[i][key])
                for i in (0, 1)
                for key in original[i]
            ) and torch.equal(original[2], current[2])
            encoder._check()
            report.update(
                control_unchanged=unchanged,
                caller_rng_unchanged=torch.equal(caller_rng, torch.get_rng_state()),
            )
            if not unchanged or not report["caller_rng_unchanged"]:
                raise ValueError("frozen control or caller RNG changed")
            before_step()
            report["complete"] = True
        finally:
            if encoder is not None:
                report["encoder_work"] = encoder.work
            if learner is not None:
                report["learner_work"] = learner.work
            report.update(
                elapsed_seconds=monotonic() - started,
                sampled_process_rss=asdict(sampler.snapshot()),
            )
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace existing adaptation evidence")
    report: dict[str, Any] = {}

    def save():
        args.output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8"
        )

    try:
        run_case(args.model_dir, args.head, report, save=save)
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
