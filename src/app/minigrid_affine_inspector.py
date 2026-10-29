"""Recorded affine TRAIN curves, causal receipts and native public-view playback.

Consumes bounded JSON and returns detached source-linked presentation data. It
does not restore weights, authenticate history, run models or select experiments.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .minigrid_inspector_projection import (
    ACTIONS,
    _boolean,
    _close,
    _envelope,
    _episode,
    _integer,
    _number,
    _records,
    _require,
    _status,
    _initial,
    decode_public_observation,
)
from .minigrid_learnability_inspector import _flatten, _missing, _proofs

DEFINITION = "portable_affine_train_and_play_v1"
METHOD = "arrived_public_cue_affine_fit"
POINTS = (0, 2, 4, 6)
FIT_ROWS = (9, 9, 18, 18, 27, 27)
TOTAL_ROWS = (0, 18, 54, 108)
SOURCES = tuple(
    (f"minigrid-source-{context}-{cue}", context, cue)
    for context in ("A", "B", "C_train")
    for cue in ("blue", "red")
)
SLOTS = tuple((context, cue) for context in ("B", "C_train") for cue in ("blue", "red"))
LAYOUTS = dict(
    B=dict(context="B", rotation=1, branch_row=4, swap_keys=True, wall_color="blue"),
    C_train=dict(context="C_train", rotation=2, branch_row=3, swap_keys=True, wall_color="red"),
)


@dataclass(frozen=True)
class AffineRecipe:
    """Fixed recorded formats, never user-supplied training configuration."""

    name: str
    training_definition: str
    points: tuple[int, ...]
    fit_rows: tuple[int, ...]
    total_rows: tuple[int, ...]
    sources: tuple[tuple[str, str, str], ...]
    slots: tuple[tuple[str, str], ...]
    layouts: dict[str, dict[str, Any]]
    game_role: str
    phase: str
    catalogue: dict[str, int]


NORMAL = AffineRecipe(
    "normal_train",
    "minigrid_arrived_abc_capacity2_train_stream_v1",
    POINTS,
    FIT_ROWS,
    TOTAL_ROWS,
    SOURCES,
    SLOTS,
    LAYOUTS,
    "knownTRAIN_BC_afterC",
    "Known after C",
    dict(
        arrival_row_exposures=54,
        replay_row_exposures=54,
        fit_row_exposures=108,
        peak_bank_rows=36,
        input_array_bytes=256608,
        peak_joint_template_array_bytes=64152,
        peak_bank_observation_target_float64_equivalent_bytes=170784,
    ),
)
RECURRING_DEFINITION = "portable_affine_recurring_train_v1"
RECURRING = AffineRecipe(
    "recurring_train",
    "minigrid_arrived_recurring_capacity3_train_stream_v1",
    (0, 2, 4, 6, 8, 10),
    (9, 9, 18, 18, 27, 27, 36, 36, 36, 36),
    (0, 18, 54, 108, 180, 252),
    tuple(
        (f"minigrid-source-{c}-{q}", c, q)
        for c in ("A", "B", "C_train", "G_train")
        for q in ("blue", "red")
    ),
    tuple((c, q) for c in ("G_train", "A") for q in ("blue", "red")),
    dict(
        G_train=dict(
            context="G_train", rotation=0, branch_row=5, swap_keys=False, wall_color="grey"
        ),
        A=dict(context="A", rotation=0, branch_row=3, swap_keys=False, wall_color="grey"),
    ),
    "knownTRAIN_GA_return",
    "Known G and returned A",
    dict(
        arrival_row_exposures=90,
        replay_row_exposures=162,
        fit_row_exposures=252,
        peak_bank_rows=54,
        input_array_bytes=598752,
        peak_joint_template_array_bytes=85536,
        peak_bank_observation_target_float64_equivalent_bytes=256176,
    ),
)

A4_DEFINITION = "portable_affine_a4_adaptation_v1"
A4 = AffineRecipe(
    "a4_train",
    "minigrid_affine_a3_a4_a3_train_v1",
    POINTS,
    (9, 9, 18, 18, 18, 18),
    (0, 18, 54, 90),
    tuple(
        (f"minigrid-source-{identity}-{cue}", label, cue)
        for identity, label in (("A", "A3"), ("A4-train", "A4"))
        for cue in ("blue", "red")
    ),
    tuple((label, cue) for label in ("A3", "A4") for cue in ("blue", "red")),
    {
        label: dict(context="A", rotation=0, branch_row=row, swap_keys=False, wall_color="grey")
        for label, row in (("A3", 3), ("A4", 4))
    },
    "knownTRAIN_A3A4_adaptation",
    "A3/A4 TRAIN",
    dict(
        arrival_row_exposures=54,
        replay_row_exposures=36,
        fit_row_exposures=90,
        peak_bank_rows=18,
        input_array_bytes=213840,
        peak_joint_template_array_bytes=42768,
        peak_bank_observation_target_float64_equivalent_bytes=85392,
    ),
)


def _object(value, name):
    _require(type(value) is dict, f"invalid {name}")
    return value


def _fixed(record, key, value):
    _require(type(record.get(key)) is type(value) and record[key] == value, f"invalid {key}")


def _budget(record, updates, *, readout=False, recipe=NORMAL, done=False):
    _object(record, "budget")
    limits = _object(record.get("limits"), "budget limits")
    _require(record.get("initial_limits") == limits, "original budget limits changed")
    for key, value in dict(
        max_training_updates=0 if readout else recipe.points[-1],
        max_wall_seconds=150 if readout else 90,
        max_process_rss_bytes=512 * 1024**2,
    ).items():
        _fixed(limits, key, value)
    _require(limits.get("max_replay_examples") == (0 if readout else None), "replay limit mismatch")
    _fixed(record, "updates_completed", updates)
    _fixed(record, "replay_examples_completed", 0)
    start = _number(record.get("started_at"), "budget start", 0)
    end = _number(record.get("ended_at"), "budget end", start)
    if recipe is NORMAL or not readout or "elapsed_seconds" in record:
        _number(record.get("elapsed_seconds"), "budget elapsed", 0)
    if recipe is RECURRING:
        _require(
            not done or end - start <= limits["max_wall_seconds"],
            "completed budget exceeds original time limit",
        )
        if readout:
            _require(record.get("observed_budget_started_at") == start, "readout clock renewed")


def _state(record, point, recipe=NORMAL):
    _object(record, "native checkpoint")
    _fixed(record, "completed_updates", point)
    _fixed(record, "training_examples", recipe.total_rows[recipe.points.index(point)])
    _require(record.get("failed_work") is None, "observed checkpoint contains failed work")
    heads = record.get("heads")
    _require(type(heads) is list and len(heads) == 2, "native checkpoint requires two heads")
    for head in heads:
        if point == 0:
            _require(head is None, "untrained checkpoint contains fitted head")
        else:
            _fixed(_object(head, "head"), "fit_rows", recipe.fit_rows[point - 1])


def _score(score):
    _object(score, "TRAIN score")
    for key in ("actions", "expected"):
        values = score.get(key)
        _require(type(values) is list and len(values) == 9, "TRAIN score requires nine rows")
        for value in values:
            _integer(value, 0, 6, key)
    correct = sum(a == b for a, b in zip(score["actions"], score["expected"], strict=True))
    _close(score.get("accuracy"), correct / 9, "TRAIN accuracy")
    _number(score.get("ce"), "recorded TRAIN CE", 0)


def _endpoints(record, done, recipe=NORMAL):
    endpoints = _records(record.get("endpoints", []), len(recipe.points), "TRAIN endpoints")
    _require(
        [e.get("updates") for e in endpoints] == list(recipe.points[: len(endpoints)]),
        "TRAIN checkpoint order mismatch",
    )
    for i, endpoint in enumerate(endpoints):
        if recipe in (RECURRING, A4):
            _fixed(endpoint, "updates", recipe.points[i])
        _state(endpoint.get("state"), recipe.points[i], recipe)
        scores = _object(endpoint.get("scores"), "TRAIN scores")
        _require(set(scores) <= {s[0] for s in recipe.sources}, "non-TRAIN or unknown score source")
        _require(
            (not done and i == len(endpoints) - 1) or len(scores) == len(recipe.sources),
            "incomplete preceding TRAIN scores",
        )
        for score in scores.values():
            _score(score)
    _require(not done or len(endpoints) == len(recipe.points), "complete TRAIN lacks checkpoints")
    calls = sum(len(e["scores"]) for e in endpoints)
    for key, value in dict(score_calls_completed=calls, score_rows_completed=calls * 9).items():
        if key in record:
            _fixed(record, key, value)
    if "score_calls_attempted" in record:
        attempts = _integer(
            record["score_calls_attempted"],
            calls,
            min(len(recipe.points) * len(recipe.sources), calls + 1),
            "score attempts",
        )
        _fixed(record, "score_rows_attempted", attempts * 9)
    return endpoints


def _catalogue(record, receipts, done, recipe=NORMAL):
    wakes, costs = [], []
    value: Any
    catalogue = record.get("catalogue")
    if catalogue is not None:
        _object(catalogue, "catalogue")
        for key, value in recipe.catalogue.items():
            _fixed(catalogue, key, value)
        wakes = _records(catalogue.get("wakes"), recipe.points[-1], "catalogue wakes")
        _require(len(wakes) == recipe.points[-1], "catalogue plan lacks wakes")
        for i, wake in enumerate(wakes):
            for key, value in dict(
                observed_at=i,
                arrived_at=i + 1,
                cue=("blue", "red")[i % 2],
                current_rows=9,
                replay_rows=recipe.fit_rows[i] - 9,
            ).items():
                _fixed(wake, key, value)
            _integer(
                wake.get("bank_rows_after"), 0, recipe.catalogue["peak_bank_rows"], "bank rows"
            )
        if recipe is RECURRING:
            _recurring_wakes(wakes)
        elif recipe is A4:
            _a4_wakes(wakes)
        committed = catalogue.get("committed_rows")
        if committed is not None:
            n = len(receipts)
            _require(
                committed
                == dict(
                    arrivals=9 * n,
                    replay=sum(recipe.fit_rows[:n]) - 9 * n,
                    native=sum(recipe.fit_rows[:n]),
                ),
                "committed catalogue exposure disagrees with receipts",
            )
        _require(not done or committed is not None, "complete catalogue lacks charged exposure")
        costs.extend(
            _flatten(catalogue, "/training/catalogue", "Catalogue (plan and committed prefix)")
        )
    _require(not done or catalogue is not None, "complete TRAIN lacks catalogue costs")
    return wakes, costs


def _curves(endpoints, recipe=NORMAL):
    curves = []
    for i, point in enumerate(recipe.points):
        scores = endpoints[i]["scores"] if i < len(endpoints) else {}
        for identity, context, cue in recipe.sources:
            score = scores.get(identity)
            curves.append(
                dict(
                    source_updates=0,
                    phase="Chronological TRAIN",
                    context=context,
                    cue=cue,
                    role="train",
                    updates=point,
                    status="complete" if score is not None else "not_run",
                    ce=None if score is None else score["ce"],
                    accuracy=None if score is None else score["accuracy"],
                    pointer=None if score is None else f"/training/endpoints/{i}/scores/{identity}",
                )
            )
    return curves


def _training(record, complete, recipe=NORMAL):
    value: Any
    endpoints, receipts, wakes, costs, proofs = [], [], [], [], []
    if record is not None:
        _object(record, "TRAIN report")
        _require(_status(record.get("status")), "invalid TRAIN status")
        _fixed(record, "definition", recipe.training_definition)
        done = record["status"] == "complete"
        _require(not complete or done, "complete workflow lacks complete TRAIN")
        for key in ("source_unchanged", "actor_unchanged"):
            if record.get(key) is not None:
                _boolean(record[key], key)
            _require(not done or record.get(key) is True, "complete TRAIN lacks preservation proof")
        endpoints = _endpoints(record, done, recipe)
        receipt_evidence = record.get("committed_receipts")
        receipts = (
            []
            if receipt_evidence is None
            else _records(receipt_evidence, recipe.points[-1], "TRAIN receipts")
        )
        for i, receipt in enumerate(receipts):
            for key, value in dict(
                observed_at=i, arrived_at=i + 1, applied_at=i + 1, update_number=i + 1
            ).items():
                _fixed(receipt, key, value)
        _require(not done or len(receipts) == recipe.points[-1], "complete TRAIN lacks receipts")
        _require(
            not done or record.get("budget") is not None, "complete TRAIN lacks original budget"
        )
        if record.get("budget") is not None:
            _object(record["budget"], "TRAIN budget")
            # Why: a failed receipt readback does not erase permanently charged
            # updates. Preserve the observed budget with an unknown receipt prefix.
            charged = (
                len(receipts)
                if receipt_evidence is not None
                else _integer(
                    record["budget"].get("updates_completed"),
                    0,
                    recipe.points[-1],
                    "charged TRAIN updates",
                )
            )
            _budget(record["budget"], charged, recipe=recipe, done=done)
            costs.extend(_flatten(record["budget"], "/training/budget", "TRAIN budget"))
        _require(
            not done or record.get("receipt_capture_error") is None,
            "complete receipt capture failed",
        )
        if done:
            stream = _object(record.get("stream"), "completed stream")
            for key, value in dict(
                completed_updates=recipe.points[-1],
                training_examples=recipe.total_rows[-1],
                score_forward_calls=0,
                evaluation_scope="training_only",
                num_classes=7,
            ).items():
                _fixed(stream, key, value)
            _require(stream.get("applied_updates") == receipts, "stream/receipt evidence differs")
        wakes, catalogue_costs = _catalogue(record, receipts, done, recipe)
        costs.extend(catalogue_costs)
        proofs.extend(_proofs(record, "/training"))
        costs.extend(
            _flatten(
                {k: record[k] for k in record if k.startswith("score_")},
                "/training",
                "TRAIN diagnostics",
            )
        )
        if "failure_work" in record:
            costs.extend(
                _flatten(
                    dict(failure_work=record["failure_work"]), "/training", "Failed TRAIN work"
                )
            )
    else:
        _require(not complete, "complete workflow lacks TRAIN report")
    return _curves(endpoints, recipe), receipts, wakes, costs, proofs


def _recurring_wakes(wakes):
    """Check the recorded arrived-only three-episode banks, without fitting."""
    bank: dict[str, list[tuple[str, int, str]]] = {q: [] for q in ("blue", "red")}
    for i, wake in enumerate(wakes):
        cue = ("blue", "red")[i % 2]
        context = ("A", "B", "C_train", "G_train", "A")[i // 2]
        new = (f"minigrid-source-{context}-{cue}:wake:{i}", i + 1, context)
        old = bank[cue]
        bank[cue] = [*old, new][-3:]
        expected = dict(
            current_episode_id=new[0],
            retained_episode_id=old[-1][0] if old else None,
            retained_arrived_at=old[-1][1] if old else None,
            retained_episode_ids=[v[0] for v in old],
            retained_arrival_ticks=[v[1] for v in old],
            bank_episode_ids_after=[v[0] for q in ("blue", "red") for v in bank[q]],
            bank_rows_after=min(i + 1, 6) * 9,
            source_spans=[
                dict(episode_id=v[0], context=v[2], start=j * 9, stop=(j + 1) * 9)
                for j, v in enumerate([*old, new])
            ],
        )
        for key, value in expected.items():
            _fixed(wake, key, value)


def _recurring_protocol(source):
    protocol = _object(source.get("protocol"), "recurring protocol")
    expected = dict(
        definition=RECURRING_DEFINITION,
        new_layout=RECURRING.layouts["G_train"],
        returned_layout=RECURRING.layouts["A"],
        new_fit_calls=10,
        new_fit_rows=252,
        historical_model_reads=0,
        readout_training_updates=0,
        readout_replay_examples=0,
        max_workflow_seconds=300,
        work_deadline_seconds=290,
        max_process_rss_bytes=512 * 1024**2,
    )
    for key, value in expected.items():
        _fixed(protocol, key, value)
    training = _object(protocol.get("training"), "recurring training protocol")
    expected_train = dict(
        definition=RECURRING.training_definition,
        label_delay=1,
        checkpoint_updates=list(RECURRING.points),
        fit_calls=10,
        fit_rows=252,
        score_calls=48,
        score_rows=432,
        classes=7,
        seed=None,
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
        arrival_row_exposures=90,
        catalogue_replay_row_exposures=162,
        distinct_source_rows=72,
    )
    for key, value in expected_train.items():
        _fixed(training, key, value)
    _require(
        all(type(v) is int for v in training["checkpoint_updates"]), "invalid checkpoint types"
    )
    costs = _flatten(
        {k: protocol[k] for k in expected}, "/protocol", "Protocol plan (not paid work)"
    )
    costs += _flatten(training, "/protocol/training", "TRAIN protocol plan (not paid work)")
    return costs


def _teacher_transitions(record, *, require_success=True):
    transitions = _records(record.get("transitions", []), 16, "teacher transitions")
    attempted = _integer(record.get("environment_steps_attempted"), 0, 16, "teacher step attempts")
    returned = _integer(record.get("environment_steps"), 0, attempted, "teacher returned steps")
    _require(len(transitions) == attempted, "teacher attempts contradict recorded transitions")
    _require(
        sum(t.get("applied") is True for t in transitions) == returned,
        "teacher applied steps differ",
    )
    prior = (
        _initial(record["initial_observation"])
        if record.get("initial_observation") is not None
        else None
    )
    terminal, rewards = False, []
    for step in transitions:
        _require(not terminal, "teacher continued after terminal observation")
        decode_public_observation(step.get("observation"))
        decode_public_observation(step.get("next_observation"))
        _require(
            prior is None or prior == step["observation"], "teacher observation continuity differs"
        )
        _integer(step.get("action"), 0, 6, "teacher action")
        _require(
            _boolean(step.get("attempted"), "teacher attempted"), "unattempted teacher transition"
        )
        _boolean(step.get("applied"), "teacher applied")
        rewards.append(_number(step.get("reward"), "teacher reward", -1, 1))
        terminal = _boolean(step.get("terminated"), "teacher terminated") | _boolean(
            step.get("truncated"), "teacher truncated"
        )
        prior = step["next_observation"]
    _close(record.get("total_return"), sum(rewards), "teacher accumulated return")
    if transitions:
        for key in ("terminated", "truncated"):
            _require(
                _boolean(record.get(key), key) == transitions[-1][key], "teacher ending differs"
            )
    if record.get("status") == "complete":
        _require(attempted == returned and terminal, "complete teacher has unfinished steps")
        if require_success:
            _require(
                attempted == 9 and record["total_return"] > 0,
                "complete teacher lacks nine successful steps",
            )


def _recurring_collection(source, required, recipe=RECURRING):
    is_a4 = recipe is A4
    value: Any
    records = _records(source.get("collection", []), 2, "G teacher collection")
    good = True
    costs = []
    for i, record in enumerate(records):
        _require(good, "collection continues after incomplete teacher evidence")
        # A lost native readback may retain only an unknown-work marker. It can
        # never admit fitting, another source, or a claim about its unobserved role.
        unknown = (
            record.get("context") is None and record.get("failure_work_may_be_unobserved") is True
        )
        if not unknown:
            cue = ("blue", "red")[i]
            for key, value in dict(
                context="A" if is_a4 else "G_train",
                cue=cue,
                role="train",
                episode_id=f"minigrid-source-{'A4-train' if is_a4 else 'G_train'}-{cue}",
            ).items():
                _fixed(record, key, value)
            if record.get("status") != "construction_failed":
                _require(_status(record.get("status")), "invalid teacher status")
                for key, value in dict(
                    environment_id="Circadian-MiniGrid-CueKeyDoor-v1",
                    encoding_id="minigrid_cue_key_door_public_image_direction_148_v1",
                    observation_dim=148,
                    num_classes=7,
                    seed=17,
                    max_steps=16,
                    layout=recipe.layouts["A4" if is_a4 else "G_train"],
                    reset_calls=1,
                    action_names=list(ACTIONS),
                ).items():
                    _fixed(record, key, value)
                if record.get("closed") is not None:
                    _boolean(record["closed"], "teacher closed")
                _teacher_transitions(
                    record,
                    require_success=not is_a4
                    or not (record.get("primary_error") or record.get("cleanup_error")),
                )
        good = (
            not unknown
            and record.get("status") == "complete"
            and record.get("closed") is True
            and not record.get("primary_error")
            and not record.get("cleanup_error")
        )
        costs += _flatten(
            {
                k: v
                for k, v in record.items()
                if k not in ("transitions", "initial_observation", "action_names")
                and not (is_a4 and k == "total_return")
            },
            f"/collection/{i}",
            "Original A4 TRAIN teacher collection"
            if is_a4
            else "Original G TRAIN teacher collection",
        )
        if is_a4 and "total_return" in record:
            # Returns are signed outcomes, unlike nonnegative compute counters.
            costs.append(
                dict(
                    label="Original A4 teacher return",
                    value=record["total_return"],
                    pointer=f"/collection/{i}/total_return",
                )
            )
    _require(
        not required
        or len(records) == 2
        and good
        and (is_a4 or source.get("collection_capture_truncated") is False),
        "TRAIN requires two complete closed original G sources",
    )
    return costs


def _recurring_scope(source, complete):
    costs = _recurring_protocol(source)
    training = source.get("training")
    costs += _recurring_collection(source, training is not None or complete)
    phases = source.get("attempted_phases", [])
    _require(
        type(phases) is list and phases == ["collection", "training", "games"][: len(phases)],
        "recurring phase order differs",
    )
    _require(
        (training is None or "training" in phases)
        and (source.get("games") is None or "games" in phases),
        "recorded work precedes phase admission",
    )
    _require(not complete or len(phases) == 3, "complete workflow lacks phase admissions")
    budget = source.get("workflow_budget")
    _require(not complete or budget is not None, "complete workflow lacks original budget")
    if budget is not None:
        _object(budget, "workflow budget")
        for key, value in dict(
            max_wall_seconds=300, work_deadline_seconds=290, max_process_rss_bytes=512 * 1024**2
        ).items():
            _fixed(budget, key, value)
        start = _number(budget.get("started_at"), "workflow start", 0)
        end = _number(budget.get("ended_at"), "workflow end", start)
        _number(budget.get("elapsed_seconds"), "workflow elapsed", 0)
        if budget.get("sampled_peak_rss_bytes") is not None:
            _integer(budget["sampled_peak_rss_bytes"], 0, 2**53 - 1, "workflow RSS")
        for phase in (training, dict(budget=source.get("readout_budget"))):
            nested = phase.get("budget") if type(phase) is dict else None
            if nested is not None:
                _object(nested, "phase budget")
                _number(nested.get("started_at"), "phase start", start, end)
                _number(nested.get("ended_at"), "phase end", start, end)
                _require(
                    start <= nested["started_at"] <= nested["ended_at"] <= end,
                    "phase clock outside original workflow",
                )
        if training and training.get("budget") and source.get("readout_budget"):
            _require(
                training["budget"]["ended_at"] <= source["readout_budget"]["started_at"],
                "readout precedes training completion",
            )
        if complete:
            _number(budget["elapsed_seconds"], "complete workflow elapsed", 0, 300)
            _integer(
                budget.get("sampled_peak_rss_bytes"), 0, 512 * 1024**2, "complete workflow RSS"
            )
        costs += _flatten(budget, "/workflow_budget", "Workflow recorded scope")
    costs += _flatten(
        {k: source[k] for k in ("attempted_phases", "collection_capture_truncated") if k in source},
        "",
        "Recorded phase admission/capture",
    )
    return costs


def _native_costs(record, records, *, pointer="/games", label="Known-game work"):
    costs = []
    done = record["status"] == "complete"
    _require(not done or record.get("work") is not None, "complete native report lacks costs")
    if record.get("work") is not None:
        work = _object(record["work"], "native work")
        for key in (
            "raw_score_forwards_attempted",
            "raw_score_forwards_completed",
            "raw_score_rows_attempted",
            "raw_score_rows_completed",
            "live_sessions_started",
        ):
            _fixed(work, key, 0)
        for key, field in (
            ("live_prediction_calls_attempted", "prediction_calls_attempted"),
            ("live_prediction_calls_returned", "prediction_calls_returned"),
        ):
            _fixed(work, key, sum(e.get(field, 0) for e in records))
        for key, field in (
            ("environment_steps_attempted", "environment_steps_attempted"),
            ("environment_steps_returned", "environment_steps"),
        ):
            known = sum(
                e["environment"][field] for e in records if e.get("environment") is not None
            )
            value = _integer(work.get(key), known, 64, key)
            _require(not done or value == known, "native work disagrees with completed games")
        environments = [e["environment"] for e in records if e.get("environment") is not None]
        resets = _integer(
            work.get("environment_resets_attempted"),
            len(environments),
            len(records),
            "reset attempts",
        )
        returned = _integer(
            work.get("environment_resets_completed"), len(environments), resets, "reset returns"
        )
        _require(not done or resets == returned == 4, "complete native reset count mismatch")
        costs.extend(_flatten(work, f"{pointer}/work", label))
    return costs


def _games(record, complete, recipe=NORMAL):
    records, proofs, costs = [], [], []
    if record is not None:
        _object(record, "native games")
        _require(_status(record.get("status")), "invalid native status")
        for key, value in dict(
            method=METHOD,
            role=recipe.game_role,
            updates=recipe.points[-1],
            source_fit_calls=recipe.points[-1],
            source_fit_rows=recipe.total_rows[-1],
            new_fit_calls=0,
            new_optimizer_updates=0,
        ).items():
            _fixed(record, key, value)
        done = record["status"] == "complete"
        _require(not complete or done, "complete workflow lacks complete native games")
        _require(not done or record.get("state_unchanged") is True, "complete native state changed")
        _require(record.get("scores") == [], "native readout must not add labelled scores")
        records = _records(record.get("episodes", []), 4, "native episodes")
        _require(not done or len(records) == 4, "complete native report lacks four games")
        _require(
            not done or all(e.get("status") == "complete" for e in records),
            "complete native report has unfinished episode",
        )
        proofs.extend(_proofs(record, "/games"))
    else:
        _require(not complete, "complete workflow lacks native report")
    episodes = []
    failed = False
    for i, (context, cue) in enumerate(recipe.slots):
        if i >= len(records):
            game = _missing(0, recipe.phase, context, cue, "train", recipe.points[-1])
        else:
            row = records[i]
            pointer = f"/games/episodes/{i}"
            _require(
                not failed or row.get("status") == "not_run",
                "native work continues after failed episode",
            )
            if row.get("status") == "not_run":
                for key, value in dict(
                    context=context,
                    cue=cue,
                    source_role="train",
                    prediction_calls_attempted=0,
                    prediction_calls_returned=0,
                ).items():
                    _fixed(row, key, value)
                _require(
                    not row.get("predictions")
                    and row.get("environment") is None
                    and row.get("success") is None
                    and row.get("return") is None,
                    "unrun game contains observed outcome",
                )
                if recipe is RECURRING:
                    _require(row.get("state_unchanged") is None, "unrun game claims preservation")
                game = _missing(0, recipe.phase, context, cue, "train", recipe.points[-1])
                game.update(pointer=pointer, record=deepcopy(row), derived=False)
            else:
                game = _episode(
                    row,
                    pointer,
                    METHOD,
                    recipe.points[-1],
                    cue,
                    context=context,
                    role="train",
                    expected_layout=recipe.layouts[context],
                    allow_observed_failure_tail=True,
                )
                game.update(source_updates=0, phase=recipe.phase, context=context, derived=False)
            failed = failed or row.get("status") != "complete"
        episodes.append(game)
    if record is not None:
        costs.extend(_native_costs(record, records))
    return episodes, proofs, costs


def project_affine_workflow_report(report: dict[str, Any], *, source_label: str) -> dict[str, Any]:
    """Validate recorded consistency, retaining unknowns without granting authority."""
    _envelope(report, source_label)
    if report.get("definition") == A4_DEFINITION:
        return _project_a4(report, source_label)
    _require(
        report.get("definition") in (DEFINITION, RECURRING_DEFINITION), "invalid affine definition"
    )

    recipe = RECURRING if report["definition"] == RECURRING_DEFINITION else NORMAL
    _require(_status(report.get("status")), "invalid workflow status")
    source = deepcopy(report)
    complete = source["status"] == "complete"
    curves, receipts, wakes, costs, proofs = _training(source.get("training"), complete, recipe)
    if recipe is RECURRING:
        costs.extend(_recurring_scope(source, complete))
        proofs.extend(_proofs(source, ""))
    if source.get("games") is not None:
        _require(
            type(source.get("training")) is dict and source["training"].get("status") == "complete",
            "native games precede completed TRAIN",
        )
    episodes, game_proofs, game_costs = _games(source.get("games"), complete, recipe)
    if source.get("readout_budget") is not None:
        _budget(source["readout_budget"], 0, readout=True, recipe=recipe, done=complete)
        costs.extend(_flatten(source["readout_budget"], "/readout_budget", "Readout budget"))
    _require(
        not complete or source.get("readout_budget") is not None,
        "complete readout lacks original budget",
    )
    return dict(
        schema_id="minigrid_affine_inspector_v1",
        **(dict(recipe=recipe.name) if recipe is RECURRING else {}),
        source_label=source_label,
        origin="recorded_artifact",
        source=source,
        conditions=[
            dict(
                key="Conventional affine learner",
                pointer="",
                status=source["status"],
                source_role="train",
                episodes=episodes,
                curves=curves,
                receipts=receipts,
                wakes=wakes,
                proofs=proofs + game_proofs,
                costs=costs + game_costs,
            )
        ],
    )


def _a4_scope(source, complete):
    protocol = _object(source.get("protocol"), "A4 protocol")
    for key, value in dict(
        definition=A4_DEFINITION,
        source_origin="current_run_A4_TRAIN",
        historical_source_reads=0,
        teacher_step_limit=32,
    ).items():
        _fixed(protocol, key, value)
    _require(
        "source_sha256" not in protocol and "source_bytes" not in protocol,
        "fresh collection cannot claim historical file binding",
    )
    planned = _object(protocol.get("training"), "A4 training plan")
    for key, value in dict(
        definition=A4.training_definition,
        label_delay=1,
        checkpoint_updates=list(A4.points),
        fit_calls=6,
        fit_rows=90,
        score_calls=16,
        score_rows=144,
        arrival_row_exposures=54,
        catalogue_replay_row_exposures=36,
        distinct_source_rows=36,
        classes=7,
        wall_seconds=90,
        peak_rss_bytes=512 * 1024**2,
    ).items():
        _fixed(planned, key, value)
    costs = _flatten(planned, "/protocol/training", "TRAIN plan (not paid work)")
    required = source.get("training") is not None or complete
    costs += _recurring_collection(source, required, A4)
    provenance = source.get("source")
    _require(not required or provenance is not None, "A4 training lacks collection provenance")
    if provenance is not None:
        _object(provenance, "fresh provenance")
        for key, value in dict(
            origin="current_run_A4_TRAIN", teacher_steps=18, environment_episodes=2, history_rows=18
        ).items():
            _fixed(provenance, key, value)
        _require(
            "source_sha256" not in provenance and "source_bytes" not in provenance,
            "fresh provenance claims historical binding",
        )
        _number(provenance.get("collection_seconds"), "collection interval", 0)
        costs += _flatten(provenance, "/source", "Current-run collection scope")
    budget = _object(source.get("readout_budget"), "A4 cumulative budget")
    limits = _object(budget.get("limits"), "A4 cumulative limits")
    _require(budget.get("initial_limits") == limits, "original A4 limits changed")
    for key, value in dict(
        max_training_updates=0,
        max_replay_examples=0,
        max_wall_seconds=150,
        max_process_rss_bytes=512 * 1024**2,
    ).items():
        _fixed(limits, key, value)
    for key in ("updates", "replay"):
        _fixed(budget, key, 0)
    start = _number(budget.get("started_at"), "A4 start", 0)
    end = _number(budget.get("ended_at"), "A4 end", start)
    _number(budget.get("observed_started_at"), "observed A4 start", 0)
    _require(budget.get("observed_started_at") == start, "original A4 clock renewed")
    _require(not complete or end - start <= 150, "complete A4 exceeds original clock")
    if budget.get("sampled_peak_rss_bytes") is not None:
        _integer(
            budget["sampled_peak_rss_bytes"], 0, 512 * 1024**2 if complete else 2**53 - 1, "A4 RSS"
        )
    training = source.get("training")
    if training is not None and training.get("budget") is not None:
        nested = training["budget"]
        _number(nested.get("started_at"), "TRAIN start in cumulative owner", start, end)
        _number(nested.get("ended_at"), "TRAIN end in cumulative owner", nested["started_at"], end)
    return costs + _flatten(budget, "/readout_budget", "Cumulative collection/TRAIN/game budget")


def _a4_wakes(wakes):
    previous: dict[str, tuple[str, int]] = {}
    for i, wake in enumerate(wakes):
        cue = ("blue", "red")[i % 2]
        identity = "A4-train" if i in (2, 3) else "A"
        current = f"minigrid-source-{identity}-{cue}:wake:{i}"
        old = previous.get(cue)
        spans = [dict(episode_id=old[0], context="A", start=0, stop=9)] if old else []
        start = 9 if old else 0
        spans.append(dict(episode_id=current, context="A", start=start, stop=start + 9))
        for key, value in dict(
            current_episode_id=current,
            retained_episode_id=old[0] if old else None,
            retained_arrived_at=old[1] if old else None,
            source_spans=spans,
            bank_rows_after=min(i + 1, 2) * 9,
        ).items():
            _fixed(wake, key, value)
        previous[cue] = (current, i + 1)


def _a4_phase(record, updates, path, complete):
    _object(record, "A4 native phase")
    for key, value in dict(
        method=METHOD,
        role=A4.game_role,
        updates=updates,
        source_fit_calls=updates,
        source_fit_rows=A4.total_rows[A4.points.index(updates)],
        new_fit_calls=0,
        new_optimizer_updates=0,
    ).items():
        _fixed(record, key, value)
    _require(_status(record.get("status")), "invalid A4 phase status")
    done = record["status"] == "complete"
    _require(not complete or done, "paid phase incomplete")
    _require(not done or record.get("state_unchanged") is True, "complete A4 state changed")
    _require(record.get("scores") == [], "A4 game must not add labelled scores")
    rows = _records(record.get("episodes", []), 4, "A4 games")
    _require(
        not done or len(rows) == 4 and all(r.get("status") == "complete" for r in rows),
        "complete A4 phase needs four completed games",
    )
    games, failed = [], False
    label = {2: "After A3", 4: "After A4", 6: "Returned A3"}[updates]
    for index, (context, cue) in enumerate(A4.slots):
        game = _missing(0, label, context, cue, "train", updates)
        if index < len(rows):
            row, pointer = rows[index], f"{path}/episodes/{index}"
            _require(
                not failed or row.get("status") == "not_run", "A4 game continues after failed work"
            )
            if row.get("status") == "not_run":
                for key, value in dict(
                    context="A",
                    cue=cue,
                    source_role="train",
                    prediction_calls_attempted=0,
                    prediction_calls_returned=0,
                ).items():
                    _fixed(row, key, value)
                _require(
                    not row.get("predictions")
                    and row.get("environment") is None
                    and row.get("success") is None
                    and row.get("return") is None
                    and row.get("state_unchanged") is None,
                    "unrun A4 game claims observed work",
                )
                game.update(pointer=pointer, record=deepcopy(row), derived=False)
            else:
                layout = A4.layouts[context]
                game = _episode(
                    row,
                    pointer,
                    METHOD,
                    updates,
                    cue,
                    context="A",
                    role="train",
                    expected_layout=layout,
                    allow_observed_failure_tail=True,
                    expected_episode_id=f"live-{METHOD}-{updates}-A-row{layout['branch_row']}-{cue}",
                )
                game.update(source_updates=0, phase=label, context=context, derived=False)
            failed = failed or row.get("status") != "complete"
        games.append(game)
    return (
        games,
        _proofs(record, path),
        _native_costs(record, rows, pointer=path, label=label + " native work"),
    )


def _a4_summary(source):
    summary = _object(source.get("summary"), "A4 summary")
    if source["status"] != "complete":
        _fixed(summary, "complete", False)
        _require(
            all(
                summary.get(k) is None
                for k in (
                    "acquisition_gain",
                    "old_retention_delta",
                    "return_gain",
                    "learning_recovery",
                    "success_by_stage",
                )
            ),
            "failed A4 aggregate claims outcomes",
        )
        return
    _fixed(summary, "complete", True)
    outcomes = [[e["success"] for e in phase["games"]["episodes"]] for phase in source["games"]]
    rates = [dict(old_A3=sum(o[:2]) / 2, new_A4=sum(o[2:]) / 2) for o in outcomes]
    _require(summary.get("success_by_stage") == rates, "A4 summary disagrees with games")
    for recorded, expected in zip(summary["success_by_stage"], rates, strict=True):
        for key in ("old_A3", "new_A4"):
            _close(recorded[key], expected[key], key)
    for key, value in dict(
        acquisition_gain=rates[1]["new_A4"] - rates[0]["new_A4"],
        old_retention_delta=rates[1]["old_A3"] - rates[0]["old_A3"],
        return_gain=rates[2]["old_A3"] - rates[1]["old_A3"],
    ).items():
        _close(summary.get(key), value, key)
    recovery = [outcomes[0][i] and not outcomes[1][i] and outcomes[2][i] for i in range(2)]
    _require(
        type(summary.get("learning_recovery")) is list
        and all(type(v) is bool for v in summary["learning_recovery"])
        and summary["learning_recovery"] == recovery,
        "A4 recovery differs from loss/regain",
    )


def _project_a4(report, source_label):
    _require(_status(report.get("status")), "invalid A4 workflow status")
    source = deepcopy(report)
    complete = source["status"] == "complete"
    costs = _a4_scope(source, complete)
    curves, receipts, wakes, train_costs, proofs = _training(source.get("training"), complete, A4)
    phases = _records(source.get("games", []), 3, "paid A4 phases")
    _require(not complete or len(phases) == 3, "complete A4 lacks three phases")
    failed = source.get("failed_phase")
    training = source.get("training")
    _require(
        not (phases or failed) or type(training) is dict and training.get("status") == "complete",
        "A4 games precede training",
    )
    _require(failed is None or not complete and len(phases) < 3, "invalid failed A4 phase position")
    games: list[dict[str, Any]] = []
    for index, updates in enumerate((2, 4, 6)):
        phase = phases[index] if index < len(phases) else failed if index == len(phases) else None
        if phase is None:
            label = {2: "After A3", 4: "After A4", 6: "Returned A3"}[updates]
            games.extend(_missing(0, label, c, q, "train", updates) for c, q in A4.slots)
            continue
        _object(phase, "A4 stage")
        _fixed(phase, "updates", updates)
        path = f"/games/{index}/games" if index < len(phases) else "/failed_phase/games"
        episodes, flags, work = _a4_phase(phase.get("games"), updates, path, index < len(phases))
        games += episodes
        proofs += flags
        costs += work
    _a4_summary(source)
    return dict(
        schema_id="minigrid_affine_inspector_v1",
        recipe=A4.name,
        source_label=source_label,
        origin="recorded_artifact",
        source=source,
        conditions=[
            dict(
                key="Conventional affine learner",
                pointer="",
                status=source["status"],
                source_role="train",
                episodes=games,
                curves=curves,
                receipts=receipts,
                wakes=wakes,
                proofs=proofs,
                costs=costs + train_costs,
            )
        ],
    )
