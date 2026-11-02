"""Ordinary probe callback accounting; no training loop or input retention.

The existing runner owns private forks, roles, clocks and slots. This adapter
seals a detached public view while retaining actual scalar receipts separately,
so corrupted public evidence cannot erase costs or become a new baseline.
"""

from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any

from src.adapters.torch_ce_probe_update import DEFINITION, apply_ce_probe_update
from src.app.coupled_attribution import _same_value
from src.app.probe_return_phase import paid_report_guard
from src.core.probe_update import ProbeUpdateFailed, ProbeUpdateReceipt


class CEProbeWork:
    def __init__(self, row, require):
        if "ce_rehearsal_work" in row:
            raise ValueError("paid CE probe work cannot be renewed")
        self._row, self._require = row, require
        self._paid: dict[str, Any] = dict(
            definition="ordinary_ce_probe_updates_v1",
            attempted_updates=0,
            attempted_rows=0,
            returned_updates=0,
            applied_updates=0,
            native_sgd_updates=0,
            events=[],
        )
        self._view: dict[str, Any] | None = None
        self._guard: Any = None
        self._publish()

    def check(self):
        if self._view is not None:
            if self._row.get("ce_rehearsal_work") is not self._view:
                raise ValueError("paid CE probe evidence replaced")
            self._guard()

    def _publish(self):
        self.check()
        self._view = deepcopy(self._paid)
        self._row["ce_rehearsal_work"] = self._view
        self._guard, _ = paid_report_guard(self._view)

    @property
    def preserved(self):
        return deepcopy(self._paid)

    def callback(self, learner, current, retained, *, phase, check_boundary):
        self._require()
        if len(self._paid["events"]) >= 32:
            raise ValueError("ordinary CE probe application allowance spent")
        event = dict(
            phase=phase,
            current_rows=len(current.features),
            status="running",
            work_known=False,
            applied=False,
        )
        self._paid["events"].append(event)
        self._paid["attempted_updates"] += 1
        self._paid["attempted_rows"] += len(current.features)
        self._publish()
        receipt = None
        failure = None

        def boundary():
            self._require()
            check_boundary()
            self._require()

        try:
            receipt = apply_ce_probe_update(
                learner, current, retained, phase=phase, check_boundary=boundary
            )
            self._paid["returned_updates"] += 1
        except Exception as error:
            failure = error
            if type(error) is ProbeUpdateFailed:
                receipt = error.receipt
        # Keep costs privately BEFORE attempting to publish over an unchanged view.
        # A refused public view remains corrupted, and can never be re-baselined.
        try:
            if receipt is not None:
                if type(receipt) is not ProbeUpdateReceipt or receipt.definition != DEFINITION:
                    raise ValueError("ordinary CE callback requires its exact scalar receipt")
                receipt = replace(receipt)
                event.update(asdict(receipt), work_known=True)
                self._paid["applied_updates"] += int(receipt.applied)
            event["status"] = "complete" if failure is None else "failed"
            if failure is None and (receipt is None or not receipt.applied):
                raise ValueError("ordinary CE callback did not apply an update")
            self._publish()
            self._require()
        except Exception as error:
            event["status"] = "failed"
            if failure is None:
                failure = error
        if failure is not None:
            if type(receipt) is ProbeUpdateReceipt:
                raise ProbeUpdateFailed(str(failure), receipt) from failure
            raise failure
        return receipt

    def complete(self, probe):
        self._require()
        if tuple(
            self._paid[k]
            for k in ("attempted_updates", "attempted_rows", "returned_updates", "applied_updates")
        ) != (32, 144, 32, 32):
            raise ValueError("ordinary CE probe applications incomplete")
        phases = (probe, probe.get("return_phase", {}))
        for index, (row, phase) in enumerate(zip(phases, ("fresh", "return"), strict=True)):
            auxiliary = row.get("auxiliary_updates", {})
            expected = self._paid["events"][16 * index : 16 * (index + 1)]
            if (
                row.get("status") != "complete"
                or row.get("completed_updates") != 16
                or type(auxiliary.get("native_sgd_updates")) is not int
                or auxiliary["native_sgd_updates"] != 0
                or not _same_value(auxiliary.get("events"), expected)
                or any(e["phase"] != phase or e["status"] != "complete" for e in expected)
            ):
                raise ValueError("ordinary CE probe receipt/phase accounting differs")
