"""Runtime coordination between the GA worker loop and its native supervisor."""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any, Callable

from ga_harness.events import EventRecorder
from ga_harness.supervisor import Decision, ProgressiveSupervisor


class HarnessControl:
    """Record the worker and schedule step, stall, and completion checks."""

    def __init__(
        self,
        recorder: EventRecorder,
        supervisor: ProgressiveSupervisor | None,
        interrupt: Callable[[], None],
        step_interval: int = 10,
        stall_timeout_sec: float = 180.0,
        max_interventions: int = 3,
    ) -> None:
        self.recorder = recorder
        self.supervisor = supervisor
        self.interrupt = interrupt
        self.step_interval = step_interval
        self.stall_timeout_sec = stall_timeout_sec
        self.max_interventions = max_interventions
        self._condition = threading.Condition()
        self._active_action: str | None = None
        self._active_kind: str | None = None
        self._last_feedback = 0.0
        self._stalled_action: str | None = None
        self._closed = False
        self._follow_up = False
        self._interventions = 0
        self._watchdog = threading.Thread(target=self._watch_stalls, daemon=True)
        self._watchdog.start()

    @classmethod
    def from_env(
        cls,
        recorder: EventRecorder,
        supervisor: ProgressiveSupervisor | None,
        interrupt: Callable[[], None],
    ) -> "HarnessControl":
        return cls(
            recorder,
            supervisor,
            interrupt,
            step_interval=int(os.environ.get("GA_SUPERVISOR_STEP_INTERVAL", "10")),
            stall_timeout_sec=float(os.environ.get("GA_STALL_TIMEOUT_SEC", "180")),
            max_interventions=int(os.environ.get("GA_MAX_INTERVENTIONS", "3")),
        )

    def action_started(self, kind: str, action_id: str, turn: int) -> None:
        with self._condition:
            self._active_kind = kind
            self._active_action = action_id
            self._last_feedback = time.monotonic()
            self._condition.notify_all()
        if kind == "model":
            self.recorder.begin_assistant(turn)
        self.recorder.emit("action_started", action_kind=kind, action_id=action_id, turn=turn)

    def progress(self, kind: str, text: str = "") -> None:
        with self._condition:
            if self._active_action is None:
                return
            self._last_feedback = time.monotonic()
            self._condition.notify_all()

    def action_finished(self, action_id: str) -> None:
        with self._condition:
            if self._active_action == action_id:
                self._active_action = None
                self._active_kind = None
                self._last_feedback = 0.0
                self._condition.notify_all()
        self.recorder.emit("action_finished", action_id=action_id)

    def record_turn(
        self,
        turn: int,
        response: Any,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        metrics: dict[str, int] | None = None,
        interrupted: bool = False,
    ) -> None:
        calls = [
            {
                "id": call.get("id") or f"turn-{turn}-call-{index}",
                "name": call.get("tool_name", ""),
                "arguments": call.get("args") or {},
            }
            for index, call in enumerate(tool_calls, 1)
            if call.get("tool_name") != "no_tool"
        ]
        self.recorder.finish_assistant(
            turn=turn,
            reasoning=getattr(response, "thinking", "") or "",
            content=getattr(response, "content", "") or "",
            tool_calls=calls,
            tool_results=tool_results,
            metrics=metrics,
            interrupted=interrupted,
        )

    def has_stall(self) -> bool:
        with self._condition:
            return self._stalled_action is not None

    def take_stall(self) -> str | None:
        with self._condition:
            target = self._stalled_action
            self._stalled_action = None
            return target

    def review_stall(self, target: str, state: str) -> str:
        if self.supervisor is None:
            return ""
        decision = self.supervisor.evaluate("stall", target, state)
        if decision.action != "intervene":
            raise RuntimeError("A no-feedback review must intervene")
        return self._apply(decision, reason_type="stalled")

    def needs_follow_up(self) -> bool:
        return self.supervisor is not None and self._follow_up

    def check_step(self, step: int, state: str) -> str:
        if self.supervisor is None:
            return ""
        trigger = "follow_up" if self._follow_up else "step_interval"
        if not self._follow_up and step % self.step_interval:
            return ""
        decision = self.supervisor.evaluate(trigger, f"step-{step}", state)
        if decision.action == "continue":
            self._follow_up = False
            return ""
        return self._apply(decision, reason_type=decision.reason or "drift")

    def check_completion(self, state: str) -> str:
        if self.supervisor is None:
            return ""
        decision = self.supervisor.evaluate("completion", "completion", state)
        if decision.action == "continue":
            self._follow_up = False
            return ""
        return self._apply(decision, reason_type=decision.reason or "incomplete")

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._watchdog.join(timeout=2)

    def summary(self) -> dict[str, Any]:
        checks = len(self.supervisor.history) if self.supervisor else 0
        return {
            "enabled": self.supervisor is not None,
            "n_checks": checks,
            "n_interventions": self._interventions,
            "step_interval": self.step_interval,
            "stall_timeout_sec": self.stall_timeout_sec,
            "token_metrics": (
                self.supervisor.token_metrics() if self.supervisor is not None else {}
            ),
        }

    def _apply(self, decision: Decision, reason_type: str) -> str:
        if self._interventions >= self.max_interventions:
            self._follow_up = False
            self.recorder.emit(
                "supervisor_limit_reached",
                maximum=self.max_interventions,
                rejected_correction=decision.correction,
            )
            return ""
        self._interventions += 1
        self._follow_up = True
        intervention_id = str(uuid.uuid4())
        self.recorder.emit(
            "supervisor_correction",
            content=decision.correction,
            intervention_id=intervention_id,
            reason_type=reason_type,
            hint_level=decision.level,
            origin="supervisor",
            target=decision.target,
        )
        return decision.correction

    def _watch_stalls(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                if self.supervisor is None or self._active_action is None:
                    self._condition.wait(timeout=1.0)
                    continue
                action = self._active_action
                remaining = self.stall_timeout_sec - (
                    time.monotonic() - self._last_feedback
                )
                if remaining > 0:
                    self._condition.wait(timeout=remaining)
                    continue
                if action != self._active_action:
                    continue
                self._stalled_action = action
                self._active_action = None
                self._active_kind = None
                self._last_feedback = 0.0
            self.recorder.emit(
                "interrupted",
                action_id=action,
                reason="no_feedback",
                timeout_sec=self.stall_timeout_sec,
            )
            self.interrupt()
