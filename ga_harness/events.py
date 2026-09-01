"""Structured runtime events and live ATIF trajectory output for GenericAgent."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class EventRecorder:
    """Keep one canonical event stream and project it into live ATIF."""

    def __init__(self, logs_dir: str | Path, model: str, version: str = "0.1.0"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.logs_dir / "events.jsonl"
        self.trajectory_path = self.logs_dir / "trajectory.json"
        self.session_id = f"ga_{uuid.uuid4().hex}"
        self.model = model
        self.version = version
        self._events: list[dict[str, Any]] = []
        self._draft: dict[str, Any] | None = None
        self._sequence = 0
        self._lock = threading.RLock()
        self._totals = {"prompt": 0, "completion": 0, "cached": 0}

    def emit(self, kind: str, **payload: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "sequence": self._sequence,
                "kind": kind,
                "timestamp": utc_now(),
                "session_id": self.session_id,
                **payload,
            }
            self._events.append(event)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            if kind in {
                "user_prompt",
                "supervisor_correction",
                "manual_correction",
                "assistant",
            }:
                self._write_trajectory_locked()
            return event

    def begin_assistant(self, turn: int) -> None:
        with self._lock:
            self._draft = {
                "turn": turn,
                "timestamp": utc_now(),
                "started_monotonic": time.monotonic(),
            }

    def finish_assistant(
        self,
        *,
        turn: int,
        reasoning: str,
        content: str,
        tool_calls: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        metrics: dict[str, int] | None = None,
        interrupted: bool = False,
    ) -> None:
        with self._lock:
            draft = self._draft
            self._draft = None
        started_at = (
            draft["timestamp"]
            if draft is not None and draft.get("turn") == turn
            else utc_now()
        )
        duration_ms = None
        if draft is not None and draft.get("turn") == turn:
            duration_ms = max(
                0,
                round((time.monotonic() - draft["started_monotonic"]) * 1000),
            )
        metrics = metrics or {}
        self._totals["prompt"] += int(metrics.get("prompt_tokens") or 0)
        self._totals["completion"] += int(metrics.get("completion_tokens") or 0)
        self._totals["cached"] += int(metrics.get("cached_tokens") or 0)
        self.emit(
            "assistant",
            turn=turn,
            started_at=started_at,
            duration_ms=duration_ms,
            reasoning=reasoning,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            metrics=metrics,
            interrupted=interrupted,
        )

    def finalize(self, status: str, **extra: Any) -> None:
        self.emit("result", status=status, **extra)
        with self._lock:
            self._write_trajectory_locked(final=True)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def observable_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return behavior evidence without private model reasoning."""
        visible: list[dict[str, Any]] = []
        with self._lock:
            for event in self._events[-limit:]:
                item = dict(event)
                item.pop("reasoning", None)
                visible.append(item)
        return visible

    def worker_metrics(self) -> dict[str, int]:
        return {
            "total_prompt_tokens": self._totals["prompt"],
            "total_completion_tokens": self._totals["completion"],
            "total_cached_tokens": self._totals["cached"],
        }

    def _write_trajectory_locked(self, final: bool = False) -> None:
        steps: list[dict[str, Any]] = []
        for event in self._events:
            step = self._event_to_step(event, len(steps) + 1)
            if step is not None:
                steps.append(step)
        if not steps:
            return
        trajectory: dict[str, Any] = {
            "schema_version": "ATIF-v1.7",
            "session_id": self.session_id,
            "trajectory_id": self.session_id,
            "agent": {
                "name": "generic-agent",
                "version": self.version,
                "model_name": self.model,
            },
            "steps": steps,
            "extra": {
                "live": not final,
                "event_sequence": self._sequence,
                "worker_metrics": self.worker_metrics(),
            },
        }
        if final:
            trajectory["final_metrics"] = {
                "total_prompt_tokens": self._totals["prompt"],
                "total_completion_tokens": self._totals["completion"],
                "total_cached_tokens": self._totals["cached"],
                "total_steps": len(steps),
            }
        atomic_write_json(self.trajectory_path, trajectory)

    def _event_to_step(
        self, event: dict[str, Any], step_id: int
    ) -> dict[str, Any] | None:
        kind = event["kind"]
        if kind in {"user_prompt", "supervisor_correction", "manual_correction"}:
            extra = {"kind": kind}
            for key in ("intervention_id", "reason_type", "hint_level", "origin"):
                if event.get(key) is not None:
                    extra[key] = event[key]
            return {
                "step_id": step_id,
                "timestamp": event["timestamp"],
                "source": "user",
                "message": event.get("content", ""),
                "extra": extra,
            }
        if kind != "assistant":
            return None
        calls = [
            {
                "tool_call_id": call.get("id") or f"call_{index}",
                "function_name": call.get("name", ""),
                "arguments": call.get("arguments") or {},
            }
            for index, call in enumerate(event.get("tool_calls") or [], 1)
        ]
        results = [
            {
                "source_call_id": result.get("tool_use_id") or None,
                "content": str(result.get("content", "")),
            }
            for result in event.get("tool_results") or []
        ]
        step: dict[str, Any] = {
            "step_id": step_id,
            "timestamp": event.get("started_at") or event["timestamp"],
            "source": "agent",
            "model_name": self.model,
            "message": event.get("content", ""),
            "llm_call_count": 1,
            "extra": {
                "kind": "assistant",
                "turn": event.get("turn"),
                "interrupted": bool(event.get("interrupted")),
                "started_at": event.get("started_at") or event["timestamp"],
                "finished_at": event["timestamp"],
                "duration_ms": event.get("duration_ms"),
            },
        }
        if event.get("reasoning"):
            step["reasoning_content"] = event["reasoning"]
        if calls:
            step["tool_calls"] = calls
        if results:
            step["observation"] = {"results": results}
        metrics = event.get("metrics") or {}
        if metrics:
            step["metrics"] = metrics
        return step
