"""GA-owned progressive, read-only supervisor."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ga_harness.events import EventRecorder, utc_now
from ga_harness.model import build_client
from ga_harness.supervisor_tools import READ_ONLY_TOOLS, ReadOnlyWorkspace


@dataclass(frozen=True)
class Decision:
    action: str
    reason: str = ""
    correction: str = ""
    level: str = "constraint"
    target: str = ""



class SupervisionAudit:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = 0

    def emit(
        self, check_id: str, kind: str, payload: dict[str, Any], attempt: int = 1
    ) -> None:
        self.sequence += 1
        record = {
            "sequence": self.sequence,
            "check_id": check_id,
            "attempt": attempt,
            "kind": kind,
            "timestamp": utc_now(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class ProgressiveSupervisor:
    """Use a separate model session to judge Worker progress and corrections."""

    def __init__(
        self,
        instruction: str,
        workspace: str | Path,
        memory_dir: str | Path,
        recorder: EventRecorder,
        max_tool_rounds: int = 3,
        decision_attempts: int = 2,
        client=None,
    ) -> None:
        self.instruction = instruction
        self.workspace = Path(workspace).resolve()
        self.memory_dir = Path(memory_dir).resolve()
        self.recorder = recorder
        self.tools = ReadOnlyWorkspace(self.workspace, recorder)
        self.client = client or build_client("supervisor")
        self.client.log_path = False
        self.max_tool_rounds = max_tool_rounds
        self.decision_attempts = decision_attempts
        self.history: list[dict[str, Any]] = []
        self.last_level: str | None = None
        self.audit = SupervisionAudit(recorder.logs_dir / "supervision.jsonl")
        self._set_system_prompt()

    def evaluate(self, trigger: str, target: str, state: str = "") -> Decision:
        check_id = str(uuid.uuid4())
        snapshot = self._snapshot(trigger, target, state)
        self.audit.emit(check_id, "check_started", {"trigger": trigger, "target": target})
        prompt = self._decision_prompt(snapshot)
        response = None
        error = None
        for attempt in range(1, self.decision_attempts + 1):
            response = self._review_with_tools(prompt, check_id, attempt)
            try:
                decision = self._parse_decision(
                    response.content or response.thinking, trigger, target
                )
                break
            except (ValueError, json.JSONDecodeError) as caught:
                error = caught
                self.audit.emit(
                    check_id,
                    "invalid_decision",
                    {
                        "error": str(caught),
                        "raw_content": response.content,
                        "raw_reasoning": response.thinking,
                    },
                    attempt,
                )
                prompt = (
                    "Your prior response violated the JSON decision contract. "
                    "Return only one valid JSON object now; do not use tools."
                )
        else:
            raise ValueError(
                f"Supervisor failed its decision contract after {self.decision_attempts} attempts: {error}"
            )
        self._remember_decision(trigger, target, decision)
        self.audit.emit(
            check_id,
            "decision",
            {
                "raw_content": response.content,
                "raw_reasoning": response.thinking,
                "decision": decision.__dict__,
            },
            attempt,
        )
        self.audit.emit(
            check_id, "check_finished", {"decision": decision.action}, attempt
        )
        return decision

    def _review_with_tools(self, prompt: str, check_id: str, attempt: int):
        tool_results: list[dict[str, Any]] = []
        response = None
        for round_number in range(self.max_tool_rounds + 1):
            response = self._chat(prompt, tool_results)
            self.audit.emit(
                check_id,
                "model_response",
                {
                    "round": round_number,
                    "content": response.content,
                    "reasoning": response.thinking,
                },
                attempt,
            )
            if not response.tool_calls:
                return response
            tool_results = self._execute_tools(response, check_id, attempt)
            prompt = "Use the read-only results and return the final decision JSON."
        raise RuntimeError("Supervisor exceeded the read-only tool-round limit")

    def _execute_tools(self, response, check_id: str, attempt: int):
        results = []
        for call in response.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            result = self.tools.call(call.function.name, arguments)
            self.audit.emit(
                check_id,
                "tool_result",
                {"name": call.function.name, "arguments": arguments, "result": result},
                attempt,
            )
            results.append({"tool_use_id": call.id, "content": result})
        return results

    def _remember_decision(self, trigger: str, target: str, decision: Decision) -> None:
        self.history.append(
            {
                "trigger": trigger,
                "target": target,
                "action": decision.action,
                "reason": decision.reason,
                "level": decision.level,
            }
        )
        if decision.action == "intervene":
            self.last_level = decision.level
        elif trigger == "follow_up":
            self.last_level = None

    def _chat(self, prompt: str, tool_results: list[dict[str, Any]]):
        messages = [{"role": "user", "content": prompt}]
        if tool_results:
            messages[0]["tool_results"] = tool_results
        generator = self.client.chat(messages, tools=READ_ONLY_TOOLS)
        try:
            while True:
                next(generator)
        except StopIteration as stop:
            return stop.value

    def _set_system_prompt(self) -> None:
        memory = self._memory_context()
        self.client.set_system(
            "You are GenericAgent's read-only supervisor. Never perform the task or "
            "modify files. Judge observable progress, tool results, and artifacts. "
            "Do not inspect hidden tests, verifier files, solutions, or grading data. "
            "A reasonable diagnostic step is progress; repeated work without a new "
            "artifact is drift. Any correction must be short and actionable.\n\n"
            + memory
        )

    def _memory_context(self) -> str:
        parts = []
        for name in ("global_mem_insight.txt", "supervisor_sop.md"):
            path = self.memory_dir / name
            if path.is_file():
                content = path.read_text(encoding="utf-8", errors="replace")
                parts.append(f"[{name}]\n{content}")
        return "\n\n".join(parts)

    def _snapshot(self, trigger: str, target: str, state: str) -> dict[str, Any]:
        return {
            "original_instruction": self.instruction,
            "trigger": trigger,
            "target": target,
            "current_state": state[-20_000:],
            "trajectory": self.recorder.observable_events(30),
            "supervision_history": self.history[-10:],
            "workspace_top_level": self.tools.list_directory({"path": "."}),
        }

    def _decision_prompt(self, snapshot: dict[str, Any]) -> str:
        trigger = snapshot["trigger"]
        level_rule = (
            "This is a follow-up. If the previous correction was not applied, "
            "intervene at the next progressive level."
            if trigger == "follow_up"
            else "Any intervention must use level=constraint."
        )
        if trigger == "stall":
            level_rule = "The action was forcibly interrupted after no feedback; intervene."
        if trigger == "completion":
            level_rule += " Approve only if required artifacts and evidence are complete."
        return (
            f"{level_rule}\n"
            "Return only JSON with keys action, reason, correction, level, target. "
            "action is continue or intervene. continue requires empty correction. "
            "constraint states the violated requirement and missing evidence only; "
            "procedure may give a method; answer may give a concrete recovery direction "
            "but never a hidden answer.\nSnapshot:\n"
            + json.dumps(snapshot, ensure_ascii=False)
        )

    def _parse_decision(self, text: str, trigger: str, target: str) -> Decision:
        payload = self._json_object(text)
        action = payload.get("action")
        if action not in {"continue", "intervene"}:
            raise ValueError(f"Invalid supervisor action: {action!r}")
        correction = str(payload.get("correction") or "").strip()
        if action == "intervene" and not correction:
            raise ValueError("Supervisor intervention has no correction")
        if action == "continue":
            correction = ""
        if action == "intervene":
            level = self._next_level() if trigger == "follow_up" else "constraint"
        else:
            level = "constraint"
        return Decision(
            action=action,
            reason=str(payload.get("reason") or ""),
            correction=correction,
            level=level,
            target=str(payload.get("target") or target),
        )

    def _next_level(self) -> str:
        return {None: "constraint", "constraint": "procedure", "procedure": "answer"}.get(
            self.last_level, "answer"
        )

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        candidates = [text.strip(), *re.findall(r"```(?:json)?\s*(.*?)```", text, re.S)]
        decoder = json.JSONDecoder()
        for candidate in reversed(candidates):
            for index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate, index)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
        raise ValueError("Supervisor did not return a JSON object")
