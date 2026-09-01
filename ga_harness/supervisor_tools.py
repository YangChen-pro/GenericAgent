"""Read-only, workspace-confined tools exposed to the GA supervisor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ga_harness.events import EventRecorder

READ_ONLY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from the task workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List paths below a task workspace directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Search literal text in task workspace files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_trajectory",
            "description": "Read recent observable Worker events without reasoning.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_PROTECTED_NAMES = {
    "verifier",
    "grader",
    "grading",
    "rubric",
    "reference_answer",
    "reward",
}


class ReadOnlyWorkspace:
    """Expose bounded evidence while preventing traversal and grading-data access."""

    def __init__(self, workspace: str | Path, recorder: EventRecorder) -> None:
        self.workspace = Path(workspace).resolve()
        self.recorder = recorder

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            if name == "read_file":
                return self.read_file(arguments)
            if name == "list_directory":
                return self.list_directory(arguments)
            if name == "search_text":
                return self.search_text(arguments)
            if name == "recent_trajectory":
                return json.dumps(self.recorder.observable_events(20), ensure_ascii=False)
            return f"Read-only tool error: unknown tool {name}"
        except Exception as error:
            return f"Read-only tool error: {type(error).__name__}: {error}"

    def read_file(self, arguments: dict[str, Any]) -> str:
        path = self.safe_path(str(arguments.get("path") or "."))
        if not path.is_file():
            raise ValueError("Not a file")
        return path.read_text(encoding="utf-8", errors="replace")[:32_000]

    def list_directory(self, arguments: dict[str, Any]) -> str:
        path = self.safe_path(str(arguments.get("path") or "."))
        if not path.is_dir():
            raise ValueError("Not a directory")
        items = []
        for item in sorted(path.rglob("*")):
            relative = self._allowed_relative(item)
            if relative is None:
                continue
            display = item.relative_to(self.workspace)
            if len(display.parts) <= 2:
                items.append(display.as_posix() + ("/" if item.is_dir() else ""))
            if len(items) >= 400:
                break
        return "\n".join(items)

    def search_text(self, arguments: dict[str, Any]) -> str:
        root = self.safe_path(str(arguments.get("path") or "."))
        query = str(arguments.get("query") or "")
        if not query:
            raise ValueError("query is required")
        matches = []
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if len(matches) >= 400 or not path.is_file():
                continue
            relative = self._allowed_relative(path)
            if relative is None:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                if query in line:
                    matches.append(f"{relative.as_posix()}:{number}:{line[:500]}")
                    if len(matches) >= 400:
                        break
        return "\n".join(matches)[:32_000]

    def safe_path(self, raw: str) -> Path:
        path = (self.workspace / raw).resolve()
        if path != self.workspace and self.workspace not in path.parents:
            raise ValueError("Path escapes the task workspace")
        relative = path.relative_to(self.workspace)
        if self._protected(relative):
            raise ValueError("Grading and solution paths are unavailable during execution")
        return path

    def _allowed_relative(self, path: Path) -> Path | None:
        try:
            relative = path.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return None
        return None if self._protected(relative) else relative

    @staticmethod
    def _protected(relative: Path) -> bool:
        return bool({part.lower() for part in relative.parts} & _PROTECTED_NAMES)
