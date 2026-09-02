"""Read-only task and memory tools exposed to the GA supervisor."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from ga_harness.events import EventRecorder


def _tool(name: str, description: str, properties: dict[str, Any], required=()):
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = list(required)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


READ_ONLY_TOOLS = [
    _tool(
        "read_file",
        "Read a text file from the task workspace.",
        {"path": {"type": "string"}},
        ("path",),
    ),
    _tool(
        "list_directory",
        "List paths below a task workspace directory.",
        {"path": {"type": "string"}},
    ),
    _tool(
        "search_text",
        "Search literal text in task workspace files.",
        {"path": {"type": "string"}, "query": {"type": "string"}},
        ("query",),
    ),
    _tool(
        "recent_trajectory",
        "Read recent observable Worker events without reasoning.",
        {},
    ),
    _tool(
        "memory_list",
        "List this Supervisor's L2/L3 memory files. L1 is already in context.",
        {"path": {"type": "string"}},
    ),
    _tool(
        "memory_read",
        "Read one file from this Supervisor's L2/L3 memory bank.",
        {"path": {"type": "string"}},
        ("path",),
    ),
]

_PROTECTED_NAMES = {
    "verifier",
    "grader",
    "grading",
    "rubric",
    "reference_answer",
    "reward",
}
_MEMORY_HIDDEN = {
    "memory_management_sop.md",
    "global_mem_insight.txt",
    "l4_raw_sessions",
    "worker",
}


class ReadOnlyWorkspace:
    """Expose bounded evidence without crossing task or Supervisor-bank roots."""

    def __init__(
        self,
        workspace: str | Path,
        recorder: EventRecorder,
        memory_dir: str | Path | None = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.memory_dir = Path(memory_dir).resolve() if memory_dir else None
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
            if name == "memory_list":
                return self.memory_list(arguments)
            if name == "memory_read":
                return self.memory_read(arguments)
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

    def memory_list(self, arguments: dict[str, Any]) -> str:
        root = self._memory_path(str(arguments.get("path") or "."), allow_root=True)
        if not root.is_dir():
            raise ValueError("Not a memory directory")
        items = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.memory_dir)
            if self._memory_protected(relative):
                continue
            items.append(relative.as_posix())
            if len(items) >= 200:
                break
        return "\n".join(items)

    def memory_read(self, arguments: dict[str, Any]) -> str:
        path = self._memory_path(str(arguments.get("path") or ""))
        if not path.is_file():
            raise ValueError("Not a memory file")
        return path.read_text(encoding="utf-8", errors="replace")[:32_000]

    def safe_path(self, raw: str) -> Path:
        path = self._confined_path(self.workspace, raw)
        relative = path.relative_to(self.workspace)
        if self._protected(relative):
            raise ValueError("Grading and solution paths are unavailable during execution")
        return path

    def _memory_path(self, raw: str, *, allow_root: bool = False) -> Path:
        if self.memory_dir is None:
            raise ValueError("Supervisor memory is unavailable")
        path = self._confined_path(self.memory_dir, raw)
        relative = path.relative_to(self.memory_dir)
        if (not allow_root or relative.parts) and self._memory_protected(relative):
            raise ValueError("Only this Supervisor's L2/L3 memory is available")
        return path

    @staticmethod
    def _confined_path(root: Path, raw: str) -> Path:
        lexical = PurePosixPath(raw)
        if lexical.is_absolute() or ".." in lexical.parts:
            raise ValueError("Absolute paths and parent traversal are unavailable")
        path = (root / raw).resolve()
        if path != root and root not in path.parents:
            raise ValueError("Path escapes its read-only root")
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

    @staticmethod
    def _memory_protected(relative: Path) -> bool:
        lowered = {part.lower() for part in relative.parts}
        return not relative.parts or bool(lowered & (_PROTECTED_NAMES | _MEMORY_HIDDEN))
