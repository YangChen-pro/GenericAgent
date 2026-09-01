"""Shared GenericAgent paths, prompts, and tool-schema configuration."""

from __future__ import annotations

import json
import locale
import os
import sys
import time

os.environ.setdefault(
    "GA_LANG",
    "zh"
    if any(
        key in (locale.getlocale()[0] or "").lower()
        for key in ("zh", "chinese")
    )
    else "en",
)

script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY_DIR = os.path.join(script_dir, "memory")


def load_tool_schema(suffix="", banned_tools=None):
    """Load one schema copy and optionally remove explicitly disabled tools."""
    path = os.path.join(script_dir, f"assets/tools_schema{suffix}.json")
    raw = open(path, "r", encoding="utf-8").read()
    schema = json.loads(raw if os.name == "nt" else raw.replace("powershell", "bash"))
    banned = set(banned_tools or ())
    return [
        tool
        for tool in schema
        if tool.get("function", {}).get("name") not in banned
    ]


def ensure_default_memory() -> None:
    os.makedirs(DEFAULT_MEMORY_DIR, exist_ok=True)
    language = "_en" if os.environ.get("GA_LANG") == "en" else ""
    files = (
        ("global_mem.txt", None),
        ("global_mem_insight.txt", f"assets/global_mem_insight_template{language}.txt"),
    )
    for name, template in files:
        target = os.path.join(DEFAULT_MEMORY_DIR, name)
        if os.path.exists(target):
            continue
        content = "# [Global Memory - L2]\n" if template is None else ""
        source = os.path.join(script_dir, template) if template else None
        if source and os.path.exists(source):
            content = open(source, encoding="utf-8").read()
        open(target, "w", encoding="utf-8").write(content)


def get_system_prompt(memory_dir=None, workspace=None):
    """Build the regular GA system prompt against a selected memory workspace."""
    from ga import get_global_memory

    language = "_en" if os.environ.get("GA_LANG") == "en" else ""
    path = os.path.join(script_dir, f"assets/sys_prompt{language}.txt")
    with open(path, "r", encoding="utf-8") as handle:
        prompt = handle.read()
    prompt += f"\nToday: {time.strftime('%Y-%m-%d %a')}\n"
    prompt += get_global_memory(memory_dir=memory_dir, cwd=workspace)
    return prompt


def configure_stdio() -> None:
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    elif hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    elif hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(errors="replace")


ensure_default_memory()
