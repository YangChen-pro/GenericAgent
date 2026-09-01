"""Standalone ``ga run`` entrypoint for benchmark and developer use."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ga_config import script_dir
from ga_runtime import GenericAgent
from ga_harness.control import HarnessControl
from ga_harness.events import EventRecorder, atomic_write_json, utc_now
from ga_harness.memory import MemoryWorkspace, merge_delta
from ga_harness.model import build_client, env_model_config
from ga_harness.supervisor import ProgressiveSupervisor


def _load_env_file(path: str | Path | None) -> None:
    if not path:
        return
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Environment file not found: {source}")
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or not key.replace("_", "a").isalnum():
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _git_commit(root: Path) -> str:
    configured = os.environ.get("GA_BASE_COMMIT")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _memory_overlay(store: Path | None, role: str, clean: bool) -> Path | None:
    if store is None or clean:
        return None
    path = store / role / "overlay"
    return path if path.exists() else None


def _prepare_memories(args, state_dir: Path):
    baseline = Path(script_dir) / "memory"
    store = Path(args.memory_store).resolve() if args.memory_store else None
    root = state_dir / "runtime-memory"
    worker = MemoryWorkspace(
        baseline, root, "worker", _memory_overlay(store, "worker", args.clean_memory)
    )
    supervisor = MemoryWorkspace(
        baseline,
        root,
        "supervisor",
        _memory_overlay(store, "supervisor", args.clean_memory),
    )
    return baseline, store, worker, supervisor, worker.prepare(), supervisor.prepare()


def _drain_queue(agent: GenericAgent, result: dict[str, Any]) -> None:
    """Print the final answer without duplicating every queued live snapshot."""
    output = result.get("output") or ""
    if output:
        print(output, flush=True)


def _distill_supervisor_memory(
    client,
    memory_dir: Path,
    state_dir: Path,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}
    prompt = (
        "The supervised run has ended. Maintain your own GA long-term memory now. "
        "Read memory_management_sop.md and update only durable, task-agnostic lessons "
        "about supervision quality. Never store the task text, answer, exact commands, "
        "filenames, constants, verifier information, or grading criteria. If there is "
        "no safe general lesson, make no change and finish."
    )
    agent = GenericAgent(
        client=client,
        workspace=memory_dir,
        state_dir=state_dir / "supervisor-distill",
        memory_dir=memory_dir,
        disable_ask_user=True,
        disable_memory_write=False,
        harness_mode=True,
        max_turns=12,
    )
    agent.verbose = False
    result = agent.execute_task(prompt, source="memory", raise_errors=True)
    return {"enabled": True, "exit_reason": result.get("exit_reason")}


def _persist_memories(
    args,
    baseline: Path,
    store: Path | None,
    workspaces: dict[str, MemoryWorkspace],
    logs_dir: Path,
    base_commit: str,
    allow_write: bool = True,
) -> dict[str, Any]:
    delta_root = logs_dir / "memory-delta"
    summaries = {}
    for role, workspace in workspaces.items():
        destination = workspace.export_delta(delta_root / role, base_commit)
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        summaries[role] = {
            "changed": len(manifest["files"]),
            "deleted": len(manifest["deleted"]),
            "rejected": len(manifest.get("rejected", [])),
            "path": str(destination),
        }
        if store is None or args.no_memory_write or not allow_write:
            continue
        target = store / role
        if args.clean_memory:
            workspace.export_delta(target, base_commit)
        else:
            merge_delta(baseline, target, destination, base_commit, role)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ga run")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instruction-file")
    group.add_argument("--instruction")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--env-file")
    parser.add_argument("--memory-store")
    parser.add_argument("--clean-memory", action="store_true")
    parser.add_argument("--no-memory-write", action="store_true")
    parser.add_argument("--no-supervisor", action="store_true")
    parser.add_argument(
        "--enable-browser-tools",
        action="store_true",
        help="Enable GA web tools; disabled by default in CLI/Docker harness runs.",
    )
    parser.add_argument("--max-turns", type=int, default=180)
    return parser


def _setup_run(args):
    _load_env_file(args.env_file)
    workspace = Path(args.workspace).resolve()
    state_dir = Path(args.state_dir).resolve()
    logs_dir = Path(args.logs_dir).resolve()
    if not workspace.is_dir():
        raise NotADirectoryError(f"Workspace not found: {workspace}")
    state_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    instruction = (
        Path(args.instruction_file).read_text(encoding="utf-8")
        if args.instruction_file
        else args.instruction
    )
    memories = _prepare_memories(args, state_dir)
    baseline, store, worker_ws, supervisor_ws, worker_memory, supervisor_memory = memories
    model = str(env_model_config("worker")["model"])
    recorder = EventRecorder(logs_dir, model=model)
    recorder.emit("user_prompt", content=instruction, origin="task")
    agent = GenericAgent(
        client=build_client("worker"),
        workspace=workspace,
        state_dir=state_dir / "worker",
        memory_dir=worker_memory,
        disable_ask_user=True,
        disable_memory_write=args.no_memory_write,
        disable_browser_tools=not args.enable_browser_tools,
        harness_mode=True,
        max_turns=args.max_turns,
    )
    supervisor = None
    if not args.no_supervisor:
        supervisor = ProgressiveSupervisor(
            instruction, workspace, supervisor_memory, recorder
        )
    control = HarnessControl.from_env(
        recorder, supervisor, agent.interrupt_current_action
    )
    agent.attach_control(control)
    return {
        "workspace": workspace,
        "state_dir": state_dir,
        "logs_dir": logs_dir,
        "instruction": instruction,
        "baseline": baseline,
        "store": store,
        "workspaces": {"worker": worker_ws, "supervisor": supervisor_ws},
        "supervisor_memory": supervisor_memory,
        "model": model,
        "recorder": recorder,
        "agent": agent,
        "supervisor": supervisor,
        "control": control,
    }


def _execute_run(args, context):
    status, error, result = "completed", None, {}
    distillation = {"enabled": False}
    try:
        result = context["agent"].execute_task(
            context["instruction"], source="task", raise_errors=True
        )
        _drain_queue(context["agent"], result)
        if context["supervisor"] is not None:
            distillation = _distill_supervisor_memory(
                context["supervisor"].client,
                context["supervisor_memory"],
                context["state_dir"],
                not args.no_memory_write,
            )
    except Exception as caught:
        status = "error"
        error = f"{type(caught).__name__}: {caught}"
        print(error, file=sys.stderr, flush=True)
    finally:
        context["control"].close()
    return status, error, result, distillation


def _finish_run(args, context, timing, outcome):
    started_at, started = timing
    status, error, result, distillation = outcome
    base_commit = _git_commit(Path(script_dir))
    memory = _persist_memories(
        args,
        context["baseline"],
        context["store"],
        context["workspaces"],
        context["logs_dir"],
        base_commit,
        allow_write=status == "completed",
    )
    exit_reason = _json_safe(result.get("exit_reason"))
    summary = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at,
        "duration_sec": round(time.monotonic() - started, 3),
        "session_id": context["recorder"].session_id,
        "model": context["model"],
        "supervision": context["control"].summary(),
        "exit_reason": exit_reason,
        "memory": memory,
        "supervisor_memory_distillation": distillation,
        "error": error,
    }
    atomic_write_json(context["logs_dir"] / "ga-summary.json", summary)
    context["recorder"].finalize(
        status, error=error, exit_reason=exit_reason
    )
    return 0 if status == "completed" else 1


def run(args) -> int:
    timing = (utc_now(), time.monotonic())
    context = _setup_run(args)
    outcome = _execute_run(args, context)
    return _finish_run(args, context, timing, outcome)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
