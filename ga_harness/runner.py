"""Standalone ``ga run`` entrypoint for benchmark and developer use."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from ga_config import script_dir
from ga_runtime import GenericAgent
from ga_harness.control import HarnessControl
from ga_harness.curator import consolidate_memory, distill_run_memory
from ga_harness.events import EventRecorder, atomic_write_json, utc_now
from ga_harness.memory import MemoryBank, publish_store
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


def _prepare_memories(args, state_dir: Path):
    source_root = Path(script_dir)
    store = Path(args.memory_store).resolve() if args.memory_store else None
    runtime_root = state_dir / "runtime-memory"
    banks: dict[str, MemoryBank] = {}
    paths: dict[str, Path] = {}
    for role in ("worker", "supervisor"):
        source = getattr(args, f"{role}_memory_source")
        role_store = store / role if store else None
        bank = MemoryBank(source_root, runtime_root, role, source, role_store)
        banks[role] = bank
        paths[role] = bank.prepare()
    return source_root, store, banks, paths


def _drain_queue(agent: GenericAgent, result: dict[str, Any]) -> None:
    """Print the final answer without duplicating every queued live snapshot."""
    output = result.get("output") or ""
    if output:
        print(output, flush=True)


def _persist_memories(
    args,
    context,
    base_commit: str,
    allow_promote: bool,
) -> dict[str, Any]:
    delta_root = context["logs_dir"] / "memory-delta"
    summaries: dict[str, Any] = {}
    for role, bank in context["banks"].items():
        candidate = bank.export_delta(delta_root / role, base_commit)
        manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        promote = allow_promote and getattr(args, f"promote_{role}_memory")
        item: dict[str, Any] = {
            "source": getattr(args, f"{role}_memory_source"),
            "promote": promote,
            "changed": len(manifest["files"]),
            "deleted": len(manifest["deleted"]),
            "rejected": len(manifest.get("rejected", [])),
            "candidate": str(candidate),
        }
        if promote:
            try:
                if context["store"] is None:
                    raise ValueError(
                        f"Cannot promote {role} memory without --memory-store"
                    )
                role_store = context["store"] / role
                with tempfile.TemporaryDirectory(
                    prefix=f"ga-{role}-promote-", dir=context["state_dir"]
                ) as temporary:
                    staged = Path(temporary) / "store"
                    report = consolidate_memory(
                        role,
                        role_store,
                        None,
                        [candidate],
                        staged,
                        base_commit=base_commit,
                        source_root=context["source_root"],
                        state_dir=temporary,
                    )
                    item["generation"] = publish_store(staged, role_store)
                    item["curator"] = report
            except Exception as error:
                item["promotion_error"] = f"{type(error).__name__}: {error}"
        summaries[role] = item
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
    for role in ("worker", "supervisor"):
        parser.add_argument(
            f"--{role}-memory-source",
            choices=("initial", "latest"),
            default="latest",
        )
        parser.add_argument(
            f"--no-promote-{role}-memory",
            dest=f"promote_{role}_memory",
            action="store_false",
            default=True,
        )
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
    source_root, store, banks, paths = _prepare_memories(args, state_dir)
    model = str(env_model_config("worker")["model"])
    recorder = EventRecorder(logs_dir, model=model)
    recorder.emit("user_prompt", content=instruction, origin="task")
    agent = GenericAgent(
        client=build_client("worker"),
        workspace=workspace,
        state_dir=state_dir / "worker",
        memory_dir=paths["worker"],
        disable_ask_user=True,
        disable_memory_write=False,
        disable_browser_tools=not args.enable_browser_tools,
        harness_mode=True,
        max_turns=args.max_turns,
    )
    agent.verbose = False
    supervisor = None
    if not args.no_supervisor:
        supervisor = ProgressiveSupervisor(
            instruction, workspace, paths["supervisor"], recorder
        )
    control = HarnessControl.from_env(recorder, supervisor, agent.interrupt_current_action)
    agent.attach_control(control)
    return {
        "workspace": workspace,
        "state_dir": state_dir,
        "logs_dir": logs_dir,
        "instruction": instruction,
        "source_root": source_root,
        "store": store,
        "banks": banks,
        "model": model,
        "recorder": recorder,
        "agent": agent,
        "supervisor": supervisor,
        "supervisor_distillation": {"enabled": supervisor is not None},
        "control": control,
    }


def _execute_run(context):
    status, error, result = "completed", None, {}
    try:
        result = context["agent"].execute_task(
            context["instruction"], source="task", raise_errors=True
        )
        _drain_queue(context["agent"], result)
        if context["supervisor"] is not None:
            try:
                context["supervisor_distillation"] = distill_run_memory(
                    "supervisor",
                    context["banks"]["supervisor"].root,
                    [
                        context["logs_dir"] / "trajectory.json",
                        context["logs_dir"] / "supervision.jsonl",
                    ],
                    source_root=context["source_root"],
                    state_dir=context["state_dir"],
                )
            except Exception as distill_error:
                context["supervisor_distillation"] = {
                    "error": f"{type(distill_error).__name__}: {distill_error}"
                }
    except Exception as caught:
        status = "error"
        error = f"{type(caught).__name__}: {caught}"
        print(error, file=sys.stderr, flush=True)
    finally:
        context["control"].close()
    return status, error, result


def _finish_run(args, context, timing, outcome):
    started_at, started = timing
    status, error, result = outcome
    base_commit = _git_commit(Path(script_dir))
    memory = _persist_memories(
        args, context, base_commit, allow_promote=status == "completed"
    )
    exit_reason = _json_safe(result.get("exit_reason"))
    summary = {
        "schema_version": 2,
        "status": status,
        "started_at": started_at,
        "duration_sec": round(time.monotonic() - started, 3),
        "session_id": context["recorder"].session_id,
        "model": context["model"],
        "supervision": context["control"].summary(),
        "exit_reason": exit_reason,
        "memory": memory,
        "supervisor_memory_distillation": context["supervisor_distillation"],
        "error": error,
    }
    atomic_write_json(context["logs_dir"] / "ga-summary.json", summary)
    context["recorder"].finalize(status, error=error, exit_reason=exit_reason)
    return 0 if status == "completed" else 1


def run(args) -> int:
    timing = (utc_now(), time.monotonic())
    context = _setup_run(args)
    outcome = _execute_run(context)
    return _finish_run(args, context, timing, outcome)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
