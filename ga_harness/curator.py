"""Semantic consolidation for role-isolated GenericAgent memory banks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ga_config import script_dir
from ga_harness.memory import (
    L0_NAME,
    MemoryBank,
    MemoryRole,
    materialize_bank,
    tree_digest,
    validate_memory,
)

Curator = Callable[[Path, str], dict[str, Any] | None]


def _prompt(role: MemoryRole, candidate_count: int) -> str:
    role_focus = (
        "reusable task-solving methods and verified technical practice"
        if role == "worker"
        else "how to improve Worker supervision, intervention, and completion judgment"
    )
    return f"""
You are the temporary GenericAgent Memory Curator for the {role} bank.
Consolidate {candidate_count} completed-run candidate memories into `memory/`.
The current latest bank is already copied into `memory/`; preserve every useful
existing lesson. `evidence/job_snapshot/` is the frozen bank the runs started
from. Each `evidence/candidate-NNN/` is one run's resulting bank.

Use your file tools to compare the snapshot and all candidates, then make small,
semantic edits only under `memory/`. Merge compatible lessons instead of letting
a later candidate overwrite an earlier one. Keep only durable, action-verified,
task-agnostic knowledge about {role_focus}. Do not copy task names, answers,
exact hidden checks, verifier text or numeric grading thresholds. Do not learn
from infrastructure failures. Do not edit `memory/{L0_NAME}`.

Maintain the L1-L4 contract from `{L0_NAME}`:
- L1 `global_mem_insight.txt` is an index/high-frequency red-line layer and must
  remain at most 30 lines.
- L2 `global_mem.txt` stores stable cross-run facts.
- L3 stores concise reusable SOPs.
- L4 stores only de-identified, task-agnostic summaries.

The {role} bank belongs only to {role}; never import knowledge from the other
role. If candidates add no safe reusable lesson, leave `memory/` unchanged.
Finish normally after re-reading the files you changed.
""".strip()


def _run_ga_curator(workspace: Path, prompt: str, role: MemoryRole) -> dict[str, Any]:
    from ga_harness.model import build_client
    from ga_runtime import GenericAgent

    agent = GenericAgent(
        client=build_client(role),
        workspace=workspace,
        state_dir=workspace / ".state",
        memory_dir=workspace / "memory",
        disable_ask_user=True,
        disable_memory_write=True,
        disable_browser_tools=True,
        harness_mode=True,
        max_turns=40,
    )
    agent.verbose = False
    result = agent.execute_task(prompt, source="memory-curator", raise_errors=True)
    exit_reason = result.get("exit_reason") if isinstance(result, dict) else None
    outcome = exit_reason.get("result") if isinstance(exit_reason, dict) else None
    if outcome != "CURRENT_TASK_DONE":
        raise RuntimeError(f"Memory Curator did not finish: {outcome or 'unknown'}")
    return result


def _candidate_has_changes(path: Path) -> bool:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return bool(payload.get("files") or payload.get("deleted"))


def distill_run_memory(
    role: MemoryRole,
    memory_dir: str | Path,
    evidence_paths: Sequence[str | Path],
    *,
    source_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    curator: Curator | None = None,
) -> dict[str, Any]:
    """Curate one completed run into its materialized role bank."""
    source = Path(source_root or script_dir).resolve()
    memory_path = Path(memory_dir).resolve()
    parent = Path(state_dir).resolve() if state_dir else memory_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"ga-{role}-run-curator-", dir=parent) as raw:
        workspace = Path(raw) / "workspace"
        curated = workspace / "memory"
        shutil.copytree(memory_path, curated)
        evidence = workspace / "evidence" / "run"
        evidence.mkdir(parents=True)
        for source_path in map(Path, evidence_paths):
            if source_path.is_file():
                shutil.copy2(source_path, evidence / source_path.name)
        prompt = f"""
You are the temporary GenericAgent Memory Curator for the {role} bank after one
completed run. Read the observable run records in `evidence/run/` and maintain
only `memory/`. Preserve useful existing lessons. Store only durable,
action-verified, task-agnostic guidance. Never retain task identity, answers,
exact commands, hidden checks, verifier data, grading criteria, numeric task
constants, or infrastructure failures. Do not edit `memory/{L0_NAME}`. Keep L1
at most 30 lines; put concise reusable detail in L3 and only de-identified
summaries in L4. If there is no safe new lesson, leave the bank unchanged.
""".strip()
        before = tree_digest(curated)
        result = (
            curator(workspace, prompt) or {}
            if curator is not None
            else _run_ga_curator(workspace, prompt, role)
        )
        validate_memory(curated, source, role)
        replacement = memory_path.with_name(f".{memory_path.name}.{uuid.uuid4().hex}.tmp")
        shutil.copytree(curated, replacement)
        backup = memory_path.with_name(f".{memory_path.name}.{uuid.uuid4().hex}.old")
        os.replace(memory_path, backup)
        try:
            os.replace(replacement, memory_path)
        except Exception:
            os.replace(backup, memory_path)
            raise
        shutil.rmtree(backup)
        return {
            "role": role,
            "before_sha256": before,
            "after_sha256": tree_digest(memory_path),
            "changed": before != tree_digest(memory_path),
            "result": result,
        }


def consolidate_memory(
    role: MemoryRole,
    current_latest: str | Path | None,
    job_snapshot: str | Path | None,
    candidates: Sequence[str | Path],
    output: str | Path,
    *,
    base_commit: str,
    source_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    curator: Curator | None = None,
) -> dict[str, Any]:
    """Semantically merge candidate banks and export one validated store.

    Inputs and output are sparse role stores. Candidates are materialized against
    the role-specific initial bank before the Curator sees them.
    """
    source = Path(source_root or script_dir).resolve()
    candidate_paths = [Path(path).resolve() for path in candidates]
    candidate_paths = [
        path
        for path in candidate_paths
        if (path / "manifest.json").is_file() and _candidate_has_changes(path)
    ]
    output_path = Path(output).resolve()
    temp_parent = Path(state_dir).resolve() if state_dir else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"ga-{role}-curator-", dir=temp_parent) as raw:
        root = Path(raw)
        workspace = root / "workspace"
        memory = materialize_bank(source, role, current_latest, workspace / "memory")
        evidence = workspace / "evidence"
        materialize_bank(source, role, job_snapshot, evidence / "job_snapshot")
        for index, candidate in enumerate(candidate_paths, 1):
            materialize_bank(
                source,
                role,
                candidate,
                evidence / f"candidate-{index:03d}",
            )
        before = tree_digest(memory)
        result: dict[str, Any] = {}
        if candidate_paths:
            prompt = _prompt(role, len(candidate_paths))
            result = (
                curator(workspace, prompt) or {}
                if curator is not None
                else _run_ga_curator(workspace, prompt, role)
            )
        validate_memory(memory, source, role)
        bank = MemoryBank(source, root / "export-state", role, "initial")
        bank.root = memory
        bank.export_delta(output_path, base_commit)
        return {
            "role": role,
            "candidate_count": len(candidate_paths),
            "before_sha256": before,
            "after_sha256": tree_digest(memory),
            "output_sha256": tree_digest(output_path),
            "changed": before != tree_digest(memory),
            "result": result,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ga memory-consolidate")
    parser.add_argument("--role", choices=("worker", "supervisor"), required=True)
    parser.add_argument("--current-latest")
    parser.add_argument("--job-snapshot")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--source-root", default=script_dir)
    parser.add_argument("--state-dir")
    parser.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = consolidate_memory(
        args.role,
        args.current_latest,
        args.job_snapshot,
        args.candidate,
        args.output,
        base_commit=args.base_commit,
        source_root=args.source_root,
        state_dir=args.state_dir,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.report:
        Path(args.report).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
