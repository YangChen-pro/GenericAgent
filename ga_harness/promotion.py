"""Idempotent atomic publication for consolidated GA memories."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ga_config import script_dir
from ga_harness.events import atomic_write_json, utc_now
from ga_harness.memory import MemoryRole, publish_store, resolve_store, tree_digest

Curator = Callable[[Path, str], dict[str, Any] | None]


def _candidate_has_changes(path: Path) -> bool:
    payload = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return bool(payload.get("files") or payload.get("deleted"))


def _candidate_paths(candidates: Sequence[str | Path]) -> list[Path]:
    paths = [Path(path).resolve() for path in candidates]
    return [
        path
        for path in paths
        if (path / "manifest.json").is_file() and _candidate_has_changes(path)
    ]


def _promotion_inputs(
    promotion_id: str,
    role: MemoryRole,
    snapshot: str | Path | None,
    candidates: Sequence[Path],
) -> tuple[str, dict[str, Any]]:
    snapshot_path = Path(snapshot).resolve() if snapshot else None
    snapshot_hash = tree_digest(Path("/__ga_empty_memory__"))
    if snapshot_path is not None and snapshot_path.exists():
        snapshot_hash = tree_digest(resolve_store(snapshot_path) or snapshot_path)
    payload = {
        "promotion_id": promotion_id,
        "role": role,
        "snapshot_sha256": snapshot_hash,
        "candidate_sha256": [tree_digest(path) for path in candidates],
    }
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return identity, payload


def _current_hash(store: Path) -> str | None:
    current = resolve_store(store)
    return tree_digest(current) if current is not None else None


def _record(
    receipt: Path,
    inputs: dict[str, Any],
    identity: str,
    output_hash: str,
    status: str,
    curator: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        **inputs,
        "identity": identity,
        "output_sha256": output_hash,
        "created_at": utc_now(),
        "status": status,
        "curator": curator,
    }
    atomic_write_json(receipt, record)
    return record


def _existing_receipt(store: Path, receipt: Path) -> dict[str, Any] | None:
    if not receipt.is_file():
        return None
    record = json.loads(receipt.read_text(encoding="utf-8"))
    generation = store / "versions" / str(record["output_sha256"])
    if not generation.is_dir():
        raise ValueError(f"Promotion receipt has no generation: {receipt}")
    return {**record, "status": "already_applied"}


def _cleanup_prepared(staged: Path, marker: Path) -> None:
    shutil.rmtree(staged, ignore_errors=True)
    marker.unlink(missing_ok=True)


def _valid_publication_source(path: Path, output_hash: str) -> bool:
    return (path / "manifest.json").is_file() and tree_digest(path) == output_hash


def _recover_prepared(
    store: Path,
    staged: Path,
    marker: Path,
    receipt: Path,
    inputs: dict[str, Any],
    identity: str,
) -> dict[str, Any] | None:
    if not marker.is_file():
        return None
    prepared = json.loads(marker.read_text(encoding="utf-8"))
    output_hash = str(prepared["output_sha256"])
    generation = store / "versions" / output_hash
    current_hash = _current_hash(store)
    if current_hash == output_hash and generation.is_dir():
        record = _record(
            receipt,
            inputs,
            identity,
            output_hash,
            "recovered_published",
            prepared.get("curator", {}),
        )
        _cleanup_prepared(staged, marker)
        return record
    if current_hash == prepared.get("base_sha256"):
        source = staged if _valid_publication_source(staged, output_hash) else generation
        if _valid_publication_source(source, output_hash):
            published = publish_store(source, store)
            record = _record(
                receipt,
                inputs,
                identity,
                published,
                "recovered_prepared",
                prepared.get("curator", {}),
            )
            _cleanup_prepared(staged, marker)
            return record
    # The current bank advanced or the staged output is incomplete. Rebase the
    # same candidates on the now-current bank instead of rolling history back.
    _cleanup_prepared(staged, marker)
    return None


def _consolidate(
    role: MemoryRole,
    store: Path,
    snapshot: str | Path | None,
    candidates: Sequence[Path],
    staged: Path,
    *,
    base_commit: str,
    source: Path,
    state_dir: str | Path | None,
    curator: Curator | None,
) -> dict[str, Any]:
    from ga_harness.curator import consolidate_memory

    parent = Path(state_dir).resolve() if state_dir else store.parent
    parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(staged, ignore_errors=True)
    return consolidate_memory(
        role,
        store,
        snapshot,
        candidates,
        staged,
        base_commit=base_commit,
        source_root=source,
        state_dir=parent,
        curator=curator,
    )


def _apply_new_promotion(
    role: MemoryRole,
    store: Path,
    snapshot: str | Path | None,
    candidates: Sequence[Path],
    staged: Path,
    marker: Path,
    receipt: Path,
    inputs: dict[str, Any],
    identity: str,
    *,
    base_commit: str,
    source: Path,
    state_dir: str | Path | None,
    curator: Curator | None,
) -> dict[str, Any]:
    base_hash = _current_hash(store)
    report = _consolidate(
        role, store, snapshot, candidates, staged,
        base_commit=base_commit, source=source, state_dir=state_dir, curator=curator,
    )
    manifest = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    if not (manifest.get("files") or manifest.get("deleted")):
        shutil.rmtree(staged)
        return {
            "schema_version": 1,
            **inputs,
            "identity": identity,
            "created_at": utc_now(),
            "status": "unchanged",
            "curator": report,
        }
    prepared = {
        "base_sha256": base_hash,
        "output_sha256": tree_digest(staged),
        "curator": report,
    }
    atomic_write_json(marker, prepared)
    output_hash = publish_store(staged, store)
    record = _record(receipt, inputs, identity, output_hash, "applied", report)
    _cleanup_prepared(staged, marker)
    return record


def promote_memory(
    role: MemoryRole,
    role_store: str | Path,
    job_snapshot: str | Path | None,
    candidates: Sequence[str | Path],
    *,
    base_commit: str,
    promotion_id: str,
    source_root: str | Path | None = None,
    state_dir: str | Path | None = None,
    curator: Curator | None = None,
) -> dict[str, Any]:
    """Consolidate and atomically promote one role, idempotently."""
    source = Path(source_root or script_dir).resolve()
    store = Path(role_store).resolve()
    candidate_paths = _candidate_paths(candidates)
    identity, inputs = _promotion_inputs(
        promotion_id, role, job_snapshot, candidate_paths
    )
    receipt = store / "promotions" / f"{identity}.json"
    existing = _existing_receipt(store, receipt)
    if existing is not None:
        return existing
    if not candidate_paths:
        return {**inputs, "identity": identity, "status": "no_candidates"}
    staged = store / "promotions" / "staged" / identity
    marker = staged.with_suffix(".json")
    recovered = _recover_prepared(store, staged, marker, receipt, inputs, identity)
    if recovered is not None:
        return recovered
    return _apply_new_promotion(
        role, store, job_snapshot, candidate_paths, staged, marker, receipt,
        inputs, identity, base_commit=base_commit, source=source,
        state_dir=state_dir, curator=curator,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ga memory-promote")
    parser.add_argument("--role", choices=("worker", "supervisor"), required=True)
    parser.add_argument("--role-store", required=True)
    parser.add_argument("--job-snapshot")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--promotion-id", required=True)
    parser.add_argument("--source-root", default=script_dir)
    parser.add_argument("--state-dir")
    parser.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = promote_memory(
        args.role,
        args.role_store,
        args.job_snapshot,
        args.candidate,
        base_commit=args.base_commit,
        promotion_id=args.promotion_id,
        source_root=args.source_root,
        state_dir=args.state_dir,
    )
    encoded = json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n"
    if args.report:
        Path(args.report).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
