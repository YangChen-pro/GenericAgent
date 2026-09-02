"""Role-isolated, layered memory banks for GenericAgent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal

from ga_harness.events import atomic_write_json, utc_now

MemoryRole = Literal["worker", "supervisor"]
MemorySource = Literal["initial", "latest"]
ROLES: tuple[MemoryRole, ...] = ("worker", "supervisor")
L0_NAME = "memory_management_sop.md"
L1_NAME = "global_mem_insight.txt"
L2_NAME = "global_mem.txt"
L4_NAME = "L4_raw_sessions"

_IGNORED_NAMES = {"__pycache__", "file_access_stats.json", ".DS_Store"}
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
_SENSITIVE_PARTS = {".env", "credentials", "secrets"}
_SENSITIVE_NAME = re.compile(
    r"(?:^|[._-])(api[_-]?key|auth[_-]?token|access[_-]?token|password|secret)(?:[._-]|$)",
    re.IGNORECASE,
)
_SECRET_CONTENT = re.compile(
    rb"(?i)(?:api[_-]?key|auth[_-]?token|access[_-]?token|password|secret)"
    rb"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{16,}|sk-[A-Za-z0-9_-]{16,}"
)


def role_initial_root(source_root: str | Path, role: MemoryRole) -> Path:
    """Return the committed initial memory bank for one role."""
    root = Path(source_root).resolve()
    if role == "worker":
        return root / "memory"
    if role == "supervisor":
        return root / "ga_harness" / "supervisor_memory"
    raise ValueError(f"Unsupported memory role: {role}")


def shared_l0_path(source_root: str | Path) -> Path:
    return Path(source_root).resolve() / "memory" / L0_NAME


def _included(relative: Path) -> bool:
    if any(part in _IGNORED_NAMES for part in relative.parts):
        return False
    if any(part.lower() in _SENSITIVE_PARTS for part in relative.parts):
        return False
    if any(_SENSITIVE_NAME.search(part) for part in relative.parts):
        return False
    return relative.suffix.lower() not in _IGNORED_SUFFIXES


def memory_files(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    if not base.exists():
        return {}
    return {
        path.relative_to(base).as_posix(): path
        for path in base.rglob("*")
        if path.is_file() and not path.is_symlink() and _included(path.relative_to(base))
    }


def file_digest(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(root: str | Path) -> str:
    value = hashlib.sha256()
    for relative, path in sorted(memory_files(root).items()):
        value.update(relative.encode("utf-8"))
        value.update(b"\0")
        value.update(file_digest(path).encode("ascii"))
        value.update(b"\n")
    return value.hexdigest()


def _contains_secret(path: Path) -> bool:
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return True
        return _SECRET_CONTENT.search(path.read_bytes()) is not None
    except OSError:
        return True


def copy_memory(source: str | Path, target: str | Path) -> None:
    source_path, target_path = Path(source), Path(target)
    if not source_path.exists():
        return
    for path in source_path.rglob("*"):
        relative = path.relative_to(source_path)
        if not _included(relative) or path.is_symlink():
            continue
        destination = target_path / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def resolve_store(store: str | Path | None) -> Path | None:
    """Resolve a role store, preferring its atomically published generation."""
    if store is None:
        return None
    root = Path(store).resolve()
    pointer = root / "current.json"
    if pointer.is_file():
        generation = str(json.loads(pointer.read_text(encoding="utf-8"))["generation"])
        resolved = (root / "versions" / generation).resolve()
        if root not in resolved.parents:
            raise ValueError("Memory generation escapes its role store")
        return resolved if (resolved / "manifest.json").is_file() else None
    current = root / "current"
    if current.exists():
        resolved = current.resolve()
        if root not in resolved.parents:
            raise ValueError("Memory generation escapes its role store")
        return resolved if (resolved / "manifest.json").is_file() else None
    return root if (root / "manifest.json").is_file() else None


def _safe_relative(raw: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe memory path: {raw}")
    return relative


def _apply_store(store: str | Path | None, target: Path, role: MemoryRole) -> None:
    resolved = resolve_store(store)
    if resolved is None:
        return
    manifest = json.loads((resolved / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("role") != role:
        raise ValueError(f"Expected {role} memory, found {manifest.get('role')!r}")
    copy_memory(resolved / "overlay", target)
    for raw_relative in manifest.get("deleted", []):
        relative = _safe_relative(str(raw_relative))
        if relative.as_posix() == L0_NAME:
            continue
        path = target / relative
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def _validate_structure(root: Path, role: MemoryRole) -> None:
    for required in (L0_NAME, L1_NAME, L2_NAME):
        path = root / required
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{role} memory is missing {required}")
    if len((root / L1_NAME).read_text(encoding="utf-8").splitlines()) > 30:
        raise ValueError(f"{role} memory L1 exceeds 30 lines")


def _validate_contract(root: Path, source_root: str | Path, role: MemoryRole) -> None:
    _validate_structure(root, role)
    if file_digest(root / L0_NAME) != file_digest(shared_l0_path(source_root)):
        raise ValueError(f"{role} memory modified shared read-only {L0_NAME}")


def _rejection_reason(
    relative: Path, path: Path, initial: Path | None = None
) -> str | None:
    if path.is_symlink():
        return "symlink"
    if not path.is_file():
        return None
    if not _included(relative):
        return "sensitive_or_unsupported_path"
    unchanged_initial = (
        initial is not None and initial.is_file() and file_digest(initial) == file_digest(path)
    )
    if _contains_secret(path) and not unchanged_initial:
        return "sensitive_content"
    return None


def validate_memory(root: str | Path, source_root: str | Path, role: MemoryRole) -> None:
    """Fail closed when a materialized bank violates the shared layer contract."""
    memory = Path(root).resolve()
    _validate_contract(memory, source_root, role)
    initial_files = memory_files(role_initial_root(source_root, role))
    for path in memory.rglob("*"):
        relative = path.relative_to(memory)
        reason = _rejection_reason(relative, path, initial_files.get(relative.as_posix()))
        if reason is not None:
            raise ValueError(
                f"{role} memory contains a rejected file: {path.name} ({reason})"
            )


class MemoryBank:
    """Materialize and export one role without exposing the other role's bank."""

    def __init__(
        self,
        source_root: str | Path,
        state_root: str | Path,
        role: MemoryRole,
        source: MemorySource = "latest",
        store: str | Path | None = None,
    ) -> None:
        if source not in {"initial", "latest"}:
            raise ValueError(f"Unsupported memory source: {source}")
        self.source_root = Path(source_root).resolve()
        self.initial = role_initial_root(self.source_root, role)
        self.root = Path(state_root).resolve() / role / "memory"
        self.role = role
        self.source = source
        self.store = Path(store).resolve() if store else None

    def prepare(self) -> Path:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        copy_memory(self.initial, self.root)
        shutil.copy2(shared_l0_path(self.source_root), self.root / L0_NAME)
        if self.source == "latest":
            _apply_store(self.store, self.root, self.role)
        # L0 is always restored after applying an untrusted prior overlay.
        shutil.copy2(shared_l0_path(self.source_root), self.root / L0_NAME)
        validate_memory(self.root, self.source_root, self.role)
        return self.root

    def _initial_snapshot(self, staging: Path) -> Path:
        initial_root = staging / ".initial"
        initial_root.mkdir(parents=True)
        copy_memory(self.initial, initial_root)
        shutil.copy2(shared_l0_path(self.source_root), initial_root / L0_NAME)
        validate_memory(initial_root, self.source_root, self.role)
        return initial_root

    def _delta_entries(
        self, initial_files: dict[str, Path], overlay_dir: Path
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
        changed: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        accepted: set[str] = set()
        for path in sorted(self.root.rglob("*")):
            relative_path = path.relative_to(self.root)
            relative = relative_path.as_posix()
            if relative == L0_NAME:
                accepted.add(relative)
                if file_digest(path) != file_digest(shared_l0_path(self.source_root)):
                    rejected.append({"path": relative, "reason": "shared_read_only"})
                continue
            initial = initial_files.get(relative)
            reason = _rejection_reason(relative_path, path, initial)
            if reason is not None:
                rejected.append({"path": relative, "reason": reason})
                continue
            if not path.is_file():
                continue
            accepted.add(relative)
            digest = file_digest(path)
            if initial is not None and digest == file_digest(initial):
                continue
            target = overlay_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            changed.append(
                {
                    "path": relative,
                    "status": "modified" if initial else "added",
                    "sha256": digest,
                }
            )
        deleted = sorted((set(initial_files) - accepted) - {L0_NAME})
        return changed, deleted, rejected

    @staticmethod
    def _replace_candidate(staging: Path, destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(staging, destination)

    def export_delta(self, destination: str | Path, base_commit: str) -> Path:
        """Export a full role state as a sparse delta over its initial bank."""
        _validate_structure(self.root, self.role)
        destination = Path(destination).resolve()
        staging = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        shutil.rmtree(staging, ignore_errors=True)
        overlay_dir = staging / "overlay"
        overlay_dir.mkdir(parents=True)
        initial_root = self._initial_snapshot(staging)
        initial_files = memory_files(initial_root)
        changed, deleted, rejected = self._delta_entries(initial_files, overlay_dir)
        initial_digest = tree_digest(initial_root)
        shutil.rmtree(initial_root)
        manifest = {
            "schema_version": 2,
            "role": self.role,
            "source": self.source,
            "base_commit": base_commit,
            "created_at": utc_now(),
            "initial_sha256": initial_digest,
            "materialized_sha256": tree_digest(self.root),
            "files": changed,
            "deleted": deleted,
            "rejected": rejected,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        self._replace_candidate(staging, destination)
        return destination


def materialize_bank(
    source_root: str | Path,
    role: MemoryRole,
    store: str | Path | None,
    destination: str | Path,
) -> Path:
    """Materialize a role store into a caller-owned directory."""
    target = Path(destination).resolve()
    bank = MemoryBank(source_root, target.parent / f".{target.name}-state", role, "latest", store)
    bank.root = target
    return bank.prepare()


def publish_store(staged_store: str | Path, role_store: str | Path) -> str:
    """Publish an immutable generation and atomically switch the current pointer."""
    staged = Path(staged_store).resolve()
    role_root = Path(role_store).resolve()
    if not (staged / "manifest.json").is_file():
        raise ValueError(f"Incomplete memory store: {staged}")
    digest = tree_digest(staged)
    versions = role_root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    generation = versions / digest
    if not generation.exists():
        temporary = versions / f".{digest}.{uuid.uuid4().hex}.tmp"
        shutil.copytree(staged, temporary)
        try:
            os.replace(temporary, generation)
        except OSError:
            shutil.rmtree(temporary, ignore_errors=True)
            if not generation.exists():
                raise
    publication = {
        "schema_version": 1,
        "generation": digest,
        "published_at": utc_now(),
    }
    atomic_write_json(role_root / "current.json", publication)
    atomic_write_json(role_root / "publication.json", publication)
    return digest
