"""Isolated GA memory workspaces and sparse overlays."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from ga_harness.events import atomic_write_json, utc_now

_IGNORED_NAMES = {
    "__pycache__",
    "file_access_stats.json",
    ".DS_Store",
}
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


def _included(relative: Path) -> bool:
    if any(part in _IGNORED_NAMES for part in relative.parts):
        return False
    if any(part.lower() in _SENSITIVE_PARTS for part in relative.parts):
        return False
    if any(_SENSITIVE_NAME.search(part) for part in relative.parts):
        return False
    if relative.suffix.lower() in _IGNORED_SUFFIXES:
        return False
    return True


def _files(root: Path) -> dict[str, Path]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and _included(path.relative_to(root))
    }


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _contains_secret(path: Path) -> bool:
    try:
        if path.stat().st_size > 8 * 1024 * 1024:
            return True
        return _SECRET_CONTENT.search(path.read_bytes()) is not None
    except OSError:
        return True


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        return
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if not _included(relative):
            continue
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


class MemoryWorkspace:
    """Materialize official memory plus an optional sparse writable overlay."""

    def __init__(
        self,
        baseline: str | Path,
        state_root: str | Path,
        role: str,
        overlay: str | Path | None = None,
    ) -> None:
        self.baseline = Path(baseline).resolve()
        self.root = Path(state_root).resolve() / role / "memory"
        self.role = role
        self.overlay = Path(overlay).resolve() if overlay else None

    def prepare(self) -> Path:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        _copy_tree(self.baseline, self.root)
        if self.overlay and self.overlay.exists():
            _copy_tree(self.overlay, self.root)
            manifest_path = self.overlay.parent / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for relative in manifest.get("deleted", []):
                    target = self.root / relative
                    if target.is_file() or target.is_symlink():
                        target.unlink()
                    elif target.is_dir():
                        shutil.rmtree(target)
        return self.root

    def export_delta(self, destination: str | Path, base_commit: str) -> Path:
        destination = Path(destination).resolve()
        staging = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        if staging.exists():
            shutil.rmtree(staging)
        overlay_dir = staging / "overlay"
        overlay_dir.mkdir(parents=True)
        baseline_files = _files(self.baseline)
        runtime_files = _files(self.root)
        changed: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for relative, runtime_path in sorted(runtime_files.items()):
            if _contains_secret(runtime_path):
                rejected.append({"path": relative, "reason": "sensitive_content"})
                continue
            baseline_path = baseline_files.get(relative)
            digest = _digest(runtime_path)
            if baseline_path is not None and digest == _digest(baseline_path):
                continue
            target = overlay_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime_path, target)
            changed.append(
                {
                    "path": relative,
                    "status": "modified" if baseline_path else "added",
                    "sha256": digest,
                }
            )
        deleted = sorted(set(baseline_files) - set(runtime_files))
        manifest = {
            "schema_version": 1,
            "role": self.role,
            "base_commit": base_commit,
            "created_at": utc_now(),
            "files": changed,
            "deleted": deleted,
            "rejected": rejected,
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
        return destination


def merge_delta(
    baseline: str | Path,
    current_store: str | Path,
    delta: str | Path,
    base_commit: str,
    role: str,
) -> None:
    """Atomically replace one rolling overlay with the materialized new state."""
    baseline = Path(baseline).resolve()
    current_store = Path(current_store).resolve()
    delta = Path(delta).resolve()
    merge_root = current_store.parent / f".{role}-merge-{uuid.uuid4().hex}"
    workspace = MemoryWorkspace(
        baseline,
        merge_root,
        role,
        current_store / "overlay" if current_store.exists() else None,
    )
    runtime = workspace.prepare()
    _copy_tree(delta / "overlay", runtime)
    delta_manifest = delta / "manifest.json"
    if delta_manifest.exists():
        payload = json.loads(delta_manifest.read_text(encoding="utf-8"))
        for relative in payload.get("deleted", []):
            target = runtime / relative
            if target.is_file() or target.is_symlink():
                target.unlink()
            elif target.is_dir():
                shutil.rmtree(target)
    workspace.export_delta(current_store, base_commit)
    shutil.rmtree(merge_root, ignore_errors=True)
