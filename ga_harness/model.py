"""Model construction for the standalone GenericAgent harness."""

from __future__ import annotations

import os
from typing import Any

from llmcore import NativeOAISession, NativeToolClient


def env_model_config(role: str = "worker") -> dict[str, Any]:
    """Read role overrides while keeping worker and supervisor sessions separate."""
    role_prefix = "GA_SUPERVISOR_" if role == "supervisor" else "GA_"

    def value(name: str, *fallbacks: str) -> str | None:
        keys = [role_prefix + name]
        if role == "supervisor":
            keys.append("GA_" + name)
        keys.extend(fallbacks)
        for key in keys:
            item = os.environ.get(key)
            if item:
                return item
        return None

    api_base = value("API_BASE", "OPENAI_BASE_URL")
    api_key = value("API_KEY", "OPENAI_API_KEY")
    model = value("MODEL", "OPENAI_MODEL", "ANTHROPIC_MODEL")
    missing = [
        name
        for name, item in (
            ("API base", api_base),
            ("API key", api_key),
            ("model", model),
        )
        if not item
    ]
    if missing:
        raise ValueError("Missing GA model configuration: " + ", ".join(missing))
    config: dict[str, Any] = {
        "name": role,
        "apikey": api_key,
        "apibase": api_base,
        "model": model,
        "api_mode": value("API_MODE") or "chat_completions",
        "stream": True,
        "omit_thinking": False,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    for key, item in {
        "context_win": value("CONTEXT_WINDOW"),
        "max_tokens": value("MAX_OUTPUT_TOKENS"),
        "read_timeout": value("READ_TIMEOUT_SEC"),
        "max_retries": value("MAX_RETRIES"),
    }.items():
        if item is not None:
            config[key] = int(item)
    for key, name in (
        ("reasoning_effort", "REASONING_EFFORT"),
        ("service_tier", "SERVICE_TIER"),
    ):
        item = value(name)
        if item is not None:
            config[key] = item
    return config


def build_client(role: str = "worker") -> NativeToolClient:
    """Build a fresh native tool client; worker and supervisor never share history."""
    return NativeToolClient(NativeOAISession(env_model_config(role)))
