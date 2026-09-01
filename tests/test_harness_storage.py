import sys

# A legacy frontend test installs collection-time module stubs. Restore the real
# core modules before importing harness code.
for _module_name in ("agent_loop", "llmcore", "agentmain"):
    _module = sys.modules.get(_module_name)
    if _module is not None and not getattr(_module, "__file__", None):
        sys.modules.pop(_module_name, None)

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ga_harness.events import EventRecorder
from ga_harness.memory import MemoryWorkspace, merge_delta
from ga_harness.model import env_model_config
from ga_harness.supervisor import ProgressiveSupervisor
from ga_harness.supervisor_tools import ReadOnlyWorkspace
from ga_runtime import GenericAgent


class DummyBackend:
    def __init__(self):
        self.history = []
        self.name = "dummy"
        self.model = "dummy"
        self.maxlen_multiplier = 1
        self.extra_sys_prompt = ""
        self.stream_observer = None
        self.should_stop = lambda: False
        self.active_response = None


class DummyClient:
    def __init__(self):
        self.backend = DummyBackend()
        self.last_tools = ""


class DecisionClient(DummyClient):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.system = ""

    def set_system(self, value):
        self.system = value

    def chat(self, messages, tools=None):
        value = self.outputs.pop(0)
        response = SimpleNamespace(content=value, thinking="raw", tool_calls=[])

        def generate():
            if False:
                yield ""
            return response

        return generate()


def test_live_trajectory_is_always_valid_json(tmp_path):
    recorder = EventRecorder(tmp_path, "model")
    recorder.emit("user_prompt", content="task")
    recorder.begin_assistant(1)
    failures = []

    def reader():
        for _ in range(300):
            try:
                json.loads((tmp_path / "trajectory.json").read_text())
            except Exception as error:
                failures.append(error)

    thread = threading.Thread(target=reader)
    thread.start()
    for _ in range(200):
        recorder.stream_delta("reasoning", "r")
        recorder.stream_delta("content", "c")
    thread.join()
    recorder.finish_assistant(
        turn=1,
        reasoning="r" * 200,
        content="c" * 200,
        tool_calls=[],
        tool_results=[],
    )
    live = json.loads((tmp_path / "trajectory.json").read_text())
    recorder.finalize("completed")
    final = json.loads((tmp_path / "trajectory.json").read_text())
    assert not failures
    assert final["steps"] == live["steps"]
    assert "final_metrics" in final
    assert final["extra"]["live"] is False


def test_worker_and_supervisor_memory_are_isolated_and_merge(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "global_mem.txt").write_text("base")
    state = tmp_path / "state"
    worker = MemoryWorkspace(baseline, state, "worker")
    supervisor = MemoryWorkspace(baseline, state, "supervisor")
    worker_root, supervisor_root = worker.prepare(), supervisor.prepare()
    (worker_root / "global_mem.txt").write_text("worker")
    (worker_root / "new.md").write_text("new")
    (worker_root / "leak.md").write_text("api_key=abcdefghijklmnop")
    assert (supervisor_root / "global_mem.txt").read_text() == "base"
    delta = worker.export_delta(tmp_path / "delta", "abc")
    manifest = json.loads((delta / "manifest.json").read_text())
    assert {entry["path"] for entry in manifest["files"]} == {
        "global_mem.txt",
        "new.md",
    }
    assert manifest["rejected"] == [
        {"path": "leak.md", "reason": "sensitive_content"}
    ]
    store = tmp_path / "store" / "worker"
    merge_delta(baseline, store, delta, "abc", "worker")
    assert (store / "overlay" / "global_mem.txt").read_text() == "worker"


def test_sensitive_filenames_never_enter_overlay(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    state = tmp_path / "state"
    memory = MemoryWorkspace(baseline, state, "worker")
    root = memory.prepare()
    (root / "api_key.md").write_text("not actually a key")
    (root / ".env").write_text("TOKEN=secret")
    delta = memory.export_delta(tmp_path / "delta", "abc")
    manifest = json.loads((delta / "manifest.json").read_text())
    assert manifest["files"] == []
    assert not any((delta / "overlay").rglob("*"))


def test_ask_user_and_memory_write_are_independent(tmp_path):
    agent = GenericAgent(
        client=DummyClient(),
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        memory_dir=tmp_path,
        disable_ask_user=True,
        disable_memory_write=False,
        harness_mode=True,
    )
    names = {tool["function"]["name"] for tool in agent.tools_schema}
    assert "ask_user" not in names
    assert "start_long_term_update" in names


def test_model_config_preserves_thinking_without_sampling_defaults(monkeypatch):
    monkeypatch.setenv("GA_API_BASE", "http://example/v1")
    monkeypatch.setenv("GA_API_KEY", "test")
    monkeypatch.setenv("GA_MODEL", "model")
    for name in (
        "GA_REASONING_EFFORT",
        "GA_MAX_OUTPUT_TOKENS",
        "GA_SERVICE_TIER",
    ):
        monkeypatch.delenv(name, raising=False)
    config = env_model_config("worker")
    assert "temperature" not in config
    assert "reasoning_effort" not in config
    assert "max_tokens" not in config
    assert config["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }


def test_supervisor_cannot_escape_or_read_verifier(tmp_path):
    workspace = tmp_path / "task"
    workspace.mkdir()
    (workspace / "visible.txt").write_text("ok")
    (workspace / "verifier").mkdir()
    (workspace / "verifier" / "secret.txt").write_text("answer")
    recorder = EventRecorder(tmp_path / "logs", "model")
    tools = ReadOnlyWorkspace(workspace, recorder)
    assert tools.call("read_file", {"path": "visible.txt"}) == "ok"
    assert "escapes" in tools.call("read_file", {"path": "../outside"})
    assert "unavailable" in tools.call(
        "read_file", {"path": "verifier/secret.txt"}
    )
    assert "verifier" not in tools.list_directory({"path": "."})


def test_invalid_supervisor_json_is_retried_and_audited(tmp_path):
    workspace = tmp_path / "task"
    memory = tmp_path / "memory"
    workspace.mkdir()
    memory.mkdir()
    recorder = EventRecorder(tmp_path / "logs", "model")
    client = DecisionClient(
        ["not json", '{"action":"continue","reason":"ok","correction":""}']
    )
    supervisor = ProgressiveSupervisor(
        "task", workspace, memory, recorder, client=client, decision_attempts=2
    )
    decision = supervisor.evaluate("step_interval", "step-10", "state")
    assert decision.action == "continue"
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervision.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records].count("invalid_decision") == 1
    assert [record["kind"] for record in records].count("model_response") == 2
