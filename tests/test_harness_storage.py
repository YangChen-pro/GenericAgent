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
from ga_harness.model import build_client, env_model_config
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
        response = (
            value
            if not isinstance(value, str)
            else SimpleNamespace(content=value, thinking="raw", tool_calls=[])
        )

        def generate():
            if False:
                yield ""
            return response

        return generate()


def test_live_trajectory_is_always_valid_json(tmp_path):
    recorder = EventRecorder(tmp_path, "model")
    recorder.emit("user_prompt", content="task")
    failures = []
    stopped = threading.Event()

    def reader():
        while not stopped.is_set():
            try:
                json.loads((tmp_path / "trajectory.json").read_text())
            except Exception as error:
                failures.append(error)

    thread = threading.Thread(target=reader)
    thread.start()
    for turn in range(1, 101):
        recorder.begin_assistant(turn)
        recorder.finish_assistant(
            turn=turn,
            reasoning="r",
            content="c",
            tool_calls=[],
            tool_results=[],
            metrics={"prompt_tokens": 2, "completion_tokens": 1},
        )
    stopped.set()
    thread.join()
    live = json.loads((tmp_path / "trajectory.json").read_text())
    recorder.finalize("completed")
    final = json.loads((tmp_path / "trajectory.json").read_text())
    assert not failures
    assert final["steps"] == live["steps"]
    assert final["final_metrics"] == {
        "total_prompt_tokens": 200,
        "total_completion_tokens": 100,
        "total_cached_tokens": 0,
        "total_steps": 101,
    }
    assert final["extra"]["live"] is False


def test_assistant_draft_is_not_published_until_step_finishes(tmp_path):
    recorder = EventRecorder(tmp_path, "model")
    recorder.emit("user_prompt", content="task")
    before = json.loads((tmp_path / "trajectory.json").read_text())

    recorder.begin_assistant(1)

    during = json.loads((tmp_path / "trajectory.json").read_text())
    assert during == before


def test_reasoning_and_tool_only_steps_are_explicit_and_keep_start_time(tmp_path):
    recorder = EventRecorder(tmp_path, "model")
    recorder.begin_assistant(1)
    recorder.finish_assistant(
        turn=1,
        reasoning="private reasoning",
        content="",
        tool_calls=[],
        tool_results=[],
    )
    recorder.begin_assistant(2)
    recorder.finish_assistant(
        turn=2,
        reasoning="",
        content="",
        tool_calls=[{"id": "call-1", "name": "read_file", "arguments": {"path": "x"}}],
        tool_results=[{"tool_use_id": "call-1", "content": "result"}],
    )

    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    reasoning_step, tool_step = trajectory["steps"]
    assert reasoning_step["timestamp"] == reasoning_step["extra"]["started_at"]
    assert reasoning_step["extra"]["finished_at"] >= reasoning_step["timestamp"]
    assert isinstance(reasoning_step["extra"]["duration_ms"], int)
    assert reasoning_step["message"] == ""
    assert reasoning_step["reasoning_content"] == "private reasoning"
    assert tool_step["message"] == ""
    assert tool_step["tool_calls"][0]["function_name"] == "read_file"
    assert tool_step["observation"]["results"][0]["content"] == "result"

def test_supervisor_client_does_not_inherit_worker_summary_protocol(monkeypatch):
    backends = []

    def make_backend(config):
        backend = DummyBackend()
        backends.append(backend)
        return backend

    monkeypatch.setenv("GA_API_BASE", "http://example/v1")
    monkeypatch.setenv("GA_API_KEY", "test")
    monkeypatch.setenv("GA_MODEL", "model")
    monkeypatch.setattr("ga_harness.model.NativeOAISession", make_backend)

    worker_client = build_client("worker")
    client = build_client("supervisor")
    client.set_system("Return one JSON object.")

    assert "<summary>" in worker_client.backend.system
    assert client.backend.system == "Return one JSON object."
    assert "<summary>" not in client.backend.system


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


def test_harness_mode_disables_browser_tools_and_legacy_model_logs(tmp_path):
    state_dir = tmp_path / "state"
    agent = GenericAgent(
        client=DummyClient(),
        workspace=tmp_path,
        state_dir=state_dir,
        memory_dir=tmp_path,
        harness_mode=True,
    )
    names = {tool["function"]["name"] for tool in agent.tools_schema}
    assert "web_scan" not in names
    assert "web_execute_js" not in names
    assert agent.log_path is False
    assert not (state_dir / "model_responses").exists()


def test_interactive_mode_keeps_legacy_model_logs(tmp_path):
    state_dir = tmp_path / "state"
    agent = GenericAgent(
        client=DummyClient(),
        workspace=tmp_path,
        state_dir=state_dir,
        memory_dir=tmp_path,
    )
    assert Path(agent.log_path).parent == state_dir / "model_responses"
    assert (state_dir / "model_responses").is_dir()


def test_harness_browser_tools_can_be_explicitly_enabled(tmp_path):
    agent = GenericAgent(
        client=DummyClient(),
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        memory_dir=tmp_path,
        disable_browser_tools=False,
        harness_mode=True,
    )
    names = {tool["function"]["name"] for tool in agent.tools_schema}
    assert {"web_scan", "web_execute_js"} <= names


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
        [
            '<summary>not allowed</summary>\n{"action":"continue","reason":"bad","correction":"","level":null}',
            '{"action":"continue","reason":"ok","correction":"","level":null,"target":"step-10"}',
        ]
    )
    supervisor = ProgressiveSupervisor(
        "task", workspace, memory, recorder, client=client, decision_attempts=2
    )
    assert client.log_path is False
    decision = supervisor.evaluate("step_interval", "step-10", "state")
    assert decision.action == "continue"
    assert decision.level is None
    assert "<summary>" not in client.system
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervision.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records].count("invalid_decision") == 1
    assert [record["kind"] for record in records].count("model_response") == 2
    assert not (tmp_path / "logs" / "supervisor-model-responses.txt").exists()


def test_supervisor_audits_input_snapshot_and_read_only_tool_results(tmp_path):
    workspace = tmp_path / "task"
    memory = tmp_path / "memory"
    workspace.mkdir()
    memory.mkdir()
    (workspace / "visible.txt").write_text("observable evidence")
    recorder = EventRecorder(tmp_path / "logs", "model")
    tool_response = SimpleNamespace(
        content="",
        thinking="Need direct evidence.",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "cached_tokens": 3},
        tool_calls=[
            SimpleNamespace(
                id="read-1",
                function=SimpleNamespace(
                    name="read_file",
                    arguments='{"path":"visible.txt"}',
                ),
            )
        ],
    )
    client = DecisionClient(
        [
            tool_response,
            '{"action":"continue","reason":"evidence is sufficient","correction":"","level":null,"target":"step-10"}',
        ]
    )
    supervisor = ProgressiveSupervisor(
        "task instruction", workspace, memory, recorder, client=client
    )

    decision = supervisor.evaluate("step_interval", "step-10", "worker state")

    assert decision.level is None
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "supervision.jsonl").read_text().splitlines()
    ]
    snapshot = next(record for record in records if record["kind"] == "snapshot")
    assert snapshot["payload"]["original_instruction"] == "task instruction"
    assert snapshot["payload"]["current_state"] == "worker state"
    tool_result = next(record for record in records if record["kind"] == "tool_result")
    assert tool_result["payload"] == {
        "name": "read_file",
        "arguments": {"path": "visible.txt"},
        "result": "observable evidence",
    }
    model_events = [record for record in records if record["kind"] == "model_response"]
    assert model_events[0]["payload"]["metrics"] == {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "cached_tokens": 3,
    }
    assert supervisor.token_metrics() == {
        "total_prompt_tokens": 10,
        "total_completion_tokens": 4,
        "total_cached_tokens": 3,
    }


def test_harness_put_task_does_not_allocate_unconsumed_display_queue(tmp_path):
    agent = GenericAgent(
        client=DummyClient(),
        workspace=tmp_path,
        state_dir=tmp_path / "state",
        memory_dir=tmp_path,
        harness_mode=True,
    )
    output = agent.put_task("task")
    queued = agent.task_queue.get_nowait()
    assert queued["output"] is output
    assert output.__class__.__name__ == "_DiscardDisplayQueue"


def test_stream_capture_uses_chunks_and_is_cleared_after_response(monkeypatch):
    from llmcore import NativeOAISession, NativeToolClient

    session = NativeOAISession(
        {
            "name": "test",
            "apikey": "key",
            "apibase": "http://example/v1",
            "model": "model",
            "stream": True,
        }
    )
    session.retain_raw_response = False
    session.history = []

    def raw_ask(messages):
        session.capture_stream("reasoning", "r")
        session.capture_stream("content", "c")
        if False:
            yield ""
        return [
            {"type": "thinking", "thinking": "r"},
            {"type": "text", "text": "c"},
        ]

    monkeypatch.setattr(session, "raw_ask", raw_ask)
    client = NativeToolClient(session)
    generator = client.chat([{"role": "user", "content": "task"}])
    with pytest.raises(StopIteration) as stopped:
        while True:
            next(generator)
    response = stopped.value.value
    assert response.thinking == "r"
    assert response.content == "c"
    assert response.raw == ""
    assert session._stream_capture == {"reasoning": [], "content": []}
