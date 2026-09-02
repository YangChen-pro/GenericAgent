# ruff: noqa: E402
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
from ga_harness.curator import consolidate_memory
from ga_harness.memory import MemoryBank, publish_store, resolve_store, tree_digest
from ga_harness.model import build_client, env_model_config
from ga_harness.supervisor import ProgressiveSupervisor
from ga_harness.supervisor_tools import READ_ONLY_TOOLS, ReadOnlyWorkspace
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


def _source_root(tmp_path):
    source = tmp_path / "source"
    worker = source / "memory"
    supervisor = source / "ga_harness" / "supervisor_memory"
    for root, insight in ((worker, "worker-index"), (supervisor, "supervisor-index")):
        (root / "L4_raw_sessions").mkdir(parents=True)
        (root / "global_mem_insight.txt").write_text(insight)
        (root / "global_mem.txt").write_text("# L2\n")
    (worker / "memory_management_sop.md").write_text("shared-l0")
    (worker / "worker_sop.md").write_text("worker-only")
    (supervisor / "supervisor_sop.md").write_text("supervisor-only")
    return source


def test_role_initial_banks_are_isolated_and_l0_is_shared(tmp_path):
    source = _source_root(tmp_path)
    worker = MemoryBank(source, tmp_path / "state", "worker", "initial")
    supervisor = MemoryBank(source, tmp_path / "state", "supervisor", "initial")
    worker_root, supervisor_root = worker.prepare(), supervisor.prepare()

    assert (worker_root / "worker_sop.md").read_text() == "worker-only"
    assert not (worker_root / "supervisor_sop.md").exists()
    assert (supervisor_root / "supervisor_sop.md").read_text() == "supervisor-only"
    assert not (supervisor_root / "worker_sop.md").exists()
    assert (worker_root / "memory_management_sop.md").read_text() == "shared-l0"
    assert (supervisor_root / "memory_management_sop.md").read_text() == "shared-l0"


def test_initial_and_latest_sources_are_independent(tmp_path):
    source = _source_root(tmp_path)
    seed = MemoryBank(source, tmp_path / "seed", "worker", "initial")
    root = seed.prepare()
    (root / "learned.md").write_text("latest")
    candidate = seed.export_delta(tmp_path / "candidate", "abc")
    publish_store(candidate, tmp_path / "store" / "worker")

    initial = MemoryBank(
        source, tmp_path / "initial-state", "worker", "initial", tmp_path / "store" / "worker"
    ).prepare()
    latest = MemoryBank(
        source, tmp_path / "latest-state", "worker", "latest", tmp_path / "store" / "worker"
    ).prepare()

    assert not (initial / "learned.md").exists()
    assert (latest / "learned.md").read_text() == "latest"


def test_sensitive_files_and_l0_changes_never_enter_candidate(tmp_path):
    source = _source_root(tmp_path)
    bank = MemoryBank(source, tmp_path / "state", "worker", "initial")
    root = bank.prepare()
    (root / "api_key.md").write_text("not actually a key")
    (root / ".env").write_text("TOKEN=secret")
    (root / "memory_management_sop.md").write_text("modified")

    candidate = bank.export_delta(tmp_path / "candidate", "abc")
    manifest = json.loads((candidate / "manifest.json").read_text())

    assert manifest["files"] == []
    assert {item["path"] for item in manifest["rejected"]} == {
        ".env",
        "api_key.md",
        "memory_management_sop.md",
    }
    assert not any((candidate / "overlay").iterdir())


def test_curator_semantically_combines_same_file_candidates(tmp_path):
    source = _source_root(tmp_path)
    candidates = []
    for index, lesson in enumerate(("first", "second"), 1):
        bank = MemoryBank(source, tmp_path / f"state-{index}", "supervisor", "initial")
        root = bank.prepare()
        (root / "supervisor_sop.md").write_text(lesson)
        candidates.append(bank.export_delta(tmp_path / f"candidate-{index}", "abc"))

    def curator(workspace, prompt):
        assert "candidate-001" in prompt or (workspace / "evidence" / "candidate-001").is_dir()
        memory = workspace / "memory" / "supervisor_sop.md"
        values = [
            (workspace / "evidence" / f"candidate-{i:03d}" / "supervisor_sop.md").read_text()
            for i in (1, 2)
        ]
        memory.write_text("\n".join(values))
        return {"done": True}

    output = tmp_path / "output"
    report = consolidate_memory(
        "supervisor",
        None,
        None,
        candidates,
        output,
        base_commit="abc",
        source_root=source,
        curator=curator,
    )
    latest = MemoryBank(source, tmp_path / "latest", "supervisor", "latest", output).prepare()
    assert (latest / "supervisor_sop.md").read_text() == "first\nsecond"
    assert report["candidate_count"] == 2


def test_publish_store_switches_valid_current_generation(tmp_path):
    source = _source_root(tmp_path)
    bank = MemoryBank(source, tmp_path / "state", "worker", "initial")
    root = bank.prepare()
    (root / "learned.md").write_text("v1")
    candidate = bank.export_delta(tmp_path / "candidate", "abc")
    generation = publish_store(candidate, tmp_path / "store" / "worker")
    assert resolve_store(tmp_path / "store" / "worker") == (
        tmp_path / "store" / "worker" / "versions" / generation
    ).resolve()
    assert tree_digest(resolve_store(tmp_path / "store" / "worker")) == generation


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
    memory = tmp_path / "supervisor-memory"
    memory.mkdir()
    (memory / "global_mem_insight.txt").write_text("L1")
    (memory / "global_mem.txt").write_text("L2")
    (memory / "supervisor_sop.md").write_text("L3")
    (memory / "L4_raw_sessions").mkdir()
    recorder = EventRecorder(tmp_path / "logs", "model")
    tools = ReadOnlyWorkspace(workspace, recorder, memory)
    assert tools.call("read_file", {"path": "visible.txt"}) == "ok"
    assert "parent traversal" in tools.call("read_file", {"path": "../outside"})
    assert "unavailable" in tools.call(
        "read_file", {"path": "verifier/secret.txt"}
    )
    assert "verifier" not in tools.list_directory({"path": "."})
    assert tools.call("memory_read", {"path": "global_mem.txt"}) == "L2"
    assert tools.call("memory_read", {"path": "supervisor_sop.md"}) == "L3"
    assert "Only this Supervisor" in tools.call(
        "memory_read", {"path": "global_mem_insight.txt"}
    )
    assert "Only this Supervisor" in tools.call(
        "memory_read", {"path": "L4_raw_sessions/hidden.md"}
    )
    assert "Absolute paths" in tools.call("memory_read", {"path": "/tmp/x"})
    assert "Absolute paths" in tools.call("memory_read", {"path": "../worker/x"})


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
    assert "supervisor_sop.md" not in client.system
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


def test_standalone_memory_defaults_are_latest_and_independently_promoted():
    from ga_harness.runner import build_parser

    args = build_parser().parse_args(
        [
            "--instruction",
            "task",
            "--workspace",
            ".",
            "--state-dir",
            "state",
            "--logs-dir",
            "logs",
        ]
    )
    assert args.worker_memory_source == "latest"
    assert args.supervisor_memory_source == "latest"
    assert args.promote_worker_memory is True
    assert args.promote_supervisor_memory is True


@pytest.mark.parametrize(
    ("worker_flag", "supervisor_flag", "worker_expected", "supervisor_expected"),
    [
        (False, False, True, True),
        (True, False, False, True),
        (False, True, True, False),
        (True, True, False, False),
    ],
)
def test_standalone_memory_promotion_switches_are_independent(
    worker_flag, supervisor_flag, worker_expected, supervisor_expected
):
    from ga_harness.runner import build_parser

    argv = [
        "--instruction",
        "task",
        "--workspace",
        ".",
        "--state-dir",
        "state",
        "--logs-dir",
        "logs",
    ]
    if worker_flag:
        argv.append("--no-promote-worker-memory")
    if supervisor_flag:
        argv.append("--no-promote-supervisor-memory")
    args = build_parser().parse_args(argv)
    assert args.promote_worker_memory is worker_expected
    assert args.promote_supervisor_memory is supervisor_expected


def test_initial_run_promotion_rebases_on_current_latest(tmp_path):
    from ga_harness.promotion import promote_memory

    source = _source_root(tmp_path)
    latest_bank = MemoryBank(source, tmp_path / "latest-seed", "worker", "initial")
    latest_root = latest_bank.prepare()
    (latest_root / "existing.md").write_text("keep-existing")
    latest_candidate = latest_bank.export_delta(tmp_path / "latest-candidate", "abc")
    store = tmp_path / "store" / "worker"
    publish_store(latest_candidate, store)

    trial_bank = MemoryBank(source, tmp_path / "trial", "worker", "initial")
    trial_root = trial_bank.prepare()
    (trial_root / "new.md").write_text("new-candidate")
    trial_candidate = trial_bank.export_delta(tmp_path / "trial-candidate", "abc")

    def curator(workspace, prompt):
        memory = workspace / "memory"
        assert (memory / "existing.md").read_text() == "keep-existing"
        candidate = workspace / "evidence" / "candidate-001" / "new.md"
        (memory / "new.md").write_text(candidate.read_text())
        return {"done": True}

    first = promote_memory(
        "worker",
        store,
        None,
        [trial_candidate],
        base_commit="abc",
        promotion_id="job-1",
        source_root=source,
        curator=curator,
    )
    materialized = MemoryBank(
        source, tmp_path / "result", "worker", "latest", store
    ).prepare()
    assert (materialized / "existing.md").read_text() == "keep-existing"
    assert (materialized / "new.md").read_text() == "new-candidate"

    second = promote_memory(
        "worker",
        store,
        None,
        [trial_candidate],
        base_commit="abc",
        promotion_id="job-1",
        source_root=source,
        curator=lambda *_: pytest.fail("idempotent promotion reran curator"),
    )
    assert first["output_sha256"] == second["output_sha256"]
    assert second["status"] == "already_applied"


def test_memory_export_failure_does_not_change_completed_task_status(tmp_path):
    from argparse import Namespace
    from ga_harness.runner import _persist_memories

    class BrokenBank:
        def export_delta(self, *args, **kwargs):
            raise ValueError("invalid candidate")

    args = Namespace(
        worker_memory_source="initial",
        supervisor_memory_source="initial",
        promote_worker_memory=False,
        promote_supervisor_memory=False,
    )
    context = {
        "logs_dir": tmp_path / "logs",
        "banks": {"worker": BrokenBank(), "supervisor": BrokenBank()},
        "store": tmp_path / "store",
    }
    summary = _persist_memories(args, context, "abc", allow_promote=True)
    assert summary["worker"]["error"] == "ValueError: invalid candidate"
    assert summary["supervisor"]["error"] == "ValueError: invalid candidate"



def test_supervisor_injects_only_l1_and_exposes_memory_tools(tmp_path):
    workspace = tmp_path / "task"
    memory = tmp_path / "memory"
    workspace.mkdir()
    memory.mkdir()
    (memory / "global_mem_insight.txt").write_text("INDEX_ONLY")
    (memory / "global_mem.txt").write_text("DEEP_L2_SECRET")
    (memory / "supervisor_sop.md").write_text("DEEP_L3_SECRET")
    recorder = EventRecorder(tmp_path / "logs", "model")
    client = DecisionClient([])

    ProgressiveSupervisor("task", workspace, memory, recorder, client=client)

    assert "INDEX_ONLY" in client.system
    assert "DEEP_L2_SECRET" not in client.system
    assert "DEEP_L3_SECRET" not in client.system
    assert {"memory_list", "memory_read"} <= {
        tool["function"]["name"] for tool in READ_ONLY_TOOLS
    }


def _promotion_fixture(tmp_path):
    source = _source_root(tmp_path)
    bank = MemoryBank(source, tmp_path / "promotion-state", "worker", "initial")
    root = bank.prepare()
    (root / "learned.md").write_text("durable lesson")
    candidate = bank.export_delta(tmp_path / "promotion-candidate", "abc")

    def curator(workspace, prompt):
        memory = workspace / "memory"
        learned = workspace / "evidence" / "candidate-001" / "learned.md"
        (memory / "learned.md").write_text(learned.read_text())
        return {"done": True}

    return source, candidate, curator


def test_sensitive_candidate_files_are_rejected_without_polluting_overlay(tmp_path):
    source = _source_root(tmp_path)
    bank = MemoryBank(source, tmp_path / "state", "worker", "initial")
    root = bank.prepare()
    (root / "notes.md").write_text("API_KEY=abcdefghijklmnopqrstuvwxyz")
    (root / ".env").write_text("TOKEN=secret")

    candidate = bank.export_delta(tmp_path / "candidate", "abc")
    manifest = json.loads((candidate / "manifest.json").read_text())

    assert not (candidate / "overlay" / "notes.md").exists()
    assert not (candidate / "overlay" / ".env").exists()
    assert {item["path"] for item in manifest["rejected"]} == {".env", "notes.md"}


def test_no_promote_keeps_candidate_and_does_not_change_latest(tmp_path):
    from argparse import Namespace
    from ga_harness.runner import _persist_memories

    source = _source_root(tmp_path)
    seed = MemoryBank(source, tmp_path / "seed", "worker", "initial")
    seed_root = seed.prepare()
    (seed_root / "existing.md").write_text("existing")
    store = tmp_path / "store"
    publish_store(seed.export_delta(tmp_path / "seed-delta", "abc"), store / "worker")
    before = tree_digest(resolve_store(store / "worker"))

    trial = MemoryBank(source, tmp_path / "trial", "worker", "initial")
    trial_root = trial.prepare()
    (trial_root / "new.md").write_text("candidate")
    args = Namespace(worker_memory_source="initial", promote_worker_memory=False)
    context = {
        "logs_dir": tmp_path / "logs",
        "banks": {"worker": trial},
        "store": store,
    }

    summary = _persist_memories(args, context, "abc", allow_promote=True)

    assert summary["worker"]["changed"] == 1
    assert Path(summary["worker"]["candidate"]).is_dir()
    assert tree_digest(resolve_store(store / "worker")) == before


def test_curator_failure_leaves_latest_and_no_partial_publication(tmp_path):
    from ga_harness.promotion import promote_memory

    source, candidate, _ = _promotion_fixture(tmp_path)
    store = tmp_path / "store" / "worker"

    def fail(*_):
        raise RuntimeError("curator failed")

    with pytest.raises(RuntimeError, match="curator failed"):
        promote_memory(
            "worker",
            store,
            None,
            [candidate],
            base_commit="abc",
            promotion_id="failed-job",
            source_root=source,
            curator=fail,
        )

    assert resolve_store(store) is None
    assert not list((store / "promotions").glob("*.json"))
    assert not list((store / "promotions" / "staged").glob("*"))


def test_prepared_promotion_resume_does_not_rerun_curator(tmp_path, monkeypatch):
    import ga_harness.promotion as promotion

    source, candidate, curator = _promotion_fixture(tmp_path)
    store = tmp_path / "store" / "worker"
    real_publish = promotion.publish_store
    calls = 0

    def counted_curator(workspace, prompt):
        nonlocal calls
        calls += 1
        return curator(workspace, prompt)

    monkeypatch.setattr(
        promotion,
        "publish_store",
        lambda *_: (_ for _ in ()).throw(RuntimeError("crash before publish")),
    )
    with pytest.raises(RuntimeError, match="crash before publish"):
        promotion.promote_memory(
            "worker",
            store,
            None,
            [candidate],
            base_commit="abc",
            promotion_id="prepared-job",
            source_root=source,
            curator=counted_curator,
        )
    monkeypatch.setattr(promotion, "publish_store", real_publish)

    result = promotion.promote_memory(
        "worker",
        store,
        None,
        [candidate],
        base_commit="abc",
        promotion_id="prepared-job",
        source_root=source,
        curator=lambda *_: pytest.fail("prepared recovery reran curator"),
    )

    assert calls == 1
    assert result["status"] == "recovered_prepared"
    assert tree_digest(resolve_store(store)) == result["output_sha256"]


def test_published_promotion_resume_matches_output_hash(tmp_path, monkeypatch):
    import ga_harness.promotion as promotion

    source, candidate, curator = _promotion_fixture(tmp_path)
    store = tmp_path / "store" / "worker"
    real_atomic = promotion.atomic_write_json

    def crash_before_receipt(path, payload):
        path = Path(path)
        if path.parent.name == "promotions":
            raise RuntimeError("crash before receipt")
        return real_atomic(path, payload)

    monkeypatch.setattr(promotion, "atomic_write_json", crash_before_receipt)
    with pytest.raises(RuntimeError, match="crash before receipt"):
        promotion.promote_memory(
            "worker",
            store,
            None,
            [candidate],
            base_commit="abc",
            promotion_id="published-job",
            source_root=source,
            curator=curator,
        )
    published_hash = tree_digest(resolve_store(store))
    monkeypatch.setattr(promotion, "atomic_write_json", real_atomic)

    result = promotion.promote_memory(
        "worker",
        store,
        None,
        [candidate],
        base_commit="abc",
        promotion_id="published-job",
        source_root=source,
        curator=lambda *_: pytest.fail("published recovery reran curator"),
    )

    assert result["status"] == "recovered_published"
    assert result["output_sha256"] == published_hash


def test_atomic_write_json_serializes_model_response_and_leaves_no_temp(tmp_path):
    from ga_harness.events import atomic_write_json, json_safe

    class Response:
        thinking = "reasoning"
        content = "answer"
        tool_calls = False

    target = tmp_path / "ga-summary.json"
    atomic_write_json(target, json_safe({"result": {"data": Response()}}))

    payload = json.loads(target.read_text())
    assert payload["result"]["data"] == {
        "thinking": "reasoning",
        "content": "answer",
    }
    assert not list(tmp_path.glob(".ga-summary.json.*.tmp"))


def test_atomic_write_json_serializes_before_creating_temp_file(tmp_path):
    from ga_harness.events import atomic_write_json

    target = tmp_path / "ga-summary.json"
    cyclic = {}
    cyclic["self"] = cyclic

    with pytest.raises(ValueError, match="Circular reference"):
        atomic_write_json(target, cyclic)

    assert not target.exists()
    assert not list(tmp_path.glob(".ga-summary.json.*.tmp"))
