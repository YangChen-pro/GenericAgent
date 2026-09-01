import sys

# A legacy frontend test installs collection-time module stubs. Restore the real
# core modules before importing harness code.
for _module_name in ("agent_loop", "llmcore", "agentmain"):
    _module = sys.modules.get(_module_name)
    if _module is not None and not getattr(_module, "__file__", None):
        sys.modules.pop(_module_name, None)

import json
import time
from pathlib import Path
from types import SimpleNamespace

from agent_loop import BaseHandler, StepOutcome, exhaust, agent_runner_loop
from ga_harness.control import HarnessControl
from ga_harness.events import EventRecorder
from ga_harness.supervisor import Decision


class FakeCall:
    def __init__(self, name="work", arguments="{}", call_id="call"):
        self.function = SimpleNamespace(name=name, arguments=arguments)
        self.id = call_id


class FakeResponse:
    def __init__(self, content="working", thinking="reasoning", tool=True):
        self.content = content
        self.thinking = thinking
        self.tool_calls = [FakeCall()] if tool else []


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []
        self.last_tools = ""

    def chat(self, messages, tools=None):
        self.seen.append(messages)
        response = self.responses.pop(0)

        def generate():
            if False:
                yield ""
            return response

        return generate()


class FakeParent:
    task_dir = None

    def __init__(self):
        self.clears = 0

    def clear_action_interrupt(self):
        self.clears += 1


class FakeHandler(BaseHandler):
    def __init__(self):
        self.parent = FakeParent()
        self._done_hooks = []

    def do_work(self, args, response):
        yield "tool feedback"
        return StepOutcome({"ok": True}, next_prompt="continue")

    def do_no_tool(self, args, response):
        return StepOutcome(response, next_prompt=None)


class FakeSupervisor:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls = []
        self.history = []

    def evaluate(self, trigger, target, state):
        self.calls.append((trigger, target, state))
        decision = self.decisions.pop(0) if self.decisions else Decision("continue")
        self.history.append({"trigger": trigger, "action": decision.action})
        return decision


def run_loop(tmp_path, supervisor, turns=11, responses=None):
    recorder = EventRecorder(tmp_path, "model")
    client = FakeClient(responses or [FakeResponse() for _ in range(turns)])
    handler = FakeHandler()
    control = HarnessControl(
        recorder,
        supervisor,
        lambda: None,
        step_interval=10,
        stall_timeout_sec=10,
    )
    try:
        result = exhaust(
            agent_runner_loop(
                client,
                "system",
                "task",
                handler,
                [],
                max_turns=turns,
                verbose=False,
                control=control,
            )
        )
    finally:
        control.close()
    return result, client, recorder


def test_first_check_is_step_ten(tmp_path):
    supervisor = FakeSupervisor()
    run_loop(tmp_path, supervisor, turns=11)
    step_checks = [call for call in supervisor.calls if call[0] == "step_interval"]
    assert [(trigger, target) for trigger, target, _ in step_checks] == [
        ("step_interval", "step-10")
    ]
    assert all(target != "step-1" for _, target, _ in supervisor.calls)


def test_correction_is_user_event_then_immediate_follow_up(tmp_path):
    supervisor = FakeSupervisor(
        [
            Decision(
                "intervene",
                reason="drift",
                correction="Stop looping and inspect the artifact.",
            ),
            Decision("continue"),
        ]
    )
    _, client, recorder = run_loop(tmp_path, supervisor, turns=11)
    assert [call[0] for call in supervisor.calls] == ["step_interval", "follow_up"]
    assert "Stop looping" in client.seen[10][0]["content"]
    kinds = [event["kind"] for event in recorder.events]
    index = kinds.index("supervisor_correction")
    assert kinds[index - 1] == "assistant"
    assert kinds[index + 1] == "action_started"
    corrections = [
        event for event in recorder.events if event["kind"] == "supervisor_correction"
    ]
    assert len(corrections) == 1
    assert corrections[0]["session_id"] == recorder.session_id
    trajectory = json.loads((tmp_path / "trajectory.json").read_text())
    steps = [step for step in trajectory["steps"] if step["source"] == "user"]
    assert steps == [
        {
            "step_id": steps[0]["step_id"],
            "timestamp": steps[0]["timestamp"],
            "source": "user",
            "message": "Stop looping and inspect the artifact.",
            "extra": {
                "kind": "supervisor_correction",
                "intervention_id": corrections[0]["intervention_id"],
                "reason_type": "drift",
                "hint_level": "constraint",
                "origin": "supervisor",
            },
        }
    ]


def test_progress_prevents_stall_until_feedback_stops(tmp_path):
    supervisor = FakeSupervisor([Decision("intervene", correction="recover")])
    interrupted = []
    control = HarnessControl(
        EventRecorder(tmp_path, "model"),
        supervisor,
        lambda: interrupted.append(True),
        stall_timeout_sec=0.12,
    )
    try:
        control.action_started("tool", "tool-1", 1)
        for _ in range(3):
            time.sleep(0.05)
            control.progress("tool", "still running")
            assert not interrupted
        deadline = time.monotonic() + 1
        while not interrupted and time.monotonic() < deadline:
            time.sleep(0.01)
        assert interrupted
        assert control.take_stall() == "tool-1"
        assert control.review_stall("tool-1", "state") == "recover"
    finally:
        control.close()


def test_unsupervised_control_never_interrupts(tmp_path):
    interrupted = []
    control = HarnessControl(
        EventRecorder(tmp_path, "model"),
        None,
        lambda: interrupted.append(True),
        stall_timeout_sec=0.03,
    )
    try:
        control.action_started("tool", "tool-1", 1)
        time.sleep(0.1)
        assert not interrupted
        assert control.check_step(10, "state") == ""
        assert control.check_completion("state") == ""
    finally:
        control.close()


def test_completion_gate_can_reject_then_approve(tmp_path):
    supervisor = FakeSupervisor(
        [
            Decision("intervene", correction="Produce the required artifact."),
            Decision("continue"),
        ]
    )
    responses = [
        FakeResponse(content="done?", tool=False),
        FakeResponse(content="now done", tool=False),
    ]
    result, client, _ = run_loop(tmp_path, supervisor, turns=3, responses=responses)
    assert result["result"] == "CURRENT_TASK_DONE"
    assert [call[0] for call in supervisor.calls] == ["completion", "follow_up"]
    assert "Produce the required artifact" in client.seen[1][0]["content"]


def test_partial_reasoning_precedes_correction(tmp_path):
    supervisor = FakeSupervisor(
        [Decision("intervene", correction="change course"), Decision("continue")]
    )
    responses = [FakeResponse(thinking=f"thought-{turn}") for turn in range(1, 12)]
    _, _, recorder = run_loop(tmp_path, supervisor, turns=11, responses=responses)
    correction_index = next(
        index
        for index, event in enumerate(recorder.events)
        if event["kind"] == "supervisor_correction"
    )
    previous = recorder.events[correction_index - 1]
    assert previous["kind"] == "assistant"
    assert previous["reasoning"] == "thought-10"
