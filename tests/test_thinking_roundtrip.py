import json

import llmcore
from llmcore import NativeOAISession


class FakeHTTPResponse:
    status_code = 200
    headers = {}

    def __init__(self, lines):
        self._lines = lines
        self.raw = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_lines(self):
        return iter(self._lines)

    def close(self):
        pass


def drain(generator):
    chunks = []
    try:
        while True:
            chunks.append(next(generator))
    except StopIteration as stop:
        return chunks, stop.value


def response_lines(reasoning="reasoning", content="answer"):
    events = [
        {"choices": [{"delta": {"reasoning_content": reasoning}}]},
        {"choices": [{"delta": {"content": content}}]},
    ]
    return [
        *(f"data: {json.dumps(event)}".encode() for event in events),
        b"data: [DONE]",
    ]


def new_session():
    return NativeOAISession(
        {
            "name": "worker",
            "apikey": "secret",
            "apibase": "http://model/v1",
            "model": "model",
            "stream": True,
            "omit_thinking": False,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        }
    )


def test_thinking_is_returned_to_next_turn_without_sampling_defaults(monkeypatch):
    payloads = []
    def post(url, **kwargs):
        payloads.append(kwargs["json"])
        return FakeHTTPResponse(response_lines())

    monkeypatch.setattr(llmcore.requests, "post", post)
    session = new_session()
    session_id = session._session_id
    drain(session.ask({"role": "user", "content": [{"type": "text", "text": "one"}]}))
    drain(session.ask({"role": "user", "content": [{"type": "text", "text": "two"}]}))

    first = payloads[0]
    assert first["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert "temperature" not in first
    assert "max_tokens" not in first
    assert "reasoning_effort" not in first
    prior = [message for message in payloads[1]["messages"] if message["role"] == "assistant"]
    assert prior[0]["reasoning_content"] == "reasoning"
    assert session._session_id == session_id


def test_interrupted_stream_keeps_partial_reasoning_in_history(monkeypatch):
    monkeypatch.setattr(
        llmcore.requests,
        "post",
        lambda *args, **kwargs: FakeHTTPResponse(response_lines("partial thought", "late")),
    )
    session = new_session()
    stop = {"value": False}
    session.should_stop = lambda: stop["value"]
    generator = session.ask(
        {"role": "user", "content": [{"type": "text", "text": "task"}]}
    )
    assert next(generator) == "partial thought"
    stop["value"] = True
    _, response = drain(generator)
    assert response.thinking == "partial thought"
    assistant = session.history[-1]
    assert assistant["role"] == "assistant"
    assert any(
        block.get("type") == "thinking" and block.get("thinking") == "partial thought"
        for block in assistant["content"]
    )
