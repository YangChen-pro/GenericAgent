"""GenericAgent turn loop with optional harness control hooks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

try:
    from plugins.hooks import trigger as _hook
except ImportError:
    _hook = lambda *args, **kwargs: None


@dataclass
class StepOutcome:
    data: Any
    next_prompt: Optional[str] = None
    should_exit: bool = False


class BaseHandler:
    def turn_end_callback(
        self, response, tool_calls, tool_results, turn, next_prompt, exit_reason
    ):
        return next_prompt

    def dispatch(self, tool_name, args, response, index=0, tool_num=1):
        method_name = f"do_{tool_name}"
        if hasattr(self, method_name):
            args["_index"] = index
            args["_tool_num"] = tool_num
            _hook("tool_before", locals())
            ret = yield from try_call_generator(
                getattr(self, method_name), args, response
            )
            _hook("tool_after", locals())
            return ret
        if tool_name == "bad_json":
            return StepOutcome(None, next_prompt=args.get("msg", "bad_json"))
        yield f"未知工具: {tool_name}\n"
        return StepOutcome(None, next_prompt=f"未知工具 {tool_name}")


def try_call_generator(func, *args, **kwargs):
    ret = func(*args, **kwargs)
    if hasattr(ret, "__iter__") and not isinstance(ret, (str, bytes, dict, list)):
        ret = yield from ret
    return ret


def exhaust(generator):
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return stop.value


def json_default(value):
    return list(value) if isinstance(value, set) else str(value)


def get_pretty_json(data):
    if isinstance(data, dict) and "script" in data:
        data = data.copy()
        data["script"] = data["script"].replace("; ", ";\n  ")
    return json.dumps(data, indent=2, ensure_ascii=False).replace("\\n", "\n")


def _tool_calls(response):
    if not response.tool_calls:
        return [{"tool_name": "no_tool", "args": {}}]
    return [
        {
            "tool_name": call.function.name,
            "args": json.loads(call.function.arguments),
            "id": call.id,
        }
        for call in response.tool_calls
    ]


def _metrics():
    try:
        from llmcore import STATS

        return {
            "prompt_tokens": int(STATS.get("inp") or 0),
            "completion_tokens": int(STATS.get("out") or 0),
            "cached_tokens": int(STATS.get("cached") or 0),
        }
    except Exception:
        return {}


def _clear_action_interrupt(handler):
    clear = getattr(handler.parent, "clear_action_interrupt", None)
    if clear:
        clear()


def _run_model(client, messages, tools_schema, verbose, control, turn):
    action_id = f"turn-{turn}-model"
    if control:
        control.action_started("model", action_id, turn)
    try:
        response_generator = client.chat(messages=messages, tools=tools_schema)
        if verbose:
            response = yield from response_generator
            yield "\n\n"
        else:
            response = exhaust(response_generator)
            cleaned = _clean_content(response.content)
            if cleaned:
                yield cleaned + "\n"
        return response
    finally:
        if control:
            control.action_finished(action_id)


def _run_one_tool(handler, call, response, index, total, verbose, control, turn):
    name, args = call["tool_name"], call["args"]
    action_id = f"turn-{turn}-tool-{index + 1}"
    if control:
        control.action_started("tool", action_id, turn)
    if name != "no_tool":
        if verbose:
            yield f"🛠️ Tool: `{name}`  📥 args:\n````text\n{get_pretty_json(args)}\n````\n"
        else:
            yield f"🛠️ {name}({_compact_tool_args(name, args)})\n"
    handler.current_turn = turn
    generator = handler.dispatch(name, args, response, index=index, tool_num=total)
    try:
        try:
            first = next(generator)
            if control:
                control.progress("tool", str(first))
            if verbose:
                yield "`````\n" + first
            outcome = (yield from generator) if verbose else exhaust(generator)
            if verbose:
                yield "`````\n"
        except StopIteration as stop:
            outcome = stop.value
        return outcome
    finally:
        if control:
            control.action_finished(action_id)


def _append_tool_result(results, call, outcome):
    if outcome.data is None or call["tool_name"] == "no_tool":
        return
    data = (
        json.dumps(outcome.data, ensure_ascii=False, default=json_default)
        if isinstance(outcome.data, (dict, list))
        else str(outcome.data)
    )
    results.append({"tool_use_id": call.get("id", ""), "content": data})


def _record_turn(control, turn, response, calls, results):
    stalled = control.take_stall() if control else None
    if control:
        control.record_turn(turn, response, calls, results, _metrics(), bool(stalled))
    return stalled


def _completion_or_step_review(control, turn, state, completion):
    if not control:
        return ""
    if control.needs_follow_up():
        return control.check_step(turn, state)
    if completion:
        return control.check_completion(state)
    return control.check_step(turn, state)


def _model_stall_correction(control, handler, turn, response, state):
    if not control or not control.has_stall():
        return None
    stalled = _record_turn(control, turn, response, [], [])
    correction = control.review_stall(stalled, state)
    _clear_action_interrupt(handler)
    return correction


def _run_tools(client, handler, response, calls, verbose, control, turn):
    results, prompts, exit_reason = [], set(), {}
    for index, call in enumerate(calls):
        outcome = yield from _run_one_tool(
            handler, call, response, index, len(calls), verbose, control, turn
        )
        _append_tool_result(results, call, outcome)
        if outcome.should_exit:
            exit_reason = {"result": "EXITED", "data": outcome.data}
            break
        if not outcome.next_prompt:
            exit_reason = {"result": "CURRENT_TASK_DONE", "data": outcome.data}
            break
        if outcome.next_prompt.startswith("未知工具"):
            client.last_tools = ""
        prompts.add(outcome.next_prompt)
        if control and control.has_stall():
            break
    return results, prompts, exit_reason


def _apply_done_hook(handler, prompts, exit_reason):
    hooks = getattr(handler, "_done_hooks", None)
    if (exit_reason or not prompts) and hooks and exit_reason.get("result") != "EXITED":
        prompts.add(hooks.pop(0))
        return {}
    return exit_reason


def _review_turn(control, handler, turn, response, calls, results, prompts, exit_reason):
    next_prompt = handler.turn_end_callback(
        response, calls, results, turn, "\n".join(prompts), exit_reason
    )
    _hook("turn_after", locals())
    state = (getattr(response, "content", "") or "")[-20_000:]
    state += "\n" + json.dumps(results, ensure_ascii=False)[-20_000:]
    stalled = _record_turn(control, turn, response, calls, results)
    if stalled:
        correction = control.review_stall(stalled, state)
        _clear_action_interrupt(handler)
    else:
        completion = bool(exit_reason or not prompts)
        correction = _completion_or_step_review(control, turn, state, completion)
    if correction:
        return f"{next_prompt}\n\n{correction}".strip(), {}, False
    return next_prompt, exit_reason, bool(exit_reason or not prompts)


def agent_runner_loop(
    client,
    system_prompt,
    user_input,
    handler,
    tools_schema,
    max_turns=40,
    verbose=True,
    initial_user_content=None,
    yield_info=False,
    control=None,
):
    """Run one task. Session history remains owned by ``client.backend``."""
    first_content = user_input if initial_user_content is None else initial_user_content
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_content},
    ]
    handler.max_turns = max_turns
    _hook("agent_before", locals())
    exit_reason = {}
    for turn in range(1, handler.max_turns + 1):
        if yield_info:
            yield {"turn": turn}
        yield from _turn_banner(handler, turn, verbose)
        if turn % 10 == 0:
            client.last_tools = ""
        _hook("turn_before", locals())
        _hook("llm_before", locals())
        response = yield from _run_model(
            client, messages, tools_schema, verbose, control, turn
        )
        _hook("llm_after", locals())
        state = (getattr(response, "content", "") or "")[-20_000:]
        correction = _model_stall_correction(
            control, handler, turn, response, state
        )
        if correction is not None:
            messages = [{"role": "user", "content": correction}]
            continue
        calls = _tool_calls(response)
        results, prompts, exit_reason = yield from _run_tools(
            client, handler, response, calls, verbose, control, turn
        )
        exit_reason = _apply_done_hook(handler, prompts, exit_reason)
        next_prompt, exit_reason, finished = _review_turn(
            control, handler, turn, response, calls, results, prompts, exit_reason
        )
        if finished:
            break
        messages = [
            {"role": "user", "content": next_prompt, "tool_results": results}
        ]
    _hook("agent_after", locals())
    return exit_reason or {"result": "MAX_TURNS_EXCEEDED"}


def _turn_banner(handler, turn, verbose):
    label = f"Turn {turn} ..." if handler.parent.task_dir else f"LLM Running (Turn {turn}) ..."
    yield f"\n**{label}**\n\n" if verbose else f"\n{label}\n\n"


def _clean_content(text):
    if not text:
        return ""

    def shrink_code(match):
        lines = match.group(0).split("\n")
        language = lines[0].replace("```", "").strip()
        body = [line for line in lines[1:-1] if line.strip()]
        if len(body) <= 6:
            return match.group(0)
        return f"```{language}\n" + "\n".join(body[:5]) + f"\n  ... ({len(body)} lines)\n```"

    text = re.sub(r"```[\s\S]*?```", shrink_code, text)
    for pattern in (
        r"<file_content>[\s\S]*?</file_content>",
        r"<tool_(?:use|call)>[\s\S]*?</tool_(?:use|call)>",
    ):
        text = re.sub(pattern, "", text)
    return re.sub(r"(\r?\n){3,}", "\n\n", text).strip()


def _compact_tool_args(name, args):
    compact = {key: value for key, value in args.items() if key != "_index"}
    if "path" in compact:
        compact["path"] = os.path.basename(compact["path"])
    if name == "update_working_checkpoint":
        text = compact.get("key_info", "")
        return text[:60] + "..." if len(text) > 60 else text
    if name == "ask_user":
        question = str(compact.get("question", ""))
        choices = compact.get("candidates") or []
        if choices:
            question += "\ncandidates:\n" + "\n".join(f"- {item}" for item in choices)
        return question
    text = json.dumps(compact, ensure_ascii=False)
    return text[:120] + "..." if len(text) > 120 else text
