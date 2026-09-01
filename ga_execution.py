"""Task execution and queue behavior for GenericAgent."""

from __future__ import annotations

import os
import queue
import re
import time

from agent_loop import agent_runner_loop
from ga import GenericAgentHandler, consume_file, format_error, smart_format
from ga_config import get_system_prompt


class TaskExecutionMixin:
    def put_task(self, query, source="user", images=None):
        display_queue = queue.Queue()
        self.task_queue.put(
            {
                "query": query,
                "source": source,
                "images": images or [],
                "output": display_queue,
            }
        )
        return display_queue

    def _prepare_query(self, raw_query):
        if len(raw_query) <= 2000:
            return raw_query
        path = os.path.join(
            self.state_dir, f"user_prompt_{os.getpid()}_{time.time_ns()}.md"
        )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(raw_query)
        return f"Long user prompt saved to {path}. Read and execute."

    def _new_handler(self):
        handler = GenericAgentHandler(
            self, self.history, self.workspace, memory_dir=self.memory_dir
        )
        if getattr(self, "no_print", False):
            handler.print = lambda *args, **kwargs: None
        if self.handler and "key_info" in self.handler.working:
            self._carry_working_memory(handler)
        return handler

    def _carry_working_memory(self, handler):
        key_info = re.sub(
            r"\n\[SYSTEM\] 此为.*?工作记忆[。\n]*",
            "",
            self.handler.working["key_info"],
        )
        passed = self.handler.working.get("passed_sessions", 0) + 1
        handler.working.update(key_info=key_info, passed_sessions=passed)
        handler.working["key_info"] += (
            f"\n[SYSTEM] 此为 {passed} 个对话前设置的key_info，若已在新任务，先更新或清除工作记忆。\n"
        )

    def _drain(self, generator, on_chunk):
        try:
            while True:
                on_chunk(next(generator))
                if self.stop_sig:
                    generator.close()
                    return {"result": "ABORTED"}
        except StopIteration as stop:
            return stop.value

    def _build_generator(self, query):
        system_prompt = get_system_prompt(self.memory_dir, self.workspace)
        system_prompt += "\n".join(self.extra_sys_prompts)
        system_prompt += getattr(self.llmclient.backend, "extra_sys_prompt", "")
        if self.peer_hint:
            system_prompt += (
                "\n[Peer] 用户提及其他会话状态时，只读近期 model_responses。\n"
            )
        self.handler = self._new_handler()
        self.llmclient.log_path = self.log_path
        if self.force_non_stream:
            self.llmclient.backend.stream = False
            self.llmclient.backend.read_timeout = max(
                self.llmclient.backend.read_timeout, 1200
            )
        return agent_runner_loop(
            self.llmclient,
            system_prompt,
            query,
            self.handler,
            self.tools_schema,
            max_turns=self.max_turns,
            verbose=self.verbose,
            yield_info=True,
            control=self.control,
        )

    def _chunk_consumer(self, source, display_queue, turn_responses, state):
        def consume(chunk):
            if consume_file(self.task_dir, "_stop"):
                self.abort()
            if isinstance(chunk, dict) and "turn" in chunk:
                state["turn"] = chunk["turn"]
                turn_responses.append("")
                return
            if not isinstance(chunk, str):
                return
            state["full"] += chunk
            if not turn_responses:
                turn_responses.append("")
            turn_responses[-1] += chunk
            if len(state["full"]) - state["position"] <= 30 and "LLM Running" not in chunk:
                return
            display_queue.put(
                {
                    "next": state["full"][state["position"] :]
                    if self.inc_out
                    else state["full"],
                    "source": source,
                    "turn": state["turn"],
                    "outputs": turn_responses[-2:],
                }
            )
            state["position"] = len(state["full"])

        return consume

    def execute_task(self, raw_query, source="user", display_queue=None, raise_errors=True):
        """Execute one task synchronously and return output plus loop result."""
        if self.llmclient is None:
            raise RuntimeError("No model client configured")
        if self.is_running:
            raise RuntimeError("GenericAgent is already running")
        display_queue = display_queue or queue.Queue()
        raw_query = self._handle_slash_cmd(raw_query, display_queue)
        if raw_query is None:
            return {"output": "", "exit_reason": {"result": "COMMAND"}}
        query = self._start_task(raw_query, display_queue)
        turn_responses = self.all_outputs[-1]["outputs"]
        state = {"full": "", "position": 0, "turn": 0}
        consume = self._chunk_consumer(source, display_queue, turn_responses, state)
        try:
            exit_reason = self._drain(self._build_generator(query), consume)
            return self._finish_task(
                display_queue, source, turn_responses, state, exit_reason
            )
        except Exception as error:
            return self._fail_task(
                error, display_queue, source, turn_responses, state, raise_errors
            )
        finally:
            self._reset_after_task()

    def _start_task(self, raw_query, display_queue):
        query = self._prepare_query(raw_query)
        self.is_running = True
        self._current_queue = display_queue
        self.stop_sig = False
        self.clear_action_interrupt()
        self.all_outputs.append({"input": query, "outputs": []})
        self.all_outputs = self.all_outputs[-5000:]
        rendered = smart_format(query.replace("\n", " "), max_str_len=200)
        self.history.append(f"[USER]: {rendered}")
        return query

    def _finish_task(self, display_queue, source, turns, state, exit_reason):
        if self.inc_out and state["position"] < len(state["full"]):
            display_queue.put(
                {
                    "next": state["full"][state["position"] :],
                    "source": source,
                    "turn": state["turn"],
                    "outputs": turns[-2:],
                }
            )
        result = {"output": state["full"], "exit_reason": exit_reason}
        self.last_result = result
        display_queue.put(
            {
                "done": state["full"],
                "source": source,
                "turn": state["turn"],
                "outputs": turns.copy(),
                "exit_reason": exit_reason,
            }
        )
        self.history = self.handler.history_info
        return result

    @staticmethod
    def _error_item(error, source, turns, state):
        rendered = format_error(error)
        return rendered, {
            "done": state["full"] + f"\n```\n{rendered}\n```",
            "source": source,
            "turn": state["turn"],
            "outputs": turns.copy(),
            "error": rendered,
        }

    def _fail_task(self, error, display_queue, source, turns, state, raise_errors):
        rendered, item = self._error_item(error, source, turns, state)
        print(f"Backend Error: {rendered}")
        display_queue.put(item)
        if raise_errors:
            raise error
        return {"output": state["full"], "error": rendered}

    def _reset_after_task(self):
        if self.stop_sig:
            print("User aborted the task.")
        self.is_running = False
        self.stop_sig = False
        self.clear_action_interrupt()
        if self.handler is not None and not self.handler.code_stop_signal:
            self.handler.code_stop_signal.append(1)

    def run(self):
        while True:
            task = self.task_queue.get()
            if isinstance(task, str):
                break
            try:
                self.execute_task(
                    task["query"],
                    source=task["source"],
                    display_queue=task["output"],
                    raise_errors=False,
                )
            finally:
                self.task_queue.task_done()
