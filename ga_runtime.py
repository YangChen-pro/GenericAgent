"""GenericAgent runtime object shared by interactive and benchmark entrypoints."""

from __future__ import annotations

import os
import queue
import random
import threading
import time

from ga_config import DEFAULT_MEMORY_DIR, configure_stdio, load_tool_schema, script_dir
from ga_execution import TaskExecutionMixin
from ga_session import SessionMixin

configure_stdio()

try:
    from plugins.hooks import discover_and_load

    discover_and_load()
except Exception:
    pass


class GenericAgent(SessionMixin, TaskExecutionMixin):
    """GenericAgent SDK plus an optional benchmark-harness control plane."""

    def __init__(
        self,
        client=None,
        workspace=None,
        state_dir=None,
        memory_dir=None,
        control=None,
        disable_ask_user=None,
        disable_memory_write=None,
        harness_mode=False,
        max_turns=180,
    ):
        self.workspace = os.path.abspath(workspace or os.path.join(script_dir, "temp"))
        self.state_dir = os.path.abspath(state_dir or os.path.join(script_dir, "temp"))
        self.memory_dir = os.path.abspath(memory_dir or DEFAULT_MEMORY_DIR)
        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        self.task_dir = self.state_dir if harness_mode else None
        self.harness_mode = harness_mode
        self.control = control
        self.max_turns = max_turns
        self.lock = threading.Lock()
        self.history = []
        self.handler = None
        self.all_outputs = []
        self.task_queue = queue.Queue()
        self.is_running = False
        self.stop_sig = False
        self.llm_no = 0
        self._current_queue = None
        self._action_interrupt = threading.Event()
        self.inc_out = False
        self.verbose = True
        self.peer_hint = not harness_mode
        self.force_non_stream = False
        self.last_result = None
        self.extra_sys_prompts = []
        self.intervene = self.extrakeyinfo = None
        self._configure_tools(disable_ask_user, disable_memory_write)
        self._configure_logging()
        self._configure_client(client)

    def _configure_tools(self, disable_ask_user, disable_memory_write):
        import sys

        disable_ask = (
            "--no-user-tools" in sys.argv
            if disable_ask_user is None
            else disable_ask_user
        )
        disable_memory = (
            "--no-memory-write" in sys.argv
            if disable_memory_write is None
            else disable_memory_write
        )
        banned = {"ask_user"} if disable_ask else set()
        if disable_memory:
            banned.add("start_long_term_update")
        self.tools_schema = load_tool_schema(banned_tools=banned)

    def _configure_logging(self):
        logid = f"{(time.time_ns() + random.randrange(1_000_000)) % 1_000_000:06d}"
        log_dir = os.path.join(self.state_dir, "model_responses")
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"model_responses_{logid}.txt")

    def _configure_client(self, client):
        self._injected_client = client is not None
        self.llmclient = client
        self.llmclients = [client] if client is not None else []
        if client is None:
            self.load_llm_sessions()
        self._configure_session_hooks()


GeneraticAgent = GenericAgent
