"""Model-session and interruption behavior for GenericAgent."""

from __future__ import annotations

import json
import os
import re

from ga import smart_format
from llmcore import (
    MixinSession,
    NativeClaudeSession,
    NativeOAISession,
    NativeToolClient,
    ToolClient,
    reload_mykeys,
    resolve_client,
)


class SessionMixin:
    def _sessions(self):
        if self.llmclient is None:
            return []
        backend = self.llmclient.backend
        return list(getattr(backend, "_sessions", [backend]))

    def _configure_session_hooks(self):
        for session in self._sessions():
            session.should_stop = (
                lambda: self.stop_sig or self._action_interrupt.is_set()
            )
            session.stream_observer = self.report_model_progress

    def attach_control(self, control):
        self.control = control
        self.harness_mode = control is not None
        if self.harness_mode:
            self.log_path = False
            if self.llmclient is not None:
                self.llmclient.log_path = False
        self._configure_session_hooks()

    def load_llm_sessions(self):
        if self._injected_client:
            return
        mykeys, changed = reload_mykeys()
        if not changed and self.llmclients:
            return
        oldhistory, oldname = self._old_session_state()
        sessions = self._configured_clients(mykeys)
        self._materialize_mixins(sessions)
        self.llmclients = sessions
        if not sessions:
            return
        names = [
            client.backend.name if not isinstance(client, dict) else f"BADMIXIN_{index}"
            for index, client in enumerate(sessions)
        ]
        if oldname in names:
            self.llm_no = names.index(oldname)
        self.llmclient = sessions[self.llm_no % len(sessions)]
        if oldhistory:
            self.llmclient.backend.history = oldhistory
        self._configure_session_hooks()

    def _old_session_state(self):
        try:
            return self.llmclient.backend.history, self.llmclient.backend.name
        except Exception:
            return None, None

    @staticmethod
    def _configured_clients(mykeys):
        sessions = []
        for name, config in mykeys.items():
            if not any(item in name for item in ("api", "config", "cookie")):
                continue
            try:
                if "mixin" in name:
                    sessions.append({"mixin_cfg": config})
                elif client := resolve_client(name):
                    sessions.append(client)
            except Exception:
                pass
        return sessions

    @staticmethod
    def _materialize_mixins(sessions):
        for index, session in enumerate(sessions):
            if not isinstance(session, dict) or "mixin_cfg" not in session:
                continue
            try:
                mixin = MixinSession(sessions, session["mixin_cfg"])
                native = isinstance(
                    mixin._sessions[0], (NativeClaudeSession, NativeOAISession)
                )
                sessions[index] = NativeToolClient(mixin) if native else ToolClient(mixin)
            except Exception as error:
                print(f"[ERROR] Failed to init MixinSession: {error}")

    def next_llm(self, number=-1):
        self.load_llm_sessions()
        if not self.llmclients:
            return
        self.llm_no = ((self.llm_no + 1) if number < 0 else number) % len(
            self.llmclients
        )
        previous = self.llmclient
        self.llmclient = self.llmclients[self.llm_no]
        try:
            self.llmclient.backend.history = previous.backend.history
        except Exception as error:
            raise RuntimeError("Bad Mixin configuration") from error
        self.llmclient.last_tools = ""
        self._configure_session_hooks()

    def list_llms(self):
        self.load_llm_sessions()
        return [
            (index, self.get_llm_name(client), index == self.llm_no)
            for index, client in enumerate(self.llmclients)
        ]

    def get_llm_name(self, backend=None, model=False):
        backend = self.llmclient if backend is None else backend
        if isinstance(backend, dict):
            return "BADCONFIG_MIXIN"
        if model:
            return backend.backend.model.lower()
        name = type(backend.backend).__name__.replace("Session", "")
        return f"{name}/{backend.backend.name}"

    def get_ctx_multiplier(self):
        return getattr(self.llmclient.backend, "maxlen_multiplier", 1.0)

    def _close_active_responses(self):
        for session in self._sessions():
            self._close_response(session)

    @staticmethod
    def _close_response(session):
        try:
            response = session.active_response
            raw = response.raw
            fp = getattr(getattr(raw, "_fp", None), "fp", None)
            sock = fp.raw._sock if fp else raw.connection.sock
            try:
                import socket

                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock._real_close()
            except AttributeError:
                sock.close()
        except Exception:
            pass
        try:
            session.active_response.close()
        except Exception:
            pass

    def interrupt_current_action(self):
        """Stop only the active model/tool action; keep the task loop alive."""
        self._action_interrupt.set()
        if self.handler is not None and not self.handler.code_stop_signal:
            self.handler.code_stop_signal.append(1)
        self._close_active_responses()

    def clear_action_interrupt(self):
        self._action_interrupt.clear()
        if self.handler is not None:
            self.handler.code_stop_signal.clear()

    def abort(self):
        if not self.is_running:
            return
        print("Abort current task...")
        self.stop_sig = True
        self.interrupt_current_action()

    def report_model_progress(self, kind, text):
        if self.control:
            self.control.progress(kind, text)

    def report_tool_progress(self, text):
        if self.control:
            self.control.progress("tool", text)

    def _handle_slash_cmd(self, raw_query, display_queue):
        if not raw_query.startswith("/"):
            return raw_query
        match = re.match(r"/session\.(\w+)=(.*)", raw_query.strip())
        if match:
            key, value = match.group(1), match.group(2)
            value_file = os.path.join(self.state_dir, value)
            if os.path.isfile(value_file):
                value = open(value_file, encoding="utf-8").read().strip()
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, ValueError):
                pass
            setattr(self.llmclient.backend, key, value)
            display_queue.put(
                {
                    "done": smart_format(
                        f"✅ session.{key} = {value!r}", max_str_len=500
                    ),
                    "source": "system",
                }
            )
            return None
        if raw_query.strip() == "/resume":
            return "帮我看看最近有哪些会话可以恢复。读取近期 model_responses 并总结。"
        return raw_query
