#!/usr/bin/env python3
"""Run vLLM with request-correlated host-side end-to-end tracing.

Only requests carrying ``X-Request-Id: codesign-*`` are traced. The API process
writes one JSON object per completed response to ``CODESIGN_E2E_TRACE_FILE``.
All event timestamps use ``time.monotonic_ns``; vLLM Engine metrics use the
same system monotonic clock and are converted from seconds to nanoseconds.
"""

import contextvars
from functools import wraps
import json
import os
from pathlib import Path
from threading import Lock
import time

from vllm.renderers.base import BaseRenderer
from vllm.renderers.hf import HfRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.core.sched.scheduler import Scheduler


_TRACE_ID = contextvars.ContextVar("codesign_trace_id", default=None)
_TRACES = {}
_TRACE_LOCK = Lock()
_TRACE_PREFIX = "codesign-"


def _normalize_request_id(request_id):
    if not request_id:
        return None
    if request_id.startswith("chatcmpl-"):
        request_id = request_id[len("chatcmpl-") :]
    return request_id if request_id.startswith(_TRACE_PREFIX) else None


def _mark(event, *, request_id=None, timestamp_ns=None, **values):
    trace_id = _normalize_request_id(request_id) or _TRACE_ID.get()
    if trace_id is None:
        return
    if timestamp_ns is None:
        timestamp_ns = time.monotonic_ns()
    with _TRACE_LOCK:
        trace = _TRACES.setdefault(
            trace_id,
            {"request_id": trace_id, "api_pid": os.getpid(), "events_ns": {}},
        )
        events = trace["events_ns"]
        if event.endswith("_start") or event.endswith("_enter"):
            events.setdefault(event, timestamp_ns)
        else:
            events[event] = timestamp_ns
        if values:
            trace.setdefault("values", {}).update(values)


def _record_engine_metrics(output):
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return
    request_id = getattr(output, "request_id", None)
    for name in ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts"):
        value = getattr(metrics, name, 0.0)
        if value and value > 0:
            _mark(
                f"engine_{name.removesuffix('_ts')}",
                request_id=request_id,
                timestamp_ns=round(value * 1_000_000_000),
            )


def _flush(request_id, *, error=None):
    trace_id = _normalize_request_id(request_id)
    if trace_id is None:
        return
    with _TRACE_LOCK:
        trace = _TRACES.pop(trace_id, None)
        if trace is None:
            return
        if error is not None:
            trace["error"] = repr(error)
        payload = json.dumps(trace, separators=(",", ":")) + "\n"
        path = os.environ.get("CODESIGN_E2E_TRACE_FILE")
        if not path:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
        try:
            os.write(fd, payload.encode())
        finally:
            os.close(fd)


def _append_trace_fragment(request_id, events):
    trace_id = _normalize_request_id(request_id)
    path = os.environ.get("CODESIGN_E2E_TRACE_FILE")
    if trace_id is None or not path:
        return
    payload = json.dumps(
        {
            "request_id": trace_id,
            "engine_pid": os.getpid(),
            "events_ns": events,
        },
        separators=(",", ":"),
    ) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    try:
        os.write(fd, payload.encode())
    finally:
        os.close(fd)


class _TraceASGIMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        headers = {
            key.decode("latin1").lower(): value.decode("latin1")
            for key, value in scope.get("headers", ())
        }
        request_id = _normalize_request_id(headers.get("x-request-id"))
        if request_id is None:
            return await self.app(scope, receive, send)

        token = _TRACE_ID.set(request_id)
        _mark("asgi_enter", request_id=request_id)
        body_done = False
        response_started = False
        first_body = False

        async def traced_receive():
            nonlocal body_done
            message = await receive()
            if (
                not body_done
                and message.get("type") == "http.request"
                and not message.get("more_body", False)
            ):
                body_done = True
                _mark("request_body_complete", request_id=request_id)
            return message

        async def traced_send(message):
            nonlocal response_started, first_body
            if message.get("type") == "http.response.start" and not response_started:
                response_started = True
                _mark("response_start", request_id=request_id)
            elif message.get("type") == "http.response.body":
                if message.get("body") and not first_body:
                    first_body = True
                    _mark("response_first_body", request_id=request_id)
                if not message.get("more_body", False):
                    _mark("response_end", request_id=request_id)
            await send(message)

        error = None
        try:
            await self.app(scope, traced_receive, traced_send)
        except BaseException as exc:
            error = exc
            raise
        finally:
            _flush(request_id, error=error)
            _TRACE_ID.reset(token)


def _wrap_async_method(cls, method_name, event_prefix):
    original = getattr(cls, method_name)
    if getattr(original, "_codesign_e2e_traced", False):
        return

    @wraps(original)
    async def wrapped(self, *args, **kwargs):
        values = {}
        if event_prefix == "preprocess":
            model_config = getattr(self, "model_config", None)
            values["renderer_num_workers"] = getattr(
                model_config, "renderer_num_workers", None
            )
        _mark(f"{event_prefix}_enter", **values)
        try:
            return await original(self, *args, **kwargs)
        finally:
            _mark(f"{event_prefix}_exit")

    wrapped._codesign_e2e_traced = True
    setattr(cls, method_name, wrapped)


def _install_renderer_timers():
    _wrap_async_method(OnlineRenderer, "preprocess_chat", "preprocess")
    _wrap_async_method(HfRenderer, "render_messages_async", "render_messages")
    _wrap_async_method(BaseRenderer, "tokenize_prompt_async", "tokenize_await")
    _wrap_async_method(BaseRenderer, "process_for_engine_async", "engine_input")

    original_base_init = BaseRenderer.__init__

    @wraps(original_base_init)
    def traced_base_init(self, *args, **kwargs):
        original_base_init(self, *args, **kwargs)
        original_tokenize = self._tokenize_prompt_async

        async def traced_tokenize(*call_args, **call_kwargs):
            _mark("tokenizer_executor_enter")
            try:
                return await original_tokenize(*call_args, **call_kwargs)
            finally:
                _mark("tokenizer_executor_exit")

        self._tokenize_prompt_async = traced_tokenize

    BaseRenderer.__init__ = traced_base_init

    original_hf_init = HfRenderer.__init__

    @wraps(original_hf_init)
    def traced_hf_init(self, *args, **kwargs):
        original_hf_init(self, *args, **kwargs)
        original_template = self._apply_chat_template_async

        async def traced_template(*call_args, **call_kwargs):
            _mark("template_executor_enter")
            try:
                return await original_template(*call_args, **call_kwargs)
            finally:
                _mark("template_executor_exit")

        self._apply_chat_template_async = traced_template

    HfRenderer.__init__ = traced_hf_init


def _install_engine_timers():
    original_add_request = AsyncLLM.add_request

    @wraps(original_add_request)
    async def traced_add_request(self, request_id, *args, **kwargs):
        _mark("add_request_enter", request_id=request_id)
        try:
            return await original_add_request(self, request_id, *args, **kwargs)
        finally:
            _mark("add_request_exit", request_id=request_id)

    AsyncLLM.add_request = traced_add_request

    original_generate = AsyncLLM.generate

    @wraps(original_generate)
    async def traced_generate(self, prompt, sampling_params, request_id, **kwargs):
        _mark("generate_enter", request_id=request_id)
        first_output = True
        try:
            async for output in original_generate(
                self, prompt, sampling_params, request_id, **kwargs
            ):
                if first_output:
                    first_output = False
                    _mark("engine_first_output_received", request_id=request_id)
                _record_engine_metrics(output)
                if output.finished:
                    _mark("engine_final_output_received", request_id=request_id)
                yield output
        finally:
            _mark("generate_exit", request_id=request_id)

    AsyncLLM.generate = traced_generate

    original_schedule = Scheduler.schedule

    @wraps(original_schedule)
    def traced_schedule(self, *args, **kwargs):
        call_enter_ns = time.monotonic_ns()
        output = original_schedule(self, *args, **kwargs)
        return_ns = time.monotonic_ns()
        for request in output.scheduled_new_reqs:
            _append_trace_fragment(
                request.req_id,
                {
                    "scheduler_call_enter": call_enter_ns,
                    "scheduler_return": return_ns,
                },
            )
        return output

    Scheduler.schedule = traced_schedule


def _install_app_timer():
    import vllm.entrypoints.launchers.app as app_module

    original_build_app = app_module.build_app

    @wraps(original_build_app)
    def traced_build_app(*args, **kwargs):
        app = original_build_app(*args, **kwargs)
        for route in app.routes:
            if getattr(route, "path", None) != "/v1/chat/completions":
                continue
            dependant = getattr(route, "dependant", None)
            if dependant is None:
                continue
            original_endpoint = dependant.call

            @wraps(original_endpoint)
            async def traced_endpoint(*call_args, __original=original_endpoint, **call_kwargs):
                _mark("endpoint_enter")
                try:
                    return await __original(*call_args, **call_kwargs)
                finally:
                    _mark("endpoint_return")

            dependant.call = traced_endpoint
        # vLLM's build_app eagerly constructs the middleware stack, so
        # Starlette rejects add_middleware() here. Wrap the already-built ASGI
        # stack directly while keeping the FastAPI object (and app.state)
        # visible to the launcher.
        if app.middleware_stack is None:
            app.build_middleware_stack()
        app.middleware_stack = _TraceASGIMiddleware(app.middleware_stack)
        return app

    app_module.build_app = traced_build_app


_install_renderer_timers()
_install_engine_timers()
_install_app_timer()


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
