#!/usr/bin/env python3
"""Start vLLM with low-overhead scheduler and frontend phase counters.

The wrapper does not modify the installed vLLM package. Counters are cumulative
uint64 (count, elapsed_ns) pairs stored in two mmap files so the benchmark can
take before/after deltas without adding Prometheus work to the hot path.
"""

from functools import wraps
import mmap
import os
import struct
from threading import Lock
from time import perf_counter_ns

import vllm.renderers.hf as hf_renderer_module
from vllm.renderers.base import BaseRenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.core.sched.scheduler import Scheduler


_FRONTEND_FORMAT = "<8Q"
_SCHEDULER_FORMAT = "<8Q"
_PAGES = {}
_LOCKS = {}


def _counter_page(env_name, counter_format):
    page = _PAGES.get(env_name)
    if page is not None:
        return page
    path = os.environ.get(env_name)
    if not path:
        return None
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    counter_size = struct.calcsize(counter_format)
    os.ftruncate(fd, counter_size)
    page = mmap.mmap(fd, counter_size)
    os.close(fd)
    page[:] = b"\0" * counter_size
    page.flush()
    _PAGES[env_name] = page
    _LOCKS[env_name] = Lock()
    return page


def _record(env_name, counter_format, pair_index, elapsed_ns):
    page = _counter_page(env_name, counter_format)
    if page is None:
        return
    with _LOCKS[env_name]:
        counters = list(struct.unpack(counter_format, page[:]))
        offset = pair_index * 2
        counters[offset] += 1
        counters[offset + 1] += elapsed_ns
        page[:] = struct.pack(counter_format, *counters)


def _install_scheduler_timer():
    original = Scheduler.schedule
    if getattr(original, "_codesign_cpu_timed", False):
        return

    @wraps(original)
    def timed_schedule(self, *args, **kwargs):
        started = perf_counter_ns()
        output = original(self, *args, **kwargs)
        elapsed_ns = perf_counter_ns() - started
        _record(
            "CODESIGN_SCHEDULER_METRICS_FILE",
            _SCHEDULER_FORMAT,
            0,
            elapsed_ns,
        )
        if output.scheduled_new_reqs:
            pair_index = 1  # admission
        elif output.total_num_scheduled_tokens == 0:
            pair_index = 3  # empty
        else:
            pair_index = 2  # steady
        _record(
            "CODESIGN_SCHEDULER_METRICS_FILE",
            _SCHEDULER_FORMAT,
            pair_index,
            elapsed_ns,
        )
        return output

    timed_schedule._codesign_cpu_timed = True
    Scheduler.schedule = timed_schedule


def _install_frontend_timers():
    original_template = hf_renderer_module.safe_apply_chat_template

    @wraps(original_template)
    def timed_template(*args, **kwargs):
        started = perf_counter_ns()
        try:
            return original_template(*args, **kwargs)
        finally:
            _record(
                "CODESIGN_FRONTEND_METRICS_FILE",
                _FRONTEND_FORMAT,
                1,
                perf_counter_ns() - started,
            )

    hf_renderer_module.safe_apply_chat_template = timed_template

    original_tokenize = BaseRenderer._tokenize_prompt

    @wraps(original_tokenize)
    def timed_tokenize(self, *args, **kwargs):
        started = perf_counter_ns()
        try:
            return original_tokenize(self, *args, **kwargs)
        finally:
            _record(
                "CODESIGN_FRONTEND_METRICS_FILE",
                _FRONTEND_FORMAT,
                2,
                perf_counter_ns() - started,
            )

    BaseRenderer._tokenize_prompt = timed_tokenize

    original_preprocess = OnlineRenderer.preprocess_chat

    @wraps(original_preprocess)
    async def timed_preprocess(self, *args, **kwargs):
        started = perf_counter_ns()
        try:
            return await original_preprocess(self, *args, **kwargs)
        finally:
            _record(
                "CODESIGN_FRONTEND_METRICS_FILE",
                _FRONTEND_FORMAT,
                0,
                perf_counter_ns() - started,
            )

    OnlineRenderer.preprocess_chat = timed_preprocess

    original_add_request = AsyncLLM.add_request

    @wraps(original_add_request)
    async def timed_add_request(self, *args, **kwargs):
        started = perf_counter_ns()
        try:
            return await original_add_request(self, *args, **kwargs)
        finally:
            _record(
                "CODESIGN_FRONTEND_METRICS_FILE",
                _FRONTEND_FORMAT,
                3,
                perf_counter_ns() - started,
            )

    AsyncLLM.add_request = timed_add_request


_install_scheduler_timer()
_install_frontend_timers()


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
