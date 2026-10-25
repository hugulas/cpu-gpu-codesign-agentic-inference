#!/usr/bin/env python3
"""Start vLLM with a Prometheus metric for Scheduler.schedule() wall time.

The wrapper leaves vLLM itself unchanged.  The patch is applied before the
engine subprocess is created, so it also works with vLLM's spawn-based engine.
"""

from functools import wraps
import mmap
import os
import struct
from time import perf_counter

from vllm.v1.core.sched.scheduler import Scheduler


_COUNTER_PAGE = None
_COUNTER_FORMAT = "<8Q"
_COUNTER_SIZE = struct.calcsize(_COUNTER_FORMAT)
_KIND_OFFSET = {"admission": 2, "steady": 4, "empty": 6}


def _counter_page():
    global _COUNTER_PAGE
    if _COUNTER_PAGE is None:
        path = os.environ.get("VLLM_SCHEDULER_METRICS_FILE")
        if not path:
            return None
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, _COUNTER_SIZE)
        _COUNTER_PAGE = mmap.mmap(fd, _COUNTER_SIZE)
        os.close(fd)
        _COUNTER_PAGE[:] = b"\0" * _COUNTER_SIZE
        _COUNTER_PAGE.flush()
    return _COUNTER_PAGE


def _record_counter(step_kind, elapsed):
    page = _counter_page()
    if page is None:
        return
    counters = list(struct.unpack(_COUNTER_FORMAT, page[:]))
    elapsed_ns = round(elapsed * 1_000_000_000)
    counters[0] += 1
    counters[1] += elapsed_ns
    offset = _KIND_OFFSET[step_kind]
    counters[offset] += 1
    counters[offset + 1] += elapsed_ns
    page[:] = struct.pack(_COUNTER_FORMAT, *counters)


def _install_scheduler_timer():
    original = Scheduler.schedule
    if getattr(original, "_cpu_gpu_codesign_timed", False):
        return

    @wraps(original)
    def timed_schedule(self, *args, **kwargs):
        started = perf_counter()
        output = original(self, *args, **kwargs)
        elapsed = perf_counter() - started
        if output.scheduled_new_reqs:
            step_kind = "admission"
        elif output.total_num_scheduled_tokens == 0:
            step_kind = "empty"
        else:
            step_kind = "steady"
        _record_counter(step_kind, elapsed)
        return output

    timed_schedule._cpu_gpu_codesign_timed = True
    Scheduler.schedule = timed_schedule


_install_scheduler_timer()


if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
