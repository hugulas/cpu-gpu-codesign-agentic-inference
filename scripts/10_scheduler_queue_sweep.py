#!/usr/bin/env python3
"""Sweep 1K requests and separate Engine queue from scheduler wall time."""

import argparse
import asyncio
import json
import math
import mmap
import re
import statistics
import struct
import time
from datetime import datetime
from pathlib import Path

import aiohttp


METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)$"
)
SCHEDULER_COUNTER_FORMAT = "<8Q"
SCHEDULER_FIELDS = (
    "total_count", "total_ns", "admission_count", "admission_ns",
    "steady_count", "steady_ns", "empty_count", "empty_ns",
)
FRONTEND_COUNTER_FORMAT = "<8Q"
FRONTEND_FIELDS = (
    "preprocess_count", "preprocess_ns", "template_count", "template_ns",
    "tokenize_count", "tokenize_ns", "add_request_count", "add_request_ns",
)


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        value = float(raw_value)
        if math.isfinite(value):
            values[(name, labels or "")] = value
    return values


def metric_total(metrics, name):
    return sum(value for (metric_name, _), value in metrics.items()
               if metric_name == name)


def metric_total_with_label(metrics, name, label_fragment):
    return sum(
        value for (metric_name, labels), value in metrics.items()
        if metric_name == name and label_fragment in labels
    )


def metric_delta_mean_ms(before, after, base_name):
    sum_delta = metric_total(after, f"{base_name}_sum") - metric_total(
        before, f"{base_name}_sum"
    )
    count_delta = metric_total(after, f"{base_name}_count") - metric_total(
        before, f"{base_name}_count"
    )
    return {
        "sum_delta_seconds": sum_delta,
        "count_delta": count_delta,
        "mean_ms": 1000 * sum_delta / count_delta if count_delta > 0 else None,
    }


def read_scheduler_counters(path):
    with open(path, "rb") as counter_file:
        page = mmap.mmap(counter_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            values = struct.unpack(SCHEDULER_COUNTER_FORMAT, page[:64])
        finally:
            page.close()
    return dict(zip(SCHEDULER_FIELDS, values))


def read_frontend_counters(path):
    with open(path, "rb") as counter_file:
        page = mmap.mmap(counter_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            values = struct.unpack(FRONTEND_COUNTER_FORMAT, page[:64])
        finally:
            page.close()
    return dict(zip(FRONTEND_FIELDS, values))


def frontend_delta(before, after):
    delta = {key: after[key] - before[key] for key in FRONTEND_FIELDS}
    result = {"raw_delta": delta}
    for phase in ("preprocess", "template", "tokenize", "add_request"):
        count = delta[f"{phase}_count"]
        elapsed_ns = delta[f"{phase}_ns"]
        result[phase] = {
            "count": count,
            "wall_ms_sum": elapsed_ns / 1e6,
            "mean_ms": elapsed_ns / count / 1e6 if count else None,
        }
    result["renderer_queue_and_other_mean_ms"] = (
        result["preprocess"]["mean_ms"]
        - result["template"]["mean_ms"]
        - result["tokenize"]["mean_ms"]
    )
    return result


def scheduler_delta(before, after, request_count):
    delta = {key: after[key] - before[key] for key in SCHEDULER_FIELDS}
    result = {"raw_delta": delta}
    for kind in ("total", "admission", "steady", "empty"):
        count = delta[f"{kind}_count"]
        elapsed_ns = delta[f"{kind}_ns"]
        result[kind] = {
            "step_count": count,
            "wall_ms_sum": elapsed_ns / 1e6,
            "wall_us_per_step": elapsed_ns / count / 1e3 if count else None,
        }
    result["total_wall_ms_per_request"] = (
        delta["total_ns"] / request_count / 1e6 if request_count else None
    )
    result["steps_per_request"] = (
        delta["total_count"] / request_count if request_count else None
    )
    return result


async def fetch_metrics(session, base_url):
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        return parse_metrics(await response.text())


async def send_request(session, args, request_tag):
    prompt = f"request_tag={request_tag}\n" + "benchmark " * args.context_tokens
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt + "\nSummarize the pattern."},
        ],
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    headers_at = None
    first_token_at = None
    usage = {}
    async with session.post(f"{args.base_url}/v1/chat/completions", json=payload) as response:
        headers_at = time.perf_counter()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        async for raw_line in response.content:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first_token_at is None and (delta.get("content") or delta.get("reasoning")):
                first_token_at = time.perf_counter()
    finished = time.perf_counter()
    return {
        "send_to_headers_ms": (headers_at - started) * 1000,
        "ttft_ms": (first_token_at - started) * 1000,
        "e2e_ms": (finished - started) * 1000,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


async def sample_runtime(session, args, stop_event, samples):
    while not stop_event.is_set():
        try:
            metrics = await fetch_metrics(session, args.base_url)
            samples.append({
                "timestamp": time.perf_counter(),
                "running": metric_total(metrics, "vllm:num_requests_running"),
                "waiting": metric_total(metrics, "vllm:num_requests_waiting"),
                "waiting_capacity": sum(
                    value for (name, labels), value in metrics.items()
                    if name == "vllm:num_requests_waiting_by_reason"
                    and 'reason="capacity"' in labels
                ),
            })
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), args.sample_interval)
        except asyncio.TimeoutError:
            pass


def distribution(rows, key):
    values = sorted(row[key] for row in rows if row.get(key) is not None)
    if not values:
        return {}
    p95_index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": values[p95_index],
        "max": values[-1],
    }


async def run_level(session, args, concurrency, run_nonce):
    await asyncio.sleep(args.settle_seconds)
    metrics_before = await fetch_metrics(session, args.base_url)
    scheduler_before = read_scheduler_counters(args.scheduler_metrics_file)
    frontend_before = read_frontend_counters(args.frontend_metrics_file)
    runtime_samples = []
    stop_sampler = asyncio.Event()
    sampler = asyncio.create_task(sample_runtime(session, args, stop_sampler, runtime_samples))
    rows = []
    started = time.perf_counter()
    try:
        for batch in range(args.batches):
            batch_rows = await asyncio.gather(*(
                send_request(
                    session,
                    args,
                    f"{run_nonce}-c{concurrency}-b{batch}-r{request_index}",
                )
                for request_index in range(concurrency)
            ))
            rows.extend(batch_rows)
            print(
                f"  c={concurrency:>2} batch={batch + 1}/{args.batches}: "
                f"TTFT={statistics.fmean(row['ttft_ms'] for row in batch_rows):.1f} ms, "
                f"E2E={statistics.fmean(row['e2e_ms'] for row in batch_rows):.1f} ms"
            )
    finally:
        stop_sampler.set()
        await sampler
    elapsed = time.perf_counter() - started
    metrics_after = await fetch_metrics(session, args.base_url)
    scheduler_after = read_scheduler_counters(args.scheduler_metrics_file)
    frontend_after = read_frontend_counters(args.frontend_metrics_file)

    metric_names = {
        "queue": "vllm:request_queue_time_seconds",
        "prefill": "vllm:request_prefill_time_seconds",
        "decode": "vllm:request_decode_time_seconds",
        "inference": "vllm:request_inference_time_seconds",
        "server_ttft": "vllm:time_to_first_token_seconds",
        "server_e2e": "vllm:e2e_request_latency_seconds",
    }
    server_times = {
        label: metric_delta_mean_ms(metrics_before, metrics_after, metric_name)
        for label, metric_name in metric_names.items()
    }
    server_times["renderer_start_to_queue_estimate_ms"] = (
        server_times["server_ttft"]["mean_ms"]
        - server_times["queue"]["mean_ms"]
        - server_times["prefill"]["mean_ms"]
    )
    prompt_sources = {}
    for source in ("local_compute", "local_cache_hit", "external_kv_transfer"):
        label = f'source="{source}"'
        prompt_sources[source] = metric_total_with_label(
            metrics_after, "vllm:prompt_tokens_by_source_total", label
        ) - metric_total_with_label(
            metrics_before, "vllm:prompt_tokens_by_source_total", label
        )
    request_count = len(rows)
    return {
        "concurrency": concurrency,
        "batches": args.batches,
        "request_count": request_count,
        "wall_seconds": elapsed,
        "request_throughput_per_s": request_count / elapsed,
        "client": {
            key: distribution(rows, key)
            for key in ("send_to_headers_ms", "ttft_ms", "e2e_ms")
        },
        "prompt_tokens": distribution(rows, "prompt_tokens"),
        "completion_tokens": distribution(rows, "completion_tokens"),
        "server": server_times,
        "prompt_token_sources_delta": prompt_sources,
        "scheduler": scheduler_delta(scheduler_before, scheduler_after, request_count),
        "frontend": frontend_delta(frontend_before, frontend_after),
        "runtime_gauges": {
            "sample_count": len(runtime_samples),
            "running_max": max((row["running"] for row in runtime_samples), default=0),
            "waiting_max": max((row["waiting"] for row in runtime_samples), default=0),
            "waiting_capacity_max": max(
                (row["waiting_capacity"] for row in runtime_samples), default=0
            ),
        },
    }


async def async_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_nonce = str(time.time_ns())
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency_levels) + 8)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        print("Warm-up...")
        await send_request(session, args, f"{run_nonce}-warmup")
        results = []
        for concurrency in args.concurrency_levels:
            print(f"Running concurrency={concurrency}...")
            results.append(await run_level(session, args, concurrency, run_nonce))

    report = {
        "model": args.model,
        "context_tokens_requested_approx": args.context_tokens,
        "max_tokens": args.max_tokens,
        "concurrency_levels": args.concurrency_levels,
        "boundaries": {
            "queue": "first QUEUED event to first SCHEDULED event",
            "prefill": "first SCHEDULED event to first NEW_TOKEN; includes model prefill and any preemption",
            "scheduler": "wall time inside Scheduler.schedule(), aggregated across engine steps",
            "renderer_start_to_queue": "EngineInput arrival_time is captured at renderer start; excludes HTTP receive and JSON/Pydantic validation",
            "frontend_preprocess": "OnlineRenderer.preprocess_chat wall time; includes renderer executor wait",
            "template_tokenize": "function execution wall time inside renderer workers; excludes executor wait",
            "add_request": "AsyncLLM.add_request wall time; includes input construction, local registration, and IPC send wait",
            "warning": "Scheduler steps are batched/shared; scheduler ms/request is workload-normalized CPU work, not additive per-request latency.",
        },
        "results": results,
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"scheduler_queue_sweep_{timestamp}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nResult: {output_path}")
    print("conc  render2q_ms  queue_ms  sched_ms/req  sched_us/step  prefill_ms  TTFT_ms  wait_max  req/s")
    for row in results:
        print(
            f"{row['concurrency']:>4}  "
            f"{row['server']['renderer_start_to_queue_estimate_ms']:>10.2f}  "
            f"{row['server']['queue']['mean_ms']:>8.2f}  "
            f"{row['scheduler']['total_wall_ms_per_request']:>12.3f}  "
            f"{row['scheduler']['total']['wall_us_per_step']:>13.2f}  "
            f"{row['server']['prefill']['mean_ms']:>10.2f}  "
            f"{row['server']['server_ttft']['mean_ms']:>7.2f}  "
            f"{row['runtime_gauges']['waiting_max']:>8.0f}  "
            f"{row['request_throughput_per_s']:>5.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--output-dir", default="results/scheduler-queue-sweep")
    parser.add_argument("--scheduler-metrics-file", default="/tmp/vllm_scheduler_metrics.bin")
    parser.add_argument("--frontend-metrics-file", default="/tmp/vllm_frontend_metrics.bin")
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()
    args.concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    return args


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
