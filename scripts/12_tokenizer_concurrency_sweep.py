#!/usr/bin/env python3
"""Closed-loop concurrency sweep for vLLM's chat /tokenize endpoint."""

import argparse
import asyncio
from datetime import datetime
import json
import math
import mmap
from pathlib import Path
import statistics
import struct
import time

import aiohttp


COUNTER_FORMAT = "<8Q"
COUNTER_FIELDS = (
    "preprocess_count",
    "preprocess_ns",
    "template_count",
    "template_ns",
    "tokenize_count",
    "tokenize_ns",
    "add_request_count",
    "add_request_ns",
)


def read_counters(path):
    with open(path, "rb") as counter_file:
        page = mmap.mmap(counter_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            values = struct.unpack(COUNTER_FORMAT, page[:64])
        finally:
            page.close()
    return dict(zip(COUNTER_FIELDS, values))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def distribution(values):
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


async def one_request(session, args, prompt, sequence):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": f"id={sequence}\n{prompt}"},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter_ns()
    async with session.post(f"{args.base_url}/tokenize", json=payload) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body}")
    return (time.perf_counter_ns() - started) / 1e6, body["count"]


async def run_level(session, args, prompt, concurrency):
    before = read_counters(args.frontend_metrics_file)
    sequence = 0
    latencies = []
    token_counts = []
    started = time.perf_counter()

    async def worker(worker_index):
        nonlocal sequence
        while True:
            current = sequence
            sequence += 1
            if current >= args.requests:
                return
            latency, count = await one_request(
                session, args, prompt, f"c{concurrency}-w{worker_index}-r{current}"
            )
            latencies.append(latency)
            token_counts.append(count)

    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    wall_seconds = time.perf_counter() - started
    after = read_counters(args.frontend_metrics_file)
    delta = {key: after[key] - before[key] for key in COUNTER_FIELDS}

    phases = {}
    for phase in ("preprocess", "template", "tokenize"):
        count = delta[f"{phase}_count"]
        elapsed_ns = delta[f"{phase}_ns"]
        phases[phase] = {
            "count": count,
            "wall_ms_sum": elapsed_ns / 1e6,
            "mean_ms": elapsed_ns / count / 1e6 if count else None,
        }
    phases["renderer_queue_and_other_mean_ms"] = (
        phases["preprocess"]["mean_ms"]
        - phases["template"]["mean_ms"]
        - phases["tokenize"]["mean_ms"]
    )
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "wall_seconds": wall_seconds,
        "throughput_requests_per_s": len(latencies) / wall_seconds,
        "http_latency_ms": distribution(latencies),
        "token_count": distribution(token_counts),
        "frontend_phases": phases,
        "raw_counter_delta": delta,
    }


async def async_main(args):
    prompt = "benchmark concurrency tokenizer measurement " * args.prompt_repetitions
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency_levels) + 4)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for warmup_index in range(args.warmup_requests):
            await one_request(session, args, prompt, f"warmup-{warmup_index}")

        results = []
        for concurrency in args.concurrency_levels:
            print(f"running c={concurrency}, n={args.requests}")
            row = await run_level(session, args, prompt, concurrency)
            results.append(row)
            phases = row["frontend_phases"]
            print(
                f"  {row['throughput_requests_per_s']:.1f} req/s, "
                f"HTTP p50/p95={row['http_latency_ms']['p50']:.2f}/"
                f"{row['http_latency_ms']['p95']:.2f} ms, "
                f"preprocess={phases['preprocess']['mean_ms']:.2f} ms, "
                f"template={phases['template']['mean_ms']:.2f} ms, "
                f"tokenize={phases['tokenize']['mean_ms']:.2f} ms, "
                f"queue+other={phases['renderer_queue_and_other_mean_ms']:.2f} ms"
            )

    report = {
        "model": args.model,
        "renderer_num_workers": args.renderer_num_workers,
        "prompt_repetitions": args.prompt_repetitions,
        "concurrency_levels": args.concurrency_levels,
        "requests_per_level": args.requests,
        "method": {
            "load_model": "closed-loop fixed concurrency against chat /tokenize",
            "preprocess": "OnlineRenderer.preprocess_chat wall time; includes executor wait",
            "template": "safe_apply_chat_template execution time; excludes executor wait",
            "tokenize": "BaseRenderer._tokenize_prompt execution time; excludes executor wait",
            "derived_queue_other": "preprocess - template - tokenize; includes executor queue and lightweight renderer work",
        },
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / (
        f"tokenizer_workers{args.renderer_num_workers}_{timestamp}.json"
    )
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"result: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--frontend-metrics-file", default="/tmp/vllm_frontend_metrics.bin")
    parser.add_argument("--output-dir", default="results/cpu-overhead-decomposition")
    parser.add_argument("--renderer-num-workers", type=int, required=True)
    parser.add_argument("--prompt-repetitions", type=int, default=200)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    args.concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    return args


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
