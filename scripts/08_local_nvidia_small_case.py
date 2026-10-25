#!/usr/bin/env python3
"""Run a small, boundary-correct vLLM CPU/GPU co-design experiment.

This script deliberately does not label the client-to-first-token interval as
"CPU time".  It reports client-observed timing, vLLM's internal Engine queue
time, and NVIDIA GPU samples as separate signals.
"""

import argparse
import asyncio
import csv
import json
import math
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import aiohttp


METRIC_RE = re.compile(
    r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|NaN|[+-]?Inf)$"
)


def parse_metrics(text):
    values = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, labels, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if math.isfinite(value):
            values[(name, labels or "")] = value
    return values


def metric_total(metrics, name):
    return sum(value for (metric_name, _), value in metrics.items()
               if metric_name == name)


async def fetch_metrics(session, base_url):
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        return parse_metrics(await response.text())


async def sample_gpu(samples, stop_event, interval_ms):
    fields = (
        "timestamp,index,utilization.gpu,utilization.memory,memory.used,"
        "memory.total,power.draw,temperature.gpu"
    )
    process = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
        f"--loop-ms={interval_ms}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        while not stop_event.is_set():
            try:
                raw_line = await asyncio.wait_for(process.stdout.readline(), 1.0)
            except asyncio.TimeoutError:
                continue
            if not raw_line:
                break
            columns = [column.strip() for column in raw_line.decode().split(",")]
            if len(columns) != 8:
                continue
            samples.append({
                "sample_monotonic_s": time.perf_counter(),
                "nvidia_timestamp": columns[0],
                "gpu_index": int(columns[1]),
                "gpu_util_pct": float(columns[2]),
                "memory_util_pct": float(columns[3]),
                "memory_used_mib": float(columns[4]),
                "memory_total_mib": float(columns[5]),
                "power_w": float(columns[6]),
                "temperature_c": float(columns[7]),
            })
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


async def request_once(
    session, base_url, model, context_tokens, max_tokens, request_tag="warmup"
):
    # Repetition makes the input deterministic. context_tokens is intentionally
    # approximate; the server-reported usage below is authoritative.
    # Put the unique tag before the repeated body so different measured
    # requests cannot reuse nearly the entire prompt through prefix caching.
    prompt = f"request_tag={request_tag}\n" + "benchmark " * context_tokens
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt + "\nSummarize the input pattern."},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request_start = time.perf_counter()
    first_token_at = None
    last_token_at = None
    headers_at = None
    usage = {}
    error_body = None

    async with session.post(
        f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        headers_at = time.perf_counter()
        if response.status != 200:
            error_body = await response.text()
        else:
            async for raw_line in response.content:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if delta.get("content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    last_token_at = now

    end_at = time.perf_counter()
    if error_body is not None:
        return {"error": error_body, "http_status": response.status}

    return {
        "context_tokens_requested_approx": context_tokens,
        "prompt_tokens_server": usage.get("prompt_tokens"),
        "completion_tokens_server": usage.get("completion_tokens"),
        "send_to_headers_ms": (headers_at - request_start) * 1000,
        "ttft_ms": ((first_token_at - request_start) * 1000
                    if first_token_at else None),
        "headers_to_first_token_ms": ((first_token_at - headers_at) * 1000
                                      if first_token_at else None),
        "decode_span_ms": ((last_token_at - first_token_at) * 1000
                           if first_token_at and last_token_at else None),
        "e2e_ms": (end_at - request_start) * 1000,
    }


def aggregate_requests(requests):
    output = {"request_count": len(requests)}
    for field in (
        "send_to_headers_ms", "ttft_ms", "headers_to_first_token_ms",
        "decode_span_ms", "e2e_ms", "prompt_tokens_server",
        "completion_tokens_server",
    ):
        values = [row[field] for row in requests if row.get(field) is not None]
        if values:
            ordered = sorted(values)
            output[f"{field}_mean"] = statistics.fmean(values)
            output[f"{field}_p50"] = statistics.median(values)
            output[f"{field}_max"] = ordered[-1]
    return output


async def run_scenario(session, args, label, context_tokens, concurrency, batches):
    before = await fetch_metrics(session, args.base_url)
    requests = []
    started_at = time.perf_counter()

    for batch_index in range(batches):
        rows = await asyncio.gather(*(
            request_once(
                session,
                args.base_url,
                args.model,
                context_tokens,
                args.max_tokens,
                request_tag=f"{label}-b{batch_index}-r{request_index}",
            )
            for request_index in range(concurrency)
        ))
        failures = [row for row in rows if "error" in row]
        if failures:
            raise RuntimeError(f"request failed: {failures[0]}")
        requests.extend(rows)
        print(
            f"  {label} batch {batch_index + 1}/{batches}: "
            f"mean TTFT={statistics.fmean(row['ttft_ms'] for row in rows):.1f} ms, "
            f"mean E2E={statistics.fmean(row['e2e_ms'] for row in rows):.1f} ms"
        )

    finished_at = time.perf_counter()
    after = await fetch_metrics(session, args.base_url)
    queue_sum_name = "vllm:request_queue_time_seconds_sum"
    queue_count_name = "vllm:request_queue_time_seconds_count"
    queue_sum_delta = metric_total(after, queue_sum_name) - metric_total(before, queue_sum_name)
    queue_count_delta = metric_total(after, queue_count_name) - metric_total(before, queue_count_name)

    result = {
        "label": label,
        "context_tokens_requested_approx": context_tokens,
        "concurrency": concurrency,
        "batches": batches,
        "started_monotonic_s": started_at,
        "finished_monotonic_s": finished_at,
        "requests": requests,
        "summary": aggregate_requests(requests),
        "engine_queue": {
            "sum_delta_seconds": queue_sum_delta,
            "count_delta": queue_count_delta,
            "mean_ms": (1000 * queue_sum_delta / queue_count_delta
                        if queue_count_delta > 0 else None),
            "definition": "scheduled_ts - queued_ts; vLLM Engine boundary only",
        },
    }
    return result


def summarize_gpu(samples, start, end):
    selected = [row for row in samples
                if start <= row["sample_monotonic_s"] <= end]
    if not selected:
        return {"sample_count": 0}
    utilization = [row["gpu_util_pct"] for row in selected]
    return {
        "sample_count": len(selected),
        "gpu_util_pct_mean": statistics.fmean(utilization),
        "gpu_util_pct_max": max(utilization),
        "gpu_idle_sample_fraction": sum(value == 0 for value in utilization) / len(utilization),
        "memory_used_mib_max": max(row["memory_used_mib"] for row in selected),
        "power_w_mean": statistics.fmean(row["power_w"] for row in selected),
    }


async def async_main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_samples = []
    stop_gpu = asyncio.Event()
    gpu_task = asyncio.create_task(sample_gpu(gpu_samples, stop_gpu, args.gpu_interval_ms))

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max(8, args.max_concurrency * 2))
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            print("Warm-up request...")
            warmup = await request_once(
                session, args.base_url, args.model, min(256, args.context_1), 8
            )
            if "error" in warmup:
                raise RuntimeError(f"warm-up failed: {warmup}")

            scenarios = [
                ("conc1_ctx1k", args.context_1, 1, args.batches),
                ("conc2_ctx4k", args.context_2, args.max_concurrency, args.batches),
            ]
            results = []
            for scenario in scenarios:
                print(f"Running {scenario[0]}...")
                result = await run_scenario(session, args, *scenario)
                results.append(result)
    finally:
        stop_gpu.set()
        await gpu_task

    for result in results:
        result["gpu"] = summarize_gpu(
            gpu_samples,
            result["started_monotonic_s"],
            result["finished_monotonic_s"],
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"local_small_case_{timestamp}.json"
    csv_path = output_dir / f"gpu_samples_{timestamp}.csv"
    report = {
        "base_url": args.base_url,
        "model": args.model,
        "note": (
            "Client timing, Engine queue, and GPU utilization have different "
            "boundaries. GPU idle samples alone do not prove CPU-induced starvation."
        ),
        "scenarios": results,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if gpu_samples:
        with csv_path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=gpu_samples[0].keys())
            writer.writeheader()
            writer.writerows(gpu_samples)

    print(f"\nResult: {json_path}")
    print(f"GPU samples: {csv_path}")
    for result in results:
        summary = result["summary"]
        queue_mean = result["engine_queue"]["mean_ms"]
        queue_text = f"{queue_mean:.4f} ms" if queue_mean is not None else "unavailable"
        print(
            f"{result['label']}: TTFT={summary['ttft_ms_mean']:.1f} ms, "
            f"E2E={summary['e2e_ms_mean']:.1f} ms, Engine queue={queue_text}, "
            f"GPU mean={result['gpu'].get('gpu_util_pct_mean', float('nan')):.1f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--output-dir", default="results/local-nvidia-small")
    parser.add_argument("--context-1", type=int, default=1024)
    parser.add_argument("--context-2", type=int, default=4096)
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--gpu-interval-ms", type=int, default=200)
    parser.add_argument("--request-timeout", type=float, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(async_main(parse_args()))
