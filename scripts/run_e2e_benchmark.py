#!/usr/bin/env python3
"""Run an exact-token, request-correlated vLLM E2E benchmark.

With ``--model-path`` this command owns the complete service lifecycle. Without
it, the command attaches to an existing service at ``--base-url``.
"""

import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import uuid
from urllib.error import URLError
from urllib.request import urlopen

import aiohttp


def wait_for_health(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited during startup with code {process.returncode}"
            )
        try:
            with urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become healthy within {timeout}s")


def stop_process_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def distribution(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values) / 1e6,
        "p50_ms": statistics.median(values) / 1e6,
        "p95_ms": percentile(values, 0.95) / 1e6,
        "max_ms": max(values) / 1e6,
    }


def numeric_distribution(values):
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


async def send_request(session, args, request_id, prompt):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": args.output_tokens,
        # Make output length a controlled variable rather than an upper bound.
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        # Make every request a distinct prefix-cache domain. A text nonce does
        # not prevent caching of the shared system/template prefix.
        "cache_salt": request_id,
    }
    events = {"client_start": time.monotonic_ns()}
    headers = {"X-Request-Id": request_id}
    usage = None
    async with session.post(
        f"{args.base_url}/v1/chat/completions", json=payload, headers=headers
    ) as response:
        events["client_headers"] = time.monotonic_ns()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        async for raw_line in response.content:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if "client_first_content" not in events and (
                delta.get("content") or delta.get("reasoning")
            ):
                events["client_first_content"] = time.monotonic_ns()
    events["client_end"] = time.monotonic_ns()
    return {"request_id": request_id, "events_ns": events, "usage": usage}


async def tokenize_count(session, args, prompt):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(f"{args.base_url}/tokenize", json=payload) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(f"/tokenize HTTP {response.status}: {body}")
        return body["count"]


async def calibrate_prompt(session, args, target_tokens):
    """Build text whose complete chat-template prompt is exactly target_tokens."""
    unit = args.calibration_unit
    base_count = await tokenize_count(session, args, "")
    if target_tokens < base_count:
        raise ValueError(
            f"prompt target {target_tokens} is below chat-template minimum "
            f"{base_count} tokens"
        )

    low = 0
    high = max(1, target_tokens - base_count)
    while await tokenize_count(session, args, unit * high) < target_tokens:
        high *= 2
    while low <= high:
        middle = (low + high) // 2
        prompt = unit * middle
        count = await tokenize_count(session, args, prompt)
        if count == target_tokens:
            return prompt, {
                "target_tokens": target_tokens,
                "actual_tokens": count,
                "unit": unit,
                "unit_repetitions": middle,
                "base_chat_tokens": base_count,
            }
        if count < target_tokens:
            low = middle + 1
        else:
            high = middle - 1
    raise RuntimeError(
        f"cannot construct exactly {target_tokens} prompt tokens with "
        f"calibration unit {unit!r}; nearest repetition boundary is {low}"
    )


async def run_level(session, args, concurrency, run_id, prompt):
    rows = []
    for batch in range(args.batches):
        batch_rows = await asyncio.gather(
            *(
                send_request(
                    session,
                    args,
                    f"{run_id}-c{concurrency}-b{batch}-r{index}",
                    prompt,
                )
                for index in range(concurrency)
            )
        )
        rows.extend(batch_rows)
    return rows


def load_server_traces(path, wanted_ids):
    traces = {}
    with open(path, encoding="utf-8") as trace_file:
        for line in trace_file:
            row = json.loads(line)
            request_id = row.get("request_id")
            # EngineCore sees vLLM's internal request ID, which appends eight
            # random hex characters to the externally supplied ID.
            if request_id not in wanted_ids and isinstance(request_id, str):
                request_id = re.sub(r"-[0-9a-f]{8}$", "", request_id)
            if request_id in wanted_ids:
                trace = traces.setdefault(
                    request_id,
                    {"request_id": request_id, "events_ns": {}},
                )
                trace["events_ns"].update(row.get("events_ns", {}))
                for key, value in row.items():
                    if key not in ("request_id", "events_ns"):
                        trace[key] = value
    return traces


PHASES = {
    "client_to_asgi": ("client_start", "asgi_enter"),
    "request_body_receive": ("asgi_enter", "request_body_complete"),
    "json_pydantic_dependencies": ("request_body_complete", "endpoint_enter"),
    "endpoint_to_preprocess": ("endpoint_enter", "preprocess_enter"),
    "preprocess_total": ("preprocess_enter", "preprocess_exit"),
    "chat_parse_and_template": ("render_messages_enter", "render_messages_exit"),
    "template_executor_wait_and_run": (
        "template_executor_enter",
        "template_executor_exit",
    ),
    "tokenizer_executor_wait_and_run": (
        "tokenizer_executor_enter",
        "tokenizer_executor_exit",
    ),
    "engine_input_build": ("engine_input_enter", "engine_input_exit"),
    "postprocess_to_endpoint_return": ("preprocess_exit", "endpoint_return"),
    "stream_setup_to_generate": ("endpoint_return", "generate_enter"),
    "generate_to_add_request": ("generate_enter", "add_request_enter"),
    "add_request": ("add_request_enter", "add_request_exit"),
    "ipc_core_admission": ("add_request_exit", "engine_queued"),
    "engine_queue": ("engine_queued", "engine_scheduled"),
    "scheduled_to_first_token": ("engine_scheduled", "engine_first_token"),
    "scheduler_admission_wall": ("engine_scheduled", "scheduler_return"),
    "post_scheduler_to_first_token": ("scheduler_return", "engine_first_token"),
    "engine_decode": ("engine_first_token", "engine_last_token"),
    "first_token_to_client_content": ("engine_first_token", "client_first_content"),
    "last_token_to_response_end": ("engine_last_token", "response_end"),
    "response_end_to_client_end": ("response_end", "client_end"),
    "client_ttft": ("client_start", "client_first_content"),
    "client_e2e": ("client_start", "client_end"),
}

MAIN_TIMELINE = (
    "client_start",
    "asgi_enter",
    "request_body_complete",
    "endpoint_enter",
    "preprocess_enter",
    "preprocess_exit",
    "endpoint_return",
    "generate_enter",
    "add_request_enter",
    "add_request_exit",
    "engine_queued",
    "engine_scheduled",
    "engine_first_token",
    "engine_last_token",
    "response_end",
    "client_end",
)


def analyze_level(client_rows, server_traces, concurrency, prompt_tokens_target, output_tokens_target):
    merged = []
    missing = []
    for client in client_rows:
        server = server_traces.get(client["request_id"])
        if server is None:
            missing.append(client["request_id"])
            continue
        events = {**server["events_ns"], **client["events_ns"]}
        merged.append(
            {
                "request_id": client["request_id"],
                "events_ns": events,
                "usage": client.get("usage"),
                "server_values": server.get("values", {}),
            }
        )

    phase_values = {}
    invalid_order = []
    for phase, (start, end) in PHASES.items():
        values = []
        for row in merged:
            events = row["events_ns"]
            if start in events and end in events:
                delta = events[end] - events[start]
                if delta < 0:
                    invalid_order.append((row["request_id"], phase, delta))
                else:
                    values.append(delta)
        phase_values[phase] = distribution(values)

    prompt_tokens = [
        row["usage"]["prompt_tokens"]
        for row in client_rows
        if row.get("usage") and row["usage"].get("prompt_tokens") is not None
    ]
    completion_tokens = [
        row["usage"]["completion_tokens"]
        for row in client_rows
        if row.get("usage") and row["usage"].get("completion_tokens") is not None
    ]
    renderer_workers = sorted(
        {
            row["server_values"].get("renderer_num_workers")
            for row in merged
            if row["server_values"].get("renderer_num_workers") is not None
        }
    )
    accounting_errors = []
    incomplete_timelines = []
    for row in merged:
        events = row["events_ns"]
        missing_events = [event for event in MAIN_TIMELINE if event not in events]
        if missing_events:
            incomplete_timelines.append(
                {"request_id": row["request_id"], "missing": missing_events}
            )
            continue
        phase_sum = sum(
            events[end] - events[start]
            for start, end in zip(MAIN_TIMELINE, MAIN_TIMELINE[1:])
        )
        accounting_errors.append(
            (events["client_end"] - events["client_start"]) - phase_sum
        )

    return {
        "concurrency": concurrency,
        "prompt_tokens_target": prompt_tokens_target,
        "output_tokens_target": output_tokens_target,
        "request_count": len(client_rows),
        "matched_trace_count": len(merged),
        "missing_trace_ids": missing,
        "invalid_order": invalid_order,
        "incomplete_main_timelines": incomplete_timelines,
        "main_timeline_accounting_error_ns": {
            "count": len(accounting_errors),
            "min": min(accounting_errors) if accounting_errors else None,
            "max": max(accounting_errors) if accounting_errors else None,
        },
        "prompt_tokens": numeric_distribution(prompt_tokens),
        "completion_tokens": numeric_distribution(completion_tokens),
        "length_validation": {
            "prompt_all_exact": len(prompt_tokens) == len(client_rows)
            and all(value == prompt_tokens_target for value in prompt_tokens),
            "completion_all_exact": len(completion_tokens) == len(client_rows)
            and all(value == output_tokens_target for value in completion_tokens),
        },
        "renderer_num_workers_observed": renderer_workers,
        "phases": phase_values,
        "requests": merged,
    }


async def async_main(args):
    run_id = uuid.uuid4().hex[:12]
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency_levels) + 4)
    rows_by_case = {}
    prompt_calibrations = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        request_prefix = "baseline" if args.client_only else "codesign"
        request_namespace = f"{request_prefix}-{run_id}"
        for prompt_tokens_target in args.prompt_token_targets:
            prompt, calibration = await calibrate_prompt(
                session, args, prompt_tokens_target
            )
            prompt_calibrations.append(calibration)
            warmup_id = f"{request_namespace}-p{prompt_tokens_target}-warmup"
            warmup = await send_request(session, args, warmup_id, prompt)
            warmup_usage = warmup.get("usage") or {}
            if warmup_usage.get("prompt_tokens") != prompt_tokens_target:
                raise RuntimeError(
                    f"warmup prompt length mismatch: target={prompt_tokens_target}, "
                    f"actual={warmup_usage.get('prompt_tokens')}"
                )
            if warmup_usage.get("completion_tokens") != args.output_tokens:
                raise RuntimeError(
                    f"warmup output length mismatch: target={args.output_tokens}, "
                    f"actual={warmup_usage.get('completion_tokens')}"
                )
            case_namespace = f"{request_namespace}-p{prompt_tokens_target}"
            for concurrency in args.concurrency_levels:
                print(
                    f"running prompt_tokens={prompt_tokens_target}, "
                    f"c={concurrency}, batches={args.batches}"
                )
                rows_by_case[(prompt_tokens_target, concurrency)] = await run_level(
                    session, args, concurrency, case_namespace, prompt
                )

    await asyncio.sleep(0.25)
    wanted_ids = {
        row["request_id"]
        for rows in rows_by_case.values()
        for row in rows
    }
    traces = {} if args.client_only else load_server_traces(args.trace_file, wanted_ids)
    if args.client_only:
        results = []
        for prompt_tokens_target in args.prompt_token_targets:
            for concurrency in args.concurrency_levels:
                rows = rows_by_case[(prompt_tokens_target, concurrency)]
                ttft = [
                    row["events_ns"]["client_first_content"]
                    - row["events_ns"]["client_start"]
                    for row in rows
                ]
                e2e = [
                    row["events_ns"]["client_end"] - row["events_ns"]["client_start"]
                    for row in rows
                ]
                prompt_tokens = [row["usage"]["prompt_tokens"] for row in rows]
                completion_tokens = [row["usage"]["completion_tokens"] for row in rows]
                results.append(
                    {
                        "prompt_tokens_target": prompt_tokens_target,
                        "output_tokens_target": args.output_tokens,
                        "concurrency": concurrency,
                        "request_count": len(rows),
                        "prompt_tokens": numeric_distribution(prompt_tokens),
                        "completion_tokens": numeric_distribution(completion_tokens),
                        "length_validation": {
                            "prompt_all_exact": all(
                                value == prompt_tokens_target for value in prompt_tokens
                            ),
                            "completion_all_exact": all(
                                value == args.output_tokens
                                for value in completion_tokens
                            ),
                        },
                        "client_ttft": distribution(ttft),
                        "client_e2e": distribution(e2e),
                    }
                )
    else:
        results = [
            analyze_level(
                rows_by_case[(prompt_tokens_target, concurrency)],
                traces,
                concurrency,
                prompt_tokens_target,
                args.output_tokens,
            )
            for prompt_tokens_target in args.prompt_token_targets
            for concurrency in args.concurrency_levels
        ]

    bad_lengths = [
        (row["prompt_tokens_target"], row["concurrency"], row["length_validation"])
        for row in results
        if not all(row["length_validation"].values())
    ]
    if bad_lengths:
        raise RuntimeError(f"token length validation failed: {bad_lengths}")
    if not args.client_only:
        trace_failures = []
        for row in results:
            failures = {}
            if row["matched_trace_count"] != row["request_count"]:
                failures["trace_match"] = (
                    row["matched_trace_count"],
                    row["request_count"],
                )
            if row["invalid_order"]:
                failures["invalid_order"] = row["invalid_order"]
            if row["incomplete_main_timelines"]:
                failures["incomplete_main_timelines"] = row[
                    "incomplete_main_timelines"
                ]
            accounting = row["main_timeline_accounting_error_ns"]
            if accounting["count"] != row["request_count"] or any(
                accounting[key] != 0 for key in ("min", "max")
            ):
                failures["main_timeline_accounting_error_ns"] = accounting
            if failures:
                trace_failures.append(
                    {
                        "prompt_tokens": row["prompt_tokens_target"],
                        "concurrency": row["concurrency"],
                        "failures": failures,
                    }
                )
        if trace_failures:
            raise RuntimeError(f"trace integrity validation failed: {trace_failures}")
    if not args.client_only and args.expected_renderer_workers is not None:
        mismatches = [
            (
                row["prompt_tokens_target"],
                row["concurrency"],
                row["renderer_num_workers_observed"],
            )
            for row in results
            if row["renderer_num_workers_observed"]
            != [args.expected_renderer_workers]
        ]
        if mismatches:
            raise RuntimeError(
                f"renderer worker mismatch; expected "
                f"{args.expected_renderer_workers}: {mismatches}"
            )

    report = {
        "run_id": run_id,
        "model": args.model,
        "clock": "time.monotonic_ns on the same host",
        "prompt_token_targets": args.prompt_token_targets,
        "output_tokens_target": args.output_tokens,
        "prompt_calibrations": prompt_calibrations,
        "expected_renderer_workers": args.expected_renderer_workers,
        "batches": args.batches,
        "client_only_untraced": args.client_only,
        "main_timeline": MAIN_TIMELINE,
        "phase_boundaries": PHASES,
        "results": results,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"e2e_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"result: {output_path}")
    if args.client_only:
        print("prompt conc requests TTFT_mean_ms TTFT_p95_ms E2E_mean_ms E2E_p95_ms")
        for row in results:
            print(
                f"{row['prompt_tokens_target']:>6} {row['concurrency']:>4} "
                f"{row['request_count']:>8} "
                f"{row['client_ttft']['mean_ms']:>12.2f} "
                f"{row['client_ttft']['p95_ms']:>11.2f} "
                f"{row['client_e2e']['mean_ms']:>11.2f} "
                f"{row['client_e2e']['p95_ms']:>10.2f}"
            )
        return

    print("prompt conc matched body_ms validate_ms preprocess_ms add_ms ipc_ms queue_ms prefill_ms decode_ms TTFT_ms E2E_ms")
    selected = (
        "request_body_receive",
        "json_pydantic_dependencies",
        "preprocess_total",
        "add_request",
        "ipc_core_admission",
        "engine_queue",
        "scheduled_to_first_token",
        "engine_decode",
        "client_ttft",
        "client_e2e",
    )
    for row in results:
        values = []
        for phase in selected:
            dist = row["phases"][phase]
            values.append(f"{dist['mean_ms']:.2f}" if dist else "NA")
        print(
            f"{row['prompt_tokens_target']:>6} {row['concurrency']:>4} "
            f"{row['matched_trace_count']:>7} "
            + " ".join(f"{value:>9}" for value in values)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    service = parser.add_argument_group("service lifecycle")
    service.add_argument(
        "--model-path",
        help="Launch and manage a traced vLLM service for this local model. "
        "Omit to attach to an existing service.",
    )
    service.add_argument("--base-url")
    service.add_argument("--host", default="127.0.0.1")
    service.add_argument("--port", type=int, default=8000)
    service.add_argument(
        "--parallel-tokenizer",
        action="store_true",
        help="Use/expect two renderer workers; default managed mode uses one.",
    )
    service.add_argument(
        "--renderer-workers",
        type=int,
        help="Explicit renderer worker count; overrides --parallel-tokenizer.",
    )
    service.add_argument("--max-model-len", type=int, default=8192)
    service.add_argument("--max-num-seqs", type=int, default=4)
    service.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    service.add_argument("--startup-timeout", type=float, default=180)

    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--trace-file")
    parser.add_argument("--output-dir", default="results/host-e2e-decomposition")
    parser.add_argument(
        "--prompt-tokens",
        default="1024",
        help="Comma-separated exact chat prompt token lengths.",
    )
    parser.add_argument(
        "--output-tokens",
        "--max-tokens",
        dest="output_tokens",
        type=int,
        default=16,
        help="Exact generated token length; ignore_eos is enabled.",
    )
    parser.add_argument("--calibration-unit", default="x ")
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--concurrency", default="1,4,8,16,32")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--client-only",
        action="store_true",
        help="Use untraced request IDs and report client TTFT/E2E only.",
    )
    parser.add_argument("--expected-renderer-workers", type=int)
    args = parser.parse_args()
    if args.renderer_workers is not None and args.renderer_workers < 1:
        parser.error("--renderer-workers must be >= 1")
    if args.output_tokens < 1:
        parser.error("--output-tokens must be >= 1")
    if args.batches < 1:
        parser.error("--batches must be >= 1")
    args.concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    args.prompt_token_targets = [
        int(value) for value in args.prompt_tokens.split(",")
    ]
    if not args.concurrency_levels or min(args.concurrency_levels) < 1:
        parser.error("all --concurrency values must be >= 1")
    if not args.prompt_token_targets or min(args.prompt_token_targets) < 1:
        parser.error("all --prompt-tokens values must be >= 1")

    args.base_url = args.base_url or f"http://{args.host}:{args.port}"
    selected_workers = args.renderer_workers
    if selected_workers is None and args.parallel_tokenizer:
        selected_workers = 2
    if args.model_path:
        selected_workers = selected_workers or 1
        if (
            args.expected_renderer_workers is not None
            and args.expected_renderer_workers != selected_workers
        ):
            parser.error(
                "managed service worker selection conflicts with "
                "--expected-renderer-workers"
            )
        args.expected_renderer_workers = selected_workers
    elif selected_workers is not None:
        if (
            args.expected_renderer_workers is not None
            and args.expected_renderer_workers != selected_workers
        ):
            parser.error(
                "worker selection conflicts with --expected-renderer-workers"
            )
        args.expected_renderer_workers = selected_workers
    return args


def run_managed(args):
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workers = args.expected_renderer_workers
    if args.trace_file is None:
        args.trace_file = f"/tmp/vllm_e2e_trace_{timestamp}_w{workers}.jsonl"
    server_log = output_dir / f"server_{timestamp}_w{workers}.log"

    env = os.environ.copy()
    for name in (
        "all_proxy",
        "ALL_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
    ):
        env.pop(name, None)
    env["CODESIGN_E2E_TRACE_FILE"] = args.trace_file

    server_command = [
        sys.executable,
        str(script_dir / "serve_e2e_trace.py"),
        "serve",
        args.model_path,
        "--served-model-name",
        args.model,
        "--language-model-only",
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--renderer-num-workers",
        str(workers),
        "--enforce-eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--disable-uvicorn-access-log",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print(f"renderer workers: {workers}")
    print(f"server log: {server_log}")
    with server_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            server_command,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_health(args.base_url, process, args.startup_timeout)
            asyncio.run(async_main(args))
        finally:
            stop_process_group(process)


def main():
    args = parse_args()
    if args.model_path:
        run_managed(args)
        return
    if args.trace_file is None and not args.client_only:
        raise SystemExit(
            "--trace-file is required when attaching to an existing traced service"
        )
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
