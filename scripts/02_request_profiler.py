#!/usr/bin/env python3
"""Profile synthetic chat requests with strict SSE and usage validation."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from benchmark_support import (
    build_messages,
    fetch_metrics,
    fresh_directory,
    parse_csv_ints,
    send_chat,
    write_json,
    write_results,
)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--context-lengths", default="1000,8000,32000")
    parser.add_argument("--concurrency-levels", default="1,4,16,32")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    fresh_directory(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    contexts = parse_csv_ints(args.context_lengths)
    concurrencies = parse_csv_ints(args.concurrency_levels)
    prepared = {}
    for target in contexts:
        messages, count = build_messages(tokenizer, target)
        if count + args.max_tokens > args.max_model_len:
            raise ValueError(
                f"target {target}: prompt={count} + output={args.max_tokens} "
                f"exceeds max model length {args.max_model_len}"
            )
        prepared[target] = (messages, count)

    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=max(concurrencies) * 2)
    all_results = []
    scenarios = []
    failures = 0
    url = args.api_endpoint.rstrip("/") + "/v1/chat/completions"
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for target in contexts:
            messages, count = prepared[target]
            for concurrency in concurrencies:
                name = f"ctx{target}-c{concurrency}"
                scenario_dir = args.output_dir / name
                scenario_dir.mkdir()
                (scenario_dir / "before.prom").write_text(
                    await fetch_metrics(session, args.api_endpoint)
                )
                results = []
                started = time.perf_counter()
                for iteration in range(args.num_iters):
                    batch = await asyncio.gather(
                        *(
                            send_chat(
                                session,
                                url,
                                f"{name}-i{iteration}-r{rank}",
                                args.model,
                                messages,
                                target,
                                count,
                                args.max_tokens,
                            )
                            for rank in range(concurrency)
                        )
                    )
                    results.extend(batch)
                elapsed = time.perf_counter() - started
                (scenario_dir / "after.prom").write_text(
                    await fetch_metrics(session, args.api_endpoint)
                )
                write_results(scenario_dir / "requests.jsonl", results)
                valid = [item for item in results if item.error is None]
                errors = len(results) - len(valid)
                failures += errors
                summary = {
                    "context_target": target,
                    "local_prompt_tokens": count,
                    "concurrency": concurrency,
                    "num_requests": len(results),
                    "num_success": len(valid),
                    "num_errors": errors,
                    "elapsed_s": elapsed,
                    "throughput_req_s": len(valid) / elapsed,
                }
                for field in (
                    "t_request_build_ms",
                    "t_headers_ms",
                    "t_ttft_ms",
                    "t_total_ms",
                    "t_sse_json_parse_ms",
                ):
                    values = [getattr(item, field) for item in valid]
                    if values:
                        summary[field + "_mean"] = statistics.mean(values)
                        summary[field + "_p50"] = statistics.median(values)
                write_json(scenario_dir / "summary.json", summary)
                scenarios.append(summary)
                all_results.extend(results)

    output = {
        "schema_version": 2,
        "scenarios": scenarios,
        "total_requests": len(all_results),
        "total_errors": failures,
    }
    write_json(args.output_dir / "summary.json", output)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
