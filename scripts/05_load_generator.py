#!/usr/bin/env python3
"""Run closed-batch synthetic chat load with strict result validation."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer

from benchmark_support import (
    build_messages,
    fetch_metrics,
    fresh_directory,
    parse_phases,
    send_chat,
    write_json,
    write_results,
)


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--phases", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    fresh_directory(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    phases = parse_phases(args.phases)
    prepared = {}
    for _, target, _ in phases:
        messages, count = build_messages(tokenizer, target)
        if count + args.max_tokens > args.max_model_len:
            raise ValueError(
                f"target {target}: prompt={count} + output={args.max_tokens} "
                f"exceeds max model length {args.max_model_len}"
            )
        prepared[target] = (messages, count)

    base_url = args.api_url.split("/v1/", 1)[0]
    timeout = aiohttp.ClientTimeout(total=600)
    connector = aiohttp.TCPConnector(limit=max(item[0] for item in phases) * 2)
    summaries = []
    total_errors = 0
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for phase_index, (concurrency, target, duration) in enumerate(phases):
            name = f"p{phase_index}-c{concurrency}-ctx{target}"
            phase_dir = args.output_dir / name
            phase_dir.mkdir()
            (phase_dir / "before.prom").write_text(
                await fetch_metrics(session, base_url)
            )
            messages, count = prepared[target]
            results = []
            started = time.perf_counter()
            batch_index = 0
            while time.perf_counter() - started < duration:
                batch = await asyncio.gather(
                    *(
                        send_chat(
                            session,
                            args.api_url,
                            f"{name}-b{batch_index}-r{rank}",
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
                batch_index += 1
            elapsed = time.perf_counter() - started
            (phase_dir / "after.prom").write_text(
                await fetch_metrics(session, base_url)
            )
            write_results(phase_dir / "requests.jsonl", results)
            valid = [item for item in results if item.error is None]
            errors = len(results) - len(valid)
            total_errors += errors
            summary = {
                "concurrency": concurrency,
                "context_target": target,
                "local_prompt_tokens": count,
                "planned_duration_s": duration,
                "elapsed_s": elapsed,
                "count": len(results),
                "success_count": len(valid),
                "error_count": errors,
                "throughput_req_s": len(valid) / elapsed,
                "throughput_output_tok_s": sum(
                    item.output_tokens for item in valid
                )
                / elapsed,
            }
            write_json(phase_dir / "summary.json", summary)
            summaries.append(summary)

    write_json(
        args.output_dir / "load_test_results.json",
        {"schema_version": 2, "phases": summaries, "total_errors": total_errors},
    )
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
