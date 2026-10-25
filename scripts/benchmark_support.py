#!/usr/bin/env python3
"""Shared fail-closed HTTP benchmark helpers."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class RequestResult:
    request_id: str
    target_context_tokens: int
    local_prompt_tokens: int
    output_budget: int
    context_tokens: int = 0
    output_tokens: int = 0
    content_events: int = 0
    t_request_build_ms: float = 0.0
    t_headers_ms: float = 0.0
    t_ttft_ms: float = 0.0
    t_sse_json_parse_ms: float = 0.0
    t_total_ms: float = 0.0
    finish_reason: str | None = None
    saw_done: bool = False
    error: str | None = None
    sse_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fresh_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing existing output directory: {path}")
    path.mkdir(parents=True)


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",")]
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"expected positive comma-separated integers: {value}")
    return parsed


def parse_phases(value: str) -> list[tuple[int, int, float]]:
    phases = []
    for item in value.split(","):
        concurrency, context, duration = item.split(":")
        phases.append((int(concurrency), int(context), float(duration)))
    if not phases or any(
        min(concurrency, context, duration) <= 0
        for concurrency, context, duration in phases
    ):
        raise ValueError(f"invalid phases: {value}")
    return phases


def templated_token_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers 5.x can return BatchEncoding even when return_dict was not
    # requested.  len(BatchEncoding) is the number of fields, not tokens.
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("chat template unexpectedly returned a token batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or any(
        not isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError(
            f"chat template returned unsupported token IDs: {type(token_ids)!r}"
        )
    return token_ids


def build_messages(tokenizer: Any, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    system = "You are a precise coding assistant. Analyze code and explain tradeoffs."
    unit = (
        " Review this Python service for performance, correctness, concurrency, "
        "error handling, testing, and maintainability."
    )

    def make(repetitions: int) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": unit * repetitions},
        ]

    low, high = 0, 1
    while len(templated_token_ids(tokenizer, make(high))) < target_tokens:
        high *= 2
        if high > max(1024, target_tokens * 8):
            raise RuntimeError(
                "chat-template filler failed to approach target token count"
            )
    while low + 1 < high:
        middle = (low + high) // 2
        if len(templated_token_ids(tokenizer, make(middle))) <= target_tokens:
            low = middle
        else:
            high = middle
    messages = min(
        (make(low), make(high)),
        key=lambda item: abs(
            len(templated_token_ids(tokenizer, item)) - target_tokens
        ),
    )
    count = len(templated_token_ids(tokenizer, messages))
    return messages, count


async def fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> str:
    async with session.get(base_url.rstrip("/") + "/metrics") as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"metrics HTTP {response.status}: {body[:500]}")
        return body


async def send_chat(
    session: aiohttp.ClientSession,
    url: str,
    request_id: str,
    model: str,
    messages: list[dict[str, str]],
    target_context_tokens: int,
    local_prompt_tokens: int,
    max_tokens: int,
) -> RequestResult:
    request_started = time.perf_counter()
    result = RequestResult(
        request_id=request_id,
        target_context_tokens=target_context_tokens,
        local_prompt_tokens=local_prompt_tokens,
        output_budget=max_tokens,
    )
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result.t_request_build_ms = (time.perf_counter() - request_started) * 1000
    sent_started = time.perf_counter()
    first_content = None
    usage = None
    try:
        async with session.post(url, json=payload) as response:
            result.t_headers_ms = (time.perf_counter() - sent_started) * 1000
            if response.status < 200 or response.status >= 300:
                body = await response.text()
                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                result.sse_lines.append(data)
                if data == "[DONE]":
                    result.saw_done = True
                    continue
                parse_started = time.perf_counter()
                try:
                    chunk = json.loads(data)
                finally:
                    result.t_sse_json_parse_ms += (
                        time.perf_counter() - parse_started
                    ) * 1000
                if chunk.get("error"):
                    raise RuntimeError(f"SSE error: {chunk['error']}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or delta.get("reasoning")
                    if content:
                        result.content_events += 1
                        if first_content is None:
                            first_content = time.perf_counter()
                    if choice.get("finish_reason") is not None:
                        result.finish_reason = choice["finish_reason"]
        if not result.saw_done:
            raise RuntimeError("stream ended without [DONE]")
        if result.finish_reason is None:
            raise RuntimeError("stream ended without finish_reason")
        if usage is None:
            raise RuntimeError("stream ended without usage")
        result.context_tokens = int(usage.get("prompt_tokens", -1))
        result.output_tokens = int(usage.get("completion_tokens", -1))
        if result.context_tokens != local_prompt_tokens:
            raise RuntimeError(
                "prompt usage mismatch: "
                f"local={local_prompt_tokens}, server={result.context_tokens}"
            )
        if result.output_tokens != max_tokens:
            raise RuntimeError(
                f"output usage mismatch: budget={max_tokens}, "
                f"server={result.output_tokens}"
            )
        if first_content is None:
            raise RuntimeError("stream had no non-empty content or reasoning event")
        result.t_ttft_ms = (first_content - sent_started) * 1000
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    result.t_total_ms = (time.perf_counter() - request_started) * 1000
    return result


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_results(path: Path, results: list[RequestResult]) -> None:
    with path.open("w") as output:
        for result in results:
            output.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
