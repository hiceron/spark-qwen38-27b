#!/usr/bin/env python3
"""Concurrency benchmark against :8888 — decode AND prefill (llama-bench style).

Modes
  decode (default):    short prompt, --max-tokens generated with ignore_eos.
                       Aggregate = total completion tokens / batch wall time.
  prefill/mixed:       --prompt-tokens P builds a filler prompt of ~P tokens
                       (auto-calibrated against usage.prompt_tokens, unique
                       leading tag defeats prefix caching). --max-tokens 1 =
                       pure prefill (pp test); --max-tokens 256 = mixed
                       (pp+tg, realistic agent turn). Streaming is used to
                       measure true TTFT per request.

Usage examples (run ON the Spark):
  python3 concurrency_bench.py unsloth35b --levels 1,2,4,8,12,16 --max-tokens 256
  python3 concurrency_bench.py unsloth35b --prompt-tokens 8192 --max-tokens 1 --levels 1,2,4,8,16
  python3 concurrency_bench.py unsloth35b --prompt-tokens 8192 --max-tokens 256 --levels 4,8,12,16
"""
from __future__ import annotations

import argparse
import json
import random
import string
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8888/v1/chat/completions"

FILLER_SENTENCE = (
    "The quick brown fox jumps over the lazy dog while seventeen engineers "
    "review the deployment pipeline and discuss cache invalidation strategies. "
)


def rand_tag(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def build_prompt(chars: int) -> str:
    # Unique leading tag so vLLM prefix caching cannot reuse another request's KV.
    body = (FILLER_SENTENCE * (chars // len(FILLER_SENTENCE) + 1))[:chars]
    return f"Session {rand_tag()}: summarize the following notes.\n\n{body}"


def request_stream(model: str, prompt: str, max_tokens: int, timeout: int = 1800):
    """Returns (wall_s, ttft_s, prompt_tokens, completion_tokens)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    pt = ct = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ttft is None and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    ttft = time.perf_counter() - t0
            usage = chunk.get("usage")
            if usage:
                pt = int(usage.get("prompt_tokens") or 0)
                ct = int(usage.get("completion_tokens") or 0)
    wall = time.perf_counter() - t0
    return wall, (ttft if ttft is not None else wall), pt, ct


def calibrate_chars(model: str, target_tokens: int) -> int:
    """Find char count whose prompt tokenizes to ~target_tokens (one probe)."""
    guess = target_tokens * 4
    _, _, pt, _ = request_stream(model, build_prompt(guess), 1)
    if pt <= 0:
        return guess
    chars = int(guess * target_tokens / pt)
    print(f"calibration: {guess} chars -> {pt} prompt_tokens; using {chars} chars for ~{target_tokens} tok")
    return chars


def run_level(model: str, c: int, chars: int | None, max_tokens: int):
    results: list[tuple | None] = [None] * c

    def worker(i: int):
        prompt = build_prompt(chars) if chars else (
            f"Session {rand_tag()}: explain in detail how a hash map works, "
            "including collision handling and resizing, with code examples."
        )
        try:
            results[i] = request_stream(model, prompt, max_tokens)
        except Exception as e:
            print(f"    request {i} FAILED: {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    done = [r for r in results if r]
    tot_pt = sum(r[2] for r in done)
    tot_ct = sum(r[3] for r in done)
    ttfts = sorted(r[1] for r in done)
    p95 = ttfts[max(0, int(len(ttfts) * 0.95) - 1)] if ttfts else 0.0
    return {
        "wall": wall,
        "prefill_agg": tot_pt / wall if wall else 0.0,
        "decode_agg": tot_ct / wall if wall else 0.0,
        "ttft_mean": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "ttft_p95": p95,
        "ok": len(done),
        "tot_pt": tot_pt,
        "tot_ct": tot_ct,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="0 = decode mode (short prompt); >0 = prefill/mixed mode")
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    levels = [int(x) for x in args.levels.split(",")]

    chars = None
    if args.prompt_tokens:
        chars = calibrate_chars(args.model, args.prompt_tokens)

    mode = "decode" if not args.prompt_tokens else (
        "prefill" if args.max_tokens <= 1 else "mixed"
    )
    print(f"model={args.model} mode={mode} prompt_tokens~{args.prompt_tokens or 'short'} "
          f"max_tokens={args.max_tokens} reps={args.reps}")
    run_level(args.model, 1, chars, min(args.max_tokens, 16))  # warmup
    print("conc | wall_s | prefill tok/s | decode tok/s | TTFT mean | TTFT p95 | ok")
    for c in levels:
        for _ in range(args.reps):
            r = run_level(args.model, c, chars, args.max_tokens)
            print(f"{c:4d} | {r['wall']:6.1f} | {r['prefill_agg']:13.0f} | "
                  f"{r['decode_agg']:12.1f} | {r['ttft_mean']:9.2f} | "
                  f"{r['ttft_p95']:8.2f} | {r['ok']}/{c}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
