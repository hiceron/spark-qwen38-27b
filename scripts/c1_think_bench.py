#!/usr/bin/env python3
"""C1 throughput: thinking-off (Bakeer) and thinking-on (house daily)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"

FRESH = (
    "Write a complete Python module implementing an LRU cache with TTL expiry, "
    "thread safety, and a decorator API. Include docstrings and type hints. Code only."
)
THINK_PROMPT = (
    "Write a complete, self-contained Python program: a small terminal rogue-like "
    "with a procedurally generated dungeon, FOV, items, a simple combat loop, and "
    "save/load. Put every module in one file. No placeholders."
)


def chat(prompt: str, max_tokens: int, thinking: bool, stream: bool = True) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if not thinking else 0.6,
        "stream": stream,
        "stream_options": {"include_usage": True} if stream else None,
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": True,
        },
    }
    if not stream:
        payload.pop("stream_options")
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    reason = 0
    content = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        if not stream:
            body = json.loads(r.read())
            wall = time.time() - t0
            msg = body["choices"][0]["message"]
            n = int(body["usage"]["completion_tokens"])
            return {
                "tokens": n,
                "wall_s": round(wall, 2),
                "ttft_s": None,
                "tok_s_wall": round(n / wall, 2) if wall else 0,
                "tok_s_decode": None,
                "reason_chars": len(msg.get("reasoning_content") or ""),
                "content_chars": len(msg.get("content") or ""),
            }
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("choices"):
                delta = d["choices"][0].get("delta") or {}
                rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                cc = delta.get("content") or ""
                if rc:
                    reason += len(rc)
                if cc:
                    content += len(cc)
                if ttft is None and (rc or cc):
                    ttft = time.time() - t0
            if d.get("usage"):
                n = int(d["usage"].get("completion_tokens") or 0)
    wall = time.time() - t0
    dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0
    return {
        "tokens": n,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft or 0, 3),
        "tok_s_wall": round(n / wall, 2) if wall else 0,
        "tok_s_decode": round(dec, 2),
        "reason_chars": reason,
        "content_chars": content,
    }


def main() -> None:
    rows = []
    print(f"endpoint {BASE} model {MODEL}", flush=True)
    print("warmup thinking-off 32...", flush=True)
    chat(FRESH, 32, False)
    for label, prompt, ntok, think in (
        ("c1-off-fresh-400", FRESH, 400, False),
        ("c1-on-fresh-400", THINK_PROMPT, 400, True),
        ("c1-on-fresh-4096", THINK_PROMPT, 4096, True),
    ):
        print(f"run {label}...", flush=True)
        rec = chat(prompt, ntok, think)
        rec["label"] = label
        rec["thinking"] = think
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    print("ALL", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
