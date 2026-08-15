#!/usr/bin/env python3
"""Wesche-style thinking-ON streamed chat. Long decode, not a 256-token burst.

Measures both:
  wall = completion_tokens / total_s          (his 75k/53min style)
  decode = (n-1) / (total - ttft)

Usage: python wesche_think_on_bench.py http://127.0.0.1:8888 unsloth27b [max_tokens]
"""
import json, sys, time, urllib.request

ep = sys.argv[1].rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else "unsloth27b"
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 4096

PROMPT = (
    "Write a complete, self-contained Python program: a small terminal rogue-like "
    "with a procedurally generated dungeon, FOV, items, a simple combat loop, and "
    "save/load. Put every module in one file. No placeholders. Include a short "
    "design note at the top explaining the map generator."
)

body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": max_tokens,
    "temperature": 0.6,
    "top_p": 0.95,
    "stream": True,
    "stream_options": {"include_usage": True},
    "chat_template_kwargs": {
        "enable_thinking": True,
        "reasoning_effort": "xhigh",
    },
}).encode()

t0 = time.time()
ttft = None
n = 0
reason_chars = 0
content_chars = 0
req = urllib.request.Request(
    f"{ep}/v1/chat/completions",
    body,
    {"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=3600) as r:
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
                reason_chars += len(rc)
            if cc:
                content_chars += len(cc)
            if ttft is None and (rc or cc):
                ttft = time.time() - t0
        if d.get("usage"):
            n = int(d["usage"].get("completion_tokens") or 0)

wall = time.time() - t0
dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0
print(json.dumps({
    "model": model,
    "max_tokens": max_tokens,
    "completion_tokens": n,
    "wall_s": round(wall, 1),
    "ttft_s": round(ttft or 0, 3),
    "tok_s_wall": round(n / wall, 1) if wall else 0,
    "tok_s_decode": round(dec, 1),
    "reasoning_chars": reason_chars,
    "content_chars": content_chars,
}, indent=2))
