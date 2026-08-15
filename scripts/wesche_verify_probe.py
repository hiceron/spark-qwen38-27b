#!/usr/bin/env python3
"""Wesche-style 400-token thinking-off chat probe (TTFT-inclusive)."""
import json, sys, time, urllib.request

ep = sys.argv[1].rstrip("/")
model = sys.argv[2] if len(sys.argv) > 2 else json.load(
    urllib.request.urlopen(f"{ep}/v1/models", timeout=30)
)["data"][0]["id"]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Write a 300-word story about a lighthouse keeper."}],
    "max_tokens": 400,
    "temperature": 0.7,
    "top_p": 0.8,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode()
t0 = time.time()
r = json.load(urllib.request.urlopen(
    urllib.request.Request(f"{ep}/v1/chat/completions", body, {"Content-Type": "application/json"}),
    timeout=600,
))
dt = time.time() - t0
u = r["usage"]
msg = r["choices"][0]["message"]
print(f"wesche-style thinking-off chat: {u['completion_tokens']} tok in {dt:.1f}s = {u['completion_tokens']/dt:.1f} tok/s (incl TTFT)")
print("finish", r["choices"][0]["finish_reason"], "reasoning_chars", len(msg.get("reasoning_content") or ""))
