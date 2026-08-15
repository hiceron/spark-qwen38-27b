#!/usr/bin/env python3
"""Decode-only completions bench matching keys/drowzeys methodology.

Hits /v1/completions, temp 0, stream=True, reports (n-1)/(wall-ttft).
Does NOT include their refusal suite. Usage:

  python keys_decode_bench.py http://127.0.0.1:8888 [model]
"""
import json, statistics, sys, time, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
PROMPT = ("The history of computing spans mechanical calculators to modern accelerators. " * 90)[:4000]


def get_model():
    global MODEL
    if MODEL:
        return MODEL
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]
    return MODEL


def stream(prompt, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                break
            d = json.loads(p)
            if d.get("choices") and d["choices"][0].get("text") and ttft is None:
                ttft = time.time() - t0
            if d.get("usage"):
                n = d["usage"].get("completion_tokens") or n
    total = time.time() - t0
    dec = (n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else 0
    return ttft, dec, n, total


def main():
    get_model()
    print(f"model={MODEL} base={BASE}")
    stream("Warm up.", 32)
    ttfts, decs = [], []
    for i in range(5):
        t, d, n, wall = stream(PROMPT, 512)
        ttfts.append(t)
        decs.append(d)
        print(f"rep {i+1}: n={n} ttft={t:.3f}s decode={d:.1f} tok/s wall={wall:.1f}s")
    print(json.dumps({
        "ttft_ms_median": round(statistics.median(ttfts) * 1000),
        "single_stream_tps_median": round(statistics.median(decs), 1),
        "reps": [round(x, 1) for x in decs],
    }, indent=2))


if __name__ == "__main__":
    main()
