#!/usr/bin/env python3
"""Think-on prose + code + short prefill/TTFT. House clock."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b-sglang"

PROSE = (
    "Write a long essay on the history of computing from Babbage to modern GPUs. "
    "Cover mechanical engines, vacuum tubes, transistors, VLSI, and CUDA. "
    "No bullet lists. Continuous prose."
)
CODE = (
    "Write a complete, self-contained Python program: a small terminal rogue-like "
    "with a procedurally generated dungeon, FOV, items, a simple combat loop, and "
    "save/load. Put every module in one file. No placeholders."
)
PREFILL = "Repeat the word spark. " * 400


def chat(prompt: str, max_tokens: int, thinking: bool) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.6 if thinking else 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": thinking,
            "preserve_thinking": True,
        },
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    n = 0
    prompt_n = 0
    reason = 0
    content = 0
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
                    reason += len(rc)
                if cc:
                    content += len(cc)
                if ttft is None and (rc or cc):
                    ttft = time.time() - t0
            if d.get("usage"):
                n = int(d["usage"].get("completion_tokens") or 0)
                prompt_n = int(d["usage"].get("prompt_tokens") or 0)
    wall = time.time() - t0
    dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0
    return {
        "tokens": n,
        "prompt_tokens": prompt_n,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft or 0, 3),
        "tok_s_wall": round(n / wall, 2) if wall else 0,
        "tok_s_decode": round(dec, 2),
        "prefill_tok_s": round(prompt_n / ttft, 1) if ttft and prompt_n else 0,
        "reason_chars": reason,
        "content_chars": content,
    }


def main() -> None:
    print(f"endpoint {BASE} model {MODEL}", flush=True)
    print("warmup think-off 32...", flush=True)
    chat("Say hi.", 32, False)
    rows = []
    for label, prompt, ntok, think in (
        ("prefill-probe", PREFILL, 8, False),
        ("c1-on-prose-800", PROSE, 800, True),
        ("c1-on-code-800", CODE, 800, True),
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
