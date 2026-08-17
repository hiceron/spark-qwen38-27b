#!/usr/bin/env python3
"""C1: brief thinking then Bakeer edit-heavy body (tests 50+ with thinking on)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"


def _source(n_classes: int = 45) -> str:
    src = '"""Inventory service helpers."""\nfrom dataclasses import dataclass, field\n'
    for i in range(1, n_classes + 1):
        src += f'''

@dataclass
class Item{i}:
    """Represents inventory item {i}."""
    sku: str
    quantity: int = 0
    price_cents: int = 0
    tags: list[str] = field(default_factory=list)

    def restock(self, n: int) -> None:
        """Add n units to the on-hand quantity."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.quantity += n

    def total_value(self) -> int:
        """Return the total value of this line in cents."""
        return self.quantity * self.price_cents
'''
    return src


PROMPT = (
    "Think in at most two short sentences, then immediately output the COMPLETE "
    "modified file and nothing else. Add a `discount(self, pct: int) -> int` method "
    "to EVERY Item class, returning the discounted total value.\n\n```python\n"
    + _source()
    + "\n```"
)


def run(max_tokens: int, label: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {
            "enable_thinking": True,
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
    wall = time.time() - t0
    dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0
    rec = {
        "label": label,
        "thinking": True,
        "tokens": n,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft or 0, 3),
        "tok_s_wall": round(n / wall, 2) if wall else 0,
        "tok_s_decode": round(dec, 2),
        "reason_chars": reason,
        "content_chars": content,
    }
    print(json.dumps(rec), flush=True)
    return rec


def main() -> None:
    print(f"endpoint {BASE} model {MODEL}", flush=True)
    print("warmup...", flush=True)
    run(32, "warmup")
    rows = [run(3000, "c1-briefthink-edit-3000"), run(3000, "c1-briefthink-edit-3000-b")]
    print("ALL", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
