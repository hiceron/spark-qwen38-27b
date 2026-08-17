#!/usr/bin/env python3
"""C1 thinking-on on Bakeer's edit-heavy prompt (the 59/75 workload)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8888"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-27b"
METRICS = BASE + "/metrics"


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


EDIT = (
    "Here is a Python module. Add a `discount(self, pct: int) -> int` method to EVERY "
    "Item class, returning the discounted total value. Output the COMPLETE modified "
    "file, nothing else.\n\n```python\n" + _source() + "\n```"
)


def spec_counters() -> dict:
    try:
        body = urllib.request.urlopen(METRICS, timeout=30).read().decode()
    except Exception:
        return {}
    keys = {
        "vllm:spec_decode_num_drafts_total": "drafts",
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted",
    }
    out = {}
    for line in body.splitlines():
        for prefix, name in keys.items():
            if line.startswith(prefix):
                out[name] = float(line.split()[-1])
    return out


def run(prompt: str, max_tokens: int, thinking: bool, label: str) -> dict:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0 if not thinking else 0.6,
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
    before = spec_counters()
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
    after = spec_counters()
    drafts = after.get("drafts", 0) - before.get("drafts", 0)
    draft_tokens = after.get("draft_tokens", 0) - before.get("draft_tokens", 0)
    accepted = after.get("accepted", 0) - before.get("accepted", 0)
    accept_pct = accepted / draft_tokens * 100 if draft_tokens else 0.0
    mean = 1 + accepted / drafts if drafts else 0.0
    dec = (n - 1) / (wall - ttft) if ttft and wall > ttft and n > 1 else 0
    rec = {
        "label": label,
        "thinking": thinking,
        "tokens": n,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft or 0, 3),
        "tok_s_wall": round(n / wall, 2) if wall else 0,
        "tok_s_decode": round(dec, 2),
        "reason_chars": reason,
        "content_chars": content,
        "accept_pct": round(accept_pct, 1),
        "mean_tok_pass": round(mean, 2),
    }
    print(json.dumps(rec), flush=True)
    return rec


def main() -> None:
    print(f"endpoint {BASE} model {MODEL}", flush=True)
    print("warmup thinking-off 32...", flush=True)
    run(EDIT, 32, False, "warmup")
    rows = []
    for label in ("c1-on-edit-3000", "c1-on-edit-3000-b"):
        print(f"run {label}...", flush=True)
        rows.append(run(EDIT, 3000, True, label))
    print("ALL", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
