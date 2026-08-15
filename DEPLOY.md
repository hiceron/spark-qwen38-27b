# Deploy — Qwen3.8-27B-NVFP4 on one DGX Spark

**Universal rules** (each learned the hard way):

1. **Never set `--gpu-memory-utilization` above 0.80.** The OS lives in the same 121 GiB.
2. Between model switches: stop the old container, `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`, then boot.
3. After every boot, send **one real chat request**. `/health` can pass over a dead engine.
4. Speculation scratch sits *outside* `gpu-memory-utilization`. `util` × `max-num-seqs` is a joint budget.

## Daily — MTP k=2, 256k, coding-solo U=0.74

See [README.md](README.md#daily-recipe-coding-solo). Voice coexist: same command, `--gpu-memory-utilization 0.55`.

Wrapper with MTP / DFlash-graft / DSpark / YaRN switches:

```bash
# voice coexist
./scripts/boot-unsloth38-27b.sh 0.55 4 auto 262144

# coding-solo
./scripts/boot-unsloth38-27b.sh 0.74 4 auto 262144

# MTP k=3
SPEC_K=3 ./scripts/boot-unsloth38-27b.sh 0.55 4 auto 262144
```

`IMAGE` defaults to `eugr/spark-vllm:latest`. House daily is `IMAGE=spark-vllm:qwen38` (ffmpeg added). `--attention-backend triton_attn` is required for FP8 KV on GB10.

## What I tried that is not daily

| Attempt | Result |
|---|---|
| 3.6 DFlash graft, k=10 | Boots. 13 tok/s. Smaller KV. Not daily |
| `RadixArk/Qwen3.8-27B-DSpark` on vLLM | Dies in `get_draft_quant_config` |
| `--kv-cache-dtype nvfp4` | Rejected by `triton_attn` |

## keys MTP-3 recipe (not yet my daily)

Published at [drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark). Their champion is MTP-3, **util 0.90**, `eugr/spark-vllm-b12x:nightly-20260813`, port 8078, decode-only `/v1/completions` bench. **0.90 is above the house ceiling** — do not run it on a box that also holds voice or ComfyUI.

## Wesche ladder

Published at [Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder](https://github.com/Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder). Their NVFP4 daily is MTP-3, `triton_attn`, fp8 KV, util 0.80, `--reasoning-parser qwen3`. Their 23.7 tok/s is a long streamed chat with thinking on — different basis than keys' 31.7.
