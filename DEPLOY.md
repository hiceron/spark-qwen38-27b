# Deploy — Qwen3.8-27B-NVFP4 on one DGX Spark

**Universal rules** (each learned the hard way):

1. **Never set `--gpu-memory-utilization` above 0.80.** The OS lives in the same 121 GiB.
2. Between model switches: stop the old container, `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches`, then boot.
3. After every boot, send **one real chat request**. `/health` can pass over a dead engine.
4. Speculation scratch sits *outside* `gpu-memory-utilization`. `util` × `max-num-seqs` is a joint budget.

## Coding-solo — DSpark k=14, U=0.85 (2026-08-17)

Voice **off**. ASR/TTS images stay on disk. Needs ~103 GiB free (`clear-ram` after `voice-stop`).

```bash
GMU=0.85 K=14 ./scripts/boot-bakeer-dspark.sh
```

Laptop: `03-models/qwen38-sglang/4-vLLM-Coding.bat`. Hermes id `unsloth27b`.

Measured C1 on this box: edit-off **75.3 / 78.5**, brief-think-on **44.1 / 72.4**, unconstrained think-on **16.7–24.1**.

## Live / previous daily — MTP k=2, 256k, coding-solo U=0.74

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
| `Doopeworld/Qwen3.8-27B-DSpark-vLLM` k=14 | Boots. Coding-solo winner. See README §3 |
| `--kv-cache-dtype nvfp4` | Rejected by `triton_attn` |

## keys MTP-3 recipe (not yet my daily)

Published at [drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark). Their 32 is thinking-off `/v1/completions`. I reproduced 32.1. **House does not use that number** — thinking stays on, and that path is ~18 tok/s. Do not run their util 0.90 next to voice or ComfyUI.

## Wesche ladder

Published at [Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder](https://github.com/Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder). Their NVFP4 daily is MTP-3, `triton_attn`, fp8 KV, util 0.80, `--reasoning-parser qwen3`. Their 23.7 tok/s is a long streamed chat with thinking on — different basis than keys' 31.7.
