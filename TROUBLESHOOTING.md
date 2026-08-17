# Troubleshooting — Qwen3.8-27B on GB10

## `/health` lies

vLLM's health endpoint can return ready while EngineCore is dead. After every boot, send one real `chat/completions` before taking any number.

## Unified memory not released

Stopping a container can leave 50–90 GiB of ghost allocations. If `MemAvailable` is under ~60 GiB, do not boot:

```
docker stop -t 30 <name>
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
```

If that does not free it, reboot the node.

## `triton_attn` is not optional for FP8 KV

On sm_121, FlashAttention cannot serve FP8 KV. Without `--attention-backend triton_attn` the 3.8 NVFP4 boot does not have a working attention backend for this quant.

## NVFP4-KV / TurboQuant-KV

`--kv-cache-dtype nvfp4` is accepted by the API server and rejected by EngineCore:

```
TRITON_ATTN ... Reason: ['kv_cache_dtype not supported']
```

Raise `--gpu-memory-utilization` instead. Stay ≤ 0.80.

## DSpark on vLLM + Unsloth NVFP4

**RadixArk** (`RadixArk/Qwen3.8-27B-DSpark`) still dies:

```
ValueError: hf_overrides must be a dict for get_quant_config
```

The draft `ModelConfig` is built with `hf_overrides=None`. Passing `'{}'` on the target does not fix the draft. Their card is SGLang + official FP8.

**Doopeworld** (`Doopeworld/Qwen3.8-27B-DSpark-vLLM`) boots. Use `scripts/boot-bakeer-dspark.sh`. If EngineCore says free memory < desired util (0.85 needs ~103 GiB), stop ASR/TTS (**do not** delete those images) and `clear-ram`. Do not delete Comfy / H3 / Krea weights to make space — drop NVFP4 only if you must, and redownload later.

## ffmpeg missing in stock eugr

Text generation works. Vision / video wants ffmpeg. House image `spark-vllm:qwen38` adds 6.1.1.

## Do not take 3.6 recipes as 3.8 gospel

Different architecture, different checkpoint, different image. Re-measure.
