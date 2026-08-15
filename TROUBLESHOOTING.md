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

```
ValueError: hf_overrides must be a dict for get_quant_config
```

The draft `ModelConfig` is built with `hf_overrides=None`. Passing `'{}'` on the target does not fix the draft. RadixArk's card is SGLang + official FP8.

## ffmpeg missing in stock eugr

Text generation works. Vision / video wants ffmpeg. House image `spark-vllm:qwen38` adds 6.1.1.

## Do not take 3.6 recipes as 3.8 gospel

Different architecture, different checkpoint, different image. Re-measure.
