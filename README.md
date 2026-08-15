# Qwen3.8-27B-NVFP4 on one DGX Spark — native MTP won, the grafts did not

**Blog:** https://hiceron.github.io/spark-qwen38-27b/ · **Author:** [Dawid / @Hiceron2](https://x.com/Hiceron2)

I replaced the house 27B with [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
the day it dropped. It boots. It thinks. The baked MTP head is the daily winner.
The things that were supposed to be faster — a 3.6 DFlash graft, RadixArk DSpark,
NVFP4-KV — either slowed the box or never served a token.

**Everything here is measured on one machine** — GB10 / sm_121, 121 GiB unified,
driver 595.71.05 — same streaming bench, one variable at a time.

This is a **new repo**. The Qwen3.6 NVFP4 work stays at
[spark-nvfp4-lab](https://github.com/hiceron/spark-nvfp4-lab). I do not copy those
numbers across as 3.8 expectations.

Work done with Grok (xAI) and Rem, my personal AI agent.

---

## Standing on other people's work

| Source | What I took |
|---|---|
| **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** | The model — hybrid GDN + vision, native 262k, baked MTP |
| **[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)** | The checkpoint I actually serve |
| **[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)** | The GB10 vLLM image. House daily is a one-line ffmpeg fork of it |
| **[MiaAI](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-RTX-6000)** | `--attention-backend triton_attn` as the GB10 path that serves FP8 KV |
| **[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)** | First public 3.8 DSpark drafter (SGLang + official FP8). I tried it on Unsloth NVFP4 + vLLM; it did not boot |
| **[keys / drowzeys](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark)** | MTP-3 completions bench (thinking off). I reproduced 32.1. That is not the house number |
| **[Wesche](https://github.com/Weschera/Qwen3.8-27B-DGX-Spark-Quant-Ladder)** | Quant ladder + a long thinking-on chat figure. Closer to real use than keys |

Where my numbers differ from theirs, that is tool / prompt / thinking / image —
not a claim that they are wrong.

---

## The setup

| | |
|---|---|
| Box | 1× DGX Spark, GB10 (sm_121), 121 GiB unified, ~273 GB/s |
| Weights | `unsloth/Qwen3.8-27B-NVFP4` snapshot `a767244` · 22 GiB |
| Image | `spark-vllm:qwen38` — eugr vLLM **0.27.2rc1.dev88**, transformers 5.15.0, flashinfer 0.6.18, ffmpeg 6.1.1 |
| Attention | `triton_attn` (required for FP8 KV on sm_121) |
| Bench | `scripts/concurrency_bench.py` — **streaming chat**, thinking on. Reads ~15–25% lower than a non-streaming tool |

---

## The findings

### 1. Native MTP is the daily winner

Same weights, same image, one variable. First real completion on every boot — `/health` lies.

| Spec | Decode c=1 | Decode c=4 agg | KV @262k | Daily? |
|---|---|---|---|---|
| **MTP k=2**, U=0.55 | **17.9 / 18.4** | 55.4 / 66.1 | **3.61×** | **yes** |
| MTP k=3, U=0.55 | 18.2 / 19.6 | 60.8 / 68.9 | 3.47× | spare |
| 3.6 DFlash k=10 graft | 13.1 / 13.3 | 34.4 / 34.8 | 2.40× | no |

k=3 is a hair faster and a hair hungrier. I keep k=2 because the KV pool is the
scarce resource on a 262k house profile. The DFlash number is a 3.6 drafter glued
onto a 3.8 target. It answered. It is just slower.

### 2. Util is the KV lever. NVFP4-KV is not, on this backend

`--kv-cache-dtype nvfp4` exists in 0.27.2. APIServer accepts it. EngineCore:

```
ValueError: Selected backend AttentionBackendEnum.TRITON_ATTN
is not valid for this configuration. Reason: ['kv_cache_dtype not supported']
```

FA2 cannot serve FP8 on GB10, so the backend that makes FP8 work is the one that
rejects NVFP4-KV and TurboQuant-KV. Raising util 0.55 → **0.74** is what moved KV:
**5.91× @262k** (1,549,032 tokens). House ceiling stays **0.80**.

### 3. DSpark exists. It does not boot on this stack

`RadixArk/Qwen3.8-27B-DSpark` is real. Their card is SGLang + `Qwen/Qwen3.8-27B-FP8`.
On Unsloth NVFP4 + vLLM 0.27.2 the engine starts loading both weights, then dies
in `get_draft_quant_config` (`hf_overrides` is `None` on the draft `ModelConfig`).
No tok/s from me — it never served a token. There is no `z-lab/Qwen3.8-27B-DFlash`.

### 4. The house number is thinking-on ~18. 32 is a bench I will not quote as daily

Thinking stays on. Without it this model is the wrong 27B.

I replayed keys' completions bench (temp 0, thinking off, decode-only) so I would
know what 32 is. I got **26.4** at k=2 and **32.1** at k=3 — their method, this box.
That is not Hermes. That is not agents. I do not use it.

**Usable, thinking on, streaming chat:** 17.9–18.4 (k=2) · 18.2–19.6 (k=3).

Wesche's **23.7 thinking-on** would be the real win (~25% up). I replayed a
4096-token xhigh coding stream on this box: **17.2 tok/s**, all reasoning.
Not reproduced. His 75k run and official nightly image are still open.

---

## Daily recipe (coding-solo)

```bash
docker run -d --name unsloth38-27b --gpus all --ipc=host --net=host \
  -e CUTE_DSL_ARCH=sm_121a \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/vllm-cache:/root/.cache/vllm \
  --entrypoint vllm eugr/spark-vllm:latest \
  serve unsloth/Qwen3.8-27B-NVFP4 \
  --served-model-name unsloth27b \
  --host 0.0.0.0 --port 8888 --trust-remote-code \
  --quantization compressed-tensors \
  --attention-backend triton_attn \
  --gpu-memory-utilization 0.74 --max-model-len 262144 \
  --max-num-seqs 4 --max-num-batched-tokens 32768 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --skip-mm-profiling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice --hf-overrides '{}'
```

Voice coexist uses the same flags at **U=0.55**. Wrapper: `scripts/boot-unsloth38-27b.sh`.
Never set util above 0.80 on unified memory.

---

## Contents

| File | |
|---|---|
| [DEPLOY.md](DEPLOY.md) | Copy-paste recipes |
| [QWEN38-RUNLOG.md](QWEN38-RUNLOG.md) | Every boot, including the two that died |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Failures and the zombie `/health` rule |
| [scripts/](scripts/) | Boot + streaming bench |

## See also

- [spark-nvfp4-lab](https://github.com/hiceron/spark-nvfp4-lab) — Qwen3.6 NVFP4 (separate story)
- [spark-ds4-flash-tuning](https://github.com/hiceron/spark-ds4-flash-tuning)
- [spark-minimax-h3-lab](https://github.com/hiceron/spark-minimax-h3-lab)

## License

MIT for the scripts. Measurements are facts — use them, and please say where they came from.
