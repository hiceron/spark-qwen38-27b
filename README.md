# Qwen3.8-27B-NVFP4 on one DGX Spark — DSpark boots. 75 tok/s is the edit path.

**Blog:** https://hiceron.github.io/spark-qwen38-27b/ · **Author:** [Dawid / @Hiceron2](https://x.com/Hiceron2)

I replaced the house 27B with [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)
the day it dropped. Day-one: baked MTP was the only thing that served tokens.
2026-08-17: [Bakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)'s
Doopeworld DSpark recipe boots on this box. C1 edit-heavy thinking-off is
**75.3 / 78.5 tok/s**. Brief thinking + the same edit is **44.1 / 72.4**.
Long unconstrained thinking is still **16.7–24.1**. That split is the whole story.

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
| **[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)** | First public 3.8 DSpark (SGLang + official FP8). On Unsloth NVFP4 + vLLM it still does not boot |
| **[0xBakeer](https://github.com/0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark)** + **[Doopeworld DSpark](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM)** | The vLLM DSpark that actually serves here. I reproduced his 75 tok/s edit-heavy C1 |
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

### 1. Native MTP is still the live-talk winner. DSpark is the coding-solo winner.

Same weights, same image, one variable. First real completion on every boot — `/health` lies.

| Spec | Decode c=1 | Decode c=4 agg | KV @262k | Daily? |
|---|---|---|---|---|
| **MTP k=2**, U=0.55 | **17.9 / 18.4** | 55.4 / 66.1 | **3.61×** | live + voice |
| MTP k=3, U=0.55 | 18.2 / 19.6 | 60.8 / 68.9 | 3.47× | spare |
| 3.6 DFlash k=10 graft | 13.1 / 13.3 | 34.4 / 34.8 | 2.40× | no |
| **DSpark k=14**, U=0.85 | **75.3 / 78.5** edit-off · **44.1 / 72.4** brief-think | — | 4.43× (1,161,326 tok) | **coding solo** |

k=3 is a hair faster and a hair hungrier than k=2. Live talk stays MTP/SGLang
because voice needs the leftover RAM. The DFlash number is a 3.6 drafter glued
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

### 3. RadixArk DSpark still does not boot. Doopeworld DSpark does.

`RadixArk/Qwen3.8-27B-DSpark` is real. Their card is SGLang + `Qwen/Qwen3.8-27B-FP8`.
On Unsloth NVFP4 + vLLM 0.27.2 the engine starts loading both weights, then dies
in `get_draft_quant_config` (`hf_overrides` is `None` on the draft `ModelConfig`).

[`Doopeworld/Qwen3.8-27B-DSpark-vLLM`](https://huggingface.co/Doopeworld/Qwen3.8-27B-DSpark-vLLM)
is the 5-layer vLLM drafter Bakeer ships. Same Unsloth NVFP4, house image
`spark-vllm:qwen38`, `k=14`, `gmu=0.85`, `--enable-prefix-caching`. It served
tokens. Measured 2026-08-17, C1, this box:

| Workload | Thinking | tok/s | Accept / mean tok/pass |
|---|---|---:|---|
| Fresh 400 (Bakeer `edit_bench`) | off | **27.42** | 18.1 % / 3.54 |
| Edit-heavy 3000 | off | **75.31 / 78.51** | 68.3 % / 10.56 |
| Fresh 400 / 4096 (house stream) | on, unconstrained | **16.69 / 24.06** | all reasoning |
| Edit-heavy 3000 | on, unconstrained | **23.41 / 31.98** | 15–23 % |
| Edit-heavy 3000 | on, brief think | **43.11 / 69.76** wall · **44.1 / 72.4** decode | short think, then copy |

Bakeer published 29.55 fresh / 72.6–75.0 edit at `k=14`, thinking **off**. I
reproduced the 75. The 59 he showed me with thinking on is the brief-think +
edit path, not a 4k free-think. Unconstrained thinking is still ~24 because the
drafter cannot copy a chain of thought that is not already in the prompt.

Voice / TTS / STT / Comfy / H3 weights were **not** deleted. Voice containers
were stopped for the 0.85 sole-occupant boot.

### 4. The house live number is still thinking-on ~18–24. 75 is not that job.

Thinking stays on for live talk. Without it this model is the wrong 27B.

I replayed keys' completions bench (temp 0, thinking off, decode-only) so I would
know what 32 is. I got **26.4** at k=2 and **32.1** at k=3 — their method, this box.
That is not Hermes live. I do not use it as the daily claim.

**Usable, thinking on, unconstrained stream:** 16.7–24.1 (DSpark) · 17.9–19.6 (MTP).
**Usable, thinking on, brief + structured edit:** **44–72** (DSpark k=14).
**Coding-solo, thinking off, edit-heavy:** **75–78**.

Wesche's **23.7 thinking-on** sits on the unconstrained path. I replayed a
4096-token xhigh coding stream on MTP: **17.2 tok/s**. DSpark 4096 think-on:
**24.1**. Still not 50 on that job.

---

## Coding-solo recipe (DSpark k=14, 2026-08-17)

Voice off. Do not run this next to ASR/TTS — 0.85 wants ~103 GiB free.

```bash
# scripts/boot-bakeer-dspark.sh
# Unsloth NVFP4 + Doopeworld DSpark, Bakeer flags, house image
export GMU=0.85 K=14
./scripts/boot-bakeer-dspark.sh
```

Laptop switcher: `03-models/qwen38-sglang/4-vLLM-Coding.bat` → Hermes `unsloth27b`.

## Live-talk recipe (MTP k=2, voice on)

Voice coexist stays MTP / SGLang at **U=0.55**. Wrapper: `scripts/boot-unsloth38-27b.sh`.
Never set util above 0.80 when ASR/TTS or Comfy share the box.

## Previous coding-solo (MTP k=2, kept)

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
