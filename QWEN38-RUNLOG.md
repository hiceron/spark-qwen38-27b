# Qwen3.8-27B-NVFP4 run log

Started 2026-08-15. Every boot/config/bench goes here. 3.6 numbers stay in FINDINGS.md and are **not** copied as 3.8 expectations.

Hardware: one DGX Spark GB10, 121 GiB unified, same `concurrency_bench.py`.

## 2026-08-15 — prep

- Retired AEON uncensored + Unsloth Qwen3.6-27B/35B-A3B weights. Record: `dgx spark/RETIRED-AEON-AND-QWEN36.md`.
- Kept: Qwen3-ASR, Qwen3-TTS, ComfyUI H3/Krea/LTX, DeepSeek V4 Flash.
- Archived 3.3 GiB `dflash-drafter-qwen36-27b` only for a later graft experiment.
- New boot: `spark-deploy/boot-unsloth38-27b.sh`
- Image target: `eugr/spark-vllm:latest` (no 3.8 recipe in eugr as of 2026-08-14).
- First candidate: `configs/qwen38-27b-mtp-k2-256k.env`

### Image / package preflight

- `eugr/spark-vllm:latest` sha256:a0cb301789ef created 2026-08-14
- python 3.12.3 · **vLLM 0.27.2rc1.dev88** · **transformers 5.15.0** · flashinfer 0.6.18 · cutlass 4.7.0
- ffmpeg **missing** in image (text boot OK; vision/video may need ffmpeg later)
- All version floors passed.

### Weight download

- `unsloth/Qwen3.8-27B-NVFP4` snapshot `a767244` · 22 GiB in HF cache.

### Boot #0 — MTP k=2, 256k, U=0.55, N=4, kv=auto, triton_attn

- Health + real completion OK. Thinking on. Content: `spark qwen ready`.
- KV: **3.61× @262144** (945,803 tokens, 32.98 GiB). Resolved `float8_e4m3fn`.
- Decode streaming: c=1 **17.9 / 18.4** tok/s · c=4 agg **55.4 / 66.1**.
- Prefill ~8k: **1377–1382** tok/s · TTFT 5.9 s.
- File: `results/qwen38-mtp-k2-256k-20260815.txt`

### Boot #1 — MTP k=3, same otherwise

- KV: **3.47× @262144** (909,449 tokens).
- Decode: c=1 **18.2 / 19.6** · c=4 agg **60.8 / 68.9**.
- Slightly faster, slightly less KV. Daily default stays **k=2**.
- File: `results/qwen38-mtp-k3-256k-20260815.txt`

### Boot #2 — DFlash k=10 graft (3.6 drafter)

- **Boots and answers.** First completion 2.26s, content `spark qwen ready`.
- KV: **2.40× @262144** (629,437 tokens) — drafter ate ~1.2× of the MTP pool.
- Decode: c=1 **13.1 / 13.3** tok/s · c=4 agg **34.4 / 34.8**.
- Slower than native MTP, smaller KV. **Not daily.**
- File: `results/qwen38-dflash-k10-256k-20260815.txt`

### Boot #3 — DSpark RadixArk/Qwen3.8-27B-DSpark k=5 — FAIL

- Engine resolves `DSparkDraftModel` and starts loading 21.8 GiB NVFP4 target + 2.6 GiB BF16 draft.
- Dies in `get_draft_quant_config`: `hf_overrides must be a dict`.
- Retry with `--hf-overrides '{}'` on the **target** does not fix the **draft** `ModelConfig` (`hf_overrides=None`).
- Card is SGLang + `Qwen/Qwen3.8-27B-FP8`, not Unsloth NVFP4 + vLLM.
- File: `results/qwen38-dspark-k5-20260815.txt`

### Boot #4 — `--kv-cache-dtype nvfp4` — FAIL

- Flag exists; APIServer accepts it.
- EngineCore: `TRITON_ATTN` / `kv_cache_dtype not supported`.
- FA2 cannot serve FP8 on SM121, so the working FP8 backend is the one that rejects NVFP4-KV / TurboQuant-KV.
- File: `results/qwen38-kv-nvfp4-20260815.txt`

### Sweep — MTP k=2 batched tokens 8k / 16k / 32k (image `spark-vllm:qwen38`)

| batched | KV @262k | decode c=1 | decode c=4 | pp8192 |
|---|---|---|---|---|
| 8192 | 3.68× | 17.2 / 18.2 | 57.5 / 64.9 | 1331 / 1332 |
| 16384 | **3.79×** | 17.4 / 19.9 | 58.0 / 66.4 | 1348 / 1356 |
| 32768 | 3.51× | **18.3 / 19.0** | **62.7 / 65.2** | 1315 / 1316 |

Keep 32768 for long agent turns. File: `results/qwen38-mtp-k2-batched-sweep-20260815.txt`

### Boot #5 — coding-solo U=0.74 N=4 MTP k=2 — KV only

- GPU KV cache: **1,549,032 tokens** · **5.91× @262144**
- Util is the working KV lever vs U=0.55 (3.51–3.79×).
- File: `results/qwen38-mtp-k2-u074-kv-20260815.txt`
- Box left on this recipe (`IMAGE=spark-vllm:qwen38`).

### Image note

House daily image is `spark-vllm:qwen38` — eugr 0.27.2rc1 plus ffmpeg 6.1.1. Stock `eugr/spark-vllm:latest` boots the text path; vision/video wants ffmpeg.

### Still open

- 512k YaRN switcher written, not booted.
- SGLang + RadixArk NVFP4 + DSpark not measured here (published 38.28 tok/s on Spark — their number, their tool).
- Non-streaming rerun of MTP k=2 (this bench reads ~15–25% lower).
- Lab GitHub/Pages: own repo `spark-qwen38-27b` (the 3.6 lab is a separate story).
### Replay — keys decode-only bench (their method, this box)

Same `spark-vllm:qwen38` image. `/v1/completions`, temp 0, `(n-1)/(wall-ttft)`, 512-token gen, median of 5.

| Boot | My median | keys published |
|---|---|---|
| MTP k=2 U=0.74 | **26.4** | 26.3 |
| MTP k=3 U=0.74 | **32.1** | 31.7 |

Did **not** need their util 0.90, `FLASHINFER_CUDA_ARCH_LIST=12.1a`, or `eugr/spark-vllm-b12x:nightly-20260813`. The 32 is MTP-3 + decode-only completions, thinking off.

Wesche `verify.sh`-style 400-token thinking-off chat (TTFT-inclusive): k=2 **19.1** · k=3 **17.2**.

### Replay — Wesche thinking-ON long stream (the number that would actually matter)

23.7 with thinking on is ~25% above our ~18 house number. I undersold that.

Replay on this box, MTP k=3 U=0.74, xhigh, temp 0.6, streamed chat, 4096-token cap, coding prompt:

- **17.2 tok/s** wall and decode (4096 tok / 238.6 s)
- All 4096 stayed in reasoning (`content_chars=0`)
- No 25% jump at 4k

His 75k / 53 min job and `vllm/vllm-openai:nightly` are still untested here.

Box left on **MTP k=3 U=0.74** (KV 5.73× @262k).

## 2026-08-17 — Bakeer DSpark reproduction (goal ≥50 C1)

Voice/TTS/STT/Comfy/H3 **images and weights kept**. Stopped `qwen3-asr` + `qwen3-tts` + `firecrawl-api-1` only. `clear-ram` → MemAvailable **118 GiB**. Did not delete NVFP4 (22G) and did not touch NAS.

### Boot #6 — Doopeworld DSpark k=14, U=0.85 — OK

- Image `spark-vllm:qwen38` (0.27.2rc1), weights `unsloth/Qwen3.8-27B-NVFP4`, drafter `Doopeworld/Qwen3.8-27B-DSpark-vLLM`
- Flags match Bakeer `serve.sh` (prefix cache, batched 16384, no extra `triton_attn` / `hf-overrides`)
- KV: **1,161,326 tokens** · **4.43× @262144**
- First EngineCore death (40.2 GiB free vs 85.14 needed) was ghost RAM + voice still up. After voice-stop + clear-ram it booted.
- RadixArk DSpark + Unsloth still does not boot (`hf_overrides`). This is a different drafter.

### Benches (C1, this box)

Bakeer `edit_bench.py` (thinking **off**, his method):

| label | tok | wall | tok/s | accept | mean tok/pass |
|---|---:|---:|---:|---:|---:|
| warmup | 32 | 2.3s | 13.85 | 13.0% | 2.82 |
| fresh-code | 400 | 14.6s | **27.42** | 18.1% | 3.54 |
| EDIT-heavy | 3000 | 39.8s | **75.31** | 68.3% | 10.56 |
| EDIT-heavy #2 | 3000 | 38.2s | **78.51** | 68.3% | 10.56 |

Bakeer published k=14: 29.55 fresh / 72.63–75.01 edit. Reproduced.

House `c1_think_bench.py` (streaming):

| label | thinking | tok/s wall | tok/s decode | notes |
|---|---|---:|---:|---|
| c1-off-fresh-400 | off | 27.39 | 27.86 | matches Bakeer fresh |
| c1-on-fresh-400 | on | 16.69 | 16.84 | 1576 reason chars, 0 content |
| c1-on-fresh-4096 | on | 24.06 | 24.10 | 13000 reason chars, 0 content |

Edit-heavy with thinking **on** (`c1_think_edit_bench.py`):

| label | tok/s wall | decode | reason/content chars | accept |
|---|---:|---:|---|---:|
| c1-on-edit-3000 | 23.41 | 24.00 | 10474 / 199 | 15.1% |
| c1-on-edit-3000-b | 31.98 | 32.52 | 6800 / 4181 | 23.0% |

Brief-think then edit (`c1_brief_think_edit.py`, thinking **on**):

| label | tok/s wall | decode | reason/content |
|---|---:|---:|---|
| c1-briefthink-edit-3000 | 43.11 | **44.10** | 3699 / 7612 |
| c1-briefthink-edit-3000-b | **69.76** | **72.41** | 639 / 10252 |

Files: `results/bakeer-edit-20260817.txt`, `c1-think-20260817.txt`, `c1-think-edit-20260817.txt`, `c1-briefthink-edit-20260817.txt`.

### House wire

- **4-vLLM-Coding** → this DSpark recipe (voice off, U=0.85, Hermes `unsloth27b`).
- Daily live stays SGLang 0.55 + ASR/TTS. Unconstrained thinking-on is still ~24.

Box left on **DSpark k=14 U=0.85** (`qwen38-dspark-vllm`, served `qwen3.8-27b`).
