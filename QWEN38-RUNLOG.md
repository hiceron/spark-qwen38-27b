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
- keys MTP-3 (`drowzeys/keys-vLLm.0.27-…`) and Wesche ladder: recipe read, replay in progress. Their 31.7 / 23.7 are not my numbers.
