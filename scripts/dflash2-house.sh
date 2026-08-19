#!/usr/bin/env bash
# House DFlash2 on :8888. Coding only. No 100g docker cap.
# Refuses if trellis2 is running (measured CUDA OOM on HDRI).
# Usage: MEM_FRACTION=0.70 bash dflash2-house.sh
set -euo pipefail
ROOT="${ROOT:-$HOME/Desktop/qwen38-r0b0tlab-sglang}"
CKPT="${CKPT:-$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121}"
DRAFT="${DRAFT:-$HOME/models/z-lab/Qwen3.8-27B-DFlash2}"
IMAGE="${IMAGE:-qwen38-27b-sglang-dflash2-sm121:house}"
NAME="${NAME:-qwen3.8-27b-sglang}"
PORT="${PORT:-8888}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MAX_REQ="${MAX_CONCURRENT_REQUESTS:-8}"

if docker inspect -f '{{.State.Running}}' trellis2 2>/dev/null | grep -qx true; then
  echo "trellis2 is up — DFlash2 cannot coexist. Stop TRELLIS first or use switch-mode.sh coding."
  exit 1
fi

mkdir -p "$CKPT" "$DRAFT"

python3 - <<PY
from huggingface_hub import snapshot_download
import os
ckpt=os.environ.get("CKPT","$CKPT")
draft=os.environ.get("DRAFT","$DRAFT")
print("downloading body if needed", flush=True)
snapshot_download("r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121", local_dir="$CKPT")
print("downloading DFlash2 if needed", flush=True)
snapshot_download("z-lab/Qwen3.8-27B-DFlash2", local_dir="$DRAFT")
print("downloads ok", flush=True)
PY

python3 - "$DRAFT/config.json" <<'PY'
import json,sys
cfg=json.load(open(sys.argv[1]))
if cfg.get("architectures") != ["DFlash2DraftModel"]:
    raise SystemExit("draft is not DFlash2DraftModel")
df=cfg.get("dflash_config") or {}
if not (df.get("conv_kernel_size") and df.get("selector_rank")):
    raise SystemExit("draft config lacks DFlash2 conv/selector fields")
print("draft config ok")
PY

# Overlay on the house image (already local). Do not pull the 38G campaign digest.
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  tmp=$(mktemp -d)
  cp -a "$ROOT/docker/." "$tmp/"
  cat > "$tmp/Dockerfile.house" <<'DF'
FROM lmsysorg/sglang:qwen38-27b
COPY python/sglang/kernels/ops/speculative/dflash.py /sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py
COPY python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
COPY python/sglang/srt/models/dflash.py /sgl-workspace/sglang/python/sglang/srt/models/dflash.py
COPY python/sglang/srt/speculative/dflash2_compat.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash2_compat.py
COPY python/sglang/srt/speculative/dflash_utils.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
COPY python/sglang/srt/speculative/dflash_worker_v2.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py
DF
  docker build -f "$tmp/Dockerfile.house" -t "$IMAGE" "$tmp"
  rm -rf "$tmp"
fi

# Only remove LLM containers. Never trellis2.
docker rm -f "$NAME" qwen38-sglang qwen38-dspark-vllm >/dev/null 2>&1 || true

docker run -d --name "$NAME" --gpus all --ipc=host --net=host \
  --cpus 14 --shm-size 32g \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  -v "$CKPT:/model:ro" \
  -v "$DRAFT:/draft:ro" \
  "$IMAGE" \
  sglang serve \
  --trust-remote-code \
  --model-path /model \
  --served-model-name qwen3.8-27b-sglang \
  --host 0.0.0.0 --port "$PORT" \
  --attention-backend flashinfer \
  --kv-cache-dtype auto \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --context-length 262144 \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_REQ" \
  --disable-prefill-cuda-graph \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /draft \
  --speculative-num-draft-tokens "${DFLASH_BLOCK_SIZE:-8}"

echo "launched $NAME dflash2 mem=$MEM_FRACTION port=$PORT"
echo -n "waiting "
for i in $(seq 1 180); do
  if curl -sf -m 4 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "ready http://127.0.0.1:${PORT}/v1"
    docker logs "$NAME" 2>&1 | grep -E "mem_fraction|KV|max_running|context_len|DFLASH|Error" | tail -25
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo
    echo "container died"
    docker logs "$NAME" 2>&1 | tail -80
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo
echo "timeout"
docker logs "$NAME" 2>&1 | tail -80
exit 1
