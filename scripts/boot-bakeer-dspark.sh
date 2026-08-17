#!/usr/bin/env bash
# Boot Bakeer's DSpark-on-vLLM recipe against house Unsloth NVFP4.
# Does NOT delete TTS / ASR / Comfy / image weights. Stops SGLang only.
# Drafter must already be in HF cache (no huggingface-cli required).
set -euo pipefail
# shellcheck disable=SC1090
source "$HOME/.bashrc" 2>/dev/null || true

IMAGE="${IMAGE:-spark-vllm:qwen38}"
NAME="${NAME:-qwen38-dspark-vllm}"
PORT="${PORT:-8888}"
MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
DRAFTER="${DRAFTER:-Doopeworld/Qwen3.8-27B-DSpark-vLLM}"
SERVED="${SERVED:-unsloth27b}"
SERVED_ALIAS="${SERVED_ALIAS:-qwen3.8-27b}"
GMU="${GMU:-0.85}"
K="${K:-14}"
MAX_LEN="${MAX_LEN:-262144}"
MAX_BATCHED="${MAX_BATCHED:-16384}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/vllm-cache}"

echo "=== stop SGLang (leave ASR/TTS/Comfy images and weights) ==="
bash "$HOME/Desktop/Qwen3.8-27B-SGLang-DGX-Spark/stop.sh" 2>/dev/null || true
docker rm -f qwen3.8-27b-sglang 2>/dev/null || true
docker rm -f unsloth38-27b 2>/dev/null || true
docker rm -f "$NAME" 2>/dev/null || true

DRAFT_DIR="$HF_CACHE/hub/models--${DRAFTER//\//--}"
if [[ ! -d "$DRAFT_DIR" ]]; then
  echo "missing drafter cache: $DRAFT_DIR"
  echo "download with: python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('$DRAFTER')\""
  exit 1
fi
echo "=== drafter present: $DRAFT_DIR ==="
mkdir -p "$VLLM_CACHE"

SPEC_CFG="{\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}"

echo "starting $NAME :: $MODEL :: dspark k=$K gmu=$GMU image=$IMAGE"
# Flags match 0xBakeer/Qwen3.8-27B-4-bit-on-a-single-DGX-Spark serve.sh.
# House extras (triton_attn / compressed-tensors / hf-overrides) omitted on purpose.
docker run -d --name "$NAME" --gpus all --ipc=host --net=host \
  -e HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  --entrypoint vllm \
  "$IMAGE" \
  serve "$MODEL" \
  --served-model-name "$SERVED" ${SERVED_ALIAS:+"$SERVED_ALIAS"} \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GMU" \
  --max-num-batched-tokens "$MAX_BATCHED" \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt.image 2 \
  --limit-mm-per-prompt.video 0 \
  --speculative-config "$SPEC_CFG"

echo -n "waiting for readiness on :$PORT "
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo
    echo "ready http://127.0.0.1:${PORT}/v1  model=$SERVED"
    docker logs "$NAME" 2>&1 | grep -E "GPU KV cache|Maximum concurrency|KV Cache is allocated|Error" | tail -20
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo
    echo "container exited:"
    docker logs "$NAME" 2>&1 | tail -60
    exit 1
  fi
  echo -n "."
  sleep 5
done
echo
echo "timed out"
docker logs "$NAME" 2>&1 | tail -60
exit 1
