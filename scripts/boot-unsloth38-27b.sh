#!/usr/bin/env bash
# Boot unsloth/Qwen3.8-27B-NVFP4 on eugr/spark-vllm (or a local ffmpeg fork).
# Usage: boot-unsloth38-27b.sh [gpu_util] [max_seqs] [kv=auto|fp8|bf16|nvfp4] [len=262144|524288]
#
# Env:
#   IMAGE          default eugr/spark-vllm:latest
#                  (house daily is a one-line fork spark-vllm:qwen38 with ffmpeg)
#   NAME           default unsloth38-27b
#   ATTN_BACKEND   default triton_attn  (required for FP8 KV on sm_121)
#   SPEC=mtp|dflash|dspark|none   (default mtp)
#   SPEC_K         mtp default 2; dflash default 10; dspark default 5
#   SPEC_JSON      full --speculative-config JSON (wins)
#   DFLASH_MODEL   local path to a DFlash drafter (required if SPEC=dflash)
#   DSPARK_MODEL   default RadixArk/Qwen3.8-27B-DSpark
#   YARN=1         force YaRN even if len=262144
#   BATCHED_TOKENS default 32768
#
# Measured 2026-08-15: MTP k=2 is the daily winner. DSpark does not boot on
# Unsloth NVFP4 + vLLM 0.27.2 (draft hf_overrides=None). nvfp4 KV is rejected
# by triton_attn. Do not set util above 0.80.
set -euo pipefail

UTIL="${1:-0.55}"
SEQS="${2:-4}"
KVD="${3:-auto}"
LEN="${4:-262144}"
IMAGE="${IMAGE:-eugr/spark-vllm:latest}"
NAME="${NAME:-unsloth38-27b}"
ATTN_BACKEND="${ATTN_BACKEND:-triton_attn}"
MODEL_ID="${MODEL_ID:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED="${SERVED_MODEL_NAME:-unsloth27b}"
DFLASH_MODEL="${DFLASH_MODEL:-}"
DSPARK_MODEL="${DSPARK_MODEL:-RadixArk/Qwen3.8-27B-DSpark}"

if [ -n "${DFLASH:-}" ] && [ -z "${SPEC:-}" ]; then
  SPEC=dflash
fi
SPEC="${SPEC:-mtp}"

SPEC_ARGS=()
DRAFTER_MOUNT=()
if [ "$SPEC" = "none" ] || [ -n "${NO_SPEC:-}" ]; then
  echo "Speculative decoding DISABLED"
elif [ -n "${SPEC_JSON:-}" ]; then
  SPEC_ARGS=(--speculative-config "$SPEC_JSON")
  echo "Speculative config: $SPEC_JSON"
elif [ "$SPEC" = "dflash" ]; then
  if [ -z "$DFLASH_MODEL" ] || [ ! -d "$DFLASH_MODEL" ]; then
    echo "ABORT: set DFLASH_MODEL to a local DFlash drafter directory"
    exit 4
  fi
  SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/dflash-drafter\",\"num_speculative_tokens\":${SPEC_K:-10}}")
  DRAFTER_MOUNT=(-v "$DFLASH_MODEL:/models/dflash-drafter:ro")
  echo "Speculative config: dflash k=${SPEC_K:-10} from $DFLASH_MODEL"
elif [ "$SPEC" = "dspark" ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"${DSPARK_MODEL}\",\"num_speculative_tokens\":${SPEC_K:-5}}")
  echo "Speculative config: dspark k=${SPEC_K:-5} model=$DSPARK_MODEL"
else
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K:-2}}")
  echo "Speculative config: mtp k=${SPEC_K:-2}"
fi

# Always a dict: DSpark's get_quant_config crashes if hf_overrides is None.
YARN_ARGS=(--hf-overrides '{}')
EXTRA_ENV_FLAGS=(-e CUTE_DSL_ARCH=sm_121a -e VLLM_FLOAT32_MATMUL_PRECISION=high)
if [ "$LEN" -ge 524288 ] || [ -n "${YARN:-}" ]; then
  EXTRA_ENV_FLAGS+=(-e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1)
  YARN_ARGS=(--hf-overrides '{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":2.0,"original_max_position_embeddings":262144}}}')
  echo "YaRN ON: max-model-len=$LEN factor=2.0"
fi

KV_ARGS=()
if [ "$KVD" != "auto" ]; then
  KV_ARGS=(--kv-cache-dtype "$KVD")
  echo "KV dtype forced: $KVD"
else
  echo "KV dtype: auto (checkpoint fp8 scheme)"
fi

python3 - "$UTIL" <<'PY'
import sys
u = float(sys.argv[1])
if u > 0.80:
    sys.exit("ABORT: gpu-memory-utilization %.2f exceeds house ceiling 0.80" % u)
PY

for old in "$NAME" unsloth38-27b; do
  docker update --restart=no "$old" 2>/dev/null || true
  docker stop -t 30 "$old" 2>/dev/null || true
  docker rm -f "$old" 2>/dev/null || true
done

avail=0
for _ in $(seq 1 24); do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 85 ] && break
  sleep 5
done
echo "MemAvailable before boot: ${avail} GiB"
if [ "$avail" -lt 60 ]; then
  echo "ABORT: unified memory not released (${avail} GiB) — drop_caches, then reboot."
  exit 2
fi

mkdir -p "$HOME/vllm-cache"

docker run -d --name "$NAME" --gpus all --ipc=host --net=host \
  "${EXTRA_ENV_FLAGS[@]}" \
  ${EXTRA_ENV:-} \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/vllm-cache:/root/.cache/vllm" \
  "${DRAFTER_MOUNT[@]}" \
  --entrypoint vllm "$IMAGE" \
  serve "$MODEL_ID" \
  --served-model-name "$SERVED" unsloth38 \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --attention-backend "$ATTN_BACKEND" \
  "${KV_ARGS[@]}" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$LEN" \
  --max-num-seqs "$SEQS" \
  --max-num-batched-tokens "${BATCHED_TOKENS:-32768}" \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --skip-mm-profiling \
  "${SPEC_ARGS[@]}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --default-chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true}' \
  --limit-mm-per-prompt '{"image":8,"video":2}' \
  "${YARN_ARGS[@]}"

echo "$NAME starting (image=$IMAGE util=$UTIL seqs=$SEQS kv=$KVD len=$LEN spec=$SPEC); waiting for health"
for _ in $(seq 1 180); do
  if curl -sf -m 3 http://127.0.0.1:8888/health >/dev/null 2>&1; then
    echo "LLM ready — send one real chat request before trusting /health"
    docker logs "$NAME" 2>&1 | grep -E "Maximum concurrency|GPU KV cache|kv_cache" | tail -20 || true
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "CONTAINER DIED during startup — last logs:"
    docker logs --tail 40 "$NAME" 2>&1
    exit 3
  fi
  sleep 10
done
echo "LLM health check timed out"
docker logs --tail 40 "$NAME" 2>&1
exit 1
