#!/usr/bin/env bash
set -euo pipefail
CKPT="$HOME/models/r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121"
DRAFT="$HOME/models/z-lab/Qwen3.8-27B-DFlash2"
ROOT="$HOME/Desktop/qwen38-r0b0tlab-sglang"
IMAGE="qwen38-27b-sglang-dflash2-sm121:house"
mkdir -p "$CKPT" "$DRAFT"
echo "DL body"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121', local_dir='$CKPT'); print('body ok')"
echo "DL dflash2"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('z-lab/Qwen3.8-27B-DFlash2', local_dir='$DRAFT'); print('draft ok')"
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
echo PREP_OK
ls -lh "$CKPT"/model-00001-of-00004.safetensors "$DRAFT"/model.safetensors
docker image inspect "$IMAGE" --format '{{.RepoTags}} {{.Size}}'
