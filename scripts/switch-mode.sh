#!/usr/bin/env bash
# House Qwen3.8-27B modes on :8888. One LLM at a time.
#   daily   — SGLang 0.55 + voice. TRELLIS may stay. Not DFlash2.
#   coding  — DFlash2 0.70. Stops TRELLIS + H3 + voice.
#   video   — SGLang 0.40 + H3. Stops TRELLIS. Not DFlash2.
#   trellis — SGLang 0.55 + trellis2. Stops H3 + voice. Not DFlash2.
set -euo pipefail

HOUSE="${HOUSE:-$HOME/Desktop/Qwen3.8-27B-SGLang-DGX-Spark}"
AEON="${AEON:-$HOME/Desktop/aeon}"
NAME="${NAME:-qwen3.8-27b-sglang}"
MODE="${1:-}"

llm_running() {
  docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -qx true
}

llm_args() {
  docker inspect "$NAME" --format '{{join .Args " "}}' 2>/dev/null || true
}

is_dflash() {
  llm_args | grep -q 'speculative-algorithm DFLASH'
}

is_house_mem() {
  llm_args | grep -q "mem-fraction-static ${1}"
}

need_house() {
  local mem="$1"
  if ! llm_running; then
    return 0
  fi
  if is_dflash; then
    return 0
  fi
  if ! is_house_mem "$mem"; then
    return 0
  fi
  return 1
}

stop_h3() {
  bash "$AEON/deploy-voice-stack.sh" comfyui-stop >/dev/null 2>&1 || true
}

stop_voice() {
  bash "$AEON/deploy-voice-stack.sh" voice-stop >/dev/null 2>&1 || true
}

stop_trellis() {
  if docker inspect -f '{{.State.Running}}' trellis2 2>/dev/null | grep -qx true; then
    echo "stopping trellis2 (this mode cannot sit next to TRELLIS)"
    docker stop trellis2 >/dev/null
  fi
}

stop_llm() {
  bash "$HOUSE/stop.sh" || true
  docker rm -f "$NAME" qwen38-dspark-vllm unsloth38-27b qwen38-sglang >/dev/null 2>&1 || true
  bash "$AEON/deploy-voice-stack.sh" clear-ram || true
}

start_house() {
  local mem="$1" conc="$2"
  cd "$HOUSE"
  export MEM_FRACTION_STATIC="$mem" MAX_CONCURRENT_REQUESTS="$conc"
  bash ./start.sh
}

set_mode() {
  echo "$1" > "$AEON/.spark-mode"
  echo "spark-mode=$1"
}

status() {
  echo "---"
  docker ps --format '{{.Names}} {{.Status}}' | grep -E 'qwen3|trellis|comfy|asr|tts' || true
  echo -n "8888: "
  curl -sf -m 4 http://127.0.0.1:8888/v1/models >/dev/null 2>&1 && echo up || echo down
}

case "$MODE" in
  daily)
    echo "mode=daily house SGLang 0.55 + voice (TRELLIS may stay)"
    stop_h3
    if need_house 0.55; then
      stop_llm
      start_house 0.55 4
    else
      echo "house SGLang 0.55 already up"
    fi
    bash "$AEON/deploy-voice-stack.sh" voice
    set_mode sglang-qwen38
    ;;
  coding)
    echo "mode=coding DFlash2 0.70 (TRELLIS off, H3 off, voice off)"
    stop_h3
    stop_trellis
    stop_voice
    if llm_running && is_dflash && is_house_mem 0.70; then
      echo "DFlash2 0.70 already up"
    else
      stop_llm
      export MEM_FRACTION=0.70 MAX_CONCURRENT_REQUESTS=8
      bash "$HOUSE/dflash2-house.sh"
    fi
    set_mode sglang-qwen38-dflash2
    ;;
  video)
    echo "mode=video slim SGLang 0.40 + H3 (TRELLIS off)"
    stop_trellis
    stop_voice
    stop_llm
    start_house 0.40 2
    bash "$AEON/deploy-voice-stack.sh" comfyui-start-h3
    bash "$AEON/deploy-voice-stack.sh" h3-coexist-status || true
    set_mode sglang-qwen38-h3
    ;;
  trellis)
    echo "mode=trellis house SGLang 0.55 + trellis2 (H3 off, voice off, not DFlash2)"
    stop_h3
    stop_voice
    if need_house 0.55; then
      stop_llm
      start_house 0.55 4
    else
      echo "house SGLang 0.55 already up"
    fi
    if ! docker start trellis2 >/dev/null 2>&1; then
      echo "trellis2 container missing — cannot start TRELLIS"
      exit 1
    fi
    echo -n "waiting for TRELLIS :7860 "
    ok=0
    for _ in $(seq 1 36); do
      if curl -sf -m 4 http://127.0.0.1:7860/ >/dev/null 2>&1; then
        echo
        echo "TRELLIS ready http://127.0.0.1:7860"
        ok=1
        break
      fi
      echo -n "."
      sleep 5
    done
    if [[ "$ok" -ne 1 ]]; then
      echo
      echo "TRELLIS container started but Gradio not answering yet"
      docker logs trellis2 2>&1 | tail -20 || true
    fi
    set_mode sglang-qwen38-trellis
    ;;
  *)
    echo "usage: $0 daily|coding|video|trellis"
    exit 1
    ;;
esac

status
