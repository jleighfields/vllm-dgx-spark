#!/usr/bin/env bash
# start.sh — start vLLM (NVIDIA container) and LiteLLM proxy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=model.conf
source "$SCRIPT_DIR/model.conf"

VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"
CONTAINER_NAME="vllm-server"
VLLM_PORT=8000
LITELLM_PORT=4000
LITELLM_PID_FILE="$SCRIPT_DIR/.litellm.pid"
LITELLM="$SCRIPT_DIR/.venv/bin/litellm"
MODEL_DIR="$SCRIPT_DIR/models/$MODEL_DIR_NAME"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-300}"
    log "Waiting for $label ..."
    local elapsed=0
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -ge $timeout ]] && die "$label did not become ready within ${timeout}s"
        [[ $((elapsed % 30)) -eq 0 ]] && log "  still waiting ($elapsed s)..."
    done
    log "$label is ready."
}

# ── prereqs ───────────────────────────────────────────────────────────────────

[[ -x "$LITELLM" ]] || die "litellm not found. Run: uv sync --project $SCRIPT_DIR"

# Check model is ready
[[ -f "$MODEL_DIR/config.json" ]] || die "Model not found at $MODEL_DIR. Run: ./download-model.sh"

# ── vLLM container ────────────────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "vLLM container already running."
elif docker ps -a --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    # Container exists but is stopped — restart it to preserve compiled kernel caches
    log "Restarting stopped vLLM container (preserving compiled kernel caches) ..."
    docker start "$CONTAINER_NAME"
    log "Container restarted. Loading model ..."
else
    log "Pulling vLLM image (no-op if already cached) ..."
    docker pull "$VLLM_IMAGE"
    log "Image ready."

    log "Starting vLLM container ..."
    # The official NVIDIA vLLM container (26.01) lacks proper NVFP4 kernel support
    # for the DGX Spark's GB10 GPU (SM121). CUTLASS FP4 GEMM tiles are sized for
    # B200's 228 KiB shared memory but GB10 only has 99 KiB, causing silent fallback
    # to slower paths. Avarok's patched container fixes this with a software E2M1
    # conversion fallback and Marlin MoE backend routing for SM121.
    # See: https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83
    extra_env_args=()
    for kv in $EXTRA_DOCKER_ENVS; do extra_env_args+=(-e "$kv"); done

    docker run -d \
        --gpus all \
        --net host \
        --ipc host \
        --name "$CONTAINER_NAME" \
        -v "$MODEL_DIR:/model" \
        -v "$SCRIPT_DIR/.cache/vllm:/root/.cache/vllm" \
        -e MODEL=/model \
        -e PORT="$VLLM_PORT" \
        -e GPU_MEMORY_UTIL=0.85 \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        -e MAX_NUM_SEQS=128 \
        -e VLLM_DEEP_GEMM_WARMUP=skip \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e VLLM_EXTRA_ARGS="--served-model-name $SERVED_MODEL_NAME --quantization $QUANTIZATION --enable-prefix-caching --attention-backend flashinfer --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER${EXTRA_VLLM_FLAGS:+ $EXTRA_VLLM_FLAGS}" \
        "${extra_env_args[@]+"${extra_env_args[@]}"}" \
        "$VLLM_IMAGE" \
        serve
    #
    # --served-model-name — the name vLLM advertises on its /v1/models endpoint.
    #   Must match the `model: openai/<name>` value in litellm-config.yaml.
    # --quantization — set in model.conf (compressed-tensors for RedHatAI NVFP4,
    #   modelopt_fp4 for Cirrascale NVFP4, awq/fp8 for other formats).
    #
    # MAX_MODEL_LEN — set to the model's full context. Claude Code requests up to
    #   32K output tokens, so anything less than input+32K will cause vLLM to
    #   reject the request with a 400 error. vLLM only allocates as much KV cache
    #   as GPU memory allows; requests exceeding available KV cache are queued,
    #   not rejected.
    #
    # SM121 (DGX Spark GB10) MoE compatibility — set in model.conf EXTRA_DOCKER_ENVS:
    #   VLLM_NVFP4_GEMM_BACKEND=marlin — routes dense FP4 GEMM to Marlin (SM121 has
    #       99 KiB shared memory vs B200's 228 KiB; default CUTLASS tiles don't fit).
    #   VLLM_USE_FLASHINFER_MOE_FP4=0 — disables FlashInfer MoE backends; FlashInfer
    #       0.6.3 compiles Blackwell SM120 TMA kernels using cvt.e2m1x2 PTX, which is
    #       not supported on SM121, crashing vLLM at startup.
    #   VLLM_TEST_FORCE_FP8_MARLIN=1 — forces Marlin as the MoE backend (the "FP8"
    #       name is misleading — this also applies to NVFP4 MoE layers).
    #
    # Startup optimization notes:
    #   -v .cache/vllm:/root/.cache/vllm — persists torch compile cache across
    #       container restarts. First cold start is slow; subsequent restarts skip
    #       recompilation entirely.
    #   VLLM_DEEP_GEMM_WARMUP=skip — skips DeepGEMM JIT warmup (saves minutes;
    #       first-token latency may spike briefly on initial requests).

    log "Container started. Loading NVFP4 model (allow 15 min) ..."
fi

wait_for_http "http://localhost:${VLLM_PORT}/health" "vLLM" 3600

# ── LiteLLM ──────────────────────────────────────────────────────────────────

# Generate litellm-config.yaml from model.conf so they always stay in sync
LITELLM_CONFIG="$SCRIPT_DIR/litellm-config.yaml"
cat > "$LITELLM_CONFIG" <<YAML
# litellm-config.yaml — AUTO-GENERATED by start.sh from model.conf
# To change the model, edit model.conf and restart. Do not edit this file directly.
# Routes Claude Code (Anthropic Messages API) → vLLM (OpenAI API :${VLLM_PORT})

model_list:
  # model: openai/<name> — "openai/" tells LiteLLM to use the OpenAI-compatible
  #   API when forwarding to vLLM. The name after "openai/" must match
  #   --served-model-name passed to vLLM (set in model.conf as SERVED_MODEL_NAME).
  # model_info tells LiteLLM the context window so modify_params can cap
  #   max_tokens in outgoing requests (Claude Code requests up to 32K output).
  - model_name: claude-sonnet-4-6
    litellm_params:
      model: openai/${SERVED_MODEL_NAME}
      api_base: http://localhost:${VLLM_PORT}/v1
      api_key: "none"
      max_tokens: ${MAX_TOKENS}
    model_info:
      max_tokens: ${MAX_MODEL_LEN}
      max_output_tokens: ${MAX_TOKENS}

  - model_name: claude-opus-4-6
    litellm_params:
      model: openai/${SERVED_MODEL_NAME}
      api_base: http://localhost:${VLLM_PORT}/v1
      api_key: "none"
      max_tokens: ${MAX_TOKENS}
    model_info:
      max_tokens: ${MAX_MODEL_LEN}
      max_output_tokens: ${MAX_TOKENS}

  - model_name: claude-haiku-4-5-20251001
    litellm_params:
      model: openai/${SERVED_MODEL_NAME}
      api_base: http://localhost:${VLLM_PORT}/v1
      api_key: "none"
      max_tokens: ${MAX_TOKENS}
    model_info:
      max_tokens: ${MAX_MODEL_LEN}
      max_output_tokens: ${MAX_TOKENS}

litellm_settings:
  drop_params: true
  ignore_invalid_params: true
  modify_params: true

general_settings: {}
YAML

log "Generated litellm-config.yaml (model: $SERVED_MODEL_NAME)"

if [[ -f "$LITELLM_PID_FILE" ]] && kill -0 "$(cat "$LITELLM_PID_FILE")" 2>/dev/null; then
    log "LiteLLM already running (pid $(cat "$LITELLM_PID_FILE"))."
else
    log "Starting LiteLLM proxy on port ${LITELLM_PORT} ..."
    "$LITELLM" \
        --config "$LITELLM_CONFIG" \
        --port "$LITELLM_PORT" \
        >> "$SCRIPT_DIR/litellm.log" 2>&1 &
    echo $! > "$LITELLM_PID_FILE"
    log "LiteLLM started (pid $!), logging to litellm.log"
    wait_for_http "http://localhost:${LITELLM_PORT}/health" "LiteLLM" 30
fi

# ── done ─────────────────────────────────────────────────────────────────────

LOCAL_IP=$(hostname -I | awk '{print $1}')
cat <<EOF

Services running:

  vLLM (prefix caching) → http://localhost:${VLLM_PORT}
  LiteLLM proxy         → http://localhost:${LITELLM_PORT}

Claude Code (this machine):
  source "$SCRIPT_DIR/use-local.sh" && claude

Claude Code (another machine on the network):
  export ANTHROPIC_BASE_URL=http://${LOCAL_IP}:${LITELLM_PORT}
  export ANTHROPIC_AUTH_TOKEN=none
  claude

Monitor prefix cache hit rate:
  curl -s http://localhost:${VLLM_PORT}/metrics | grep prefix_cache

Container logs:
  docker logs $CONTAINER_NAME --follow
EOF
