#!/usr/bin/env bash
# start.sh — start vLLM (NVIDIA container) and LiteLLM proxy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v22"
CONTAINER_NAME="vllm-server"
VLLM_PORT=8000
LITELLM_PORT=4000
LITELLM_PID_FILE="$SCRIPT_DIR/.litellm.pid"
LITELLM="$SCRIPT_DIR/.venv/bin/litellm"
MODEL_DIR="$SCRIPT_DIR/models/Qwen3-Coder-Next-NVFP4"

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
[[ -f "$MODEL_DIR/config.json" ]] || die "NVFP4 model not found at $MODEL_DIR. Run: ./download-model.sh"

# ── vLLM container ────────────────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "vLLM container already running."
else
    # Remove any stopped container with the same name
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

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
    docker run -d \
        --gpus all \
        --net host \
        --ipc host \
        --name "$CONTAINER_NAME" \
        -v "$MODEL_DIR:/model" \
        -v "$SCRIPT_DIR/.cache/vllm:/root/.cache/vllm" \
        -e MODEL=/model \
        -e PORT="$VLLM_PORT" \
        -e GPU_MEMORY_UTIL=0.90 \
        -e MAX_MODEL_LEN=262144 \
        -e MAX_NUM_SEQS=128 \
        -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
        -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
        -e VLLM_NVFP4_GEMM_BACKEND=marlin \
        -e VLLM_DEEP_GEMM_WARMUP=skip \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e VLLM_EXTRA_ARGS="--served-model-name Qwen/Qwen3-Coder-Next-FP8 --quantization modelopt_fp4 --enable-prefix-caching --attention-backend flashinfer --kv-cache-dtype fp8 --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
        "$VLLM_IMAGE" \
        serve
    #
    # MAX_MODEL_LEN=262144 — set to the model's full 256K context. Claude Code
    #   requests up to 32K output tokens, so anything less than input+32K will
    #   cause vLLM to reject the request with a 400 error. vLLM only allocates
    #   as much KV cache as GPU memory allows; requests exceeding available KV
    #   cache are queued, not rejected.
    #
    # Startup optimization notes:
    #   -v .cache/vllm:/root/.cache/vllm — persists torch compile cache across
    #       container restarts. First cold start is slow; subsequent restarts skip
    #       recompilation entirely.
    #   VLLM_DEEP_GEMM_WARMUP=skip — skips DeepGEMM JIT warmup (saves minutes;
    #       first-token latency may spike briefly on initial requests).

    log "Container started. Loading pre-quantized NVFP4 model (allow 15 min) ..."
fi

wait_for_http "http://localhost:${VLLM_PORT}/health" "vLLM" 3600

# ── LiteLLM ──────────────────────────────────────────────────────────────────

if [[ -f "$LITELLM_PID_FILE" ]] && kill -0 "$(cat "$LITELLM_PID_FILE")" 2>/dev/null; then
    log "LiteLLM already running (pid $(cat "$LITELLM_PID_FILE"))."
else
    log "Starting LiteLLM proxy on port ${LITELLM_PORT} ..."
    "$LITELLM" \
        --config "$SCRIPT_DIR/litellm-config.yaml" \
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
