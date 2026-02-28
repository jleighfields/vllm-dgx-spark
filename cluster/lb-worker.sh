#!/usr/bin/env bash
# cluster/lb-worker.sh — start vLLM on this node only (no LiteLLM)
#
# Run this on EACH Spark in the cluster.
# After all workers are up, run cluster/lb-proxy.sh on one node.
#
# Usage:
#   ./cluster/lb-worker.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"
CONTAINER_NAME="vllm-server"
VLLM_PORT=8000
MODEL_DIR="$PROJECT_DIR/models/Qwen3-Coder-Next-NVFP4"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-600}"
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

[[ -f "$MODEL_DIR/config.json" ]] || die "NVFP4 model not found at $MODEL_DIR. Run: ./download-model.sh"

# ── vLLM container ────────────────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "vLLM container already running."
else
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    log "Pulling vLLM image ..."
    docker pull "$VLLM_IMAGE"
    log "Image ready."

    log "Starting vLLM ..."
    # The official NVIDIA vLLM container (26.01) lacks proper NVFP4 kernel support
    # for the DGX Spark's GB10 GPU (SM121). Avarok's patched container fixes this.
    # See: https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83
    docker run -d \
        --gpus all \
        --net host \
        --ipc host \
        --name "$CONTAINER_NAME" \
        -v "$MODEL_DIR:/model" \
        -v "$PROJECT_DIR/.cache/vllm:/root/.cache/vllm" \
        -e MODEL=/model \
        -e PORT="$VLLM_PORT" \
        -e GPU_MEMORY_UTIL=0.85 \
        -e MAX_MODEL_LEN=262144 \
        -e MAX_NUM_SEQS=128 \
        -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
        -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
        -e VLLM_NVFP4_GEMM_BACKEND=marlin \
        -e VLLM_DEEP_GEMM_WARMUP=skip \
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        -e VLLM_EXTRA_ARGS="--served-model-name Qwen/Qwen3-Coder-Next-NVFP4 --quantization modelopt_fp4 --enable-prefix-caching --attention-backend flashinfer --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
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

LOCAL_IP=$(hostname -I | awk '{print $1}')
log "vLLM ready at http://${LOCAL_IP}:${VLLM_PORT}"
log ""
log "Add this node to the proxy. On the proxy node run:"
log "  ./cluster/lb-proxy.sh ${LOCAL_IP} [other-ip] ..."
