#!/usr/bin/env bash
# cluster-ray-head.sh — start a multi-node vLLM instance via Ray (HEAD node)
#
# Use this when a model is too large to fit on a single Spark and must be
# sharded across multiple GPUs using tensor parallelism.
#
# For models that fit on one Spark (e.g. Qwen3-Coder-Next-FP8 at ~80 GB),
# use the load-balancing approach instead (cluster-lb-worker.sh /
# cluster-lb-proxy.sh) — it's simpler and gives better throughput.
#
# Cluster setup order:
#   1. Run this script on the HEAD node first.
#   2. Run cluster-ray-worker.sh on each WORKER node, passing the head IP.
#   3. vLLM will start once all workers have joined.
#
# Usage:
#   ./cluster-ray-head.sh <num_nodes>
#
# Example (3-node cluster):
#   ./cluster-ray-head.sh 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v22"
CONTAINER_NAME="vllm-ray-head"
VLLM_PORT=8000
RAY_PORT=6379
MODEL_DIR="$SCRIPT_DIR/models/Qwen3-Coder-Next-NVFP4"
NUM_NODES="${1:?Usage: $0 <num_nodes>}"
# Each Spark has 1 GPU; total tensor-parallel size = number of nodes
TOTAL_TP="$NUM_NODES"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

wait_for_http() {
    local url="$1" label="$2" timeout="${3:-900}"
    log "Waiting for $label ..."
    local elapsed=0
    until curl -sf "$url" > /dev/null 2>&1; do
        sleep 5; elapsed=$((elapsed + 5))
        [[ $elapsed -ge $timeout ]] && die "$label did not become ready within ${timeout}s"
        [[ $((elapsed % 60)) -eq 0 ]] && log "  still waiting ($elapsed s) — waiting for $NUM_NODES nodes to join..."
    done
    log "$label is ready."
}

# ── prereqs ───────────────────────────────────────────────────────────────────

[[ -f "$MODEL_DIR/config.json" ]] || die "NVFP4 model not found at $MODEL_DIR. Run: ./download-model.sh"

# ── Ray head + vLLM container ─────────────────────────────────────────────────

if docker ps --filter "name=$CONTAINER_NAME" --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Ray head container already running."
else
    docker rm "$CONTAINER_NAME" 2>/dev/null || true

    log "Pulling vLLM image ..."
    docker pull "$VLLM_IMAGE"
    log "Image ready."

    LOCAL_IP=$(hostname -I | awk '{print $1}')
    log "Starting Ray head node + vLLM (tp=$TOTAL_TP across $NUM_NODES nodes) ..."
    log "Ray address: ${LOCAL_IP}:${RAY_PORT}"

    # The official NVIDIA vLLM container (26.01) lacks proper NVFP4 kernel support
    # for the DGX Spark's GB10 GPU (SM121). Avarok's patched container fixes this.
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
        -e HEAD_IP="$LOCAL_IP" \
        -e TENSOR_PARALLEL_SIZE="$TOTAL_TP" \
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

    log "Head container started. Loading pre-quantized NVFP4 model (allow 15 min) ..."
fi

LOCAL_IP=$(hostname -I | awk '{print $1}')
log ""
log "Now run on each of the $((NUM_NODES - 1)) worker node(s):"
log "  ./cluster-ray-worker.sh ${LOCAL_IP}"
log ""
log "vLLM will begin loading once all $NUM_NODES nodes have joined."
log "Watch progress: docker logs $CONTAINER_NAME --follow"

wait_for_http "http://localhost:${VLLM_PORT}/health" "vLLM ($NUM_NODES-node cluster)" 3600

# ── LiteLLM ──────────────────────────────────────────────────────────────────

LITELLM="$SCRIPT_DIR/.venv/bin/litellm"
LITELLM_PID_FILE="$SCRIPT_DIR/.litellm.pid"
LITELLM_PORT=4000

[[ -x "$LITELLM" ]] || die "litellm not found. Run: uv sync --project $SCRIPT_DIR"

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
fi

# ── done ─────────────────────────────────────────────────────────────────────

cat <<EOF

$NUM_NODES-node Ray cluster ready.

Claude Code (this machine):
  source "$SCRIPT_DIR/use-local.sh" && claude

Claude Code (another machine):
  export ANTHROPIC_BASE_URL=http://${LOCAL_IP}:${LITELLM_PORT}
  export ANTHROPIC_AUTH_TOKEN=none
  claude

Stop:
  ./cluster-ray-stop.sh          # on head node
  ./cluster-ray-stop.sh          # on each worker node
EOF
