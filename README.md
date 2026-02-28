# vLLM on DGX Spark — Qwen3-Coder-Next (NVFP4) with Prefix Caching

Serves **Qwen3-Coder-Next** using the NVIDIA vLLM container with NVFP4 quantization
and prefix caching enabled, bridged to Claude Code via LiteLLM.

```
Claude Code → LiteLLM :4000 (Anthropic API) → vLLM :8000 (OpenAI API) → Qwen3-Coder-Next-NVFP4
```

## Why vLLM on DGX Spark?

| | vLLM (this project) | llama.cpp (Docker Model Runner) |
|---|---|---|
| **Model format** | NVFP4 safetensors (pre-quantized) | GGUF MXFP4_MOE |
| **Model size on disk** | ~40 GB | 43.7 GB |
| **Prefix caching** | Yes — GPU KV cache reuse | Limited |
| **Throughput** | High (optimised for batching) | Moderate |
| **First-token latency** | Lower after cache warm-up | Higher (no prefix cache) |
| **Context length** | 256K (full native) | 256K (full native) |
| **Setup complexity** | Moderate | Simple |

**Prefix caching explained:** Claude Code sends a large, identical system prompt at the
start of every request. vLLM detects the repeated prefix, stores its KV states in GPU
memory after the first request, and skips recomputing them on all subsequent requests.
The result is noticeably faster first-token latency for every turn after the first.

## Memory Footprint

| Component | Size |
|-----------|------|
| NVFP4 model weights (pre-quantized by Cirrascale) | ~40 GB |
| KV cache (fp8, 256K ctx) | ~24 GB |
| Total | ~64 GB |

DGX Spark has 128 GB unified memory. At `gpu_memory_utilization=0.85` (~109 GB), this
leaves ~45 GB of headroom. NVFP4 is NVIDIA's Blackwell-native FP4 format (E2M1, 16-value
blocks) — ~10-15% faster than the open MXFP4 standard on the GB10 due to native silicon
pathways.

## Prerequisites

- NVIDIA DGX Spark (or compatible GPU with 128 GB memory)
- Docker with GPU access (`--gpus all`)
- [uv](https://docs.astral.sh/uv/) package manager (`~/.local/bin/uv`)
- ~40 GB free disk space for model storage (`models/` directory)

## Setup (run once)

```bash
cd ~/Documents/vllm-dgx-spark

# 1. Create the project venv and install dependencies (LiteLLM, huggingface-hub)
#    This must run before download-model.sh, which uses `uv run` to invoke Python.
~/.local/bin/uv sync

# 2. Download the pre-quantized NVFP4 model (~40 GB)
#    Downloads from HuggingFace into models/Qwen3-Coder-Next-NVFP4/.
#    Takes a while depending on your connection speed.
./download-model.sh
```

## Single Node

```
Claude Code → LiteLLM :4000 → vLLM :8000 → GPU
```

### Start

```bash
cd ~/Documents/vllm-dgx-spark
./start.sh
```

The vLLM container image (`avarok/dgx-vllm-nvfp4-kernel:v23`) is pulled automatically
on first run. The first cold start is slow (up to 15-30 minutes) because vLLM must
compile torch kernels; this compilation is cached in `.cache/vllm/` so subsequent
starts are much faster (typically 5-10 minutes). `start.sh` will wait up to 60 minutes
for vLLM to become healthy before giving up.

Watch progress with `docker logs vllm-server --follow`.

### Use Claude Code (on local machine)

```bash
source ~/Documents/vllm-dgx-spark/use-local.sh
claude
```

### Use Claude Code (from another machine)

Replace the IP with your DGX Spark's address:

```bash
export ANTHROPIC_BASE_URL=http://192.168.0.7:4000
export ANTHROPIC_AUTH_TOKEN=none
claude
```

### Stop

```bash
./stop.sh
```

---

## Cluster: Option A — Load Balancing (recommended for throughput)

```
Claude Code → LiteLLM :4000 → vLLM :8000 (Spark 1)
                             → vLLM :8000 (Spark 2)
                             → vLLM :8000 (Spark 3)
```

**Use this when:** You want to serve more concurrent users or increase total throughput.
Each Spark runs a full independent copy of the model. LiteLLM round-robins requests
across all workers. Scales linearly — N Sparks = N× throughput.

**Requires:** Model downloaded on every worker node (`./download-model.sh` on each).

### Start workers (run on each Spark)

```bash
cd ~/Documents/vllm-dgx-spark
./cluster/lb-worker.sh
```

### Start proxy (run on one node, after all workers are ready)

```bash
./cluster/lb-proxy.sh 192.168.0.10 192.168.0.11 192.168.0.12
```

### Stop

```bash
# On the proxy node:
./stop.sh

# On each worker node:
./stop.sh
```

### Open firewall for cross-node access

```bash
# Allow vLLM port from other Sparks (run on each worker)
sudo ufw allow from 192.168.0.0/24 to any port 8000

# Allow LiteLLM port for Claude Code clients (run on proxy node)
sudo ufw allow from 192.168.0.0/24 to any port 4000
```

---

## Cluster: Option B — Ray Multi-node (for models too large for one Spark)

```
Claude Code → LiteLLM :4000 → vLLM :8000 (head node)
                                    ↕ Ray cluster
                               GPU (Spark 1) + GPU (Spark 2) + GPU (Spark 3)
                               [model sharded across all GPUs]
```

**Use this when:** A model's weights don't fit in one Spark's 128 GB memory and must
be split across multiple GPUs using tensor parallelism. For example, a 200 GB model
would need at least 2 Sparks with this approach.

**Not needed for Qwen3-Coder-Next-NVFP4** (~40 GB weights + ~24 GB KV cache = ~64 GB,
which fits on a single Spark). Use Option A instead for this model.

**How it works:** vLLM uses [Ray](https://docs.ray.io) to coordinate across nodes.
Each GPU holds one shard of the model. All GPUs collaborate to process every request,
so latency per request can be lower, but total throughput doesn't scale beyond what
a single logical instance can handle.

### Prerequisites

- Ray ports open between nodes: TCP 6379 (GCS), plus ephemeral ports for object store
- Model downloaded only on the HEAD node (workers get weights via Ray)

```bash
# Open Ray ports on all nodes
sudo ufw allow from 192.168.0.0/24 to any port 6379
```

### Start (3-node example)

```bash
# 1. On head node — pass total number of nodes
./cluster/ray-head.sh 3

# 2. On each worker node — pass head node IP
./cluster/ray-worker.sh 192.168.0.10
```

vLLM starts automatically on the head node once all workers have joined.
Watch progress: `docker logs vllm-ray-head --follow`

### Stop

```bash
# Run on head node AND each worker node
./cluster/ray-stop.sh
```

---

## Files

| File | Purpose |
|------|---------|
| `download-model.sh` | Download pre-quantized Qwen3-Coder-Next-NVFP4 from HuggingFace |
| `start.sh` | Single-node: start vLLM + LiteLLM |
| `stop.sh` | Single-node / LB proxy: stop services |
| `use-local.sh` | Source to configure Claude Code env vars |
| `litellm-config.yaml` | Routes Claude model names → vLLM endpoint |
| `cluster/lb-worker.sh` | Cluster (Option A): start vLLM on this node |
| `cluster/lb-proxy.sh` | Cluster (Option A): start LiteLLM load-balancing proxy |
| `cluster/ray-head.sh` | Cluster (Option B): start Ray head + vLLM |
| `cluster/ray-worker.sh` | Cluster (Option B): join Ray cluster as worker |
| `cluster/ray-stop.sh` | Cluster (Option B): stop Ray containers on this node |

## Ports

| Port | Service | API format |
|------|---------|-----------|
| 8000 | vLLM | OpenAI-compatible (`/v1`) |
| 4000 | LiteLLM proxy | Anthropic-compatible (`/v1/messages`) |
| 6379 | Ray GCS (Option B only) | Ray cluster coordination |

## How It Works

LiteLLM acts as a protocol translator. Claude Code speaks the Anthropic Messages API,
but vLLM speaks the OpenAI Chat Completions API. LiteLLM sits in between, accepting
Anthropic-format requests on port 4000 and forwarding them as OpenAI-format requests
to vLLM on port 8000. The `litellm-config.yaml` maps Claude model names
(`claude-sonnet-4-6`, `claude-opus-4-6`, `claude-haiku-4-5-20251001`) to the local
vLLM endpoint so Claude Code works without modification.

Tool calling is enabled via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`,
so Claude Code's tool-use requests (file edits, bash commands, etc.) are translated
into the model's native tool-calling format.

### Why the avarok container?

The official NVIDIA vLLM container (`nvcr.io/nvidia/vllm`) does not work for NVFP4
inference on the DGX Spark's GB10 GPU (SM121) through at least release 26.02. The
root cause is a CUTLASS FP4 GEMM tile size mismatch: CUTLASS tiles were compiled for
B200's 228 KiB shared memory but GB10 only has 99 KiB, causing a `Failed to run
cutlass FP4 gemm on sm120` error.

`avarok/dgx-vllm-nvfp4-kernel` fixes this with four runtime patches:

1. **`fix_flashinfer_e2m1_sm121.py`** — software E2M1 conversion via bit manipulation,
   replacing the missing `cvt.rn.satfinite.e2m1x2.f32` PTX instruction on GB10
2. **`fix_flashinfer_nvfp4_moe_backend.py`** — fixes upstream vLLM bug in NVFP4 MoE
   backend routing
3. **`fix_capability_121_v112.py`** — routes SM 12.1 to SM 12.0 optimized paths for
   FlashInfer and CUTLASS
4. **`fix_mtp_nvfp4_exclusion.py`** — removes NVFP4 exclusion from speculative
   decoding (MTP)

As of vLLM v0.16.0 (released February 25, 2026), native SM121 CUTLASS tile support
was merged (PR #33517), which should eventually make the avarok patches unnecessary.
However, v0.16.0 has not yet been packaged into an official NVIDIA NGC container and
has not been independently validated for the full modelopt_fp4 inference path on GB10.
Until then, the avarok container remains the only pre-built option with verified
NVFP4/modelopt_fp4 support on DGX Spark.

References:
- [Avarok NVFP4 breakthrough post](https://blog.avarok.net/we-unlocked-nvfp4-on-dgx-spark-and-its-20-faster-than-awq-72b0f3e58b83)
- [avarok/dgx-vllm on GitHub](https://github.com/Avarok-Cybersecurity/dgx-vllm)
- [vLLM PR #33517 — SM121 CUTLASS support](https://github.com/vllm-project/vllm/pull/33517)

## Useful Commands

```bash
# View vLLM logs (model loading progress, errors)
docker logs vllm-server
docker logs vllm-server --follow

# View LiteLLM logs
tail -f ~/Documents/vllm-dgx-spark/litellm.log

# Check GPU memory usage
nvidia-smi

# Undo Claude Code local config
source ~/Documents/vllm-dgx-spark/use-local.sh --reset
```

## Metrics

vLLM exposes a Prometheus metrics endpoint at `http://localhost:8000/metrics`.

### Prefix cache hit rate

The most important metric for this use case. A high hit rate means vLLM is
successfully skipping recomputation of Claude Code's system prompt.

```bash
curl -s http://localhost:8000/metrics | grep -E "prefix_cache_(hit|miss)_rate"
```

A value above `0.7` is good. It will be `0` on the very first request and rise
quickly as subsequent requests share the same system prompt prefix.

### GPU KV cache utilisation

```bash
curl -s http://localhost:8000/metrics | grep gpu_cache_usage
```

### Request throughput and latency

```bash
# Tokens generated per second
curl -s http://localhost:8000/metrics | grep generation_tokens_total

# Time to first token (seconds)
curl -s http://localhost:8000/metrics | grep time_to_first_token
```

### Watch all metrics live

```bash
watch -n 2 'curl -s http://localhost:8000/metrics | grep -E "(prefix_cache|gpu_cache|time_to_first)"'
```

## Troubleshooting

**Model not downloaded**
```bash
./download-model.sh   # downloads ~40 GB
```

**Container exits immediately**
```bash
docker logs vllm-server   # check for OOM or other errors
```

**vLLM takes too long to start**
The first cold start is slow because vLLM compiles torch kernels for your GPU.
`start.sh` waits up to 60 minutes. The torch compile cache is persisted in
`.cache/vllm/`, so subsequent starts skip recompilation (typically 5-10 minutes).
If the container exits during compilation, check logs for OOM errors and consider
lowering `MAX_MODEL_LEN`.

**Reduce memory if needed**
Edit `start.sh`: change `MAX_MODEL_LEN=262144` to `131072` or lower.

**LiteLLM auth error in Claude Code**
Ensure `ANTHROPIC_AUTH_TOKEN=none` is exported. LiteLLM does not require authentication
by default.

**Ray workers not joining (Option B)**
Check that port 6379 is open between nodes and that all nodes are running the same
vLLM image version. Check head node logs: `docker logs vllm-ray-head --follow`.
