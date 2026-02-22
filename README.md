# vLLM — Qwen3-Coder-Next-FP8 with Prefix Caching

Serves **Qwen3-Coder-Next-FP8** using the NVIDIA vLLM container with prefix caching
enabled, bridged to Claude Code via LiteLLM.

```
Claude Code → LiteLLM :4000 (Anthropic API) → vLLM :8000 (OpenAI API) → Qwen3-Coder-Next-FP8
```

## vLLM vs llama.cpp (Docker Model Runner)

| | vLLM (this project) | llama.cpp (`~/Documents/vllm_docker`) |
|---|---|---|
| **Model format** | FP8 safetensors (native) | GGUF MXFP4_MOE |
| **Model size on disk** | ~80 GB | 43.7 GB |
| **Prefix caching** | Yes — GPU KV cache reuse | Limited |
| **Throughput** | High (optimised for batching) | Moderate |
| **First-token latency** | Lower after cache warm-up | Higher (no prefix cache) |
| **Context length** | 256K (full native, MXFP4) | 256K (full native) |
| **Setup complexity** | Moderate | Simple |

**Prefix caching explained:** Claude Code sends a large, identical system prompt at the
start of every request. vLLM detects the repeated prefix, stores its KV states in GPU
memory after the first request, and skips recomputing them on all subsequent requests.
The result is noticeably faster first-token latency for every turn after the first.

## Memory footprint

| Component | Size |
|-----------|------|
| NVFP4 model weights (pre-quantized by `quantize.sh`) | ~40 GB |
| KV cache (fp8, 256K ctx) | ~24 GB |
| Total | ~64 GB |

DGX Spark has 128 GB unified memory. At `gpu_memory_utilization=0.90` (~115 GB), this
leaves ~51 GB of headroom. NVFP4 is NVIDIA's Blackwell-native FP4 format (E2M1, 16-value
blocks) — ~10-15% faster than the open MXFP4 standard on the GB10 due to native silicon
pathways. The one-time quantization step (`quantize.sh`) runs in 20–40 min; after that
vLLM loads the pre-quantized weights directly in 5–10 min.

## Prerequisites

- Docker with GPU access
- ~80 GB free for model storage (`models/` directory)

## Setup (run once)

```bash
cd ~/Documents/vllm

# 1. Install LiteLLM into project venv
~/.local/bin/uv sync

# 2. Download the model (~80 GB — takes a while)
./download-model.sh

# 3. Quantize FP8 → NVFP4 (one-time, 20–40 min)
#    Produces a pre-quantized checkpoint that vLLM loads directly without
#    runtime quantization. Saves ~20+ min on every subsequent start.
#    NVFP4 uses Blackwell-native E2M1 format — ~10-15% faster than MXFP4 on GB10.
./quantize.sh
```

## Single Node

```
Claude Code → LiteLLM :4000 → vLLM :8000 → GPU
```

### Start

```bash
cd ~/Documents/vllm
./start.sh
```

Model takes 5–10 minutes to load from the pre-quantized NVFP4 checkpoint. Watch progress with `docker logs vllm-server --follow`.

### Use Claude Code (on this machine)

```bash
source ~/Documents/vllm/use-local.sh
claude
```

### Use Claude Code (from another machine)

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
cd ~/Documents/vllm
./cluster-lb-worker.sh
```

### Start proxy (run on one node, after all workers are ready)

```bash
./cluster-lb-proxy.sh 192.168.0.10 192.168.0.11 192.168.0.12
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

**Not needed for Qwen3-Coder-Next-FP8** (~80 GB weights + ~12 GB KV cache = ~92 GB,
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
./cluster-ray-head.sh 3

# 2. On each worker node — pass head node IP
./cluster-ray-worker.sh 192.168.0.10
```

vLLM starts automatically on the head node once all workers have joined.
Watch progress: `docker logs vllm-ray-head --follow`

### Stop

```bash
# Run on head node AND each worker node
./cluster-ray-stop.sh
```

---

## Files

| File | Purpose |
|------|---------|
| `download-model.sh` | Download Qwen3-Coder-Next-FP8 from HuggingFace |
| `quantize.sh` | One-time: quantize FP8 → NVFP4 using NVIDIA ModelOpt |
| `start.sh` | Single-node: start vLLM + LiteLLM |
| `stop.sh` | Single-node / LB proxy: stop services |
| `use-local.sh` | Source to configure Claude Code env vars |
| `litellm-config.yaml` | Routes Claude model names → vLLM endpoint |
| `cluster-lb-worker.sh` | Cluster (Option A): start vLLM on this node |
| `cluster-lb-proxy.sh` | Cluster (Option A): start LiteLLM load-balancing proxy |
| `cluster-ray-head.sh` | Cluster (Option B): start Ray head + vLLM |
| `cluster-ray-worker.sh` | Cluster (Option B): join Ray cluster as worker |
| `cluster-ray-stop.sh` | Cluster (Option B): stop Ray containers on this node |
| `litellm.log` | LiteLLM output (created at runtime) |

## Ports

| Port | Service | API format |
|------|---------|-----------|
| 8000 | vLLM | OpenAI-compatible (`/v1`) |
| 4000 | LiteLLM proxy | Anthropic-compatible (`/v1/messages`) |
| 6379 | Ray GCS (Option B only) | Ray cluster coordination |

## Useful Commands

```bash
# View vLLM logs (model loading progress, errors)
docker logs vllm-server
docker logs vllm-server --follow

# View LiteLLM logs
tail -f ~/Documents/vllm/litellm.log

# Check GPU memory usage
nvidia-smi
```

## Metrics

vLLM exposes a Prometheus metrics endpoint at `http://localhost:8000/metrics`.

### Prefix cache hit rate

The most important metric for this use case. A high hit rate means vLLM is
successfully skipping recomputation of Claude Code's system prompt.

```bash
# Hit rate (hits / (hits + misses))
curl -s http://localhost:8000/metrics | grep -E "prefix_cache_(hit|miss)_rate"
```

Expected output after a few requests:
```
vllm:prefix_cache_hit_rate{model_name="Qwen/Qwen3-Coder-Next-FP8"} 0.85
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
./download-model.sh   # downloads ~80 GB
```

**Container exits immediately**
```bash
docker logs vllm-server   # check for OOM or other errors
```

**vLLM takes too long to start**
`start.sh` waits up to 30 minutes. Loading the NVFP4 checkpoint normally takes 5–10 min.
If you haven't run `./quantize.sh` yet, do that first (one-time, 20–40 min).

**Reduce memory if needed**
Edit `start.sh`: change `--max-model-len 262144` to `--max-model-len 131072` or lower.

**LiteLLM auth error in Claude Code**
Ensure `ANTHROPIC_AUTH_TOKEN=none` is exported.

**Ray workers not joining (Option B)**
Check that port 6379 is open between nodes and that all nodes are running the same
vLLM image version. Check head node logs: `docker logs vllm-ray-head --follow`.

## Quantization Notes

`quantize.sh` runs inside the NVIDIA vLLM Docker container rather than using a local
Python venv. This was necessary to work around three issues on the DGX Spark (ARM64):

1. **`nvidia-modelopt` ≤ 0.27.0 + PyTorch 2.7+:** Older modelopt versions import
   `torch.onnx._type_utils`, which was removed in PyTorch 2.7. Fix: use
   `nvidia-modelopt>=0.35.0`, which officially requires `torch>2.6` and doesn't
   use the removed API.

2. **`nvidia-modelopt` requires `huggingface-hub<1.0`:** The main project needs
   `huggingface-hub>=1.4.1` for LiteLLM and `snapshot_download`. These constraints
   are mutually exclusive, so modelopt cannot be installed in the same venv as
   the project. Fix: run quantization in the Docker container which has its own
   Python environment.

3. **ARM64 PyTorch from PyPI is CPU-only:** The standard `torch` wheel on PyPI
   for `aarch64` does not include CUDA support, so the FP8 model loader fails with
   `No GPU or XPU found`. The NVIDIA vLLM container ships a CUDA-enabled torch
   built for the GB10, so quantization runs on GPU as expected.
