# vLLM on DGX Spark — Qwen3 (NVFP4) with Prefix Caching

Serves a **Qwen3 NVFP4** model using the official NVIDIA NGC vLLM container with
NVFP4 quantization and prefix caching enabled, bridged to Claude Code via LiteLLM.
The active model — and the container image / launch style — are configured in a
single `model.conf` file. (The project originally required the patched Avarok
container; as of NGC 26.04 the official container supports NVFP4 on the DGX Spark's
GB10 GPU, so it is now the default, with Avarok kept as a one-line fallback. See
["Why this container?"](#why-this-container) and `prompt-processing-tuning.md`
Test 5.)

```
Claude Code → LiteLLM :4000 (Anthropic API) → vLLM :8000 (OpenAI API) → Qwen3-NVFP4
```

## Why vLLM on DGX Spark?

| | vLLM (this project) | llama.cpp (Docker Model Runner) |
|---|---|---|
| **Model format** | NVFP4 safetensors (modelopt_fp4 / compressed-tensors) | GGUF MXFP4_MOE |
| **Model size on disk** | ~16 GB (default) | 43.7 GB |
| **Prefix caching** | Yes — GPU KV cache reuse (with this project's `cch` hook) | Limited |
| **Throughput** | High (optimised for batching) | Moderate |
| **First-token latency** | Lower after cache warm-up | Higher (no prefix cache) |
| **Context length** | 512K (via YaRN ×2) — native 256K | 256K (full native) |
| **Setup complexity** | Moderate | Simple |

**Prefix caching explained:** Claude Code sends a large, identical system prompt at the
start of every request. vLLM detects the repeated prefix, stores its KV states in GPU
memory after the first request, and skips recomputing them on all subsequent requests.
The result is noticeably faster first-token latency for every turn after the first.

## Switching Models

Edit `model.conf` — all scripts source it automatically:

```bash
# model.conf — edit this file to switch models
VLLM_IMAGE="nvcr.io/nvidia/vllm:26.04-py3"              # container image to run
VLLM_LAUNCH_STYLE="ngc"                                 # "ngc" | "avarok" (see below)
MODEL_REPO="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"     # HuggingFace repo to download
MODEL_DIR_NAME="Qwen3-Coder-30B-A3B-Instruct-FP4"        # local subdir under models/
SERVED_MODEL_NAME="NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4"  # name vLLM advertises
QUANTIZATION="modelopt_fp4"                              # vLLM quantization format
MAX_MODEL_LEN=524288                                     # 512K via YaRN ×2 (model native is 262144)
TOOL_CALL_PARSER="qwen3_coder"                           # tool call parser
MAX_TOKENS=16384                                         # max LiteLLM response tokens
# YaRN ×2 rope-scaling extends 256K → 512K. Default sized for long Claude Code
# coding sessions where dense history (file reads + tool flows) would otherwise
# trigger Claude Code's auto-compaction around ~230K; 512K pushes that threshold
# to ~480K. A controlled A/B (2026-05-02) showed gen throughput is essentially
# identical between 256K and 512K at matched workloads — prompt size dominates,
# not context config — so adopting 512K is essentially free for caching/TTFT/
# throughput, with the only real cost being a ~7× → ~3.49× max-context
# concurrency drop (irrelevant for 1-2 concurrent sessions).
EXTRA_VLLM_FLAGS='--hf-overrides {"max_position_embeddings":524288,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}'
# EXTRA_DOCKER_ENVS is computed automatically from VLLM_LAUNCH_STYLE (see model.conf):
#   ngc    — Marlin is selected by the --moe-backend marlin CLI flag, so only the
#            YaRN bypass (VLLM_ALLOW_LONG_MAX_MODEL_LEN=1) is passed as an env.
#   avarok — the three NVFP4-MoE-on-SM121 env flags + the YaRN bypass.
```

After editing `model.conf`:
```bash
./stop.sh
./download-model.sh   # if switching to a model not yet downloaded
./start.sh
```

`start.sh` regenerates `litellm-config.yaml` automatically on every run, so it
always stays in sync with `model.conf`.

## Memory Footprint

| Component | Size |
|-----------|------|
| NVFP4 model weights (`NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`, current) | ~16 GB |
| KV cache pool (fp8_e4m3, sized by `gpu_memory_utilization`) | ~85 GB |
| Total at runtime | ~101 GB |

DGX Spark has 128 GB unified memory. At `gpu_memory_utilization=0.85` (~109 GB),
the KV cache pool can hold the equivalent of **~3.5× max-context (512K)
sequences** simultaneously, or many more short sequences via vLLM paging. Each
in-flight 512K-token sequence costs ~25 GB of pool space (~12 GB at the model's
native 256K). NVFP4 is NVIDIA's Blackwell-native FP4 format (E2M1, 16-value
blocks).

## Model Alternatives

The current model (`NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`) is a pure `qwen3_moe`
MoE — no Mamba layers — which is why prefix caching works on avarok v23. The previous
default (`Cirrascale/Qwen3-Coder-Next-NVFP4`) is a hybrid GatedDeltaNet+MoE
architecture; vLLM's prefix-caching support for those hybrid layers was broken in
avarok v23 (0% hit rate). The four Mamba APC fixes it needs are now in the official
NGC image (vLLM 0.19/0.20), so the upstream blocker is cleared — reviving it on NGC
is plausible but untested/deferred. See `prompt-processing-tuning.md` Tests 2 & 5
and the Cirrascale alternative-config block in `model.conf`.

| Model | Architecture | Quantization | Weights | Prefix caching | Notes |
|---|---|---|---|---|---|
| `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` *(current)* | `qwen3_moe` — pure MoE, no Mamba | NVFP4 (modelopt_fp4) | ~16 GB | ✅ Works (NGC + cch hook; also avarok v23) | Coding-specialized, **512K context via YaRN ×2** (native 256K), ~32-45 t/s solo throughput. |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` | `qwen3_moe` — pure MoE, no Mamba | FP8 | ~18 GB | ✅ Works | Official Qwen FP8 release; same coder line. |
| `Qwen/Qwen2.5-Coder-32B-Instruct` | `qwen2` — pure dense transformer | BF16 | ~64 GB | ✅ Works | Gold-standard 32B coder; no NVFP4 available. |
| `BCCard/Qwen2.5-Coder-32B-Instruct-FP8-Dynamic` | `qwen2` — pure dense transformer | FP8 | ~32 GB | ✅ Works | Community FP8 of above. |
| `Qwen/Qwen3-32B-FP8` | `qwen3` — pure dense transformer | FP8 | ~32 GB | ✅ Works | General purpose, not coding-specialized. |
| `RedHatAI/Qwen3-32B-NVFP4` | `qwen3` — pure dense transformer | NVFP4 (compressed-tensors) | ~18 GB | ✅ Works | General purpose. |
| `Cirrascale/Qwen3-Coder-Next-NVFP4` | Hybrid GatedDeltaNet+MoE | NVFP4 (modelopt_fp4) | ~40 GB | ⚠️ Broken on avarok v23; unblocked on NGC (untested) | Fastest raw throughput when caching works. Revival needs the NGC image (vLLM 0.19+, has the Mamba APC fixes) AND the cch hook (see below). |

**Note on prefix caching for Claude Code workloads:** Claude Code prepends a
per-request `x-anthropic-billing-header: ... cch=<hex>;` to every system prompt,
where `cch` changes on every turn. Without intervention this defeats vLLM's
prefix cache regardless of model (observed: 0.06% per-request hit rate). This
project ships a LiteLLM pre-call hook (`litellm_hooks.py`) that strips that
header before forwarding to vLLM, restoring near-100% per-request cache hits.
Full diagnosis in `prompt-processing-tuning.md` Test 3.

## Prerequisites

- NVIDIA DGX Spark (or compatible GPU with 128 GB memory)
- Docker with GPU access (`--gpus all`)
- [uv](https://docs.astral.sh/uv/) package manager (`~/.local/bin/uv`)
- ~20 GB free disk space for the default model (more if you keep multiple
  alternatives under `models/`)

## Setup (run once)

```bash
cd ~/Documents/vllm-dgx-spark

# 1. (Optional) Edit model.conf to select which model to run.
#    The default is NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4.

# 2. Create the project venv and install dependencies (LiteLLM, huggingface-hub)
#    This must run before download-model.sh, which uses `uv run` to invoke Python.
~/.local/bin/uv sync

# 3. Download the pre-quantized NVFP4 model
#    Downloads from HuggingFace into models/<MODEL_DIR_NAME>/.
#    Default model is ~16 GB; larger alternatives in model.conf can reach 40 GB+.
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

The vLLM container image (default `nvcr.io/nvidia/vllm:26.04-py3`, set by
`VLLM_IMAGE` in `model.conf`) is pulled automatically on first run. The first cold
start is slow (up to 15-30 minutes) because vLLM must compile torch kernels; this
compilation is cached in `.cache/vllm/` so subsequent starts are much faster
(typically 5-10 minutes). Switching `VLLM_IMAGE` to a different vLLM version
invalidates that cache, so the first start on a new image recompiles. `start.sh`
will wait up to 60 minutes for vLLM to become healthy before giving up.

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
./stop.sh           # stop services; keep container (preserves compiled kernel caches)
./stop.sh --clean   # stop and remove container (required when switching models)
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

**Not needed for the default model** (`NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`,
~16 GB weights — even with 512K context via YaRN ×2 there's ample headroom for
~3.5× concurrent max-context sequences on a single Spark). Use Option A instead.

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
| `model.conf` | **Model + container configuration** — edit this to switch models, context length, or the vLLM image / launch style (`VLLM_IMAGE`, `VLLM_LAUNCH_STYLE`) |
| `download-model.sh` | Download the model configured in `model.conf` from HuggingFace |
| `start.sh` | Single-node: start vLLM + LiteLLM (regenerates `litellm-config.yaml`) |
| `stop.sh` | Single-node / LB proxy: stop services |
| `use-local.sh` | Source to configure Claude Code env vars |
| `litellm-config.yaml` | Auto-generated by `start.sh` — do not edit directly |
| `litellm_hooks.py` | LiteLLM pre-call hook that strips Claude Code's per-request `cch=` header so prefix caching works (see `prompt-processing-tuning.md` Test 3) |
| `cleanup-models.sh` | Interactively delete unused model dirs + HuggingFace hub cache duplicates; never offers the active model |
| `prompt-processing-tuning.md` | Investigation log: prefix-cache failures, root causes, fixes |
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

Tool calling is enabled via `--enable-auto-tool-choice --tool-call-parser <parser>`,
where the parser is set per-model in `model.conf` (`qwen3_coder` for the Qwen3-Coder
family, `qwen3_xml` for the general-purpose Qwen3 dense models like RedHatAI's). Claude
Code's tool-use requests (file edits, bash commands, etc.) are translated into the
model's native tool-calling format.

### Why this container?

**Default: official NGC (`nvcr.io/nvidia/vllm:26.04-py3`, vLLM 0.19.0).** When this
project was first built (Feb–May 2026), the official NVIDIA container could *not* run
NVFP4 inference on the DGX Spark's GB10 GPU (SM121): a CUTLASS FP4 GEMM tile-size
mismatch (tiles compiled for B200's 228 KiB shared memory vs GB10's 99 KiB) produced
`Failed to run cutlass FP4 gemm on sm120`. Native SM121 support landed upstream in
vLLM v0.16.0 (PR #33517) and, as of NGC **26.04** (vLLM 0.19.0), is packaged with
the SM121 Marlin/PTX fixes in an official container. On GB10, FP4 MoE still runs
fastest on **Marlin** (native FP4 kernels exist but don't yet outpace Marlin Int4),
selected with the `--moe-backend marlin` CLI flag. This is now the default. See
`prompt-processing-tuning.md` Test 5 for the migration findings and `model.conf` for
the `VLLM_IMAGE` / `VLLM_LAUNCH_STYLE` knobs.

**Fallback: avarok (`avarok/dgx-vllm-nvfp4-kernel:v23`).** Before the official
container worked, this patched image was the only pre-built option with NVFP4 support
on the Spark. It bundles four runtime patches:

1. **`fix_flashinfer_e2m1_sm121.py`** — software E2M1 conversion via bit manipulation,
   replacing the missing `cvt.rn.satfinite.e2m1x2.f32` PTX instruction on GB10
2. **`fix_flashinfer_nvfp4_moe_backend.py`** — fixes upstream vLLM bug in NVFP4 MoE
   backend routing
3. **`fix_capability_121_v112.py`** — routes SM 12.1 to SM 12.0 optimized paths for
   FlashInfer and CUTLASS
4. **`fix_mtp_nvfp4_exclusion.py`** — removes NVFP4 exclusion from speculative
   decoding (MTP)

To use it, set `VLLM_IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"` and
`VLLM_LAUNCH_STYLE="avarok"` in `model.conf`, then `./stop.sh --clean && ./start.sh`.
(avarok has not published anything past v23 / vLLM 0.16; the official NGC image is
the maintained path going forward.)

References:
- [vLLM Release Notes — NVIDIA NGC (26.04 = 0.19.0)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)
- [State of native NVFP4 kernel support on GB10 — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/state-of-native-nvfp4-kernel-support-on-gb10/372559)
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
./download-model.sh   # ~16 GB for the default model
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
Edit `model.conf`: lower `MAX_MODEL_LEN` (e.g. `131072` instead of `262144`).

**LiteLLM auth error in Claude Code**
Ensure `ANTHROPIC_AUTH_TOKEN=none` is exported. LiteLLM does not require authentication
by default.

**Container crashes at startup with `cvt.e2m1x2 not supported on sm_121`**
*(avarok launch style only)*

FlashInfer 0.6.3 (shipped in both v22 and v23 of the avarok container) includes
Blackwell SM120 TMA grouped GEMM kernels that use the `cvt.e2m1x2` PTX instruction,
which is not available on SM121 (GB10). When vLLM selects the `FLASHINFER_CUTLASS`
MoE backend, it tries to JIT-compile these kernels and crashes.

Fix: when running `VLLM_LAUNCH_STYLE="avarok"`, the three Marlin env vars are set
automatically (`EXTRA_DOCKER_ENVS` is computed from the launch style in `model.conf`).
If you see this crash, confirm the launch style is `avarok` and that the env vars
reach the container (`docker inspect vllm-server | grep VLLM_`). The NGC launch
style avoids this path entirely by selecting Marlin via `--moe-backend marlin`.
Then restart with a fresh container (the failed compilation may have left a corrupted
cache in the container's writable layer):
```bash
./stop.sh --clean
./start.sh
```

**Ray workers not joining (Option B)**
Check that port 6379 is open between nodes and that all nodes are running the same
vLLM image version. Check head node logs: `docker logs vllm-ray-head --follow`.

**Prefix cache hit rate stuck near 0%**
Should be > 0.7 after a couple of Claude Code requests. If it's stuck near 0%
with the per-request signature being exactly +32 hits, the `cch_stripper` hook
isn't loading. Verify with:
```bash
grep "\[cch_stripper\]" ~/Documents/vllm-dgx-spark/litellm.log | tail
```
If there are no `[cch_stripper] ... pre_call: stripped N billing-header item(s)`
lines, the hook isn't firing — check that `start.sh` set `PYTHONPATH` and that
`litellm-config.yaml` lists `litellm_hooks.cch_stripper` under `callbacks`. Full
context in `prompt-processing-tuning.md` Test 3.
