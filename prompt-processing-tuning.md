# Prompt Processing Speed — Tuning Notes

## Overview

Claude Code sends a large, identical system prompt at the start of every request.
The key optimisation for this workload is **prefix caching**: vLLM detects the
repeated prefix, stores its KV states in GPU memory after the first request, and
skips recomputing them on all subsequent requests. Without it, every request must
fully prefill the entire prompt from scratch, increasing time-to-first-token (TTFT)
proportionally with context length.

**The problem:** Qwen3-Coder-Next uses a hybrid **GatedDeltaNet + MoE** architecture
(`Qwen3NextForCausalLM` in vLLM). Unlike pure transformer models, hybrid models that
include Mamba or DeltaNet layers require special handling for prefix caching — the
recurrent state of those layers cannot be cached and replayed the same way transformer
KV cache blocks can. vLLM's support for this is marked experimental and, as tested
below, produces a **0% prefix cache hit rate** in practice.

**Impact:** Every Claude Code request fully recomputes the system prompt. At observed
prefill throughput of ~1,000–6,700 tokens/s (depending on torch compile cache warmup),
a typical Claude Code system prompt of ~50K tokens takes 7–50 seconds of prefill
before the first output token. TTFT histogram shows 6 out of 25 requests taking 20–40s.

**Root cause summary:** Not a configuration issue — it is an architectural limitation
of the current vLLM build for this model. See tests below for details.

> **Follow-up:** Switched to `RedHatAI/Qwen3-32B-NVFP4` (pure transformer, prefix
> caching works) as of Feb 2026. Revisit Qwen3-Coder-Next when:
> - A new avarok container image is released based on a vLLM commit after 2026-02-27
> - vLLM PR #34798 (kernel-level chunk alignment for Mamba1) is merged
> - vLLM issue #26201 (hybrid model prefix caching tracking) is closed
>
> Check https://github.com/vllm-project/vllm/issues/26201 for status.

---

## Baseline

- **Model:** Qwen3-Coder-Next-NVFP4
- **Container:** `avarok/dgx-vllm-nvfp4-kernel:v23`
- **vLLM version:** v0.16.0rc2.dev236+g3b30e6150.d20260221
- **Use case:** Claude Code (large, repeated system prompt per request)

---

## Tests

### 1. FP8 KV Cache — `--kv-cache-dtype fp8`

**Status: Removed from args (no effect — container defaults to fp8_e4m3)**

**Hypothesis:** FP8 KV cache reduces GPU memory usage, leaving more room for larger
context and more concurrent requests.

**Result:** Prefix caching produced 0 hits across 856,441 queried tokens over multiple
sessions. Removing the flag from `VLLM_EXTRA_ARGS` had no effect — the container image
defaults to `kv_cache_dtype=fp8_e4m3` regardless. To override, would need to explicitly
pass `--kv-cache-dtype auto`.

**Conclusion:** Not the root cause of 0% prefix cache hit rate. See test 2.

---

### 2. Prefix Caching with Mamba Layers

**Status: Broken — architectural limitation in current vLLM build**

**Hypothesis:** `--enable-prefix-caching` would cache the repeated Claude Code system
prompt KV states, skipping recomputation on subsequent requests.

**Result:** 0% prefix cache hit rate across all sessions and container restarts.
vLLM logs reveal the root cause:

```
WARNING: Mamba cache mode is set to 'align' for Qwen3NextForCausalLM by default
         when prefix caching is enabled
INFO: Prefix caching in Mamba cache 'align' mode is currently enabled.
      Its support for Mamba layers is experimental.
```

Qwen3-Coder-Next is a hybrid Mamba+attention model. Prefix caching for Mamba layers
is marked experimental in this vLLM build and does not produce cache hits.

**Conclusion:** Prefix caching is non-functional for this model in the current vLLM
build. This is a vLLM limitation, not a configuration issue.

**Mamba prefix caching upstream status (as of Feb 2026):**

Both `v22` and `v23` avarok container images are built from the **same vLLM commit**
(`3b30e6150`, 2026-02-16). `v23` is a non-functional rebuild of `v22` — the Dockerfile
is byte-for-byte identical and the `version` label inside the image still reads `22`.
The ~18 MB size difference is from non-deterministic apt package resolution.

Mamba prefix caching fix status in v22/v23:

| Fix | Merged | In v22/v23? |
|---|---|---|
| Mamba1 + Mamba2 APC (basic support) | Earlier | ✅ Present |
| `align`-mode async scheduling fix (PR #33997) | 2026-02-14 | ✅ Present |
| Mamba `all`-mode garbage output bugfix (PR #34874) | 2026-02-23 | ❌ Missing |
| Perf: pinned memory for async H2D transfer (PR #35480) | 2026-02-27 | ❌ Missing |
| Zero freed SSM cache blocks bugfix (PR #35219) | Not yet merged | ❌ Missing |
| Kernel-level chunk alignment for Mamba1 (PR #34798) | Not yet merged | ❌ Missing |

The most critical missing fix — kernel-level chunk alignment (PR #34798) — is likely
a key contributor to the 0% hit rate observed. It is not yet merged in mainline vLLM.

**Action:** Watch for avarok image `v24`+ based on a vLLM commit after 2026-02-27.
References:
- https://github.com/vllm-project/vllm/issues/26201
- https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/

---

## Raw Prefill Throughput

Despite prefix caching being non-functional, raw prompt throughput is strong:

| Observation | Prompt throughput |
|---|---|
| Initial (post-restart) | ~1,000–2,600 tokens/s |
| After torch compile cache warms | ~5,700–6,700 tokens/s |

At 6,700 tokens/s, a 100K token system prompt prefills in ~15s. Typical Claude Code
requests with smaller contexts will have significantly lower TTFT.

Chunked prefill is **enabled by default** in this vLLM build (`enable_chunked_prefill=True`
visible in startup config), so that optimization is already active.

---

## Options Not Yet Tested

### Override KV cache dtype — `--kv-cache-dtype auto`

Forces fp16 KV cache instead of the container's fp8_e4m3 default. Unlikely to fix
Mamba prefix caching but worth testing to rule out fp8/Mamba interaction.

### Increased batch token budget — `--max-num-batched-tokens`

Controls how many tokens are processed per scheduler step. Default is typically
2048–4096. Higher values may improve throughput for long prompts.

### Multi-step scheduling — `--num-scheduler-steps`

Reduces scheduler overhead by running multiple forward passes per scheduling call.
Values of 4–8 are common starting points.

---

## Key Metrics

```bash
# Prefix cache hit rate (target > 0.7 after warm-up)
curl -s http://localhost:8000/metrics | grep -E "prefix_cache_(hits|queries)_total"

# Time to first token distribution
curl -s http://localhost:8000/metrics | grep time_to_first_token

# Live log (includes prompt throughput and hit rate every 10s)
docker logs vllm-server --follow | grep "Prefix cache hit rate"

# Watch all at once
watch -n 2 'curl -s http://localhost:8000/metrics | grep -E "(prefix_cache|gpu_cache|time_to_first)"'
```
