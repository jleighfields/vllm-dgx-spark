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

> **Follow-up (updated 2026-05-02):** Switched the active model to
> `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (pure `qwen3_moe`, no Mamba layers) to
> sidestep the architectural blocker. **Hit rate still ~0%.** Diagnosis showed the
> deeper cause — a per-request hash in Claude Code's system prompt poisoning the
> cache regardless of model architecture. Fixed via a LiteLLM pre-call hook;
> per-request hit rate went from 0.34% → **99.96%** within two requests. See
> **Test 3** below.
>
> The Mamba upstream story is unchanged: all four missing APC fixes (#34874,
> #35480, #34798, #35219) merged into vLLM `main` between 2026-02-23 and
> 2026-03-10, but no avarok image newer than v23 (2026-02-21) has been published,
> so the previous Cirrascale model would now require **both** an avarok v24+ rebuild
> *and* the Test 3 hook to actually cache. Tracking issue #26201 remains open as a
> roll-up. https://github.com/vllm-project/vllm/issues/26201

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

Mamba prefix caching fix status in v22/v23 (table updated 2026-05-02):

| Fix | Merged | In v22/v23? |
|---|---|---|
| Mamba1 + Mamba2 APC (basic support) | Earlier | ✅ Present |
| `align`-mode async scheduling fix (PR #33997) | 2026-02-14 | ✅ Present |
| Mamba `all`-mode garbage output bugfix (PR #34874) | 2026-02-23 | ❌ Missing |
| Perf: pinned memory for async H2D transfer (PR #35480) | 2026-02-27 | ❌ Missing |
| Kernel-level chunk alignment for Mamba1 (PR #34798) | 2026-03-01 | ❌ Missing |
| Zero freed SSM cache blocks bugfix (PR #35219) | 2026-03-10 | ❌ Missing |

All four are now merged in vLLM `main`. The most critical fix — kernel-level chunk
alignment (PR #34798) — landed 2026-03-01. The blocker for this project is no
longer upstream code: it is that the avarok container hasn't been refreshed past
v23 (2026-02-21), so none of these fixes are reachable without rebuilding the image.

**Action:** Watch for avarok image `v24`+ based on a vLLM commit after 2026-03-10,
or build a custom image from current vLLM `main` plus the avarok runtime patches.
References:
- https://github.com/vllm-project/vllm/issues/26201
- https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/

---

### 3. The cch Prefix Killer — Claude Code's billing header (FIXED)

**Status: Fixed via LiteLLM pre-call hook (2026-05-02). Per-request hit rate
0.06% → 99.96%.**

**Symptom:** After switching from `Cirrascale/Qwen3-Coder-Next-NVFP4` (hybrid
Mamba) to `NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4` (pure `qwen3_moe`) — a model
where vLLM's prefix caching definitively works in v23 — the cumulative hit rate
stayed at ~0.3%. Across ~30 real Claude Code requests of ~50K tokens each, every
single request showed **exactly +32 cache hits** (= 2 vLLM cache blocks × 16
tokens each = the chat-template BOS opener) regardless of prompt size.

**Hypothesis test (isolation):** Sent two identical ~3,200-token prompts directly
to vLLM via `curl /v1/completions`. Result: req 1 cold (620ms), req 2 nearly all
hits (155ms — **4× speedup**), combined hit rate 50%. **vLLM caching itself is
fine.** The bug must be upstream of vLLM.

**Diagnostic (LiteLLM `--detailed_debug`):** Restarted only the LiteLLM proxy
with detailed_debug writing to a separate `litellm-debug.log`. Captured two
consecutive Claude Code request bodies and diffed them. Every request started
with this exact text as the first item of the system content array:

```
x-anthropic-billing-header: cc_version=2.1.121.dbb; cc_entrypoint=cli; cch=<5-hex>;
```

`cc_version` and `cc_entrypoint` are stable; **`cch` changes on every request**
(observed: `cch=3049b` → `cch=df8a0` for two consecutive turns). Because this
header lives at the very start of the system prompt, vLLM's prefix matcher
diverges at byte ~50 and the entire downstream 50K-token system prompt becomes
uncacheable. The "+32 hits" we kept seeing was just the chat-template prefix
that *precedes* the system content.

**Fix:** A LiteLLM `CustomLogger` pre-call hook strips any content item whose
text starts with `x-anthropic-billing-header:`, before forwarding to vLLM. The
billing header has no consumer in this stack (we are not Anthropic's billing
system), so removing it has zero functional impact.

Implementation:
- `litellm_hooks.py` — `CCHStripper` class scans both `data["system"]`
  (Anthropic format) and `data["messages"]` (post-OpenAI-conversion format)
  and removes matching items.
- `start.sh` — registers the hook via
  `litellm_settings.callbacks: ["litellm_hooks.cch_stripper"]` in the
  auto-generated config; sets `PYTHONPATH=$SCRIPT_DIR` on the LiteLLM launch so
  the module is importable.

Important detail discovered while wiring it up: when Claude Code hits LiteLLM's
`/v1/messages?beta=true` endpoint, the proxy invokes `async_pre_call_hook` with
`call_type="anthropic_messages"` **before** converting to OpenAI format. In that
state, the system prompt is in `data["system"]` (a list of content blocks), not
in `data["messages"]` with `role="system"`. A first-cut hook that only scanned
`messages` did nothing — it has to handle both shapes.

**Result (verified with live Claude Code traffic, 2026-05-02):**

| | Before hook | After hook (req 2 onward) |
|---|---|---|
| Per-request hit rate | 32 / ~50K ≈ **0.06%** | **99.96%** (~20 misses = the user's actual question content) |
| Cumulative hit rate | 0.34% across ~30 reqs | climbed past 27% within 4 cached reqs and rising |
| TTFT | ~13s for 50K-token prompts | dropping fast as cumulative climbs |

**Implication for Cirrascale and any other hybrid-Mamba model:** This fix is
**necessary but not sufficient** for those models. They still need a vLLM build
with the four merged Mamba APC fixes (i.e. avarok v24+) before caching engages
at all. Both fixes are independent and both are required.

References:
- `litellm_hooks.py`
- `start.sh` (config generation, LiteLLM launch with `PYTHONPATH`)

---

### 4. Extending Context to 512K via YaRN ×2 (DEPLOYED)

**Status: Adopted as the default 2026-05-02. Costs ~13% generation throughput
in steady state (~33 t/s → ~28.5 t/s) but doubles the context ceiling and
preserves all caching/quality behavior on prompts < native 256K.**

**Motivation:** With prefix caching working at native 256K (Test 3), there was
headroom in the KV pool to extend the model's context ceiling. The Qwen3-Coder
model card claims native 262144 with YaRN-extension up to 1M.

**Implementation:** vLLM v0.16 doesn't accept a top-level `--rope-scaling` flag
in this build (returns `unrecognized arguments`). The correct mechanism is
`--hf-overrides`, which merges a JSON dict into the loaded HF config:

```
EXTRA_VLLM_FLAGS='--hf-overrides {"max_position_embeddings":524288,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}'
```

Two non-obvious gotchas hit during deployment:
1. vLLM's `ModelConfig` validation rejects `max_model_len > config.max_position_embeddings`
   *before* `--hf-overrides` is merged. Workaround: set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
   in the container env. (Also overriding `max_position_embeddings` in the JSON
   dict is harmless and may help on future vLLM versions where the check moves.)
2. The container's torch_compile_cache hash does NOT include `max_model_len`,
   so changing 256K → 512K only triggered cudagraph re-capture for batch sizes
   1–256 (~3 min), not full kernel recompile. Total cold restart was ~10 min,
   not the 15-30 min I'd budgeted.

**Verification (live Claude Code traffic, 2026-05-02 after switch):**

Initial read (~10 min, 18 reqs) showed gen throughput within noise of 256K
baseline. After more data accumulated (~28 reqs over ~30 min), the steady-state
emerged: **gen throughput is meaningfully lower at 512K**. Hit rate, TTFT, and
quality behavior on short prompts are unchanged.

| Metric | At 256K (before) | At 512K (steady state) | Delta |
|---|---|---|---|
| Generation throughput | ~33–34 t/s | **~28.5 t/s** | **~13% slower** |
| Per-cached-request hit rate | ~99.96% | ~99.93–99.97% | identical |
| Avg TTFT (cached requests) | sub-second | sub-second | identical |
| Max-context concurrency | ~5× | ~3.49× | tighter (irrelevant <2 sessions) |
| Errors / OOM | 0 | 0 | — |

A 1000-token response is ~30s at 256K vs ~35s at 512K — a ~5s difference per
turn that's noticeable but not painful for interactive use.

**Likely causes of the gen throughput drop:**
- YaRN rope scaling adds a small constant per-token cost on every decode step,
  applied on all requests regardless of length.
- Slightly larger CUDA graph footprint at the 512K shape.

**Decision:** Adopted as default. Rationale: this project is used for long
Claude Code coding sessions — heavy file reads, multi-step tool flows, deep
codebase exploration. In those sessions the conversation history grows
quickly (often 10–20K tokens per turn vs ~2K for short Q&A), and the 512K
ceiling pushes Claude Code's auto-compaction threshold from ~230K (at 256K
native) to ~480K. That keeps more session history *in scope* before older
turns get summarized away — usually more valuable than the ~5s/turn the
larger context costs. Roll back to native 256K only if your sessions are
predominantly short interactive Q&A — instructions in the "Rollback to native
256K" block in `model.conf`.

**Caveats not yet evaluated:**
- YaRN×2 is theoretically applied on every request, including short prompts.
  At factor=2.0 quality should be near-identical to native RoPE in the
  interpolated range, but no formal A/B has been run. If subtle weirdness on
  short-context tasks ever appears, suspect this first.
- No prompt >262K has actually been served yet, so YaRN-extended attention
  quality for genuinely long contexts is untested.

References:
- `model.conf` (active config)
- HuggingFace [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) (model card claims 1M with YaRN)

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
