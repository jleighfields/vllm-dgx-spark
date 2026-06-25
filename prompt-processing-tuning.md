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
> The Mamba upstream story (updated 2026-06-25): all four missing APC fixes
> (#34874, #35480, #34798, #35219) merged into vLLM `main` between 2026-02-23 and
> 2026-03-10. avarok still has not published anything past v23 (2026-02-21) — but
> those fixes are now shipping in the **official NGC vLLM container** (26.04 =
> vLLM 0.19.0, 26.05 = vLLM 0.20.1), which also now supports NVFP4 on SM121. So the
> Cirrascale revival no longer depends on a hypothetical avarok v24: it needs the
> official NGC image *and* the Test 3 hook. The container migration that makes this
> reachable is **Test 5** below (Cirrascale itself remains deferred/untested).
> Tracking issue #26201 remains open as a roll-up.
> https://github.com/vllm-project/vllm/issues/26201

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

**Status: Adopted as the default 2026-05-02. Generation throughput is within
measurement noise of native 256K at matched prompt sizes (initial readings
suggested a ~13% drop, but a controlled A/B with both configs showed the
apparent gap was sample-size and prompt-size effects, not a real YaRN tax).
The actual win is +2× compaction headroom for long sessions.**

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

**Verification (live Claude Code traffic, 2026-05-02):**

Initial read at 512K (~28 reqs) suggested a ~13% gen throughput drop vs an
earlier 256K snapshot — that snapshot turned out to be misleading. A
controlled A/B (~70 reqs at 512K, ~50 reqs at 256K with similar workload)
showed the dominant factor in gen throughput is **prompt size**, not context
config: large cached prompts (~150K) decode at ~23-27 t/s on either config
because each decode step has to read more KV cache; small prompts (~50K) hit
~30 t/s on either config.

| Metric | At 256K (matched workload) | At 512K (matched workload) |
|---|---|---|
| Gen throughput at ~150K prompts | ~23-25 t/s | ~25-27 t/s |
| Gen throughput at ~50-70K prompts | ~30 t/s | ~30 t/s |
| Per-cached-request hit rate | ~99.97% | ~99.96% |
| Avg TTFT (cached requests) | sub-second | sub-second |
| Max-context concurrency | ~7× | ~3.49× |
| Errors / OOM | 0 | 0 |

Any small remaining 256K-vs-512K gap is within run-to-run variance. The 5%
or so that 512K *might* be slower (if anything) on some workloads is
inconsequential next to the prompt-size effect.

**Decision:** Adopted as default. Rationale: this project is used for long
agentic coding sessions — heavy file reads, multi-step tool flows, deep
codebase exploration. In those sessions the conversation history grows
quickly (often 10–20K tokens per turn vs ~2K for short Q&A), and the 512K
ceiling pushes the upstream client's auto-compaction threshold from ~230K
(at 256K native) to ~480K — roughly 2× more turns of verbatim history before
any of it gets summarized into a single message. With gen throughput
essentially the same at matched workloads, the only meaningful trade-off is
the ~7× → ~3.49× max-context concurrency drop, which doesn't bind for 1-2
concurrent sessions. Roll back to native 256K only if your sessions are
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

### 5. Migrating off avarok to the official NGC container (2026-06-25)

**Status: DONE — migrated to NGC 26.04 (vLLM 0.19.0), validated on live Claude
Code traffic 2026-06-25. NGC 26.05 (0.20.1) was tested and rejected (gibberish on
real workloads — see results table). avarok v23 retained as a one-line fallback.**

**Motivation:** The project used the avarok `dgx-vllm-nvfp4-kernel:v23` container
only because, when it was built, the official NVIDIA container could not run NVFP4
on the DGX Spark's GB10 GPU (SM121) — a CUTLASS FP4 GEMM tile-size mismatch (tiles
sized for B200's 228 KiB shared memory vs GB10's 99 KiB) crashed it. Two open
items were tracked: (1) avarok never shipped a v24+, and (2) upstream's native
SM121 fix (vLLM v0.16.0, PR #33517) had not been packaged into an official NGC
container. A status check on 2026-06-25 resolved item (2).

**Findings (web research, 2026-06-25):**

| Question | Finding |
|---|---|
| avarok past v23? | **No.** Docker Hub shows only v21/v22/v23 + `latest`; `latest` == v23, pushed ~Feb 2026. Item (1) still open on avarok's side. |
| Official NGC caught up? | **Yes.** `nvcr.io/nvidia/vllm:26.04-py3` = vLLM **0.19.0** with documented SM121 Marlin/PTX fixes ("the de-facto stable Spark NGC"); `26.05-py3` = vLLM **0.20.1**. |
| Native FP4 kernels on GB10 now? | They exist in current FlashInfer but **still don't outpace Marlin Int4**; gap "substantially narrowed." W4A4 ≈ non-existent, W4A16 better. → **Keep Marlin.** |
| How is Marlin selected on NGC? | The `--moe-backend marlin` CLI flag — replaces avarok's `VLLM_NVFP4_GEMM_BACKEND` / `VLLM_USE_FLASHINFER_MOE_FP4=0` / `VLLM_TEST_FORCE_FP8_MARLIN=1` env trio. |
| Mamba APC fixes reachable? | **Yes**, bundled in vLLM 0.19/0.20 — so the hybrid Cirrascale model is now upstream-unblocked (revival still deferred/untested). |

**Decision:** Migrate to the official NGC container (default `26.04-py3`), keep the
same model (`NVFP4/Qwen3-Coder-30B-A3B-Instruct-FP4`), the same 512K YaRN profile,
and the same `cch_stripper` hook (Test 3 — still required; orthogonal to the
container). avarok stays available as a one-line fallback.

**Implementation:** The avarok and NGC images use different launch conventions, so
the swap is not just an image-tag change:
- `model.conf` — new `VLLM_IMAGE` and `VLLM_LAUNCH_STYLE` (`ngc` | `avarok`) vars;
  the SM121 env-flag section is now launch-style-aware (NGC drops the avarok env
  trio in favor of `--moe-backend marlin`).
- `start.sh` — `docker run` branches on `VLLM_LAUNCH_STYLE`: avarok keeps the
  env-var `serve` entrypoint; NGC issues a full `vllm serve /model --flags…`
  pass-through command.
- Cluster scripts (`lb-worker.sh`, `ray-head.sh`) remain avarok-only for now and
  set the avarok env trio locally so the `model.conf` default change doesn't break
  them.

**Open verify items — RESOLVED during live testing (2026-06-25):**
- `QUANTIZATION`: **`modelopt_fp4` is accepted by NGC 0.19/0.20** (startup args show
  `quantization: 'modelopt_fp4'`). No switch to `modelopt` needed.
- `--attention-backend`: NGC default works fine; left unset.
- `CUTE_DSL_ARCH=sm_121a`: not needed — Marlin is selected via `--moe-backend
  marlin` and boots clean without it.

Both NGC startups log `Using 'MARLIN' NvFp4 MoE backend` and two benign warnings:
`Your GPU does not have native support for FP4 ... weight-only ... Marlin` (expected)
and `w1_weight_scale_2 must match w3_weight_scale_2. Accuracy may be affected`
(checkpoint property — relevant to the 26.05 finding below).

**Results (live Claude Code traffic, 2026-06-25):**

| Metric | avarok v23 (baseline) | NGC 26.04 (vLLM 0.19.0) | NGC 26.05 (vLLM 0.20.1) |
|---|---|---|---|
| Boots clean on SM121 | ✅ | ✅ | ✅ |
| Quantization arg accepted | `modelopt_fp4` | ✅ `modelopt_fp4` | ✅ `modelopt_fp4` |
| MARLIN MoE backend selected | (env trio) | ✅ `--moe-backend marlin` | ✅ `--moe-backend marlin` |
| Prefix cache hit rate (cch hook on) | ~99.96% per-req | **85% cumulative & climbing** | n/a (pulled before warm-up) |
| Gen throughput | ~30 / ~23-27 t/s | **22–46 t/s (in range)** | n/a |
| Avg TTFT | sub-second | ~1.0s incl. cold starts | n/a |
| Tool calling via qwen3_coder | ✅ | ✅ | (untested — see below) |
| **Output quality on real workload** | ✅ | ✅ **coherent** | ❌ **GIBBERISH** |

**26.05 (vLLM 0.20.1) regression — DO NOT USE (yet):** Short prompts and direct
`/v1/completions` or `/v1/chat/completions` (even at temperature 1.0) are perfectly
coherent on 26.05. But real Claude Code workloads — large ~50K+ system prompt +
tool definitions + heavy prefix caching + YaRN 512K — produce **garbage output**
(random tokens, runs of `%`, broken markup). The likely culprit is the FP4/Marlin
SM121 accuracy gap (the `weight_scale_2` warning) compounding through the
long-context / prefix-cache path in 0.20.1; 0.19.0 (26.04) does not exhibit it.
Reproduced and rolled back same day.

**Decision: keep NGC 26.04 as the default.** It boots clean, caches (~85% and
rising), matches avarok throughput, and produces coherent output on the real
workload. avarok v23 remains the one-line fallback; **26.05 is pinned off** with a
warning in `model.conf` — retry only on a newer NGC tag and re-run the real-workload
quality check before trusting it.

References:
- [avarok/dgx-vllm-nvfp4-kernel — Docker Hub tags](https://hub.docker.com/r/avarok/dgx-vllm-nvfp4-kernel/tags)
- [vLLM Release Notes — NVIDIA Docs (26.04 = 0.19.0, 26.05 = 0.20.1)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)
- [State of native NVFP4 kernel support on GB10 — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/state-of-native-nvfp4-kernel-support-on-gb10/372559)
- [Marlin Fix: NVFP4 Actually Works on SM121 (DGX Spark) — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/marlin-fix-nvfp4-actually-works-on-sm121-dgx-spark/365119)

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
