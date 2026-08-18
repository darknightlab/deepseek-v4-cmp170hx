# Official upstream patch queue

This directory rebases the CMP 170HX stack onto a pinned official
`vllm-project/vllm` main commit. It leaves the legacy `c3046d1` workflow
untouched.

## Pinned base

```text
vllm-project/vllm 402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
2026-08-17 [Bugfix][CI] Release the shared ColBERT engine before test_colbert_hf_comparison (#52608)
```

The pin makes native builds reproducible. Move it only by deliberately rebasing
and retesting all three layers.

## Patch layers

1. `0001-DSv4-SM80-add-minimal-...patch`

   Minimal SM80 correctness support. It contains the Ampere sparse-MLA backend,
   software FP8 encode/decode, Triton sparse attention and indexer logits,
   DeepGEMM/CuTeDSL fallbacks, the GDN build guard fix, and focused kernel tests.
   It does not contain communication, Marlin, parser, DSpark serving, disk KV,
   or unrelated fork experiments.

2. `0002-DSv4-SM80-add-optional-fork-performance-...patch`

   Optional, reachable fork optimizations: deterministic top-k/sampling,
   Marlin tuning, router GEMV, hierarchical all-reduce, local argmax, cache
   kernel tuning, indexer-logits tuning, and multi-stream controls. Removing
   this patch preserves the SM80 execution path and downstream behavior while
   providing a correctness/performance control arm.

3. `0003-Apply-deepseek-v4-cmp170hx-downstream-...patch`

   Project serving fixes: DSpark+PP, long-context top-k/row chunking, parser and
   Responses compatibility, structured output, and bounded O_DIRECT disk KV
   offload for arbitrary PP payload sizes.

The old single residual included provider code whose callers had been lost
while resolving conflicts. Unreachable MHC int8, DSpark fused-Markov,
PinnedStagingPool, query-sharding helpers, and their inert environment variables
are intentionally absent from all three layers.

## Prepare a checkout

Apply all layers (default):

```bash
./scripts/prepare-upstream-vllm.sh /opt/vllm-upstream
```

Skip the optional performance layer:

```bash
VLLM_APPLY_FORK_PERFORMANCE=0 \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-sm80-control
```

Probe a newer official commit without changing the pin:

```bash
VLLM_REF=origin/main ./scripts/prepare-upstream-vllm.sh /opt/vllm-candidate
```

A clean apply is only an applicability result. Update `UPSTREAM_VLLM_REF` and
regenerate the queue before treating a newer base as supported.

## Build

From the prepared checkout:

```bash
cp /path/to/deepseek-v4-cmp170hx/docker/Dockerfile.splitbuild .
cp /path/to/deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
podman build -f Dockerfile.splitbuild -t darknightlab/vllm-170hx:upstream-main .
```

The queue changes native headers and, with the performance layer enabled,
native CUDA sources. Use a full sm_80 source build; a Python-only bind mount is
not sufficient for a clean build from the official base.

## Verification status

Completed:

- confirmed the fork is exactly one squash commit ahead of its merge-base;
- reviewed all 18 conflict files and retained newer official implementations
  where the fork behavior was optional or superseded;
- verified the minimal-only, minimal+downstream, and full three-layer queues all
  apply cleanly from the pinned commit and compile as Python trees;
- completed a full CUDA 13 / Torch 2.13 sm_80 native build;
- completed three-rank CMP 170HX PP startup, sparse-MLA warmup, DSpark capture,
  `/v1/models`, and a real chat completion;
- initialized three bounded 5 GiB O_DIRECT disk KV files with per-rank aligned
  physical strides;
- removed incompatible fork protocols found by native build and hardware
  execution, including partial per-group KV zeroing and cudagraph pad-up.

Still required:

- a clean hardware A/B run with `VLLM_APPLY_FORK_PERFORMANCE=0`;
- four-rank PP verification;
- sustained long-context, concurrent, eviction, disk cache-hit recovery, and
  performance comparison against the legacy `c3046d1` image.

The queue is hardware-startup verified, not yet performance-certified.
