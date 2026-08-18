# Official upstream patch queue

Pinned base:

```text
vllm-project/vllm 402547d7f02bdbfc5dce5d27dc21f50dd4d627b6
```

The queue is ordered, feature-scoped, and reproducible. Patch `0001` is the
minimal SM80 execution path. Each subsequent optimization has its own patch;
`0018` is the project downstream layer. Experimental patches are never enabled
by the default profile.

## Profiles

```bash
# Minimal SM80 + downstream serving, no fork performance work
VLLM_FORK_PERFORMANCE_PROFILE=none \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-sm80-minimal

# Default: independently scoped, reachable optimizations 0002-0011
./scripts/prepare-upstream-vllm.sh /opt/vllm-upstream

# Include compile-checked experimental ports 0012-0017 and 0019-0021
VLLM_FORK_PERFORMANCE_PROFILE=all \
  ./scripts/prepare-upstream-vllm.sh /opt/vllm-upstream-all
```

## Patch index

| Patch | Feature | Default |
|---|---|---|
| 0001 | Minimal Ampere correctness: backend, software FP8, Triton sparse MLA/MQA, fallbacks | always |
| 0002 | Deterministic persistent top-k and sampler | verified |
| 0003 | Optional Marlin FP8-to-BF16 dequantization | verified |
| 0004 | SM80 router BF16 Triton GEMV | verified |
| 0005 | Deterministic MoE alignment | verified, runtime flag off |
| 0006 | A100 custom-AR crossover and hierarchical all-reduce | verified |
| 0007 | Vocab-parallel local argmax infrastructure | verified |
| 0008 | Wide sparse KV gather/dequantization | verified |
| 0009 | Sparse compressor warp sizing | verified |
| 0010 | SM80 MQA/indexer logits tuning | verified |
| 0011 | Multi-stream capture safety and control | verified |
| 0012 | Marlin occupancy/warp perturbations | experiment, flags off |
| 0013 | Cudagraph PIECEWISE pad-up with current `max_query_len` API | experiment |
| 0014 | Pinned input metadata staging pools | experiment |
| 0015 | Adaptive Marlin MoE block-size selector | experiment |
| 0016 | Persistent Marlin MoE workspace | experiment |
| 0017 | Attention input-GEMM fusion and replicated-GEMM token sharding | experiment |
| 0018 | 170HX downstream: DSpark+PP, long context, parsers, structured output, disk KV | always |
| 0019 | DSpark vocab-sharded Markov local argmax | experiment |
| 0020 | DSpark fused sequential Markov argmax | experiment |
| 0021 | Indexer prefill/decode query-row sharding | experiment |

The patch boundaries are functional boundaries, not one-file boundaries. A
feature patch includes all of its required caller, metadata, kernel, config,
and test changes.

## Deliberately not exported yet

Two original fork groups still require a semantic port before they can be
represented as usable official-main patches:

- MHC int8 all-reduce / fused sqrsum / prenorm sharding;
- the deep sparse-MLA campaign (8192-row ragged scan, ratio-128 query blocking,
  uniform decode grouping/split tuning, and exact-tile specialization).

Their fork versions overwrite newer official metadata/kernel APIs. Keeping a
raw diff that does not apply or compile would recreate the half-connected-code
problem this split is meant to solve. They remain tracked in
`REBASE_NOTES.md` and are not silently folded into another patch.

## Build

```bash
cp /path/to/deepseek-v4-cmp170hx/docker/Dockerfile.splitbuild .
cp /path/to/deepseek-v4-cmp170hx/docker/dockerignore.txt .dockerignore
podman build -f Dockerfile.splitbuild -t darknightlab/vllm-170hx:upstream-main .
```

Patch `0001` changes a native header. Several verified/experimental patches
change CUDA sources. Use a full sm_80 build for a clean image.

## Verification

Completed:

- `none`, `verified`, and `all` profiles apply cleanly from the pin;
- all three profiles pass Python compilation;
- the `all` profile matches its submitted Git tree exactly;
- the pre-split active feature set completed a full sm_80 build and PP3
  startup, DSpark capture, bounded disk-KV initialization, model listing, and
  a real chat request.

Still required:

- full native build and hardware A/B for the new per-feature profiles;
- hardware verification of each experimental patch individually;
- four-card and sustained long-context testing.
